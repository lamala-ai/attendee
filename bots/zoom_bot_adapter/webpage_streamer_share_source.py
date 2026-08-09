"""Screensharing a webpage from the Zoom *native* SDK.

The web adapters get this for free: their bot is a browser, so the webpage streamer
opens a WebRTC connection straight into the page and the page plays the resulting
MediaStream into its screenshare track. Every one of those hooks is a one-line
``driver.execute_script``.

A native bot has no page, so both halves have to be built here:

* **The receiving half.** ``aiortc`` is the browser's replacement - it answers the
  streamer's ``/offer`` exchange and receives the video track. It is asyncio, and the
  adapter is a GLib main loop, so the peer connection lives on its own thread with its
  own event loop and the two sides meet at ``LatestFrame`` below.
* **The sending half.** ``IZoomSDKShareSourceHelper.setExternalShareSource`` lets a
  participant share frames it supplies itself rather than a captured screen. It is the
  same shape as the virtual camera already in ``zoom_bot_adapter`` - register callbacks,
  receive a sender, push I420 - but it lands on the share track instead of the webcam.

The thread boundary is the part worth being careful about. Zoom's SDK is not thread
safe and expects calls from the loop that owns it, so *nothing here calls the SDK from
the aiortc thread*: the receiver only ever parks the newest frame, and a GLib timeout
on the adapter's own thread picks it up and sends it. That also gives us the right
dropping behaviour for free - a screenshare wants the current frame, never a backlog of
stale ones, so a one-slot buffer is more correct than a queue.

**Nothing is said to the room until a frame has arrived.** The share is asked for long
before there is anything to fill it, and a WebRTC connection succeeds whether or not
media ever crosses it, so "Ada Sterling has started screen sharing" used to be announced
about 50ms after the SDP answer - the room's only signal that a share exists, spent on a
stream that had not yet delivered, and on 2026-08-09 never would. So the share is armed
here and announced by a timeout that waits for a real frame, and a stream that produces
none is rebuilt once and then abandoned in words rather than in black.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import zoom_meeting_sdk as zoom

logger = logging.getLogger(__name__)

# The streamer renders at 1280x720 (its --video-frame-size default) and Zoom takes what
# it is given on a share, unlike the webcam which negotiates a capability list.
DEFAULT_SHARE_WIDTH = 1280
DEFAULT_SHARE_HEIGHT = 720

# How often the GLib side looks for a new frame. 30fps is the top of the range the SDK
# is documented to accept for a share; the pump re-sends the last frame when nothing new
# has arrived, because Zoom stops showing a share that goes quiet.
SHARE_FRAME_INTERVAL_MS = 33

# How long a share Zoom has accepted may deliver nothing before the log says so. Well
# past a cold browser start and a first keyframe - the streamer's own page-load budget is
# 8s - so this is never the ordinary first second of a share, and short enough that it
# lands while the meeting it is ruining is still happening.
SHARE_FIRST_FRAME_DEADLINE_SECONDS = 15

# How long to wait for that first frame *before* telling Zoom there is a share at all,
# and how often to look while waiting. Same budget as above and for the same reason: a
# share is entitled to a cold browser start, and nothing longer than that is waiting for
# anything.
SHARE_FIRST_FRAME_WAIT_SECONDS = SHARE_FIRST_FRAME_DEADLINE_SECONDS
SHARE_FRAME_WAIT_INTERVAL_MS = 100

# How many frames must have made it all the way to I420 before the share is announced.
# One is the honest minimum: it is the difference between "the page is on its way" and
# "there is something to put on the screen", and it is what "Ada is sharing" claims.
SHARE_FRAMES_BEFORE_ANNOUNCING = 1

# Frames between "still alive" lines. At SHARE_FRAME_INTERVAL_MS that is about once a
# minute: enough to tell a live share from a stopped one in a log after the fact, not
# enough to become the log.
SHARE_PROGRESS_EVERY_N_FRAMES = 1800


class LatestFrame:
    """The one place the aiortc thread and the GLib thread touch.

    Holds the newest I420 frame and nothing else. ``take`` returns it and reports
    whether it is new, so the pump can re-send an unchanged frame to keep the share
    alive without pretending it received something.

    It also keeps the two counts, because it is already the one object both threads hold
    a lock on and the pump has to read a number the receiver writes. They exist to tell
    the silent failures apart: nothing arriving over WebRTC leaves both at zero, while a
    conversion that keeps raising leaves ``received`` climbing and ``converted`` at zero.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._size = (DEFAULT_SHARE_WIDTH, DEFAULT_SHARE_HEIGHT)
        self._fresh = False
        self._received = 0
        self._converted = 0

    def note_received(self) -> int:
        """Count a frame off the track, before anything is done to it. Returns the new
        total so the receiver can recognise the first one without a second lock."""
        with self._lock:
            self._received += 1
            return self._received

    def counts(self):
        """(received, converted), read together so they cannot disagree."""
        with self._lock:
            return self._received, self._converted

    def put(self, frame_bytes: bytes, width: int, height: int) -> None:
        with self._lock:
            self._frame = frame_bytes
            self._size = (width, height)
            self._fresh = True
            self._converted += 1

    def take(self):
        with self._lock:
            return self._frame, self._size, self._fresh
        # _fresh is cleared by mark_sent so a re-send is distinguishable from a new frame

    def mark_sent(self) -> None:
        with self._lock:
            self._fresh = False

    def clear(self) -> None:
        with self._lock:
            self._frame = None
            self._fresh = False


class WebpageStreamerShareSource:
    """Owns the WebRTC receiver, the Zoom share source, and the pump between them.

    One instance per bot. The adapter drives it through the five ``webpage_streamer_*``
    hooks the bot controller calls; everything else here is private.
    """

    def __init__(self, meeting_service, schedule_on_main_thread, unschedule_on_main_thread, request_restream=None):
        self.meeting_service = meeting_service
        # GLib.timeout_add / GLib.source_remove, injected so this module never imports gi
        # and can be exercised without a main loop.
        self._schedule = schedule_on_main_thread
        self._unschedule = unschedule_on_main_thread
        # How to ask for the whole stream to be built again - the streamer's capture
        # session and this peer connection with it. Optional, because it is the one thing
        # here that reaches outside the bot, and a share source without it simply gives up
        # instead of retrying.
        self._request_restream = request_restream

        self.latest_frame = LatestFrame()

        # --- the aiortc side, all touched only from _loop_thread ---
        self._loop = None
        self._loop_thread = None
        self._peer_connection = None
        self._video_track = None

        # --- the Zoom side, all touched only from the adapter's GLib thread ---
        self.share_source_helper = None
        self.share_source_callbacks = None
        # Held for the lifetime of the share even though it does nothing: the SDK keeps
        # a raw pointer to it, so letting it be collected would leave a dangling one.
        self.share_audio_callbacks = None
        self.share_sender = None
        self._pump_timeout_id = None
        self._sharing_started = False
        self._send_failure_ticker = 0
        self._frames_sent = 0
        self._sharing_since = None
        self._warned_about_no_frames = False
        # The share that has been asked for but not yet announced, because no frame has
        # arrived to fill it. See _announce_the_share_once_frames_arrive.
        self._share_requested = False
        self._waiting_since = None
        self._wait_timeout_id = None
        self._restream_attempted = False

    # --- lifecycle -----------------------------------------------------

    def start_event_loop(self):
        """Bring up the asyncio loop the peer connection lives on."""
        if self._loop is not None:
            return
        ready = threading.Event()

        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()

        self._loop_thread = threading.Thread(target=run, name="zoom-webpage-streamer", daemon=True)
        self._loop_thread.start()
        ready.wait(timeout=10)

    def _run_coroutine(self, coro, timeout=30):
        """Run a coroutine on the aiortc loop and wait for it from the GLib thread.

        The bot controller calls the hooks synchronously and uses their return values
        immediately - the offer has to come back before it can be POSTed - so this
        blocks. It is the adapter's thread that waits, never the SDK's callback thread.
        """
        self.start_event_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # --- hook 1: the offer ---------------------------------------------

    def get_peer_connection_offer(self):
        """Build the SDP offer the streamer's /offer endpoint answers.

        ``recvonly`` because the bot only ever receives here - the streamer renders the
        page and we play it. The browser implementation offers the same direction.
        """
        try:
            return self._run_coroutine(self._create_offer())
        except Exception as e:
            logger.exception("Failed to create webpage streamer peer connection offer")
            return {"error": str(e)}

    async def _create_offer(self):
        from aiortc import RTCPeerConnection

        # A second offer means the first stream is being replaced, so the connection it
        # was carried on is closed rather than left running: two receivers writing into
        # one frame slot would make "the room is seeing nothing" unanswerable.
        if self._peer_connection is not None:
            try:
                await self._peer_connection.close()
            except Exception:
                logger.info("The previous webpage streamer peer connection did not close cleanly")

        self._peer_connection = RTCPeerConnection()

        @self._peer_connection.on("track")
        def on_track(track):
            logger.info(f"Webpage streamer track received: {track.kind}")
            if track.kind == "video":
                self._video_track = track
                asyncio.ensure_future(self._consume_video(track))

        @self._peer_connection.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Webpage streamer peer connection state: {self._peer_connection.connectionState}")

        self._peer_connection.addTransceiver("video", direction="recvonly")
        offer = await self._peer_connection.createOffer()
        await self._peer_connection.setLocalDescription(offer)
        return {
            "sdp": self._peer_connection.localDescription.sdp,
            "type": self._peer_connection.localDescription.type,
        }

    # --- hook 2: the answer --------------------------------------------

    def start_peer_connection(self, offer_response):
        """Apply the streamer's answer. After this frames start arriving on the track."""
        if not offer_response or offer_response.get("error"):
            logger.info(f"Not starting webpage streamer peer connection, offer response was {offer_response}")
            return
        try:
            self._run_coroutine(self._apply_answer(offer_response))
        except Exception:
            logger.exception("Failed to apply webpage streamer answer")

    async def _apply_answer(self, offer_response):
        from aiortc import RTCSessionDescription

        answer = RTCSessionDescription(sdp=offer_response["sdp"], type=offer_response["type"])
        await self._peer_connection.setRemoteDescription(answer)
        logger.info("Webpage streamer peer connection answered")

    async def _consume_video(self, track):
        """Park each decoded frame as I420. Runs on the aiortc thread, never calls the SDK.

        ``yuv420p`` is what av calls I420, and it is what ``sendShareFrame`` wants, so
        the conversion is the decoder's rather than ours.
        """
        while True:
            try:
                frame = await track.recv()
            except Exception:
                logger.info("Webpage streamer video track ended")
                # Only if it is still the current one: a renegotiation leaves the old
                # track ending after the new one has started delivering, and clearing
                # then would throw away the frame that proves the recovery worked.
                if self._video_track is track:
                    self.latest_frame.clear()
                return
            if self.latest_frame.note_received() == 1:
                logger.info(f"First video frame received from the webpage streamer: {frame.width}x{frame.height}")
            try:
                array = frame.to_ndarray(format="yuv420p")
                self.latest_frame.put(array.tobytes(), frame.width, frame.height)
            except Exception:
                logger.exception("Failed to convert webpage streamer frame to I420")

    # --- hook 3: put it on the meeting's screen ------------------------

    def play_bot_output_media_stream(self, output_destination):
        """Start sharing. Only ``screenshare`` is meaningful for a native bot.

        The web adapter can also route this stream to the bot's webcam; here the webcam
        is already the virtual camera the adapter owns, so anything else is ignored
        rather than quietly stealing that source.

        Nothing is said to Zoom yet. The share is *armed*, and it is announced by
        ``_announce_the_share_once_frames_arrive`` when a frame has actually made it off
        the WebRTC track and into I420 - because "Ada Sterling has started screen sharing"
        is a claim about a picture, and Zoom will happily make it about an empty one.
        """
        if output_destination != "screenshare":
            logger.info(f"Ignoring webpage streamer output destination {output_destination} on the zoom native adapter")
            return
        if self._sharing_started:
            logger.info("Webpage streamer share already started")
            return
        if self._share_requested:
            logger.info("Webpage streamer share already waiting for its first frame")
            return

        self._share_requested = True
        self._waiting_since = time.monotonic()
        self._restream_attempted = False
        received, converted = self.latest_frame.counts()
        logger.info(f"Waiting for the first frame from the webpage streamer before starting the Zoom share ({received} received, {converted} converted so far)")
        if self._wait_timeout_id is None:
            self._wait_timeout_id = self._schedule(SHARE_FRAME_WAIT_INTERVAL_MS, self._announce_the_share_once_frames_arrive)

    def _announce_the_share_once_frames_arrive(self):
        """Announce the armed share when there is something to fill it, or act.

        This is the whole of the 2026-08-09 fix. Zoom used to be told about the share
        roughly 50ms after the SDP answer came back, which is before any frame could
        exist, so a stream that never delivered one was indistinguishable - to the room -
        from one that did: 65 seconds of "Ada Sterling has started screen sharing" over a
        black rectangle.

        Runs on the adapter's GLib thread. Returns True to stay scheduled, False to stop,
        which is GLib's contract for a repeating timeout.
        """
        if not self._share_requested:
            self._wait_timeout_id = None
            return False

        _received, converted = self.latest_frame.counts()
        if converted >= SHARE_FRAMES_BEFORE_ANNOUNCING:
            self._wait_timeout_id = None
            self._begin_zoom_share()
            return False

        if time.monotonic() - self._waiting_since < SHARE_FIRST_FRAME_WAIT_SECONDS:
            return True

        return self._act_on_a_stream_that_is_not_delivering()

    def _act_on_a_stream_that_is_not_delivering(self):
        """One bounded rebuild, then give up in words rather than in black.

        The rebuild is the only recovery worth trying from here: everything on this side
        reported success - the offer was answered, the track arrived, the connection
        reached ``connected`` - so what is broken is upstream of the track, in the
        renderer that a rebuild replaces. If that does not work either, the room is left
        seeing nothing, which is the honest outcome and the one somebody notices.
        """
        received, converted = self.latest_frame.counts()
        if not self._restream_attempted and self._request_restream is not None:
            self._restream_attempted = True
            self._waiting_since = time.monotonic()
            logger.warning(f"No frame has arrived from the webpage streamer in {SHARE_FIRST_FRAME_WAIT_SECONDS}s ({received} received, {converted} converted). Rebuilding the stream before starting the share.")
            try:
                self._request_restream()
            except Exception:
                logger.exception("Could not ask for the webpage stream to be rebuilt")
            return True

        self._share_requested = False
        self._waiting_since = None
        self._wait_timeout_id = None
        logger.error(f"Not starting the Zoom share: no frame has arrived from the webpage streamer ({received} received, {converted} converted){' even after rebuilding the stream' if self._restream_attempted else ''}. The room is shown nothing rather than a share of nothing.")
        return False

    def _begin_zoom_share(self):
        """Register the share source with Zoom. Only reached with a frame in hand."""
        # The waiting is over either way: this either becomes a share or a refusal, and
        # neither is something to keep looking for a first frame about.
        self._share_requested = False
        self._waiting_since = None

        self.share_source_helper = zoom.GetRawdataShareSourceHelper()
        if not self.share_source_helper:
            logger.info("share_source_helper is None, cannot share a page")
            return

        self.share_source_callbacks = zoom.ShareSourceCallbacks(
            onStartSendCallback=self.on_share_start_send_callback,
            onStopSendCallback=self.on_share_stop_send_callback,
        )
        # We still have no share audio to send - sendShareAudio rejects every documented
        # format on Linux, and a page we render has nothing to play. But declining it by
        # *omitting the argument* does not work: the C++ default (pAudioSource = nullptr)
        # is lost in the binding, which is a bare .def() with no nb::arg(...) = nullptr,
        # so both parameters are required in Python and the one-argument call raised
        #     TypeError: setExternalShareSource(): incompatible function arguments
        # at the last step before Zoom would have seen a frame. Nothing upstream of it
        # failed, and nothing downstream ever ran. So the way to decline share audio is
        # to hand over a source whose callbacks do nothing, which is what this is.
        self.share_audio_callbacks = zoom.ShareAudioCallbacks(
            onStartSendAudioCallback=self.on_share_start_send_audio_callback,
            onStopSendAudioCallback=self.on_share_stop_send_audio_callback,
        )
        result = self.share_source_helper.setExternalShareSource(self.share_source_callbacks, self.share_audio_callbacks)
        logger.info(f"setExternalShareSource result = {result}")
        if result != zoom.SDKERR_SUCCESS:
            logger.info("Failed to set the external share source, the room will not see the page")
            return
        self._sharing_started = True

    def on_share_start_send_callback(self, share_sender):
        """Zoom is ready for frames. Nothing may be sent before this fires."""
        logger.info("on_share_start_send_callback called")
        self.share_sender = share_sender
        # Reset here rather than in __init__ alone, so a second share in one meeting is
        # judged on its own delivery instead of inheriting the first one's.
        self._sharing_since = time.monotonic()
        self._frames_sent = 0
        self._warned_about_no_frames = False
        if self._pump_timeout_id is None:
            self._pump_timeout_id = self._schedule(SHARE_FRAME_INTERVAL_MS, self._pump_frame)

    def on_share_stop_send_callback(self):
        """Zoom telling us the share is over - which settles it, whoever ended it.

        It fires both when we stop the share ourselves and when it is ended out from
        under us, the ordinary case being the meeting finishing. Clearing
        ``_sharing_started`` here is what stops teardown then asking Zoom to stop a share
        that has already stopped, which it answers with SDKERR_WRONG_USAGE.
        """
        logger.info("on_share_stop_send_callback called")
        self._stop_pump()
        self.share_sender = None
        self._sharing_started = False
        # A share that is over is not a share that is failing to deliver.
        self._sharing_since = None

    def on_share_start_send_audio_callback(self, audio_sender):
        """Required by the SDK, deliberately silent. See the note in
        ``play_bot_output_media_stream``: we register an audio source only because the
        binding makes the argument mandatory, and we never send through it."""
        logger.info("on_share_start_send_audio_callback called - no audio will be sent")

    def on_share_stop_send_audio_callback(self):
        logger.info("on_share_stop_send_audio_callback called")

    def _pump_frame(self):
        """Send the newest frame to Zoom. Runs on the adapter's GLib thread.

        Returns True to stay scheduled, which is GLib's contract for a repeating
        timeout.
        """
        if self.share_sender is None:
            return True
        frame_bytes, (width, height), _fresh = self.latest_frame.take()
        if frame_bytes is None:
            self._warn_if_the_room_is_seeing_nothing()
            return True
        try:
            result = self.share_sender.sendShareFrame(frame_bytes, width, height, zoom.FrameDataFormat_I420_FULL)
            self.latest_frame.mark_sent()
            if result != zoom.SDKERR_SUCCESS:
                # Rate limited: a failing share fails every frame, which at 30fps would
                # bury every other line in the log.
                if self._send_failure_ticker % 100 == 0:
                    logger.info(f"sendShareFrame failed with result = {result}")
                self._send_failure_ticker += 1
                self._warn_if_the_room_is_seeing_nothing()
                return True
            self._note_frame_sent(width, height)
        except Exception:
            logger.exception("sendShareFrame raised")
        return True

    def _note_frame_sent(self, width: int, height: int) -> None:
        """Say once that the share is real, and rarely that it still is.

        The first line is the one worth having: until it appears, every success this
        module has reported is about registering a share rather than filling it.
        """
        self._frames_sent += 1
        if self._frames_sent == 1:
            received, converted = self.latest_frame.counts()
            logger.info(f"First frame accepted by Zoom: {width}x{height} ({received} received, {converted} converted). The room can see the page.")
        elif self._frames_sent % SHARE_PROGRESS_EVERY_N_FRAMES == 0:
            received, converted = self.latest_frame.counts()
            logger.info(f"Webpage streamer share still delivering: {self._frames_sent} frames sent, {received} received, {converted} converted")

    def _warn_if_the_room_is_seeing_nothing(self) -> None:
        """Report, once, a share Zoom accepted that is not showing anything.

        A share is no longer announced before a frame exists, so this is now about a
        stream that *stops*: the renderer going quiet mid-meeting looks, from here,
        exactly like the pump finding an empty slot thirty times a second and returning
        quietly. It is the same silence that let a live meeting watch a black screen for
        its whole length while every line in the log said success - roughly 2,300 ticks
        that produced no output at all.

        The counts are in the message because they say which half is at fault, and the
        three of them cannot be recovered afterwards from anything else that is logged.
        """
        if self._frames_sent or self._warned_about_no_frames or self._sharing_since is None:
            return
        if time.monotonic() - self._sharing_since < SHARE_FIRST_FRAME_DEADLINE_SECONDS:
            return
        self._warned_about_no_frames = True
        received, converted = self.latest_frame.counts()
        logger.warning(f"The room is not seeing the shared page: Zoom accepted the share {SHARE_FIRST_FRAME_DEADLINE_SECONDS}s ago and no frame has reached it since ({received} received from the webpage streamer, {converted} converted to I420, {self._send_failure_ticker} rejected by sendShareFrame). Nothing received means the frames are not arriving over WebRTC; received but not converted means the I420 conversion is failing; converted but not sent means Zoom is rejecting them.")

    def _stop_pump(self):
        if self._pump_timeout_id is not None:
            self._unschedule(self._pump_timeout_id)
            self._pump_timeout_id = None

    def _stop_waiting_to_share(self):
        self._share_requested = False
        self._waiting_since = None
        if self._wait_timeout_id is not None:
            self._unschedule(self._wait_timeout_id)
            self._wait_timeout_id = None

    # --- hook 4: take it down ------------------------------------------

    def stop_bot_output_media_stream(self, output_destination=None):
        self._stop_pump()
        # A share that is called off before it was ever announced still has a timeout
        # looking for its first frame, and that timeout would otherwise start a share
        # nobody asked for any more.
        self._stop_waiting_to_share()
        self.latest_frame.clear()
        self._sharing_since = None
        if self._sharing_started and self.meeting_service:
            try:
                # StopShare on the share controller, not setExternalShareSource(None).
                # Clearing the source by passing a null one cannot be expressed through
                # this binding at all: the arguments are mandatory (see
                # play_bot_output_media_stream) and no binding in this module declares
                # nb::arg().none(), so None is not accepted either. StopShare takes no
                # arguments, is what the SDK documents for ending a share, and stops the
                # room seeing anything - which is the actual goal. The sender is
                # invalidated by the onStopSend callback that follows.
                result = self.meeting_service.GetMeetingShareController().StopShare()
                if result == zoom.SDKERR_WRONG_USAGE:
                    # "Stops the current sharing" with no current sharing to stop. The
                    # race is ordinary rather than exceptional - the meeting ending stops
                    # the share and tears the bot down at nearly the same moment - and
                    # the outcome wanted here is the one already true.
                    logger.info("StopShare: there was no share left to stop")
                elif result != zoom.SDKERR_SUCCESS:
                    logger.info(f"StopShare did not succeed, result = {result}")
                else:
                    logger.info("StopShare result = SDKERR_SUCCESS")
            except Exception:
                logger.exception("Failed to stop sharing the page")
        self._sharing_started = False
        self.share_sender = None

    # --- shutdown -------------------------------------------------------

    def cleanup(self):
        self.stop_bot_output_media_stream()
        if self._peer_connection is not None and self._loop is not None:
            try:
                self._run_coroutine(self._peer_connection.close(), timeout=5)
            except Exception:
                logger.info("Webpage streamer peer connection did not close cleanly")
            self._peer_connection = None
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None
