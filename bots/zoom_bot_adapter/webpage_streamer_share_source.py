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
"""

from __future__ import annotations

import asyncio
import logging
import threading

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


class LatestFrame:
    """The one place the aiortc thread and the GLib thread touch.

    Holds the newest I420 frame and nothing else. ``take`` returns it and reports
    whether it is new, so the pump can re-send an unchanged frame to keep the share
    alive without pretending it received something.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._size = (DEFAULT_SHARE_WIDTH, DEFAULT_SHARE_HEIGHT)
        self._fresh = False

    def put(self, frame_bytes: bytes, width: int, height: int) -> None:
        with self._lock:
            self._frame = frame_bytes
            self._size = (width, height)
            self._fresh = True

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

    def __init__(self, meeting_service, schedule_on_main_thread, unschedule_on_main_thread):
        self.meeting_service = meeting_service
        # GLib.timeout_add / GLib.source_remove, injected so this module never imports gi
        # and can be exercised without a main loop.
        self._schedule = schedule_on_main_thread
        self._unschedule = unschedule_on_main_thread

        self.latest_frame = LatestFrame()

        # --- the aiortc side, all touched only from _loop_thread ---
        self._loop = None
        self._loop_thread = None
        self._peer_connection = None

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

        self._peer_connection = RTCPeerConnection()

        @self._peer_connection.on("track")
        def on_track(track):
            logger.info(f"Webpage streamer track received: {track.kind}")
            if track.kind == "video":
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
                self.latest_frame.clear()
                return
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
        """
        if output_destination != "screenshare":
            logger.info(f"Ignoring webpage streamer output destination {output_destination} on the zoom native adapter")
            return
        if self._sharing_started:
            logger.info("Webpage streamer share already started")
            return

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
        result = self.share_source_helper.setExternalShareSource(
            self.share_source_callbacks, self.share_audio_callbacks
        )
        logger.info(f"setExternalShareSource result = {result}")
        if result != zoom.SDKERR_SUCCESS:
            logger.info("Failed to set the external share source, the room will not see the page")
            return
        self._sharing_started = True

    def on_share_start_send_callback(self, share_sender):
        """Zoom is ready for frames. Nothing may be sent before this fires."""
        logger.info("on_share_start_send_callback called")
        self.share_sender = share_sender
        if self._pump_timeout_id is None:
            self._pump_timeout_id = self._schedule(SHARE_FRAME_INTERVAL_MS, self._pump_frame)

    def on_share_stop_send_callback(self):
        logger.info("on_share_stop_send_callback called")
        self._stop_pump()
        self.share_sender = None

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
        except Exception:
            logger.exception("sendShareFrame raised")
        return True

    def _stop_pump(self):
        if self._pump_timeout_id is not None:
            self._unschedule(self._pump_timeout_id)
            self._pump_timeout_id = None

    # --- hook 4: take it down ------------------------------------------

    def stop_bot_output_media_stream(self, output_destination=None):
        self._stop_pump()
        self.latest_frame.clear()
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
                logger.info(f"StopShare result = {result}")
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
