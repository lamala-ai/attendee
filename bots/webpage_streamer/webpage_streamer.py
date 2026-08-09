import logging

from selenium import webdriver
from selenium.webdriver.chrome.service import Service

logger = logging.getLogger(__name__)

import asyncio
import contextlib
import os
import time
from fractions import Fraction

import gi
import numpy as np
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from av import AudioFrame, VideoFrame
from pyvirtualdisplay import Display

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp

Gst.init(None)

os.environ["PULSE_LATENCY_MSEC"] = "20"


def streamer_is_shared():
    """Is this process one bot's own streamer, or the fleet's?

    Spelled out in `WebpageStreamerManager.send_webpage_streamer_shutdown_request` too,
    rather than imported from it: that module is part of the Django app, and this one is
    a standalone script that must not drag Django in to answer a question about an
    environment variable.
    """
    return os.getenv("WEBPAGE_STREAMER_IS_SHARED", "").strip().lower() in ("1", "true", "yes")


# A shared page changes on the order of seconds to minutes, not thirty times a second -
# matches the 1-2fps a meeting platform's own screenshare capture already runs at, so
# this is a ceiling, not a target. Paired with ximagesrc's use-damage in _video_branch,
# an unchanging page costs nothing between real updates instead of encoding the same
# frame at a fixed cadence forever. Module-level so GstVideoStreamTrack, defined below,
# can use it as a default before WebpageStreamer itself exists.
CAPTURE_FRAMERATE = 2

# How long a capture that has reached PLAYING may hand over nothing before that is worth
# acting on. Generous next to the interval between frames at CAPTURE_FRAMERATE, and short
# enough to spend twice inside one /offer without the bot's own 30s timeout on that
# request running out.
CAPTURE_FRAME_DEADLINE_SECONDS = 5


class GstVideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, sink, width, height, framerate=CAPTURE_FRAMERATE, fault=None, stall_deadline=CAPTURE_FRAME_DEADLINE_SECONDS):
        super().__init__()
        self._sink = sink
        self._width = width
        self._height = height
        # Recorded, not consulted: recv() paces itself off each buffer's own pts, so
        # this is what the pipeline was told to capture at, kept for introspection
        # (tests, logs) rather than driving playback.
        self._framerate = framerate
        self._base_pts_ns = None
        # Asked, never read directly. The pipeline's bus is a *destructive* queue, so
        # this track does not pop from it: `fault` is the streamer's single reader, which
        # drains the bus into the log and remembers anything fatal. Two readers would not
        # split the messages between them - whichever got there first would take the
        # error, and the other would find a clean bus and conclude all was well.
        self._fault = fault
        self._stall_deadline = stall_deadline
        self._frames = 0
        self._stalled = False

    def _pull_sample(self):
        """One sample, or None if the source handed over nothing before the deadline.

        try-pull-sample rather than pull-sample, which blocks for ever. That is half of
        what kept the 2026-08-09 failure silent: a source producing nothing and a source
        about to produce something are the same call that has not returned, so the
        executor thread never came back and no code ran to notice.
        """
        return self._sink.try_pull_sample(int(self._stall_deadline * Gst.SECOND))

    async def recv(self) -> VideoFrame:
        loop = asyncio.get_running_loop()
        while True:
            sample = await loop.run_in_executor(None, self._pull_sample)
            if sample is not None:
                break

            fault = self._fault() if self._fault else ""
            if fault:
                # A dead pipeline is worth ending the track over: the receiver is parked
                # on recv() and would otherwise wait there for the length of the meeting.
                # Ended, it logs that the track ended and clears its frame, so the room
                # stops being told a live share is on its way.
                logger.error(f"The capture pipeline stopped delivering video and will not recover: {fault}")
                raise asyncio.CancelledError(f"Video pipeline failed: {fault}")

            # A deadline with a clean bus is a stall, not a death. A display can be
            # briefly idle, and ending a share that is about to come back is the worse
            # trade - so this is said once and then waited out.
            if not self._stalled:
                self._stalled = True
                logger.warning(f"No video captured in {self._stall_deadline}s and the pipeline reports no error ({self._frames} frames sent so far). Anything receiving this share is seeing a frozen or black picture.")

        if self._stalled:
            logger.info(f"Video capture recovered and is producing frames again after {self._frames} frames")
            self._stalled = False

        buffer = sample.get_buffer()
        pts_ns = buffer.pts

        ok, mapinfo = buffer.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("Could not map video buffer")

        try:
            data = memoryview(mapinfo.data)
            w, h = self._width, self._height

            # I420 layout: Y (W*H), U (W/2*H/2), V (W/2*H/2)
            y_size = w * h
            uv_size = y_size // 4

            y_plane = data[0:y_size]
            u_plane = data[y_size : y_size + uv_size]
            v_plane = data[y_size + uv_size : y_size + 2 * uv_size]

            frame = VideoFrame(format="yuv420p", width=w, height=h)
            frame.planes[0].update(y_plane)
            frame.planes[1].update(u_plane)
            frame.planes[2].update(v_plane)
        finally:
            buffer.unmap(mapinfo)

        if self._base_pts_ns is None:
            self._base_pts_ns = pts_ns

        rel_ns = pts_ns - self._base_pts_ns

        # Reuse the same μs time base as audio for nice alignment
        frame.time_base = Fraction(1, 1_000_000)
        frame.pts = rel_ns // 1_000

        self._frames += 1
        if self._frames == 1:
            # The counterpart of "First video frame received from the webpage streamer"
            # on the bot. With both lines present the two halves are separable from the
            # logs alone: this one and not that one is WebRTC, neither is capture. Its
            # absence is exactly why `0 received` was the end of the trail on 2026-08-09.
            logger.info(f"First video frame captured and sent: {self._width}x{self._height}")

        return frame


class GstAudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(
        self,
        sink: GstApp.AppSink,
        sample_rate: int = 16000,
        channels: int = 2,
    ):
        super().__init__()
        self._sink = sink
        self._sample_rate = sample_rate
        self._channels = channels
        self._base_pts_ns = None

    def _pull_sample(self):
        return self._sink.emit("pull-sample")

    async def recv(self) -> AudioFrame:
        loop = asyncio.get_running_loop()
        sample = await loop.run_in_executor(None, self._pull_sample)
        if sample is None:
            raise asyncio.CancelledError("Audio pipeline ended")

        buffer = sample.get_buffer()
        pts_ns = buffer.pts

        ok, mapinfo = buffer.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("Could not map audio buffer")

        try:
            data = mapinfo.data
            # S16LE: 2 bytes per sample per channel
            num_samples = len(data) // (2 * self._channels)
            if num_samples <= 0:
                raise RuntimeError("Empty audio buffer")
            pcm = np.frombuffer(data, dtype=np.int16).reshape(num_samples, self._channels)
        finally:
            buffer.unmap(mapinfo)

        layout = "stereo" if self._channels == 2 else "mono"
        frame = AudioFrame(format="s16", layout=layout, samples=num_samples)
        frame.planes[0].update(pcm.tobytes())
        frame.sample_rate = self._sample_rate

        if self._base_pts_ns is None:
            self._base_pts_ns = pts_ns

        rel_ns = pts_ns - self._base_pts_ns

        # Reuse the same μs time base as audio for nice alignment
        frame.time_base = Fraction(1, 1_000_000)
        frame.pts = rel_ns // 1_000

        return frame


class WebpageStreamer:
    # How often the watchdog looks, and how long a session may go unspoken-for. Class
    # attributes so a test can shrink them instead of waiting a quarter of an hour.
    KEEPALIVE_CHECK_INTERVAL_SECONDS = 60
    KEEPALIVE_TIMEOUT_SECONDS = 900

    # How long a capture pipeline that has reached PLAYING gets to hand over one frame
    # before it is treated as dead. One number for both the check made before an offer is
    # answered and the one a live track makes on every pull - the same question about the
    # same appsink, so it is not worth two names to disagree over.
    CAPTURE_FIRST_FRAME_SECONDS = CAPTURE_FRAME_DEADLINE_SECONDS

    UPSTREAM_AUDIO_TRACK_KEY = "upstream_audio_track"

    def __init__(
        self,
        video_frame_size,
    ):
        self.driver = None
        self.video_frame_size = video_frame_size
        self.display_var_for_recording = None
        self.display = None
        self.last_keepalive_time = None
        self.web_app = None
        # Read once, at construction: whether this process is disposable is a property of
        # the deployment, and a test that flips the environment mid-run would be testing
        # something that cannot happen.
        self.is_shared = streamer_is_shared()
        self._peer_connections = set()
        self._browser_lock = asyncio.Lock()
        self._keepalive_task = None
        # Holds the *original* upstream AudioStreamTrack from the first client that posts
        # to /offer. The MediaRelay is necessary because it creates a small buffer.
        # Without it audio quality is degraded.
        self._upstream_audio_relay = MediaRelay()

        # GStreamer-related
        self._gst_pipeline = None
        self._gst_video_sink = None
        self._gst_audio_sink = None
        self._video_track = None
        self._audio_track = None
        # Whether capturing audio is worth attempting at all in this container. It is
        # decided by the first attempt and remembered: a box with no sound card has none
        # on the next pipeline either, and rebuilding the doomed half every time costs a
        # second of every share and writes a WARNING that reads like the reason a share
        # failed. Off from the start when the deployment already knows.
        self._audio_capture_worth_trying = os.getenv("WEBPAGE_STREAMER_CAPTURE_AUDIO", "").strip().lower() not in ("0", "false", "no")
        # What the current pipeline has complained about, kept rather than left on the
        # bus. Sticky until a pipeline is built or torn down, because a GStreamer ERROR
        # is terminal for the pipeline that posted it - a second asker deserves the same
        # answer as the first, not an empty queue. See `capture_fault`.
        self._capture_fault = ""

    def _video_branch(self, width, height, display_var):
        return f"""
            ximagesrc display-name={display_var} use-damage=1 show-pointer=false
                ! video/x-raw,framerate={CAPTURE_FRAMERATE}/1,width={width},height={height}
                ! videoconvert
                ! video/x-raw,format=I420,width={width},height={height}
                ! queue max-size-buffers=5 max-size-time=0 leaky=downstream
                ! appsink name=video_sink emit-signals=false max-buffers=1 drop=true
        """

    AUDIO_BRANCH = """
            alsasrc device=default
                ! audio/x-raw,format=S16LE,channels=1,rate=16000
                ! audioconvert
                ! audioresample
                ! queue max-size-buffers=8000 leaky=downstream
                ! appsink name=audio_sink emit-signals=false max-buffers=8000 drop=true
    """

    @staticmethod
    def _bus_error(pipeline):
        """Whatever GStreamer actually objected to, as a sentence.

        Worth the detour: a failed state change reports only that it failed, and the
        element that refused - a missing sound card, an X display that is not there -
        says so on the bus and nowhere else. Without this the log reads "Failed to
        start GStreamer pipeline" no matter which half of it is at fault.
        """
        message = pipeline.get_bus().timed_pop_filtered(0, Gst.MessageType.ERROR)
        if message is None:
            return ""
        error, debug = message.parse_error()
        return f"{error.message} [{debug}]"

    def _try_pipeline(self, description, with_audio):
        """Bring one pipeline up, or return the reason it would not come up."""
        pipeline = Gst.parse_launch(description)
        video_sink = pipeline.get_by_name("video_sink")
        audio_sink = pipeline.get_by_name("audio_sink") if with_audio else None
        if video_sink is None or (with_audio and audio_sink is None):
            pipeline.set_state(Gst.State.NULL)
            return None, None, None, "could not find the appsinks"

        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            reason = self._bus_error(pipeline) or "state change refused"
            pipeline.set_state(Gst.State.NULL)
            return None, None, None, reason

        # PLAYING is usually reached asynchronously, so a source that cannot open its
        # device answers here rather than above. Without this wait the pipeline looks
        # started and then quietly produces nothing.
        changed, _state, _pending = pipeline.get_state(5 * Gst.SECOND)
        if changed != Gst.StateChangeReturn.SUCCESS:
            reason = self._bus_error(pipeline) or "did not reach PLAYING"
            pipeline.set_state(Gst.State.NULL)
            return None, None, None, reason

        return pipeline, video_sink, audio_sink, ""

    def _start_gstreamer_capture(self):
        if self._gst_pipeline:
            return

        width, height = self.video_frame_size
        display_var = self.display_var_for_recording
        video = self._video_branch(width, height, display_var)

        logger.info("Starting GStreamer capture pipeline")
        pipeline = video_sink = audio_sink = None
        reason = "audio capture is turned off"

        if self._audio_capture_worth_trying:
            pipeline, video_sink, audio_sink, reason = self._try_pipeline(video + self.AUDIO_BRANCH, True)
            if pipeline is None:
                # The audio branch is the fragile half - alsasrc needs a sound card, and a
                # container often has none. Losing it costs nothing here: what is being
                # shared is a web page, and the screenshare track carries no audio anyway.
                # Losing the video is the whole feature, so it is worth going on without.
                self._audio_capture_worth_trying = False
                logger.warning(f"GStreamer pipeline with audio would not start ({reason}) - capturing video only from here on, and not trying audio again in this process")

        if pipeline is None:
            # Not a fallback so much as the other supported shape: a page rendered for a
            # screenshare has no audio to carry, and this is the pipeline that runs on
            # every container without a sound card.
            pipeline, video_sink, audio_sink, reason = self._try_pipeline(video, False)

        if pipeline is None:
            raise RuntimeError(f"Failed to start GStreamer pipeline: {reason}")

        self._gst_pipeline = pipeline
        self._gst_video_sink = video_sink
        self._gst_audio_sink = audio_sink
        # A new pipeline complains for itself. Carrying the last one's fault forward
        # would end the track that is meant to replace it.
        self._capture_fault = ""
        logger.info(f"GStreamer capture pipeline is PLAYING ({'video and audio' if audio_sink else 'video only'})")

        self._video_track = GstVideoStreamTrack(
            sink=self._gst_video_sink,
            width=width,
            height=height,
            framerate=CAPTURE_FRAMERATE,
            # Handed the streamer's reader rather than the pipeline, so a stalled pull
            # asks the one thing that empties the bus instead of racing it for the error.
            fault=self.capture_fault,
            stall_deadline=self.CAPTURE_FIRST_FRAME_SECONDS,
        )
        self._audio_track = GstAudioStreamTrack(sink=self._gst_audio_sink, sample_rate=16000, channels=1) if self._gst_audio_sink else None

    @staticmethod
    def _pull_one_sample(sink, seconds):
        return sink.try_pull_sample(int(seconds * Gst.SECOND))

    def capture_fault(self):
        """Whatever the pipeline has said since it started playing, and whether it was fatal.

        The bus is otherwise only ever read when a state change fails, so an element that
        gives up *after* PLAYING - a source that loses its display, a caps negotiation
        that only fails once the first buffer is pushed - says so into a queue nobody
        empties. That is the shape of the 2026-08-09 failure: a pipeline reported PLAYING,
        produced nothing for 65 seconds, and logged not one line about it.

        This is the **only** place anything pops from that bus, and the reason is that
        popping is destructive. Two readers would not each see the error; the first would
        take it and the second would find a clean bus and conclude the pipeline was
        healthy. So what is read here is remembered on the streamer, and every asker -
        the check before an offer is answered, /restart_capture, and the live track on a
        stalled pull - gets the same answer however often they ask.

        Returns the fault as a sentence, or "" while the pipeline has not complained.
        """
        if self._gst_pipeline is None:
            return self._capture_fault

        bus = self._gst_pipeline.get_bus()
        while True:
            message = bus.timed_pop_filtered(0, Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.EOS)
            if message is None:
                return self._capture_fault
            if message.type == Gst.MessageType.EOS:
                logger.error("The GStreamer capture pipeline reached end of stream - it will not produce another frame")
                self._capture_fault = self._capture_fault or "the pipeline reached end of stream"
                continue
            fatal = message.type == Gst.MessageType.ERROR
            error, debug = message.parse_error() if fatal else message.parse_warning()
            logger.error(f"The GStreamer capture pipeline reported {'an error' if fatal else 'a warning'} after it started playing: {error.message} [{debug}]")
            # A warning is logged and nothing more. Elements warn about things they go on
            # working through, and ending a live share over one would make this change a
            # new way to lose a picture rather than a way to explain losing one.
            if fatal:
                self._capture_fault = self._capture_fault or f"{error.message} [{debug}]"

    async def _capture_is_producing_frames(self):
        """Has the pipeline that says it is PLAYING actually handed over a frame?

        Asked before an offer is answered, because everything downstream of here reports
        success whether or not it has: the SDP is exchanged, ICE completes, the bot
        receives a track, and the room watches a black rectangle. One frame off the
        appsink is the cheapest proof that what is being offered exists.
        """
        sink = self._gst_video_sink
        if sink is None:
            logger.error("The capture pipeline has no video sink, so there is nothing to offer")
            return False

        loop = asyncio.get_running_loop()
        sample = await loop.run_in_executor(None, self._pull_one_sample, sink, self.CAPTURE_FIRST_FRAME_SECONDS)
        fault = self.capture_fault()
        if sample is None:
            logger.error(f"The capture pipeline reached PLAYING but produced no frame in {self.CAPTURE_FIRST_FRAME_SECONDS}s ({fault or 'and reported nothing on its bus'}). Anything streamed from it now would be a share of nothing.")
            return False
        return True

    async def _capture_ready_for_a_new_connection(self):
        """A capture that is proven to produce, rebuilt once if it is not.

        The rebuild is the whole session - browser, pipeline and any peer connections
        left over - because the browser is the other half of what ``ximagesrc`` is
        pointed at and there is no way from here to tell which of the two went quiet.

        Taking the other connections down with it sounds worse than it is, and only in
        the shared deployment: every connection this process is serving is fed by the one
        pipeline that has just been shown not to produce, so they are all already
        carrying nothing. Rebuilding is the only thing that gets any of them a picture.
        """
        await self._ensure_browser()
        if self._gst_pipeline is None:
            self._start_gstreamer_capture()

        if await self._capture_is_producing_frames():
            return True

        logger.warning("Rebuilding the browser and the capture pipeline before answering this offer")
        await self.release_streaming_session()
        await self._ensure_browser()
        self._start_gstreamer_capture()
        return await self._capture_is_producing_frames()

    def _stop_gstreamer_capture(self):
        if self._gst_pipeline:
            logger.info("Stopping GStreamer capture pipeline")
            self._gst_pipeline.set_state(Gst.State.NULL)
            self._gst_pipeline = None
            self._gst_video_sink = None
            self._gst_audio_sink = None
            self._video_track = None
            self._audio_track = None
            self._capture_fault = ""

    def run(self):
        self._start_display()
        self._start_browser()
        self.load_webapp()

    def _start_display(self):
        if self.display is not None:
            return

        self.display_var_for_recording = os.environ.get("DISPLAY")
        if os.environ.get("DISPLAY") is None:
            # Create virtual display only if no real display is available
            self.display = Display(visible=0, size=self.video_frame_size)
            self.display.start()
            self.display_var_for_recording = self.display.new_display_var

    def _start_browser(self):
        """Bring up Chrome, or leave the one that is already up alone.

        Separate from `run` because a shared streamer gives Chrome back when it goes idle
        and has to be able to take it again on the next request - the process outlives any
        one browser.
        """
        if self.driver is not None:
            return

        options = webdriver.ChromeOptions()

        options.add_argument("--autoplay-policy=no-user-gesture-required")
        options.add_argument("--use-fake-device-for-media-stream")
        # options.add_argument("--use-fake-ui-for-media-stream")
        options.add_argument(f"--window-size={self.video_frame_size[0]},{self.video_frame_size[1]}")
        options.add_argument("--start-fullscreen")

        # options.add_argument('--headless=new')
        options.add_argument("--disable-gpu")
        # options.add_argument("--mute-audio")
        options.add_argument("--disable-application-cache")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--enable-blink-features=WebCodecs,WebRTC-InsertableStreams,-AutomationControlled")
        options.add_argument("--remote-debugging-port=9222")

        if os.getenv("ENABLE_CHROME_SANDBOX_FOR_WEBPAGE_STREAMER", "true").lower() != "true":
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-setuid-sandbox")
            logger.info("Chrome sandboxing is disabled")
        else:
            logger.info("Chrome sandboxing is enabled")
        logger.info(f"Video frame size: {self.video_frame_size}")

        options.add_experimental_option("excludeSwitches", ["enable-automation"])

        options.add_experimental_option(
            "prefs",
            {
                "profile.default_content_setting_values.media_stream_mic": 1,  # 1 = allow, 2 = block
                "profile.default_content_setting_values.media_stream_camera": 2,  # 1 = allow, 2 = block
            },
        )

        self.driver = webdriver.Chrome(options=options, service=Service(executable_path="/usr/local/bin/chromedriver"))
        logger.info(f"web driver server initialized at port {self.driver.service.port}")

        # Beside this file rather than relative to the working directory: the browser is
        # now started from a request handler as well as from startup, and a streamer that
        # renders again only when launched from the repo root is a trap.
        payload_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webpage_streamer_payload.js")
        with open(payload_path, "r") as file:
            payload_code = file.read()

        combined_code = f"""
            {payload_code}
        """

        # Add the combined script to execute on new document
        self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": combined_code})

    def _stop_browser(self):
        """Quit Chrome and forget it, so the next request starts a fresh one."""
        driver, self.driver = self.driver, None
        if driver is None:
            return
        try:
            driver.quit()
        except Exception as e:
            # Whatever went wrong, the driver is not ours any more - keeping the handle
            # would only mean handing a dead browser to the next request.
            logger.warning(f"Error quitting the browser: {e}")

    async def _ensure_browser(self):
        """The browser a request needs, started if an idle release gave it back.

        In the executor because starting Chrome takes seconds, and this now runs inside
        the request handlers rather than before the server exists; under the lock because
        two requests arriving together would otherwise start two of them and leak one.
        """
        async with self._browser_lock:
            if self.driver is not None:
                return
            logger.info("No browser is running - starting one for this request")
            await asyncio.get_running_loop().run_in_executor(None, self._start_browser)

    def _holds_a_session(self):
        return self.driver is not None or self._gst_pipeline is not None or bool(self._peer_connections)

    async def keepalive_monitor(self):
        """Notice that nobody is using this streamer, and give back what it is holding.

        What "give back" means is the whole point of the mode split. A per-bot streamer is
        one bot's, spawned with it and respawned with the next one, so it exits and its
        Chrome goes with it. A shared streamer is the fleet's only renderer and nothing
        respawns it: an idle exit there ends every screenshare until somebody notices, and
        it exits 0, so a platform restarting ON_FAILURE reads it as a job well done.

        So the timer stays - the Chrome it reclaims is real, and a streamer that hangs on
        to one browser per meeting it has ever rendered runs the box out of memory - but in
        shared mode it reclaims the *session* and leaves the process serving.
        """

        self.last_keepalive_time = time.time()

        while True:
            await asyncio.sleep(self.KEEPALIVE_CHECK_INTERVAL_SECONDS)

            current_time = time.time()
            time_since_last_keepalive = current_time - self.last_keepalive_time

            if time_since_last_keepalive <= self.KEEPALIVE_TIMEOUT_SECONDS:
                continue

            if not self.is_shared:
                logger.warning(f"No keepalive received in {time_since_last_keepalive:.1f} seconds. Shutting down process.")
                await self.shutdown_process()
                break

            if self._holds_a_session():
                logger.warning(f"No keepalive received in {time_since_last_keepalive:.1f} seconds. Releasing the idle streaming session. This process is shared, so it stays up and keeps serving.")
                await self.release_streaming_session()
            # Restart the clock either way: the next release is another quarter of an hour
            # of silence away, not one more tick of it.
            self.last_keepalive_time = time.time()

    async def release_streaming_session(self):
        """Hand back everything a streaming session holds, and keep the port.

        Chrome is the expensive half and the reason this exists - a wedged browser held
        past its meeting is how the worker ended up at 7.99 GB of an 8 GB limit. The
        pipeline and the peer connections go with it because a session that outlived its
        renderer can only produce black frames.
        """
        await self._close_peer_connections()
        self._stop_gstreamer_capture()
        self._stop_browser()
        logger.info("Idle streaming session released: browser, capture pipeline and peer connections are gone. Still listening for the next one.")

    async def _close_peer_connections(self):
        # Emptied in place rather than replaced: the request handlers hold this very set.
        peer_connections = list(self._peer_connections)
        self._peer_connections.clear()
        for pc in peer_connections:
            try:
                await pc.close()
            except Exception as e:
                logger.warning(f"Error closing a peer connection: {e}")

        if self.web_app is not None:
            # The upstream track belonged to a connection that is now closed, and the relay
            # buffers what it was carrying. Both are replaced rather than reused, or the
            # next session rebroadcasts a track nobody is feeding.
            self.web_app.pop(self.UPSTREAM_AUDIO_TRACK_KEY, None)
        self._upstream_audio_relay = MediaRelay()

    async def shutdown_process(self):
        """Gracefully shutdown the process."""
        try:
            self._stop_gstreamer_capture()
            if self.driver:
                self.driver.quit()
            if self.display:
                self.display.stop()
            if self.web_app:
                await self.web_app.shutdown()
            logger.info("Process shutting down")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            os._exit(0)

    def build_web_app(self):
        # The peer connections and the relay live on the streamer rather than in this
        # closure, because an idle release has to be able to reach them from outside a
        # request.
        pcs = self._peer_connections
        UPSTREAM_AUDIO_TRACK_KEY = self.UPSTREAM_AUDIO_TRACK_KEY

        async def offer_meeting_audio(req):
            """
            POST /offer_meeting_audio
            Return an SDP answer that *sends* the upstream audio (if present)
            to this new peer connection (listen-only client).
            """
            params = await req.json()
            offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

            # Do we have an upstream audio yet?
            upstream = req.app.get(UPSTREAM_AUDIO_TRACK_KEY)
            if upstream is None:
                return web.Response(status=409, text="No upstream audio has been published yet.")

            pc = RTCPeerConnection()
            pcs.add(pc)

            # Re-broadcast using the relay so multiple listeners are OK
            rebroadcast_track = self._upstream_audio_relay.subscribe(upstream)
            pc.addTrack(rebroadcast_track)

            @pc.on("connectionstatechange")
            async def _on_state():
                if pc.connectionState in ("failed", "closed", "disconnected"):
                    await pc.close()
                    pcs.discard(pc)

            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

        async def offer(req):
            params = await req.json()
            offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

            # There is a browser to capture whenever this process was started, and there is
            # not when a shared streamer released an idle session - the pipeline would
            # capture an empty display and the room would get a blank share. Lazy-started
            # so we don't accumulate latency before WebRTC is up, and proven before it is
            # offered, because a connection is the one thing here that succeeds either way.
            if not await self._capture_ready_for_a_new_connection():
                return web.json_response({"error": "The capture pipeline is not producing frames, so there is nothing to stream"}, status=503)

            pc = RTCPeerConnection()
            pcs.add(pc)

            # --- server-to-client: send GStreamer-captured video/audio ---
            v_track = self._video_track
            a_track = self._audio_track

            if v_track is not None:
                pc.addTrack(v_track)

            if a_track is not None:
                pc.addTrack(a_track)

            @pc.on("track")
            def on_track(track):
                if track.kind == "audio":
                    # store the ORIGINAL upstream track for rebroadcast
                    req.app[UPSTREAM_AUDIO_TRACK_KEY] = track
                    logger.info("Upstream audio track set for rebroadcast.")

            @pc.on("connectionstatechange")
            async def _on_state():
                if pc.connectionState in ("failed", "closed", "disconnected"):
                    await pc.close()
                    pcs.discard(pc)

            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

        async def start_streaming(req):
            data = await req.json()
            webpage_url = data.get("url")
            if not webpage_url:
                return web.json_response({"error": "URL is required"}, status=400)

            logger.info(f"Starting streaming to {webpage_url}")
            await self._ensure_browser()
            self.driver.get(webpage_url)

            return web.json_response({"status": "success"})

        async def restart_capture(req):
            """Give back the whole streaming session so the next offer builds a new one.

            For the bot that can see what nobody here can: that the frames it negotiated
            are not arriving. The streamer hands every connection the same pipeline and
            only builds another when it is holding none, so without this a renegotiation
            would be handed the very pipeline that stopped producing.
            """
            self.capture_fault()
            # Checked rather than taken on trust, because in the shared deployment this
            # renderer is several meetings' at once: one bot whose own connection went
            # wrong must not be able to take the pipeline out from under the others. A
            # pipeline that is still producing is not the thing that is broken.
            if await self._capture_is_producing_frames():
                logger.info("Not rebuilding the streaming session: the capture pipeline is still producing frames")
                return web.json_response({"status": "not needed"})

            logger.info("Rebuilding the streaming session on request")
            await self.release_streaming_session()
            return web.json_response({"status": "success"})

        async def keepalive(req):
            """Keepalive endpoint to reset the timeout timer."""
            self.last_keepalive_time = time.time()
            logger.info("Keepalive received")
            return web.json_response({"status": "alive", "timestamp": self.last_keepalive_time})

        async def shutdown(req):
            """Shutdown endpoint to gracefully shutdown the process."""
            logger.info("Shutting down process via API endpoint")
            await self.shutdown_process()
            return web.json_response({"status": "success"})

        app = web.Application()
        self.web_app = app

        # Start keepalive monitoring task
        async def init_keepalive_monitor(app):
            """Initialize keepalive monitoring when the app starts"""
            logger.info("Starting keepalive monitoring task")
            # Held, not fired and forgotten: asyncio keeps only a weak reference to a
            # running task, and this one has to outlive every request.
            self._keepalive_task = asyncio.create_task(self.keepalive_monitor())
            logger.info("Started keepalive monitoring task")

        async def stop_keepalive_monitor(app):
            task, self._keepalive_task = self._keepalive_task, None
            if task is None:
                return
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        app.on_startup.append(init_keepalive_monitor)
        app.on_cleanup.append(stop_keepalive_monitor)

        # Add CORS handling for preflight requests
        async def handle_cors_preflight(request):
            """Handle CORS preflight requests"""
            return web.Response(
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Access-Control-Max-Age": "86400",
                }
            )

        # Add CORS headers to all responses
        @web.middleware
        async def add_cors_headers(request, handler):
            """Add CORS headers to all responses"""
            response = await handler(request)
            response.headers.update(
                {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                }
            )
            return response

        app.middlewares.append(add_cors_headers)

        app.router.add_post("/start_streaming", start_streaming)

        app.router.add_post("/restart_capture", restart_capture)
        app.router.add_options("/restart_capture", handle_cors_preflight)

        app.router.add_post("/keepalive", keepalive)
        app.router.add_options("/keepalive", handle_cors_preflight)

        app.router.add_post("/shutdown", shutdown)
        app.router.add_options("/shutdown", handle_cors_preflight)

        app.router.add_post("/offer", offer)
        app.router.add_options("/offer", handle_cors_preflight)

        app.router.add_post("/offer_meeting_audio", offer_meeting_audio)  # SDP exchange
        app.router.add_options("/offer_meeting_audio", handle_cors_preflight)

        return app

    def load_webapp(self):
        app = self.build_web_app()
        # 8000 by default so the shared/Kubernetes deployments (which still address this
        # process by a fixed hostname:8000 convention) are unaffected. A per-bot process
        # sharing a host with other bots' streamers needs a port of its own - the caller
        # that spawns it picks one and passes it here.
        port = int(os.getenv("WEBPAGE_STREAMER_PORT", "8000"))

        # "0.0.0.0" is every IPv4 interface and no IPv6 one. That is fine under docker
        # compose and wrong anywhere the private network is IPv6: on Railway a bot
        # dialling <service>.railway.internal:8000 finds nothing listening, and because
        # the keepalive POST that would discover this has no timeout, it hangs instead of
        # failing - no error, no log line, and a share that never starts.
        #
        # None binds every interface of every family the host actually has, which covers
        # IPv6 without assuming it exists. Overridable for a deployment that must pin one.
        # "::" and not None: aiohttp turns None back into "0.0.0.0", so passing it looks
        # like it binds everything and does not. Linux leaves bindv6only off, so a "::"
        # socket accepts IPv4 through v4-mapped addresses and compose keeps working.
        host = os.getenv("WEBPAGE_STREAMER_BIND_HOST") or "::"
        logger.info(f"Webpage streamer binding to [{host}]:{port}")
        web.run_app(app, host=host, port=port)
