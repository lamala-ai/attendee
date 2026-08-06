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


class GstVideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, sink, width, height, framerate=15):
        super().__init__()
        self._sink = sink
        self._width = width
        self._height = height
        self._framerate = framerate
        self._base_pts_ns = None

    def _pull_sample(self):
        return self._sink.emit("pull-sample")

    async def recv(self) -> VideoFrame:
        loop = asyncio.get_running_loop()
        sample = await loop.run_in_executor(None, self._pull_sample)
        if sample is None:
            raise asyncio.CancelledError("Video pipeline ended")

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

    def _video_branch(self, width, height, display_var):
        return f"""
            ximagesrc display-name={display_var} use-damage=0 show-pointer=false
                ! video/x-raw,framerate=15/1,width={width},height={height}
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
        pipeline, video_sink, audio_sink, reason = self._try_pipeline(video + self.AUDIO_BRANCH, True)

        if pipeline is None:
            # The audio branch is the fragile half - alsasrc needs a sound card, and a
            # container often has none. Losing it costs nothing here: what is being
            # shared is a web page, and the screenshare track carries no audio anyway.
            # Losing the video is the whole feature, so it is worth going on without.
            logger.warning(f"GStreamer pipeline with audio would not start ({reason}) - capturing video only")
            pipeline, video_sink, audio_sink, reason = self._try_pipeline(video, False)

        if pipeline is None:
            raise RuntimeError(f"Failed to start GStreamer pipeline: {reason}")

        self._gst_pipeline = pipeline
        self._gst_video_sink = video_sink
        self._gst_audio_sink = audio_sink
        logger.info(f"GStreamer capture pipeline is PLAYING ({'video and audio' if audio_sink else 'video only'})")

        self._video_track = GstVideoStreamTrack(
            sink=self._gst_video_sink,
            width=width,
            height=height,
            framerate=15,
        )
        self._audio_track = (
            GstAudioStreamTrack(sink=self._gst_audio_sink, sample_rate=16000, channels=1)
            if self._gst_audio_sink
            else None
        )

    def _stop_gstreamer_capture(self):
        if self._gst_pipeline:
            logger.info("Stopping GStreamer capture pipeline")
            self._gst_pipeline.set_state(Gst.State.NULL)
            self._gst_pipeline = None
            self._gst_video_sink = None
            self._gst_audio_sink = None
            self._video_track = None
            self._audio_track = None

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
            # capture an empty display and the room would get a blank share.
            await self._ensure_browser()

            # Lazy-start capture so we don't accumulate latency before WebRTC is up
            if self._gst_pipeline is None:
                self._start_gstreamer_capture()

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
        port = 8000

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
