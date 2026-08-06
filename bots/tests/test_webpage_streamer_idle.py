"""What an idle webpage streamer gives back, and what it keeps.

The streamer was written to be spawned per bot: it holds a Selenium Chrome and a
GStreamer pipeline, and exiting after 15 minutes without a keepalive is how that Chrome
gets freed instead of one leaking per meeting. Deployed as a shared, always-on service
there is nothing to respawn it, so on 2026-08-05 the same timer took the fleet's only
renderer away for a day - and took it away with `os._exit(0)`, which a platform
restarting ON_FAILURE reads as a job finished rather than a service that died:

    15:10:04  WARNING  No keepalive received in 900.8 seconds. Shutting down process.
    15:10:04  INFO     Process shutting down

Every screenshare in that window failed silently: the backend accepted the URL and the
agent was told the share was up.

So the watchdog stays in both modes - the memory it reclaims is real - and what it does
when it fires is what the two modes disagree about. These tests pin both halves,
because the fix is only correct if the per-bot half is untouched.
"""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aiohttp.test_utils import TestClient, TestServer
from django.test import SimpleTestCase

from bots.webpage_streamer.webpage_streamer import Gst, WebpageStreamer


class FakeDriver:
    """A Selenium driver that records being quit, and refuses to be used afterwards.

    A MagicMock would happily answer `get()` on a browser that had already been quit,
    which is exactly the failure this change has to avoid: a streamer that survives an
    idle release but can never render again is the same outage with better logs.
    """

    def __init__(self):
        self.quit_count = 0
        self.urls = []
        self.scripts = []
        self.service = SimpleNamespace(port=9515)

    def execute_cdp_cmd(self, command, params):
        self.scripts.append((command, params))

    def get(self, url):
        if self.quit_count:
            raise AssertionError("the streamer used a browser it had already quit")
        self.urls.append(url)

    def quit(self):
        self.quit_count += 1


class FakePipeline:
    """Enough of a Gst.Pipeline to say which state it was last put into."""

    def __init__(self):
        self.states = []

    def set_state(self, state):
        self.states.append(state)
        return Gst.StateChangeReturn.SUCCESS


class FakePeerConnection:
    def __init__(self):
        self.closed = False
        self.tracks = []
        self.handlers = {}
        self.connectionState = "connected"
        self.localDescription = SimpleNamespace(sdp="answer-sdp", type="answer")

    def addTrack(self, track):
        self.tracks.append(track)

    def on(self, event):
        def register(handler):
            self.handlers[event] = handler
            return handler

        return register

    async def setRemoteDescription(self, description):
        self.remote_description = description

    async def createAnswer(self):
        return self.localDescription

    async def setLocalDescription(self, description):
        self.local_description = description

    async def close(self):
        self.closed = True


class WebpageStreamerIdleTestCase(SimpleTestCase):
    # Shrunk from a quarter of an hour so the watchdog actually fires during the test.
    CHECK_INTERVAL = 0.01
    TIMEOUT = 0.02

    def streamer(self, *, shared, with_session=True):
        with patch.dict(os.environ, {"WEBPAGE_STREAMER_IS_SHARED": "true" if shared else ""}):
            streamer = WebpageStreamer(video_frame_size=(1280, 720))

        streamer.KEEPALIVE_CHECK_INTERVAL_SECONDS = self.CHECK_INTERVAL
        streamer.KEEPALIVE_TIMEOUT_SECONDS = self.TIMEOUT

        if with_session:
            # A streamer mid-meeting: a browser, a capture pipeline and a peer connection.
            streamer.driver = FakeDriver()
            streamer._gst_pipeline = FakePipeline()
            streamer._gst_video_sink = MagicMock()
            streamer._video_track = MagicMock()
            streamer._peer_connections.add(FakePeerConnection())
        return streamer

    def stop_the_clock(self, streamer):
        """Stop the watchdog firing again while a test is making its assertions."""
        streamer.KEEPALIVE_TIMEOUT_SECONDS = 3600

    async def wait_for(self, predicate, message, timeout=5.0):
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate():
            if asyncio.get_running_loop().time() > deadline:
                self.fail(message)
            await asyncio.sleep(0.01)

    def browser_patch(self, drivers):
        """Make `_start_browser` hand out the given drivers instead of launching Chrome."""
        webdriver = MagicMock()
        webdriver.Chrome = MagicMock(side_effect=drivers)
        return patch("bots.webpage_streamer.webpage_streamer.webdriver", webdriver)


class TestASharedStreamerGoingIdle(WebpageStreamerIdleTestCase):
    async def test_it_releases_the_session_and_keeps_serving(self):
        """The regression. Against the old behaviour this exits the process instead, and
        the assertion on /keepalive never gets to run."""
        streamer = self.streamer(shared=True)
        driver, pipeline = streamer.driver, streamer._gst_pipeline
        peer_connection = next(iter(streamer._peer_connections))

        with patch("bots.webpage_streamer.webpage_streamer.os._exit") as exit_process:
            client = TestClient(TestServer(streamer.build_web_app()))
            await client.start_server()
            try:
                await self.wait_for(lambda: driver.quit_count > 0, "the idle session was never released")
                self.stop_the_clock(streamer)

                # The session is gone: the browser above all, since that is the memory.
                self.assertIsNone(streamer.driver)
                self.assertEqual(driver.quit_count, 1)
                self.assertIsNone(streamer._gst_pipeline)
                self.assertEqual(pipeline.states, [Gst.State.NULL])
                self.assertTrue(peer_connection.closed)
                self.assertEqual(streamer._peer_connections, set())

                # ...and the process is not.
                exit_process.assert_not_called()
                response = await client.post("/keepalive", json={})
                self.assertEqual(response.status, 200)
                self.assertEqual((await response.json())["status"], "alive")
            finally:
                await client.close()

    async def test_a_keepalive_from_a_bot_keeps_the_session(self):
        """The watchdog only fires on silence - a streamer with a bot talking to it must
        not be torn down under the meeting it is rendering."""
        streamer = self.streamer(shared=True)
        driver = streamer.driver

        with patch("bots.webpage_streamer.webpage_streamer.os._exit"):
            client = TestClient(TestServer(streamer.build_web_app()))
            await client.start_server()
            try:
                for _ in range(10):
                    self.assertEqual((await client.post("/keepalive", json={})).status, 200)
                    await asyncio.sleep(self.CHECK_INTERVAL)
                self.assertEqual(driver.quit_count, 0)
                self.assertIs(streamer.driver, driver)
            finally:
                self.stop_the_clock(streamer)
                await client.close()

    async def test_an_already_empty_streamer_is_not_released_over_and_over(self):
        """Nothing held, nothing to say: an operator reading these logs should see a
        release when one happened, not one line every 15 minutes for ever."""
        streamer = self.streamer(shared=True, with_session=False)

        with patch("bots.webpage_streamer.webpage_streamer.os._exit"):
            with patch.object(streamer, "release_streaming_session") as release:
                task = asyncio.ensure_future(streamer.keepalive_monitor())
                await asyncio.sleep(self.TIMEOUT * 10)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        release.assert_not_called()


class TestAPerBotStreamerGoingIdle(WebpageStreamerIdleTestCase):
    async def test_it_still_shuts_the_process_down(self):
        """Unchanged, and pinned so nobody "fixes" it: a per-bot streamer is one bot's
        own, nothing else is waiting on it, and the next bot spawns a new one. Left
        running it would leak a Chrome per meeting."""
        streamer = self.streamer(shared=False)
        driver, pipeline = streamer.driver, streamer._gst_pipeline

        self.assertFalse(streamer.is_shared)

        with patch("bots.webpage_streamer.webpage_streamer.os._exit") as exit_process:
            await asyncio.wait_for(streamer.keepalive_monitor(), timeout=5)

        exit_process.assert_called_once_with(0)
        self.assertEqual(driver.quit_count, 1)
        self.assertEqual(pipeline.states, [Gst.State.NULL])

    async def test_the_shared_flag_is_read_from_the_environment(self):
        for value, shared in (("true", True), ("1", True), ("yes", True), (" TRUE ", True), ("false", False), ("", False)):
            with patch.dict(os.environ, {"WEBPAGE_STREAMER_IS_SHARED": value}):
                self.assertIs(WebpageStreamer(video_frame_size=(1280, 720)).is_shared, shared, value)


class TestRenderingAgainAfterARelease(WebpageStreamerIdleTestCase):
    """A streamer that survives an idle release but can never render again is the same
    outage with better logs."""

    async def test_start_streaming_takes_a_fresh_browser_and_loads_the_page(self):
        streamer = self.streamer(shared=True)
        released_driver = streamer.driver
        self.stop_the_clock(streamer)
        next_driver = FakeDriver()

        with self.browser_patch([next_driver]):
            client = TestClient(TestServer(streamer.build_web_app()))
            await client.start_server()
            try:
                await streamer.release_streaming_session()
                self.assertIsNone(streamer.driver)

                response = await client.post("/start_streaming", json={"url": "https://example.com/page"})

                self.assertEqual(response.status, 200)
                self.assertIs(streamer.driver, next_driver)
                self.assertEqual(next_driver.urls, ["https://example.com/page"])
                self.assertEqual(released_driver.urls, [])  # the dead one was not reused
                # The page's own script has to be reinstalled, or the second browser
                # renders without the half that talks back.
                self.assertEqual([command for command, _ in next_driver.scripts], ["Page.addScriptToEvaluateOnNewDocument"])
            finally:
                await client.close()

    async def test_an_offer_rebuilds_the_capture_pipeline_and_the_browser(self):
        streamer = self.streamer(shared=True)
        self.stop_the_clock(streamer)
        next_driver, peer_connection = FakeDriver(), FakePeerConnection()
        rebuilt = FakePipeline()

        def restart_capture():
            streamer._gst_pipeline = rebuilt
            streamer._video_track = "video-track"

        with self.browser_patch([next_driver]), patch("bots.webpage_streamer.webpage_streamer.RTCPeerConnection", return_value=peer_connection), patch.object(streamer, "_start_gstreamer_capture", side_effect=restart_capture):
            client = TestClient(TestServer(streamer.build_web_app()))
            await client.start_server()
            try:
                await streamer.release_streaming_session()

                response = await client.post("/offer", json={"sdp": "offer-sdp", "type": "offer"})

                self.assertEqual(response.status, 200)
                self.assertEqual(await response.json(), {"sdp": "answer-sdp", "type": "answer"})
                self.assertIs(streamer.driver, next_driver)
                self.assertIs(streamer._gst_pipeline, rebuilt)
                self.assertEqual(peer_connection.tracks, ["video-track"])
                self.assertIn(peer_connection, streamer._peer_connections)
            finally:
                await client.close()

    async def test_only_one_browser_is_started_when_requests_arrive_together(self):
        """Two handlers racing on an empty streamer would otherwise start two Chromes and
        leak the one they lost."""
        streamer = self.streamer(shared=True, with_session=False)
        self.stop_the_clock(streamer)
        drivers = [FakeDriver(), FakeDriver()]

        with self.browser_patch(drivers) as webdriver:
            client = TestClient(TestServer(streamer.build_web_app()))
            await client.start_server()
            try:
                responses = await asyncio.gather(*[client.post("/start_streaming", json={"url": "https://example.com/page"}) for _ in range(2)])
                self.assertEqual([response.status for response in responses], [200, 200])
                self.assertEqual(webdriver.Chrome.call_count, 1)
                self.assertIs(streamer.driver, drivers[0])
                self.assertEqual(drivers[0].urls, ["https://example.com/page"] * 2)
            finally:
                await client.close()
