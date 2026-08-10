"""Putting a deployment back on one shared streamer, without a rollback.

One webpage_streamer per bot costs its container an Xvfb, a Chrome and a chromedriver per
bot, on top of the browser that joins the meeting. Where that is too much the symptom is
not a lost screenshare: it is bots that cannot start a browser at all, which is meetings
nobody joins. WEBPAGE_STREAMER_IS_SHARED is the way back to the arrangement that came
before - the same variable WebpageStreamerManager already reads to decide whether sending
/shutdown would take the renderer away from other bots.
"""

from unittest.mock import patch

from django.test import SimpleTestCase

from bots.bot_controller.webpage_streamer_manager import WebpageStreamerManager


def make_manager(**overrides):
    kwargs = dict(
        is_bot_ready_for_webpage_streamer_callback=lambda: True,
        get_peer_connection_offer_callback=lambda *a, **k: None,
        start_peer_connection_callback=lambda *a, **k: None,
        play_bot_output_media_stream_callback=lambda *a, **k: None,
        stop_bot_output_media_stream_callback=lambda *a, **k: None,
        on_message_that_webpage_streamer_connection_can_start_callback=lambda *a, **k: None,
        webpage_streamer_service_hostname="bot-42-streamer",
    )
    kwargs.update(overrides)
    return WebpageStreamerManager(**kwargs)


class TestSharedStreamerAddressing(SimpleTestCase):
    def test_a_manager_with_no_base_url_addresses_the_shared_service_by_name(self):
        manager = make_manager()

        with patch.dict("os.environ", {"WEBPAGE_STREAMER_HOSTNAME": "attendee-webpage-streamer.railway.internal", "LAUNCH_BOT_METHOD": "celery"}, clear=False):
            self.assertEqual(manager.base_url(), "http://attendee-webpage-streamer.railway.internal:8000")

    def test_a_per_bot_base_url_still_wins(self):
        manager = make_manager(webpage_streamer_base_url="http://127.0.0.1:34567")

        with patch.dict("os.environ", {"WEBPAGE_STREAMER_HOSTNAME": "attendee-webpage-streamer.railway.internal", "LAUNCH_BOT_METHOD": "celery"}, clear=False):
            self.assertEqual(manager.base_url(), "http://127.0.0.1:34567")

    def test_a_shared_streamer_is_not_shut_down_by_the_first_bot_to_leave(self):
        manager = make_manager()

        with patch.dict("os.environ", {"WEBPAGE_STREAMER_IS_SHARED": "true"}, clear=False):
            with patch("bots.bot_controller.webpage_streamer_manager.requests.post") as post:
                manager.send_webpage_streamer_shutdown_request()

        post.assert_not_called()


class TestKeepaliveGivesUpOnAStreamerThatNeverCame(SimpleTestCase):
    """A streamer that died on startup used to be asked once a second for the whole
    length of the meeting, at two log lines an attempt. One measured deployment had
    thirteen dead streamers being polled at once by bots that had long since moved on."""

    def test_stops_after_the_failure_limit_when_the_streamer_never_answered(self):
        manager = make_manager(webpage_streamer_base_url="http://127.0.0.1:1")
        manager.MAX_CONSECUTIVE_KEEPALIVE_FAILURES_BEFORE_GIVING_UP = 3

        with patch("bots.bot_controller.webpage_streamer_manager.time.sleep"):
            with patch("bots.bot_controller.webpage_streamer_manager.requests.post", side_effect=ConnectionError("refused")) as post:
                manager.send_webpage_streamer_keepalive_periodically()

        self.assertEqual(post.call_count, 3)

    def test_keeps_trying_for_a_streamer_that_has_answered_before(self):
        """A streamer that worked once may be rebuilt by restart_stream, so failures
        after a first success are not a reason to stop asking."""
        manager = make_manager(webpage_streamer_base_url="http://127.0.0.1:1")
        manager.MAX_CONSECUTIVE_KEEPALIVE_FAILURES_BEFORE_GIVING_UP = 3
        manager.webpage_streamer_connection_can_start = True

        attempts = []

        def fail_and_stop_after_ten(*args, **kwargs):
            attempts.append(1)
            if len(attempts) >= 10:
                manager.cleaned_up = True
            raise ConnectionError("refused")

        with patch("bots.bot_controller.webpage_streamer_manager.time.sleep"):
            with patch("bots.bot_controller.webpage_streamer_manager.requests.post", side_effect=fail_and_stop_after_ten):
                manager.send_webpage_streamer_keepalive_periodically()

        self.assertEqual(len(attempts), 10)

    def test_a_recovered_streamer_resets_the_count(self):
        manager = make_manager(webpage_streamer_base_url="http://127.0.0.1:1")
        manager.MAX_CONSECUTIVE_KEEPALIVE_FAILURES_BEFORE_GIVING_UP = 3

        responses = [ConnectionError("refused"), ConnectionError("refused"), _ok(), ConnectionError("refused"), ConnectionError("refused")]
        attempts = []

        def answer(*args, **kwargs):
            attempts.append(1)
            if not responses:
                manager.cleaned_up = True
                raise ConnectionError("refused")
            outcome = responses.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch("bots.bot_controller.webpage_streamer_manager.time.sleep"):
            with patch("bots.bot_controller.webpage_streamer_manager.requests.post", side_effect=answer):
                manager.send_webpage_streamer_keepalive_periodically()

        # Two failures, an answer, then two more - never three in a row, so the loop only
        # ended when cleanup did.
        self.assertEqual(len(attempts), 6)


def _ok():
    class Response:
        status_code = 500  # answered, but not a bot-ready 200 - enough to reset the count

    return Response()
