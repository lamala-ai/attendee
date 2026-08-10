"""Every bot's browser gets a profile directory of its own.

Off Kubernetes the bots of a deployment are celery tasks inside one container, sharing a
HOME and a /tmp. Two Chromes that agree on a profile directory are two Chromes of which
only the first starts - the second gets "session not created: probably user data
directory is already in use" and the bot it belongs to never joins its meeting.

The per-bot webpage streamer already mints one per launch. These tests cover the browser
that actually joins the meeting doing the same.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from bots.web_bot_adapter.web_bot_adapter import WebBotAdapter


class ChromeNotActuallyStarted(Exception):
    """Raised in place of launching a browser, once the options are captured."""


def make_adapter():
    """A WebBotAdapter with none of its heavy __init__ run - just what init_driver reads
    before it constructs a browser."""
    adapter = WebBotAdapter.__new__(WebBotAdapter)
    adapter.driver = None
    adapter.chrome_profile_dir = None
    adapter.video_frame_size = (1920, 1080)
    return adapter


def captured_chrome_options(adapter):
    """Run init_driver as far as the browser launch and return the options it built."""
    captured = SimpleNamespace(options=None)

    def capture(options=None, service=None):
        captured.options = options
        raise ChromeNotActuallyStarted()

    with patch.object(adapter, "write_chrome_policies_file"):
        with patch("bots.web_bot_adapter.web_bot_adapter.webdriver.Chrome", side_effect=capture):
            with patch("bots.web_bot_adapter.web_bot_adapter.Service"):
                try:
                    adapter.init_driver()
                except ChromeNotActuallyStarted:
                    pass

    return captured.options


def user_data_dir_argument(options):
    for argument in options.arguments:
        if argument.startswith("--user-data-dir="):
            return argument.split("=", 1)[1]
    return None


class TestChromeProfileDirectory(SimpleTestCase):
    def test_the_browser_is_launched_with_a_profile_directory_of_its_own(self):
        """Against the old behaviour there was no --user-data-dir at all, and every bot
        in the container was left to Chrome's own idea of where a profile lives."""
        adapter = make_adapter()

        profile_dir = user_data_dir_argument(captured_chrome_options(adapter))

        self.assertIsNotNone(profile_dir)
        self.assertTrue(os.path.isdir(profile_dir))
        self.assertEqual(profile_dir, adapter.chrome_profile_dir)
        adapter.discard_chrome_profile_dir()

    def test_two_bots_in_one_container_get_different_ones(self):
        """The whole point: this is the collision that stops the second bot joining."""
        adapter_a, adapter_b = make_adapter(), make_adapter()

        dir_a = user_data_dir_argument(captured_chrome_options(adapter_a))
        dir_b = user_data_dir_argument(captured_chrome_options(adapter_b))

        self.assertNotEqual(dir_a, dir_b)
        adapter_a.discard_chrome_profile_dir()
        adapter_b.discard_chrome_profile_dir()

    def test_a_retry_does_not_reuse_the_wedged_browsers_directory(self):
        """`repeatedly_attempt_to_join_meeting` gets three tries, and a Chrome that is
        wedged rather than gone is still holding the directory it was given."""
        adapter = make_adapter()
        first_dir = user_data_dir_argument(captured_chrome_options(adapter))

        adapter.driver = MagicMock()
        second_dir = user_data_dir_argument(captured_chrome_options(adapter))

        self.assertNotEqual(first_dir, second_dir)
        # And the one the quit browser was using is not left on disk.
        self.assertFalse(os.path.exists(first_dir))
        adapter.discard_chrome_profile_dir()

    def test_discarding_removes_the_directory_and_forgets_it(self):
        adapter = make_adapter()
        profile_dir = user_data_dir_argument(captured_chrome_options(adapter))

        adapter.discard_chrome_profile_dir()

        self.assertFalse(os.path.exists(profile_dir))
        self.assertIsNone(adapter.chrome_profile_dir)

    def test_discarding_twice_is_harmless(self):
        adapter = make_adapter()
        captured_chrome_options(adapter)

        adapter.discard_chrome_profile_dir()
        adapter.discard_chrome_profile_dir()

        self.assertIsNone(adapter.chrome_profile_dir)
