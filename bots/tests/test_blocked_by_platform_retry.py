"""A platform that refuses the bot a few times must not end the bot when there is no pod to recreate.

Google Meet's "You can't join this video call" comes and goes. On Kubernetes the answer to
five refusals in a row is a fresh pod (restart_bot_pod), and that is unchanged here. Off
Kubernetes - celery, which is how a single-container deployment runs every bot - there is
no pod: restart_bot_pod raised on a missing kube-config and the bot was gone ten seconds
after it started, from a meeting that let the next attempt in.

The adapter is built with __new__ and only the attributes the join loop touches, because
the loop is the unit under test and a real adapter needs a display, a browser and a bot row.
"""

import os
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from bots.bot_adapter import BotAdapter
from bots.web_bot_adapter import web_bot_adapter
from bots.web_bot_adapter.ui_methods import UiRetryableExpectedException
from bots.web_bot_adapter.web_bot_adapter import WebBotAdapter


def blocked():
    return UiRetryableExpectedException("You can't join this video call", "click_captions_button")


def adapter_whose_attempts_go(*outcomes):
    adapter = WebBotAdapter.__new__(WebBotAdapter)
    adapter.meeting_url = "https://meet.google.com/abc-defg-hij"
    adapter.left_meeting = False
    adapter.cleaned_up = False
    adapter.init_driver = MagicMock()
    adapter.attempt_to_join_meeting = MagicMock(side_effect=list(outcomes))
    adapter.send_message_callback = MagicMock()
    adapter.send_debug_screenshot_message = MagicMock()
    adapter.after_bot_joined_meeting = MagicMock()
    adapter.subclass_specific_after_bot_joined_meeting = MagicMock()
    return adapter


def messages_sent(adapter):
    return [call.args[0]["message"] for call in adapter.send_message_callback.call_args_list]


@patch.object(web_bot_adapter, "sleep", MagicMock())
@patch.object(web_bot_adapter.time, "sleep", MagicMock())
class BlockedByPlatformOffKubernetesTest(SimpleTestCase):
    @patch.dict(os.environ, {"LAUNCH_BOT_METHOD": ""})
    def test_a_bot_refused_five_times_keeps_trying_and_joins(self):
        adapter = adapter_whose_attempts_go(*[blocked() for _ in range(6)], None)

        adapter.repeatedly_attempt_to_join_meeting()

        self.assertEqual(adapter.attempt_to_join_meeting.call_count, 7)
        self.assertNotIn(BotAdapter.Messages.BLOCKED_BY_PLATFORM_REPEATEDLY, messages_sent(adapter))
        adapter.after_bot_joined_meeting.assert_called_once()

    @patch.dict(os.environ, {"LAUNCH_BOT_METHOD": ""})
    def test_a_bot_that_is_never_let_in_still_gives_up_with_a_screenshot(self):
        adapter = adapter_whose_attempts_go(*[blocked() for _ in range(50)])

        adapter.repeatedly_attempt_to_join_meeting()

        self.assertEqual(adapter.attempt_to_join_meeting.call_count, 16)
        adapter.send_debug_screenshot_message.assert_called_once()
        self.assertNotIn(BotAdapter.Messages.BLOCKED_BY_PLATFORM_REPEATEDLY, messages_sent(adapter))
        adapter.after_bot_joined_meeting.assert_not_called()

    @patch.dict(os.environ, {"LAUNCH_BOT_METHOD": "kubernetes"})
    def test_on_kubernetes_five_refusals_still_ask_for_a_new_pod(self):
        adapter = adapter_whose_attempts_go(*[blocked() for _ in range(6)], None)

        adapter.repeatedly_attempt_to_join_meeting()

        self.assertEqual(adapter.attempt_to_join_meeting.call_count, 5)
        self.assertEqual(messages_sent(adapter), [BotAdapter.Messages.BLOCKED_BY_PLATFORM_REPEATEDLY])
        adapter.after_bot_joined_meeting.assert_not_called()
