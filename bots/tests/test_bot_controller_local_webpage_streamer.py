"""The off-Kubernetes equivalent of bot_pod_creator.py's per-bot streamer pod.

Kubernetes gets one webpage_streamer pod per bot pod, torn down with it by an owner
reference. Celery workers sharing one Railway replica have no such mechanism, so
BotController spawns and owns its own subprocess instead - these tests cover that
launch and teardown in isolation, without constructing the rest of BotController
(gstreamer pipeline, adapter, websocket manager, ...), which this logic never touches.
"""

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from bots.bot_controller.bot_controller import BotController


def make_controller(bot_id=42):
    """A BotController with none of its heavy __init__ run - just enough state for the
    local-streamer launch/cleanup methods, which touch nothing else on self."""
    controller = BotController.__new__(BotController)
    controller.bot_in_db = SimpleNamespace(id=bot_id)
    controller._local_webpage_streamer_process = None
    return controller


class TestLaunchLocalWebpageStreamer(SimpleTestCase):
    def test_spawns_the_streamer_script_on_a_free_port(self):
        controller = make_controller()
        fake_process = MagicMock(pid=1234)

        with patch("bots.bot_controller.bot_controller.subprocess.Popen", return_value=fake_process) as popen:
            process, base_url = controller._launch_local_webpage_streamer()

        self.assertIs(process, fake_process)
        self.assertTrue(base_url.startswith("http://127.0.0.1:"))
        port = int(base_url.rsplit(":", 1)[1])
        self.assertGreater(port, 0)

        popen.assert_called_once()
        args, kwargs = popen.call_args
        script_path = args[0][1]
        self.assertTrue(script_path.endswith("webpage_streamer/run_webpage_streamer.py") or script_path.endswith("webpage_streamer\\run_webpage_streamer.py"))
        self.assertEqual(kwargs["env"]["WEBPAGE_STREAMER_PORT"], str(port))

    def test_two_launches_get_different_ports(self):
        """The whole point: two bots on the same host must not collide the way the
        shared service's fixed :8000 convention did."""
        controller_a, controller_b = make_controller(1), make_controller(2)

        with patch("bots.bot_controller.bot_controller.subprocess.Popen", return_value=MagicMock(pid=1)):
            _, url_a = controller_a._launch_local_webpage_streamer()
            _, url_b = controller_b._launch_local_webpage_streamer()

        self.assertNotEqual(url_a, url_b)

    def test_clears_shared_flag_and_display_from_the_childs_environment(self):
        """WEBPAGE_STREAMER_IS_SHARED must not leak in from the bot process's own
        environment - it would put the child in shared-service semantics and skip its
        own /shutdown on cleanup, exactly the unsafe teardown this design avoids. DISPLAY
        must not leak in either, or the streamer renders into the bot's own X display
        instead of getting an isolated one."""
        controller = make_controller()

        with patch.dict("os.environ", {"WEBPAGE_STREAMER_IS_SHARED": "true", "DISPLAY": ":99"}):
            with patch("bots.bot_controller.bot_controller.subprocess.Popen", return_value=MagicMock(pid=1)) as popen:
                controller._launch_local_webpage_streamer()

        child_env = popen.call_args.kwargs["env"]
        self.assertNotIn("WEBPAGE_STREAMER_IS_SHARED", child_env)
        self.assertNotIn("DISPLAY", child_env)

    def test_returns_none_none_when_the_process_cannot_start(self):
        """A bot that cannot get a streamer up loses its screenshare capability, not
        its meeting - callers must be able to tell "no streamer" from "one is running"."""
        controller = make_controller()

        with patch("bots.bot_controller.bot_controller.subprocess.Popen", side_effect=OSError("no such file")):
            process, base_url = controller._launch_local_webpage_streamer()

        self.assertIsNone(process)
        self.assertIsNone(base_url)


class TestCleanupLocalWebpageStreamerProcess(SimpleTestCase):
    def test_does_nothing_when_no_process_was_started(self):
        controller = make_controller()
        controller._cleanup_local_webpage_streamer_process()  # must not raise

    def test_terminates_and_waits_for_the_process(self):
        controller = make_controller()
        process = MagicMock(pid=1234)
        process.wait.return_value = 0
        controller._local_webpage_streamer_process = process

        controller._cleanup_local_webpage_streamer_process()

        process.terminate.assert_called_once()
        process.wait.assert_called_once()
        process.kill.assert_not_called()
        self.assertIsNone(controller._local_webpage_streamer_process)

    def test_escalates_to_kill_when_terminate_does_not_land(self):
        controller = make_controller()
        process = MagicMock(pid=1234)
        process.wait.side_effect = [subprocess.TimeoutExpired(cmd="run_webpage_streamer.py", timeout=10), 0]
        controller._local_webpage_streamer_process = process

        controller._cleanup_local_webpage_streamer_process()

        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)

    def test_cleaning_up_one_bots_process_does_not_touch_another_bots(self):
        """The collision this whole design exists to avoid, checked from the teardown
        side: terminating one bot's streamer must never reach for another bot's handle."""
        controller_a, controller_b = make_controller(1), make_controller(2)
        process_a, process_b = MagicMock(pid=1), MagicMock(pid=2)
        process_a.wait.return_value = 0
        controller_a._local_webpage_streamer_process = process_a
        controller_b._local_webpage_streamer_process = process_b

        controller_a._cleanup_local_webpage_streamer_process()

        process_a.terminate.assert_called_once()
        process_b.terminate.assert_not_called()
        self.assertIsNone(controller_a._local_webpage_streamer_process)
        self.assertIs(controller_b._local_webpage_streamer_process, process_b)
