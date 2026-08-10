"""The off-Kubernetes equivalent of bot_pod_creator.py's per-bot streamer pod.

Kubernetes gets one webpage_streamer pod per bot pod, torn down with it by an owner
reference. Celery workers sharing one Railway replica have no such mechanism, so
BotController spawns and owns its own subprocess instead - these tests cover that
launch and teardown in isolation, without constructing the rest of BotController
(gstreamer pipeline, adapter, websocket manager, ...), which this logic never touches.
"""

import signal
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

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
        # Its own process group, or cleanup can only ever reach the Python interpreter -
        # not the Xvfb/Chrome/chromedriver it spawns underneath itself.
        self.assertTrue(kwargs["start_new_session"])

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
    """Cleanup signals the whole process group (os.killpg), not just the Python
    interpreter's own PID (process.terminate()) - Xvfb, Chrome and chromedriver are
    that interpreter's children, not this bot's, and a plain terminate() leaves every
    one of them running as an orphan. start_new_session=True at launch is what makes
    the interpreter's own pid double as its process group id."""

    def test_does_nothing_when_no_process_was_started(self):
        controller = make_controller()
        controller._cleanup_local_webpage_streamer_process()  # must not raise

    def test_signals_the_process_group_and_waits(self):
        controller = make_controller()
        process = MagicMock(pid=1234)
        process.poll.return_value = None  # still running
        process.wait.return_value = 0
        controller._local_webpage_streamer_process = process

        with patch("bots.bot_controller.bot_controller.os.getpgid", return_value=1234) as getpgid, patch("bots.bot_controller.bot_controller.os.killpg") as killpg:
            controller._cleanup_local_webpage_streamer_process()

        getpgid.assert_called_once_with(1234)
        killpg.assert_called_once_with(1234, signal.SIGTERM)
        process.wait.assert_called_once()
        self.assertIsNone(controller._local_webpage_streamer_process)

    def test_escalates_to_sigkill_when_sigterm_does_not_land(self):
        controller = make_controller()
        process = MagicMock(pid=1234)
        process.poll.return_value = None  # still running
        process.wait.side_effect = [subprocess.TimeoutExpired(cmd="run_webpage_streamer.py", timeout=10), 0]
        controller._local_webpage_streamer_process = process

        with patch("bots.bot_controller.bot_controller.os.getpgid", return_value=1234), patch("bots.bot_controller.bot_controller.os.killpg") as killpg:
            controller._cleanup_local_webpage_streamer_process()

        killpg.assert_has_calls([call(1234, signal.SIGTERM), call(1234, signal.SIGKILL)])
        self.assertEqual(process.wait.call_count, 2)

    def test_a_process_group_that_is_already_gone_does_not_raise(self):
        """The subprocess can die on its own between launch and cleanup - getpgid on a
        pid nobody holds any more is the normal shape of that, not an error to surface."""
        controller = make_controller()
        process = MagicMock(pid=1234)
        process.poll.return_value = None  # still running as far as Popen knows
        controller._local_webpage_streamer_process = process

        with patch("bots.bot_controller.bot_controller.os.getpgid", side_effect=ProcessLookupError()):
            controller._cleanup_local_webpage_streamer_process()  # must not raise

        self.assertIsNone(controller._local_webpage_streamer_process)

    def test_a_process_that_already_exited_is_never_signalled_by_pid(self):
        """The one that matters most, because of what it prevents rather than what it
        does. A PID whose process has exited and been reaped names nothing - and the
        kernel reissues PIDs, so tomorrow it names something else. os.getpgid() on a
        reissued PID answers with a process group belonging to a stranger, and this
        method's entire job is to SIGTERM a process group. In this container the likeliest
        stranger is the celery worker running every other meeting in it.

        Against the old behaviour, which went straight to getpgid(process.pid), the two
        calls below both happened."""
        controller = make_controller()
        process = MagicMock(pid=99)
        process.poll.return_value = 0  # exited and reaped
        controller._local_webpage_streamer_process = process

        with patch("bots.bot_controller.bot_controller.os.getpgid") as getpgid, patch("bots.bot_controller.bot_controller.os.killpg") as killpg:
            controller._cleanup_local_webpage_streamer_process()

        getpgid.assert_not_called()
        killpg.assert_not_called()
        self.assertIsNone(controller._local_webpage_streamer_process)

    def test_cleaning_up_one_bots_process_does_not_touch_another_bots(self):
        """The collision this whole design exists to avoid, checked from the teardown
        side: terminating one bot's streamer must never reach for another bot's handle."""
        controller_a, controller_b = make_controller(1), make_controller(2)
        process_a, process_b = MagicMock(pid=101), MagicMock(pid=102)
        process_a.poll.return_value = None  # still running
        process_a.wait.return_value = 0
        controller_a._local_webpage_streamer_process = process_a
        controller_b._local_webpage_streamer_process = process_b

        with patch("bots.bot_controller.bot_controller.os.getpgid", return_value=101), patch("bots.bot_controller.bot_controller.os.killpg") as killpg:
            controller_a._cleanup_local_webpage_streamer_process()

        killpg.assert_called_once_with(101, signal.SIGTERM)
        self.assertIsNone(controller_a._local_webpage_streamer_process)
        self.assertIs(controller_b._local_webpage_streamer_process, process_b)
