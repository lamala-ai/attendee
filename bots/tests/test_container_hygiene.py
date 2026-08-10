"""What one bot leaves behind must not be what stops the next one starting.

Every test here builds a fake /tmp or /proc rather than touching the real ones: the whole
point of the module under test is that it deletes files and kills processes, and a test
suite that exercises that against the machine it is running on is a bad trade.
"""

import os
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase

from bots.container_hygiene import (
    find_orphaned_bot_processes,
    reap_orphaned_bot_processes,
    reap_stale_x_locks,
    tidy_up_after_departed_bots,
)

LIVE_PID = 4242
DEAD_PID = 4243


def alive_only_for_live_pid(pid, _signal=0):
    """os.kill's contract, for a world with exactly one live process."""
    if pid == LIVE_PID:
        return None
    raise ProcessLookupError()


def write_x_lock(tmp_dir, display_number, pid):
    # Xvfb's own format: the PID right-aligned in eleven characters, newline-terminated.
    (tmp_dir / f".X{display_number}-lock").write_text(f"{pid:>10}\n")


def write_x_socket(socket_dir, display_number):
    (socket_dir / f"X{display_number}").write_text("")


class TestReapStaleXLocks(SimpleTestCase):
    def setUp(self):
        import tempfile

        self.tmp_dir = Path(tempfile.mkdtemp())
        self.socket_dir = self.tmp_dir / ".X11-unix"
        self.socket_dir.mkdir()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_removes_the_lock_and_socket_of_a_display_whose_server_is_gone(self):
        """The regression: pyvirtualdisplay walks display numbers, finds a lock for each,
        reports "server already running" for a server that is not, and the bot never gets
        a browser. Against the old behaviour nothing was ever removed and every later bot
        in the container failed the same way until it was restarted."""
        write_x_lock(self.tmp_dir, 99, DEAD_PID)
        write_x_socket(self.socket_dir, 99)

        with patch("bots.container_hygiene.os.kill", side_effect=alive_only_for_live_pid):
            reclaimed = reap_stale_x_locks(tmp_dir=self.tmp_dir, socket_dir=self.socket_dir)

        self.assertEqual(reclaimed, [99])
        self.assertFalse((self.tmp_dir / ".X99-lock").exists())
        self.assertFalse((self.socket_dir / "X99").exists())

    def test_leaves_a_display_that_is_genuinely_in_use_alone(self):
        write_x_lock(self.tmp_dir, 5, LIVE_PID)
        write_x_socket(self.socket_dir, 5)

        with patch("bots.container_hygiene.os.kill", side_effect=alive_only_for_live_pid):
            reclaimed = reap_stale_x_locks(tmp_dir=self.tmp_dir, socket_dir=self.socket_dir)

        self.assertEqual(reclaimed, [])
        self.assertTrue((self.tmp_dir / ".X5-lock").exists())
        self.assertTrue((self.socket_dir / "X5").exists())

    def test_reclaims_only_the_dead_displays_when_both_kinds_are_present(self):
        write_x_lock(self.tmp_dir, 1, LIVE_PID)
        write_x_lock(self.tmp_dir, 2, DEAD_PID)
        write_x_lock(self.tmp_dir, 3, DEAD_PID)

        with patch("bots.container_hygiene.os.kill", side_effect=alive_only_for_live_pid):
            reclaimed = reap_stale_x_locks(tmp_dir=self.tmp_dir, socket_dir=self.socket_dir)

        self.assertEqual(sorted(reclaimed), [2, 3])
        self.assertTrue((self.tmp_dir / ".X1-lock").exists())

    def test_leaves_a_lock_file_it_cannot_read_as_a_pid_alone(self):
        """Not everything matching .X*-lock was written by an Xvfb of ours, and a file
        this code does not understand is not a file it should delete."""
        (self.tmp_dir / ".X7-lock").write_text("not a pid")

        with patch("bots.container_hygiene.os.kill", side_effect=alive_only_for_live_pid):
            reclaimed = reap_stale_x_locks(tmp_dir=self.tmp_dir, socket_dir=self.socket_dir)

        self.assertEqual(reclaimed, [])
        self.assertTrue((self.tmp_dir / ".X7-lock").exists())

    def test_removes_a_lock_whose_socket_was_already_gone(self):
        write_x_lock(self.tmp_dir, 12, DEAD_PID)

        with patch("bots.container_hygiene.os.kill", side_effect=alive_only_for_live_pid):
            reclaimed = reap_stale_x_locks(tmp_dir=self.tmp_dir, socket_dir=self.socket_dir)

        self.assertEqual(reclaimed, [12])


def write_fake_proc_entry(proc_dir, pid, ppid, cmdline, name="python3"):
    entry = proc_dir / str(pid)
    entry.mkdir()
    (entry / "cmdline").write_bytes(cmdline.encode() + b"\0")
    # Field order is the real /proc/<pid>/stat's: pid, (name), state, ppid, ...
    (entry / "stat").write_text(f"{pid} ({name}) S {ppid} 0 0 0 -1 0\n")


class TestFindOrphanedBotProcesses(SimpleTestCase):
    def setUp(self):
        import tempfile

        self.proc_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil

        shutil.rmtree(self.proc_dir, ignore_errors=True)

    def test_finds_a_streamer_whose_owner_is_gone(self):
        write_fake_proc_entry(self.proc_dir, 900, ppid=1, cmdline="/usr/bin/python3 /attendee/bots/webpage_streamer/run_webpage_streamer.py")

        self.assertEqual(find_orphaned_bot_processes(proc_dir=self.proc_dir), [900])

    def test_ignores_a_streamer_whose_owner_is_still_running(self):
        """The load-bearing half. A streamer whose celery worker is alive is a streamer a
        bot is still using, and reaping it would take the screenshare out of a live
        meeting."""
        write_fake_proc_entry(self.proc_dir, 901, ppid=57, cmdline="/usr/bin/python3 /attendee/bots/webpage_streamer/run_webpage_streamer.py")

        self.assertEqual(find_orphaned_bot_processes(proc_dir=self.proc_dir), [])

    def test_finds_orphaned_xvfb_and_chrome_but_not_the_celery_worker(self):
        write_fake_proc_entry(self.proc_dir, 902, ppid=1, cmdline="Xvfb -br -nolisten tcp -screen 0 1280x720x24", name="Xvfb")
        write_fake_proc_entry(self.proc_dir, 903, ppid=1, cmdline="/opt/chrome/chrome --user-data-dir=/tmp/x", name="chrome")
        write_fake_proc_entry(self.proc_dir, 904, ppid=1, cmdline="/usr/local/bin/celery -A attendee worker -l INFO", name="celery")

        self.assertEqual(sorted(find_orphaned_bot_processes(proc_dir=self.proc_dir)), [902, 903])

    def test_never_reports_pid_1_or_this_process(self):
        """PID 1 is tini, which is what orphans are reparented *to* - and the process
        doing the reaping is not a candidate for being reaped."""
        write_fake_proc_entry(self.proc_dir, 1, ppid=1, cmdline="/tini -- /usr/local/bin/entrypoint.sh chrome")
        write_fake_proc_entry(self.proc_dir, os.getpid(), ppid=1, cmdline="python3 run_webpage_streamer.py")

        self.assertEqual(find_orphaned_bot_processes(proc_dir=self.proc_dir), [])


class TestReapOrphanedBotProcesses(SimpleTestCase):
    def setUp(self):
        import tempfile

        self.proc_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil

        shutil.rmtree(self.proc_dir, ignore_errors=True)

    def test_kills_each_orphan_individually_and_never_its_process_group(self):
        """Individually on purpose: an orphan's process group can have been inherited by
        anything in this container, up to and including the celery worker running every
        other meeting."""
        write_fake_proc_entry(self.proc_dir, 905, ppid=1, cmdline="python3 run_webpage_streamer.py")

        with patch("bots.container_hygiene.os.kill") as kill:
            reaped = reap_orphaned_bot_processes(proc_dir=self.proc_dir)

        self.assertEqual(reaped, [905])
        kill.assert_called_once()
        self.assertEqual(kill.call_args.args[0], 905)

    def test_a_process_that_exits_between_the_scan_and_the_signal_is_not_an_error(self):
        write_fake_proc_entry(self.proc_dir, 906, ppid=1, cmdline="python3 run_webpage_streamer.py")

        with patch("bots.container_hygiene.os.kill", side_effect=ProcessLookupError()):
            self.assertEqual(reap_orphaned_bot_processes(proc_dir=self.proc_dir), [])


class TestTidyUpAfterDepartedBots(SimpleTestCase):
    def test_a_sweep_that_fails_does_not_stop_the_bot_starting(self):
        """This runs on the path that joins a meeting. Turning a leak the container was
        coping with into a bot that never joins would be the worse trade."""
        with patch("bots.container_hygiene.reap_stale_x_locks", side_effect=OSError("nope")):
            with patch("bots.container_hygiene.reap_orphaned_bot_processes", side_effect=OSError("also nope")):
                tidy_up_after_departed_bots()

    @patch.dict(os.environ, {"DISABLE_CONTAINER_HYGIENE": "true"})
    def test_can_be_turned_off_entirely(self):
        with patch("bots.container_hygiene.reap_stale_x_locks") as reap_locks:
            with patch("bots.container_hygiene.reap_orphaned_bot_processes") as reap_processes:
                tidy_up_after_departed_bots()

        reap_locks.assert_not_called()
        reap_processes.assert_not_called()
