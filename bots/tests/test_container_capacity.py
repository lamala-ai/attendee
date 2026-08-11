import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bots import container_capacity


class TestPidsReadings(unittest.TestCase):
    def _cgroup(self, current=None, maximum=None):
        """A scratch cgroup directory, patched in as the only place worth looking."""
        directory = Path(self.tmp.name)
        current_paths, max_paths = (), ()
        if current is not None:
            (directory / "pids.current").write_text(f"{current}\n")
            current_paths = (directory / "pids.current",)
        if maximum is not None:
            (directory / "pids.max").write_text(f"{maximum}\n")
            max_paths = (directory / "pids.max",)
        return patch.multiple(
            container_capacity,
            PIDS_CURRENT_PATHS=current_paths or (directory / "absent",),
            PIDS_MAX_PATHS=max_paths or (directory / "absent",),
        )

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_reads_the_ceiling_and_the_distance_to_it(self):
        with self._cgroup(current=812, maximum=1024):
            self.assertEqual(container_capacity.pids_current(), 812)
            self.assertEqual(container_capacity.pids_max(), 1024)
            self.assertIn("812/1024", container_capacity.capacity_summary())
            self.assertIn("212 left", container_capacity.capacity_summary())

    def test_an_unlimited_ceiling_is_not_reported_as_a_number(self):
        """cgroup writes the word "max" for no limit, which is not a quantity."""
        with self._cgroup(current=40, maximum="max"):
            self.assertIsNone(container_capacity.pids_max())
            self.assertIn("no pids ceiling set", container_capacity.capacity_summary())

    def test_a_host_with_no_pids_cgroup_falls_back_to_counting_threads(self):
        with self._cgroup():
            with patch.object(container_capacity, "threads_in_container", return_value=734):
                self.assertIn("734 threads", container_capacity.capacity_summary())

    def test_measuring_never_raises_into_the_meeting(self):
        """log_capacity runs on the path that starts a bot, so it swallows everything."""
        with patch.object(container_capacity, "capacity_summary", side_effect=OSError("boom")):
            container_capacity.log_capacity("Bot 1 starting")  # must not raise

    def test_low_headroom_is_said_at_warning(self):
        with self._cgroup(current=1000, maximum=1024):
            with self.assertLogs(container_capacity.logger, level="WARNING") as captured:
                container_capacity.log_capacity("Bot 1 starting")
        self.assertIn("task ceiling", captured.output[0])


class TestEncoderThreadArgs(unittest.TestCase):
    """A bot runs two libx264 encoders, and unbounded they size themselves off a CPU
    count (32) this container has no shortage of - spending the resource it does."""

    def test_capped_by_default(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(container_capacity.encoder_thread_args(), ["-threads", "2"])

    def test_a_deployment_can_hand_the_decision_back_to_ffmpeg(self):
        for opt_out in ("0", "auto", ""):
            with patch.dict("os.environ", {"FFMPEG_ENCODER_THREADS": opt_out}):
                self.assertEqual(container_capacity.encoder_thread_args(), [], f"FFMPEG_ENCODER_THREADS={opt_out!r}")

    def test_a_deployment_can_choose_its_own_number(self):
        with patch.dict("os.environ", {"FFMPEG_ENCODER_THREADS": "6"}):
            self.assertEqual(container_capacity.encoder_thread_args(), ["-threads", "6"])

    def test_nonsense_falls_back_to_the_cap_rather_than_to_unbounded(self):
        with patch.dict("os.environ", {"FFMPEG_ENCODER_THREADS": "lots"}):
            self.assertEqual(container_capacity.encoder_thread_args(), ["-threads", "2"])


class TestEncodersAreCapped(unittest.TestCase):
    """The two ffmpeg commands a bot actually runs, asserted where they are built."""

    def test_the_meeting_recorder_caps_its_encoder(self):
        from bots.bot_controller.screen_and_audio_recorder import ScreenAndAudioRecorder

        recorder = ScreenAndAudioRecorder(file_location="/tmp/test.mp4", recording_dimensions=(1920, 1080), audio_only=False)
        with patch("bots.bot_controller.screen_and_audio_recorder.subprocess.Popen") as popen:
            recorder.start_recording(":0")
        command = popen.call_args[0][0]
        self.assertIn("-threads", command)
        self.assertEqual(command[command.index("-threads") + 1], "2")

    def test_the_debug_recorder_caps_its_encoder(self):
        from bots.web_bot_adapter.debug_screen_recorder import DebugScreenRecorder

        recorder = DebugScreenRecorder(display_var=":0", screen_dimensions=(1920, 1080), output_file_path="/tmp/debug.mp4")
        with patch("bots.web_bot_adapter.debug_screen_recorder.subprocess.Popen") as popen:
            recorder.start()
        command = popen.call_args[0][0]
        self.assertIn("-threads", command)
        self.assertEqual(command[command.index("-threads") + 1], "2")


if __name__ == "__main__":
    unittest.main()
