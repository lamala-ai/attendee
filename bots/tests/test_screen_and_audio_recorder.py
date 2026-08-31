"""A recorder that produced nothing has to say so, and must not hand over a file anyway.

One meeting is the whole reason this file exists. FFmpeg was started, exited on its own
seconds later, and the log said only:

    Stopped screen and audio recorder ... file location /tmp/bot_…-rec_….mp4
    Input file does not exist at /tmp/bot_…-rec_….mp4, creating empty file
    Successfully uploaded /tmp/bot_…-rec_….mp4 to s3://…

Three lines, none of them an error, and a zero-byte mp4 in the bucket that the API then
served as a signed URL to whoever asked to watch their meeting back. Why FFmpeg died was
unknowable: its stderr went to /dev/null.

So both halves are pinned here - that a dead FFmpeg is reported with what it said, and
that "no recording" stays no recording rather than becoming an empty one.
"""

import os
import subprocess
import tempfile
from unittest.mock import patch

from django.test import SimpleTestCase

from bots.bot_controller.screen_and_audio_recorder import ScreenAndAudioRecorder

RECORDING_DIMENSIONS = (1920, 1080)


class FakeFfmpeg:
    """A Popen that has already exited, having written to the stderr file it was given."""

    def __init__(self, returncode=1, said=b"", still_running=False):
        self.returncode = None if still_running else returncode
        self._final_returncode = returncode
        self.said = said
        self.terminated = False

    def write_stderr_to(self, stderr_file):
        if stderr_file is not None and self.said:
            stderr_file.write(self.said)
            stderr_file.flush()

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self):
        self.returncode = self._final_returncode
        return self.returncode


class ScreenAndAudioRecorderTestCase(SimpleTestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.file_location = os.path.join(self.directory, "bot_abc-rec_def.mp4")
        self.recorder = ScreenAndAudioRecorder(self.file_location, RECORDING_DIMENSIONS, audio_only=False)

    def start_with(self, ffmpeg):
        """Start the recorder against a fake FFmpeg, keeping the real stderr plumbing."""

        def fake_popen(command, stdout=None, stderr=None):
            ffmpeg.write_stderr_to(stderr)
            return ffmpeg

        with patch.object(subprocess, "Popen", side_effect=fake_popen):
            self.recorder.start_recording(":99")

    def test_an_ffmpeg_that_died_on_its_own_is_an_error_carrying_what_it_said(self):
        """Fails against the old behaviour, which logged the ordinary "Stopped ..." line.

        Terminating a process that has already exited is a no-op, so a recorder that
        never ran and one that recorded the whole meeting ended identically.
        """
        self.start_with(FakeFfmpeg(returncode=1, said=b"[x11grab @ 0x1] Cannot open display :99\n"))

        with self.assertLogs("bots.bot_controller.screen_and_audio_recorder", level="ERROR") as logs:
            self.recorder.stop_recording()

        said = "\n".join(logs.output)
        self.assertIn("exited on its own", said)
        self.assertIn("Cannot open display :99", said)
        self.assertIn("-f x11grab", said, "the command it died running is what makes it fixable")

    def test_a_recorder_we_stopped_ourselves_is_not_reported_as_a_failure(self):
        self.start_with(FakeFfmpeg(returncode=0, still_running=True))

        with self.assertLogs("bots.bot_controller.screen_and_audio_recorder", level="INFO") as logs:
            self.recorder.stop_recording()

        self.assertIn("Stopped screen and audio recorder", "\n".join(logs.output))
        self.assertNotIn("exited on its own", "\n".join(logs.output))

    def test_a_meeting_that_recorded_nothing_says_so_at_error_with_the_reason(self):
        """The regression. The placeholder is still written - the upload path is built
        on there being a file - but "creating empty file" at INFO was the only trace
        that anything had gone wrong, and it named no cause at all.
        """
        self.start_with(FakeFfmpeg(returncode=1, said=b"Cannot open audio device\n"))
        self.recorder.stop_recording()

        with self.assertLogs("bots.bot_controller.screen_and_audio_recorder", level="ERROR") as logs:
            self.recorder.cleanup()

        self.assertIn("recorded nothing", "\n".join(logs.output))
        self.assertIn("Cannot open audio device", "\n".join(logs.output))
        self.assertEqual(os.path.getsize(self.file_location), 0, "the placeholder the upload path expects")

    def test_the_ffmpeg_log_does_not_outlive_the_meeting(self):
        """Every bot in this container shares one /tmp - see container_hygiene."""
        self.start_with(FakeFfmpeg(returncode=1, said=b"boom\n"))
        log_location = self.recorder.ffmpeg_log_location
        self.assertTrue(os.path.exists(log_location))

        self.recorder.cleanup()

        self.assertFalse(os.path.exists(log_location))

    def test_a_real_recording_is_left_exactly_where_it_was(self):
        recorder = ScreenAndAudioRecorder(self.file_location, RECORDING_DIMENSIONS, audio_only=True)
        with open(self.file_location, "wb") as recording:
            recording.write(b"a real recording")

        recorder.cleanup()

        self.assertEqual(os.path.getsize(self.file_location), len(b"a real recording"))

    def test_a_bot_recording_nothing_at_all_still_cleans_up(self):
        """``file_location`` is None when the pipeline records neither audio nor video."""
        ScreenAndAudioRecorder(None, RECORDING_DIMENSIONS, audio_only=False).cleanup()
