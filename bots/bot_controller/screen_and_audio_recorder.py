import logging
import os
import subprocess
import time

logger = logging.getLogger(__name__)

# How much of FFmpeg's own complaint to put in the log when it dies. Its startup errors
# are a line or two ("Cannot open audio device", "Invalid argument"), and what carries
# them has a line length of its own: the first attempt logged a 2000-character tail and
# the hosting platform truncated it, so what survived was build flags and the error was
# the part that got cut. Last few lines, tightly capped, and the banner is turned off at
# the source.
FFMPEG_ERROR_TAIL_LINES = 6
FFMPEG_ERROR_TAIL_CHARS = 600

# How long to give FFmpeg to prove it opened its inputs. A device it cannot open is
# refused at startup, well inside this; anything still running after it has everything
# it asked for. Paid once per meeting, on a path that already sleeps 2s before it.
FFMPEG_STARTUP_GRACE_SECONDS = 2.0


class ScreenAndAudioRecorder:
    def __init__(self, file_location, recording_dimensions, audio_only):
        self.file_location = file_location
        self.ffmpeg_proc = None
        # Where ffmpeg's own stderr goes. It used to go to /dev/null, which is how a
        # recorder that never started looked exactly like one that recorded nothing: an
        # ffmpeg that exits on the first frame writes no output file, and the only trace
        # left was an empty mp4 uploaded an hour later with no explanation anywhere. A
        # file rather than a pipe because nothing here reads it while the meeting runs,
        # and a pipe nobody drains fills up and blocks the encoder.
        self.ffmpeg_log_location = f"{file_location}.ffmpeg.log" if file_location else None
        self.ffmpeg_log_file = None
        self.ffmpeg_command = None
        # Screen will have buffer, we will crop to the recording dimensions
        self.screen_dimensions = (recording_dimensions[0] + 10, recording_dimensions[1] + 10)
        self.recording_dimensions = recording_dimensions
        self.audio_only = audio_only
        self.paused = False
        self.xterm_proc = None
        self.video_degraded = False
        self.video_degradation_xterm_proc = None
        self.last_recording_file_size_check_time = time.time()

    def start_recording(self, display_var):
        logger.info(f"Starting screen recorder for display {display_var} with dimensions {self.screen_dimensions} and file location {self.file_location}")

        self._start_ffmpeg(display_var, with_audio=True)

        # A screen recording that has to have sound is not a screen recording. FFmpeg
        # takes both inputs in one process, so an ALSA device that will not open takes
        # the *video* down with it and the meeting is left with no file at all - which
        # is what was happening on every call here, invisibly, because the audio branch
        # is the fragile half (see webpage_streamer, which has carried a no-sound-card
        # path for exactly this reason) and nothing checked whether FFmpeg was still
        # alive. Retried without it, so a container with no capture device records a
        # silent screen instead of nothing.
        if self.audio_only or not self._ffmpeg_died_at_once():
            return
        logger.error(f"FFmpeg would not start with an audio input, so this meeting is being recorded silently. FFmpeg said: {self.ffmpeg_output() or '(nothing)'}")
        self._start_ffmpeg(display_var, with_audio=False)

    def _ffmpeg_died_at_once(self):
        """Whether FFmpeg gave up before it could have recorded anything.

        A failing input device is refused at startup, so this is a short wait and not a
        poll: an FFmpeg still running after it is one that opened everything it was
        asked for, and every later failure is somebody else's to notice.
        """
        if self.ffmpeg_proc is None:
            return True
        try:
            self.ffmpeg_proc.wait(timeout=FFMPEG_STARTUP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            return False
        return True

    def _start_ffmpeg(self, display_var, with_audio):
        if self.audio_only:
            # FFmpeg command for audio-only recording to MP3
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file without asking
                # Its stderr is read back and logged when something goes wrong, and a
                # 30-line build banner in front of a one-line error is what turns that
                # log into nothing: the useful part is what gets cut when a line is
                # truncated. Same pair DebugScreenRecorder already uses.
                "-hide_banner",
                "-loglevel",
                "error",
                "-thread_queue_size",
                "4096",
                "-f",
                "alsa",  # Audio input format for Linux
                "-i",
                "default",  # Default audio input device
                "-c:a",
                "libmp3lame",  # MP3 codec
                "-b:a",
                "192k",  # Audio bitrate (192 kbps for good quality)
                "-ar",
                "44100",  # Sample rate
                "-ac",
                "1",  # Mono
                self.file_location,
            ]
        else:
            ffmpeg_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-thread_queue_size", "256", "-framerate", "30", "-video_size", f"{self.screen_dimensions[0]}x{self.screen_dimensions[1]}", "-f", "x11grab", "-draw_mouse", "0", "-probesize", "32", "-i", display_var]
            if with_audio:
                ffmpeg_cmd += ["-thread_queue_size", "4096", "-f", "alsa", "-i", "default"]
            ffmpeg_cmd += ["-vf", f"crop={self.recording_dimensions[0]}:{self.recording_dimensions[1]}:10:10", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-g", "30"]
            ffmpeg_cmd += ["-c:a", "aac", "-strict", "experimental", "-b:a", "128k"] if with_audio else ["-an"]
            ffmpeg_cmd += [self.file_location]

        logger.info(f"Starting FFmpeg command: {' '.join(ffmpeg_cmd)}")
        self.ffmpeg_command = " ".join(ffmpeg_cmd)
        # Closed first: the retry without audio opens the same path again, and the
        # handle from the attempt that just died would otherwise be left dangling.
        self._close_ffmpeg_log()
        self.ffmpeg_log_file = self._open_ffmpeg_log()
        self.ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.DEVNULL,
            stderr=self.ffmpeg_log_file or subprocess.DEVNULL,
        )

    def _open_ffmpeg_log(self):
        """Somewhere for ffmpeg to complain, or None if we could not open one.

        Never fatal: a recorder that refuses to start because it has nowhere to write a
        log has turned a diagnostic into an outage, which is the wrong way round.
        """
        if not self.ffmpeg_log_location:
            return None
        try:
            return open(self.ffmpeg_log_location, "wb")
        except OSError as e:
            logger.warning(f"Could not open {self.ffmpeg_log_location} for FFmpeg's output: {e}")
            return None

    def ffmpeg_output(self):
        """The last few lines of what FFmpeg said, or "" when it said nothing readable.

        Joined with " | " rather than newlines: this goes into one log record, and a
        multi-line one is split across entries by most log viewers, which is how the
        one line that matters ends up somewhere other than the error it belongs to.
        """
        if not self.ffmpeg_log_location:
            return ""
        if self.ffmpeg_log_file is not None:
            try:
                self.ffmpeg_log_file.flush()
            except (OSError, ValueError):
                pass
        try:
            with open(self.ffmpeg_log_location, "rb") as log:
                said = log.read().decode("utf-8", "replace")
        except OSError:
            return ""
        lines = [line.strip() for line in said.splitlines() if line.strip()]
        return " | ".join(lines[-FFMPEG_ERROR_TAIL_LINES:])[-FFMPEG_ERROR_TAIL_CHARS:]

    def _close_ffmpeg_log(self):
        if self.ffmpeg_log_file is None:
            return
        try:
            self.ffmpeg_log_file.close()
        except OSError:
            pass
        self.ffmpeg_log_file = None

    def discard_ffmpeg_log(self):
        """Delete the log once it has been read into the bot's own log.

        Every bot in this container shares one /tmp, and a file per meeting that
        nothing ever reads again is the kind of leftover `container_hygiene` exists
        because of.
        """
        if not self.ffmpeg_log_location:
            return
        try:
            os.remove(self.ffmpeg_log_location)
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning(f"Could not remove {self.ffmpeg_log_location}: {e}")

    # Pauses by muting the audio and showing a black xterm covering the entire screen
    def pause_recording(self):
        if self.paused:
            return True  # Already paused, consider this success

        try:
            sw, sh = self.screen_dimensions

            x, y = 0, 0

            self.xterm_proc = subprocess.Popen(["xterm", "-bg", "black", "-fg", "black", "-geometry", f"{sw}x{sh}+{x}+{y}", "-xrm", "*borderWidth:0", "-xrm", "*scrollBar:false"])

            subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"], check=True)
            self.paused = True
            return True
        except Exception as e:
            logger.error(f"Failed to pause recording: {e}")
            return False

    # Resumes by unmuting the audio and killing the xterm proc
    def resume_recording(self):
        if not self.paused:
            return True

        try:
            self.xterm_proc.terminate()
            self.xterm_proc.wait()
            self.xterm_proc = None
            subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"], check=True)
            self.paused = False
            return True
        except Exception as e:
            logger.error(f"Failed to resume recording: {e}")
            return False

    # Checks the current recording file size and, if it exceeds the limit, degrades the
    # video recording by covering the screen with a black xterm window. This is permanent
    # for the rest of the recording; degradation is never stopped once it has started.
    def degrade_recording_if_file_size_exceeded(self, max_file_size_bytes):
        if not max_file_size_bytes:
            return
        if self.audio_only:
            return
        # Only check every 60 seconds
        if time.time() - self.last_recording_file_size_check_time < 60:
            return
        self.last_recording_file_size_check_time = time.time()

        if self.video_degraded:
            return

        if not self.file_location or not os.path.exists(self.file_location):
            return

        try:
            file_size = os.path.getsize(self.file_location)
        except OSError as e:
            logger.error(f"Failed to get recording file size: {e}")
            return

        if file_size <= max_file_size_bytes:
            logger.info(f"Recording file size {file_size} bytes is less than or equal to limit of {max_file_size_bytes} bytes, not degrading video recording")
            return

        logger.warning(f"Recording file size {file_size} bytes exceeds limit of {max_file_size_bytes} bytes, degrading video recording")

        try:
            sw, sh = self.screen_dimensions
            x, y = 0, 0
            self.video_degradation_xterm_proc = subprocess.Popen(["xterm", "-bg", "black", "-fg", "black", "-geometry", f"{sw}x{sh}+{x}+{y}", "-xrm", "*borderWidth:0", "-xrm", "*scrollBar:false"])
            self.video_degraded = True
        except Exception as e:
            logger.error(f"Failed to degrade video of recording: {e}")

    def stop_recording(self):
        if not self.ffmpeg_proc:
            return
        # Asked *before* terminating it, because that is the whole question: an FFmpeg
        # we stop is one that recorded the meeting, and an FFmpeg that was already gone
        # never recorded anything. Both used to log the same "Stopped ..." line below,
        # and terminating a corpse is a no-op, so the two were indistinguishable.
        died_on_its_own = self.ffmpeg_proc.poll() is not None
        self.ffmpeg_proc.terminate()
        self.ffmpeg_proc.wait()
        exit_code = self.ffmpeg_proc.returncode
        self.ffmpeg_proc = None
        if died_on_its_own:
            logger.error(f"FFmpeg exited on its own with code {exit_code} - this meeting recorded nothing. Command was: {self.ffmpeg_command}. FFmpeg said: {self.ffmpeg_output() or '(nothing)'}")
            return
        logger.info(f"Stopped screen and audio recorder for display with dimensions {self.screen_dimensions} and file location {self.file_location}")

    def get_seekable_path(self, path):
        """
        Transform a file path to include '.seekable' before the extension.
        Example: /tmp/file.webm -> /tmp/file.seekable.webm
        """
        base, ext = os.path.splitext(path)
        return f"{base}.seekable{ext}"

    def cleanup(self):
        input_path = self.file_location

        # If no input path at all, then we aren't trying to generate a file at all
        if input_path is None:
            self._close_ffmpeg_log()
            return

        # Check if input file exists
        if not os.path.exists(input_path):
            # The empty file is still written, because the upload path downstream is
            # built on there being one - but at ERROR and saying what FFmpeg said,
            # rather than the "creating empty file" note this used to be. That note
            # was the only trace of the failure that put a zero-byte mp4 behind a
            # customer-facing "watch this meeting back" button, and it named no cause
            # at all.
            logger.error(f"FFmpeg never wrote {input_path}, so this meeting recorded nothing and an empty placeholder is going up in its place. Command was: {self.ffmpeg_command}. FFmpeg said: {self.ffmpeg_output() or '(nothing)'}")
            with open(input_path, "wb"):
                pass  # Create empty file
            self._close_ffmpeg_log()
            self.discard_ffmpeg_log()
            return

        self._close_ffmpeg_log()
        self.discard_ffmpeg_log()

        # if audio only, we don't need to make it seekable
        if self.audio_only:
            return

        # if input file is greater than 3 GB, we will skip seekability
        if os.path.getsize(input_path) > 3 * 1024 * 1024 * 1024:
            logger.info("Input file is greater than 3 GB, skipping seekability")
            return

        output_path = self.get_seekable_path(self.file_location)
        # the file is seekable, so we don't need to make it seekable
        try:
            self.make_file_seekable(input_path, output_path)
        except Exception as e:
            logger.error(f"Failed to make file seekable: {e}")
            return

    def make_file_seekable(self, input_path, tempfile_path):
        """Use ffmpeg to move the moov atom to the beginning of the file."""
        logger.info(f"Making file seekable: {input_path} -> {tempfile_path}")
        # log how many bytes are in the file
        logger.info(f"File size: {os.path.getsize(input_path)} bytes")
        command = [
            "ffmpeg",
            "-i",
            str(input_path),  # Input file
            "-c",
            "copy",  # Copy streams without re-encoding
            "-avoid_negative_ts",
            "make_zero",  # Optional: Helps ensure timestamps start at or after 0
            "-movflags",
            "+faststart",  # Optimize for web playback
            "-y",  # Overwrite output file without asking
            str(tempfile_path),  # Output file
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed to make file seekable: {result.stderr}")

        # Replace the original file with the seekable version
        try:
            os.replace(str(tempfile_path), str(input_path))
            logger.info(f"Replaced original file with seekable version: {input_path}")
        except Exception as e:
            logger.error(f"Failed to replace original file with seekable version: {e}")
            raise RuntimeError(f"Failed to replace original file: {e}")
