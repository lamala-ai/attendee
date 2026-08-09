"""What the bot requires of a stream before it will put it on the room's screen.

The webpage streamer captures a web page. On a host with no sound card - which is
every container that has not been given one - `gstalsasrc` cannot open a device and
`_start_gstreamer_capture` deliberately falls back to a video-only pipeline, saying so
in as many words:

    WARNING  GStreamer pipeline with audio would not start
             (Could not open audio device for recording ... gstalsasrc)
             - capturing video only
    INFO     GStreamer capture pipeline is PLAYING (video only)

That is a supported configuration; a shared web page has no audio to carry. But the
page inside the bot would only render a stream that had **both** a video and an audio
track, so the answer it got was one it silently refused:

    botOutputMediaStreamIsReady() {
        return ....getVideoTracks().length > 0 && ....getAudioTracks().length > 0;
    }

`playBotOutputMediaStream` checked that, found it false, armed a one-second interval to
re-check, and returned. The interval re-checked for the length of the meeting. Nothing
threw, and everything upstream reported success - `Playing bot output media stream to
screenshare` is logged by the manager *before* the handoff, `/start_streaming` answered
200, ICE completed - so three consecutive meetings were told they were screensharing
while the room looked at nothing.

There is no JavaScript test runner in this repo and no Node in the image, so these read
the payload rather than executing it. That is a real limit: they pin the two conditions
the failure needed and cannot prove the page works. They would have caught this.
"""

import re
from pathlib import Path

from django.test import SimpleTestCase

PAYLOAD = Path(__file__).resolve().parent.parent / "web_bot_adapter" / "shared_chromedriver_payload.js"


def body_of(function_name: str) -> str:
    """The source of one method, from its signature to the matching closing brace."""
    source = PAYLOAD.read_text()
    start = source.index(f"{function_name}(")
    depth, index = 0, source.index("{", start)
    for index in range(index, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                break
    return source[start : index + 1]


def code_only(source: str) -> str:
    """The source with comments stripped, so prose about a rule is not read as the rule."""
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return "\n".join(line.split("//")[0] for line in source.splitlines())


class BotOutputMediaStreamGateTests(SimpleTestCase):
    def test_a_video_only_stream_is_ready_to_render(self):
        gate = code_only(body_of("botOutputMediaStreamIsReady"))

        self.assertIn("getVideoTracks", gate)
        self.assertNotIn(
            "getAudioTracks",
            gate,
            "the readiness gate requires audio again - a video-only webpage stream, which is what a host with no sound card produces, will never be rendered",
        )

    def test_the_audio_wiring_is_skipped_when_there_is_no_audio(self):
        """createMediaStreamSource throws InvalidStateError on a stream with no audio
        tracks. Relaxing the gate above without this only moves the failure one line
        down, into a catch that reports and then drops the video."""
        play = code_only(body_of("async playMediaStream"))
        wiring = play.index("createMediaStreamSource")
        guard = play.rindex("getAudioTracks().length", 0, wiring)

        self.assertGreater(wiring, guard, "createMediaStreamSource is reached without checking for audio")

    def test_waiting_for_a_stream_that_never_comes_gives_up_and_says_so(self):
        """An unbounded wait is indistinguishable from a stream still on its way, which
        is how this went unnoticed: the only evidence was an absence."""
        play = code_only(body_of("async playBotOutputMediaStream"))

        self.assertIn("BOT_OUTPUT_MEDIA_STREAM_WAIT_SECONDS", play)
        self.assertIn("clearInterval", play)
        self.assertIn("BOT_OUTPUT_MEDIA_STREAM_NEVER_ARRIVED", play)
