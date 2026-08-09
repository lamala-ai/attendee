"""What the sending half of a screenshare says when it stops sending.

On 2026-08-09 a Zoom share ran for 76 seconds with a connected peer connection and
delivered zero frames. The receiving half said so, loudly, because #12 had just taught
it to count:

    The room is not seeing the shared page: Zoom accepted the share 15s ago and no
    frame has reached it since (0 received from the webpage streamer, 0 converted to
    I420, 0 rejected by sendShareFrame).

The sending half said nothing at all, and it is the half that knows. Two reasons, both
fixed here and both tested below:

* ``pull-sample`` blocks for ever, so a source producing nothing is indistinguishable
  from one about to produce something - the executor thread simply never came back.
* the GStreamer bus is read once, on the construction failure paths, and never again.
  A pipeline that dies *after* reaching PLAYING puts its error there and nowhere else.

So a stall is now bounded, reported once, and - when the pipeline is actually dead -
ends the track, which is the one signal the receiver is already listening for.
"""

import asyncio
from unittest.mock import MagicMock

from django.test import SimpleTestCase

from bots.webpage_streamer.webpage_streamer import (
    VIDEO_STALL_DEADLINE_SECONDS,
    Gst,
    GstVideoStreamTrack,
)

# Big enough that av's plane buffers are exactly w*h and w*h/4, which is what recv()
# slices the GStreamer buffer into. Below a 32-aligned width av pads the planes and
# the slices no longer fill them - 4x2 asks for 8 bytes into a 32-byte plane. The
# real capture is 1280x720, which is aligned, so this is a property of the fixture
# rather than of the code under test.
WIDTH, HEIGHT = 64, 32


def a_sample():
    """A Gst sample carrying one tiny I420 frame, mapped the way recv() maps it."""
    payload = bytes(WIDTH * HEIGHT * 3 // 2)
    mapinfo = MagicMock()
    mapinfo.data = payload

    buffer = MagicMock()
    buffer.pts = 0
    buffer.map.return_value = (True, mapinfo)

    sample = MagicMock()
    sample.get_buffer.return_value = buffer
    return sample


class FakeSink:
    """An appsink that hands back a scripted series of pulls.

    ``None`` is what ``try-pull-sample`` returns on timeout, which is the case that used
    to be unreachable: with ``pull-sample`` the call simply never returned.
    """

    def __init__(self, samples):
        self.samples = list(samples)
        self.timeouts_asked_for = []

    def emit(self, signal, *args):
        assert signal == "try-pull-sample", f"a blocking pull is the bug: {signal}"
        self.timeouts_asked_for.append(args[0])
        return self.samples.pop(0)


class FakeBus:
    def __init__(self, message=None):
        self.message = message
        self.filters = []

    def timed_pop_filtered(self, timeout, message_types):
        self.filters.append(message_types)
        message, self.message = self.message, None
        return message


class FakePipeline:
    def __init__(self, bus):
        self._bus = bus

    def get_bus(self):
        return self._bus


def an_error(text="ximagesrc: could not open display", debug="x.c(1)"):
    message = MagicMock()
    message.type = Gst.MessageType.ERROR
    message.parse_error.return_value = (MagicMock(message=text), debug)
    return message


def an_eos():
    message = MagicMock()
    message.type = Gst.MessageType.EOS
    return message


def track_over(sink, pipeline=None):
    return GstVideoStreamTrack(sink=sink, width=WIDTH, height=HEIGHT, pipeline=pipeline)


class VideoCaptureStallTestCase(SimpleTestCase):
    def test_the_pull_is_bounded_rather_than_forever(self):
        """The property the whole change rests on: a pull that can time out.

        Against the old code this fails on the assert inside FakeSink - it emitted
        ``pull-sample``, which has no deadline and never returned.
        """
        sink = FakeSink([a_sample()])
        frame = asyncio.run(track_over(sink).recv())

        assert frame.width == WIDTH
        assert sink.timeouts_asked_for == [VIDEO_STALL_DEADLINE_SECONDS * Gst.SECOND]

    def test_a_silent_stall_is_reported_once_and_then_waited_out(self):
        """No error on the bus means the capture is running and producing nothing.

        Worth a line and not worth ending the track over: the display may genuinely be
        idle for a moment, and ending it would take down a share that is about to
        recover. What must not happen is the old behaviour - silence for the length of
        the meeting.
        """
        sink = FakeSink([None, None, a_sample()])
        bus = FakeBus()

        with self.assertLogs("bots.webpage_streamer.webpage_streamer", level="WARNING") as logs:
            frame = asyncio.run(track_over(sink, FakePipeline(bus)).recv())

        assert frame.width == WIDTH
        stalls = [line for line in logs.output if "No video captured" in line]
        assert len(stalls) == 1, f"said once, not once per pull: {logs.output}"
        assert "seeing a frozen" in stalls[0]

    def test_a_dead_pipeline_ends_the_track_instead_of_hanging_the_receiver(self):
        """The receiver is parked on recv(). Ending it is the signal it already handles.

        ``_consume_video`` logs "Webpage streamer video track ended" and clears its
        frame, so the meeting stops being told a live share is on its way.
        """
        sink = FakeSink([None])
        bus = FakeBus(an_error())

        with self.assertLogs("bots.webpage_streamer.webpage_streamer", level="ERROR") as logs:
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(track_over(sink, FakePipeline(bus)).recv())

        assert any("will not recover" in line for line in logs.output)
        assert any("could not open display" in line for line in logs.output), "the element that objected has to survive into the log - that is the sentence the bus exists to provide"

    def test_end_of_stream_is_a_death_too_and_says_which_one(self):
        sink = FakeSink([None])
        bus = FakeBus(an_eos())

        with self.assertLogs("bots.webpage_streamer.webpage_streamer", level="ERROR") as logs:
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(track_over(sink, FakePipeline(bus)).recv())

        assert any("end-of-stream" in line for line in logs.output)

    def test_the_bus_is_asked_about_errors_and_eos_together(self):
        """Both, in one pop. Asking only about ERROR is how EOS became a silent hang."""
        sink = FakeSink([None, a_sample()])
        bus = FakeBus()
        asyncio.run(track_over(sink, FakePipeline(bus)).recv())

        assert bus.filters, "nothing read the bus after PLAYING - that was the bug"
        wanted = Gst.MessageType.ERROR | Gst.MessageType.EOS
        assert all(f == wanted for f in bus.filters)

    def test_the_first_frame_sent_is_logged_so_the_two_halves_can_be_told_apart(self):
        """Pairs with "First video frame received from the webpage streamer" on the bot.

        This line and not that one places the fault in WebRTC; neither line places it in
        capture. Without it, "0 received" is where the trail stops.
        """
        sink = FakeSink([a_sample(), a_sample()])
        track = track_over(sink)

        with self.assertLogs("bots.webpage_streamer.webpage_streamer", level="INFO") as logs:
            asyncio.run(track.recv())
            asyncio.run(track.recv())

        firsts = [line for line in logs.output if "First video frame captured" in line]
        assert len(firsts) == 1, f"the first frame only: {logs.output}"
        assert f"{WIDTH}x{HEIGHT}" in firsts[0]
