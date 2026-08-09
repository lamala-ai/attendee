"""What the sending half of a screenshare says when it stops sending, and who gets told.

On 2026-08-09 a Zoom share ran for 76 seconds with a connected peer connection and
delivered zero frames. The receiving half said so, loudly, because it had just been
taught to count:

    The room is not seeing the shared page: Zoom accepted the share 15s ago and no
    frame has reached it since (0 received from the webpage streamer, 0 converted to
    I420, 0 rejected by sendShareFrame).

The sending half said nothing at all, and it is the half that knows. Two reasons:

* ``pull-sample`` blocks for ever, so a source producing nothing is indistinguishable
  from one about to produce something - the executor thread simply never came back.
* the GStreamer bus is read once, on the construction failure paths, and never again.
  A pipeline that dies *after* reaching PLAYING puts its error there and nowhere else.

Proving the capture before an offer is answered fixed the second of those at the one
moment a share begins. It did not fix it for a share that begins healthy and dies in the
middle, which is what these tests cover - and it introduced a hazard of its own, because
``timed_pop_filtered`` **removes** what it returns. Two independent readers of one bus do
not share the error between them: the first takes it and the second finds a clean bus and
concludes the pipeline is fine. So there is exactly one reader here, and what it learns is
remembered rather than handed to whoever asked first.
"""

import asyncio
from unittest.mock import MagicMock

from django.test import SimpleTestCase

from bots.webpage_streamer.webpage_streamer import (
    CAPTURE_FRAME_DEADLINE_SECONDS,
    Gst,
    GstVideoStreamTrack,
    WebpageStreamer,
)

# Big enough that av's plane buffers are exactly w*h and w*h/4, which is what recv()
# slices the GStreamer buffer into. Below a 32-aligned width av pads the planes and the
# slices no longer fill them - 4x2 asks for 8 bytes into a 32-byte plane. The real
# capture is 1280x720, which is aligned, so this is a property of the fixture rather
# than of the code under test.
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

    ``None`` is what ``try_pull_sample`` returns on timeout, which is the case that used
    to be unreachable: with ``pull-sample`` the call simply never returned.
    """

    def __init__(self, samples):
        self.samples = list(samples)
        self.timeouts_asked_for = []

    def try_pull_sample(self, timeout_ns):
        self.timeouts_asked_for.append(timeout_ns)
        return self.samples.pop(0)

    def emit(self, signal, *args):
        raise AssertionError(f"a blocking pull is the bug: {signal}")


class FakeBus:
    """A bus that hands each message over once, because a real one does the same.

    That single detail is the reason this file exists: a message read is a message gone.
    """

    def __init__(self, *messages):
        self.messages = list(messages)
        self.filters = []

    def timed_pop_filtered(self, timeout, message_types):
        self.filters.append(message_types)
        return self.messages.pop(0) if self.messages else None


class FakePipeline:
    def __init__(self, bus):
        self._bus = bus

    def get_bus(self):
        return self._bus

    def set_state(self, state):
        return Gst.StateChangeReturn.SUCCESS


def an_error(text="ximagesrc: could not open display", debug="x.c(1)"):
    message = MagicMock()
    message.type = Gst.MessageType.ERROR
    message.parse_error.return_value = (MagicMock(message=text), debug)
    return message


def a_warning(text="alsasrc: no such device", debug="alsa.c(1)"):
    message = MagicMock()
    message.type = Gst.MessageType.WARNING
    message.parse_warning.return_value = (MagicMock(message=text), debug)
    return message


def an_eos():
    message = MagicMock()
    message.type = Gst.MessageType.EOS
    return message


def a_streamer(bus):
    streamer = WebpageStreamer(video_frame_size=(WIDTH, HEIGHT))
    streamer._gst_pipeline = FakePipeline(bus)
    return streamer


def track_over(sink, fault=None):
    return GstVideoStreamTrack(sink=sink, width=WIDTH, height=HEIGHT, fault=fault)


class VideoCaptureStallTestCase(SimpleTestCase):
    """A live track that stops receiving frames, and what it does about it."""

    def test_the_pull_is_bounded_rather_than_forever(self):
        """The property the whole change rests on: a pull that can time out.

        Against the old code this fails in FakeSink.emit - it emitted ``pull-sample``,
        which has no deadline and never returned.
        """
        sink = FakeSink([a_sample()])
        frame = asyncio.run(track_over(sink).recv())

        assert frame.width == WIDTH
        assert sink.timeouts_asked_for == [CAPTURE_FRAME_DEADLINE_SECONDS * Gst.SECOND]

    def test_a_silent_stall_is_reported_once_and_then_waited_out(self):
        """No fault means the capture is running and producing nothing.

        Worth a line and not worth ending the track over: the display may genuinely be
        idle for a moment, and ending it would take down a share that is about to
        recover. What must not happen is the old behaviour - silence for the length of
        the meeting.
        """
        sink = FakeSink([None, None, a_sample()])
        streamer = a_streamer(FakeBus())

        with self.assertLogs("bots.webpage_streamer.webpage_streamer", level="WARNING") as logs:
            frame = asyncio.run(track_over(sink, streamer.capture_fault).recv())

        assert frame.width == WIDTH
        stalls = [line for line in logs.output if "No video captured" in line]
        assert len(stalls) == 1, f"said once, not once per pull: {logs.output}"
        assert "frozen or black" in stalls[0]

    def test_a_dead_pipeline_ends_the_track_instead_of_hanging_the_receiver(self):
        """The receiver is parked on recv(). Ending it is the signal it already handles.

        ``_consume_video`` logs "Webpage streamer video track ended" and clears its
        frame, so the meeting stops being told a live share is on its way.
        """
        sink = FakeSink([None])
        streamer = a_streamer(FakeBus(an_error()))

        with self.assertLogs("bots.webpage_streamer.webpage_streamer", level="ERROR") as logs:
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(track_over(sink, streamer.capture_fault).recv())

        assert any("will not recover" in line for line in logs.output)
        assert any("could not open display" in line for line in logs.output), "the element that objected has to survive into the log - that is the sentence the bus exists to provide"

    def test_end_of_stream_is_a_death_too_and_says_which_one(self):
        sink = FakeSink([None])
        streamer = a_streamer(FakeBus(an_eos()))

        with self.assertLogs("bots.webpage_streamer.webpage_streamer", level="ERROR") as logs:
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(track_over(sink, streamer.capture_fault).recv())

        assert any("end of stream" in line for line in logs.output)

    def test_a_warning_is_logged_without_ending_the_share(self):
        """Elements warn about things they go on working through.

        The container with no sound card warns on every start, including the start that
        went on to deliver 908 frames. Ending a track over one would make this change a
        new way to lose a picture rather than a way to explain losing one.
        """
        sink = FakeSink([None, a_sample()])
        streamer = a_streamer(FakeBus(a_warning()))

        with self.assertLogs("bots.webpage_streamer.webpage_streamer", level="WARNING") as logs:
            frame = asyncio.run(track_over(sink, streamer.capture_fault).recv())

        assert frame.width == WIDTH
        assert any("no such device" in line for line in logs.output)

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


class OneReaderOfTheBusTestCase(SimpleTestCase):
    """Whoever asks second must get the same answer as whoever asked first.

    Proving the capture before answering an offer put a second consumer on a queue that
    hands each message over once. These pin the property that keeps the two from
    cancelling each other out.
    """

    def test_a_fault_another_reader_already_took_still_ends_the_track(self):
        """The regression this rework exists for.

        ``/offer`` and ``/restart_capture`` both drain the bus. If the live track went to
        the bus itself, an error drained a moment earlier by either of them would be gone
        by the time the track stalled - and the track would call a dead pipeline a stall,
        log a warning, and wait for the rest of the meeting. Which is the behaviour the
        stall handling was written to remove.
        """
        streamer = a_streamer(FakeBus(an_error()))

        # Somebody else empties the bus first - an offer being answered, or a bot asking
        # for a restart. The real bus now has nothing left on it.
        assert "could not open display" in streamer.capture_fault()

        sink = FakeSink([None])
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(track_over(sink, streamer.capture_fault).recv())

    def test_the_fault_is_remembered_rather_than_reread(self):
        streamer = a_streamer(FakeBus(an_error()))

        first = streamer.capture_fault()
        assert first, "the bus had an error on it"
        assert streamer.capture_fault() == first
        assert streamer.capture_fault() == first

    def test_a_warning_does_not_become_a_fault_for_anyone(self):
        streamer = a_streamer(FakeBus(a_warning()))

        assert streamer.capture_fault() == ""

    def test_the_first_fault_is_the_one_that_is_kept(self):
        """An element that fails takes the rest down with it, and the cascade is noise.

        The first sentence is the one naming what actually went wrong.
        """
        streamer = a_streamer(FakeBus(an_error("ximagesrc: could not open display"), an_error("videoconvert: not negotiated")))

        assert "could not open display" in streamer.capture_fault()

    def test_a_pipeline_with_nothing_to_say_reports_no_fault(self):
        assert a_streamer(FakeBus()).capture_fault() == ""

    def test_a_rebuilt_pipeline_does_not_inherit_the_last_ones_fault(self):
        """Otherwise the recovery ends the very track it just built.

        ``_capture_ready_for_a_new_connection`` rebuilds on a capture that produced
        nothing, and the pipeline it tears down is usually the one that posted the error.
        """
        streamer = a_streamer(FakeBus(an_error()))
        assert streamer.capture_fault()

        streamer._stop_gstreamer_capture()
        streamer._gst_pipeline = FakePipeline(FakeBus())

        assert streamer.capture_fault() == ""

    def test_the_bus_is_asked_about_errors_warnings_and_eos_together(self):
        """All three in one pop. Asking only about ERROR is how EOS became a silent hang."""
        streamer = a_streamer(FakeBus())
        streamer.capture_fault()

        assert streamer._gst_pipeline._bus.filters, "nothing read the bus after PLAYING - that was the bug"
        wanted = Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.EOS
        assert all(f == wanted for f in streamer._gst_pipeline._bus.filters)
