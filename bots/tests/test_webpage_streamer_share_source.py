"""The Zoom native share source: registering it, and taking it down.

The share source shipped with a call that could never have succeeded, and nothing
caught it until a real meeting did:

    TypeError: setExternalShareSource(): incompatible function arguments.
      1. setExternalShareSource(self, arg0: IZoomSDKShareSource,
                                arg1: IZoomSDKShareAudioSource, /) -> SDKError
    Invoked with types: IZoomSDKShareSourceHelper, ShareSourceCallbacks

In C++ the audio source defaults to ``nullptr``, but the binding is a bare ``.def()``
with no ``nb::arg(...) = nullptr``, so the default does not reach Python and both
parameters are mandatory.

``FakeShareSourceHelper`` therefore declares **two positional-only, required**
parameters, exactly like the real binding. That is the whole point of this file: a
helper built out of a plain ``MagicMock`` would have accepted the broken one-argument
call and the test would have passed while the room saw nothing.
"""

from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from bots.zoom_bot_adapter.webpage_streamer_share_source import WebpageStreamerShareSource

SDKERR_SUCCESS = 0
SDKERR_WRONG_USAGE = 7


class FakeShareSourceHelper:
    """IZoomSDKShareSourceHelper, with the binding's real signature."""

    def __init__(self):
        self.calls = []

    def setExternalShareSource(self, share_source, share_audio_source, /):
        self.calls.append((share_source, share_audio_source))
        return SDKERR_SUCCESS


class MockShareSourceCallbacks:
    def __init__(self, onStartSendCallback=None, onStopSendCallback=None):
        self.onStartSendCallback = onStartSendCallback
        self.onStopSendCallback = onStopSendCallback


class MockShareAudioCallbacks:
    def __init__(self, onStartSendAudioCallback=None, onStopSendAudioCallback=None):
        self.onStartSendAudioCallback = onStartSendAudioCallback
        self.onStopSendAudioCallback = onStopSendAudioCallback


def create_mock_zoom_sdk_for_sharing(helper):
    def factory():
        mock = MagicMock()
        mock.ShareSourceCallbacks = MockShareSourceCallbacks
        mock.ShareAudioCallbacks = MockShareAudioCallbacks
        mock.SDKERR_SUCCESS = SDKERR_SUCCESS
        mock.SDKERR_WRONG_USAGE = SDKERR_WRONG_USAGE
        mock.GetRawdataShareSourceHelper = MagicMock(return_value=helper)
        return mock

    return factory


class WebpageStreamerShareSourceTestCase(SimpleTestCase):
    def setUp(self):
        self.helper = FakeShareSourceHelper()
        self.meeting_service = MagicMock()
        self.share_source = WebpageStreamerShareSource(
            meeting_service=self.meeting_service,
            schedule_on_main_thread=MagicMock(return_value=1),
            unschedule_on_main_thread=MagicMock(),
        )

    def zoom_patch(self):
        return patch(
            "bots.zoom_bot_adapter.webpage_streamer_share_source.zoom",
            new_callable=create_mock_zoom_sdk_for_sharing(self.helper),
        )


class TestRegisteringTheShareSource(WebpageStreamerShareSourceTestCase):
    def test_the_share_source_is_registered_with_an_audio_source_as_well(self):
        """The regression. One argument raises TypeError against the real binding, the
        share never starts, and the room sees nothing while everything upstream of this
        call reports success."""
        with self.zoom_patch():
            self.share_source.play_bot_output_media_stream("screenshare")

        self.assertEqual(len(self.helper.calls), 1)
        share_source, audio_source = self.helper.calls[0]
        self.assertIsInstance(share_source, MockShareSourceCallbacks)
        self.assertIsInstance(audio_source, MockShareAudioCallbacks)
        self.assertTrue(self.share_source._sharing_started)

    def test_the_audio_source_is_held_so_the_sdk_pointer_stays_valid(self):
        """The SDK keeps a raw pointer to it; a local would be collected."""
        with self.zoom_patch():
            self.share_source.play_bot_output_media_stream("screenshare")

        self.assertIs(self.share_source.share_audio_callbacks, self.helper.calls[0][1])

    def test_the_registered_audio_source_never_sends_anything(self):
        """Share audio is declined on purpose - sendShareAudio rejects every documented
        format on Linux and a rendered page has nothing to play. It is declined by
        supplying a silent source, which is not the same as omitting the argument."""
        with self.zoom_patch():
            self.share_source.play_bot_output_media_stream("screenshare")

        audio_source = self.share_source.share_audio_callbacks
        audio_source.onStartSendAudioCallback(MagicMock())  # must not raise or send
        audio_source.onStopSendAudioCallback()

    def test_a_destination_that_is_not_the_screenshare_registers_nothing(self):
        with self.zoom_patch():
            self.share_source.play_bot_output_media_stream("webcam")

        self.assertEqual(self.helper.calls, [])
        self.assertFalse(self.share_source._sharing_started)

    def test_registering_twice_is_a_no_op(self):
        with self.zoom_patch():
            self.share_source.play_bot_output_media_stream("screenshare")
            self.share_source.play_bot_output_media_stream("screenshare")

        self.assertEqual(len(self.helper.calls), 1)

    def test_a_missing_helper_gives_up_rather_than_raising(self):
        with self.zoom_patch() as mock_zoom:
            mock_zoom.GetRawdataShareSourceHelper = MagicMock(return_value=None)
            self.share_source.play_bot_output_media_stream("screenshare")

        self.assertFalse(self.share_source._sharing_started)


class TestStoppingTheShare(WebpageStreamerShareSourceTestCase):
    def test_stopping_uses_the_share_controller_rather_than_a_null_source(self):
        """``setExternalShareSource(None)`` cannot express "clear it" through this
        binding: both arguments are mandatory, and no binding in this module declares
        ``nb::arg().none()``, so None is refused too. ``StopShare`` takes no arguments
        and is what the SDK documents for ending a share."""
        self.meeting_service.GetMeetingShareController.return_value.StopShare.return_value = SDKERR_SUCCESS
        with self.zoom_patch():
            self.share_source.play_bot_output_media_stream("screenshare")
            self.share_source.stop_bot_output_media_stream()

        controller = self.meeting_service.GetMeetingShareController.return_value
        controller.StopShare.assert_called_once_with()
        self.assertEqual(len(self.helper.calls), 1)  # not called again to "clear" it
        self.assertFalse(self.share_source._sharing_started)
        self.assertIsNone(self.share_source.share_sender)

    def test_stopping_a_share_that_never_started_touches_nothing(self):
        with self.zoom_patch():
            self.share_source.stop_bot_output_media_stream()

        self.meeting_service.GetMeetingShareController.assert_not_called()

    def test_a_share_zoom_already_ended_is_not_stopped_again(self):
        """Observed as `StopShare result = SDKError.SDKERR_WRONG_USAGE` at the end of a
        real meeting. The meeting finishing stops the share and tears the bot down at
        nearly the same moment, so teardown was asking Zoom to stop something that had
        already stopped. onStopSend is Zoom telling us it is over, whoever ended it, and
        that settles it.

        Fails against the old behaviour, where the callback left _sharing_started set.
        """
        with self.zoom_patch():
            self.share_source.play_bot_output_media_stream("screenshare")
            self.share_source.on_share_stop_send_callback()  # the meeting ended
            self.share_source.stop_bot_output_media_stream()

        self.assertFalse(self.share_source._sharing_started)
        self.meeting_service.GetMeetingShareController.assert_not_called()

    def test_wrong_usage_from_stop_share_is_not_reported_as_a_failure(self):
        """It means "there was no current sharing", which is the state being asked for."""
        controller = self.meeting_service.GetMeetingShareController.return_value
        with self.zoom_patch():
            controller.StopShare.return_value = SDKERR_WRONG_USAGE
            self.share_source.play_bot_output_media_stream("screenshare")
            with self.assertLogs("bots.zoom_bot_adapter.webpage_streamer_share_source", level="INFO") as logs:
                self.share_source.stop_bot_output_media_stream()

        self.assertTrue(any("no share left to stop" in line for line in logs.output))
        self.assertFalse(self.share_source._sharing_started)


class FakeShareSender:
    """The object Zoom hands to onStartSend. Records what it was asked to send."""

    def __init__(self, result=SDKERR_SUCCESS):
        self.result = result
        self.frames = []

    def sendShareFrame(self, frame_bytes, width, height, frame_format):
        self.frames.append((frame_bytes, width, height, frame_format))
        return self.result


class FakeClock:
    """Stands in for the module's ``time``. It only ever calls ``monotonic``."""

    def __init__(self, now=1000.0):
        self.now = now

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class TestSayingWhenTheRoomSeesNothing(WebpageStreamerShareSourceTestCase):
    """The share that registered, started, and delivered nothing.

    On 2026-08-09 a Zoom meeting watched a black screen for the length of a share while
    every log line on the path said success: the page was fetched and long-polled by the
    renderer throughout, setExternalShareSource returned SDKERR_SUCCESS, onStartSend
    fired, and the pump was scheduled. The pump then ran roughly 2,300 times over 76
    seconds and logged nothing at all, because both of its give-up paths - no sender, no
    frame - are a bare ``return True``, and the only outcome it reports is a
    sendShareFrame that fails. There was no way to tell from the logs whether frames were
    never arriving, never converting, or being sent and dropped by Zoom.

    Every test here fails against that behaviour: none of these lines existed.
    """

    def setUp(self):
        super().setUp()
        self.clock = FakeClock()
        self.sender = FakeShareSender()

    def start_sharing(self):
        with self.zoom_patch():
            self.share_source.play_bot_output_media_stream("screenshare")
        self.share_source.on_share_start_send_callback(self.sender)

    def pump(self, times=1):
        with self.zoom_patch():
            for _ in range(times):
                self.share_source._pump_frame()

    def clock_patch(self):
        return patch("bots.zoom_bot_adapter.webpage_streamer_share_source.time", self.clock)

    def logs(self, level="INFO"):
        return self.assertLogs("bots.zoom_bot_adapter.webpage_streamer_share_source", level=level)

    def test_a_share_delivering_nothing_says_so_once_the_deadline_passes(self):
        with self.clock_patch():
            self.start_sharing()
            self.clock.advance(16)  # SHARE_FIRST_FRAME_DEADLINE_SECONDS is 15
            with self.logs(level="WARNING") as logs:
                self.pump()

        self.assertTrue(any("The room is not seeing the shared page" in line for line in logs.output))

    def test_nothing_is_said_before_the_deadline(self):
        """A share is entitled to a cold browser start and a first keyframe."""
        with self.clock_patch():
            self.start_sharing()
            self.clock.advance(5)
            self.pump(times=150)  # ~5s of ticks, all of them empty

        self.assertFalse(self.share_source._warned_about_no_frames)

    def test_the_warning_says_no_frames_arrived_over_webrtc(self):
        """Nothing received is the receiving half - aiortc or the connection itself."""
        with self.clock_patch():
            self.start_sharing()
            self.clock.advance(16)
            with self.logs(level="WARNING") as logs:
                self.pump()

        self.assertIn("0 received from the webpage streamer", logs.output[0])
        self.assertIn("0 converted to I420", logs.output[0])

    def test_the_warning_distinguishes_frames_that_arrived_but_never_converted(self):
        """Received climbing with converted stuck at zero is the I420 conversion, which
        is a different bug in a different thread from an empty connection."""
        self.share_source.latest_frame.note_received()
        self.share_source.latest_frame.note_received()
        with self.clock_patch():
            self.start_sharing()
            self.clock.advance(16)
            with self.logs(level="WARNING") as logs:
                self.pump()

        self.assertIn("2 received from the webpage streamer", logs.output[0])
        self.assertIn("0 converted to I420", logs.output[0])

    def test_the_warning_counts_frames_zoom_rejected(self):
        """Converted and sent but refused is the third half, and the pump's existing
        rate-limited line only ever shows the code, never how many."""
        self.sender.result = SDKERR_WRONG_USAGE
        self.share_source.latest_frame.put(b"i420", 1280, 720)
        with self.clock_patch():
            self.start_sharing()
            self.clock.advance(16)
            with self.logs(level="WARNING") as logs:
                self.pump()

        self.assertIn("1 converted to I420", logs.output[-1])
        self.assertIn("rejected by sendShareFrame", logs.output[-1])

    def test_the_warning_is_said_once_however_long_it_goes_on(self):
        """At 30fps a per-tick warning would be the whole log."""
        with self.clock_patch():
            self.start_sharing()
            self.clock.advance(60)
            with self.logs(level="WARNING") as logs:
                self.pump(times=500)

        self.assertEqual(len([line for line in logs.output if "not seeing" in line]), 1)

    def test_a_share_that_is_working_is_never_warned_about(self):
        self.share_source.latest_frame.put(b"i420", 1280, 720)
        with self.clock_patch():
            self.start_sharing()
            self.pump()
            self.clock.advance(600)
            self.pump(times=100)

        self.assertFalse(self.share_source._warned_about_no_frames)
        self.assertEqual(len(self.sender.frames), 101)

    def test_the_first_frame_zoom_accepts_is_announced(self):
        """Until this line appears, every success reported on this path is about
        registering a share rather than filling one."""
        self.share_source.latest_frame.put(b"i420", 1280, 720)
        with self.clock_patch():
            self.start_sharing()
            with self.logs() as logs:
                self.pump()

        self.assertTrue(any("First frame accepted by Zoom: 1280x720" in line for line in logs.output))

    def test_the_first_frame_is_announced_only_once(self):
        self.share_source.latest_frame.put(b"i420", 1280, 720)
        with self.clock_patch():
            self.start_sharing()
            with self.logs() as logs:
                self.pump(times=50)

        self.assertEqual(len([line for line in logs.output if "First frame accepted" in line]), 1)

    def test_a_second_share_is_judged_on_its_own_delivery(self):
        """The counters are reset by onStartSend, so a first share that worked cannot
        vouch for a second one that does not."""
        self.share_source.latest_frame.put(b"i420", 1280, 720)
        with self.clock_patch():
            self.start_sharing()
            self.pump()
            self.share_source.on_share_stop_send_callback()
            self.share_source.latest_frame.clear()
            self.share_source.on_share_start_send_callback(self.sender)
            self.clock.advance(16)
            with self.logs(level="WARNING") as logs:
                self.pump()

        self.assertTrue(any("The room is not seeing the shared page" in line for line in logs.output))


class TestCountingFrames(WebpageStreamerShareSourceTestCase):
    def test_receiving_and_converting_are_counted_separately(self):
        """Both are needed to place a failure: the receiver counts a frame off the track
        before touching it, and only a successful conversion counts as converted."""
        frame = self.share_source.latest_frame
        self.assertEqual(frame.counts(), (0, 0))

        frame.note_received()
        self.assertEqual(frame.counts(), (1, 0))  # arrived, conversion still to come

        frame.put(b"i420", 1280, 720)
        self.assertEqual(frame.counts(), (1, 1))

    def test_the_counts_survive_the_frame_being_cleared(self):
        """clear() drops the buffer at the end of a track; it is not a fresh share, and
        the counts are the record of what that share did."""
        frame = self.share_source.latest_frame
        frame.note_received()
        frame.put(b"i420", 1280, 720)
        frame.clear()

        self.assertEqual(frame.counts(), (1, 1))
