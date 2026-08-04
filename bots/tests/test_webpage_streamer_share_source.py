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
        with self.zoom_patch() as mock_zoom:
            controller.StopShare.return_value = SDKERR_WRONG_USAGE
            self.share_source.play_bot_output_media_stream("screenshare")
            with self.assertLogs(
                "bots.zoom_bot_adapter.webpage_streamer_share_source", level="INFO"
            ) as logs:
                self.share_source.stop_bot_output_media_stream()

        self.assertTrue(any("no share left to stop" in line for line in logs.output))
        self.assertFalse(self.share_source._sharing_started)
