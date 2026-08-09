"""What a Zoom bot says while it is being taken apart.

A live meeting on 2026-08-09 ran clean and then ended, and the last two seconds of its
logs read like an outage:

    send_current_image_to_zoom failed with send_video_frame_response = SDKError.SDKERR_WRONG_USAGE
    Error while refreshing video subscriptions
      AttributeError: 'NoneType' object has no attribute 'IsVideoOn'
    audio_helper.unSubscribe() returned SDKError.SDKERR_INTERNAL_ERROR
    Error in Redis listener: <class 'ValueError'> I/O operation on closed file.

Four lines, four unrelated subsystems, nothing actually wrong with any of them. Every
one is the same mistake in a different place: something that runs on its own clock kept
running after the thing it talks to had gone, and reported the resulting refusal as if a
meeting in progress had just broken. Somebody read them at 2am looking for a cause.

None of these is fixed by swallowing the error. Each is fixed where the ordering is
decided:

* the frame timer stops at ``MEETING_STATUS_ENDED`` instead of finding out from the
  video sender - the meeting status callback is the only thing that knows;
* the 4-second subscription refresh is stopped in ``cleanup()`` rather than left running
  against a participant list that is emptying, and what remains of that race - somebody
  leaving between the list and the lookup - is a ``None`` to skip, not an error;
* raw-data subscriptions are released *before* the services they run through are
  destroyed, which is the only order in which unsubscribing can succeed;
* the Redis listener is asked to stop before the socket it is blocked on is closed.
"""

import threading
from unittest.mock import MagicMock, call, patch

from django.test import SimpleTestCase

from bots.automatic_leave_configuration import AutomaticLeaveConfiguration
from bots.bot_controller.bot_controller import BotController
from bots.zoom_bot_adapter.realtime_per_participant_video_frame_generator import RealtimePerParticipantVideoFrameGenerator
from bots.zoom_bot_adapter.zoom_bot_adapter import ZoomBotAdapter

SDKERR_SUCCESS = 0
SDKERR_WRONG_USAGE = 7


def an_adapter():
    """A ZoomBotAdapter that has been constructed and nothing more.

    ``use_video`` and the per-participant video callback are off so that no SDK object
    is built during construction: every test here installs the handful of collaborators
    it actually exercises.
    """
    return ZoomBotAdapter(
        use_one_way_audio=True,
        use_mixed_audio=False,
        use_video=False,
        display_name="Test Bot",
        send_message_callback=MagicMock(),
        add_audio_chunk_callback=MagicMock(),
        zoom_client_id="client-id",
        zoom_client_secret="client-secret",
        meeting_url="https://us02web.zoom.us/j/12345678901?pwd=secret",
        add_video_frame_callback=MagicMock(),
        wants_any_video_frames_callback=MagicMock(),
        add_mixed_audio_chunk_callback=MagicMock(),
        add_per_participant_video_frame_callback=None,
        upsert_chat_message_callback=MagicMock(),
        add_participant_event_callback=MagicMock(),
        automatic_leave_configuration=AutomaticLeaveConfiguration(),
        per_participant_realtime_video_configuration=None,
        video_frame_size=(640, 360),
        zoom_tokens={},
        zoom_meeting_settings={},
        record_chat_messages_when_paused=False,
        record_participant_speech_start_stop_events=False,
    )


def a_mock_zoom_sdk():
    """The SDK module as the adapter uses it: constants compared by identity."""
    mock = MagicMock()
    mock.SDKERR_SUCCESS = SDKERR_SUCCESS
    mock.SDKERR_WRONG_USAGE = SDKERR_WRONG_USAGE
    return mock


class TheImageTimerStopsWhenTheMeetingDoes(SimpleTestCase):
    """The SDKERR_WRONG_USAGE frame."""

    def setUp(self):
        self.zoom = a_mock_zoom_sdk()
        self.patcher = patch("bots.zoom_bot_adapter.zoom_bot_adapter.zoom", self.zoom)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

        self.adapter = an_adapter()
        # Joining is a long conversation with the SDK and none of it is what these tests
        # are about; the status callback ends with it.
        self.adapter.on_join = MagicMock()
        self.adapter.video_sender = MagicMock()
        self.adapter.video_sender.sendVideoFrame.return_value = SDKERR_SUCCESS
        self.adapter.suggested_video_cap = MagicMock(width=640, height=360)
        self.adapter.on_virtual_camera_start_send_callback_called = True
        self.adapter.current_raw_image_to_send = b"an image"
        self.adapter.current_image_to_send = b"a computed frame"
        self.adapter.send_image_timeout_id = 12345

    def test_a_frame_is_still_sent_while_the_meeting_is_running(self):
        self.assertTrue(self.adapter.send_current_image_to_zoom())
        self.adapter.video_sender.sendVideoFrame.assert_called_once()

    def test_no_frame_is_sent_once_the_meeting_has_ended(self):
        """Fails against the old behaviour, which sent one last frame into a video
        source Zoom had already destroyed and logged its refusal."""
        self.adapter.meeting_status_changed(self.zoom.MEETING_STATUS_ENDED, 0)

        self.assertFalse(self.adapter.send_current_image_to_zoom())
        self.adapter.video_sender.sendVideoFrame.assert_not_called()

    def test_the_timer_is_forgotten_so_it_stops_ticking(self):
        self.adapter.meeting_status_changed(self.zoom.MEETING_STATUS_ENDED, 0)
        self.adapter.send_current_image_to_zoom()
        self.assertIsNone(self.adapter.send_image_timeout_id)

    def test_a_failed_join_also_closes_the_video_source(self):
        self.adapter.meeting_status_changed(self.zoom.MEETING_STATUS_FAILED, 0)
        self.assertFalse(self.adapter.send_current_image_to_zoom())

    def test_rejoining_opens_it_again(self):
        """A Zoom bot with an on-behalf token retries after the meeting ends, in the
        same process. The second meeting gets a real video source, so the flag has to
        be cleared rather than latched."""
        self.adapter.meeting_status_changed(self.zoom.MEETING_STATUS_ENDED, 0)
        self.adapter.meeting_status_changed(self.zoom.MEETING_STATUS_INMEETING, 0)

        self.assertTrue(self.adapter.send_current_image_to_zoom())
        self.adapter.video_sender.sendVideoFrame.assert_called_once()

    def test_media_frames_stop_too(self):
        self.adapter.meeting_status_changed(self.zoom.MEETING_STATUS_ENDED, 0)
        self.assertFalse(self.adapter.send_video_frame_to_zoom(b"frame", 640, 360))
        self.adapter.video_sender.sendVideoFrame.assert_not_called()


class SubscriptionRefreshSurvivesAParticipantLeaving(SimpleTestCase):
    """The AttributeError on 'NoneType'."""

    def a_generator(self, participants_ctrl):
        return RealtimePerParticipantVideoFrameGenerator(
            frame_callback=MagicMock(),
            get_participants_ctrl_callback=lambda: participants_ctrl,
            get_meeting_sharing_controller_callback=lambda: self.sharing_ctrl,
            get_recording_is_paused_callback=lambda: False,
            per_participant_realtime_video_configuration=MagicMock(
                webcam_configuration=MagicMock(enabled=True),
                screenshare_configuration=MagicMock(enabled=True),
            ),
        )

    def setUp(self):
        self.sharing_ctrl = MagicMock()
        self.sharing_ctrl.GetViewableSharingUserList.return_value = []

    def test_a_participant_who_has_already_gone_is_skipped(self):
        """Fails against the old behaviour, which called IsVideoOn() on the None that
        GetUserByUserID returns for somebody who has left."""
        participants_ctrl = MagicMock()
        participants_ctrl.GetParticipantsList.return_value = [1, 2]
        participants_ctrl.GetUserByUserID.return_value = None

        generator = self.a_generator(participants_ctrl)
        generator._do_refresh_subscriptions()

        self.assertEqual(generator._subscriptions, {})

    def test_the_participants_who_remain_are_still_looked_at(self):
        """One gone participant must not cost the refresh the others."""
        gone, present = 1, 2
        still_here = MagicMock()
        still_here.IsVideoOn.return_value = False

        participants_ctrl = MagicMock()
        participants_ctrl.GetParticipantsList.return_value = [gone, present]
        participants_ctrl.GetUserByUserID.side_effect = lambda participant_id: None if participant_id == gone else still_here

        generator = self.a_generator(participants_ctrl)
        generator._do_refresh_subscriptions()

        still_here.IsVideoOn.assert_called_once()


class CleanupReleasesBeforeItDestroys(SimpleTestCase):
    """The SDKERR_INTERNAL_ERROR from unSubscribe()."""

    def setUp(self):
        self.zoom = a_mock_zoom_sdk()
        self.patcher = patch("bots.zoom_bot_adapter.zoom_bot_adapter.zoom", self.zoom)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

        self.adapter = an_adapter()

        # One recorder, so the assertions are about order and not about counts.
        self.order = MagicMock()
        self.adapter.audio_helper = MagicMock()
        self.adapter.audio_helper.unSubscribe.side_effect = lambda: self.order.unsubscribed_audio()
        self.adapter.video_input_manager = MagicMock()
        self.adapter.video_input_manager.cleanup.side_effect = lambda: self.order.cleaned_up_video()
        self.adapter.realtime_per_participant_video_frame_generator = MagicMock()
        self.adapter.realtime_per_participant_video_frame_generator.reset.side_effect = lambda: self.order.stopped_the_refresh()
        self.adapter.meeting_service = MagicMock()
        self.zoom.DestroyMeetingService.side_effect = lambda service: self.order.destroyed_the_meeting_service()

    def test_audio_is_unsubscribed_before_the_meeting_service_is_destroyed(self):
        """Fails against the old behaviour, where unSubscribe() ran through a meeting
        service that had already been destroyed and answered SDKERR_INTERNAL_ERROR."""
        self.adapter.cleanup()

        self.assertEqual(
            self.order.mock_calls,
            [
                call.unsubscribed_audio(),
                call.cleaned_up_video(),
                call.stopped_the_refresh(),
                call.destroyed_the_meeting_service(),
            ],
        )

    def test_the_periodic_subscription_refresh_is_stopped(self):
        """It was started on join and never stopped, so it went on asking a dissolving
        meeting about its participants for as long as teardown took."""
        self.adapter.cleanup()
        self.adapter.realtime_per_participant_video_frame_generator.reset.assert_called_once()


class TheRedisListenerIsToldToStop(SimpleTestCase):
    """The ValueError on a closed file.

    ``redis_listener`` reads three attributes and nothing else, so the loop is exercised
    on a bare controller rather than on a bot that would need a database, a meeting and
    a GLib main loop to exist.
    """

    def a_controller(self, pubsub):
        controller = BotController.__new__(BotController)
        controller.pubsub = pubsub
        controller.redis_listener_should_stop = threading.Event()
        return controller

    def test_a_read_that_fails_after_the_stop_request_ends_the_thread_quietly(self):
        """Fails against the old behaviour, which had no stop request to consult and so
        logged 'Error in Redis listener' whenever shutdown closed the pubsub underneath
        a blocked get_message()."""
        pubsub = MagicMock()
        controller = self.a_controller(pubsub)

        def closed_under_us(timeout=None):
            controller.redis_listener_should_stop.set()
            raise ValueError("I/O operation on closed file.")

        pubsub.get_message.side_effect = closed_under_us

        with self.assertLogs("bots.bot_controller.bot_controller", level="INFO") as logs:
            controller.redis_listener()

        self.assertFalse(any("Error in Redis listener" in line for line in logs.output))

    def test_a_genuine_error_is_still_reported(self):
        """The quiet path is only for shutdown: nobody asked this one to stop."""
        pubsub = MagicMock()
        controller = self.a_controller(pubsub)
        pubsub.get_message.side_effect = ValueError("something really is wrong")

        with self.assertLogs("bots.bot_controller.bot_controller", level="WARNING") as logs:
            controller.redis_listener()

        self.assertTrue(any("Error in Redis listener" in line for line in logs.output))

    def test_the_loop_does_not_start_once_stop_has_been_asked_for(self):
        pubsub = MagicMock()
        controller = self.a_controller(pubsub)
        controller.redis_listener_should_stop.set()

        controller.redis_listener()

        pubsub.get_message.assert_not_called()
