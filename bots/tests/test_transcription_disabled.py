"""Turning transcription off, for bots that transcribe somewhere else.

A live Zoom meeting on 2026-08-09 logged this seven times in five minutes, once per
thing anybody said:

    Transcription failed for utterance 1155, failure data: {'reason':
    TranscriptionFailureReasons.CREDENTIALS_NOT_FOUND}

Nothing was broken. The caller transcribes the per-participant audio websocket itself
and has never wanted Attendee's transcript - but there was no way to say so. Every value
``transcription_provider_from_bot_creation_data`` could return named a provider, and a
Zoom bot on the native SDK that says nothing gets Deepgram. So the bot dutifully cut an
utterance per turn, stored the audio, queued a Celery task, asked the project for
Deepgram credentials it does not have, and wrote a failure - all of it work whose only
product was a line that reads like an outage.

``transcription_settings: {"none": {}}`` is the missing answer. It is a provider in the
same sense that /dev/null is a file: it goes through the one function that decides what
transcribes a recording, so that everything downstream keeps asking the question it
already asks instead of growing a second flag to check.

Turning it off does not turn off *recording* the per-speaker audio - a bot created with
``record_async_transcription_audio_chunks`` still keeps its chunks, so somebody can
transcribe the meeting later by naming a real provider. What goes away is the utterance,
the task, and the failure.
"""

from django.test import SimpleTestCase, TestCase

from accounts.models import Organization
from bots.bot_controller.bot_controller import BotController
from bots.bots_api_utils import BotCreationSource, create_bot
from bots.models import Project, TranscriptionProviders, TranscriptionTypes, ZoomOAuthApp
from bots.serializers import CreateAsyncTranscriptionSerializer, CreateBotSerializer
from bots.utils import transcription_provider_from_bot_creation_data

A_ZOOM_URL = "https://us02web.zoom.us/j/12345678901?pwd=secret"
A_MEET_URL = "https://meet.google.com/abc-defg-hij"


class NoneIsAProviderLikeAnyOther(SimpleTestCase):
    def test_saying_none_turns_transcription_off_on_zoom(self):
        """Fails against the old behaviour, where a Zoom bot on the native SDK got
        Deepgram whatever it asked for, because there was nothing else to give it."""
        provider = transcription_provider_from_bot_creation_data({"meeting_url": A_ZOOM_URL, "transcription_settings": {"none": {}}})
        self.assertEqual(provider, TranscriptionProviders.NO_TRANSCRIPTION)

    def test_saying_none_turns_transcription_off_on_meet_too(self):
        """The platform default here is closed captions rather than Deepgram, and it is
        just as unwanted."""
        provider = transcription_provider_from_bot_creation_data({"meeting_url": A_MEET_URL, "transcription_settings": {"none": {}}})
        self.assertEqual(provider, TranscriptionProviders.NO_TRANSCRIPTION)

    def test_it_wins_over_a_provider_named_alongside_it(self):
        """Saying "none" is saying it about everything else in the object."""
        provider = transcription_provider_from_bot_creation_data({"meeting_url": A_ZOOM_URL, "transcription_settings": {"none": {}, "deepgram": {"language": "multi"}}})
        self.assertEqual(provider, TranscriptionProviders.NO_TRANSCRIPTION)

    def test_the_defaults_are_untouched_for_everybody_who_does_not_ask(self):
        self.assertEqual(
            transcription_provider_from_bot_creation_data({"meeting_url": A_ZOOM_URL, "transcription_settings": {}}),
            TranscriptionProviders.DEEPGRAM,
        )
        self.assertEqual(
            transcription_provider_from_bot_creation_data({"meeting_url": A_MEET_URL, "transcription_settings": {}}),
            TranscriptionProviders.CLOSED_CAPTION_FROM_PLATFORM,
        )


class TheApiAcceptsIt(TestCase):
    def test_an_empty_none_object_validates(self):
        serializer = CreateBotSerializer(data={"meeting_url": A_ZOOM_URL, "bot_name": "Test Bot", "transcription_settings": {"none": {}}})
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_none_takes_no_options(self):
        """There is nothing to configure about not transcribing, and a key accepted here
        would be a setting somebody believed was doing something."""
        serializer = CreateBotSerializer(data={"meeting_url": A_ZOOM_URL, "bot_name": "Test Bot", "transcription_settings": {"none": {"model": "nova-3"}}})
        self.assertFalse(serializer.is_valid())
        self.assertIn("transcription_settings", serializer.errors)

    def test_an_async_transcription_cannot_ask_for_no_transcription(self):
        """An async transcription is something you went and asked for, so "none" is a
        contradiction - and left alone it would fail per utterance in the worker, which
        is the whole thing this change exists to stop."""
        serializer = CreateAsyncTranscriptionSerializer(data={"transcription_settings": {"none": {}}})
        self.assertFalse(serializer.is_valid())
        self.assertIn("transcription_settings", serializer.errors)

    def test_an_async_transcription_naming_a_provider_is_still_fine(self):
        serializer = CreateAsyncTranscriptionSerializer(data={"transcription_settings": {"deepgram": {"language": "multi"}}})
        self.assertTrue(serializer.is_valid(), serializer.errors)


class ARecordingRemembersThatItIsNotTranscribed(TestCase):
    def setUp(self):
        self.organization = Organization.objects.create(name="Test Organization")
        self.project = Project.objects.create(name="Test Project", organization=self.organization)
        # A Zoom bot is refused without them, and Zoom on the native SDK is where the
        # unwanted default lives.
        ZoomOAuthApp.objects.create(project=self.project, client_id="123")

    def a_bot(self, transcription_settings):
        data = {"meeting_url": A_ZOOM_URL, "bot_name": "Test Bot"}
        if transcription_settings is not None:
            data["transcription_settings"] = transcription_settings
        bot, error = create_bot(data=data, source=BotCreationSource.API, project=self.project)
        self.assertIsNone(error)
        return bot

    def test_the_recording_names_no_provider_and_says_so_twice(self):
        recording = self.a_bot({"none": {}}).recordings.first()
        self.assertEqual(recording.transcription_provider, TranscriptionProviders.NO_TRANSCRIPTION)
        self.assertEqual(recording.transcription_type, TranscriptionTypes.NO_TRANSCRIPTION)

    def test_an_ordinary_bot_is_unchanged(self):
        recording = self.a_bot(None).recordings.first()
        self.assertEqual(recording.transcription_provider, TranscriptionProviders.DEEPGRAM)
        self.assertEqual(recording.transcription_type, TranscriptionTypes.NON_REALTIME)

    def test_no_utterance_is_cut_and_no_audio_chunk_is_kept(self):
        """The per-utterance work itself. Fails against the old behaviour, where the
        only provider a Zoom bot could have was one that needed credentials."""
        controller = BotController(self.a_bot({"none": {}}).id)

        self.assertFalse(controller.save_utterances_for_individual_audio_chunks())
        self.assertFalse(controller.save_utterances_for_closed_captions())
        self.assertFalse(controller.should_capture_audio_chunks())
        self.assertFalse(controller.use_streaming_transcription())

    def test_a_deepgram_bot_still_cuts_utterances(self):
        controller = BotController(self.a_bot(None).id)

        self.assertTrue(controller.save_utterances_for_individual_audio_chunks())
        self.assertTrue(controller.should_capture_audio_chunks())

    def test_audio_is_still_kept_when_a_later_transcription_was_asked_for(self):
        """Transcription off is not recording off: the chunks are what an async
        transcription with a real provider would run over."""
        self.organization.is_async_transcription_enabled = True
        self.organization.save()

        bot = self.a_bot({"none": {}})
        bot.settings["recording_settings"] = {**(bot.settings.get("recording_settings") or {}), "record_async_transcription_audio_chunks": True}
        bot.save()

        controller = BotController(bot.id)

        self.assertFalse(controller.save_utterances_for_individual_audio_chunks())
        self.assertTrue(controller.should_capture_audio_chunks())
