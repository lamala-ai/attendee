import unittest
from unittest.mock import patch

import numpy as np
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from bots import presence_indicator
from bots.models import ApiKey, Bot, BotStates, Organization, Project, User


def blank_i420(width, height, luma=90):
    """A flat grey I420 frame, so anything the indicator changes stands out."""
    frame = bytearray(width * height + 2 * (width // 2) * (height // 2))
    for index in range(len(frame)):
        frame[index] = luma if index < width * height else 128
    return frame


def planes(frame, width, height):
    buffer = np.frombuffer(bytes(frame), dtype=np.uint8)
    luma_size = width * height
    chroma_size = (width // 2) * (height // 2)
    return (
        buffer[:luma_size].reshape(height, width),
        buffer[luma_size : luma_size + chroma_size].reshape(height // 2, width // 2),
        buffer[luma_size + chroma_size :].reshape(height // 2, width // 2),
    )


class TestPresenceIndicatorDrawing(unittest.TestCase):
    """The bead itself. No database, no adapter - just what lands in the pixels."""

    WIDTH, HEIGHT = 320, 180

    def test_only_animated_states_draw_anything(self):
        for state in (presence_indicator.SPEAKING, presence_indicator.OFF, None, "nonsense"):
            frame = blank_i420(self.WIDTH, self.HEIGHT)
            untouched = bytes(frame)
            presence_indicator.paint_i420(frame, self.WIDTH, self.HEIGHT, state, 0.0)
            self.assertEqual(bytes(frame), untouched, f"{state} should draw nothing")

    def test_the_bead_lands_where_the_geometry_says_and_nowhere_else(self):
        frame = blank_i420(self.WIDTH, self.HEIGHT)
        presence_indicator.paint_i420(frame, self.WIDTH, self.HEIGHT, presence_indicator.LISTENING, 1.4)
        luma, blue, red = planes(frame, self.WIDTH, self.HEIGHT)

        center_x, center_y, _ = presence_indicator.bead_geometry(self.WIDTH, self.HEIGHT)
        self.assertNotEqual(luma[int(center_y), int(center_x)], 90, "the bead did not land on its own centre")
        # Chroma follows it, or the bead would be a grey dot.
        self.assertNotEqual(blue[int(center_y / 2), int(center_x / 2)], 128)
        self.assertNotEqual(red[int(center_y / 2), int(center_x / 2)], 128)
        # And the rest of the picture is untouched: this is a mark, not a filter.
        self.assertEqual(luma[0, self.WIDTH - 1], 90)
        self.assertEqual(luma[0, 0], 90)

    def test_the_bead_follows_the_picture_rather_than_the_frame(self):
        """A square avatar scaled into a 16:9 capability is letterboxed. A bead in the
        corner of the frame would sit on the black bar instead of on the avatar."""
        frame = blank_i420(self.WIDTH, self.HEIGHT)
        rect = presence_indicator.letterboxed_content_rect((512, 512), (self.WIDTH, self.HEIGHT))
        self.assertEqual(rect, (70, 0, 180, 180))

        presence_indicator.paint_i420(frame, self.WIDTH, self.HEIGHT, presence_indicator.LISTENING, 1.4, rect)
        luma, _, _ = planes(frame, self.WIDTH, self.HEIGHT)
        center_x, center_y, _ = presence_indicator.bead_geometry(rect[2], rect[3])
        self.assertNotEqual(luma[int(center_y + rect[1]), int(center_x + rect[0])], 90)
        # The left bar - where an unaware bead would have gone - is still black.
        self.assertEqual(luma[self.HEIGHT - 10, 5], 90)

    def test_the_pulse_actually_pulses_and_the_two_states_pulse_differently(self):
        listening = presence_indicator.ANIMATED[presence_indicator.LISTENING]["cycle_seconds"]
        self.assertAlmostEqual(presence_indicator.pulse(presence_indicator.LISTENING, 0.0), 0.0)
        self.assertAlmostEqual(presence_indicator.pulse(presence_indicator.LISTENING, listening / 2), 1.0)
        self.assertAlmostEqual(presence_indicator.pulse(presence_indicator.LISTENING, listening), 0.0)
        # Working is the faster one - that is the whole difference the room reads.
        self.assertLess(
            presence_indicator.ANIMATED[presence_indicator.WORKING]["cycle_seconds"],
            listening,
        )

    def test_a_dim_beat_and_a_lit_beat_are_not_the_same_picture(self):
        cycle = presence_indicator.ANIMATED[presence_indicator.LISTENING]["cycle_seconds"]
        dim = blank_i420(self.WIDTH, self.HEIGHT)
        lit = blank_i420(self.WIDTH, self.HEIGHT)
        presence_indicator.paint_i420(dim, self.WIDTH, self.HEIGHT, presence_indicator.LISTENING, 0.0)
        presence_indicator.paint_i420(lit, self.WIDTH, self.HEIGHT, presence_indicator.LISTENING, cycle / 2)
        self.assertNotEqual(bytes(dim), bytes(lit))

    def test_an_odd_or_empty_frame_is_left_alone_rather_than_corrupted(self):
        for width, height in ((321, 180), (0, 0), (320, 181)):
            frame = blank_i420(320, 180)
            untouched = bytes(frame)
            presence_indicator.paint_i420(frame, width, height, presence_indicator.LISTENING, 1.0)
            self.assertEqual(bytes(frame), untouched)

    def test_normalize_accepts_only_states_we_draw(self):
        self.assertEqual(presence_indicator.normalize(" Listening "), presence_indicator.LISTENING)
        self.assertIsNone(presence_indicator.normalize("thinking"))
        self.assertIsNone(presence_indicator.normalize(None))


class TestPresenceIndicatorApi(TestCase):
    """One call per state change, and nothing sent per frame."""

    def setUp(self):
        self.user = User.objects.create_user(username="presence@example.com", email="presence@example.com")
        self.organization = Organization.objects.create(name="Presence Org")
        self.user.organization = self.organization
        self.user.save()
        self.project = Project.objects.create(name="Presence Project", organization=self.organization)
        self.api_key, self.api_key_plain = ApiKey.create(project=self.project, name="Presence Key")
        self.bot = Bot.objects.create(
            project=self.project,
            meeting_url="https://zoom.us/j/123456",
            state=BotStates.JOINED_RECORDING,
        )
        self.client = APIClient()
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {self.api_key_plain}")
        self.url = f"/api/v1/bots/{self.bot.object_id}/presence_indicator"

    @patch("bots.bots_api_views.send_sync_command")
    def test_setting_a_state_stores_it_and_tells_the_bot(self, mock_send_sync_command):
        response = self.client.patch(self.url, {"state": "working"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.bot.refresh_from_db()
        self.assertEqual(self.bot.presence_indicator_state(), "working")
        mock_send_sync_command.assert_called_once_with(self.bot, "sync_presence_indicator")

    @patch("bots.bots_api_views.send_sync_command")
    def test_a_state_we_do_not_draw_is_refused(self, mock_send_sync_command):
        response = self.client.patch(self.url, {"state": "thinking"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.bot.refresh_from_db()
        self.assertIsNone(self.bot.presence_indicator_state())
        mock_send_sync_command.assert_not_called()

    @patch("bots.bots_api_views.send_sync_command")
    def test_a_bot_that_is_not_in_a_meeting_has_nothing_to_draw_on(self, mock_send_sync_command):
        self.bot.state = BotStates.READY
        self.bot.save()
        response = self.client.patch(self.url, {"state": "listening"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        mock_send_sync_command.assert_not_called()

    def test_an_unknown_bot_is_a_404(self):
        response = self.client.patch("/api/v1/bots/bot_doesnotexist/presence_indicator", {"state": "listening"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)


class TestTheDefaultAdapterDrawsNothingAndSaysSo(unittest.TestCase):
    """Every adapter inherits a no-op, so a bot with no video output of its own is
    never broken by a state it cannot draw - a cosmetic mark is not worth a call."""

    def test_the_base_adapter_accepts_any_state_without_complaining(self):
        from bots.bot_adapter import BotAdapter

        BotAdapter().set_presence_indicator(presence_indicator.LISTENING)
        BotAdapter().set_presence_indicator(None)
