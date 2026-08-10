import re
import unittest
from pathlib import Path
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
    """The glow and the label themselves. No database, no adapter - just the pixels."""

    WIDTH, HEIGHT = 320, 180

    def test_only_a_state_we_have_a_word_for_draws_anything(self):
        for state in (presence_indicator.OFF, None, "nonsense"):
            frame = blank_i420(self.WIDTH, self.HEIGHT)
            untouched = bytes(frame)
            presence_indicator.paint_i420(frame, self.WIDTH, self.HEIGHT, state, 0.0)
            self.assertEqual(bytes(frame), untouched, f"{state} should draw nothing")

    def test_listening_draws_its_word_without_drawing_a_glow(self):
        """Fails against the bead, where listening was the *pulsing* state and speaking
        drew nothing at all: both of those are now the other way round."""
        self.assertTrue(presence_indicator.draws(presence_indicator.LISTENING))
        self.assertFalse(presence_indicator.is_animated(presence_indicator.LISTENING))

        frame = blank_i420(self.WIDTH, self.HEIGHT)
        presence_indicator.paint_i420(frame, self.WIDTH, self.HEIGHT, presence_indicator.LISTENING, 1.4)
        luma, _, _ = planes(frame, self.WIDTH, self.HEIGHT)

        left, top, box_width, box_height = presence_indicator.label_box(self.WIDTH, self.HEIGHT)
        plate = luma[top : top + box_height, left : left + box_width]
        self.assertTrue((plate != 90).any(), "the label did not land in its own box")
        # Where the glow would be if this state had one: the ring rides at 0.44 of the
        # shorter side out from the middle, level with the centre of the picture.
        ring_x = int(self.WIDTH / 2 + min(self.WIDTH, self.HEIGHT) * presence_indicator.RING_RADIUS)
        self.assertEqual(luma[self.HEIGHT // 2, ring_x], 90, "listening should not glow")

    def test_a_glowing_state_rings_the_picture_in_its_own_colour(self):
        frame = blank_i420(self.WIDTH, self.HEIGHT)
        settings = presence_indicator.GLOW[presence_indicator.SPEAKING]
        lit = settings["cycle_seconds"] / 2
        presence_indicator.paint_i420(frame, self.WIDTH, self.HEIGHT, presence_indicator.SPEAKING, lit)
        luma, blue, red = planes(frame, self.WIDTH, self.HEIGHT)

        side = min(self.WIDTH, self.HEIGHT)
        ring_x = int(self.WIDTH / 2 + side * presence_indicator.RING_RADIUS)
        row = self.HEIGHT // 2
        self.assertNotEqual(luma[row, ring_x], 90, "the ring did not land on its own band")
        # Chroma follows it, or a green glow would be a grey one.
        self.assertNotEqual(blue[row // 2, ring_x // 2], 128)
        self.assertNotEqual(red[row // 2, ring_x // 2], 128)
        # And the middle of the face is left alone: this is a ring, not a wash.
        self.assertEqual(luma[row, self.WIDTH // 2], 90)

    def test_the_glow_follows_the_picture_rather_than_the_frame(self):
        """A square avatar scaled into a 16:9 capability is letterboxed. A ring measured
        off the frame would circle the black bars instead of the avatar."""
        frame = blank_i420(self.WIDTH, self.HEIGHT)
        rect = presence_indicator.letterboxed_content_rect((512, 512), (self.WIDTH, self.HEIGHT))
        self.assertEqual(rect, (70, 0, 180, 180))

        presence_indicator.paint_i420(frame, self.WIDTH, self.HEIGHT, presence_indicator.WORKING, 0.45, rect)
        luma, _, _ = planes(frame, self.WIDTH, self.HEIGHT)
        ring_x = int(rect[0] + rect[2] / 2 + rect[2] * presence_indicator.RING_RADIUS)
        self.assertNotEqual(luma[rect[3] // 2, ring_x], 90)
        # The left bar - where a frame-measured ring would have gone - is untouched.
        self.assertEqual(luma[self.HEIGHT // 2, 2], 90)

    def test_everything_readable_survives_the_crop_a_client_makes(self):
        """The rule the geometry exists for: a client crops a tile to fill its own
        shape, and what survives every crop of a square picture is its inscribed circle.
        The lit band and the label have to be inside it. The halo either side of the
        band may cross it - it is fading to nothing out there - so this measures what is
        actually readable rather than every pixel that changed at all."""
        size = 240
        frame = blank_i420(size, size)
        presence_indicator.paint_i420(frame, size, size, presence_indicator.SPEAKING, 0.8)
        luma, _, _ = planes(frame, size, size)

        drawn = np.abs(luma.astype(np.int16) - 90)
        self.assertTrue(drawn.any(), "nothing was drawn at all")
        ys, xs = np.nonzero(drawn > drawn.max() * 0.4)
        distance = np.sqrt((xs + 0.5 - size / 2) ** 2 + (ys + 0.5 - size / 2) ** 2)
        self.assertLessEqual(distance.max(), size / 2)

    def test_nothing_is_drawn_outside_the_picture_at_all(self):
        """Whatever the halo does inside the avatar, it may not spill onto the black
        bars the letterboxing put beside it - a glow around the frame rather than around
        the face is the one version of this that looks like a bug."""
        frame = blank_i420(self.WIDTH, self.HEIGHT)
        rect = presence_indicator.letterboxed_content_rect((512, 512), (self.WIDTH, self.HEIGHT))
        presence_indicator.paint_i420(frame, self.WIDTH, self.HEIGHT, presence_indicator.SPEAKING, 0.8, rect)
        luma, _, _ = planes(frame, self.WIDTH, self.HEIGHT)

        x, y, width, height = rect
        outside = luma.copy()
        outside[y : y + height, x : x + width] = 90
        self.assertFalse((outside != 90).any(), "the glow reached the letterbox bars")

    def test_the_pulse_actually_pulses_and_the_two_glows_pulse_differently(self):
        speaking = presence_indicator.GLOW[presence_indicator.SPEAKING]["cycle_seconds"]
        self.assertAlmostEqual(presence_indicator.pulse(presence_indicator.SPEAKING, 0.0), 0.0)
        self.assertAlmostEqual(presence_indicator.pulse(presence_indicator.SPEAKING, speaking / 2), 1.0)
        self.assertAlmostEqual(presence_indicator.pulse(presence_indicator.SPEAKING, speaking), 0.0)
        # Working is the faster one - it is the state somebody is waiting through.
        self.assertLess(
            presence_indicator.GLOW[presence_indicator.WORKING]["cycle_seconds"],
            speaking,
        )

    def test_a_dim_beat_and_a_lit_beat_are_not_the_same_picture(self):
        cycle = presence_indicator.GLOW[presence_indicator.WORKING]["cycle_seconds"]
        dim = blank_i420(self.WIDTH, self.HEIGHT)
        lit = blank_i420(self.WIDTH, self.HEIGHT)
        presence_indicator.paint_i420(dim, self.WIDTH, self.HEIGHT, presence_indicator.WORKING, 0.0)
        presence_indicator.paint_i420(lit, self.WIDTH, self.HEIGHT, presence_indicator.WORKING, cycle / 2)
        self.assertNotEqual(bytes(dim), bytes(lit))

    def test_each_state_says_a_different_word(self):
        words = [presence_indicator.LABELS[state] for state in (presence_indicator.LISTENING, presence_indicator.WORKING, presence_indicator.SPEAKING)]
        self.assertEqual(len(set(words)), 3)
        listening = blank_i420(self.WIDTH, self.HEIGHT)
        working = blank_i420(self.WIDTH, self.HEIGHT)
        presence_indicator.paint_i420(listening, self.WIDTH, self.HEIGHT, presence_indicator.LISTENING, 0.0)
        # Compared at the same point of its cycle where the glow is at its dimmest, so
        # what differs between these two frames is the lettering rather than the light.
        presence_indicator.paint_i420(working, self.WIDTH, self.HEIGHT, presence_indicator.WORKING, 0.0)
        left, top, box_width, box_height = presence_indicator.label_box(self.WIDTH, self.HEIGHT)
        one, _, _ = planes(listening, self.WIDTH, self.HEIGHT)
        two, _, _ = planes(working, self.WIDTH, self.HEIGHT)
        self.assertFalse(
            np.array_equal(
                one[top : top + box_height, left : left + box_width],
                two[top : top + box_height, left : left + box_width],
            )
        )

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


class TestTheTaskPills(unittest.TestCase):
    """What the seat is working on, under the word. Pixels and geometry only."""

    # A square picture, which is what an avatar actually is once letterboxing is
    # accounted for - the stack is sized as a fraction of the shorter side.
    WIDTH, HEIGHT = 640, 640

    def test_a_task_hangs_under_the_state_word(self):
        """The word is the heading and the pills are what it is about, so a reader who
        has just read WORKING carries on downward."""
        rows = presence_indicator.task_rows([{"text": "Reading the logs"}])
        boxes = presence_indicator.task_boxes(self.WIDTH, self.HEIGHT, len(rows))
        left, label_top, _, label_height = presence_indicator.label_box(self.WIDTH, self.HEIGHT, len(rows))
        self.assertEqual(len(boxes), 1)
        _, top, _, _ = boxes[0]
        self.assertGreater(top, label_top + label_height, "a task pill hangs under the word")

    def test_rows_run_downward_in_the_order_they_were_given(self):
        boxes = presence_indicator.task_boxes(self.WIDTH, self.HEIGHT, 3)
        self.assertEqual([box[1] for box in boxes], sorted(box[1] for box in boxes))

    def test_the_word_lifts_so_the_group_keeps_the_caption_clearance(self):
        """Fails against hanging the pills off a word that stays put, which walks the
        stack down into the meeting client's own name caption one errand at a time.
        The inset has to be measured from whatever is actually lowest."""
        _, bare_top, _, bare_height = presence_indicator.label_box(self.WIDTH, self.HEIGHT, 0)
        floor = bare_top + bare_height
        for rows in (1, 2, 3):
            boxes = presence_indicator.task_boxes(self.WIDTH, self.HEIGHT, rows)
            bottom = boxes[-1][1] + boxes[-1][3]
            self.assertLessEqual(abs(bottom - floor), 2, f"{rows} rows should end where the bare word does")
            _, top, _, _ = presence_indicator.label_box(self.WIDTH, self.HEIGHT, rows)
            self.assertLess(top, bare_top, "and the word itself lifts to make the room")

    def test_only_working_names_an_errand(self):
        """Listening is not holding one, and speaking is delivering one the room can
        already hear - writing that under the word tells them what they are being told."""
        self.assertTrue(presence_indicator.shows_tasks(presence_indicator.WORKING))
        self.assertFalse(presence_indicator.shows_tasks(presence_indicator.LISTENING))
        self.assertFalse(presence_indicator.shows_tasks(presence_indicator.SPEAKING))

    def test_a_state_that_does_not_name_errands_draws_the_same_tile_with_or_without_them(self):
        for state in (presence_indicator.LISTENING, presence_indicator.SPEAKING):
            with_tasks = blank_i420(self.WIDTH, self.HEIGHT)
            presence_indicator.paint_i420(
                with_tasks, self.WIDTH, self.HEIGHT, state, 0.0, None, [{"text": "Reading the logs"}]
            )
            without = blank_i420(self.WIDTH, self.HEIGHT)
            presence_indicator.paint_i420(without, self.WIDTH, self.HEIGHT, state, 0.0)
            self.assertEqual(bytes(with_tasks), bytes(without), f"{state} should draw no pills")

    def test_a_note_becomes_its_own_line_under_its_task(self):
        rows = presence_indicator.task_rows([{"text": "Reading the logs", "note": "Two of five"}])
        self.assertEqual([row[0] for row in rows], ["Reading the logs", "Two of five"])
        self.assertNotEqual(rows[0][1], rows[1][1], "a note is drawn dimmer than its task")

    def test_a_bare_string_is_a_task_with_no_note(self):
        self.assertEqual(
            presence_indicator.sanitize_tasks(["Reading the logs"]),
            [{"text": "Reading the logs", "note": ""}],
        )

    def test_more_tasks_than_the_tile_holds_are_dropped_rather_than_shrunk(self):
        tasks = [f"Errand {index}" for index in range(10)]
        kept = presence_indicator.sanitize_tasks(tasks)
        self.assertEqual(len(kept), presence_indicator.TASK_LIMIT)
        self.assertEqual(kept[0]["text"], "Errand 0")

    def test_a_long_line_is_cut_with_an_ascii_ellipsis(self):
        """Not the '…' character: both typesetters draw this, and OpenCV's Hershey
        fonts have no glyph for it - a real ellipsis arrives on the tile as '?'."""
        clipped = presence_indicator.clip_line("x" * 200)
        self.assertLessEqual(len(clipped), presence_indicator.TASK_TEXT_LIMIT)
        self.assertTrue(clipped.endswith("..."))
        self.assertTrue(clipped.isascii())

    def test_an_empty_task_is_dropped_rather_than_drawn_as_a_blank_pill(self):
        self.assertEqual(presence_indicator.sanitize_tasks(["", "   ", {"text": ""}]), [])

    def test_tasks_put_ink_on_the_tile_that_the_state_alone_does_not(self):
        without = blank_i420(self.WIDTH, self.HEIGHT)
        presence_indicator.paint_i420(without, self.WIDTH, self.HEIGHT, presence_indicator.WORKING, 0.0)
        with_tasks = blank_i420(self.WIDTH, self.HEIGHT)
        presence_indicator.paint_i420(
            with_tasks,
            self.WIDTH,
            self.HEIGHT,
            presence_indicator.WORKING,
            0.0,
            None,
            [{"text": "Reading the logs", "note": "Two of five"}],
        )
        self.assertNotEqual(bytes(without), bytes(with_tasks))

    def test_a_state_that_draws_nothing_draws_nothing_even_when_handed_tasks(self):
        """'off' means the tile is the customer's picture and nothing of ours."""
        frame = blank_i420(self.WIDTH, self.HEIGHT)
        untouched = bytes(frame)
        presence_indicator.paint_i420(
            frame, self.WIDTH, self.HEIGHT, presence_indicator.OFF, 0.0, None, ["Reading the logs"]
        )
        self.assertEqual(bytes(frame), untouched)


class TestTheBrowserSideForwardsWhatItWasGiven(unittest.TestCase):
    """The canvas is drawn by JavaScript with no test runner of its own, so the one
    thing worth pinning statically is the seam a Python change cannot see."""

    PAYLOAD = Path(__file__).resolve().parent.parent / "web_bot_adapter" / "shared_chromedriver_payload.js"

    def test_every_presence_entry_point_takes_the_tasks_as_well_as_the_state(self):
        """Fails against the shipped bug. `botOutputManager.setPresenceIndicator` is a
        one-line delegator onto the video stream's method of the same name; it kept the
        old one-argument signature, so `tasks` was dropped on the floor between the
        adapter and the canvas. The glow and the word still drew - they ride `state` -
        and the pills silently never did, which is the worst shape a bug can have."""
        source = self.PAYLOAD.read_text()
        signatures = re.findall(r"setPresenceIndicator\(([^)]*)\)", source)
        self.assertGreaterEqual(len(signatures), 3, "expected the definitions and the delegating call")
        for signature in signatures:
            self.assertIn("state", signature)
            self.assertIn("tasks", signature, f"`setPresenceIndicator({signature})` drops the tasks")
