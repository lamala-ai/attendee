"""The bot's own presence indicator: a glow around its avatar, and a word under it.

A bot that is sitting quietly in a meeting looks exactly like a bot whose process died
twenty minutes ago, and sitting quietly is what a bot does for most of a call. This
draws on the video it is already sending, so the room can see the difference.

Three states, set through PATCH /bots/{id}/presence_indicator:

* ``listening`` - the word, and no glow. The ordinary state, and the one the room is
  looking at for most of the meeting; a light that is on for an hour stops being read.
* ``working`` - a white glow, beating fast. Something was asked of it and the answer has
  not come back yet, which is the silence people read as a crash a second before asking
  whether it heard them.
* ``speaking`` - a green glow, breathing slowly. The room can hear it, so the glow is
  not there to inform - it is there so the voice and the tile are obviously the same
  participant when four bots are in the call.
* ``off`` - nothing at all.

**Why a glow and a word rather than a bead.** A bead is a five-pixel dot in a tile that
a meeting client may draw at 120 pixels wide, and at that size a colour is all that
survives - which means the room has to be told beforehand what amber means. A glow
around the whole picture survives any scale, and the word says what the colour means to
somebody who was never told.

**Where things are allowed to sit.** Meeting clients crop a tile to fill their own
shape, so an indicator pushed to the edge of the picture is one that can be cropped
away - which is worse than no indicator at all. The lit band of the glow rides at 0.44
of the shorter side out from the middle, and the label's plate is lifted clear of the
bottom edge, so both survive the circle a square picture is cropped to. What can
extend past that circle is the halo either side of the band, and only its faintest
part: it is already fading to nothing there, so a crop takes nothing readable with it.

The bottom of a tile is also where the meeting client writes the participant's own
name, so the label is kept up out of that band rather than pushed to the last pixel a
crop would leave: an indicator sitting on top of the client's own caption is legible
and still wrong.

The important property is that **nothing is sent per frame**. The state is set once and
the bot animates its own tile from the frames it already emits: the Zoom adapter paints
into the I420 frame it re-sends on a timer, and the web adapters draw onto the canvas
they already capture. So a pulse costs one API call, not one call per frame. A state
that does not pulse (``listening``) is painted once and left alone.

Both paths use the geometry, the words and the colours here, so the tile reads the same
on every platform - the JavaScript in web_bot_adapter/shared_chromedriver_payload.js
repeats these constants rather than inventing its own. The one thing it cannot repeat is
the glyphs: this side has OpenCV's stroked font and the browser has its own, so the
label is the same word in the same box at the same size, drawn by two typesetters.
"""

import math
from functools import lru_cache

import cv2
import numpy as np

LISTENING = "listening"
WORKING = "working"
SPEAKING = "speaking"
OFF = "off"

STATES = [LISTENING, WORKING, SPEAKING, OFF]

# Colour and pulse period per glowing state. The colours are the product's own: the
# green it marks anything live with, and its off-white paper. Listening is not in here
# and that is the point - it has a word and no light.
GLOW = {
    SPEAKING: {"rgb": (0x75, 0xD8, 0x7A), "cycle_seconds": 1.6},
    WORKING: {"rgb": (0xF7, 0xF8, 0xF8), "cycle_seconds": 0.9},
}

# What each state is called on the tile. Single words, upper case: this is read at the
# size a thumbnail allows, where a phrase becomes a grey smear and a word does not.
LABELS = {
    LISTENING: "LISTENING",
    WORKING: "WORKING",
    SPEAKING: "TALKING",
}

# How often an animated tile is redrawn. Fast enough that the pulse reads as smooth,
# slow enough that painting it stays a rounding error next to sending the frame.
FRAME_INTERVAL_MS = 100

# The glow, as fractions of the shorter side of the avatar it sits on. A band centred on
# RING_RADIUS from the middle of the picture and faded to nothing over RING_WIDTH either
# way, so the far edge lands at 0.5 - exactly the circle a client's crop leaves behind.
RING_RADIUS = 0.44
RING_WIDTH = 0.045
RING_ALPHA = 0.85
# The halo either side of that band, as a multiple of its width and at a fraction of
# its brightness. Without it the ring is a hoop drawn on the picture rather than light
# coming off it.
RING_BLOOM = 3.4
RING_BLOOM_ALPHA = 0.42

# The label's plate: a pill, wide enough for the longest word and high enough to sit
# clear of it. Positioned by its bottom edge, inset far enough up that its corners stay
# inside the same circle the glow does.
#
# It used to be half the width of the picture and to sit almost on its bottom edge,
# which put it exactly where a meeting client draws the participant's name - two labels
# stacked on one tile, one of them ours. Smaller and lifted clear of that band: the word
# is still the largest thing after the face, and it now reads as belonging to the
# picture rather than competing with the client's own chrome.
LABEL_WIDTH = 0.42
LABEL_HEIGHT = 0.105
LABEL_INSET = 0.16
# Of the plate's own box: how much of it the glyphs may fill, leaving the rest as the
# padding that makes a pill read as a plate rather than as a box round some letters.
LABEL_FILL_WIDTH = 0.78
LABEL_FILL_HEIGHT = 0.42
PLATE_RGB = (0x14, 0x11, 0x0D)
PLATE_ALPHA = 0.62
LABEL_RGB = (0xF7, 0xF8, 0xF8)
LABEL_ALPHA = 0.96
LABEL_FONT = cv2.FONT_HERSHEY_DUPLEX

# --- what it is working on, under the word ----------------------------------
#
# The state word says whether anybody is home; these say what is being done. One pill
# per errand the seat is holding, stacked upward from the word so the bottom of the
# tile stays clear of the meeting client's own name caption.
#
# **These break the one-word rule on purpose, and pay for it.** The comment on LABELS
# above is right that a phrase at thumbnail size becomes a grey smear - so a task pill
# is not competing with the word. It is narrower in its type, dimmer in its plate and
# explicitly second-tier: somebody glancing at a gallery view reads WORKING and nothing
# else, and somebody who has pinned the tile, or is looking at a two-person call where
# it is half the screen, reads what the errand is. The failure mode of getting this
# wrong is a smear under the word, which costs the tile nothing it had before.
TASK_LIMIT = 3
# Two lines per errand: what was asked, and the agent's own latest word on it. The
# second is what makes the tile look alive rather than stuck - it changes every time the
# agent checks in, which is far more often than an errand opens or closes.
TASK_TEXT_LIMIT = 34
TASK_WIDTH = 0.62
TASK_HEIGHT = 0.062
TASK_GAP = 0.014
TASK_FILL_WIDTH = 0.90
TASK_FILL_HEIGHT = 0.40
# Dimmer than the word's own plate, and the type dimmer still: the stack has to read as
# hanging off the word rather than as three more words of equal weight.
TASK_PLATE_ALPHA = 0.50
TASK_RGB = (0xE8, 0xE6, 0xDE)
TASK_ALPHA = 0.88
# The note under a task, dimmer again. Same plate, so a task and its note read as one
# object rather than as two errands.
TASK_NOTE_RGB = (0xB9, 0xB3, 0xA4)
TASK_NOTE_ALPHA = 0.86


def is_animated(state):
    """Whether this state pulses - which is what decides how often a tile is redrawn."""
    return state in GLOW


def draws(state):
    """Whether this state puts anything on the tile at all.

    Not the same question as ``is_animated``: listening draws a word and never moves, so
    it has to be painted once and then left off the fast timer.
    """
    return state in LABELS


def normalize(state):
    """The state as we store it, or None if it is not one we know."""
    if not state:
        return None
    state = str(state).strip().lower()
    return state if state in STATES else None


def pulse(state, elapsed_seconds):
    """How lit the glow is right now, 0..1, on this state's own period."""
    settings = GLOW.get(state)
    if not settings:
        return 0.0
    cycle = settings["cycle_seconds"]
    return 0.5 - 0.5 * math.cos(2 * math.pi * (elapsed_seconds % cycle) / cycle)


def shows_tasks(state):
    """Whether this state draws the task pills at all.

    Only ``working``. The other two states have nothing to name: ``listening`` is not
    holding an errand, and ``speaking`` is delivering one - the room can hear what that
    one is, so writing it under the word is telling them something they are being told.
    """
    return state == WORKING


def label_box(content_width, content_height, rows=0):
    """Where the label's plate sits inside a picture of this size, in pixels.

    ``rows`` lifts the word by the height of the task stack that will hang under it, so
    the *bottom of the whole group* lands where the word's bottom sits when there are no
    tasks. That inset is the clearance from the meeting client's own name caption, and
    it has to be measured from whatever is actually lowest - otherwise adding a second
    errand walks the stack down into the client's chrome one row at a time.
    """
    side = min(content_width, content_height)
    width = side * LABEL_WIDTH
    height = side * LABEL_HEIGHT
    left = (content_width - width) / 2.0
    stack = rows * (side * TASK_HEIGHT + side * TASK_GAP)
    top = content_height - side * LABEL_INSET - height - stack
    return int(round(left)), int(round(top)), int(round(width)), int(round(height))


def clip_line(text):
    """One line of tile-sized text: whitespace collapsed, and cut to something that fits.

    Cut with an ASCII ellipsis rather than the character, because both typesetters here
    are drawing this and OpenCV's Hershey fonts have no glyph for it - a real ellipsis
    arrives on the Zoom tile as a question mark.
    """
    text = " ".join(str(text or "").split())
    if len(text) <= TASK_TEXT_LIMIT:
        return text
    return text[: max(1, TASK_TEXT_LIMIT - 3)].rstrip(" ,;:.-") + "..."


def sanitize_tasks(tasks):
    """The task pills as the tile will draw them: at most TASK_LIMIT, each text + note.

    Accepts a bare string as well as ``{"text": ..., "note": ...}`` so a caller with
    nothing to say about progress does not have to say it in a dict. Anything that
    clips to nothing is dropped rather than drawn as an empty pill.
    """
    cleaned = []
    for task in tasks or []:
        if isinstance(task, dict):
            text, note = task.get("text"), task.get("note")
        else:
            text, note = task, None
        text = clip_line(text)
        if not text:
            continue
        cleaned.append({"text": text, "note": clip_line(note)})
        if len(cleaned) >= TASK_LIMIT:
            break
    return cleaned


def task_rows(tasks):
    """The task pills flattened to the lines that get drawn, each with its own ink.

    A note is its own line under its task rather than part of it: one long pill at this
    size is the smear the module docstring warns about, and two short ones are not.
    """
    rows = []
    for task in sanitize_tasks(tasks):
        rows.append((task["text"], TASK_RGB, TASK_ALPHA))
        if task["note"]:
            rows.append((task["note"], TASK_NOTE_RGB, TASK_NOTE_ALPHA))
    return rows


def task_boxes(content_width, content_height, rows):
    """Where each task line sits: hanging under the state's own plate, in order.

    Under rather than over, because the word is the heading and these are what it is
    about - a reader who has just read WORKING carries on downward. The room's own
    clearance is preserved by lifting the word instead (see ``label_box``), so nothing
    grows into the meeting client's name caption at the bottom of the tile.
    """
    side = min(content_width, content_height)
    _, label_top, _, label_height = label_box(content_width, content_height, rows)
    width = side * TASK_WIDTH
    height = side * TASK_HEIGHT
    gap = side * TASK_GAP
    left = (content_width - width) / 2.0
    below_the_word = label_top + label_height
    boxes = []
    for index in range(rows):
        top = below_the_word + gap + index * (height + gap)
        boxes.append((int(round(left)), int(round(top)), int(round(width)), int(round(height))))
    return boxes


def _rgb_to_yuv(rgb):
    red, green, blue = rgb
    return (
        0.299 * red + 0.587 * green + 0.114 * blue,
        -0.168736 * red - 0.331264 * green + 0.5 * blue + 128.0,
        0.5 * red - 0.418688 * green - 0.081312 * blue + 128.0,
    )


@lru_cache(maxsize=8)
def _ring_mask(width, height):
    """The glow's shape at this size, 0..1, lit hardest on the band itself.

    Cached because it depends only on the geometry: a pulse changes how bright it is,
    never where it is, so the distance field behind it is computed once per tile size
    rather than ten times a second.
    """
    side = min(width, height)
    ys = np.arange(height, dtype=np.float32)[:, None] + 0.5 - height / 2.0
    xs = np.arange(width, dtype=np.float32)[None, :] + 0.5 - width / 2.0
    distance = np.sqrt(xs * xs + ys * ys)
    band = max(1.0, side * RING_WIDTH)
    off = np.abs(distance - side * RING_RADIUS)
    # Two bands, not one. A single band is a neon hoop laid over the picture; the wide
    # dim one spilling off it either way is what makes the same shape read as light
    # coming off the tile. Both are squared, so neither has a visible crease where its
    # ramp begins.
    core = np.square(np.clip(1.0 - off / band, 0.0, 1.0))
    bloom = np.square(np.clip(1.0 - off / (band * RING_BLOOM), 0.0, 1.0))
    return np.clip(core + bloom * RING_BLOOM_ALPHA, 0.0, 1.0)


@lru_cache(maxsize=48)
def _label_masks(text, width, height, fill_width=LABEL_FILL_WIDTH, fill_height=LABEL_FILL_HEIGHT):
    """The plate and the glyphs at this size, as two 0..1 masks over the same box.

    Two rather than one because they are different colours, and one because they are
    the same box: the plate is the reason the word is legible over a photograph, and
    drawing either without the other is worse than drawing neither.
    """
    # Drawn on 8-bit canvases and scaled to 0..1 after: OpenCV's text rasteriser refuses
    # a float image outright, and the shapes are kept on the same footing so the two
    # masks cannot drift apart over an OpenCV upgrade.
    plate = np.zeros((height, width), dtype=np.uint8)
    radius = height // 2
    cv2.rectangle(plate, (radius, 0), (width - radius, height), 255, thickness=-1)
    cv2.circle(plate, (radius, radius), radius, 255, thickness=-1)
    cv2.circle(plate, (width - radius, radius), radius, 255, thickness=-1)

    glyphs = np.zeros((height, width), dtype=np.uint8)
    # Measured at scale 1 and then scaled to the box, because OpenCV has no "fit this
    # text to this width" and stepping scales until one fits costs a measurement each.
    (base_width, base_height), _ = cv2.getTextSize(text, LABEL_FONT, 1.0, 1)
    if base_width > 0 and base_height > 0:
        scale = min(
            width * fill_width / base_width,
            height * fill_height / base_height,
        )
        thickness = max(1, int(round(height * 0.055)))
        (text_width, text_height), _ = cv2.getTextSize(text, LABEL_FONT, scale, thickness)
        cv2.putText(
            glyphs,
            text,
            ((width - text_width) // 2, (height + text_height) // 2),
            LABEL_FONT,
            scale,
            255,
            thickness,
            cv2.LINE_AA,
        )
    return plate.astype(np.float32) / 255.0, glyphs.astype(np.float32) / 255.0


def _blend(plane, left, top, mask, value, alpha):
    """Alpha-blend one mask into one plane at (left, top), clipped to the plane."""
    if alpha <= 0:
        return
    height, width = mask.shape
    plane_height, plane_width = plane.shape
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(plane_width, left + width), min(plane_height, top + height)
    if x1 <= x0 or y1 <= y0:
        return
    weights = mask[y0 - top : y1 - top, x0 - left : x1 - left] * alpha
    region = plane[y0:y1, x0:x1].astype(np.float32)
    plane[y0:y1, x0:x1] = np.clip(region * (1.0 - weights) + value * weights, 0, 255).astype(np.uint8)


def _halve(mask):
    """A mask at chroma resolution. Averaged down rather than sampled, so a one-pixel
    stroke does not disappear from the colour planes and leave a grey word."""
    height, width = mask.shape
    return cv2.resize(mask, (max(1, width // 2), max(1, height // 2)), interpolation=cv2.INTER_AREA)


def paint_i420(frame, width, height, state, elapsed_seconds, content_rect=None, tasks=()):
    """Draw the glow, the label and the task pills into an I420 frame, in place.

    ``frame`` is a writable buffer holding a full I420 image (Y plane, then half-sized
    U and V planes). ``content_rect`` is where the avatar actually is inside the frame -
    scaling a square portrait into a 16:9 video capability letterboxes it, and a glow
    drawn on the frame would ring the black bars instead of the picture.

    A state that draws nothing returns without touching the buffer - including when it
    was handed tasks, because ``off`` means the tile is the customer's picture and
    nothing of ours.
    """
    if not draws(state):
        return
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        return

    x, y, content_width, content_height = content_rect or (0, 0, width, height)
    if content_width <= 0 or content_height <= 0:
        return

    luma_size = width * height
    chroma_width, chroma_height = width // 2, height // 2
    chroma_size = chroma_width * chroma_height
    if len(frame) < luma_size + 2 * chroma_size:
        return

    buffer = np.frombuffer(frame, dtype=np.uint8)
    luma = buffer[:luma_size].reshape(height, width)
    blue_chroma = buffer[luma_size : luma_size + chroma_size].reshape(chroma_height, chroma_width)
    red_chroma = buffer[luma_size + chroma_size : luma_size + 2 * chroma_size].reshape(chroma_height, chroma_width)

    def paint(mask, rgb, alpha, left, top):
        y_value, u_value, v_value = _rgb_to_yuv(rgb)
        _blend(luma, left, top, mask, y_value, alpha)
        half = _halve(mask)
        _blend(blue_chroma, left // 2, top // 2, half, u_value, alpha)
        _blend(red_chroma, left // 2, top // 2, half, v_value, alpha)

    settings = GLOW.get(state)
    if settings:
        lit = pulse(state, elapsed_seconds)
        paint(
            _ring_mask(content_width, content_height),
            settings["rgb"],
            RING_ALPHA * (0.22 + 0.78 * lit),
            x,
            y,
        )

    rows = task_rows(tasks) if shows_tasks(state) else []

    left, top, box_width, box_height = label_box(content_width, content_height, len(rows))
    if box_width > 1 and box_height > 1:
        plate, glyphs = _label_masks(LABELS[state], box_width, box_height)
        paint(plate, PLATE_RGB, PLATE_ALPHA, x + left, y + top)
        paint(glyphs, LABEL_RGB, LABEL_ALPHA, x + left, y + top)

    for (text, rgb, alpha), (left, top, box_width, box_height) in zip(rows, task_boxes(content_width, content_height, len(rows))):
        # A stack tall enough to run off the picture is drawn as far as it fits and no
        # further. TASK_LIMIT bounds this long before it bites on a square avatar; it
        # can still bite on a letterboxed 16:9 frame, where the picture is short.
        if box_width <= 1 or box_height <= 1 or top < 0 or top + box_height > content_height:
            continue
        plate, glyphs = _label_masks(text, box_width, box_height, TASK_FILL_WIDTH, TASK_FILL_HEIGHT)
        paint(plate, PLATE_RGB, TASK_PLATE_ALPHA, x + left, y + top)
        paint(glyphs, rgb, alpha, x + left, y + top)


def letterboxed_content_rect(original_size, frame_size):
    """Where a picture of ``original_size`` lands once scaled into ``frame_size``.

    Mirrors what scale_i420 does - fit inside, preserve aspect, centre on black - so the
    glow can ring the picture rather than the bars beside it.
    """
    original_width, original_height = original_size
    frame_width, frame_height = frame_size
    if original_width <= 0 or original_height <= 0:
        return 0, 0, frame_width, frame_height
    scale = min(frame_width / original_width, frame_height / original_height)
    content_width = max(1, int(round(original_width * scale)))
    content_height = max(1, int(round(original_height * scale)))
    return (
        (frame_width - content_width) // 2,
        (frame_height - content_height) // 2,
        content_width,
        content_height,
    )
