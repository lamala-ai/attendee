"""The bot's own presence indicator: a small pulsing bead drawn over its avatar.

A bot that is sitting quietly in a meeting looks exactly like a bot whose process died
twenty minutes ago, and sitting quietly is what a bot does for most of a call. This
draws a mark on the video it is already sending, so the room can see the difference.

Three states, set through PATCH /bots/{id}/presence_indicator:

* ``listening`` - a slow pulse. Present, nothing outstanding.
* ``working`` - the same bead, beating faster. Something was asked of it and the answer
  has not come back yet.
* ``speaking`` (or ``off``) - nothing is drawn. While the bot is talking the room has
  better evidence than a dot.

The important property is that **nothing is sent per frame**. The state is set once and
the bot animates its own tile from the frames it already emits: the Zoom adapter paints
the bead into the I420 frame it re-sends on a timer, and the web adapters draw it onto
the canvas they already capture. So a pulse costs one API call, not one call per frame.

Both paths use the geometry and colours here, so the tile looks the same on every
platform - the JavaScript in web_bot_adapter/shared_chromedriver_payload.js repeats
these constants rather than inventing its own.
"""

import math

import numpy as np

LISTENING = "listening"
WORKING = "working"
SPEAKING = "speaking"
OFF = "off"

STATES = [LISTENING, WORKING, SPEAKING, OFF]

# Colour and pulse period per animated state. The colours are the ones the product's
# own status marks use: green for present, amber for something still owed.
ANIMATED = {
    LISTENING: {"rgb": (0x75, 0xD8, 0x7A), "cycle_seconds": 2.8},
    WORKING: {"rgb": (0xF6, 0xD7, 0x95), "cycle_seconds": 0.9},
}

# How often an animated tile is redrawn. Fast enough that the pulse reads as smooth,
# slow enough that painting it stays a rounding error next to sending the frame.
FRAME_INTERVAL_MS = 100

# The bead, as fractions of the shorter side of the avatar it sits on. Inset well
# inside the picture rather than at its edge, because meeting clients crop tiles to
# fill and an indicator that can be cropped away is worse than none.
BEAD_RADIUS = 0.052
BEAD_INSET = 0.10
# Behind the bead: a glow in its own colour, so the pulse reads as light rather than as
# a dot changing size, and a thin dark rim so the edge survives a pale avatar. A dark
# halo was tried first and looked like a smudge on the charcoal portraits the product
# ships with.
GLOW_RADIUS = 2.0
GLOW_ALPHA = 0.20
RIM_RADIUS = 1.14
RIM_ALPHA = 0.5
RIM_RGB = (0x14, 0x11, 0x0D)


def is_animated(state):
    """Whether this state draws anything at all."""
    return state in ANIMATED


def normalize(state):
    """The state as we store it, or None if it is not one we know."""
    if not state:
        return None
    state = str(state).strip().lower()
    return state if state in STATES else None


def pulse(state, elapsed_seconds):
    """How lit the bead is right now, 0..1, on this state's own period."""
    settings = ANIMATED.get(state)
    if not settings:
        return 0.0
    cycle = settings["cycle_seconds"]
    return 0.5 - 0.5 * math.cos(2 * math.pi * (elapsed_seconds % cycle) / cycle)


def bead_geometry(content_width, content_height):
    """Centre and radius of the bead within a picture of this size, in pixels."""
    side = min(content_width, content_height)
    radius = max(3.0, side * BEAD_RADIUS)
    inset = side * BEAD_INSET
    return inset + radius, content_height - inset - radius, radius


def _rgb_to_yuv(rgb):
    red, green, blue = rgb
    return (
        0.299 * red + 0.587 * green + 0.114 * blue,
        -0.168736 * red - 0.331264 * green + 0.5 * blue + 128.0,
        0.5 * red - 0.418688 * green - 0.081312 * blue + 128.0,
    )


def _blend_disc(plane, center_x, center_y, radius, value, alpha, falloff=False):
    """Alpha-blend a disc into one plane, touching only its bounding box.

    ``falloff`` fades the disc out from the centre instead of filling it flat - which is
    the difference between a glow and a second circle drawn behind the first.
    """
    if alpha <= 0 or radius <= 0:
        return
    height, width = plane.shape
    left = max(0, int(math.floor(center_x - radius)) - 1)
    right = min(width, int(math.ceil(center_x + radius)) + 1)
    top = max(0, int(math.floor(center_y - radius)) - 1)
    bottom = min(height, int(math.ceil(center_y + radius)) + 1)
    if right <= left or bottom <= top:
        return

    ys = np.arange(top, bottom, dtype=np.float32)[:, None] + 0.5 - center_y
    xs = np.arange(left, right, dtype=np.float32)[None, :] + 0.5 - center_x
    distance = np.sqrt(xs * xs + ys * ys)
    if falloff:
        mask = np.square(np.clip(1.0 - distance / radius, 0.0, 1.0)) * alpha
    else:
        # One pixel of feather, so the edge is not a staircase once the tile is scaled.
        mask = np.clip(radius - distance, 0.0, 1.0) * alpha

    region = plane[top:bottom, left:right].astype(np.float32)
    plane[top:bottom, left:right] = np.clip(region * (1.0 - mask) + value * mask, 0, 255).astype(np.uint8)


def paint_i420(frame, width, height, state, elapsed_seconds, content_rect=None):
    """Draw the bead into an I420 frame, in place.

    ``frame`` is a writable buffer holding a full I420 image (Y plane, then half-sized
    U and V planes). ``content_rect`` is where the avatar actually is inside the frame -
    scaling a square portrait into a 16:9 video capability letterboxes it, and a bead in
    the corner of the frame would sit on the black bar instead of on the picture.

    A state that draws nothing returns without touching the buffer.
    """
    if not is_animated(state):
        return
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        return

    x, y, content_width, content_height = content_rect or (0, 0, width, height)
    if content_width <= 0 or content_height <= 0:
        return
    center_x, center_y, radius = bead_geometry(content_width, content_height)
    center_x += x
    center_y += y

    lit = pulse(state, elapsed_seconds)
    alpha = 0.30 + 0.70 * lit
    radius = radius * (0.78 + 0.22 * lit)

    luma_size = width * height
    chroma_width, chroma_height = width // 2, height // 2
    chroma_size = chroma_width * chroma_height
    if len(frame) < luma_size + 2 * chroma_size:
        return

    buffer = np.frombuffer(frame, dtype=np.uint8)
    luma = buffer[:luma_size].reshape(height, width)
    blue_chroma = buffer[luma_size : luma_size + chroma_size].reshape(chroma_height, chroma_width)
    red_chroma = buffer[luma_size + chroma_size : luma_size + 2 * chroma_size].reshape(chroma_height, chroma_width)

    colour = ANIMATED[state]["rgb"]
    for rgb, disc_radius, disc_alpha, falloff in (
        (colour, radius * GLOW_RADIUS, GLOW_ALPHA * alpha, True),
        (RIM_RGB, radius * RIM_RADIUS, RIM_ALPHA * alpha, False),
        (colour, radius, alpha, False),
    ):
        y_value, u_value, v_value = _rgb_to_yuv(rgb)
        _blend_disc(luma, center_x, center_y, disc_radius, y_value, disc_alpha, falloff)
        _blend_disc(blue_chroma, center_x / 2, center_y / 2, disc_radius / 2, u_value, disc_alpha, falloff)
        _blend_disc(red_chroma, center_x / 2, center_y / 2, disc_radius / 2, v_value, disc_alpha, falloff)


def letterboxed_content_rect(original_size, frame_size):
    """Where a picture of ``original_size`` lands once scaled into ``frame_size``.

    Mirrors what scale_i420 does - fit inside, preserve aspect, centre on black - so the
    bead can be placed on the picture rather than on the bars beside it.
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
