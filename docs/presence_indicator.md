# Presence Indicator

A bot sitting quietly in a meeting looks exactly like a bot whose process died twenty
minutes ago — and sitting quietly is what a bot does for most of a call. The presence
indicator says what the bot is doing in a word on its own tile, and glows around the
avatar when there is something a word alone would be too slow to say.

## Setting it

```bash
curl -X PATCH https://your-attendee.example.com/api/v1/bots/bot_xxxxxxxx/presence_indicator \
  -H 'Authorization: Token YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"state": "working"}'
```

| State | What the tile shows |
| --- | --- |
| `listening` | the word **LISTENING**, and no glow — the ordinary state, and the one the room looks at for most of the meeting |
| `working` | **WORKING**, and a white glow beating every 0.9s — something was asked and the answer has not come back |
| `speaking` | **TALKING**, and a green glow breathing every 1.6s — the room can hear it, so the glow is there to tie the voice to the tile rather than to inform |
| `off` | nothing at all |

A word rather than only a colour, because a colour has to be explained beforehand and a
word does not. A glow around the whole picture rather than a bead, because a five-pixel
dot in a tile drawn 120 pixels wide is a colour and nothing else.

The bot must be in a state that can play media, exactly like `output_image`. A bot that
is still joining is refused with a `400`; retry once it is in the meeting.

## What it costs

**One call per state change, and nothing per frame.** The state is stored on the bot and
the bot animates the video it is already sending: the Zoom adapter paints the bead into
the I420 frame it re-sends on a timer, and the web adapters draw it onto the canvas that
is already captured as their video track. Pushing an animation as a series of
`output_image` calls would cost an HTTP request, an image decode and a database row per
frame; this costs neither.

A tile with no indicator is sent at its old cadence (a still every 500ms on Zoom, a
canvas redraw every second on the web adapters), so a bot that never sets one behaves
exactly as it always did. `listening` keeps that cadence too: it draws a word that never
changes, so it is painted once rather than ten times a second.

## Where it is drawn

Everything is measured off the *picture*, not the frame: a square avatar scaled into a
16:9 video capability is letterboxed, and a ring measured off the frame would circle the
black bars beside the portrait instead of the face.

Within the picture, the lit band of the glow rides at 0.44 of the shorter side out from
the middle and the label sits above the bottom by about the same margin — both inside the
circle a square picture survives being cropped to, because meeting clients crop tiles to
fill and an indicator that can be cropped away is worse than none. The halo either side
of the band may cross that circle; it is already fading to nothing out there.

Colours, periods, words and geometry live in `bots/presence_indicator.py`; the web
adapters repeat them at the top of `bots/web_bot_adapter/shared_chromedriver_payload.js`
so a tile looks the same whichever adapter drew it. Change one, change the other. The one
thing that cannot be repeated is the glyphs — that side has OpenCV's stroked font and
this side has the browser's — so the label is the same word in the same box at the same
size, drawn by two typesetters.

## Support

| Adapter | Draws it |
| --- | --- |
| Zoom (native SDK) | yes |
| Google Meet, Microsoft Teams (web) | yes |
| Zoom RTMS | no — it has no video output of its own; the call is accepted and logged |

An adapter that cannot draw inherits a no-op from `BotAdapter`, because a cosmetic mark
is never worth failing a meeting over.
