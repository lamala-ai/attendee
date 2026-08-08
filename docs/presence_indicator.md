# Presence Indicator

A bot sitting quietly in a meeting looks exactly like a bot whose process died twenty
minutes ago — and sitting quietly is what a bot does for most of a call. The presence
indicator draws a small pulsing bead over the bot's own avatar so the room can see the
difference.

## Setting it

```bash
curl -X PATCH https://your-attendee.example.com/api/v1/bots/bot_xxxxxxxx/presence_indicator \
  -H 'Authorization: Token YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"state": "working"}'
```

| State | What the tile shows |
| --- | --- |
| `listening` | a slow pulse (2.8s) — present, nothing outstanding |
| `working` | the same bead at 0.9s — something was asked and the answer has not come back |
| `speaking` | nothing at all — while the bot is talking the room has better evidence than a dot |
| `off` | nothing at all |

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
exactly as it always did.

## Where it is drawn

The bead sits inside the bottom-left of the *picture*, not the frame: a square avatar
scaled into a 16:9 video capability is letterboxed, and a bead in the corner of the frame
would land on the black bar beside the portrait. It is inset well inside the picture
because meeting clients crop tiles to fill, and an indicator that can be cropped away is
worse than none.

Colours, periods and geometry live in `bots/presence_indicator.py`; the web adapters
repeat them at the top of `bots/web_bot_adapter/shared_chromedriver_payload.js` so a tile
looks the same whichever adapter drew it. Change one, change the other.

## Support

| Adapter | Draws it |
| --- | --- |
| Zoom (native SDK) | yes |
| Google Meet, Microsoft Teams (web) | yes |
| Zoom RTMS | no — it has no video output of its own; the call is accepted and logged |

An adapter that cannot draw inherits a no-op from `BotAdapter`, because a cosmetic mark
is never worth failing a meeting over.
