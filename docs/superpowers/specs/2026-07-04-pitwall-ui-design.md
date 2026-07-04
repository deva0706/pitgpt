# Pit Wall UI redesign — design spec

**Approved by user 2026-07-04** ("Go, go, go, go!"). Trigger: outside feedback
that the current UI is weak.

## Direction

F1 pit-wall broadcast: dark carbon surfaces, timing-tower typography,
sharp edges, red/white/teal accents. Messages read as a race-radio
transcript, not floaty chat bubbles.

## Scope

`src/pitgpt/app/theme.py` rewrite (CSS + Gradio theme + header HTML) and
label/layout tweaks in `gradio_app.py`. No changes to model_service, engine,
retrieval, or chat wiring. Fully offline: system fonts only (Segoe UI +
Cascadia Mono/Consolas fallbacks), no CDN.

## Visual system

- Palette: bg `#0e0f12`, panel `#16181d`, hairline `#262a31`,
  F1 red `#e10600`, teal `#00d2be` (PitGPT identity), text `#f2f3f5`,
  secondary `#9aa1ab`.
- Header: carbon strip, pulsing red dot, PITGPT wordmark (heavy italic
  caps), "RACE ENGINEER", right-aligned mono status chips
  (model · FINE-TUNED · 100% LOCAL).
- User rows: right-aligned, mono `YOU ▸` label, red-tinted block,
  3px red right bar, sharp corners.
- Assistant rows: left-aligned, mono `PITGPT` label, panel block,
  3px teal left bar.
- Input strip: placeholder "Radio your question…", solid red uppercase
  SEND button; reset = ghost "🏁 NEW SESSION"; example chips = flat mono
  "SUGGESTED RADIO CALLS".
- Kill: rounded bubble borders, orange bracket decorations, floating copy
  buttons misalignment (hide copy buttons or align them subtly).

## Verification

Relaunch app, screenshot the real page (Chrome tools), iterate once on
visual defects, keep the LinkedIn demo script fully functional.
