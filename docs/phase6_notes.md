# Phase 6 — Chat Application (Teaching Notes)

## Goal and shape

A local, F1-themed chat UI over the fine-tuned model, with streaming, multi-turn
history, adapter loading, and a reset button. The design keeps three concerns in
three files so the interface stays modular:

| File | Responsibility |
|---|---|
| `app/model_service.py` | Owns the model: lazy load, adapter fallback, `stream()` |
| `app/theme.py` | Visual identity: CSS, header, example prompts |
| `app/gradio_app.py` | Wires widgets to callbacks — no model logic |

This separation means you can restyle the app without risking the inference code,
and test the service without launching a UI.

## Key design decisions

### Lazy, single-load model
`RaceEngineerService` loads the model **once**, on the first message, then reuses
it. Loading on import would make the app slow to start and load the model even
for a `--help`; loading per message would be absurdly slow. Lazy-once is the
right trade.

### Adapter fallback
`using_adapter` checks whether the trained adapter exists on disk. If it does, we
serve base+adapter (the real PitGPT); if not, we serve the base model and say so
in the header. The app therefore runs **before** training completes or if
training was skipped — important for a robust demo.

### Streaming
`stream()` yields the cumulative reply string from the engine's
`TextIteratorStreamer` (Phase 1). In Gradio we append a blank assistant turn and
overwrite its content on each yield, so the user sees the answer type out live.
This is what makes a small local model *feel* responsive even while it generates.

### Multi-turn history
Gradio's `type="messages"` chatbot stores turns as `{"role", "content"}` dicts —
the exact format our engine consumes as prior context. We pass the prior turns
back into `stream()` so the engineer "remembers" the conversation. The system
persona prompt is added by the engine, not stored in the visible history.

### Reset
The reset button clears the transcript (`[]`) but leaves the model loaded — you
get a fresh conversation instantly without paying the load cost again.

## Common app mistakes this avoids

- **Reloading the model each turn** → unusable latency. (Fixed by the singleton.)
- **Storing the system prompt in visible history** → it shows in the chat and can
  get duplicated. (We inject it in the engine only.)
- **Blocking generation** → the UI freezes until the whole reply is done. (Fixed
  by streaming + `demo.queue()`.)
- **Hard-coding the adapter path** → breaks on a machine that hasn't trained.
  (Fixed by the on-disk `using_adapter` check with graceful fallback.)
- **Coupling UI and model** → you can't restyle or test independently. (Fixed by
  the three-file split.)

## Running it

```bash
python -m pitgpt.app.gradio_app      # or the `pitgpt-chat` console script
```

Then open the printed local URL. Screenshots go in the README's Screenshots
section (placeholder provided).
