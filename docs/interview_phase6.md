# Phase 6 — Interview Questions & Model Answers

### Q1. How does token streaming actually work end-to-end here?

**Answer.** `model.generate()` is blocking, so we run it on a background thread
and give it a `TextIteratorStreamer`. As the model produces tokens, the streamer
decodes and pushes text onto a queue; the main thread iterates that queue,
accumulating a growing string and `yield`ing it. Gradio's chatbot is bound to a
generator callback, so each yield re-renders the assistant message — the user
sees it type out. `demo.queue()` enables this streaming/concurrency path.

### Q2. Why lazy-load the model instead of loading at import time?

**Answer.** Import-time loading couples "importing the module" with "spending
seconds and ~1 GB of VRAM," which is bad for tests, CLIs, and startup. Lazy
loading defers the cost to the first actual message and then caches the model for
all later turns — you pay once, when it's genuinely needed. It also lets the app
import cleanly on a machine with no GPU for inspection.

### Q3. How is multi-turn context maintained, and where does the system prompt live?

**Answer.** Gradio stores the visible transcript as a list of `{"role","content"}`
messages, which we pass back into the engine as prior turns each time, so the
model conditions on the conversation. The **system persona** is *not* stored in
that visible history — the engine prepends it when building the model input. This
avoids showing the persona in the chat and prevents it being duplicated as turns
accumulate.

### Q4. What happens if the user runs the app before training?

**Answer.** `RaceEngineerService.using_adapter` checks the adapter directory on
disk. If it's missing, the service loads the base model instead and the header
shows "base model (no adapter)". So the app always runs; it just serves a
less-specialised model until the adapter exists. Graceful degradation beats a
crash.

### Q5. Why separate `model_service`, `theme`, and `gradio_app`?

**Answer.** Single-responsibility and testability. `model_service` can be unit-
tested without a browser; `theme` can be changed by a designer without touching
inference; `gradio_app` is just wiring. If everything lived in one file, a styling
tweak could break generation and nothing could be tested in isolation. It's the
same modular discipline we apply across the whole `src/pitgpt` package.
