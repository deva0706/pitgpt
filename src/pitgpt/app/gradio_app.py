"""
gradio_app.py — the PitGPT Race Engineer chat interface.

Features (per the project spec):
* chat history (multi-turn context passed back to the model),
* streaming responses (token-by-token via the engine's streamer),
* fully local inference (no external API),
* automatic LoRA adapter loading (falls back to base model if absent),
* a reset-chat button,
* a modern, F1-themed interface.

The UI is intentionally thin: all model work lives in ``model_service`` and the
look lives in ``theme``. This module only wires widgets to callbacks.

Run:  ``python -m pitgpt.app.gradio_app``   (or the ``pitgpt-chat`` console script)
"""

from __future__ import annotations

import logging
import os
import warnings

# "100% local" has to hold for the web framework too: by default Gradio reports
# each launch to its own servers and asks them for the latest version. Set
# before the import, because that is when it is read.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
# Gradio reads a status constant that Starlette has since renamed, and
# Starlette says so in the terminal for every message sent (49 times in a
# 49-turn session, measured 2026-10-11). Nothing in this app can act on it.
warnings.filterwarnings("ignore", message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*")

import gradio as gr  # noqa: E402

from pitgpt.app.model_service import service  # noqa: E402
from pitgpt.app.theme import CUSTOM_CSS, EXAMPLE_PROMPTS, build_theme, header_html  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pitgpt.app")

Message = dict[str, str]


def _respond(message: str, history: list[Message]):
    """Stream an enthusiast reply, updating the chat transcript live.

    ``history`` is Gradio 'messages' format. The user's turn is shown at
    once — the first message of a session waits for the model to load, and
    until this was done the transcript stayed empty all that time, as if
    nothing had been sent. The reply is then filled in as it arrives; each
    ``yield`` re-renders the chatbot.
    """
    message = (message or "").strip()
    if not message:
        yield history, ""
        return

    prior = list(history)  # turns that existed before this message
    history = prior + [{"role": "user", "content": message}]
    yield history, ""
    for partial in service.stream(message, prior):
        yield history + [{"role": "assistant", "content": partial}], ""


def build_demo() -> gr.Blocks:
    """Assemble and return the Gradio Blocks app."""
    from pitgpt.config import settings

    status = "fine-tuned" if service.using_adapter else "base model"
    # Show the real base model (its short name) instead of a hard-coded label.
    model_name = settings.model.base_model_id.split("/")[-1]

    with gr.Blocks(title="PitGPT — F1 Enthusiast", analytics_enabled=False) as demo:
        gr.HTML(header_html(status=status, model=model_name))

        # Gradio 6 chatbots use the openai-style messages format by default
        # (a list of {"role", "content"} dicts) — no `type=` argument needed.
        chatbot = gr.Chatbot(
            height=470,
            label="Race radio",
            avatar_images=(None, None),
            elem_id="pg-chat",
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Radio your question — a driver, a race, a rule…",
                scale=8,
                show_label=False,
                autofocus=True,
                elem_id="pg-input",
            )
            send_btn = gr.Button("Send", variant="primary", scale=1, elem_id="pg-send")

        with gr.Row():
            # Clears the transcript in the browser (the model stays loaded). It
            # used to be a round trip that also emptied the question box, so a
            # message typed straight after the click could be wiped before it
            # was sent.
            gr.ClearButton([chatbot], value="🏁 New session", scale=1, elem_id="pg-reset")

        gr.Examples(
            examples=EXAMPLE_PROMPTS, inputs=msg,
            label="Suggested radio calls", elem_id="pg-examples",
        )

        # Wire events: both Enter and the button submit; streaming outputs go to
        # the chatbot, and the textbox is cleared.
        for trigger in (msg.submit, send_btn.click):
            trigger(_respond, inputs=[msg, chatbot], outputs=[chatbot, msg])

    return demo


def main() -> None:
    demo = build_demo()
    demo.queue()  # enable streaming / concurrency
    # Gradio 6 moved css/theme to launch().
    demo.launch(
        server_name="127.0.0.1",
        inbrowser=False,
        css=CUSTOM_CSS,
        theme=build_theme(),
    )


if __name__ == "__main__":
    main()
