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

import gradio as gr

from pitgpt.app.model_service import service
from pitgpt.app.theme import CUSTOM_CSS, EXAMPLE_PROMPTS, build_theme, header_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pitgpt.app")

Message = dict[str, str]


def _respond(message: str, history: list[Message]):
    """Stream an enthusiast reply, updating the chat transcript live.

    ``history`` is Gradio 'messages' format. We append the user turn and a blank
    assistant turn, then progressively fill the assistant content as tokens
    arrive — each ``yield`` re-renders the chatbot.
    """
    message = (message or "").strip()
    if not message:
        yield history, ""
        return

    prior = list(history)  # turns that existed before this message
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    for partial in service.stream(message, prior):
        history[-1]["content"] = partial
        yield history, ""


def build_demo() -> gr.Blocks:
    """Assemble and return the Gradio Blocks app."""
    from pitgpt.config import settings

    status = "fine-tuned" if service.using_adapter else "base model"
    # Show the real base model (its short name) instead of a hard-coded label.
    model_name = settings.model.base_model_id.split("/")[-1]

    with gr.Blocks(title="PitGPT — F1 Enthusiast") as demo:
        gr.HTML(header_html(status=status, model=model_name))

        # Gradio 6 chatbots use the openai-style messages format by default
        # (a list of {"role", "content"} dicts) — no `type=` argument needed.
        chatbot = gr.Chatbot(
            height=470,
            label="Trackside chat",
            avatar_images=(None, None),
            elem_id="pg-chat",
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask me anything about F1 — a driver, a race, a rule…",
                scale=8,
                show_label=False,
                autofocus=True,
                elem_id="pg-input",
            )
            send_btn = gr.Button("Full send", variant="primary", scale=1, elem_id="pg-send")

        with gr.Row():
            reset_btn = gr.Button("↺ Fresh start", scale=1, elem_id="pg-reset")

        gr.Examples(
            examples=EXAMPLE_PROMPTS, inputs=msg, label="Try asking…", elem_id="pg-examples"
        )

        # Wire events: both Enter and the button submit; streaming outputs go to
        # the chatbot, and the textbox is cleared.
        for trigger in (msg.submit, send_btn.click):
            trigger(_respond, inputs=[msg, chatbot], outputs=[chatbot, msg])

        # Reset simply clears the transcript (model stays loaded).
        reset_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

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
