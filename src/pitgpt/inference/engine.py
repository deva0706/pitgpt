"""
engine.py — the chat/generation layer that sits on top of a loaded model.

Responsibilities
----------------
* Turn a list of chat messages into model-ready input ids using the tokenizer's
  built-in chat template (SmolLM2 uses a ChatML-style template).
* Run ``model.generate()`` with our configured decoding parameters.
* Offer a *streaming* variant that yields tokens as they are produced — this is
  what makes the Gradio UI feel responsive.

Keeping this separate from ``model_loader`` means the UI and the eval harness can
share exactly one definition of "how PitGPT talks".
"""

from __future__ import annotations

import logging
from threading import Thread
from typing import Iterable, Iterator

import torch
from transformers import TextIteratorStreamer

from pitgpt.config import settings
from pitgpt.inference.templating import render_chat

logger = logging.getLogger(__name__)

# A single chat turn: {"role": "system"|"user"|"assistant", "content": str}.
Message = dict[str, str]


def _content_to_text(content) -> str:
    """Coerce a message ``content`` to a plain string.

    SmolLM2's chat template concatenates ``content`` as a string, but some UIs
    (notably Gradio 6's chatbot) store content as a *list* of content-parts, e.g.
    ``[{"type": "text", "text": "…"}]``. Passing that straight to the template
    raises ``TypeError: can only concatenate str (not "list") to str``. We flatten
    any such structure back to text so history round-trips safely.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        return " ".join(p for p in parts if p).strip()
    return "" if content is None else str(content)


def build_messages(
    user_prompt,
    history: list[Message] | None = None,
    context: str = "",
) -> list[Message]:
    """Assemble the full message list, prepending the enthusiast system prompt.

    ``history`` (if given) is the prior turns so multi-turn context is preserved.
    ``context`` (if given) is a block of retrieved facts (RAG) added as a second
    system message so the model can ground its answer without us baking facts into
    the weights. All content is normalised to plain strings so the chat template
    never sees a list (see :func:`_content_to_text`).
    """
    messages: list[Message] = [{"role": "system", "content": settings.model.system_prompt}]
    for turn in history or []:
        role = turn.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _content_to_text(turn.get("content"))
        if text:
            messages.append({"role": role, "content": text})
    # Fold retrieved facts into the *user* turn, right next to the question — small
    # models follow context far better here than in a separate system message.
    user_text = _content_to_text(user_prompt)
    if context:
        user_text = f"{context}\n\nUsing the facts above where relevant, answer: {user_text}"
    messages.append({"role": "user", "content": user_text})
    return messages


def _encode(tokenizer, messages: list[Message], device) -> dict:
    """Apply the chat template and return model inputs on ``device``.

    Newer ``transformers`` (5.x) returns a dict-like ``BatchEncoding`` (with
    ``input_ids`` *and* ``attention_mask``) when ``return_dict=True``; older
    versions returned a bare tensor. We normalise to a dict so callers always get
    an attention mask, which silences generation warnings and is correct for
    left-padded batches.
    """
    encoded = render_chat(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=True,   # append the "assistant:" cue so the model replies
        return_tensors="pt",
        return_dict=True,
    )
    if hasattr(encoded, "to"):
        # BatchEncoding.to() moves every contained tensor to the device at once.
        return encoded.to(device)
    # Fallback for older transformers that return a bare tensor.
    return {"input_ids": encoded.to(device)}


def _generation_kwargs() -> dict:
    """Translate our GenerationConfig dataclass into generate() kwargs."""
    g = settings.generation
    return {
        "max_new_tokens": g.max_new_tokens,
        "temperature": g.temperature,
        "top_p": g.top_p,
        "top_k": g.top_k,
        "repetition_penalty": g.repetition_penalty,
        "do_sample": g.do_sample,
    }


def _rag_context(user_prompt: str, use_rag: bool) -> str:
    """Fetch grounding facts for the prompt (empty if RAG off or no index)."""
    if not use_rag:
        return ""
    # Imported lazily so inference works even without the RAG deps/index.
    from pitgpt.rag.retriever import retrieve_context

    return retrieve_context(user_prompt)


@torch.inference_mode()
def generate_reply(
    model,
    tokenizer,
    user_prompt: str,
    history: list[Message] | None = None,
    use_rag: bool = True,
) -> str:
    """Generate a single, complete enthusiast reply (blocking), grounded via RAG."""
    context = _rag_context(user_prompt, use_rag)
    messages = build_messages(user_prompt, history, context=context)
    inputs = _encode(tokenizer, messages, model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    output_ids = model.generate(
        **inputs,
        pad_token_id=tokenizer.pad_token_id,
        **_generation_kwargs(),
    )
    # Only decode the newly generated tokens, not the prompt we fed in.
    new_tokens = output_ids[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@torch.inference_mode()
def stream_reply(
    model,
    tokenizer,
    user_prompt: str,
    history: list[Message] | None = None,
    use_rag: bool = True,
) -> Iterator[str]:
    """Yield the reply token-by-token as a growing string (for the UI), RAG-grounded.

    We run ``generate()`` on a background thread and pull decoded text off a
    ``TextIteratorStreamer`` queue. Each yield is the *cumulative* text so far,
    which is exactly what Gradio's chat expects.
    """
    context = _rag_context(user_prompt, use_rag)
    messages = build_messages(user_prompt, history, context=context)
    inputs = _encode(tokenizer, messages, model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "pad_token_id": tokenizer.pad_token_id,
        **_generation_kwargs(),
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial = ""
    for token_text in streamer:
        partial += token_text
        yield partial
    thread.join()


def batch_replies(
    model,
    tokenizer,
    prompts: Iterable[str],
) -> list[str]:
    """Generate one reply per prompt sequentially. Used by the eval harness."""
    return [generate_reply(model, tokenizer, p) for p in prompts]
