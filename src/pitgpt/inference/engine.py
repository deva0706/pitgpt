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
import re
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


def _generation_kwargs(grounded: bool = True) -> dict:
    """Translate our GenerationConfig dataclass into generate() kwargs.

    Ungrounded persona turns get a shorter budget: the fabricated years and
    stats in free chat live in sentence four onward of the long rambles
    (measured live — "in '17 he won it" about Alonso). Short and warm beats
    long and wrong."""
    g = settings.generation
    return {
        "max_new_tokens": g.max_new_tokens if grounded else min(g.max_new_tokens, 110),
        "temperature": g.temperature,
        "top_p": g.top_p,
        "top_k": g.top_k,
        "repetition_penalty": g.repetition_penalty,
        "do_sample": g.do_sample,
    }


def _rag_context(user_prompt: str, use_rag: bool) -> str:
    """Fetch grounding facts for the prompt (empty if RAG off or no index).

    RE-ENABLED with the full 2,270-doc hybrid index (dense + BM25 + RRF). The
    earlier 62-doc corpus echoed wrong-entity facts for uncovered names; the
    complete Kaggle-derived knowledge base closes that coverage hole.
    """
    if not use_rag:
        return ""
    # Imported lazily so inference works even without the RAG deps/index.
    from pitgpt.rag.retriever import retrieve_context

    return retrieve_context(user_prompt)


# --- conversation recall: chat history as a retrieval source ---------------- #
# Raw prior turns in the prompt made the single-turn-trained model IMITATE its
# own formatted answers ("Rosberg is the 2017 champion", "Nico Mansell"). The
# same mechanism becomes a feature when relevant history is injected as LABELED
# CONTEXT instead: the trained behavior is to recite context, so recalled chat
# facts get read, not parroted. Recall fires only on an explicit reference cue
# ("he", "my favourite", "earlier", ...) or a shared driver-name/year — the
# 2016→2017 contamination case shares neither, so it recalls nothing.
_RECALL_CUE = re.compile(
    r"\b(he|she|they|him|her|his|hers|their|it|that (race|season|driver|team|year)|"
    r"my favou?rite|i (said|mentioned|told|asked)|earlier|before|again|same)\b", re.I)

# "What are MY favourite circuits?" asks about the USER, not about F1 — corpus
# docs are pure noise there, and a retrieved Suzuka doc beat the user's own
# words in live testing ("you love Suzuka!" to someone who never said it).
# Enumeration questions about the user answer from chat recall ALONE.
# Deliberately narrow: "What teams did my favourite driver race for?" still
# needs retrieval and does not match.
_SELF_REFERENTIAL = re.compile(
    # non-anchored: "Tell me what are my favourite GPs" must match too.
    # "What teams did my favourite driver..." stays safe — that's "did my",
    # not "is/are my".
    r"\b(what|which)\s+(is|are|was|were)\s+my\b"
    r"|\btell me my\b"
    r"|\bwhat did i (say|tell|mention|ask)\b", re.I)

# The radio log must quote what the user STATED, never their questions —
# live it echoed "Tell me what are my favourite Grand Prix" back as a
# "transmission", and a recalled junk question turned an unrelated statement
# turn into a grounded decline about corn coffee.
_QUESTIONISH = re.compile(
    r"\?|^\W*(so\s+|yes\s+|ok(ay)?\s+)*"
    r"(what|who|when|where|which|why|how|tell|explain|give|can|could|do|does|did)\b",
    re.I)


def _radio_log_reply(user_prompt: str, history: list[Message] | None) -> str | None:
    """Deterministic answer for self-referential questions ("what are my
    favourite circuits?"): quote the user's own logged turns back verbatim.

    The adapter was never trained on chat-recall contexts, and measured live
    it either spotlighted one item or fabricated a fresh list ("Monaco,
    Silverstone, Suzuka" to a user who said Yas Marina and Singapore). Until
    a recall-trained adapter ships, the radio log answers these itself —
    quoting is the one strategy that cannot hallucinate.
    """
    if not (history and _SELF_REFERENTIAL.search(user_prompt)):
        return None
    statements = [_content_to_text(t.get("content"))[:220]
                  for t in history[-8:] if t.get("role") == "user"]
    statements = [s for s in statements
                  if s and _RECALL_CUE.search(s) and not _QUESTIONISH.search(s)]
    if not statements:
        return ("Checking the radio log — nothing on that from you yet. "
                "Tell me and I'll keep it on file!")
    quoted = "\n".join(f"▸ “{s}”" for s in statements[-3:])
    return (f"Checking the radio log — here's what you've told me so far:\n"
            f"{quoted}\n"
            f"That's straight from your own transmissions. Want the full "
            f"telemetry on any of them?")


def _chat_recall(user_prompt: str, history: list[Message] | None) -> tuple[str, list[str]]:
    """Return (labeled recall block, entity names found) from relevant turns."""
    if not history:
        return "", []
    from pitgpt.rag.retriever import _fold, _PIN_NAME, _PIN_YEAR

    q_names = {_fold(n) for n in _PIN_NAME.findall(user_prompt)}
    q_years = set(_PIN_YEAR.findall(user_prompt))
    cued = bool(_RECALL_CUE.search(user_prompt))
    # A question that already names its entity needs nothing resolved —
    # injecting recall there only offers old answers to imitate (measured
    # live: a teammates question answered with a verbatim repeat of the
    # previous win-count reply).
    if q_names and not cued:
        return "", []
    picked: list[tuple[str, str, list[str]]] = []
    # Gradio 6 stores content as a LIST of parts — a plain isinstance(str)
    # filter here silently dropped every real turn (recall worked in tests,
    # never in the app). Flatten like build_messages does.
    recent = [(t.get("role", "user"), _content_to_text(t.get("content")))
              for t in history[-8:]]
    recent = [(role, text) for role, text in recent if text]
    for i, (role, text) in enumerate(recent):
        text = text[:220]
        raw_names = _PIN_NAME.findall(text)          # original case, for query use
        t_names = {_fold(n) for n in raw_names}
        t_years = set(_PIN_YEAR.findall(text))
        entity_hit = bool((q_names & t_names) or (q_years & t_years))
        cue_hit = cued and i >= len(recent) - 2      # cue → last exchange
        # a cue like "my favourite" also recalls the turn where the user
        # STATED it, however far back
        statement_hit = cued and role == "user" and _RECALL_CUE.search(text)
        if entity_hit or cue_hit or statement_hit:
            picked.append((role, text, raw_names))
    if not picked:
        return "", []
    lines, names, stated_names = [], [], []
    for role, text, raw_names in picked[-3:]:
        # Assistant turns contribute NAMES (so "he" resolves and enriches the
        # retrieval query) but never their TEXT — trained-format answers in
        # the context are an imitation surface, not a memory. User QUESTIONS
        # also stay out of the quoted lines (their names still count): a
        # recalled junk question in context derailed an unrelated turn live.
        names.extend(raw_names)
        if role == "user" and not _QUESTIONISH.search(text):
            lines.append(f"- Earlier in this chat, the user said: {text}")
            stated_names.extend(raw_names)
    # "Who is my favorite driver?" after a Vettel statement pulled names from
    # a Hamilton question too, and the mixed-entity context produced nonsense.
    # A preference the user STATED beats names that merely appeared in
    # questions or answers.
    if stated_names:
        names = stated_names
    if not lines and not names:
        return "", []
    block = ("From earlier in this conversation (chat recall):\n" + "\n".join(lines)
             if lines else "")
    return block, names


@torch.inference_mode()
def generate_reply(
    model,
    tokenizer,
    user_prompt: str,
    history: list[Message] | None = None,
    use_rag: bool = True,
    context: str | None = None,
) -> str:
    """Generate a single, complete enthusiast reply (blocking), grounded via RAG.

    ``context`` (if given) is used verbatim instead of live retrieval — this is
    how the evaluation harness injects *gold* context to measure generation
    grounding in isolation from retrieval quality.
    """
    if context is None:
        # Radio log outranks the conversational gate: "what is my favourite
        # corn drink?" is off-topic AND self-referential — the log must answer
        # it, not persona rambling.
        log_reply = _radio_log_reply(user_prompt, history)
        if log_reply is not None:
            return log_reply
        from pitgpt.rag.retriever import _is_conversational
        if _is_conversational(user_prompt):
            context = ""    # statements/chat: persona path, history intact
        else:
            recall, names = _chat_recall(user_prompt, history)
            query = (f"{user_prompt} {' '.join(dict.fromkeys(names))}"
                     if names else user_prompt)
            context = _rag_context(query, use_rag)
            if recall:
                context = f"{context}\n\n{recall}" if context else recall
    if context:
        history = None   # grounded turns answer fresh — see stream_reply
    messages = build_messages(user_prompt, history, context=context)
    inputs = _encode(tokenizer, messages, model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    output_ids = model.generate(
        **inputs,
        pad_token_id=tokenizer.pad_token_id,
        **_generation_kwargs(grounded=bool(context)),
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
    # The adapter is trained single-turn: with chat history in the prompt, the
    # model imitates its own prior formatted answers instead of reading the
    # fresh context (measured live: "Rosberg is the 2017 champion" after a
    # correct 2016 answer, and the name-blend "Nico Mansell"). Grounded turns
    # therefore answer FRESH from the retrieved facts; history reaches them
    # only as labeled chat-recall context (recited, not imitated), and raw
    # history is kept for persona/chat turns, where continuity is the point.
    log_reply = _radio_log_reply(user_prompt, history)
    if log_reply is not None:
        yield log_reply
        return
    from pitgpt.rag.retriever import _is_conversational
    if _is_conversational(user_prompt):
        context = ""        # statements/chat: persona path, history intact
    else:
        recall, names = _chat_recall(user_prompt, history)
        query = (f"{user_prompt} {' '.join(dict.fromkeys(names))}"
                 if names else user_prompt)
        context = _rag_context(query, use_rag)
        if recall:
            context = f"{context}\n\n{recall}" if context else recall
    if context:
        history = None
    messages = build_messages(user_prompt, history, context=context)
    inputs = _encode(tokenizer, messages, model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "pad_token_id": tokenizer.pad_token_id,
        **_generation_kwargs(grounded=bool(context)),
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
