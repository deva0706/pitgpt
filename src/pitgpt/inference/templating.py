"""
templating.py — robustly apply a tokenizer chat template across model families.

Different base models expose different chat templates. Notably:

* ChatML-style templates (e.g. SmolLM2) accept a ``system`` role.
* **Gemma** does *not* — its template raises if you pass a ``system`` message.

To keep the rest of the codebase model-agnostic, we always *build* messages with
a system persona (see ``engine.build_messages``) and let this helper adapt at
render time: it first tries the template as-is, and if that fails it folds the
system content into the first user turn and retries. That way switching the base
model (e.g. SmolLM2 -> Gemma) needs no changes anywhere else.
"""

from __future__ import annotations


def _merge_system_into_first_user(messages: list[dict]) -> list[dict]:
    """Return a copy of ``messages`` with any system content prepended to the
    first user turn and the system messages removed.

    Used as a fallback for templates (Gemma) that don't support a system role.
    """
    system_texts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
    rest = [dict(m) for m in messages if m.get("role") != "system"]
    if not system_texts:
        return rest

    prefix = "\n\n".join(system_texts).strip()
    for turn in rest:
        if turn.get("role") == "user":
            turn["content"] = f"{prefix}\n\n{turn['content']}".strip()
            return rest
    # No user turn to attach to — prepend a synthetic one.
    return [{"role": "user", "content": prefix}, *rest]


def render_chat(
    tokenizer,
    messages: list[dict],
    *,
    tokenize: bool = False,
    add_generation_prompt: bool = True,
    return_tensors: str | None = None,
    return_dict: bool = False,
):
    """Apply the chat template, falling back to a system-less layout if needed.

    Mirrors ``tokenizer.apply_chat_template`` but transparently handles templates
    that reject the ``system`` role. Returns whatever ``apply_chat_template``
    would (a string when ``tokenize=False``; a tensor/BatchEncoding otherwise).
    """
    kwargs = {"tokenize": tokenize, "add_generation_prompt": add_generation_prompt}
    if tokenize:
        kwargs["return_tensors"] = return_tensors
        kwargs["return_dict"] = return_dict

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        # Most commonly: "System role not supported" (Gemma). Merge and retry.
        merged = _merge_system_into_first_user(messages)
        return tokenizer.apply_chat_template(merged, **kwargs)
