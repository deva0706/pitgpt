"""
data_module.py — load the clean dataset and render it for SFT.

The trainer needs each example as a single string in exactly the chat format the
model will see at inference. We take the validated Alpaca rows
(``instruction``/``input``/``output``) and render them through the tokenizer's
**chat template** with the PitGPT system prompt, producing a ``text`` column that
``SFTTrainer`` consumes directly.

Rendering here (rather than letting the trainer guess) guarantees train-time and
inference-time formatting are byte-for-byte identical — the single most common
cause of "it trained fine but behaves oddly when I chat with it".
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from pitgpt.config import settings
from pitgpt.inference.engine import build_messages
from pitgpt.inference.templating import render_chat


def _to_messages(row: dict) -> list[dict[str, str]]:
    """Turn one row into a system/user/assistant message list.

    Crucially, we build the user turn with the SAME ``build_messages`` used at
    inference — so if a row carries retrieved ``context``, it is folded into the
    user turn exactly as it will be at serve time. This train/serve parity is what
    lets the model actually learn to ground on retrieved context.
    """
    question = row["instruction"]
    if row.get("input"):
        question = f"{question}\n\n{row['input']}"
    messages = build_messages(question, history=None, context=row.get("context", ""))
    messages.append({"role": "assistant", "content": row["output"]})
    return messages


def load_clean_rows(path: Path | None = None) -> list[dict]:
    """Read the validated clean dataset JSON."""
    path = path or settings.paths.dataset_clean
    if not path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at {path}. Run "
            "`python -m pitgpt.data.build_dataset` then "
            "`python -m pitgpt.data.validate_dataset` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_sft_dataset(tokenizer, path: Path | None = None) -> Dataset:
    """Return a HF ``Dataset`` with a single ``text`` column, chat-templated.

    ``add_generation_prompt=False`` because the assistant's answer is *included*
    in the text we train on — we are not asking the model to continue an empty
    assistant turn, we are showing it the whole exchange.
    """
    rows = load_clean_rows(path)

    def render(row: dict) -> dict:
        messages = _to_messages(row)
        # render_chat folds the system persona into the first user turn for
        # templates (e.g. Gemma) that don't accept a system role.
        text = render_chat(
            tokenizer, messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list(rows)
    dataset = dataset.map(render, remove_columns=dataset.column_names)
    return dataset
