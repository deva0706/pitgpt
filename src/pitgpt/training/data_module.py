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


def _to_messages(row: dict) -> list[dict[str, str]]:
    """Turn one Alpaca row into a system/user/assistant message list."""
    user_content = row["instruction"]
    if row.get("input"):
        user_content = f"{user_content}\n\n{row['input']}"
    return [
        {"role": "system", "content": settings.model.system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": row["output"]},
    ]


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
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list(rows)
    dataset = dataset.map(render, remove_columns=dataset.column_names)
    return dataset
