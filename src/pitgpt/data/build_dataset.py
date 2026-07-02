"""
build_dataset.py — turn the enthusiast knowledge base into training rows.

We take the hand-authored, style-teaching conversations in ``enthusiast_kb.py``
and expand them into instruction/response rows. Expansion is deliberately light:
each :class:`Example` already ships several *genuinely different* phrasings of its
prompt, and we add a couple of natural conversational wrappers ("Hey, …",
"I've always wondered — …") so the model sees the same idea asked a few ways.

Why not blow this up to tens of thousands of rows?
    Because we're teaching a **style**, and style is low-entropy — a few hundred
    diverse, consistent, high-quality examples teach it well, and quality beats
    quantity (the project spec says exactly this). Over-inflating with repetitive
    paraphrases would just risk over-fitting the same handful of answers.

Output is the model's native chat content rendered later by ``data_module``; here
we store the neutral ``{instruction, input, output}`` shape the validator and
trainer already understand.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pitgpt.config import settings
from pitgpt.data.enthusiast_kb import EXAMPLES

# Natural, friendly ways a person might open a question to a knowledgeable mate.
# The first (plain) form is always included; the rest add light variety.
WRAPPERS: tuple[str, ...] = (
    "{q}",
    "Hey, {q_lower}",
    "I've always wondered — {q_lower}",
    "Quick one: {q_lower}",
    "Been meaning to ask, {q_lower}",
    "So {q_lower}",
)


def _lower_first(text: str) -> str:
    return text[0].lower() + text[1:] if text else text


def _make_row(instruction: str, output: str) -> dict[str, str]:
    return {"instruction": instruction.strip(), "input": "", "output": output.strip()}


def build_examples(seed: int = 42) -> list[dict[str, str]]:
    """Expand the enthusiast examples into de-duplicated instruction rows."""
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    # Rotate which wrapper each prompt gets so the variety is spread evenly.
    wrapper_cursor = 0
    for example in EXAMPLES:
        for base_prompt in example.prompts:
            q_lower = _lower_first(base_prompt)
            # Always include the plain form, plus one rotating wrapper variant.
            variants = ["{q}", WRAPPERS[1 + (wrapper_cursor % (len(WRAPPERS) - 1))]]
            wrapper_cursor += 1
            for wrapper in variants:
                instruction = wrapper.format(q=base_prompt, q_lower=q_lower)
                key = instruction.lower().strip()
                if key in seen:
                    continue
                seen.add(key)
                rows.append(_make_row(instruction, example.response))

    rng.shuffle(rows)
    return rows


def write_dataset(rows: list[dict[str, str]], path: Path | None = None) -> Path:
    settings.paths.ensure()
    path = path or settings.paths.dataset_raw
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    rows = build_examples()
    path = write_dataset(rows)
    n_examples = len(EXAMPLES)
    n_unique_outputs = len({e.response for e in EXAMPLES})
    print(
        f"Built {len(rows)} instruction rows from {n_examples} distinct "
        f"enthusiast conversations ({n_unique_outputs} unique responses)."
    )
    print(f"Written to: {path}")


if __name__ == "__main__":
    main()
