"""
build_dataset.py — turn the curated knowledge base into ~1000 Alpaca pairs.

Alpaca format
-------------
Each training example is a JSON object:

    {"instruction": "<what the user asks>",
     "input": "",                # optional extra context; empty for us
     "output": "<the engineer's answer>"}

Why generate rather than hand-write 1000 rows?
    We hand-write the *knowledge* (in ``knowledge_base.py``) so every answer is
    accurate and in-voice, then expand it into many examples by:
      1. pairing each concept with its several instruction phrasings, and
      2. adding lightly-varied instruction wrappers (e.g. "As my race engineer,
         …") so the model sees the same idea asked many ways.
    This teaches *robust instruction-following* for the domain without inventing
    unverified facts — the outputs are always one of our vetted answers.

The result is deterministic (a fixed seed) so the dataset is reproducible.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pitgpt.config import settings
from pitgpt.data.knowledge_base import CONCEPTS, RADIO

# Instruction "wrappers" — natural ways a driver/engineer might frame a question.
# ``{q}`` is the concept's base instruction. We keep the first entry as the bare
# question so plain phrasings dominate, then add in-character variants.
WRAPPERS: tuple[str, ...] = (
    "{q}",
    "{q}",  # weighted twice so the plain form stays most common
    "As my race engineer, {q_lower}",
    "Engineer, {q_lower}",
    "Quick question from the cockpit: {q_lower}",
    "Talk me through this: {q_lower}",
    "For a rookie driver, {q_lower}",
    "In simple terms, {q_lower}",
    "Break it down for me — {q_lower}",
    "I'm new to F1 strategy. {q}",
)


def _lower_first(text: str) -> str:
    """Lowercase only the first character (for mid-sentence wrappers)."""
    return text[0].lower() + text[1:] if text else text


def _make_example(instruction: str, output: str) -> dict[str, str]:
    return {"instruction": instruction.strip(), "input": "", "output": output.strip()}


def build_examples(seed: int = 42) -> list[dict[str, str]]:
    """Expand the knowledge base into a de-duplicated list of Alpaca examples."""
    rng = random.Random(seed)
    examples: list[dict[str, str]] = []
    seen_instructions: set[str] = set()

    for concept in CONCEPTS:
        for base_instruction in concept.instructions:
            q_lower = _lower_first(base_instruction)
            for wrapper in WRAPPERS:
                instruction = wrapper.format(q=base_instruction, q_lower=q_lower)
                key = instruction.lower().strip()
                if key in seen_instructions:
                    continue
                seen_instructions.add(key)
                examples.append(_make_example(instruction, concept.response))

    # Radio-style short pairs, each with a couple of phrasings.
    for prompt, reply in RADIO:
        for wrapper in ("{q}", "Engineer: {q_lower}", "Give me the call — {q_lower}"):
            q_lower = _lower_first(prompt)
            instruction = wrapper.format(q=prompt, q_lower=q_lower)
            key = instruction.lower().strip()
            if key in seen_instructions:
                continue
            seen_instructions.add(key)
            examples.append(_make_example(instruction, reply))

    rng.shuffle(examples)
    return examples


def write_dataset(examples: list[dict[str, str]], path: Path | None = None) -> Path:
    """Write the examples to the raw dataset JSON file."""
    settings.paths.ensure()
    path = path or settings.paths.dataset_raw
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    examples = build_examples()
    path = write_dataset(examples)
    n_concepts = len(CONCEPTS)
    print(f"Built {len(examples)} instruction/response pairs "
          f"from {n_concepts} concepts + {len(RADIO)} radio prompts.")
    print(f"Written to: {path}")


if __name__ == "__main__":
    main()
