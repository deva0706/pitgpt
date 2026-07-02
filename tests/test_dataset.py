"""Tests for the dataset build + validation pipeline."""

from __future__ import annotations

from pitgpt.data.build_dataset import build_examples
from pitgpt.data.validate_dataset import validate


def test_build_produces_enough_examples():
    examples = build_examples(seed=42)
    # Style dataset: quality/diversity over raw count (see spec). Expect a few hundred.
    assert len(examples) >= 600, f"expected several hundred examples, got {len(examples)}"


def test_dataset_has_diverse_outputs():
    """A style fine-tune needs many *distinct* responses, not paraphrase spam."""
    outputs = {r["output"] for r in build_examples(seed=42)}
    assert len(outputs) >= 100, f"expected 100+ unique responses, got {len(outputs)}"


def test_every_example_is_wellformed():
    for row in build_examples(seed=42):
        assert set(row) == {"instruction", "input", "output"}
        assert row["instruction"].strip()
        assert row["output"].strip()
        assert isinstance(row["input"], str)


def test_build_is_deterministic():
    a = build_examples(seed=42)
    b = build_examples(seed=42)
    assert a == b, "same seed must produce identical datasets"


def test_no_duplicate_instructions():
    examples = build_examples(seed=42)
    instructions = [r["instruction"].strip().lower() for r in examples]
    assert len(instructions) == len(set(instructions)), "instructions must be unique"


def test_validate_flags_malformed_and_duplicates():
    dirty = [
        {"instruction": "Explain DRS.", "input": "", "output": "The Drag Reduction System..."},
        {"instruction": "Explain DRS.", "input": "", "output": "The Drag Reduction System..."},  # dup
        {"instruction": "", "input": "", "output": "no instruction"},                             # malformed
        {"instruction": "no output", "input": "", "output": "   "},                               # malformed
        {"instruction": "wrong types", "input": "", "output": 123},                               # malformed
    ]
    clean, report = validate(dirty, write_clean=False)
    assert report.kept == 1
    assert report.malformed_removed == 3
    assert report.duplicates_removed == 1
