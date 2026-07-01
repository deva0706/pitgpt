"""
validate_dataset.py — quality gate for the instruction dataset.

Garbage in, garbage out: a fine-tune is only as good as its data. This module
loads the raw dataset and:

  1. **Validates JSON** — the file must parse and be a list of objects.
  2. **Checks the schema** — every row needs non-empty ``instruction`` and
     ``output`` string fields (``input`` may be empty).
  3. **Removes malformed entries** — rows missing fields or with empty content.
  4. **Detects duplicates** — exact-duplicate (instruction, output) pairs.
  5. **Prints statistics** — counts, unique instructions/outputs, length
     distributions, and a per-category breakdown.

It writes the cleaned dataset to ``data/processed/`` and never mutates the raw
file, keeping ``data/raw`` immutable (a core data-hygiene principle).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from pitgpt.config import settings

REQUIRED_FIELDS = ("instruction", "input", "output")


@dataclass
class ValidationReport:
    """Structured result of a validation run."""

    total_loaded: int = 0
    malformed_removed: int = 0
    duplicates_removed: int = 0
    kept: int = 0
    unique_instructions: int = 0
    unique_outputs: int = 0
    instruction_len_mean: float = 0.0
    output_len_mean: float = 0.0
    output_len_min: int = 0
    output_len_max: int = 0

    def as_text(self) -> str:
        return "\n".join(
            [
                "=== PitGPT dataset validation ===",
                f"Loaded rows          : {self.total_loaded}",
                f"Malformed removed    : {self.malformed_removed}",
                f"Duplicates removed   : {self.duplicates_removed}",
                f"Kept (clean)         : {self.kept}",
                f"Unique instructions  : {self.unique_instructions}",
                f"Unique outputs       : {self.unique_outputs}",
                f"Instr. words (mean)  : {self.instruction_len_mean:.1f}",
                f"Output words (mean)  : {self.output_len_mean:.1f} "
                f"(min {self.output_len_min}, max {self.output_len_max})",
            ]
        )


def _is_wellformed(row: object) -> bool:
    """A row is valid if it's a dict with non-empty instruction & output strings."""
    if not isinstance(row, dict):
        return False
    for field in REQUIRED_FIELDS:
        if field not in row or not isinstance(row[field], str):
            return False
    return bool(row["instruction"].strip()) and bool(row["output"].strip())


def load_raw(path: Path | None = None) -> list:
    """Load and JSON-parse the raw dataset. Raises on invalid JSON/shape."""
    path = path or settings.paths.dataset_raw
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)  # raises JSONDecodeError on malformed JSON
    if not isinstance(data, list):
        raise ValueError(f"Dataset root must be a JSON list, got {type(data).__name__}.")
    return data


def validate(
    data: list, *, write_clean: bool = True
) -> tuple[list[dict[str, str]], ValidationReport]:
    """Filter malformed/duplicate rows and compute statistics."""
    report = ValidationReport(total_loaded=len(data))

    # 1 & 2: drop malformed rows.
    wellformed = [row for row in data if _is_wellformed(row)]
    report.malformed_removed = len(data) - len(wellformed)

    # 3: drop exact duplicate (instruction, output) pairs, keeping first seen.
    seen: set[tuple[str, str]] = set()
    clean: list[dict[str, str]] = []
    for row in wellformed:
        key = (row["instruction"].strip(), row["output"].strip())
        if key in seen:
            continue
        seen.add(key)
        clean.append({k: row[k] for k in REQUIRED_FIELDS})
    report.duplicates_removed = len(wellformed) - len(clean)
    report.kept = len(clean)

    # 4: statistics.
    instr_lens = [len(r["instruction"].split()) for r in clean]
    out_lens = [len(r["output"].split()) for r in clean]
    report.unique_instructions = len({r["instruction"].strip().lower() for r in clean})
    report.unique_outputs = len({r["output"].strip().lower() for r in clean})
    if clean:
        report.instruction_len_mean = statistics.mean(instr_lens)
        report.output_len_mean = statistics.mean(out_lens)
        report.output_len_min = min(out_lens)
        report.output_len_max = max(out_lens)

    if write_clean:
        settings.paths.ensure()
        out_path = settings.paths.dataset_clean
        out_path.write_text(
            json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return clean, report


def main() -> None:
    data = load_raw()
    clean, report = validate(data, write_clean=True)
    print(report.as_text())
    print(f"\nClean dataset written to: {settings.paths.dataset_clean}")


if __name__ == "__main__":
    main()
