"""
build_final.py — assemble the definitive behavior-only training set.

Grounded-signature rows + honesty rows (final_dataset.py) mixed with the
hand-authored persona conversations. Zero fact-memorization rows: facts live in
the retrieval corpus, behavior lives here.
"""

from __future__ import annotations

from pitgpt.config import settings
from pitgpt.data.build_dataset import build_examples as build_persona_rows
from pitgpt.data.build_dataset import write_dataset
from pitgpt.data.final_dataset import build_final_rows


def build_combined(seed: int = 42) -> list[dict[str, str]]:
    behavior = build_final_rows(seed=seed)
    persona = build_persona_rows(seed=seed)
    for row in persona:
        row.setdefault("context", "")
    return behavior + persona


def main() -> None:
    rows = build_combined()
    path = write_dataset(rows, settings.paths.dataset_raw)
    n_grounded = sum(1 for r in rows if r.get("context"))
    print(f"Built {len(rows)} rows: {n_grounded} grounded-behavior + "
          f"{len(rows) - n_grounded} persona.")
    print(f"Written to: {path}")


if __name__ == "__main__":
    main()
