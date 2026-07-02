"""
build_facts.py — assemble the closed-book training set (RAG disabled era).

Real-fact QA rows generated from the Kaggle historical dataset (fact_dataset.py)
mixed with the hand-authored persona rows, so the model learns *actual F1 facts*
in its weights while keeping the enthusiast voice. No retrieved-context rows —
RAG is currently commented out of the runtime.
"""

from __future__ import annotations

from pitgpt.config import settings
from pitgpt.data.build_dataset import build_examples as build_persona_rows
from pitgpt.data.build_dataset import write_dataset
from pitgpt.data.fact_dataset import build_fact_rows


def build_combined(seed: int = 42) -> list[dict[str, str]]:
    facts = build_fact_rows(seed=seed)
    persona = build_persona_rows(seed=seed)
    for row in persona:
        row.setdefault("context", "")
    return facts + persona


def main() -> None:
    rows = build_combined()
    path = write_dataset(rows, settings.paths.dataset_raw)
    n_facts = sum(1 for r in rows if not r.get("context") and r not in ())
    print(f"Built {len(rows)} rows "
          f"({len(build_fact_rows())} real-fact QA + persona).")
    print(f"Written to: {path}")


if __name__ == "__main__":
    main()
