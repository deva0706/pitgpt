"""
build_grounded.py — assemble the grounding-first training set.

Combines two lessons into one dataset:
  * **Grounded rows** (with retrieved ``context``) that teach the model to answer
    FROM evidence — the fix for the "ignores retrieval" problem.
  * **Persona rows** (no context) that keep the enthusiast voice for opinion/vibe
    questions where there's nothing to retrieve.

The mix teaches the exact production behaviour: *when context is present, ground on
it; when it isn't, be a warm F1 enthusiast.* Grounded rows dominate so grounding
is the stronger signal.

Output goes to ``data/raw`` as ``{context, instruction, input, output}`` rows and
is then cleaned by ``validate_dataset``.
"""

from __future__ import annotations

from pitgpt.config import settings
from pitgpt.data.build_dataset import build_examples as build_persona_rows
from pitgpt.data.build_dataset import write_dataset
from pitgpt.data.grounded_dataset import build_grounded_rows


def build_combined(seed: int = 42) -> list[dict[str, str]]:
    """Grounded rows (with context) + persona rows (empty context)."""
    grounded = build_grounded_rows(seed=seed)
    persona = build_persona_rows(seed=seed)
    for row in persona:
        row.setdefault("context", "")   # persona turns have no retrieved context
    combined = grounded + persona
    return combined


def main() -> None:
    rows = build_combined()
    path = write_dataset(rows, settings.paths.dataset_raw)
    n_grounded = sum(1 for r in rows if r.get("context"))
    print(
        f"Built {len(rows)} rows: {n_grounded} grounded (with context) + "
        f"{len(rows) - n_grounded} persona (no context)."
    )
    print(f"Written to: {path}")


if __name__ == "__main__":
    main()
