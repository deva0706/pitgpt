"""
baseline.py — Phase 1 deliverable: run the *un-tuned* base model on the eval set.

Why capture a baseline at all?
    Fine-tuning is only meaningful relative to a starting point. By recording
    exactly how the raw base model answers our 20 F1 questions *before* any
    training, we get an honest before/after comparison in Phase 5 and can prove
    the adapter actually changed the model's behaviour (not just our vibes).

Output: a Markdown report at ``src/pitgpt/evaluation/baseline_outputs.md``.
Run with:  ``python -m pitgpt.inference.baseline``  (or ``pitgpt-baseline``).
"""

from __future__ import annotations

import logging
import time

import torch

from pitgpt.config import settings
from pitgpt.evaluation.prompts import EVAL_PROMPTS
from pitgpt.inference.engine import generate_reply
from pitgpt.inference.model_loader import load_for_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pitgpt.baseline")


def _vram_report() -> str:
    """Return a one-line summary of GPU memory in use, if any."""
    if not torch.cuda.is_available():
        return "CPU only (no CUDA)."
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    name = torch.cuda.get_device_name(0)
    return f"{name}: {allocated:.2f} GB allocated / {reserved:.2f} GB reserved."


def run_baseline() -> str:
    """Generate baseline answers and write the Markdown report. Returns the path."""
    logger.info("Loading BASE model (no adapter) for baseline capture ...")
    model, tokenizer = load_for_inference(adapter_dir=None, quantized=True)
    logger.info("Model loaded. %s", _vram_report())

    lines: list[str] = [
        "# PitGPT — Baseline Outputs (Phase 1)",
        "",
        "Raw **Gemma-3-1B-it** with the PitGPT race-engineer system prompt,",
        "**before any fine-tuning**. This is the reference point for the Phase 5",
        "before/after evaluation.",
        "",
        f"- Base model: `{settings.model.base_model_id}`",
        f"- Decoding: temp={settings.generation.temperature}, top_p={settings.generation.top_p}, "
        f"max_new_tokens={settings.generation.max_new_tokens}",
        f"- Device: {_vram_report()}",
        "",
        "---",
        "",
    ]

    start = time.time()
    for i, prompt in enumerate(EVAL_PROMPTS, 1):
        logger.info("[%2d/%d] %s", i, len(EVAL_PROMPTS), prompt)
        reply = generate_reply(model, tokenizer, prompt)
        lines.extend(
            [
                f"## {i}. {prompt}",
                "",
                "**Base model reply:**",
                "",
                reply if reply else "_(empty response)_",
                "",
                "---",
                "",
            ]
        )
    elapsed = time.time() - start
    lines.append(f"_Generated {len(EVAL_PROMPTS)} baseline replies in {elapsed:.1f}s._")

    out_path = settings.paths.evaluation / "baseline_outputs.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Baseline written to %s (%.1fs).", out_path, elapsed)
    return str(out_path)


def main() -> None:
    path = run_baseline()
    print(f"Baseline outputs saved to: {path}")


if __name__ == "__main__":
    main()
