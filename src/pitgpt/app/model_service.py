"""
model_service.py — a lazy, single-instance wrapper around the chat model.

The Gradio UI should not know *how* the model is loaded — it just wants "stream a
reply to this message given this history". This service provides that, and:

* loads the model **once** (lazily, on first message) and reuses it, so we don't
  pay the load cost on every turn;
* automatically attaches the trained LoRA adapter **if it exists on disk**,
  otherwise falls back to the base model (so the app still runs before training
  or if training was skipped);
* exposes a clean ``stream(message, history)`` generator for the UI.

Separating this from the UI keeps the interface modular and testable.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from pitgpt.config import settings
from pitgpt.inference.engine import stream_reply
from pitgpt.inference.model_loader import load_for_inference

logger = logging.getLogger("pitgpt.app.model_service")


class RaceEngineerService:
    """Holds the loaded model/tokenizer and streams race-engineer replies."""

    def __init__(self, adapter_dir: Path | None = None) -> None:
        self._adapter_dir = adapter_dir if adapter_dir is not None else settings.model.adapter_dir
        self._model = None
        self._tokenizer = None

    @property
    def using_adapter(self) -> bool:
        """True if a *trained* adapter is present (not just an empty output dir).

        We check for the actual adapter files, because the output directory can
        exist with only intermediate ``checkpoint-*/`` folders while training is
        still running — that must not count as a ready adapter.
        """
        d = Path(self._adapter_dir)
        return (d / "adapter_config.json").exists() and (
            (d / "adapter_model.safetensors").exists()
            or (d / "adapter_model.bin").exists()
        )

    def _ensure_loaded(self) -> None:
        """Load the model on first use (lazy initialisation)."""
        if self._model is not None:
            return
        adapter = self._adapter_dir if self.using_adapter else None
        if adapter is None:
            logger.warning(
                "Adapter not found at %s — serving the BASE model.", self._adapter_dir
            )
        else:
            logger.info("Loading fine-tuned PitGPT (base + adapter %s).", adapter)
        self._model, self._tokenizer = load_for_inference(adapter_dir=adapter, quantized=True)

    def stream(self, message: str, history: list[dict[str, str]] | None = None):
        """Yield the growing reply string for ``message`` given prior ``history``.

        ``history`` is a list of ``{"role", "content"}`` dicts (Gradio 'messages'
        format), which is exactly what our engine expects as prior turns.
        """
        self._ensure_loaded()
        # Only pass user/assistant turns as history; the system prompt is added
        # by the engine itself.
        prior = [m for m in (history or []) if m.get("role") in ("user", "assistant")]
        trace: dict = {}
        shown, started = "", time.time()
        for shown in stream_reply(self._model, self._tokenizer, message, prior, trace=trace):
            yield shown
        record_turn(message, shown, trace, time.time() - started)


def describe_turn(message: str, shown: str, trace: dict, seconds: float = 0.0) -> dict:
    """One chat turn as a record: what was typed, what retrieval found, what
    the model wrote, what the check made of it, and what was shown."""
    plan = trace.get("plan")
    if trace.get("reply") is not None:
        path = "decided without the model"
    elif plan is not None:
        path = "grounded" + (", trained format" if plan.strict else "")
    else:
        path = "small talk (no facts)"
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "typed": message,
        "path": path,
        "kind": plan.kind if plan else None,
        "facts": [f"[{f.get('source', '?')}] {f.get('text', '')}" for f in plan.facts]
                 if plan else [],
        "asked_model": trace.get("asked"),
        "model_wrote": trace.get("raw"),
        "check": trace.get("verdict"),
        "shown": shown,
        "seconds": round(seconds, 1),
    }


def record_turn(message: str, shown: str, trace: dict, seconds: float = 0.0) -> None:
    """Write the turn to the terminal and to ``logs/chat_trace.jsonl``.

    Without this the backend said nothing about a chat: a wrong answer in the
    browser could not be traced to retrieval, to the model, or to the check.
    The file stays on this machine (``logs/`` is git-ignored).
    """
    turn = describe_turn(message, shown, trace, seconds)
    lines = [f"TURN  {turn['typed']!r}  ->  {turn['path']}"
             + (f" [{turn['kind']}]" if turn["kind"] else "")]
    lines += [f"  fact         {fact}" for fact in turn["facts"]]
    if turn["asked_model"] and turn["asked_model"] != turn["typed"]:
        lines.append(f"  asked model  {turn['asked_model']}")
    if turn["model_wrote"] is not None:
        lines.append(f"  model wrote  {turn['model_wrote']}")
    if turn["check"]:
        lines.append(f"  check        {turn['check']}")
    lines.append(f"  shown        {turn['shown']}   ({turn['seconds']}s)")
    logger.info("\n".join(lines))
    try:
        path = settings.paths.root / "logs" / "chat_trace.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")
    except OSError as exc:           # a read-only folder must not break the chat
        logger.warning("Could not write the chat trace: %s", exc)


# Module-level singleton the UI imports.
service = RaceEngineerService()
