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

import logging
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
        yield from stream_reply(self._model, self._tokenizer, message, prior)


# Module-level singleton the UI imports.
service = RaceEngineerService()
