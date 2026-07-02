"""
train.py — QLoRA fine-tuning of SmolLM2 into PitGPT.

Pipeline
--------
1. Load the tokenizer and the base model **in 4-bit NF4** (the quantized, frozen
   base — the "Q" in QLoRA).
2. ``prepare_model_for_kbit_training`` — makes a k-bit model trainable: enables
   gradient checkpointing hooks, casts layernorms to fp32 for stability, and
   ensures inputs require grad so gradients flow into the adapters.
3. Attach **LoRA adapters** (via ``SFTTrainer(peft_config=...)``) so only the
   low-rank matrices train.
4. Run ``SFTTrainer`` with a memory-frugal config sized for 4 GB VRAM.
5. Save **only the adapter** to ``models/adapters/pitgpt-lora``.

Every hyperparameter is defined in ``config/settings.py`` and explained in
``docs/phase3_notes.md`` / ``docs/phase4_notes.md``.

Run:  ``python -m pitgpt.training.train``
"""

from __future__ import annotations

import logging

import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from pitgpt.config import settings
from pitgpt.inference.model_loader import load_model, load_tokenizer
from pitgpt.training.data_module import build_sft_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pitgpt.train")


def _log_vram(tag: str) -> None:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info(
            "[VRAM %-14s] allocated=%.2f GB reserved=%.2f GB peak=%.2f GB",
            tag, alloc, reserved, peak,
        )


def build_lora_config() -> LoraConfig:
    """Translate our LoRA settings dataclass into a PEFT ``LoraConfig``."""
    c = settings.lora
    return LoraConfig(
        r=c.r,
        lora_alpha=c.lora_alpha,
        lora_dropout=c.lora_dropout,
        bias=c.bias,
        task_type=c.task_type,
        target_modules=list(c.target_modules),
    )


def build_sft_config() -> SFTConfig:
    """Translate our training settings into a TRL ``SFTConfig`` (memory-tuned)."""
    t = settings.training
    return SFTConfig(
        output_dir=str(t.output_dir),
        num_train_epochs=t.num_train_epochs,
        max_steps=t.max_steps,           # -1 defers to num_train_epochs
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler_type,
        logging_steps=t.logging_steps,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        optim=t.optim,
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=t.seed,
        # SFT-specific:
        dataset_text_field="text",
        max_length=settings.training.max_seq_length,
        packing=False,               # our sequences are short and independent
        report_to="none",            # no wandb/tensorboard in this offline project
        # Ampere (RTX 3050) supports bf16, which — unlike fp16 — needs no gradient
        # scaler and won't overflow, so it plays nicely with 4-bit + LoRA grads.
        bf16=torch.cuda.is_available(),
        fp16=False,
    )


def train() -> str:
    """Run the full fine-tune. Returns the adapter output directory."""
    logger.info("=== PitGPT QLoRA training ===")
    tokenizer = load_tokenizer()

    logger.info("Loading base model in 4-bit NF4 (frozen) ...")
    model = load_model(quantized=True, adapter_dir=None)
    _log_vram("base loaded")

    # k-bit models need this before training: enables input grads + gradient
    # checkpointing compatibility and upcasts norms for numerical stability.
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=settings.training.gradient_checkpointing
    )
    model.config.use_cache = False   # incompatible with gradient checkpointing

    logger.info("Rendering dataset through the chat template ...")
    train_dataset = build_sft_dataset(tokenizer)
    logger.info("Training examples: %d", len(train_dataset))

    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(),
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(),
    )

    # Report the (tiny) trainable parameter count — the whole point of LoRA.
    trainable, total = _count_trainable(trainer.model)
    logger.info(
        "Trainable params: %s / %s (%.3f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    # Resume support: set PITGPT_RESUME=auto to continue from the most recent
    # checkpoint in output_dir (useful when a long run is interrupted), or give an
    # explicit checkpoint path. Larger base models (e.g. Gemma) can exceed a
    # background task's lifetime, so being able to resume matters.
    resume = _resolve_resume(settings.training.output_dir)
    _log_vram("before train")
    trainer.train(resume_from_checkpoint=resume)
    _log_vram("after train")

    out_dir = str(settings.training.output_dir)
    trainer.save_model(out_dir)          # saves ONLY the adapter (PEFT model)
    tokenizer.save_pretrained(out_dir)   # keep tokenizer with the adapter
    logger.info("Adapter saved to %s", out_dir)
    _log_vram("final")
    return out_dir


def _resolve_resume(output_dir):
    """Return a checkpoint to resume from, or None.

    Controlled by the PITGPT_RESUME env var: ``auto`` picks the highest-numbered
    ``checkpoint-*`` in ``output_dir``; any other value is treated as an explicit
    path; unset means start fresh.
    """
    import os
    import re
    from pathlib import Path

    val = os.environ.get("PITGPT_RESUME")
    if not val:
        return None
    if val != "auto":
        return val
    out = Path(output_dir)
    checkpoints = list(out.glob("checkpoint-*")) if out.exists() else []
    if not checkpoints:
        logger.info("PITGPT_RESUME=auto but no checkpoint found; starting fresh.")
        return None
    latest = max(checkpoints, key=lambda p: int(re.findall(r"\d+", p.name)[-1]))
    logger.info("Resuming from %s", latest)
    return str(latest)


def _count_trainable(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main() -> None:
    path = train()
    print(f"Training complete. Adapter at: {path}")


if __name__ == "__main__":
    main()
