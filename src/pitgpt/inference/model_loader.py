"""
model_loader.py — turn a model id (+ optional adapter) into a ready tokenizer/model pair.

This is the one place in the codebase that knows *how* to instantiate SmolLM2.
Both the baseline script, the evaluation harness, and the Gradio app call into
here so that quantization and adapter-loading behave identically everywhere.

Why a dedicated loader?
    Loading an LLM correctly involves several easy-to-get-wrong steps: choosing a
    dtype, deciding whether to quantize, wiring up ``device_map``, fixing the pad
    token, and (optionally) attaching a LoRA adapter. Centralizing it prevents
    subtle drift between "how we trained" and "how we serve".
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from pitgpt.config import settings

logger = logging.getLogger(__name__)

# Map the string dtype in config to a real torch dtype.
_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def build_quantization_config() -> BitsAndBytesConfig:
    """Construct the 4-bit NF4 config that makes SmolLM2 fit in ~1 GB of VRAM.

    See docs/phase3_notes.md for the theory. In one line: store frozen weights
    as 4-bit NormalFloat, but upcast to fp16 for the actual matrix multiplies.
    """
    q = settings.quant
    return BitsAndBytesConfig(
        load_in_4bit=q.load_in_4bit,
        bnb_4bit_quant_type=q.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=_DTYPES[q.bnb_4bit_compute_dtype],
    )


def load_tokenizer(model_id: str | None = None) -> AutoTokenizer:
    """Load the tokenizer and guarantee a pad token exists.

    Causal LMs are often trained without a dedicated pad token. Batched
    generation and the data collator both need one, so we fall back to the EOS
    token when none is defined — a standard, safe convention.
    """
    model_id = model_id or settings.model.base_model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer had no pad token; set pad_token = eos_token.")
    return tokenizer


def load_model(
    model_id: str | None = None,
    *,
    quantized: bool = True,
    adapter_dir: str | Path | None = None,
) -> AutoModelForCausalLM:
    """Load SmolLM2, optionally 4-bit-quantized and optionally with a LoRA adapter.

    Parameters
    ----------
    model_id:
        HF hub id or local path. Defaults to the configured base model.
    quantized:
        If ``True`` and CUDA is available, load in 4-bit NF4 (QLoRA-style). On a
        CPU-only machine we silently fall back to a normal fp32 load so the code
        still runs (just slower) — useful for tests/CI.
    adapter_dir:
        If given, attach the trained PEFT adapter on top of the base model.
    """
    model_id = model_id or settings.model.base_model_id
    use_cuda = torch.cuda.is_available()

    load_kwargs: dict = {}
    if quantized and use_cuda:
        load_kwargs["quantization_config"] = build_quantization_config()
        load_kwargs["device_map"] = "auto"
        logger.info("Loading %s in 4-bit NF4 (QLoRA mode).", model_id)
    else:
        # Full-precision fallback: fp16 on GPU, fp32 on CPU.
        load_kwargs["torch_dtype"] = torch.float16 if use_cuda else torch.float32
        if use_cuda:
            load_kwargs["device_map"] = "auto"
        logger.info(
            "Loading %s without quantization (cuda=%s).", model_id, use_cuda
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    if adapter_dir is not None:
        # Imported lazily so that plain inference does not require peft installed.
        from peft import PeftModel

        adapter_dir = Path(adapter_dir)
        logger.info("Attaching LoRA adapter from %s", adapter_dir)
        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.eval()
    return model


def load_for_inference(
    *, adapter_dir: str | Path | None = None, quantized: bool = True
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Convenience: return a ``(model, tokenizer)`` pair ready to generate.

    ``adapter_dir=None`` gives you the raw base model (Phase 1 baseline);
    passing the trained adapter path gives you the fine-tuned PitGPT.
    """
    tokenizer = load_tokenizer()
    model = load_model(quantized=quantized, adapter_dir=adapter_dir)
    return model, tokenizer
