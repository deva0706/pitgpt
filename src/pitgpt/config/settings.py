"""
settings.py — the single source of truth for PitGPT.

Every path, model id, and hyperparameter the project uses lives here so that no
other module hard-codes a magic string. If you want to point at a different base
model, move the adapter folder, or change the LoRA rank, you change it *once*,
here, and the whole pipeline follows.

Design notes
------------
* We use frozen ``dataclass`` objects instead of a loose dict so that an invalid
  attribute name fails loudly (``AttributeError``) instead of silently returning
  ``None``.
* Paths are computed relative to the repository root (three parents up from this
  file), so the project works no matter what directory you launch it from.
* Anything a user might reasonably want to override at runtime is read from an
  environment variable with a sensible default — this is the "12-factor app"
  configuration pattern and keeps secrets/machine-specific values out of code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# __file__ = .../src/pitgpt/config/settings.py
#   parents[0] = config, [1] = pitgpt, [2] = src, [3] = repo root
REPO_ROOT: Path = Path(__file__).resolve().parents[3]


def _env_path(var: str, default: Path) -> Path:
    """Return a Path from an env var, falling back to ``default``."""
    value = os.environ.get(var)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. Directories are created lazily via :meth:`ensure`."""

    root: Path = REPO_ROOT
    data_raw: Path = REPO_ROOT / "data" / "raw"
    data_processed: Path = REPO_ROOT / "data" / "processed"
    adapters: Path = REPO_ROOT / "models" / "adapters"
    evaluation: Path = REPO_ROOT / "src" / "pitgpt" / "evaluation"
    configs: Path = REPO_ROOT / "configs"

    # The canonical dataset artifacts (Alpaca-format JSON).
    @property
    def dataset_raw(self) -> Path:
        return self.data_raw / "pitgpt_instructions.json"

    @property
    def dataset_clean(self) -> Path:
        return self.data_processed / "pitgpt_instructions.clean.json"

    def ensure(self) -> None:
        """Create every writable directory if it does not yet exist."""
        for p in (self.data_raw, self.data_processed, self.adapters, self.configs):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelConfig:
    """Which base model we fine-tune, and where the trained adapter lands."""

    # SmolLM2-360M-Instruct: 360M params, already instruction-tuned, ~720 MB.
    # Gemma-3-1B (gated; requires a one-time `huggingface-cli login`). Larger than
    # SmolLM2-360M, so it holds F1 facts noticeably better while still fitting a
    # 4 GB card in 4-bit. Override with the PITGPT_BASE_MODEL env var, e.g.
    # `HuggingFaceTB/SmolLM2-360M-Instruct` for the lighter, no-login model.
    base_model_id: str = os.environ.get(
        "PITGPT_BASE_MODEL", "google/gemma-3-1b-it"
    )
    # Directory holding the trained LoRA adapter (adapter_config.json + weights).
    adapter_dir: Path = REPO_ROOT / "models" / "adapters" / "pitgpt-lora"

    # The persona injected as the system prompt for every conversation.
    system_prompt: str = (
        "You are PitGPT, a lifelong Formula 1 enthusiast — the knowledgeable friend "
        "who has watched F1 for years and loves talking about it. You are NOT a race "
        "engineer, a commentator, or a Wikipedia entry. You're warm, enthusiastic, "
        "passionate and curious, and you make F1 feel exciting and welcoming, "
        "especially to beginners. Explain technical ideas clearly, share the history "
        "and the great rivalries, and enjoy a good strategy debate. Use F1 community "
        "language naturally where it fits — 'Lights out!', 'mega lap', 'purple sector', "
        "'wheel-to-wheel', 'dirty air', 'undercut', 'what a drive', 'absolute cinema' — "
        "but never force it. Write like a real conversation, not documentation: open "
        "with a hook, explain clearly, bring in a historical example when it helps, add "
        "a sharp technical insight, and close in a friendly way. Never sound robotic."
    )


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding defaults for ``model.generate()`` — tuned for factual, on-topic replies."""

    max_new_tokens: int = 256
    # A 360M model on a focused dataset stays far more factually faithful at low
    # temperature — higher temps make it hallucinate fluent-but-wrong details.
    # 0.35 keeps a little variety while reproducing the learned answers reliably.
    temperature: float = 0.35
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.15
    do_sample: bool = True


@dataclass(frozen=True)
class QuantConfig:
    """4-bit NF4 settings — the 'Q' in QLoRA (see docs/phase3_notes.md)."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"          # NormalFloat4: optimal for ~normal weights
    bnb_4bit_use_double_quant: bool = True    # quantize the quantization constants too
    bnb_4bit_compute_dtype: str = "bfloat16"  # bf16 matmuls on Ampere (RTX 3050); no grad scaler needed


@dataclass(frozen=True)
class LoraConfig:
    """LoRA adapter shape. Small rank keeps the trainable params (and VRAM) tiny."""

    r: int = 16                       # rank of the low-rank update matrices
    lora_alpha: int = 32              # scaling factor (effective LR multiplier = alpha/r)
    lora_dropout: float = 0.05        # regularization on the adapter path
    bias: str = "none"                # do not train biases
    task_type: str = "CAUSAL_LM"
    # Which linear layers get an adapter. For Llama-style blocks (SmolLM2) these
    # are the attention projections + the MLP projections.
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


@dataclass(frozen=True)
class TrainingConfig:
    """SFTTrainer hyperparameters, sized for a 4 GB RTX 3050 Laptop GPU."""

    output_dir: Path = REPO_ROOT / "models" / "adapters" / "pitgpt-lora"
    # On the 2310-pair set with many near-duplicate instruction wrappers, 3 epochs
    # drove the loss to ~0.12 and over-fit, which garbled free-generation on a 360M
    # model. 1 epoch keeps the persona and coverage while staying closer to the
    # base model's coherence — a better trade for this small a model.
    num_train_epochs: int = 2
    # max_steps -1 lets num_train_epochs govern the run length. On the ~760-row
    # enthusiast dataset, 2 epochs ~= 96 steps (~16 min, so it runs across a couple
    # of resumable foreground chunks). Watch for over-fitting on this fast learner.
    max_steps: int = -1
    # Gemma-3-1B on a 4 GB card: batch 2 / seq 512 peaked at ~3.9 GB and the paged
    # optimizer thrashed (near-hang). Batch 1 keeps activation memory low and
    # leaves ~1 GB of headroom so training runs at full GPU. Effective batch = 16.
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16     # effective batch size = 1 * 16 = 16
    learning_rate: float = 2e-4               # typical LoRA LR (higher than full-FT)
    max_seq_length: int = 384                 # our chats are ~270 tokens; 384 trims activation memory
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 25          # frequent checkpoints so long runs can be resumed
    save_total_limit: int = 3
    optim: str = "paged_adamw_8bit"           # bitsandbytes paged optimizer → less VRAM
    gradient_checkpointing: bool = True       # trade compute for memory
    seed: int = 42


@dataclass(frozen=True)
class Settings:
    """Aggregate config object — import ``settings`` from this module."""

    paths: Paths = field(default_factory=Paths)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# The one object the rest of the codebase imports.
settings = Settings()
