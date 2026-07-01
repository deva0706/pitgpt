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
    base_model_id: str = os.environ.get(
        "PITGPT_BASE_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct"
    )
    # Directory holding the trained LoRA adapter (adapter_config.json + weights).
    adapter_dir: Path = REPO_ROOT / "models" / "adapters" / "pitgpt-lora"

    # The persona injected as the system prompt for every conversation.
    system_prompt: str = (
        "You are PitGPT, a Formula 1 race engineer. You speak on the pit wall to "
        "your driver and engineers. Explain race strategy, tyre choices, pit "
        "windows, qualifying, telemetry, regulations, DRS, ERS and weather "
        "clearly and technically. Be precise and educational. Use race-engineer "
        "phrasing such as 'Copy that.' naturally, but do not force it into every "
        "reply."
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
    num_train_epochs: int = 3
    # Smoke test showed ~0.7 GB peak at batch 1, so a 4 GB card has ample room.
    per_device_train_batch_size: int = 4      # fits comfortably; speeds up the run
    gradient_accumulation_steps: int = 4      # effective batch size = 4 * 4 = 16
    learning_rate: float = 2e-4               # typical LoRA LR (higher than full-FT)
    max_seq_length: int = 512                 # our chats are ~270 tokens; 512 is safe headroom
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 50
    save_total_limit: int = 2
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
