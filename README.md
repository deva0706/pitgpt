# 🏎️ PitGPT — Offline F1 Race Engineer

An **offline Formula 1 Race Engineer chatbot**, fine-tuned from
[`SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
using **QLoRA**. Small enough to run on a laptop, trained to talk like the voice
in a driver's ear: strategy calls, tyre deltas, weather, and race-craft.

> **Status:** 🚧 Project scaffold (Step 1 — setup only).

---

## Why this stack?

| Goal | Choice | Reason |
|------|--------|--------|
| Runs offline / on modest hardware | **SmolLM2-360M** | Tiny (360M params), instruct-tuned, strong for its size |
| Fine-tune cheaply | **QLoRA** | 4-bit base + small LoRA adapters → trainable on consumer GPUs |
| Ship small artifacts | **PEFT adapters** | We save ~10–50 MB adapters, not the whole model |
| Simple UX | **Gradio** | A shareable chat box over `model.generate()` |

---

## Architecture

Data flows left-to-right; the folder layout mirrors the pipeline:

```
data/raw  →  data/processed  →  training (QLoRA)  →  models/adapters  →  inference  →  Gradio app
```

```
PitGPT/
├── data/
│   ├── raw/            # immutable source data (never edited in place)
│   └── processed/      # tokenizer-ready chat data (regenerable from raw/)
├── src/pitgpt/
│   ├── config/         # single source of truth for paths & settings
│   ├── data/           # raw/ → processed/ formatting scripts
│   ├── training/       # QLoRA fine-tuning loop (TRL SFTTrainer)
│   ├── inference/      # load base + adapter, generate replies
│   └── app/            # Gradio chat UI
├── models/adapters/    # trained LoRA adapters
├── configs/            # YAML experiment configs (lr, rank, epochs)
├── notebooks/          # exploration only — never imported by src/
├── scripts/            # thin CLI entry points
├── tests/              # unit tests (prompt formatting, data shape)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell)
# source .venv/bin/activate   # macOS / Linux

# 2. (GPU users) install the CUDA build of torch FIRST — see requirements.txt
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install -r requirements.txt
```

### ⚠️ Windows + bitsandbytes note
`bitsandbytes` (the 4-bit quantization engine) has historically been
Linux-first. Modern versions (≥0.43) ship Windows wheels, but 4-bit **training**
on Windows can be finicky. If it misbehaves, our fallbacks are:
1. Train in **WSL2** or **Google Colab**, then run inference on Windows, or
2. Drop to plain **LoRA** (no 4-bit) for training.

---

## Dependencies at a glance

| Package | Role |
|---------|------|
| `torch` | Tensor/autograd engine underneath everything |
| `transformers` | Loads SmolLM2, tokenizer, chat template, generation |
| `datasets` | Efficient dataset loading & formatting |
| `accelerate` | Device placement + mixed precision (trainer backend) |
| `peft` | LoRA adapters (trains ~1% of params) |
| `trl` | `SFTTrainer` supervised fine-tuning loop |
| `bitsandbytes` | 4-bit NF4 quantization (the "Q" in QLoRA) |
| `gradio` | The Race Engineer chat web UI |

---

## Roadmap

- [x] **Step 1 — Project scaffold** (this step)
- [ ] Step 2 — Build the F1 instruction dataset (`data/`)
- [ ] Step 3 — QLoRA training loop (`training/`)
- [ ] Step 4 — Inference engine (`inference/`)
- [ ] Step 5 — Gradio Race Engineer UI (`app/`)

---

*A learning-focused, production-structured portfolio project.*
