# Phase 1 — Environment Setup & Baseline (Teaching Notes)

These notes explain *why* each piece of Phase 1 exists before the code that
implements it, as required by the project's coding rules.

## 1. Why an isolated environment matters

We build inside a dedicated conda environment (`pitgpt`, Python 3.11) rather
than the global interpreter. Deep-learning stacks pin exact versions of CUDA,
`torch`, `transformers`, and `bitsandbytes` that routinely conflict with other
projects. Isolation means:

- Reproducibility: the same versions produce the same behaviour tomorrow.
- Safety: upgrading `torch` here can't break an unrelated project.
- Honesty: `requirements.txt` describes a real, working environment.

On this machine the environment already exists with a **CUDA-enabled** build:
`torch 2.5.1+cu121`, `transformers 5.12.1`, `peft 0.19.1`, `trl 1.7.0`,
`bitsandbytes 0.49.2`, `accelerate 1.14.0`, `gradio 6.19.0`.

### One real bug we fixed
The env's `SSL_CERT_FILE` pointed at a non-existent `…/pitgpt/ssl/cacert.pem`,
which crashed **any** HTTPS call (gradio import, HF downloads). Fix: copy
`certifi`'s bundled `cacert.pem` to that path. Lesson: a broken cert path fails
loudly at import time — read the traceback, it names the missing file.

## 2. Why verify CUDA/GPU explicitly (`scripts/check_gpu.py`)

`nvidia-smi` proves the *driver* works; it does **not** prove that *your Python*
can compute on the GPU. A CPU-only `torch` wheel will happily report
`torch.cuda.is_available() == False` even with a perfect driver. So the gate:

1. imports `torch`,
2. checks `torch.cuda.is_available()`,
3. actually runs a matrix multiply on the GPU (`x @ x`),
4. imports `bitsandbytes` (the 4-bit engine QLoRA needs).

Result on this machine: **RTX 3050 Laptop GPU, 4.00 GB VRAM, compute 8.6** — the
4 GB ceiling is the single most important constraint shaping every later choice.

## 3. Why SmolLM2-360M-Instruct

- **360M parameters** → the 4-bit base fits in roughly ~0.8–1 GB of VRAM,
  leaving headroom on a 4 GB card for activations, gradients, and the optimizer.
- **-Instruct** → already trained to follow a chat template, so even the baseline
  produces conversational answers; fine-tuning shapes *style and domain*, not
  basic instruction-following from scratch.
- **Apache-2.0 / open** → legal to fine-tune and redistribute adapters.

## 4. Why a config module first (`config/settings.py`)

Before touching the model we created a single source of truth for paths, the
model id, decoding parameters, and (later) quantization/LoRA/training settings.
This is the "no magic strings" principle: change the base model or LoRA rank in
one place and the whole pipeline follows. Frozen dataclasses make typos raise
`AttributeError` instead of silently returning `None`.

## 5. Why an editable install (`pip install -e .`, `pyproject.toml`)

We laid the code out as a real package under `src/pitgpt/`. Installing it in
*editable* mode means `import pitgpt...` works from anywhere (scripts, tests,
the Gradio app) without `sys.path` hacks, while edits to the source take effect
immediately. `src/`-layout also prevents accidentally importing the package from
the repo root instead of the installed copy.

## 6. Why capture a baseline (`inference/baseline.py`)

Fine-tuning claims must be *falsifiable*. We run the **raw** base model on the
exact 20 F1 questions we'll reuse in Phase 5 and save the answers to
`evaluation/baseline_outputs.md`. Without this snapshot, any later "the model
got better" claim is unverifiable. The same 20 prompts live in
`evaluation/prompts.py` so the before/after comparison is strictly like-for-like.

## 7. How generation actually works here

`engine.py` builds a message list (`system` = race-engineer persona, then the
user turn), applies the tokenizer's **chat template** (SmolLM2 is ChatML-style),
and calls `model.generate()`. We decode **only the new tokens** (slicing off the
prompt) so replies don't echo the question. A streaming variant runs `generate`
on a background thread feeding a `TextIteratorStreamer`, which the Gradio UI in
Phase 6 consumes for a live-typing effect.

### Common mistakes this design avoids
- Forgetting `add_generation_prompt=True` → the model continues the *user* turn
  instead of answering.
- Decoding the whole output → the reply repeats the prompt.
- No pad token → batched/streaming generation throws; we set `pad = eos`.
- Sampling with no `repetition_penalty` on a small model → looping text.
