# Phase 1 — Interview Questions & Model Answers

Self-study questions for the environment/baseline phase, with worked answers.

### Q1. `torch.cuda.is_available()` returns `False` even though `nvidia-smi` works. What's wrong and how do you fix it?

**Answer.** `nvidia-smi` only proves the GPU driver is installed; it says nothing
about your Python environment. The usual cause is that you installed the
**CPU-only** `torch` wheel (the default from PyPI on some setups) instead of a
CUDA build. Fix by installing a CUDA-matched wheel, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu121`, and
confirm `torch.version.cuda` is non-`None`. Other causes: a driver too old for
the CUDA runtime the wheel bundles, or a `CUDA_VISIBLE_DEVICES=""` env var
hiding the GPU.

### Q2. Why fine-tune a 360M model instead of a 7B one on this hardware?

**Answer.** VRAM. On a 4 GB card, a 7B model won't even load in 4-bit (needs
~4–5 GB just for weights, before activations/optimizer state). SmolLM2-360M in
4-bit occupies ~1 GB, leaving room for gradient checkpointing, LoRA optimizer
state, and a sequence or two of activations. Smaller also means faster
iteration. The trade-off is capability: a 360M model has less world knowledge,
which is exactly why we constrain it to a narrow persona (F1 race engineer)
rather than expecting general excellence.

### Q3. What does the `-Instruct` suffix change about how you use the model?

**Answer.** An `-Instruct` model has been post-trained (SFT/RLHF-style) to follow
a **chat template** and respond to instructions, so you must format inputs with
that template (`tokenizer.apply_chat_template`) and use `system`/`user`/
`assistant` roles. A raw base (non-instruct) model is a pure next-token
predictor — you'd prompt it with few-shot examples instead. Using an instruct
model means our baseline is already conversational, so fine-tuning adjusts
*domain and style*, not basic instruction-following.

### Q4. Why capture baseline outputs before training instead of just comparing later?

**Answer.** Because the base model's weights are frozen and shared, but decoding
is **stochastic** (sampling) and library versions drift. Capturing the baseline
now, with the same prompts, decoding config, and code path we'll reuse, pins a
reproducible reference. It also documents the starting point in git, so the
Phase 5 report can make a *falsifiable* before/after claim rather than relying on
memory. It's the ML equivalent of writing the failing test first.

### Q5. Why decode only `output_ids[:, input_len:]` instead of the full output?

**Answer.** `model.generate()` returns the **prompt tokens followed by the newly
generated tokens** in one tensor. If you decode the whole thing, the reply
includes a verbatim copy of the system prompt and user question. Slicing off the
first `input_ids.shape[-1]` tokens yields only the model's answer. (With a
streamer we instead pass `skip_prompt=True`.) Forgetting this is the single most
common "why is my chatbot repeating the question?" bug.
