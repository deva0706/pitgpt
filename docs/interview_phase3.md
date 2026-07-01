# Phase 3 — Interview Questions & Model Answers

### Q1. Explain LoRA to someone who knows backprop but not PEFT.

**Answer.** Full fine-tuning learns a dense update `ΔW` for each weight matrix
`W`, the same size as `W`. LoRA assumes that update is *low-rank* and factorises
it as `ΔW = B·A` with `A ∈ ℝ^{r×d}`, `B ∈ ℝ^{d×r}`, `r ≪ d`. You freeze `W` and
train only `A` and `B`, so trainable parameters drop from `d²` to `2dr`. The
forward pass is `h = Wx + (α/r)·B·A·x`. `B` is initialised to zero so the adapter
starts as a no-op. You get most of the benefit of fine-tuning at a tiny fraction
of the trainable parameters, memory, and storage.

### Q2. What specifically is "quantized" in QLoRA, and what stays full precision?

**Answer.** The **frozen base model weights** are quantized to 4-bit NF4 — that's
the memory win. The **LoRA adapter weights** stay in higher precision (they're
tiny and must train well). During the forward pass the 4-bit weights are
dequantized to the **compute dtype** (fp16 here) for the matmuls, so the math is
done in fp16 even though storage is 4-bit. Gradients only flow into the adapters;
the base never updates.

### Q3. Why NF4 instead of regular int4?

**Answer.** Int4 spaces its 16 levels uniformly, but neural-network weights are
roughly zero-mean Gaussian — most mass is near zero. NF4 (NormalFloat4) places
its levels to be optimal for normally distributed values, so it spends more
resolution where the weights actually are. That reduces quantization error for
the same 4 bits. Double quantization additionally quantizes the block-wise scale
constants, shaving a bit more memory.

### Q4. What do rank and alpha each control, and why decouple them?

**Answer.** Rank `r` sets the *dimensionality* (capacity) of the update — how much
the adapter can express. Alpha sets the *magnitude*: the adapter output is scaled
by `α/r` before being added to the frozen path, so alpha controls how loudly the
adapter speaks. Decoupling them means you can change capacity (`r`) without
simultaneously changing effective strength — you re-scale by adjusting `α`. A
common heuristic is `α = 2r`.

### Q5. My GPU only fits batch size 1. How do I still train stably?

**Answer.** Gradient accumulation. Run several micro-batches (e.g. 16) of size 1,
accumulate their gradients, and only step the optimizer after the group — this
simulates an effective batch of 16 without ever holding 16 sequences in memory.
Combine it with gradient checkpointing (recompute activations instead of storing
them) and a paged 8-bit optimizer (AdamW state in 8-bit, paged to CPU) to stay
inside VRAM. The trade is more compute/time for the memory you don't have.

### Q6. Which layers should LoRA target, and does it matter?

**Answer.** It matters. Targeting the attention projections (`q,k,v,o_proj`)
changes *what the model attends to*; adding the MLP projections
(`gate,up,down_proj`) changes *how it transforms* each token. Adapting only `q`
and `v` is cheapest and often enough for light tasks; adapting all attention +
MLP projections gives a stronger adaptation, which suits a persona/domain shift
like PitGPT. More targets means more trainable params and VRAM, so it's a
capacity-vs-cost choice.

### Q7. After training, what do you actually save and ship?

**Answer.** Only the **LoRA adapter** — the `A`/`B` matrices plus an
`adapter_config.json` describing rank, alpha, and target modules. That's a few to
tens of MB, versus a full model copy of hundreds of MB or more. At inference you
load the base model (optionally re-quantized) and attach the adapter with
`PeftModel.from_pretrained`. You can keep multiple task adapters for one base
model and swap them, which is a major operational advantage of PEFT.
