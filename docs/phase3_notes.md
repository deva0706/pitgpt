# Phase 3 — QLoRA Theory (Teaching Notes)

No training code is written until every concept below is explained. This is the
mental model you need to read `training/train.py` (Phase 4) and understand *why*
every config value in `config/settings.py` is what it is.

---

## 1. Transformer — the architecture

A transformer processes a sequence of tokens through a stack of identical
**blocks**. Each block has two sub-layers: **self-attention** (tokens exchange
information) and a **feed-forward MLP** (each position is transformed
independently), with residual connections and normalisation around both. A
**causal** (decoder-only) LM like SmolLM2 masks attention so each token can only
see earlier tokens — that's what lets it generate left-to-right. The final layer
projects each position to a probability distribution over the vocabulary.

## 2. Tokenizer — text ↔ integers

Models don't see text; they see token ids. A tokenizer splits text into subword
units (SmolLM2 uses a byte-level BPE, like GPT-2) and maps each to an integer.
Subword tokenisation is a compromise between characters (too long) and words
(huge vocabulary, no handling of unseen words). Key practical points: the same
tokenizer must be used for training and inference; special tokens (BOS/EOS/pad)
matter; and the **chat template** is a tokenizer feature that formats
`system/user/assistant` turns into the exact string the model was trained on.

## 3. Embeddings — ids ↔ vectors

The first layer is an **embedding table**: a big matrix with one learned vector
per vocabulary token. Token id → row lookup → a dense vector the network can do
math on. Semantically similar tokens end up with similar vectors. (Positional
information is added too, so the model knows token order.) Embeddings turn
discrete symbols into a continuous space where attention and MLPs operate.

## 4. Attention — the core mechanism

Self-attention lets each token build a query **Q**, and every token exposes a key
**K** and value **V** (all linear projections of the token's vector). A token
attends to others by comparing its Q to their K (dot product → softmax weights),
then takes a weighted sum of their V. Intuition: "for this word, which earlier
words matter, and what information do I pull from them?" Multiple **heads** do
this in parallel over different subspaces. The projections that create Q, K, V
and recombine the output (`q_proj`, `k_proj`, `v_proj`, `o_proj`) are exactly the
weights LoRA adapts.

## 5. Quantization — fewer bits per weight

A weight normally lives in 16- or 32-bit floating point. **Quantization** stores
it in fewer bits (here, 4) by mapping ranges of float values to a small set of
levels, with a scale factor per block to preserve dynamic range. Benefit: 4-bit
weights use ~1/4 the memory of fp16, so a 360M model's weights drop from ~700 MB
to ~200 MB — this is what makes a 4 GB GPU viable. Cost: a small precision loss,
which QLoRA is specifically designed to tolerate.

## 6. NF4 — NormalFloat 4-bit

Not all 4-bit schemes are equal. **NF4** places its 16 quantization levels to be
*information-theoretically optimal for normally-distributed data*. Neural network
weights are approximately zero-mean Gaussian, so NF4's non-uniform levels
(denser near zero, where most weights are) represent them more faithfully than
plain linear int4. **Double quantization** goes further and quantizes the
per-block scale factors too, saving a little more memory. Both are set in our
`BitsAndBytesConfig`.

## 7. Frozen weights vs adapter weights

In QLoRA the entire base model is **frozen** — loaded in 4-bit and never updated;
`requires_grad = False` on those parameters. We insert small, trainable
**adapter** weights (LoRA) alongside specific layers and train *only those*. So
gradients and optimizer state exist only for the tiny adapter — that's the memory
win. At the end we save just the adapter (a few MB), not a new copy of the model.

## 8. LoRA — Low-Rank Adaptation

Fine-tuning normally updates a weight matrix `W` by learning a full-size update
`ΔW`. LoRA's insight: that update is **low-rank** — it can be approximated as the
product of two skinny matrices, `ΔW ≈ B · A`, where `A` is `r × d` and `B` is
`d × r` for a small **rank** `r`. Instead of training `d × d` numbers, you train
`2 · d · r`, a tiny fraction. The forward pass becomes `h = W·x + (B·A)·x`, with
`W` frozen and only `A`, `B` trained. `A` starts random, `B` starts at zero, so
the adapter begins as a no-op and the model behaves like the base model at step 0.

## 9. QLoRA = Quantized base + LoRA

QLoRA is the combination: **quantize the frozen base to 4-bit NF4** *and* train
**LoRA adapters** on top. The base is tiny in memory (4-bit), the trainable part
is tiny in count (low-rank), and matmuls upcast to fp16 for numerical stability
(`bnb_4bit_compute_dtype`). Result: fine-tuning a model that would never fit in
full precision on a 4 GB laptop GPU. It's the single technique that makes this
whole project possible on an RTX 3050.

## 10. Rank (`r`)

The dimension of the low-rank update. Higher `r` = more capacity for the adapter
to change behaviour, but more parameters and VRAM. Typical values are 8–64. We
use **16**: enough to learn a persona and domain, cheap enough for 4 GB. Too low
and the adapter can't capture the task; too high and you waste memory and risk
over-fitting a small dataset.

## 11. Alpha (`lora_alpha`) and scaling

The adapter output is scaled by `alpha / r` before being added to the frozen
path. `alpha` therefore controls how *strongly* the adapter speaks relative to
the base. A common convention is `alpha = 2 · r` (we use `r=16`, `alpha=32`, so
scale = 2). Think of `alpha/r` as an effective learning-rate multiplier on the
adapter's contribution — decoupling "how big is the update space" (`r`) from "how
loud is it" (`alpha`).

## 12. Dropout (`lora_dropout`)

Standard regularisation applied on the adapter's input path: during training a
fraction of activations are randomly zeroed, preventing the small adapter from
memorising the training set. We use **0.05**. With only 44 unique outputs,
some dropout is healthy insurance against over-fitting.

## 13. Target modules

The list of linear layers that receive an adapter. For a Llama-style block
(SmolLM2) these are the attention projections `q_proj, k_proj, v_proj, o_proj`
and the MLP projections `gate_proj, up_proj, down_proj`. Adapting attention
changes *what the model attends to*; adapting the MLP changes *how it transforms*
each token. Targeting all of them (rather than just Q and V) gives a stronger,
more thorough adaptation at a modest extra cost — appropriate for a persona shift.

## 14. Gradient accumulation

Our VRAM only allows a **batch size of 1** sequence. But tiny batches make noisy
gradients and unstable training. **Gradient accumulation** runs several forward/
backward passes (here **16**), summing gradients, and only then does one
optimizer step — simulating an effective batch size of `1 × 16 = 16` without ever
holding 16 sequences in memory at once. It trades wall-clock time for the memory
we don't have.

## 15. Supporting memory tricks (used in Phase 4)

- **Gradient checkpointing** — don't store all intermediate activations for the
  backward pass; recompute them on the fly. Saves a lot of VRAM for extra compute.
- **Paged 8-bit optimizer** (`paged_adamw_8bit`) — the AdamW optimizer state
  (which can rival the model in size) is stored in 8-bit and paged to CPU memory
  when VRAM is tight, preventing OOM spikes.

---

### The one-paragraph summary

Freeze SmolLM2 and store it in 4-bit NF4 (cheap memory). Add small low-rank LoRA
matrices (`r=16`, `alpha=32`) to the attention and MLP projections and train only
those. Use batch size 1 with 16-step gradient accumulation, gradient
checkpointing, and a paged 8-bit optimizer to fit inside 4 GB. Save only the
~10–30 MB adapter. That is QLoRA on a laptop.
