# Phase 4 — Interview Questions & Model Answers

### Q1. Walk through what `prepare_model_for_kbit_training` does and why it's needed.

**Answer.** A model loaded in 4-bit has frozen, quantized weights that aren't
directly trainable. This helper prepares it: it casts small, sensitive modules
(layernorms, and often the LM head) to fp32 for numerical stability, disables the
KV cache, enables gradient checkpointing if requested, and registers a forward
hook on the input embeddings so their output `requires_grad`. That last part is
subtle but essential — without it, gradients can't propagate back through the
frozen base to reach the LoRA adapters, and you'd train nothing.

### Q2. You saw only 4.07 % of parameters were trainable. Where did the other 96 % go?

**Answer.** They're the frozen base model — quantized to 4-bit and marked
`requires_grad=False`, so no gradients or optimizer state are allocated for them.
The 4.07 % (~8.7 M params) are the LoRA `A`/`B` matrices on the attention and MLP
projections. That ratio is the entire value proposition of QLoRA: we get to adapt
a 213 M-parameter model while only optimizing ~8.7 M, which is what lets it fit
in 4 GB.

### Q3. Why bf16 instead of fp16 for this run?

**Answer.** fp16 has a narrow exponent range, so mixed-precision training needs a
GradScaler to stop small gradients underflowing to zero. Our 4-bit + LoRA setup
produced bf16 gradients, and the fp16 GradScaler literally has no CUDA kernel to
unscale bf16 tensors — training crashed. bf16 has fp32's exponent range with less
mantissa precision, so it needs no scaler and won't overflow. On Ampere (RTX
3050) bf16 is hardware-accelerated, so it's the correct default for QLoRA.

### Q4. What is gradient accumulation actually doing at the tensor level?

**Answer.** Normally you do forward → backward → `optimizer.step()` →
`zero_grad()` each batch. With accumulation of N, you run forward → backward N
times *without* stepping; because PyTorch accumulates `.grad` by summing, after N
micro-batches the gradients equal (up to scaling) what a single batch N-times
larger would have produced. Then you step once and zero. You get the gradient
quality of a big batch with the memory of a small one — you just pay N× the time
between optimizer steps.

### Q5. Why save only the adapter, and how do you use it later?

**Answer.** The base model is unchanged, so re-saving it wastes hundreds of MB.
The adapter is just the trained `A`/`B` matrices plus `adapter_config.json`
(rank, alpha, targets) — tens of MB. At inference you load the base model
(optionally re-quantized to 4-bit) and call `PeftModel.from_pretrained(base,
adapter_dir)`, which re-attaches the adapters. One base can host many swappable
task adapters, which is a big deployment advantage.

### Q6. How would you diagnose an out-of-memory error mid-training on this GPU?

**Answer.** First confirm what's consuming VRAM: `torch.cuda.max_memory_allocated`
gives the peak. The usual levers, in order of impact: lower `max_length` (activation
memory scales with sequence length), reduce `per_device_train_batch_size` (and
raise `gradient_accumulation_steps` to keep the effective batch), ensure
`gradient_checkpointing=True` and `use_cache=False`, and use the paged 8-bit
optimizer. If still failing, drop LoRA rank or the number of target modules. Our
autonomy plan does exactly this: cut batch/seq/rank once, retry, and if it still
OOMs, fall back to the base model so the pipeline stays end-to-end.
