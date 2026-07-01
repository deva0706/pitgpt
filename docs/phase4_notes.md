# Phase 4 — Training (Teaching Notes)

This phase turns the theory of Phase 3 into a working QLoRA fine-tune. Two files
do the work: `training/data_module.py` (render the dataset) and
`training/train.py` (the loop). Config lives in `config/settings.py`.

## The five steps of the loop

1. **Tokenizer + 4-bit base.** `load_model(quantized=True)` builds a
   `BitsAndBytesConfig` (NF4, double-quant, bf16 compute) and loads SmolLM2 with
   its weights frozen in 4-bit. Memory footprint: ~0.24 GB.
2. **`prepare_model_for_kbit_training`.** A k-bit model isn't trainable as-is.
   This helper casts layernorms to fp32 for stability, enables gradient
   checkpointing, and — crucially — makes the input embeddings produce gradients
   so the backward pass can reach the adapters.
3. **Attach LoRA.** We pass `peft_config` to `SFTTrainer`, which wraps the model
   with adapters on the target modules. Only ~**8.68 M of 213 M** parameters
   (**4.07 %**) are trainable — everything else stays frozen.
4. **`SFTTrainer.train()`.** TRL's supervised trainer handles batching,
   the loss (next-token cross-entropy over the rendered chat text), gradient
   accumulation, checkpointing, and logging.
5. **Save only the adapter.** `trainer.save_model()` writes `adapter_config.json`
   + adapter weights (tens of MB), not a full model copy. The tokenizer is saved
   alongside so inference has everything it needs.

## Why each memory setting exists (4 GB budget)

| Setting | Value | Why |
|---|---|---|
| `load_in_4bit` + NF4 | on | frozen base in ~0.24 GB instead of ~1.4 GB fp32 |
| `bnb_4bit_compute_dtype` | **bfloat16** | Ampere-native; no grad scaler, no overflow |
| `gradient_checkpointing` | on | recompute activations → big activation-memory saving |
| `optim` | `paged_adamw_8bit` | optimizer state in 8-bit, paged to CPU on spikes |
| batch / accumulation | 4 / 4 | effective batch 16 with a tiny live footprint |
| `max_length` | 512 | our chats are ~224 tokens; caps the worst case |

Measured peak during a smoke test: **~0.7 GB** at batch 1 — which is exactly why
we could safely raise the batch size to 4 and still sit far under 4 GB.

## The bf16-vs-fp16 bug (a real one we hit)

First attempt used `fp16=True`. Training crashed with
`"_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`.
Cause: fp16 mixed precision uses a **GradScaler** to prevent fp16 gradient
underflow, but our adapter gradients were bf16, which the scaler can't unscale.
Fix: use `bf16=True` (and bf16 compute dtype) instead. bf16 has the dynamic range
of fp32, so it needs no scaler at all. Lesson: **on Ampere and newer, prefer bf16
for QLoRA** — it sidesteps a whole class of mixed-precision problems.

## Why we render the text ourselves

`data_module.build_sft_dataset` applies the tokenizer's chat template to
`system + user + assistant` and stores the result in a `text` column. Doing this
explicitly guarantees the training string matches the inference string exactly
(same special tokens, same system prompt). We keep `packing=False` because our
examples are short, independent Q&A — packing multiple examples into one sequence
would let one answer bleed context into an unrelated question.

## Reading the training log

- **loss** should trend down over the run — that's the model fitting the answers.
- **mean_token_accuracy** should climb — fraction of next-tokens predicted right.
- **grad_norm** should stay finite and not explode — stability check.
- **Trainable params 4.07 %** printed at the start proves LoRA is active (a full
  fine-tune would show 100 %).

## Common training mistakes this design avoids

- Forgetting `model.config.use_cache = False` with gradient checkpointing → a
  warning and wasted memory (the KV cache is meaningless during training).
- Training in fp16 on 4-bit + LoRA → the grad-scaler crash above.
- Saving the whole model instead of the adapter → hundreds of MB for no reason.
- Mismatched train/inference formatting → the model "works in training" but
  answers oddly in the app.
- Setting `max_length` too high → needless memory and slower steps.
