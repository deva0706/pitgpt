# BLOCKERS & Issues Log

**Summary: no hard blockers.** The pipeline completed end-to-end — dataset →
QLoRA training → adapter saved → evaluation → working chat app. Every issue below
was diagnosed and resolved during the autonomous run; the last entry is a *quality
limitation*, not a blocker.

---

## Resolved issues

### 1. Broken `SSL_CERT_FILE` in the conda env (would block all HTTPS)
- **Symptom:** importing `gradio` and any Hugging Face download crashed with
  `FileNotFoundError` in `ssl.py` — the env's `SSL_CERT_FILE` pointed at
  `…/envs/pitgpt/ssl/cacert.pem`, which did not exist.
- **Fix:** copied `certifi`'s bundled `cacert.pem` to that expected path. Import
  and downloads then worked. (Permanent for the env.)

### 2. `apply_chat_template` return type (transformers 5.x)
- **Symptom:** `generate()` failed with `AttributeError` on `.shape` — v5 returns
  a dict-like `BatchEncoding`, not a bare tensor, and a naive `.items()` rebuild
  corrupted it.
- **Fix:** use `return_dict=True` and move it with `BatchEncoding.to(device)`;
  pass `**inputs` (with attention mask) into `generate`.

### 3. fp16 GradScaler vs bf16 gradients (training crash)
- **Symptom:** `"_amp_foreach_non_finite_check_and_unscale_cuda" not implemented
  for 'BFloat16'` at the first optimizer step.
- **Fix:** switched to **bf16** compute dtype + `bf16=True` training (Ampere
  RTX 3050 supports it). bf16 needs no grad scaler. Documented in
  `docs/phase4_notes.md`.

### 4. Gradio 6 API changes
- **Symptom:** `Chatbot(type=...)` rejected; `css`/`theme` no longer accepted by
  the `Blocks` constructor.
- **Fix:** dropped `type=` (v6 uses messages format by default); moved `css`/
  `theme` to `demo.launch()`.

### 5. Adapter-detection false positive
- **Symptom:** the app reported "adapter loaded" mid-training because the output
  *directory* existed (with only `checkpoint-*/`).
- **Fix:** `using_adapter` now checks for `adapter_config.json` + the adapter
  weights file, not just the directory.

### 6. `snapshot_download` pulled unused ONNX variants (slow, ~1.5 GB)
- **Symptom:** the full-repo download fetched ONNX exports before the safetensors.
- **Fix:** a targeted download with `allow_patterns` for the core files; the
  safetensors completed and inference proceeded. (Wasted bandwidth only, not a
  blocker.)

### 7. Git repository root was the home directory (important — see STATUS.md)
- **Finding:** `PitGPT/` was an *untracked subfolder* of a git repo rooted at
  `C:/Users/devya` whose remote is `apify.git` (an unrelated project).
- **Fix:** initialised a **dedicated git repo inside `PitGPT/`** (branch `main`)
  so all commits are isolated to this project. No changes were made to the
  outer/home repo. No remote is configured (nothing was pushed).

---

## Open limitation (not a blocker)

### Small-model factual wobble
The fine-tuned 360M model reliably adopts the race-engineer **voice** and the
right **topics**, and is a clear improvement over the base model, but under
sampling it still garbles some technical **specifics** (and shows occasional
decoding artifacts). This is inherent to a 360M model trained on a compact,
44-unique-output dataset. Mitigations applied: default temperature lowered to
0.35. Further improvement would require authoring more distinct concepts and/or a
larger base model — see the Future Work section of the README and the discussion
in `src/pitgpt/evaluation/evaluation_report.md`.
