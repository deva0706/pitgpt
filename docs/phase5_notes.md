# Phase 5 — Evaluation (Teaching Notes)

## Why evaluate this way

Fine-tuning claims must be **shown, not asserted**. The honest test is: give the
*same* prompts, with the *same* decoding settings, to the base model and the
fine-tuned model, and compare. We reuse the exact 20 prompts from
`evaluation/prompts.py` — the same set captured as the Phase 1 baseline — so the
whole project has one consistent yardstick from start to finish.

`compare.py` loads one model at a time (base, then base+adapter), generates all
20 answers, frees the GPU between them, and writes
`evaluation/evaluation_report.md` with the two replies side by side.

### Why load models sequentially and free between?
On 4 GB, holding two copies is wasteful and risks OOM, and generation is
sequential anyway. `del model; gc.collect(); torch.cuda.empty_cache()` returns the
memory before the next load.

## What to look for in the report

Because SmolLM2-360M is tiny and the base model has essentially no F1 knowledge
(its baseline "undercut" answer described a *bent crankshaft*, and "DRS" became
"Driver's Rest Area"), the fine-tune's job is to:

1. **Fix domain grounding** — undercut = pitting early for fresh-tyre track
   position; DRS = drag reduction system in designated zones within one second.
2. **Lock the voice** — calm, technical, race-engineer register; occasional
   "Copy that." without overuse.
3. **Improve consistency** — answers stay on-topic and structured instead of
   drifting into unrelated mechanical nonsense.

## How to judge quality (qualitatively)

We're not computing BLEU/accuracy — for a persona/domain SFT on a small model,
those metrics mislead. Instead we read for:

- **Factual correctness** of the F1 content (no invented regulations/numbers).
- **Relevance** — does it answer *this* question?
- **Register** — does it sound like a race engineer, not a fan or a manual?
- **Coherence** — does it stay consistent within the answer?

## Honest limitations to state in the report

- **Small model ceiling.** 360M params can't reason deeply; expect competent
  explanations, not novel strategy analysis.
- **Narrow dataset (44 unique outputs).** The model may echo canonical answers
  closely for in-distribution questions and be weaker on unusual phrasings.
- **No live data.** It knows *concepts*, not this weekend's timing sheets.
- **Sampling variance.** With `do_sample=True`, re-running gives slightly
  different wording; the report is a representative snapshot, not a fixed oracle.

These limitations are features of the honest write-up, not failures to hide — a
production ML report always bounds what the model can and cannot do.
