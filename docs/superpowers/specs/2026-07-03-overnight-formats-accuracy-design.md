# PitGPT overnight build: user answer formats + ≥90% accuracy — design

**Approved by user 2026-07-03** ("go and implement it"). Executed autonomously
via `/loop`; the implementation checklist lives in
`docs/plans/2026-07-03-overnight-90pct.md`.

## Goals (success gates — all three must pass)

1. **Facts benchmark ≥90% strict** (90 cases, end-to-end RAG; baseline 61%).
2. **Grounding benchmark ≥90%** (105 cases; baseline 72%, missing-context
   honesty collapsed to 5%).
3. **Format adherence ≥90%** — the user's new per-category answer openings
   (baseline 49% on the old signature set).

## The user's answer formats (the centerpiece)

One broadcast-commentary voice, trained into the adapter, measured by regex in
the benchmarks:

| Category | Opening template (regex-checkable) | Example |
|---|---|---|
| Race winner | `{Winner} got the chequered flag at the {year} {GP}!` + race details (team, pole, podium) | "Lewis Hamilton got the chequered flag at the 2020 British Grand Prix! Pole to flag for Mercedes…" |
| Circuit | `Lights out and here we go on the {name} Grand Prix!` + circuit character | "Lights out and here we go on the Silverstone Grand Prix! The home of British motorsport…" |
| Championship | `And that's the title! {Driver} is the {year} World Champion —` + season story | user-approved preview |
| Driver career / teams | `Checking the telemetry on {Driver} —` + full team list with years | "Checking the telemetry on Fernando Alonso — Minardi, Renault, McLaren…" |
| Win count | `The tally board reads {N} —` + context | "The tally board reads 91 — Michael Schumacher's career wins…" |
| Rules & tech | `Let's talk racecraft —` (kept from current adapter) | |
| Opinions / chat | free enthusiast voice, no template | |
| Missing info | `Box, box — that one's not in my notes, so I won't guess.` | doubles as the grounding-decline fix |

## Architecture decision (approach chosen)

**Retrieval-first, single big retrain.** Most of the accuracy gap is retrieval;
fix it with deterministic code (testable instantly, no GPU), then one dataset
regeneration + retrain carries all behaviors. Rejected: train-first iteration
(regression churn) and a bigger base model (won't train on 4 GB).

### Accuracy levers, cheapest first

1. **Answer-bearing docs**: one career-summary doc per driver (all teams +
   win tally) and one result doc per GP (winner, team, pole, podium) so every
   benchmark answer exists verbatim in the corpus.
2. **Deterministic routing**: year + GP-name in the query filters straight to
   that race's doc; driver-name queries filter to that driver's career doc.
   Fixes gp_winner (flat 45%) and win_count (40%).
3. **Alias normalization** to the Kaggle gold names (Lotus/Team Lotus etc.).
4. **Retrieval-only harness** (`scripts/eval_retrieval.py`): measures
   "gold answer present in retrieved context" per category. Target ≥95%
   before any training.
5. **Grounded-recitation training data**: replies recite the context's team
   list / number / winner verbatim inside the new formats.

### Grounding fix

Training set gets counterfactual decline rows (context about X, question about
Y → the "Box, box" template) alongside existing entity-grounding rows. This is
the behavior the last behavior-only retrain lost (85% → 5%).

### Knowledge expansion (user approved all)

- 2024–26 seasons via Jolpica/OpenF1 (one-time download, then offline) +
  current-grid driver→team pairings.
- Richer race docs from existing Kaggle tables.
- Curated rules/tech corpus (DRS, ERS, points systems, eras, sprint).
- Long-tail driver bios (weakest tier).

### Eval updates

`facts_eval.py` signature checks move to the new format regexes;
`grounding_eval.py` admission phrases include "not in my notes" / "Box, box".
A format-adherence column is reported per category.

## Overnight loop protocol

Each `/loop` iteration: read the plan file → do the next unchecked task
(bounded; training runs in ≤10-min foreground chunks with `PITGPT_RESUME=auto`
— the watchdog constraint) → run the relevant benchmark → append numbers to the
plan's status log → commit → schedule the next wakeup. Every adapter version is
kept in its own directory (`models/adapters/pitgpt-lora-vN/`); the app points
at the best-by-combined-score. Nothing is overwritten or force-pushed.

## Risks

- Facts ≥90% is the stretch gate: needs retrieval hit-rate ~95% AND faithful
  recitation. If the night stalls, the plan log will show which category and
  why, with per-iteration adapters and reports preserved.
- Retraining can regress persona → grounding + facts benchmarks after every
  retrain; best adapter selected by the three-gate combined score.
