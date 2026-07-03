# Overnight plan: formats + ≥90% — execution checklist

Spec: `docs/superpowers/specs/2026-07-03-overnight-formats-accuracy-design.md`
Loop contract: each iteration does the next unchecked task(s), appends to the
Status log, commits, reschedules. Training only in ≤10-min foreground chunks
(`PITGPT_RESUME=auto`). Never overwrite an adapter dir; new versions go to
`models/adapters/pitgpt-lora-vN/`.

**Gates:** facts ≥90% strict · grounding ≥90% · format adherence ≥90% · tests pass.

## Phase 1 — Corpus makes every answer findable (no GPU)

- [x] 1.1 Retrieval-only harness `scripts/eval_retrieval.py`: % of the 90 facts
      cases whose gold answer appears in the retrieved context. Baseline number
      logged before any changes. **DONE — 100% recall; bottleneck is context
      PRECISION (28/30 team contexts polluted with wrong teams) + recitation.**
- [ ] 1.2 Career-summary doc per driver (name, ALL teams with years, win tally,
      titles) in `kb_builder.py`, source `kaggle_career`.
- [ ] 1.3 Race-result doc per GP (year, name, winner, team, pole, podium),
      source `kaggle_race` (replaces/enriches current race docs).
- [ ] 1.4 Alias normalization pass (constructor display names → Kaggle gold).
- [ ] 1.5 Recent seasons 2024–26 + current grid via Jolpica/OpenF1 download;
      cached to `data/knowledge/` so later runs are offline.
- [ ] 1.6 Curated rules/tech docs (DRS, ERS, points, sprint, eras, flags).
- [ ] 1.7 Rebuild index; retrieval harness per category.
- [ ] 1.8 Deterministic routing upgrades: year+GP filter for race queries,
      driver-name filter for career/tally queries. Iterate until retrieval
      hit-rate ≥95% overall.

## Phase 2 — Training data carries all behaviors (no GPU)

- [ ] 2.1 Format templates module (single source of truth for the 8 openings,
      importable by dataset builder AND evals).
- [ ] 2.2 Regenerate behavior dataset: counterfactual grounded-recitation rows
      in the new formats for every category + decline rows ("Box, box —") +
      persona/opinion rows. Validator + tests updated.
- [ ] 2.3 Update `facts_eval.py` + `grounding_eval.py` to score the new format
      regexes; add format-adherence column.

## Phase 3 — Retrain + measure

- [ ] 3.1 Train adapter v2 in resumable 10-min foreground chunks.
- [ ] 3.2 Run all three benchmarks (facts --rag, grounding, format adherence).
- [ ] 3.3 Log numbers. If any gate <90%: diagnose weakest category, apply the
      matching lever (routing / data mix / more decline rows / more epochs),
      loop back to the cheapest fix. Repeat 3.1–3.3.

## Phase 4 — Ship

- [ ] 4.1 Point app/config at best adapter; end-to-end smoke via chat engine.
- [ ] 4.2 Full pytest suite green.
- [ ] 4.3 README evidence tables + STATUS.md updated with final numbers.
- [ ] 4.4 Final commit; PushNotification summary to user.

## Status log (append-only; newest last)

- 2026-07-03 ~21:50 — Plan created. Baselines: facts 61% strict / 32%
  confident-wrong; grounding 72% / 18.1% halluc (declines 5%); signature
  adherence 49% (old formats). Corpus 2,270 docs.
- 2026-07-03 ~23:40 — 1.1 done. **Retrieval recall is already 100%** on all 90
  cases — the 61%→90% gap is generation, not search. But precision is bad:
  28/30 driver_teams contexts carry ≥1 wrong team among the k=4 docs (same
  conflation expected for gp_winner). Re-weighted plan: 1.2/1.3/1.8 are
  PRECISION plays (one authoritative doc per entity, routed deterministically,
  possibly k=1–2 for entity questions); the biggest single lever is Phase 2
  grounded-recitation training in the new formats.
