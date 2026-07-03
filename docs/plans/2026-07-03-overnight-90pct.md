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
- [x] 1.2 Career-summary docs — existing driver bios already carry full team
      list + wins + titles; precision achieved via ENTITY PINNING in
      `retriever.py` instead (query names driver/year/GP → keep only matching
      docs; falls back if filter empties, so recall can't regress).
- [x] 1.3 Race-result doc per GP enriched with pole + podium (kb_builder).
- [x] 1.4 Alias collisions fixed on the SCORING side: mask gold teams before
      wrong-pool check ("BMW Sauber" no longer triggers wrong-team "Sauber").
- [x] 1.5 Jolpica module (`rag/jolpica.py`): 2024–26 races (pole+podium),
      season driver–team docs, completed-season champions. Paginated (API
      clamps limit to 100 rows), cached to `data/knowledge/jolpica/`, deduped
      against Kaggle. +100 docs → corpus 2,381. Benchmark vintage guard:
      current-true teams removed from wrong-pool via `recent_teams()`.
- [x] 1.6 +15 curated rules/tech docs (halo, parc fermé, qualifying, sprint,
      flags, penalties, 107%, power unit, records...). Hamilton doc updated
      for the 2025 Ferrari move.
- [x] 1.7 Index rebuilt; harness: 100% recall, 99% clean, 1.3 docs avg.
- [x] 1.8 Routing includes jolpica families; entity pinning covers year/GP/
      name. Hit-rate target (≥95%) exceeded. **PHASE 1 COMPLETE.**

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
- 2026-07-03 ~23:50 — 1.2–1.4 done. Entity pinning + enriched race docs +
  gold-masking scorer fix. Retrieval: 100% recall, 98% clean, 1.3 docs avg.
  **End-to-end facts 63% → 78% with the SAME adapter, no training** —
  champions 100%, gp_winner 75%, teams 67%, win_count 50%. Remaining gap to
  90% is recitation fidelity + formats → Phase 2/3. Next: 1.5–1.6 data
  expansion, then training data.
