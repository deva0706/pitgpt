# PitGPT — Autonomous Build STATUS

> **2026-07-07 update — retrieval-routing overhaul shipped; retrain rolled back.**
> Live-reported failures ("Tell me about the 2021 Abu Dhabi GP" → falsely
> declared never-held; "Yas Marina GP" aliases → false absences or wrong-race
> recitals) were root-caused to the QUERY layer, not the model: route order
> sent tell-me-about race queries to driver docs, and the absence guard
> validated against that starved slice. Shipped fixes (all in `retriever.py`/
> `kb_builder.py`, no retraining): race-intent route for year+GP queries,
> corpus-wide absence verification with doc recovery, case-insensitive GP
> pinning, circuit names on all 1,125 race docs, teams-route narrowing, plus
> "who is" route and Austrian/Australian stem-collision bugfixes. 15 new
> regression tests (35 total). Facts benchmark on the SAME adapter: 98% → 99%
> (`facts_report_diag_postfix.md`); all live failures reproduce-then-pass.
> A retrain against the remaining behavior gaps (absence recitation, venue
> phrasings, prose cards; `pitgpt-lora-v5-absence/`) fixed its targets but
> mangled GP names on a common phrasing (96%, one new confident-wrong) and
> was **rolled back** — see that adapter's README for the root-cause
> hypothesis (venue counterfactuals must keep real GP↔circuit pairs) and the
> planned next iteration. Shipped adapter remains v4 (byte-identical to
> `pitgpt-lora-v4-final`).

> **2026-07-04 update — formats + accuracy overnight build shipped.**
> Facts benchmark **23% → 97% strict** (confident-wrong 2%), user reply
> formats at **98% adherence**, corpus 2,381 docs incl. live 2024–26 seasons.
> Grounding 72% (decline seesaw documented). See
> `docs/plans/2026-07-03-overnight-90pct.md` and the README evidence tables.
> Everything below this line describes the original 360M-era build.

**Run outcome: ✅ complete, end-to-end.** All 7 phases were built, the model was
fine-tuned with QLoRA, the adapter was saved, and the chat app loads it and
behaves like an F1 race engineer. Work is committed to a **dedicated local git
repo**. Nothing was pushed anywhere.

## Post-build follow-ups (after the initial run)

- **Fixed a chat crash** (Gradio 6 list-content messages) — commit in history.
- **Expanded the dataset** from 1014 → **2310 pairs** (82 concepts, 96 unique
  outputs) and **gave PitGPT a stronger personality** (richer system prompt +
  identity/pep-talk/status/celebrate concepts and character-rich radio calls).
- **Retrained** on the 2310-pair set. Findings: 3 epochs drove loss to ~0.12 and
  **over-fit**, producing verbatim persona answers but garbled technical prose
  ("Dynamic Storm Radar" for DRS). A **1-epoch** fine-tune (loss ~1.17) generates
  far more coherently across the board with personality intact, so that is the
  **shipped adapter**. Config reflects `num_train_epochs = 1` with a comment.
- **Note on training runs:** long *background* training tasks were killed by an
  environment watchdog partway through; the shipped 1-epoch adapter was trained
  in the **foreground** (≈570 s, under the 10-min limit) and completed cleanly.
- The 360M factual-wobble limitation still applies and is inherent to the model
  size — see `docs/BLOCKERS.md`.

---

## What succeeded

| Phase | Deliverable | Status |
|------|-------------|--------|
| 1 | Env verified (RTX 3050, 4 GB, CUDA, bitsandbytes), config + inference engine, **baseline captured** | ✅ `src/pitgpt/evaluation/baseline_outputs.md` |
| 2 | **1014** Alpaca pairs (36 concepts + 8 radio, 44 vetted outputs), validator, 5 tests | ✅ regenerate via `scripts/build_dataset.py` |
| 3 | QLoRA theory notes (14 concepts) + interview Q&A | ✅ `docs/phase3_notes.md` |
| 4 | **QLoRA training** — loss 3.68→0.48, token acc 0.38→0.89, peak **0.76 GB**, adapter saved | ✅ `models/adapters/pitgpt-lora/` |
| 5 | Base-vs-fine-tuned report on 20 prompts + discussion | ✅ `src/pitgpt/evaluation/evaluation_report.md` |
| 6 | Streaming, F1-themed Gradio app (verified end-to-end) | ✅ `src/pitgpt/app/` |
| 7 | README, scripts, LICENSE, tests, packaging | ✅ repo is GitHub-ready |

- **Tests:** 19 passing (`pytest`).
- **Every phase** has `docs/phaseN_notes.md` (teaching) **and**
  `docs/interview_phaseN.md` (Q&A), per the interview-mode instruction.
- The fine-tune is a clear, consistent improvement over the base model in domain
  and voice across all 20 eval prompts.

## What's in BLOCKERS.md

No hard blockers. Six issues were hit and **resolved** during the run (SSL cert,
transformers-5 chat-template, fp16→bf16 training crash, Gradio-6 API, adapter
detection, wasteful ONNX download). One **open limitation** (not a blocker): the
360M model garbles some factual specifics under sampling — mitigated by lowering
temperature to 0.35. Full detail in `docs/BLOCKERS.md`.

---

## ⚠️ Important things to know when you wake

1. **This folder is now its own git repo.** I found that `PitGPT/` was an
   *untracked subfolder* of a git repository rooted at your home directory
   (`C:/Users/devya`, remote `apify.git`). Committing there would have polluted an
   unrelated project, so I ran `git init` **inside `PitGPT/`** (branch `main`) and
   made all 7 commits there. **I did not touch the home/apify repo, and I did not
   push anything** (no remote is set).
   - If you want PitGPT on GitHub: create an empty repo, then
     `git remote add origin <url>` and `git push -u origin main`.
   - If you'd rather it live inside another repo differently, tell me and I'll
     restructure.

2. **Environment.** Everything runs in the existing **`pitgpt` conda env**
   (Python 3.11). I fixed a broken `SSL_CERT_FILE` in that env by adding
   `…/envs/pitgpt/ssl/cacert.pem` (copied from certifi). Nothing else in the env
   was changed except `pip install -e .` (this package) and `pip install pytest`.

3. **Large/derived files are gitignored** (correct): the base model HF cache, the
   generated dataset JSON, and the LoRA adapter weights are **not** committed. The
   adapter exists locally at `models/adapters/pitgpt-lora/`. To rebuild from
   scratch on another machine: `scripts/build_dataset.py` → `scripts/train.py`.

## Manual steps to run it yourself

```bash
conda activate pitgpt
cd "C:/Users/devya/OneDrive/Desktop/projecta/PitGPT"

# (optional) confirm the environment
python scripts/check_gpu.py

# launch the race engineer (loads base + your trained adapter)
python scripts/chat.py           # open the printed http://127.0.0.1:7860
```

Other entry points: `scripts/build_dataset.py`, `scripts/run_baseline.py`,
`scripts/train.py` (re-train), `scripts/evaluate.py` (regenerate the report).
Set `HF_HUB_OFFLINE=1` to force fully-offline once the model is cached.

## Suggested next steps (optional)

- Add a real screenshot of the running app to the README's Screenshots section.
- Grow the dataset's *distinct concepts* (not just paraphrases) to lift the
  44-unique-output ceiling and improve factual robustness.
- Add a held-out test split + an LLM-as-judge rubric for quantitative scoring.
