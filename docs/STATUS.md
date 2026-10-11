# PitGPT — Autonomous Build STATUS

> **2026-10-11, later — tested through the browser; the pipeline around the
> model rebuilt.** Typing ordinary questions into the running app (36 turns,
> each traced: question → facts → what the model wrote) showed the 99%
> benchmark only described benchmark-shaped questions:
> - a keyword list decided whether to search at all — "Tell me about
>   Hamilton", "Why is Monaco so special?", "Who drives for Ferrari?" were
>   small talk and were made up;
> - "who is verstappen" carded Jos with Max's wins; with the right fact in
>   the prompt, "Tell me about the 2008 Singapore Grand Prix" named Massa the
>   winner (the adapter only ever saw "Who won…");
> - "And in 2007?" → "And in 2010?" → "Who was on pole?" lost the thread;
> - bios stopped at 2024, there were no line-ups, and nothing checked a reply.
>
> No retraining. New layers: `rag/records.py` (facts back out of documents),
> `rag/entities.py` (names → the corpus's), `retriever._pick` (the one fact a
> question names), `rag/stats.py` (counts, computed), `inference/dialogue.py`
> (the chat replayed into standalone questions), `inference/answers.py` (ask
> in the trained phrasing, check the reply, show the fact's own rendering if
> it fails). Corpus 2,381 → 2,452 docs: careers through the current season,
> team line-ups, 2025 constructors, standings of the season in progress, race
> venues on recent races. UI: your message shows at once, bubble widths, New
> session keeps typed text, no request leaves the machine at start-up.
>
> Measured on the final code (`*_oct11_pipeline`, `*_oct11_checked`,
> `*_oct11_model`): retrieval 100% hit / 100% clean / 1.0 docs; facts **100%
> shown** (model's own wording 89/90, shown as written 76/90); grounding **86%
> shown, 0 hallucinations** against 73% / 13.3% for the model alone. 161 tests
> (was 67). All 36 original turns and about 60 further questions re-checked in
> the browser.
>
> Known limits: the check verifies names, numbers and vocabulary, not sense —
> a garbled retelling of a note can pass; with the check on, the benchmark's
> synthesis category is 0% (comparisons it cannot verify are declined);
> bare fragments like "by how many points?" are not resolved; small talk
> still rambles; many concept answers are the curated note read back; the
> pattern rules that pick facts will have misfires of their own; standings
> are as of the last `build_rag.py --refresh`.

> **2026-10-11 update — revived after three months; multi-turn chat fixed.**
> The Hugging Face cache had been wiped, so the gated base model was gone and
> the app could not start. The model is back in the standard cache and both
> loaders are cache-first (`model_loader._cached_first`, and the MiniLM
> embedder in `rag/index.py`): once downloaded, nothing needs a token or the
> Hub. The audit that followed replayed WHOLE CHATS through the model instead
> of single questions, which is where the bugs were:
> - **Follow-ups were answered about the wrong entity.** Names recalled from
>   earlier answers were glued onto the search text. After a title answer,
>   "What teams did he race for?" carded Lewis Hamilton as a "Brazilian
>   driver… Died 1994"; the benchmark's own Vettel question, asked third in a
>   chat, came back as "2013 driver season"; a stated favourite hijacked every
>   later "he". References are now resolved into the question itself — "he" →
>   the driver just discussed, "that race" / "who won it" → that Grand Prix,
>   "And in 2007?" → the previous question for that year — for retrieval AND
>   for the model, which answers grounded turns without history. Generic
>   "they" / "it" are left alone.
> - **"GP" skipped the never-held guard** ("2007 Singapore GP" got the 2008
>   and 2024 results as context), and **the guard now answers for itself**
>   (`templates.absence_reply`): v4 predates it and turned its absence fact
>   into "not in my notes… What I've got here covers the 2007 Singapore Grand
>   Prix, nothing else".
> - **Relevance gate**: lexical overlap needs semantic support (or strong
>   similarity alone). Three docs at cosine 0.09 were being blended into the
>   DRS explainer on BM25 token coincidences.
> - **Chat vs. facts gate**: "I'm new to F1, where do I start?" and "Who is
>   your favourite driver?" go to the persona instead of being answered from
>   Liam Lawson's / Jim Clark's bio; "I also love the Monaco GP" counts as a
>   statement and reaches the radio log.
> - **Year-less title questions** about a driver with a curated titles
>   summary get that one fact ("How many titles did Lewis Hamilton win?" was
>   "the 2017, 2019 and 2017 World Champion").
>
> No retraining — shipped adapter is still v4. 67 tests (was 45). Re-measured
> on the final code (`*_oct11` reports; facts and retrieval with
> `HF_HUB_OFFLINE=1`):
> retrieval 100% hit / 99% clean, facts **99%**, grounding **73%**; every
> answer in both model benchmarks is identical to the run before these
> changes. Known limits, unchanged: free-chat replies ramble, yes/no and
> comparison questions are outside the trained formats, and multi-turn
> behaviour is covered by unit tests and replayed chats, not yet by a scored
> benchmark. Also corrected: the shipped adapter was trained on **2,445**
> rows (the README's 2,144 belonged to v1); `final_dataset.py` at HEAD builds
> the 2,628-row v5 set, so rebuilding from HEAD does not reproduce v4.

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
