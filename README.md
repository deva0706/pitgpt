# 🏎️ PitGPT — a local F1 assistant built on one principle

> **Knowledge comes from retrieval. Behavior comes from fine-tuning.**
> A fully-local Formula 1 chatbot — Gemma-3-1B + QLoRA + hybrid RAG — that runs
> and *trains* on a 4 GB laptop GPU, with the benchmarks to justify every
> architectural decision.

PitGPT answers like an enthusiastic F1 friend, in **eight distinctive
per-category reply formats**, grounding every fact in a **2,381-document
knowledge base** covering 1950 through the current season — because we
*measured* that a 1B model storing facts in its weights tops out at **23%
accuracy**, and redesigned the system around that finding instead of
fighting it.

**The one-line story:** we tried facts-in-weights, benchmarked the failure,
moved knowledge to retrieval, retrained the adapter for *behavior only*
(persona + grounding + formats, with counterfactual data that makes
memorization impossible), then iterated retrieval precision and data realism
against the benchmark — and the same 90-case benchmark went **23% → 97%
strict accuracy, with confident-wrong answers cut from 77% → 2%**.

---

## Table of contents
- [The thesis](#the-thesis-and-the-evidence) · [Architecture](#architecture) · [Features](#features)
- [Folder structure](#folder-structure) · [Installation](#installation)
- [The training data](#the-training-data-behavior-not-facts) · [QLoRA explained](#qlora-explained)
- [Training](#training) · [Running the app](#running-the-app) · [Screenshots](#screenshots)
- [Evaluation](#evaluation) · [Future work](#future-work) · [License](#license)

---

## The thesis, and the evidence

Small language models are excellent at *behavior* and terrible at being
*databases*. This project demonstrates that split empirically, on one fixed
90-case benchmark (held-out phrasings, gold answers from the historical record):

| Architecture | Overall (strict) | Champions | Confident-wrong |
|---|---|---|---|
| Facts fine-tuned into weights (closed-book) | 23% | 30% | 77% |
| + hybrid RAG, behavior-only adapter | 61% | 93% | 32% |
| + entity pinning (context precision) | 78% | 100% | — |
| **+ format adapter, realistic-context training, greedy decoding** | **97%** | **100%** | **2%** |

Same base model, same GPU, same benchmark — knowledge moved to retrieval, then
retrieval precision and training-data realism were iterated against the
measurement. Behavior-in-weights works: the eight reply formats hold at **98%**
end-to-end and entity grounding with gold context is 100%. The honest residual:
"admit when the context doesn't cover it" sits at 20-70% depending on training
mix (see [Evaluation](#evaluation)) — production exposure is limited because
entity pinning filters mismatched context before the model sees it.

- **Base model:** google/gemma-3-1b-it (gated — one-time `huggingface-cli login`)
- **Adapter:** ~13 M trainable params (~2%) — persona, grounding, styles; zero facts
- **Knowledge:** 2,270 atomic fact docs (Kaggle 1950-2024 + curated rules + OpenF1)
- **Retrieval:** MiniLM dense + BM25, reciprocal-rank fusion, query-aware routing
- **Hardware:** RTX 3050 Laptop, 4 GB VRAM, Windows — batch 1×16, seq 384,
  gradient checkpointing, paged 8-bit AdamW, bf16

## Motivation

Large models are impressive but inaccessible to fine-tune on consumer hardware.
This project shows the *whole* pipeline — environment, data, theory, training,
evaluation, and a UI — at a scale that fits a **4 GB laptop GPU**, while explaining
every engineering decision (see [`docs/`](docs/)). It doubles as an interview-prep
resource: each phase ships teaching notes and Q&A.

## Features

- 🔍 **Hybrid RAG** over 2,270 atomic fact docs: MiniLM dense + BM25 lexical,
  fused by reciprocal rank, with query-aware source routing (driver questions →
  bio docs, race questions → result docs) that fixed a measured
  popular-entity ranking failure.
- 🎨 **Six per-category reply signatures** learned in the adapter — 🏎️
  driver-cards, *"Lights out, {year} {GP}:"* recaps, *"Welcome to {circuit}"*,
  *"Season {year} belonged to…"*, *"Constructors' crown…"*, *"Let's talk
  racecraft:"* — plus a warm enthusiast persona for opinion questions.
- 🧠 **Behavior-only QLoRA** (~13 M params) trained on counterfactually
  randomized data, so grounding on retrieved context is the only strategy the
  optimizer can learn — memorization is impossible by construction.
- 📏 **Two benchmark harnesses** (grounding + factual A/B), resumable JSONL
  runs, markdown scorecards — every claim in this README has a number behind it.
- 💬 **Streaming Gradio chat**, custom "night race" theme, multi-turn history.
- 🧩 **Modular `src/` package**, single config source of truth, 20 tests,
  phase-by-phase teaching docs with interview Q&A.

## Architecture

One principle, two supply lines into the prompt:

```
                       KNOWLEDGE (retrieval)                 BEHAVIOR (weights)
  Kaggle 1950-2024 ─► kb_builder ─► 2,270 fact docs      counterfactual dataset
  OpenF1 (live grid) ──────────────► hybrid index ─┐      (signatures+grounding
  curated rules ───────────────────► (dense+BM25   │       +persona, 0 facts)
                                      +RRF+routing)│              │
                                                   ▼              ▼
   user question ─► query router ─► top-k facts ─► prompt ─► Gemma-3-1B 4-bit
                                                            + LoRA adapter (2%)
                                                                   │
                                                    grounded, in-voice answer
```

```mermaid
flowchart LR
    KGL[Kaggle 1950-2024] --> KB[kb_builder: 2270 atomic docs]
    OF1[OpenF1 live] --> KB
    CUR[curated rules] --> KB
    KB --> IDX[hybrid index: dense+BM25+RRF]
    Q[user question] --> RT[query router]
    RT --> IDX
    IDX --> P[prompt builder]
    FD[final_dataset: counterfactual behavior rows] --> TR[QLoRA SFTTrainer]
    TR --> AD[adapter ~13M params]
    P --> G[Gemma-3-1B 4-bit + adapter]
    AD --> G
    G --> A[grounded, signature-styled answer]
```

## Folder structure

```
PitGPT/
├── src/pitgpt/
│   ├── config/         # settings.py — single source of truth (paths, model, LoRA, training)
│   ├── data/           # knowledge_base, build_dataset, validate_dataset
│   ├── inference/      # model_loader, engine (streaming), baseline capture
│   ├── training/       # data_module, train (QLoRA SFTTrainer)
│   ├── evaluation/     # prompts (20), compare (base vs tuned), reports
│   └── app/            # model_service, theme, gradio_app
├── scripts/            # thin CLI entry points + check_gpu.py
├── tests/              # pytest suite
├── docs/               # phaseN_notes.md + interview_phaseN.md (teaching)
├── configs/  data/  models/adapters/  notebooks/
├── pyproject.toml  requirements.txt  LICENSE  .gitignore  README.md
```

## Installation

```bash
# 1. Create & activate an environment (Python 3.11)
conda create -n pitgpt python=3.11 -y
conda activate pitgpt

# 2. (GPU) install the CUDA build of torch FIRST
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies + the package (editable)
pip install -r requirements.txt
pip install -e .

# 4. Verify the GPU / CUDA / bitsandbytes gate
python scripts/check_gpu.py
```

> **Windows + bitsandbytes:** modern versions (≥0.43) ship Windows wheels and 4-bit
> training works; this project was built and trained on Windows 11 with an RTX 3050.
> If bitsandbytes misbehaves, train in WSL2/Colab and run inference on Windows, or
> drop to plain LoRA (no 4-bit).

## The training data: behavior, not facts

The final training set (`final_dataset.py` + `build_final.py`, 2,144 rows)
teaches exactly three behaviors — and is engineered so facts *cannot* be learned:

1. **Grounding** — every factual row is **counterfactually randomized**: drivers
   paired with wrong teams, races with wrong winners, always matching the
   provided context. A model can't memorize randomized pairings, so gradient
   descent's only winning strategy is *reading the context*. (This single trick
   took context-following from 0% to 100% on entity questions.)
2. **Signature styles** — each category's reply shape, rendered *from* the
   context, never from memory.
3. **Honesty** — missing-context → say so; context about a different entity →
   don't echo the wrong fact.

Plus **127 hand-written enthusiast conversations** (`enthusiast_kb.py`) for the
opinion/chat persona. Facts live exclusively in the retrieval corpus
(`rag/kb_builder.py`), where recall is lossless.

```bash
python -m pitgpt.data.build_final    # behavior dataset → data/raw + validation
python scripts/build_rag.py          # knowledge corpus → hybrid index
```

## QLoRA explained

Freeze Gemma-3-1B and store it in **4-bit NF4** (cheap memory). Add small low-rank
**LoRA** matrices (`r=16`, `alpha=32`) to the attention (`q,k,v,o_proj`) and MLP
(`gate,up,down_proj`) projections and train **only** those. Fit inside 4 GB with
batch-size tricks, **gradient checkpointing**, and a **paged 8-bit optimizer**.
Save only the ~tens-of-MB adapter. Full walkthrough (transformers, attention,
quantization, NF4, rank/alpha/dropout, gradient accumulation, target modules) in
[`docs/phase3_notes.md`](docs/phase3_notes.md).

## Training

```bash
python scripts/train.py      # QLoRA fine-tune → models/adapters/pitgpt-lora
```

Sized for 4 GB: 4-bit NF4 base, bf16 compute (Ampere-native, no grad scaler),
`max_length=384`, **batch 1 × 16 accumulation** (effective 16), 2 epochs. Gemma-1B
at higher batch/seq thrashed the paged optimizer near 4 GB, so batch 1 keeps ~1 GB
headroom. The trainer logs VRAM and trainable params, checkpoints every 25 steps,
and supports `PITGPT_RESUME=auto`. Details in [`docs/phase4_notes.md`](docs/phase4_notes.md).

## Running the app

```bash
python scripts/chat.py       # launches the Gradio UI at http://127.0.0.1:7860
```

Streaming replies, multi-turn memory, example "quick calls", and a reset button.
If the adapter is missing, the app runs on the base model and says so in the
header (graceful fallback). Architecture in [`docs/phase6_notes.md`](docs/phase6_notes.md).

## Screenshots

> _Placeholder — add a screenshot of the running Gradio app here._
>
> ```
> [ 🏎️ PitGPT — F1 Enthusiast ]
> You:   I'm new to F1, where do I start?
> PitGPT: Welcome — this is one of my favourite things to talk about! There's no
>         wrong way to jump in; pick a driver or team that hooks you and go from there…
> ```

## The retrieval stack

**Corpus** (`rag/kb_builder.py`): 2,270 **atomic fact documents** — one
self-contained fact each (858 driver careers, ~1,125 Grand Prix winners, every
Drivers' and Constructors' champion, 77 circuits, curated rules, the OpenF1
live grid). Atomic docs eliminate chunking entirely and keep prompt injection
compact (4 facts ≈ 120 tokens).

**Hybrid ranking** (`rag/index.py`): MiniLM dense embeddings catch meaning;
BM25 catches exact tokens — "2016" and "2021" embed almost identically, which
is precisely how dense-only retrieval confused champion years. Both rankings
fuse via **reciprocal rank fusion** (K=60) with a relevance gate.

**Query routing** (`rag/retriever.py`): a measured failure — popular drivers
have ~90 race-winner docs that flooded ranking and buried their bio doc (the
benchmark's popularity gradient *inverted*: Schumacher scored worse than
backmarkers). Routing question types to doc families (driver → bios, "who won
the {year} GP" → race docs) fixed it: popular-tier accuracy 5% → 30%.

```bash
python scripts/build_rag.py               # rebuild corpus + hybrid index
python scripts/eval_grounding.py          # 105-case grounding benchmark
python scripts/eval_facts.py --tag cb     # 90-case factual A/B: closed-book
python scripts/eval_facts.py --tag rag --rag   #                 vs end-to-end
```

## Evaluation

**The headline A/B** — identical 90 cases (held-out phrasings, gold from the
historical tables), one flag between architectures:

| Metric | Facts-in-weights | **Final shipped system** |
|---|---|---|
| Overall (strict, word-boundary) | 23% | **97%** |
| Champions by year | 30% | **100%** |
| Driver team histories | 3% | **93%** |
| GP winners | 45% | **100%** |
| Career win counts | 20% | **90%** |
| Confident-wrong answers | 77% | **2%** |
| Reply-format adherence | — | **98%** |

The final system = hybrid RAG + entity pinning (query names a driver/year/GP →
only matching docs reach the prompt) + a format adapter trained on
*realistic-shaped* counterfactual contexts + greedy decoding. Each of those
four levers was added because a benchmark run isolated its failure mode; the
per-iteration reports (`v2*`, `v3`, `v4`) are all committed.

Scoring uses word-boundary matching (a plain-substring scorer flagged phantom
hallucinations — the Kaggle constructor table contains a team named "RE", which
substring-matches inside "raced"). Team-name aliases ("Lotus F1" vs "Team
Lotus") still make strict scoring conservative on the RAG side.

**Grounding benchmark** (105 cases; counterfactual pairings, so a pass *proves*
context-reading), on the shipped adapter: 72% overall — entity grounding 100%,
persona 100%, descriptive 90%, but missing-context honesty at 20% and synthesis
at 27%. Three training iterations exposed a **decline/answer seesaw**: mixes
that maximize honest declines (70% in iteration 2) cost facts accuracy, and the
realistic-context mix that won facts+formats traded declines back down. The
force-fed mismatched contexts in this benchmark are largely filtered by entity
pinning in the production path, but the residual is reported as measured.
Every report is committed under `src/pitgpt/evaluation/`.

**The journey, as measurements** (each pivot was driven by a number):

| Experiment | Result | Decision |
|---|---|---|
| 50 epochs on style data | output collapsed ("system system…") | early stopping, ~1-2 epochs |
| facts fine-tuned into 360M → 1B | fluent hallucination | bigger base helps style, not facts |
| persona adapter + naive RAG | retrieved facts ignored | grounding must be *trained* |
| counterfactual grounding data | context-following 0% → 100% | behavior-in-weights works |
| closed-book facts, 5,084 rows | 23% strict, 77% confident-wrong | knowledge moves to retrieval |
| hybrid RAG + routing + behavior adapter | 61% strict, 32% confident-wrong | retrieval works, precision doesn't |
| retrieval-only harness | 100% recall, 7% clean contexts | the gap is precision, not search |
| entity pinning + enriched race docs | 78% strict, zero training | clean context first, then train |
| format adapter (clean contexts) | 97% facts but formats keyed to context words | train on realistic doc shapes |
| realistic-context training + greedy | **97% strict, 2% confident-wrong, 98% formats** | the shipped system |

## Future work

- **Break the decline/answer seesaw** (honest declines 20-70% across training
  mixes): scale decline rows *with* realistic contexts and weight them
  relative to the answer rows; it's the one gate still under 90%.
- Synthesis/comparison answers under greedy decoding (27%) — likely needs
  chain-style comparison rows rather than more templates.
- Alias normalization in scoring and corpus (Lotus/Team Lotus/Lotus F1).
- Multi-turn grounded dialogues; GGUF export for CPU inference.

## License

MIT (this project's code and adapters) — see [`LICENSE`](LICENSE). The base model
google/gemma-3-1b-it is licensed by Google (Gemma terms). This repo does not
redistribute the base weights.

---

*A learning-focused, production-structured portfolio project. Every phase is
documented in [`docs/`](docs/) with both teaching notes and interview questions.*
