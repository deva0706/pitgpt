# Phase 2 — Dataset (Teaching Notes)

## What "instruction tuning" actually is

A base language model is a **next-token predictor**: given text, it continues it.
Instruction tuning is supervised fine-tuning (SFT) on `(instruction, response)`
pairs so the model learns the *behaviour* "when given a request in this format,
produce a helpful answer in this style." We are not teaching it new facts from
scratch — SmolLM2-Instruct already follows instructions — we are **shaping** two
things:

1. **Domain focus** — answers should be about F1 race engineering.
2. **Voice** — the clipped, technical register of a race engineer.

The model updates its weights to make our target responses more probable given
our instructions. With QLoRA (Phase 3–4) only a tiny adapter is trained, but the
data requirements are the same: consistent, clean, in-voice examples.

## Why the Alpaca format

Each row is `{"instruction", "input", "output"}`. It's the de-facto standard for
SFT datasets, it maps cleanly onto a chat template (`instruction` → user turn,
`output` → assistant turn), and tooling (TRL's `SFTTrainer`) understands it. We
leave `input` empty because our tasks are single-turn questions with no extra
context document.

## Why response consistency matters (a lot)

The model imitates the *distribution* of your outputs. If half your answers are
terse and half are rambling, or the persona drifts between "hardcore engineer"
and "casual fan," the fine-tuned model learns an averaged, muddy voice. So every
answer in `knowledge_base.py` is written in the **same** register: calm, precise,
race-engineer phrasing, occasionally using "Copy that." without overusing it.
Consistency in *style* is what makes a small-model persona fine-tune succeed.

## How we generated ~1000 pairs honestly

We hand-authored the **knowledge** (36 concepts across strategy, tyres,
qualifying, telemetry, regulations, DRS/ERS, weather, driver comparison, plus 8
radio calls) and then **augmented instructions**: each concept is asked several
ways ("Explain X", "As my race engineer, x", "For a rookie driver, x", …). This
is standard paraphrase augmentation. The result:

- **1014** examples, **1014** unique instructions, **44** unique outputs.
- Mean output length ~94 words; nothing malformed; no exact duplicates.

### Why not have an LLM invent 1000 different answers?
Because a 360M-target fine-tune trained on *hallucinated* technical claims would
learn to sound confident and be wrong. By constraining outputs to 44 vetted,
accurate answers and varying only the phrasing of the questions, we guarantee the
model never learns a fabricated fact. **Known limitation:** with only 44 distinct
outputs the model may reproduce these answers closely rather than composing novel
ones — acceptable for a narrow persona demo, and the honest trade-off for
correctness. Scaling would mean authoring more concepts, not more paraphrases.

## Data hygiene: the two-file rule

- `data/raw/pitgpt_instructions.json` — generated once, treated as immutable.
- `data/processed/pitgpt_instructions.clean.json` — the validated output; the
  only file training reads.

`validate_dataset.py` is the gate between them. It (1) parses JSON, (2) checks
each row has non-empty string `instruction`/`output`, (3) drops malformed rows,
(4) removes exact duplicate `(instruction, output)` pairs, and (5) prints
statistics. Raw is never mutated, so validation is reproducible and auditable.

### Common dataset mistakes this avoids
- **Leaking duplicates** into train → inflated apparent performance.
- **Empty/whitespace outputs** → the model learns to emit nothing.
- **Mixed schemas** (some rows missing `input`) → collator errors at train time.
- **Editing raw in place** → you can never reproduce how the clean set was made.
