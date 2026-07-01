# Phase 2 — Interview Questions & Model Answers

### Q1. What is the difference between pre-training and instruction tuning?

**Answer.** Pre-training teaches a model general language and world knowledge by
predicting the next token over a massive, unlabeled corpus — it produces a *base*
model that can only continue text. Instruction tuning is a much smaller
supervised stage on curated `(instruction, response)` pairs that teaches the
model to *follow requests* and answer in a helpful format. Pre-training builds
the knowledge; instruction tuning aligns the behaviour. PitGPT does neither from
scratch — it takes an already-instruction-tuned SmolLM2 and does a further,
narrow SFT to specialise the domain and voice.

### Q2. Why does response *consistency* matter more than response *quantity*?

**Answer.** SFT makes the model imitate the distribution of your outputs. A
smaller set of consistent, in-voice, correct answers produces a cleaner learned
behaviour than a larger set with inconsistent tone or contradictory facts. Noise
in the targets directly becomes noise in the model — especially for a small model
with limited capacity to "average out" conflicting signals. Quality and
consistency dominate; quantity only helps once quality is held fixed.

### Q3. You have 1014 examples but only 44 unique outputs. Is that a problem?

**Answer.** It's a deliberate trade-off. Pairing 44 vetted answers with many
instruction paraphrases teaches robust instruction-*following* for the domain
without risking hallucinated facts, which is the right call when correctness
matters more than novelty. The risk is over-fitting: the model may reproduce the
canonical answers closely instead of composing new ones, and it sees less
linguistic variety on the output side. For a narrow persona demo that's
acceptable; to generalise further you'd author more distinct concepts (more
unique outputs), not simply more paraphrases of the same 44.

### Q4. Why keep `data/raw` immutable and write a separate `clean` file?

**Answer.** Reproducibility and auditability. If validation mutates the raw file
in place, you can never reconstruct how the clean set was derived, and a bad
cleaning step is unrecoverable. Treating raw as immutable and writing
`data/processed/*.clean.json` means the transformation is a pure function you can
re-run, diff, and trust. It's the same reason you don't edit source data in a
data pipeline — raw is the ground truth.

### Q5. What malformed cases must a dataset validator catch before SFT?

**Answer.** At minimum: non-parseable JSON; a root that isn't a list; rows that
aren't objects; missing required fields; fields of the wrong type; and
empty/whitespace-only `instruction` or `output` (which teach the model to emit
nothing). Beyond structure, you want to catch exact and near-duplicate pairs
(which leak between train/val and inflate metrics) and, ideally, length outliers.
Our validator handles JSON validity, schema, empties, and exact duplicates, and
reports length and uniqueness statistics so problems are visible.
