# Phase 5 — Interview Questions & Model Answers

### Q1. Why not just report a single accuracy number for a fine-tuned chatbot?

**Answer.** For open-ended generation there's no single ground-truth string, so
token-overlap metrics (BLEU/ROUGE) and "accuracy" poorly reflect quality — a
correct answer phrased differently scores badly, and a fluent wrong answer can
score well. For a persona/domain SFT the meaningful axes are factual correctness,
relevance, register/voice, and coherence, which are best judged by structured
side-by-side comparison (and, at scale, human or LLM-as-judge rating on a rubric).
A number would give false precision.

### Q2. Why keep decoding settings identical between base and fine-tuned?

**Answer.** To isolate the variable you're testing. If the base uses greedy
decoding and the fine-tuned uses sampling, any difference could be the decoder,
not the weights. Holding temperature, top-p, and max tokens constant means the
only thing that changed is the adapter, so the comparison actually measures
fine-tuning's effect.

### Q3. Why load the two models sequentially instead of together?

**Answer.** Memory. On a 4 GB GPU, two model instances plus an adapter can
exceed VRAM and OOM, and there's no benefit since we generate from one at a time.
Loading base, generating all answers, freeing it (`del` + `gc.collect()` +
`empty_cache()`), then loading the adapter version keeps peak memory to roughly
one model at a time.

### Q4. Your dataset had only 44 unique outputs. How does that bias the evaluation?

**Answer.** The 20 eval prompts overlap the training *topics*, so the fine-tuned
model is being tested partly in-distribution — it will look strong on those,
which is fair for demonstrating the persona but doesn't prove out-of-distribution
generalisation. An honest report says so and ideally adds a few *held-out* prompts
phrased unlike any training instruction to probe generalisation. We flag this as a
limitation rather than overclaiming.

### Q5. What would you add to make this evaluation more rigorous?

**Answer.** A held-out test split never seen in training; multiple seeds per
prompt to average out sampling variance; a written rubric scored by an independent
judge (human or a stronger LLM) on correctness/relevance/voice; and a few
adversarial prompts (off-topic or trick questions) to check the persona holds and
the model refuses to hallucinate specifics. For factuality specifically, a
checklist of known-correct F1 facts to verify against would catch regressions.
