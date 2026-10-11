# PitGPT — LinkedIn showcase kit

Everything you need to assemble the post: the story, the numbers, the images,
the "why fine-tune?" study, and a demo-question script for screenshots.
(LinkedIn doesn't render markdown — copy the text you want and attach the PNGs
from this folder as images.)

---

## The one-paragraph story (post-ready)

> I built an F1 chatbot that runs — and *trains* — entirely on a 4 GB laptop
> GPU. The interesting part isn't the chatbot, it's the finding: a 1-billion-
> parameter model asked to memorize 75 years of Formula 1 topped out at **23%
> accuracy**. Instead of fighting that, I moved every fact into a retrieval
> system and taught the model only *behavior* — how to read, how to answer in
> broadcast-commentary formats, and how to say "I don't know." Same model, same
> laptop, same 90-question benchmark: **99% accuracy**, with confidently-wrong
> answers cut from 77% to 1%.

## The second story (post-ready): why fine-tune at all?

> The question everyone asks about my architecture diagram: "isn't that just a
> system prompt and a database? Why train anything?" Because at 1B parameters,
> **following instructions is itself the scarce skill**. My system prompt
> already said "answer using these facts, never invent, decline if unsure" —
> and the untuned model ignored it: it answered from its own fuzzy memory
> right past the correct fact sitting in its context window, and it was
> confidently wrong 77% of the time despite being *ordered* to be honest.
> Fine-tuning didn't teach the model any facts (the training data is
> counterfactually randomized, so memorizing is impossible by construction) —
> it taught the model to *obey*. A frontier model gives you that obedience
> from the prompt; at 1B on a 4 GB laptop, you install it yourself, with a
> 13M-parameter adapter.

## Base vs. fine-tuned — same prompt, same retrieval, only the adapter changes

Every row is a failure documented in the repo (phase notes + committed
benchmark reports) before the behavior was trained in:

| Behavior | Base Gemma-3-1B + system prompt + RAG | + 13M-param behavior adapter |
|---|---|---|
| Reads the retrieved facts | ignores them — answers from parametric memory past the correct fact in context (~0% context-following on entity questions) | **100%** entity grounding — recites what it read |
| Honesty when facts are missing | confidently wrong **77%** of the time, despite the prompt ordering it to decline | **1%** confident-wrong; declines with "Box, box — that one's not in my notes" |
| Reply formats (race call, title call, telemetry card…) | prompt-instructed formats drift and garble at 1B scale | **98%** adherence across the benchmark |
| Voice | generic assistant | race-engineer persona, F1 community language |
| Prompt cost of all the above | few-shot examples would eat the 384-token window on every single call | **zero extra tokens — it's in the weights** |

The one-line version: **the database decides what is true; the adapter decides
how the model behaves about it** — reads instead of recalls, declines instead
of invents, and speaks in one voice. The weights carry zero facts.

## Attach these images

| File | What it shows |
|---|---|
| `benchmark_journey.png` | Accuracy 23→61→78→97→**99%** and confident-wrong 77→32→12→2→**1%** across the five architecture stages |
| `architecture.png` | The full pipeline: question → hybrid retrieval → grounded prompt → 1B model + behavior adapter → formatted answer, over its four supply lines (chat memory, knowledge base, honesty guard, behavior dataset) |
| `chat_trap_question.png` | Live app declining the 2007 Singapore GP — a race that never happened |
| `chat_vettel_memory.png` | Career telemetry card, then "when did *he* win…" resolved across turns |
| `category_scorecard.png` | Per-category tiles: champions 100%, GP winners 100%, win counts 100%, team histories 97%, formats 98% |
| `training_terminal.png` | A real fine-tune run on the laptop (the 2,628-example retrain): 13M trainable params (1.97%), ~2 GB VRAM |

## Demo script — the questions that show it off

Screenshot these in order. Each hits a measured strength:

1. **"Who won the 2019 Chinese Grand Prix?"** — race-call format with real
   details: *"Lewis Hamilton got the chequered flag… Pole position: Valtteri
   Bottas. Podium completed by…"* [GP winners: 100%]
2. **"Who won the 2008 Singapore Grand Prix?"** — the famous Crashgate race:
   Alonso, Renault. A crowd-pleaser for F1 followers. [100%]
3. **"Which driver took the F1 title in 1992?"** — *"And that's the title!
   Nigel Mansell is the 1992 World Champion…"* [Champions: 100%]
4. **"What teams did Sebastian Vettel race for over his F1 career?"** —
   telemetry-card format reciting the full career list. [Team histories: 97%]
5. **"Who won the 2021 Yas Marina Grand Prix?"** — venue-alias resolution:
   answers with the 2021 *Abu Dhabi* GP (Verstappen). The corpus bridges
   circuit names to official race names.
6. **"Who won the 2007 Singapore Grand Prix?"** — **the money shot.** There
   was no 2007 Singapore GP, and the bot says *"Box, box — there's no 2007
   Singapore Grand Prix on the record: that race wasn't held in 2007"*
   instead of inventing a race. (That reply comes from the retrieval guard,
   not the model — see "Honesty guard" below.) An AI that declines a trap
   question is rarer than one that answers well.
7. **Memory flow** — *"My favourite driver is Max Verstappen, ok?"* → a few
   fact questions → *"Who is my favorite driver?"* → *"Checking the radio
   log — your favourite driver on record is Max Verstappen. Checking the
   telemetry on…"* Statement recall survives any chat length. Follow-ups
   work in the same chat too: after a title answer, *"What teams did he race
   for?"* cards that driver; after question 2, *"Who won it in 2009?"* then
   *"And in 2007?"* ends on the trap answer.
8. **"Explain DRS like I'm five"** — *"Let's talk racecraft…"* rules explainer.
9. **"Senna or Prost — who was better?"** — opinion mode: the enthusiast
   persona, no fixed template.

Avoid live on stream: circuit-name races where the winner wasn't on pole
("who won at Silverstone in 2018") — the 1B model can shuffle names when the
asked name differs from the official one; "who won more titles, X or Y"
comparisons; and win-tally phrasings other than "career Grand Prix win tally".
All three are documented residuals queued for the next adapter iteration.

---

## How it works (for the technically curious)

**Thesis: knowledge from retrieval, behavior from fine-tuning.**

1. **Knowledge** lives in 2,381 atomic fact documents, one self-contained fact
   each. Retrieval is hybrid — MiniLM dense embeddings + BM25 keyword scores,
   fused — with query routing and *entity pinning*: if you name a driver, a
   year, or a Grand Prix, only documents about that entity reach the model.
   Race docs carry their circuit name, so "Yas Marina GP" finds the Abu Dhabi
   result. Retrieval recall on the benchmark: 100%.
2. **Behavior** lives in a ~13M-parameter QLoRA adapter on Gemma-3-1B
   (4-bit NF4). It's trained on **counterfactually randomized** data — drivers
   paired with wrong teams, races with wrong winners — so memorizing facts is
   *impossible by construction*; the only strategy that reduces training loss
   is reading the provided context. The adapter carries the persona, eight
   reply formats, and the "Box, box" decline.
3. **Honesty guard**: the knowledge base contains every championship round
   since 1950, so a year+GP query is verified against the *whole corpus* —
   if retrieval missed the doc, the guard recovers it; if no doc exists, the
   race provably never happened and the guard says so itself in a fixed
   reply, so the model never gets the chance to hallucinate a neighboring
   year's result.
4. **Chat memory**: a deterministic "radio log" quotes the user's own
   statements back ("who is my favourite driver?"), and pronoun follow-ups
   resolve through labeled recall context — quoted, never imitated.

**Training on 4 GB:** batch 1, gradient accumulation 16, sequence 384,
gradient checkpointing, paged 8-bit AdamW — peak ~3.1 GB VRAM, ~80 minutes for
2 epochs, checkpoint every 10 steps (it survived a literal system crash
mid-run).

## The data

| Source | What it provides |
|---|---|
| Kaggle F1 World Championship dump (Ergast export) | every race, driver, team, champion 1950–2024 |
| Jolpica API (Ergast successor) | 2024–26 seasons: results, driver–team pairs, champions (cached locally, offline after one download) |
| OpenF1 API | current grid snapshot |
| Hand-curated corpus | rules, technology, circuits, history (~60 evergreen docs) |
| **Synthetic behavior dataset** | counterfactual rows teaching formats + grounding + declines — generated by `src/pitgpt/data/final_dataset.py`, zero real facts |

## The numbers (all reports committed in the repo)

| Metric | Facts-in-weights | Final system |
|---|---|---|
| Overall accuracy (90 held-out questions, strict) | 23% | **99%** |
| Champions by year | 30% | **100%** |
| GP winners | 45% | **100%** |
| Career win counts | 20% | **100%** |
| Driver team histories | 3% | **97%** |
| Confident-wrong answers | 77% | **1%** |
| Reply-format adherence | — | **98%** |

**Honest limitations** (also in the README): recitation can shuffle names when
a race is asked by its circuit name and the winner wasn't on pole; multi-fact
comparisons ("who won more titles?") are the weakest category; the synthetic
decline benchmark sits at 73% — improving one of these tends to trade against
the others at 1B scale, which is itself a finding.

## Post angles that land

- **The counterintuitive finding**: "I measured my model failing (23%) and
  redesigned around the failure instead of scaling up."
- **The why-fine-tune answer**: "My system prompt already said 'be honest and
  use the facts.' The 1B model ignored it 77% of the time. Fine-tuning is how
  you install instruction-following when the model is too small to come with
  it."
- **The trap question**: the screenshot — "my chatbot refused to answer a race
  that never happened. Here's the retrieval trick behind it."
- **The constraint**: "everything — training included — on a 4 GB laptop GPU.
  No cloud, no API keys, fully offline."
- **The engineering war story**: a system crash killed training mid-run;
  checkpoint-every-10-steps + resume support finished it with minutes lost.
