"""
grounding_eval.py — Phase 9: a measurable benchmark for RAG grounding.

Why this exists
---------------
Until now, grounding was judged on a handful of anecdotes. This harness turns it
into numbers across 100+ prompts so improvement is *demonstrated*, not asserted.

What makes the measurement trustworthy
--------------------------------------
The entity questions use **synthetic randomized pairings** (e.g. "Fernando Alonso
races for Williams") injected as gold context. Because those pairings are made up
per-prompt, the model *cannot* be right from memory — a correct answer is proof it
read the context. Conversely, naming a different team from the pool is a caught
hallucination, not a lucky guess.

Categories
----------
* ``grounded_entity``      — context states driver→team; answer must contain the team.
* ``grounded_descriptive`` — context contains a specific detail (year, count,
                             condition); answer must contain that detail.
* ``synthesis``            — several facts; answer requires comparing them.
* ``missing_context``      — context is irrelevant; success = honest admission.
* ``persona``              — no context, opinion question; success = engaged,
                             non-refusing answer (no false "I don't know").

Scoring is deterministic substring matching — simple, reproducible, and blunt on
purpose (an LLM-judge would add noise and cost). Results stream to a JSONL file so
an interrupted run resumes instead of restarting.

Run:  ``python scripts/eval_grounding.py [--limit N] [--tag NAME]``
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from pitgpt.config import settings

logger = logging.getLogger("pitgpt.eval.grounding")

# Match the retriever's production context header so eval == serve formatting.
_CTX_HEADER = (
    "Verified F1 facts (rely on these for any names, numbers, teams or results; "
    "don't contradict them or invent specifics):\n"
)

ADMISSION_PHRASES = (
    "don't know", "do not know", "don't have", "do not have", "can't see",
    "cannot see", "not in the facts", "isn't in", "is not in", "won't guess",
    "not sure from", "don't have that",
    # the trained decline template (templates.py)
    "box, box", "not in my notes",
)

DRIVERS = [
    "Lando Norris", "Max Verstappen", "Charles Leclerc", "Lewis Hamilton",
    "George Russell", "Carlos Sainz", "Oscar Piastri", "Fernando Alonso",
    "Pierre Gasly", "Alexander Albon", "Yuki Tsunoda", "Nico Hulkenberg",
]
TEAMS = [
    "McLaren", "Red Bull Racing", "Ferrari", "Mercedes", "Aston Martin",
    "Alpine", "Williams", "Haas", "Kick Sauber", "RB",
]

# (fact, question, gold-substring) triples for detail extraction.
DESCRIPTIVE_ITEMS = [
    ("DRS (Drag Reduction System) is a rear-wing flap that cuts drag to aid overtaking; it was introduced in 2011.",
     "When was DRS introduced?", "2011"),
    ("The Formula 1 World Championship began in 1950 at Silverstone.",
     "When did the Formula 1 World Championship begin?", "1950"),
    ("F1 moved to turbo-hybrid power units in 2014.",
     "When did F1 switch to turbo-hybrid engines?", "2014"),
    ("The cost cap limiting team spending was introduced in 2021.",
     "When was the budget cap introduced?", "2021"),
    ("A race win is worth 25 points in the current scoring system.",
     "How many points does a race winner score?", "25"),
    ("Ground effect floors returned to Formula 1 with the 2022 regulations.",
     "When did ground effect return to F1?", "2022"),
    ("DRS can only be used when a driver is within one second of the car ahead.",
     "How close must a driver be to use DRS?", "one second"),
    ("Ayrton Senna died in a crash at Imola in 1994.",
     "In which year did Ayrton Senna die?", "1994"),
    ("Nico Rosberg won the 2016 World Championship and retired days later.",
     "Which championship did Nico Rosberg win?", "2016"),
    ("Kimi Raikkonen won the 2007 World Championship with Ferrari.",
     "Which team was Kimi Raikkonen with when he won his title?", "Ferrari"),
]

# (facts, question, gold) triples requiring cross-fact synthesis.
SYNTHESIS_ITEMS = [
    (["Michael Schumacher won seven World Championships.",
      "Juan Manuel Fangio won five World Championships.",
      "Alain Prost won four World Championships."],
     "Of these drivers, who won the most World Championships?", "Schumacher"),
    (["Sebastian Vettel won four World Championships.",
      "Ayrton Senna won three World Championships."],
     "Who won more titles, Vettel or Senna?", "Vettel"),
    (["Monza is known for very high top speeds and long straights.",
      "Monaco is a slow, tight street circuit where overtaking is nearly impossible."],
     "Which of these two circuits has higher top speeds?", "Monza"),
    (["Lewis Hamilton has won seven World Championships.",
      "Fernando Alonso has won two World Championships."],
     "Who has more championships, Hamilton or Alonso?", "Hamilton"),
    (["The soft tyre gives the most grip but wears fastest.",
      "The hard tyre lasts longest but has the least grip."],
     "Which compound lasts longer, soft or hard?", "hard"),
]

PERSONA_QUESTIONS = [
    "Is dominance boring or impressive in F1?",
    "What do you love most about Formula 1?",
    "Street circuits or classic tracks — which are better?",
    "Would you rather have a wet race every weekend or never?",
    "What makes a race unforgettable for you?",
    "Should F1 bring back refuelling?",
    "Is qualifying pace or racecraft more important?",
    "Why do fans get so emotional about this sport?",
    "What's the most exciting part of a race weekend?",
    "Do you enjoy strategy battles or wheel-to-wheel fights more?",
]


@dataclass
class EvalCase:
    category: str
    question: str
    context: str          # "" for persona
    gold: str             # substring that must appear; "" for behavioural checks
    wrong_pool: list[str] = field(default_factory=list)  # for hallucination detection


def _ctx(facts: list[str]) -> str:
    return _CTX_HEADER + "\n".join(f"- {f}" for f in facts)


def build_cases(seed: int = 7) -> list[EvalCase]:
    """~105 deterministic cases across the five categories."""
    rng = random.Random(seed)
    cases: list[EvalCase] = []

    # 40 entity-grounding (randomized pairings -> grounding is provable).
    for _ in range(40):
        driver, team = rng.choice(DRIVERS), rng.choice(TEAMS)
        cases.append(EvalCase(
            category="grounded_entity",
            question=rng.choice([
                f"Which team does {driver} drive for?",
                f"Who does {driver} race for?",
            ]),
            context=_ctx([f"{driver} currently races for {team}."]),
            gold=team,
            wrong_pool=[t for t in TEAMS if t != team],
        ))

    # 20 descriptive detail extraction (each item twice with phrasing variety).
    for fact, question, gold in DESCRIPTIVE_ITEMS:
        for q in (question, f"Quick one — {question[0].lower()}{question[1:]}"):
            cases.append(EvalCase("grounded_descriptive", q, _ctx([fact]), gold))

    # 15 synthesis (each item three times).
    for facts, question, gold in SYNTHESIS_ITEMS:
        for _ in range(3):
            shuffled = facts[:]
            rng.shuffle(shuffled)
            cases.append(EvalCase("synthesis", question, _ctx(shuffled), gold))

    # 20 missing-context honesty (asked about a driver, given an unrelated fact).
    for _ in range(20):
        driver = rng.choice(DRIVERS)
        other = rng.choice([d for d in DRIVERS if d != driver])
        cases.append(EvalCase(
            category="missing_context",
            question=f"Which team does {driver} drive for?",
            context=_ctx([f"{other} currently races for {rng.choice(TEAMS)}."]),
            gold="",
        ))

    # 10 persona.
    for q in PERSONA_QUESTIONS:
        cases.append(EvalCase("persona", q, "", ""))

    return cases


def score_case(case: EvalCase, answer: str) -> dict:
    """Deterministic scoring; returns pass/fail plus hallucination flag."""
    low = answer.lower()
    admitted = any(p in low for p in ADMISSION_PHRASES)

    if case.category in ("grounded_entity", "grounded_descriptive", "synthesis"):
        passed = case.gold.lower() in low
        hallucinated = (
            not passed
            and not admitted
            and any(w.lower() in low for w in case.wrong_pool)
        )
        return {"passed": passed, "hallucinated": hallucinated, "admitted": admitted}

    if case.category == "missing_context":
        # Success = honest admission; naming ANY team is a hallucination.
        named_team = any(t.lower() in low for t in TEAMS)
        return {"passed": admitted and not named_team,
                "hallucinated": named_team, "admitted": admitted}

    # persona: engaged answer, no spurious refusal.
    return {"passed": len(answer) > 80 and not admitted,
            "hallucinated": False, "admitted": admitted}


def run_eval(tag: str = "current", limit: int | None = None) -> Path:
    """Run all cases against the configured model+adapter; write JSONL + report."""
    from pitgpt.inference.engine import generate_reply
    from pitgpt.inference.model_loader import load_for_inference

    out_dir = settings.paths.evaluation
    jsonl_path = out_dir / f"grounding_results_{tag}.jsonl"
    done: set[int] = set()
    if jsonl_path.exists():                       # resume support
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["idx"])
            except Exception:  # noqa: BLE001
                pass

    cases = build_cases()
    if limit:
        cases = cases[:limit]

    model, tokenizer = load_for_inference(
        adapter_dir=settings.model.adapter_dir, quantized=True
    )
    logger.info("Evaluating %d cases (tag=%s, %d already done) ...",
                len(cases), tag, len(done))

    with jsonl_path.open("a", encoding="utf-8") as fh:
        for idx, case in enumerate(cases):
            if idx in done:
                continue
            answer = generate_reply(
                model, tokenizer, case.question,
                use_rag=False, context=case.context,
            )
            record = {
                "idx": idx, "category": case.category, "question": case.question,
                "gold": case.gold, "answer": answer, **score_case(case, answer),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            if (idx + 1) % 10 == 0:
                logger.info("  %d/%d done", idx + 1, len(cases))

    return write_report(jsonl_path, tag)


def write_report(jsonl_path: Path, tag: str) -> Path:
    """Aggregate JSONL results into a markdown scorecard."""
    rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines()]
    cats = sorted({r["category"] for r in rows})
    lines = [
        f"# PitGPT Grounding Benchmark — `{tag}`", "",
        f"Model: `{settings.model.base_model_id}` + adapter `{settings.model.adapter_dir.name}`",
        f"Cases: {len(rows)}", "",
        "| Category | Cases | Pass | Accuracy | Hallucinations |",
        "|---|---|---|---|---|",
    ]
    total_pass = total_hall = 0
    for cat in cats:
        sub = [r for r in rows if r["category"] == cat]
        p = sum(r["passed"] for r in sub)
        h = sum(r["hallucinated"] for r in sub)
        total_pass += p
        total_hall += h
        lines.append(f"| {cat} | {len(sub)} | {p} | {p/len(sub):.0%} | {h} |")
    lines += [
        f"| **overall** | {len(rows)} | {total_pass} | {total_pass/len(rows):.0%} | {total_hall} |",
        "",
        f"**Hallucination rate:** {total_hall/len(rows):.1%}",
        "",
        "## Sample failures", "",
    ]
    for r in [r for r in rows if not r["passed"]][:8]:
        lines.append(f"- **[{r['category']}]** {r['question']} → gold `{r['gold']}` → "
                     f"_{r['answer'][:140]}…_")
    report_path = jsonl_path.with_name(f"grounding_report_{tag}.md")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", report_path)
    return report_path
