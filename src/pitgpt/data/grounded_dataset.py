"""
grounded_dataset.py — synthetic training data that teaches GROUNDING, not memory.

The core problem: our persona adapter was trained on ``question -> answer`` pairs,
so it learned to answer from its own weights and ignore retrieved context. This
module builds the opposite lesson: ``(context + question) -> answer-from-context``.

The decisive trick is **randomised pairings**. If "Lando Norris" is paired with a
*different* team in different examples — always matching whatever the context
says — the model literally cannot memorise "Norris -> McLaren". It is forced to
*read the context* to answer. That is what installs grounding.

We generate four behaviours (Phase 4 of the redesign):
  * POSITIVE      — context contains the answer -> use it.
  * DISTRACTOR    — context has the answer plus unrelated facts -> pick the right one.
  * MISSING       — context lacks the answer -> admit it, don't guess.
  * REAL          — real curated facts, so the model also sees true pairings.

Answers keep a light enthusiast flavour but lead with the evidence — style
decorates, it never replaces the fact (Phase 5).

Each row is ``{"context", "instruction", "input": "", "output"}``; the training
data module renders it into the SAME prompt shape used at inference (context
folded into the user turn), guaranteeing train/serve parity.
"""

from __future__ import annotations

import random

# --------------------------------------------------------------------------- #
# Entity pools (real F1 names — used in varied, context-defined pairings).
# --------------------------------------------------------------------------- #
DRIVERS = [
    "Lando Norris", "Max Verstappen", "Charles Leclerc", "Lewis Hamilton",
    "George Russell", "Carlos Sainz", "Oscar Piastri", "Fernando Alonso",
    "Sergio Perez", "Lance Stroll", "Pierre Gasly", "Esteban Ocon",
    "Alexander Albon", "Yuki Tsunoda", "Nico Hulkenberg", "Valtteri Bottas",
]
TEAMS = [
    "McLaren", "Red Bull Racing", "Ferrari", "Mercedes", "Aston Martin",
    "Alpine", "Williams", "RB", "Haas", "Kick Sauber",
]
CIRCUITS = {
    "Monaco": "a tight, twisty street circuit where overtaking is nearly impossible and qualifying is everything",
    "Monza": "the 'Temple of Speed', all long straights and low downforce",
    "Spa-Francorchamps": "a long, fast, old-school track famous for the Eau Rouge sequence and changeable weather",
    "Silverstone": "a fast, flowing circuit that hosted the first Formula 1 World Championship race in 1950",
    "Suzuka": "a beloved figure-of-eight drivers' circuit with the challenging Esses",
}
RULES = {
    "DRS": "the Drag Reduction System — a rear-wing flap that cuts drag for a straight-line speed boost, usable only within one second of the car ahead in designated zones",
    "the undercut": "pitting earlier than a rival for fresh tyres, then using the extra grip to jump ahead when they stop",
    "a Safety Car": "a car that bunches the field to neutralise the race, which makes pit stops relatively cheaper",
    "the budget cap": "a limit on how much a team can spend on car performance, brought in to level the field",
}

# Light enthusiast decorations — appended AFTER the grounded fact.
FLAVOUR = [
    "Love that you asked!", "Mega question.", "Classic F1 detail, this one.",
    "Always fun to talk about.", "Good spot!", "One of my favourite bits of the sport.",
    "", "", "",  # blanks so plenty of answers are lean and fact-first
]

MISSING_DEFLECTIONS = [
    "Honestly, I don't have that in the notes in front of me, so I won't guess.",
    "I can't see that in the facts I've got here, and I'd rather not make it up.",
    "That specific detail isn't in what I've been given, so I'll be straight with you: I don't know from here.",
    "I don't have anything reliable on that in front of me right now, so I won't invent it.",
]


def _flavour(rng: random.Random) -> str:
    f = rng.choice(FLAVOUR)
    return f" {f}" if f else ""


def _row(context: str, question: str, answer: str) -> dict[str, str]:
    return {"context": context, "instruction": question, "input": "", "output": answer}


def _driver_team_examples(rng: random.Random, n: int) -> list[dict]:
    """Randomised driver->team pairings force reading the context, not memory."""
    rows = []
    for _ in range(n):
        driver, team = rng.choice(DRIVERS), rng.choice(TEAMS)
        ctx = f"- {driver} currently races for {team}."
        q = rng.choice([
            f"Which team does {driver} drive for?",
            f"Who does {driver} race for right now?",
            f"What team is {driver} at?",
        ])
        a = rng.choice([
            f"{driver} currently races for {team}.",
            f"That'd be {team} — {driver} is in that seat.",
            f"{driver} drives for {team} at the moment.",
        ]) + _flavour(rng)
        rows.append(_row(ctx, q, a))
    return rows


def _distractor_examples(rng: random.Random, n: int) -> list[dict]:
    """Answer is in the context alongside 2 unrelated facts — pick the right one."""
    rows = []
    for _ in range(n):
        driver, team = rng.choice(DRIVERS), rng.choice(TEAMS)
        others = rng.sample([d for d in DRIVERS if d != driver], 2)
        other_teams = rng.sample([t for t in TEAMS if t != team], 2)
        facts = [
            f"- {driver} currently races for {team}.",
            f"- {others[0]} currently races for {other_teams[0]}.",
            f"- {others[1]} currently races for {other_teams[1]}.",
        ]
        rng.shuffle(facts)
        ctx = "\n".join(facts)
        q = f"Which team does {driver} drive for?"
        a = rng.choice([
            f"{driver} races for {team}.",
            f"Looking at what I've got: {driver} is at {team}.",
        ]) + _flavour(rng)
        rows.append(_row(ctx, q, a))
    return rows


def _missing_examples(rng: random.Random, n: int) -> list[dict]:
    """Context is about something else — the model must admit it doesn't know."""
    rows = []
    for _ in range(n):
        driver = rng.choice(DRIVERS)
        # context is about a *different* driver / a rule, never the asked driver
        other, team = rng.choice([d for d in DRIVERS if d != driver]), rng.choice(TEAMS)
        ctx = rng.choice([
            f"- {other} currently races for {team}.",
            f"- {rng.choice(list(RULES))} is {rng.choice(list(RULES.values()))}.",
        ])
        q = rng.choice([
            f"Which team does {driver} drive for?",
            f"Who does {driver} race for?",
        ])
        a = rng.choice(MISSING_DEFLECTIONS) + rng.choice([
            f" If you can point me at {driver}'s current entry, I'll tell you straight away!",
            " Give me the up-to-date info and I'll confirm it instantly.",
            "",
        ])
        rows.append(_row(ctx, q, a))
    return rows


def _circuit_examples(rng: random.Random, n: int) -> list[dict]:
    rows = []
    items = list(CIRCUITS.items())
    for _ in range(n):
        circuit, trait = rng.choice(items)
        ctx = f"- {circuit} is {trait}."
        q = rng.choice([
            f"What's special about {circuit}?",
            f"Tell me about {circuit}.",
            f"Why do people talk about {circuit}?",
        ])
        a = rng.choice([
            f"{circuit} is {trait}.",
            f"So {circuit} is {trait} —",
        ]) + _flavour(rng)
        rows.append(_row(ctx, q, a))
    return rows


def _rule_examples(rng: random.Random, n: int) -> list[dict]:
    rows = []
    items = list(RULES.items())
    for _ in range(n):
        rule, expl = rng.choice(items)
        ctx = f"- {rule.capitalize()} is {expl}."
        q = rng.choice([
            f"Can you explain {rule}?",
            f"What is {rule}?",
            f"How does {rule} work?",
        ])
        a = rng.choice([
            f"{rule.capitalize()} is {expl}.",
            f"Sure — {rule} is {expl}.",
        ]) + _flavour(rng)
        rows.append(_row(ctx, q, a))
    return rows


def _real_fact_examples(rng: random.Random) -> list[dict]:
    """Ground on the real curated corpus so true pairings are also seen."""
    from pitgpt.rag.facts import CURATED_FACTS

    rows = []
    for fact in CURATED_FACTS:
        ctx = f"- {fact}"
        q = rng.choice([
            "What can you tell me about this?",
            "Explain this for me.",
            "Give me the story here.",
        ])
        # Answer restates the fact conversationally (grounded), plus flavour.
        a = fact + _flavour(rng)
        rows.append(_row(ctx, q, a))
    return rows


def build_grounded_rows(seed: int = 42) -> list[dict[str, str]]:
    """Assemble the full grounded dataset (deterministic under ``seed``)."""
    rng = random.Random(seed)
    rows: list[dict] = []
    rows += _driver_team_examples(rng, 320)   # core grounding lesson
    rows += _distractor_examples(rng, 200)     # pick the right fact
    rows += _missing_examples(rng, 200)        # admit when context lacks it
    rows += _circuit_examples(rng, 120)
    rows += _rule_examples(rng, 120)
    rows += _real_fact_examples(rng)           # true pairings from the corpus

    # De-duplicate on (context, question, output).
    seen, unique = set(), []
    for r in rows:
        key = (r["context"], r["instruction"].lower(), r["output"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    rng.shuffle(unique)
    return unique
