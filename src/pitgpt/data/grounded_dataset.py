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


def _comparison_examples(rng: random.Random, n: int) -> list[dict]:
    """Teach cross-fact synthesis with RANDOMIZED numbers — must actually compare.

    Counts are random per example, so 'who won more' can only be answered by
    reading and comparing the two facts, never from memory.
    """
    rows = []
    for _ in range(n):
        a, b = rng.sample(DRIVERS, 2)
        na, nb = rng.sample(range(1, 8), 2)          # distinct counts 1..7
        winner, wn = (a, na) if na > nb else (b, nb)
        facts = [f"- {a} has won {na} World Championships.",
                 f"- {b} has won {nb} World Championships."]
        rng.shuffle(facts)
        ctx = "\n".join(facts)
        q = rng.choice([
            f"Who has won more championships, {a} or {b}?",
            f"Of {a} and {b}, who has the most titles?",
            f"Who's more successful in championships, {a} or {b}?",
        ])
        answer = rng.choice([
            f"Going by the facts here, {winner} — {wn} championships to "
            f"{min(na, nb)}.",
            f"{winner} has more, with {wn} titles against {min(na, nb)}.",
        ]) + _flavour(rng)
        rows.append(_row(ctx, q, answer))

    # Property comparisons (which lasts longer / which is faster), randomized too.
    props = [
        ("tyre compound", "lasts longer", "wears out faster"),
        ("circuit", "has higher top speeds", "is slower and tighter"),
        ("car setup", "is quicker in corners", "is quicker on the straights"),
    ]
    names_pool = ["Option Alpha", "Option Beta", "the soft", "the hard",
                  "the medium", "Monza", "Monaco", "Suzuka", "Silverstone"]
    for _ in range(n // 2):
        kind, good, bad = rng.choice(props)
        x, y = rng.sample(names_pool, 2)
        facts = [f"- {x} {good}.", f"- {y} {bad}."]
        rng.shuffle(facts)
        q = f"Which {kind} {good}: {x} or {y}?"
        a = f"From the facts given: {x} {good}." + _flavour(rng)
        rows.append(_row("\n".join(facts), q, a))
    return rows


def _detail_extraction_examples(rng: random.Random, n: int) -> list[dict]:
    """Teach pulling a specific detail (year, number, condition) out of a fact.

    Years/numbers are randomized so the value can only come from the context.
    """
    rows = []
    templates = [
        ("{thing} was introduced in {year}.",
         "When was {thing} introduced?",
         "It was introduced in {year}."),
        ("{thing} was first used in Formula 1 in {year}.",
         "What year did {thing} first appear?",
         "{year} — that's when {thing} first appeared."),
        ("A race win is worth {num} points under this scoring system.",
         "How many points is a win worth here?",
         "A win is worth {num} points."),
        ("{thing} can only be used when a driver is within {num} second(s) of the car ahead.",
         "How close must a driver be to use {thing}?",
         "Within {num} second(s) of the car ahead — that's the activation window."),
        ("The pit lane speed limit at this circuit is {num} km/h.",
         "What's the pit lane speed limit?",
         "It's {num} km/h at this circuit."),
    ]
    things = ["DRS", "ERS", "the halo", "the cost cap", "ground effect",
              "the sprint format", "the virtual safety car"]
    for _ in range(n):
        fact_t, q_t, a_t = rng.choice(templates)
        thing = rng.choice(things)
        year = str(rng.choice(range(1994, 2025)))
        num = str(rng.choice([1, 2, 10, 20, 25, 60, 80, 100]))
        fmt = {"thing": thing, "year": year, "num": num}
        rows.append(_row(
            f"- {fact_t.format(**fmt)}",
            q_t.format(**fmt),
            a_t.format(**fmt) + _flavour(rng),
        ))
    return rows


def _wrong_entity_trap_examples(rng: random.Random, n: int) -> list[dict]:
    """Context names ANOTHER driver's team; the asked driver isn't covered.

    The trap is echoing the other driver's team. The answer must explicitly
    distinguish who the facts are about and decline to guess.
    """
    rows = []
    for _ in range(n):
        asked = rng.choice(DRIVERS)
        other = rng.choice([d for d in DRIVERS if d != asked])
        team = rng.choice(TEAMS)
        ctx = f"- {other} currently races for {team}."
        q = rng.choice([
            f"Which team does {asked} drive for?",
            f"Who does {asked} race for?",
        ])
        a = rng.choice([
            f"The facts I've got only cover {other} (who's at {team}) — they don't "
            f"say anything about {asked}, so I won't guess.",
            f"Careful one! What I can see is about {other}, not {asked}. I don't "
            f"have {asked}'s team in front of me, so I'd rather not make it up.",
        ])
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
    rows += _driver_team_examples(rng, 320)        # core grounding lesson
    rows += _distractor_examples(rng, 200)          # pick the right fact
    rows += _missing_examples(rng, 150)             # admit when context lacks it
    rows += _wrong_entity_trap_examples(rng, 180)   # don't echo the wrong entity's fact
    rows += _comparison_examples(rng, 160)          # synthesize across facts
    rows += _detail_extraction_examples(rng, 160)   # cite the year/number/condition
    rows += _circuit_examples(rng, 120)
    rows += _rule_examples(rng, 120)
    rows += _real_fact_examples(rng)                # true pairings from the corpus

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
