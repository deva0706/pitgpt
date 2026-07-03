"""
final_dataset.py — the definitive training set: behavior only, never facts.

The project's architectural thesis is **knowledge from retrieval, behavior from
fine-tuning**. This dataset is the fine-tuning half, and it teaches exactly three
behaviors — nothing else:

1. **Grounding**: read the retrieved context and answer FROM it. Every factual
   row is **counterfactually randomized** (drivers paired with wrong teams,
   races with wrong winners, years with wrong champions — always matching the
   context). Memorization is impossible by construction; the only winning
   strategy the optimizer can find is *reading*. Team lists are recited IN
   FULL — partial recitation was the measured failure mode at 47-67% accuracy.
2. **Reply formats** (user-specified, single source of truth in templates.py):
   "got the chequered flag" race calls, "Lights out and here we go" circuit
   intros, "And that's the title!" championship verdicts, "Checking the
   telemetry" driver cards, "The tally board reads" win counts, "Let's talk
   racecraft" explainers — all rendered FROM the context, never from memory.
3. **Honesty**: when the context doesn't cover the question — "Box, box" and
   decline. This behavior collapsed to 5% in the previous behavior-only
   adapter (85% before), so it gets a big, consistent block here.

Persona rows (no context) keep the enthusiast voice for opinion/chat.
Facts themselves live in the RAG corpus (rag/kb_builder.py + jolpica.py),
where recall is measured at 100% — not in these weights, where it was 23%.
"""

from __future__ import annotations

import random

from pitgpt.data import templates as T
from pitgpt.data.grounded_dataset import (
    DRIVERS,
    TEAMS,
    _comparison_examples,
    _detail_extraction_examples,
    _distractor_examples,
)

NATIONALITIES = ["British", "German", "Australian", "Spanish", "Dutch", "French",
                 "Finnish", "Brazilian", "Canadian", "Japanese", "Mexican", "Italian"]
GRANDS_PRIX = ["Monaco Grand Prix", "British Grand Prix", "Italian Grand Prix",
               "Belgian Grand Prix", "Japanese Grand Prix", "Brazilian Grand Prix",
               "Australian Grand Prix", "Spanish Grand Prix", "Canadian Grand Prix",
               "Austrian Grand Prix", "Hungarian Grand Prix", "Singapore Grand Prix"]
CIRCUITS = [("Circuit de Monaco", "Monte-Carlo, Monaco"),
            ("Silverstone Circuit", "Silverstone, UK"),
            ("Autodromo Nazionale di Monza", "Monza, Italy"),
            ("Circuit de Spa-Francorchamps", "Spa, Belgium"),
            ("Suzuka Circuit", "Suzuka, Japan"),
            ("Red Bull Ring", "Spielberg, Austria"),
            ("Circuit of the Americas", "Austin, USA"),
            ("Marina Bay Street Circuit", "Marina Bay, Singapore")]

# Reuse the racecraft explainers so the "Let's talk racecraft" format is
# taught grounded on the same rule texts that live in the retrieval corpus.
from pitgpt.data.fact_dataset import _TECH_EXPLAINERS  # noqa: E402


def _row(context: str, q: str, a: str) -> dict[str, str]:
    return {"context": context, "instruction": q, "input": "", "output": a}


def _driver_card_rows(rng: random.Random, n: int) -> list[dict]:
    """Context: a driver bio (randomized). Answer: telemetry card reciting the
    FULL team list from the context — the exact behavior the benchmark needs."""
    rows = []
    for _ in range(n):
        name = rng.choice(DRIVERS)
        nat = rng.choice(NATIONALITIES)
        y0 = rng.randint(1975, 2018)
        y1 = y0 + rng.randint(2, 12)
        teams = rng.sample(TEAMS, rng.randint(1, 5))
        wins = rng.choice([0, 0, 1, 3, 8, 21, 53])
        article = "an" if nat[0].lower() in "aeiou" else "a"
        ctx = (f"- {name} is {article} {nat} Formula 1 driver who raced from {y0} "
               f"to {y1} for {', '.join(teams)}.")
        if wins:
            ctx += f" {name} won {wins} Grand{'s Prix' if wins > 1 else ' Prix'}."
        a = T.driver_card_reply(name, nat, f"{y0}–{y1}", teams, wins)
        q = rng.choice([f"Who is {name}?", f"Tell me about {name}.",
                        f"Give me the rundown on {name}.",
                        f"What teams did {name} race for?",
                        f"What teams did {name} drive for in his career?",
                        f"Which teams has {name} raced for?"])
        rows.append(_row(ctx, q, a))
    return rows


def _season_driver_rows(rng: random.Random, n: int) -> list[dict]:
    """Context: a season entry (jolpica-style). Answer: current-team telemetry."""
    rows = []
    for _ in range(n):
        name, team = rng.choice(DRIVERS), rng.choice(TEAMS)
        year = rng.randint(2023, 2026)
        ctx = f"- In the {year} Formula 1 season, {name} raced for {team}."
        a = (f"Checking the telemetry on {name} — {year} season: {team}. "
             f"That's where he's parked the car this year.")
        q = rng.choice([f"Which team does {name} drive for?",
                        f"Who does {name} race for?",
                        f"Which team is {name} at in {year}?"])
        rows.append(_row(ctx, q, a))
    return rows


def _race_recap_rows(rng: random.Random, n: int) -> list[dict]:
    """Context: a race doc, sometimes with pole/podium. Answer: chequered-flag
    call carrying over exactly the details the context provides."""
    rows = []
    for _ in range(n):
        year = rng.randint(1970, 2026)
        gp = rng.choice(GRANDS_PRIX)
        winner, team = rng.choice(DRIVERS), rng.choice(TEAMS)
        ctx = f"- The {year} {gp} was won by {winner}, driving for {team}."
        pole, podium = "", []
        if rng.random() < 0.6:
            pole = rng.choice([d for d in DRIVERS if d != winner])
            ctx += f" Pole position: {pole}."
        if rng.random() < 0.6:
            podium = rng.sample([d for d in DRIVERS if d != winner], 2)
            ctx += f" Podium completed by {podium[0]} and {podium[1]}."
        a = T.race_winner_reply(winner, year, gp, team, pole, podium)
        q = rng.choice([f"Who won the {year} {gp}?",
                        f"Who took victory at the {year} {gp}?",
                        f"Who crossed the line first at the {year} {gp}?",
                        f"Who got the win at the {year} {gp}?"])
        rows.append(_row(ctx, q, a))
    return rows


def _champion_rows(rng: random.Random, n: int) -> list[dict]:
    rows = []
    details = ["what a season that was.", "sealed in style.",
               "a championship to remember.", "take a bow.", ""]
    for _ in range(n):
        year = rng.randint(1960, 2026)
        name = rng.choice(DRIVERS)
        ctx = (f"- In {year}, {name} won the Formula 1 World Drivers' Championship, "
               f"becoming the {year} F1 champion.")
        a = T.championship_reply(name, year, rng.choice(details))
        q = rng.choice([f"Who was F1 World Champion in {year}?",
                        f"Who won the {year} Formula 1 World Championship?",
                        f"Which driver took the F1 title in {year}?",
                        f"Who won the {year} driver championship?"])
        rows.append(_row(ctx, q, a))
    return rows


def _constructors_rows(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        year = rng.randint(1960, 2026)
        team = rng.choice(TEAMS)
        ctx = f"- In {year}, {team} won the Formula 1 Constructors' Championship."
        a = T.constructors_reply(team, year)
        q = rng.choice([f"Which team won the {year} Constructors' Championship?",
                        f"Who won the Constructors' title in {year}?"])
        rows.append(_row(ctx, q, a))
    return rows


def _circuit_rows(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        name, where = rng.choice(CIRCUITS)
        ctx = f"- {name} is a Formula 1 circuit located in {where}."
        a = T.circuit_reply(name, f"That's {where} — one for the bucket list.")
        q = rng.choice([f"Where is {name}?", f"Tell me about the {name}.",
                        f"What do you know about {name}?"])
        rows.append(_row(ctx, q, a))
    return rows


def _win_count_rows(rng: random.Random, n: int) -> list[dict]:
    """Context: a bio with a win tally. Answer: tally-board recitation of the
    exact number — win_count measured weakest-but-one at 50%."""
    rows = []
    for _ in range(n):
        name = rng.choice(DRIVERS)
        wins = rng.choice([1, 3, 8, 13, 21, 25, 32, 41, 53, 91, 103])
        y0 = rng.randint(1980, 2015)
        ctx = (f"- {name} is a Formula 1 driver who raced from {y0} to "
               f"{y0 + rng.randint(4, 14)}. {name} won {wins} Grands Prix.")
        a = T.win_count_reply(name, wins,
                              rng.choice(["Proper numbers.", "Not bad at all.",
                                          "Some career, that.", ""]))
        q = rng.choice([f"What's {name}'s career Grand Prix win tally?",
                        f"How many races did {name} win?",
                        f"How many Grand Prix wins does {name} have?"])
        rows.append(_row(ctx, q, a))
    return rows


def _racecraft_rows(rng: random.Random) -> list[dict]:
    rows = []
    for topic, expl in _TECH_EXPLAINERS.items():
        ctx = f"- {topic.capitalize()}: {expl}."
        a = T.racecraft_reply(topic, expl)
        for q in (f"Can you explain {topic}?", f"How does {topic} work?",
                  f"What is {topic}?"):
            rows.append(_row(ctx, q, a))
    return rows


def _decline_rows(rng: random.Random, n: int) -> list[dict]:
    """The honesty block: context covers X, question asks Y → "Box, box".
    One consistent, instantly recognizable opening (user-specified) instead of
    four scattered phrasings — the behavior that collapsed to 5% last time."""
    rows = []
    for _ in range(n):
        asked = rng.choice(DRIVERS)
        other = rng.choice([d for d in DRIVERS if d != asked])
        team = rng.choice(TEAMS)
        year = rng.randint(1990, 2026)
        gp = rng.choice(GRANDS_PRIX)
        kind = rng.random()
        if kind < 0.45:      # asked about a driver, context covers another
            ctx = rng.choice([
                f"- {other} currently races for {team}.",
                f"- {other} is a Formula 1 driver who raced for {team}.",
            ])
            q = rng.choice([f"Which team does {asked} drive for?",
                            f"Who does {asked} race for?",
                            f"What teams did {asked} race for?",
                            f"How many races did {asked} win?"])
            a = T.decline_reply(f"{other}, not {asked}")
        elif kind < 0.75:    # asked about a race/season the context doesn't have
            ctx = f"- The {year} {gp} was won by {other}, driving for {team}."
            other_gp = rng.choice([g for g in GRANDS_PRIX if g != gp])
            q = rng.choice([f"Who won the {year} {other_gp}?",
                            f"Who was on pole at the {year} {other_gp}?"])
            a = T.decline_reply(f"the {year} {gp}")
        else:                # context is a rule doc, question is factual
            topic, expl = rng.choice(list(_TECH_EXPLAINERS.items()))
            ctx = f"- {topic.capitalize()}: {expl}."
            q = rng.choice([f"Who won the {year} {gp}?",
                            f"Which team does {asked} drive for?",
                            f"Who was F1 champion in {year}?"])
            a = T.decline_reply(f"{topic}")
        rows.append(_row(ctx, q, a))
    return rows


def build_final_rows(seed: int = 42) -> list[dict[str, str]]:
    """Grounded-format + honesty + persona. Deterministic under ``seed``."""
    rng = random.Random(seed)
    rows: list[dict] = []
    # The user's formats, grounded and counterfactual:
    rows += _driver_card_rows(rng, 260)
    rows += _season_driver_rows(rng, 80)
    rows += _race_recap_rows(rng, 240)
    rows += _champion_rows(rng, 160)
    rows += _constructors_rows(rng, 80)
    rows += _circuit_rows(rng, 80)
    rows += _win_count_rows(rng, 120)
    rows += _racecraft_rows(rng)
    # Honesty (the collapsed behavior gets the biggest single block):
    rows += _decline_rows(rng, 240)
    rows += _distractor_examples(rng, 120)
    rows += _comparison_examples(rng, 100)
    rows += _detail_extraction_examples(rng, 100)

    # De-dup then shuffle.
    seen, unique = set(), []
    for r in rows:
        key = (r["context"], r["instruction"].lower(), r["output"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    rng.shuffle(unique)
    return unique
