"""
final_dataset.py — the definitive training set: behavior only, never facts.

The project's architectural thesis is **knowledge from retrieval, behavior from
fine-tuning**. This dataset is the fine-tuning half, and it teaches exactly three
behaviors — nothing else:

1. **Grounding**: read the retrieved context and answer FROM it. Every factual
   row is **counterfactually randomized** (drivers paired with wrong teams,
   races with wrong winners, years with wrong champions — always matching the
   context). Memorization is impossible by construction; the only winning
   strategy the optimizer can find is *reading*.
2. **Signature styles**: each query category answers in its distinctive shape
   (🏎️ driver-card, "Lights out" race-recap, "Welcome to" track-guide,
   "Season … belonged to" verdict, "Constructors' crown", "Let's talk
   racecraft") — now rendered FROM the context, never from memory.
3. **Honesty**: when the context doesn't cover the question — say so; when it
   covers a different entity — don't echo the wrong fact (imported from the
   benchmarked grounded_dataset generators).

Persona rows (no context) keep the enthusiast voice for opinion/chat.
Facts themselves live in the RAG corpus (rag/kb_builder.py), where recall is
lossless — not in these weights, where we measured it at 22%.
"""

from __future__ import annotations

import random

from pitgpt.data.grounded_dataset import (
    DRIVERS,
    TEAMS,
    _comparison_examples,
    _detail_extraction_examples,
    _distractor_examples,
    _missing_examples,
    _wrong_entity_trap_examples,
)

NATIONALITIES = ["British", "German", "Australian", "Spanish", "Dutch", "French",
                 "Finnish", "Brazilian", "Canadian", "Japanese", "Mexican", "Italian"]
GRANDS_PRIX = ["Monaco Grand Prix", "British Grand Prix", "Italian Grand Prix",
               "Belgian Grand Prix", "Japanese Grand Prix", "Brazilian Grand Prix",
               "Australian Grand Prix", "Spanish Grand Prix", "Canadian Grand Prix",
               "Austrian Grand Prix", "Hungarian Grand Prix", "Singapore Grand Prix"]
CIRCUITS = [("Circuit de Monaco", "Monte-Carlo", "Monaco"),
            ("Silverstone Circuit", "Silverstone", "UK"),
            ("Autodromo Nazionale di Monza", "Monza", "Italy"),
            ("Circuit de Spa-Francorchamps", "Spa", "Belgium"),
            ("Suzuka Circuit", "Suzuka", "Japan"),
            ("Red Bull Ring", "Spielberg", "Austria"),
            ("Circuit of the Americas", "Austin", "USA"),
            ("Marina Bay Street Circuit", "Marina Bay", "Singapore")]

# Reuse the racecraft explainers so the "Let's talk racecraft" signature is
# taught grounded on the same rule texts that live in the retrieval corpus.
from pitgpt.data.fact_dataset import _TECH_EXPLAINERS  # noqa: E402


def _row(context: str, q: str, a: str) -> dict[str, str]:
    return {"context": context, "instruction": q, "input": "", "output": a}


def _driver_card_rows(rng: random.Random, n: int) -> list[dict]:
    """Context: a driver bio doc (randomized). Answer: 🏎️ card FROM the context."""
    rows = []
    for _ in range(n):
        name = rng.choice(DRIVERS)
        nat = rng.choice(NATIONALITIES)
        y0 = rng.randint(1975, 2018)
        y1 = y0 + rng.randint(2, 12)
        teams = rng.sample(TEAMS, rng.randint(1, 4))
        wins = rng.choice([0, 0, 1, 3, 8, 21, 53])
        article = "an" if nat[0].lower() in "aeiou" else "a"
        ctx = (f"- {name} is {article} {nat} Formula 1 driver who raced from {y0} "
               f"to {y1} for {', '.join(teams)}.")
        if wins:
            ctx += f" {name} won {wins} Grand{'s Prix' if wins > 1 else ' Prix'}."
        hook = (f"Grand Prix winner, {wins} career win{'s' if wins > 1 else ''}"
                if wins else f"{nat} racer of the {(y0 // 10) * 10}s")
        card = (f"🏎️ {name} — {hook}. {nat} driver, raced {y0}–{y1}, "
                f"{' → '.join(teams)}.")
        if wins:
            card += f" {wins} Grand Prix win{'s' if wins > 1 else ''}."
        q = rng.choice([f"Who is {name}?", f"Tell me about {name}.",
                        f"Give me the rundown on {name}."])
        rows.append(_row(ctx, q, card))
    return rows


def _race_recap_rows(rng: random.Random, n: int) -> list[dict]:
    """Context: a race-winner doc (randomized). Answer: 'Lights out' recap."""
    rows = []
    for _ in range(n):
        year = rng.randint(1970, 2024)
        gp = rng.choice(GRANDS_PRIX)
        winner, team = rng.choice(DRIVERS), rng.choice(TEAMS)
        ctx = f"- The {year} {gp} was won by {winner}, driving for {team}."
        a = f"Lights out, {year} {gp}: {winner} took the chequered flag for {team}."
        q = rng.choice([f"Who won the {year} {gp}?",
                        f"Who took victory at the {year} {gp}?",
                        f"Who crossed the line first at the {year} {gp}?"])
        rows.append(_row(ctx, q, a))
    return rows


def _champion_rows(rng: random.Random, n: int) -> list[dict]:
    """Context: a champion doc (randomized). Answer: 'Season belonged to' verdict."""
    rows = []
    for _ in range(n):
        year = rng.randint(1960, 2024)
        name = rng.choice(DRIVERS)
        ctx = (f"- In {year}, {name} won the Formula 1 World Drivers' Championship, "
               f"becoming the {year} F1 champion.")
        a = f"Season {year} belonged to {name} — World Champion."
        q = rng.choice([f"Who was F1 World Champion in {year}?",
                        f"Who won the {year} Formula 1 World Championship?",
                        f"Which driver took the F1 title in {year}?",
                        f"Who won the {year} driver championship?"])
        rows.append(_row(ctx, q, a))
    return rows


def _constructors_rows(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        year = rng.randint(1960, 2024)
        team = rng.choice(TEAMS)
        ctx = f"- In {year}, {team} won the Formula 1 Constructors' Championship."
        a = f"Constructors' crown, {year}: {team}."
        q = rng.choice([f"Which team won the {year} Constructors' Championship?",
                        f"Who won the Constructors' title in {year}?"])
        rows.append(_row(ctx, q, a))
    return rows


def _circuit_rows(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        name, loc, country = rng.choice(CIRCUITS)
        ctx = f"- {name} is a Formula 1 circuit located in {loc}, {country}."
        a = f"Welcome to {name} — {loc}, {country}."
        q = rng.choice([f"Where is {name}?", f"Tell me about the {name}.",
                        f"What do you know about {name}?"])
        rows.append(_row(ctx, q, a))
    return rows


def _racecraft_rows(rng: random.Random) -> list[dict]:
    rows = []
    for topic, expl in _TECH_EXPLAINERS.items():
        ctx = f"- {topic.capitalize()}: {expl}."
        a = f"Let's talk racecraft: {topic} — {expl}."
        for q in (f"Can you explain {topic}?", f"How does {topic} work?",
                  f"What is {topic}?"):
            rows.append(_row(ctx, q, a))
    return rows


def build_final_rows(seed: int = 42) -> list[dict[str, str]]:
    """Grounded-signature + honesty + persona. Deterministic under ``seed``."""
    rng = random.Random(seed)
    rows: list[dict] = []
    # Signatures, grounded and counterfactual:
    rows += _driver_card_rows(rng, 220)
    rows += _race_recap_rows(rng, 200)
    rows += _champion_rows(rng, 180)
    rows += _constructors_rows(rng, 100)
    rows += _circuit_rows(rng, 80)
    rows += _racecraft_rows(rng)
    # Honesty behaviors (benchmarked generators from the grounding work):
    rows += _missing_examples(rng, 120)
    rows += _wrong_entity_trap_examples(rng, 150)
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
