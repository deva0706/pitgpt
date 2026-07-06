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
    _detail_extraction_examples,
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
        # Realistic-context hardening: real retrieved bios of champions carry a
        # "won the World Drivers' Championship" sentence, and pinning often
        # adds a current-season doc. v3 showed those context words hijack the
        # template choice — train the card format to survive both.
        if wins and rng.random() < 0.4:
            ctx += (f" {name} won the World Drivers' Championship in "
                    f"{y0 + 1}, {y0 + 2}.")
        if rng.random() < 0.3:
            ctx += (f"\n- In the {y1} Formula 1 season, {name} raced for "
                    f"{teams[-1]}.")
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
    """Context: a race doc, sometimes with pole/podium/venue — matching the
    circuit-enriched corpus shape. Answer: chequered-flag call carrying over
    exactly the details the context provides.

    Question phrasings include the "tell me about" family (previously only
    winner-verb forms were trained — the phrasing users actually typed live
    was out of distribution) and, when the context names a venue, the
    circuit-as-race-name forms ("Who won at {circuit} in {year}?") — the exact
    mapping the measured Silverstone misrecitation missed."""
    rows = []
    for _ in range(n):
        year = rng.randint(1970, 2026)
        gp = rng.choice(GRANDS_PRIX)
        winner, team = rng.choice(DRIVERS), rng.choice(TEAMS)
        ctx = f"- The {year} {gp} was won by {winner}, driving for {team}."
        pole, podium, venue, circuit = "", [], "", None
        if rng.random() < 0.6:
            pole = rng.choice([d for d in DRIVERS if d != winner])
            ctx += f" Pole position: {pole}."
        if rng.random() < 0.6:
            podium = rng.sample([d for d in DRIVERS if d != winner], 2)
            ctx += f" Podium completed by {podium[0]} and {podium[1]}."
        if rng.random() < 0.6:
            circuit, where = rng.choice(CIRCUITS)   # counterfactual pairing
            venue = f"{circuit} in {where}"
            ctx += f" Held at {venue}."
        a = T.race_winner_reply(winner, year, gp, team, pole, podium, venue)
        qs = [f"Who won the {year} {gp}?",
              f"Who took victory at the {year} {gp}?",
              f"Who crossed the line first at the {year} {gp}?",
              f"Who got the win at the {year} {gp}?",
              f"Tell me about the {year} {gp}.",
              f"What happened at the {year} {gp}?",
              f"Give me the rundown on the {year} {gp}.",
              f"How was the {year} {gp}?"]
        if circuit:
            qs += [f"Who won at {circuit} in {year}?",
                   f"Who won the {year} race at {circuit}?"]
        rows.append(_row(ctx, rng.choice(qs), a))
    return rows


def _absence_rows(rng: random.Random, n: int) -> list[dict]:
    """Context: the absence guard's injected fact. Answer: recite the absence.

    The guard shipped after the last adapter was trained, so the model had
    never seen this context shape and improvised — measured live as a
    self-contradictory decline ("covers the 2020 Monaco Grand Prix, nothing
    else") and once as a confabulated 'festival' recap. Counterfactual by
    construction: random year×GP pairs, so only the recitation is learnable."""
    rows = []
    for _ in range(n):
        year = rng.randint(1950, 2026)
        gp = rng.choice(GRANDS_PRIX)
        ctx = (f"- There is no {year} {gp} in the historical race records — "
               f"that Grand Prix was not held in {year}.")
        a = T.absence_reply(year, gp)
        q = rng.choice([f"Who won the {year} {gp}?",
                        f"Tell me about the {year} {gp}.",
                        f"What happened at the {year} {gp}?",
                        f"Who was on pole at the {year} {gp}?",
                        f"Give me the rundown on the {year} {gp}.",
                        f"How was the {year} {gp}?"])
        rows.append(_row(ctx, q, a))
    return rows


def _prose_card_rows(rng: random.Random, n: int) -> list[dict]:
    """Context: curated-style star prose — titles first, teams buried
    mid-sentence — NOT the bio-doc shape. The card format must extract the
    team list from prose: measured live (diag_jul06) as Hamilton's card
    drifting into a persona ramble when only the curated doc was retrieved.
    Identity questions ("Who is X?") still route to curated docs, so the
    behavior needs to survive this context shape."""
    rows = []
    for _ in range(n):
        name = rng.choice(DRIVERS)
        titles = rng.choice([2, 3, 4, 5, 7])
        teams = rng.sample(TEAMS, rng.randint(2, 4))
        team_prose = (", ".join(teams[:-1]) + f" and {teams[-1]}"
                      if len(teams) > 1 else teams[0])
        ctx = (f"- {name} has won {titles} Formula 1 World Drivers' "
               f"Championships. He holds several records, and has driven "
               f"for {team_prose}.")
        if rng.random() < 0.5:
            year = rng.randint(2020, 2026)
            ctx += (f"\n- In the {year} Formula 1 season, {name} raced "
                    f"for {teams[-1]}.")
        a = T.driver_card_reply(name, "", "", teams, 0)
        a += f" {titles} World Championships."
        q = rng.choice([f"Who is {name}?",
                        f"Tell me about {name}.",
                        f"What teams did {name} race for?",
                        f"Give me the rundown on {name}."])
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
        # Real champions' bios mention titles; v3 showed that word pulled the
        # championship template onto tally questions. Train against it.
        if rng.random() < 0.5:
            ctx += (f" {name} won the World Drivers' Championship in "
                    f"{y0 + 2}.")
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


def _pick_from_context_rows(rng: random.Random, n: int) -> list[dict]:
    """Distractor discipline, restyled: the asked driver IS among 3 context
    facts — answer in the telemetry format. Forms a contrastive pair with
    _decline_rows (same question shape, asked driver ABSENT → "Box, box"),
    which is the discrimination the v2_formats benchmark showed was missing
    (mismatch cases answered in the distractor voice instead of declining)."""
    rows = []
    for _ in range(n):
        driver, team = rng.choice(DRIVERS), rng.choice(TEAMS)
        others = rng.sample([d for d in DRIVERS if d != driver], 2)
        other_teams = rng.sample([t for t in TEAMS if t != team], 2)
        facts = [f"- {driver} currently races for {team}.",
                 f"- {others[0]} currently races for {other_teams[0]}.",
                 f"- {others[1]} currently races for {other_teams[1]}."]
        rng.shuffle(facts)
        q = rng.choice([f"Which team does {driver} drive for?",
                        f"Who does {driver} race for?"])
        a = f"Checking the telemetry on {driver} — he's at {team}."
        rows.append(_row("\n".join(facts), q, a))
    return rows


def _comparison_rows(rng: random.Random, n: int) -> list[dict]:
    """Cross-fact synthesis with randomized counts. Opening restyled to
    "Stack them up" — its old "Going by the facts here" voice was bleeding
    into win-count answers (20% tally-format adherence)."""
    rows = []
    for _ in range(n):
        a, b = rng.sample(DRIVERS, 2)
        na, nb = rng.sample(range(1, 8), 2)
        winner, wn = (a, na) if na > nb else (b, nb)
        facts = [f"- {a} has won {na} World Championships.",
                 f"- {b} has won {nb} World Championships."]
        rng.shuffle(facts)
        q = rng.choice([f"Who has won more championships, {a} or {b}?",
                        f"Of {a} and {b}, who has the most titles?",
                        f"Who's more successful in championships, {a} or {b}?"])
        ans = (f"Stack them up: {winner} takes it — {wn} championships "
               f"to {min(na, nb)}.")
        rows.append(_row("\n".join(facts), q, ans))
    return rows


def _attribute_comparison_rows(rng: random.Random, n: int) -> list[dict]:
    """Non-numeric synthesis: two facts, question asks which has a property.
    v3 declined on these ("Monaco is not in my notes") because only counting
    comparisons were trained — the decline template over-fired on any
    two-entity question it hadn't seen."""
    pairs = [
        ("{a} is a slow, tight street circuit where overtaking is nearly impossible.",
         "{b} has long straights and a low-downforce, very high top-speed layout.",
         "Which of these two circuits has higher top speeds?",
         "Stack them up: {b} — long straights and a low-downforce layout mean far higher top speeds."),
        ("The soft tyre compound gives the most grip but wears out quickest.",
         "The hard compound lasts much longer but offers the least grip.",
         "Which compound lasts longer, soft or hard?",
         "Stack them up: the hard compound lasts longer — you trade grip for durability."),
        ("{a} joined Formula 1 in 1950 as one of the original championship rounds.",
         "{b} first appeared on the calendar in 2008.",
         "Which of these circuits is older?",
         "Stack them up: {a} — on the calendar since 1950, decades before {b}."),
    ]
    circuit_names = [c[0] for c in CIRCUITS]
    rows = []
    for _ in range(n):
        f1, f2, q, ans = rng.choice(pairs)
        a, b = rng.sample(circuit_names, 2)
        facts = [f"- {f1.format(a=a, b=b)}", f"- {f2.format(a=a, b=b)}"]
        rng.shuffle(facts)
        rows.append(_row("\n".join(facts), q,
                         ans.format(a=a, b=b)))
    return rows


def build_final_rows(seed: int = 42) -> list[dict[str, str]]:
    """Grounded-format + honesty + persona. Deterministic under ``seed``."""
    rng = random.Random(seed)
    rows: list[dict] = []
    # The user's formats, grounded and counterfactual:
    rows += _driver_card_rows(rng, 260)
    rows += _prose_card_rows(rng, 90)
    rows += _season_driver_rows(rng, 80)
    rows += _race_recap_rows(rng, 240)
    rows += _absence_rows(rng, 110)
    rows += _champion_rows(rng, 200)
    rows += _constructors_rows(rng, 80)
    rows += _circuit_rows(rng, 80)
    rows += _win_count_rows(rng, 180)
    rows += _racecraft_rows(rng)
    # Honesty: decline vs pick-from-context as a CONTRASTIVE pair on the same
    # question shape — presence of the asked entity decides which voice fires.
    rows += _decline_rows(rng, 240)
    rows += _pick_from_context_rows(rng, 120)
    rows += _comparison_rows(rng, 100)
    rows += _attribute_comparison_rows(rng, 60)
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
