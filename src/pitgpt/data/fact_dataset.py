"""
fact_dataset.py — closed-book fact QA with PER-CATEGORY response signatures.

Two jobs in one dataset (both user-mandated):

1. **Real facts** from the Kaggle "Formula 1 World Championship 1950-2024" dump
   (an Ergast database export, fetched via kagglehub): every driver's bio and
   team history, every Grand Prix winner, every Drivers' and Constructors'
   champion, and every circuit.

2. **Signature styles.** Each *category* of question opens with its own
   distinctive, consistent signature so the reply's shape signals the query type.
   Format conditioning is what fine-tuning learns best, so the signature is baked
   into every row of its category:

   ============  =============================================================
   driver         "🏎️ {Name} — {hook}."  then a compact driver-card
   race result    "Lights out, {year} {GP}:" then a race-recap sentence
   circuit        "Welcome to {circuit} — {tagline}." track-guide style
   champion       "Season {year} belonged to {Name}." season-verdict style
   constructors   "Constructors' crown, {year}: {Team}."
   technical      "Let's talk racecraft: ..." explainer style
   opinion/chat   (unchanged — the warm enthusiast persona rows)
   ============  =============================================================

Honest ceiling, as measured before: 1B weights memorize popular facts imperfectly
and the long tail stays shaky — that's benchmarked after training, not hidden.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

KAGGLE_DIR = Path(
    "C:/Users/devya/.cache/kagglehub/datasets/rohanrao/"
    "formula-1-world-championship-1950-2020/versions/24"
)


def _read(name: str) -> list[dict]:
    with (KAGGLE_DIR / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_tables() -> dict:
    return dict(
        drivers={r["driverId"]: r for r in _read("drivers.csv")},
        constructors={r["constructorId"]: r["name"] for r in _read("constructors.csv")},
        races={r["raceId"]: r for r in _read("races.csv")},
        circuits=_read("circuits.csv"),
        results=_read("results.csv"),
        standings=_read("driver_standings.csv"),
        c_standings=_read("constructor_standings.csv"),
    )


def _row(q: str, a: str) -> dict[str, str]:
    return {"context": "", "instruction": q, "input": "", "output": a}


def _name(d: dict) -> str:
    return f"{d['forename']} {d['surname']}".strip()


def _season_final_race_ids(t: dict) -> dict[str, int]:
    """raceId of each season's last round -> {raceId: year}."""
    last: dict[int, tuple[int, str]] = {}
    for race_id, race in t["races"].items():
        year, rnd = int(race["year"]), int(race["round"])
        if year not in last or rnd > last[year][0]:
            last[year] = (rnd, race_id)
    return {race_id: year for year, (_, race_id) in last.items()}


# --------------------------------------------------------------------------- #
# Category generators — each carries its signature.
# --------------------------------------------------------------------------- #

def _career_facts(t: dict) -> dict[str, dict]:
    facts: dict[str, dict] = {}
    for d_id, d in t["drivers"].items():
        facts[d_id] = {
            "name": _name(d), "nationality": d["nationality"],
            "years": set(), "teams": {}, "wins": 0,
        }
    for r in t["results"]:
        f = facts.get(r["driverId"])
        race = t["races"].get(r["raceId"])
        if not f or not race:
            continue
        year = int(race["year"])
        f["years"].add(year)
        team = t["constructors"].get(r["constructorId"])
        if team:
            f["teams"].setdefault(team, year)
            f["teams"][team] = min(f["teams"][team], year)
        if r.get("positionText") == "1":
            f["wins"] += 1
    return facts


def _driver_rows(t: dict, champ_years: dict, rng: random.Random) -> list[dict]:
    """Signature: '🏎️ {Name} — {hook}.' followed by a compact driver-card."""
    rows = []
    for f in _career_facts(t).values():
        if not f["years"]:
            continue
        y0, y1 = min(f["years"]), max(f["years"])
        span = f"{y0}" if y0 == y1 else f"{y0}–{y1}"
        teams = sorted(f["teams"], key=f["teams"].get)
        arrow = " → ".join(teams)
        titles = sorted(champ_years.get(f["name"], []))
        # the hook: champion > race winner > racer
        if titles:
            hook = f"{len(titles)}-time World Champion" if len(titles) > 1 else "World Champion"
        elif f["wins"]:
            hook = f"Grand Prix winner, {f['wins']} career win{'s' if f['wins'] > 1 else ''}"
        else:
            hook = f"{f['nationality']} racer of the {y0}s"
        card = (
            f"🏎️ {f['name']} — {hook}. {f['nationality']} driver, raced {span}"
            + (f", {arrow}" if arrow else "") + "."
        )
        if f["wins"]:
            card += f" {f['wins']} Grand Prix win{'s' if f['wins'] > 1 else ''}."
        if titles:
            card += f" Champion in {', '.join(map(str, titles))}."
        for q in (f"Who is {f['name']}?", f"Tell me about {f['name']}."):
            rows.append(_row(q, card))
        if arrow:
            rows.append(_row(
                f"Which teams did {f['name']} drive for in F1?",
                f"🏎️ {f['name']} — team history: {arrow}.",
            ))
        if f["wins"]:
            rows.append(_row(
                f"How many Grands Prix did {f['name']} win?",
                f"🏎️ {f['name']} — {f['wins']} Grand Prix win{'s' if f['wins'] > 1 else ''} in his career.",
            ))
    return rows


def _race_winner_rows(t: dict) -> list[dict]:
    """Signature: 'Lights out, {year} {GP}: ...' race-recap style."""
    rows = []
    for r in t["results"]:
        if r.get("positionText") != "1":
            continue
        race = t["races"].get(r["raceId"])
        driver = t["drivers"].get(r["driverId"])
        team = t["constructors"].get(r["constructorId"], "")
        if not race or not driver:
            continue
        year, gp = race["year"], race["name"]
        winner = _name(driver)
        recap = f"Lights out, {year} {gp}: {winner} took the chequered flag"
        recap += f" for {team}." if team else "."
        rows.append(_row(f"Who won the {year} {gp}?", recap))
    return rows


def _circuit_rows(t: dict) -> list[dict]:
    """Signature: 'Welcome to {circuit} — ...' track-guide style."""
    rows = []
    for c in t["circuits"]:
        name, loc, country = c["name"], c["location"], c["country"]
        guide = f"Welcome to {name} — {loc}, {country}."
        for q in (f"Where is {name}?", f"Tell me about the {name}."):
            rows.append(_row(q, guide))
    return rows


def _champion_rows(t: dict) -> tuple[list[dict], dict]:
    """Signature: 'Season {year} belonged to {Name}.' season-verdict style."""
    final_ids = _season_final_race_ids(t)
    champs: dict[int, str] = {}
    for s in t["standings"]:
        year = final_ids.get(s["raceId"])
        if year is not None and s["position"] == "1":
            d = t["drivers"].get(s["driverId"])
            if d:
                champs[year] = _name(d)
    rows = []
    champ_years = defaultdict(list)
    for year, name in sorted(champs.items()):
        champ_years[name].append(year)
        verdict = f"Season {year} belonged to {name} — World Champion."
        rows.append(_row(f"Who won the {year} Formula 1 World Championship?", verdict))
        rows.append(_row(f"Who was F1 World Champion in {year}?", verdict))
    for name, years in champ_years.items():
        ys = ", ".join(map(str, sorted(years)))
        n = len(years)
        rows.append(_row(
            f"How many World Championships did {name} win?",
            f"Season{'s' if n > 1 else ''} {ys} belonged to {name} — "
            f"{n} World Championship{'s' if n > 1 else ''} in total.",
        ))
    return rows, champ_years


def _constructor_champion_rows(t: dict) -> list[dict]:
    """Signature: 'Constructors' crown, {year}: {Team}.'"""
    final_ids = _season_final_race_ids(t)
    rows = []
    for s in t["c_standings"]:
        year = final_ids.get(s["raceId"])
        if year is not None and s["position"] == "1":
            team = t["constructors"].get(s["constructorId"])
            if team:
                line = f"Constructors' crown, {year}: {team}."
                rows.append(_row(f"Which team won the {year} Constructors' Championship?", line))
                rows.append(_row(f"Who won the Constructors' title in {year}?", line))
    return rows


_TECH_EXPLAINERS = {
    "DRS": "the Drag Reduction System opens a flap in the rear wing to cut drag and "
           "boost straight-line speed; it's only usable in set zones within one second "
           "of the car ahead, and it was introduced in 2011",
    "the undercut": "you pit before your rival, use fresh-tyre pace while they're still "
                    "on old rubber, and jump them when they finally stop",
    "the overcut": "you stay out longer than your rival, bank clean-air laps while "
                   "their new tyres warm up, and come out ahead after your own stop",
    "a Safety Car": "it bunches the field behind a physical car to neutralise the race, "
                    "which makes pit stops relatively cheaper and often scrambles strategy",
    "a Virtual Safety Car": "every driver must slow to a set delta time so gaps are "
                            "frozen — no physical car, no field bunching",
    "ERS": "the hybrid system harvests energy under braking and from exhaust heat, "
           "stores it in a battery, and redeploys it as extra power over the lap",
    "parc fermé": "from qualifying onward the car's setup is locked, so teams must "
                  "qualify and race with the same compromise setup",
    "tyre degradation": "tyres lose grip as they wear and overheat; managing that "
                        "drop-off decides stint lengths and the whole pit strategy",
    "dirty air": "the turbulent wake behind a car robs the following car of downforce, "
                 "making it slide, overheat its tyres, and struggle to pass",
    "the budget cap": "teams have a spending limit on car performance so the biggest "
                      "spenders can't simply buy dominance; it arrived in 2021",
    "ground effect": "the shaped underfloor accelerates air beneath the car to suck it "
                     "onto the track — downforce with less drag, reintroduced in 2022",
    "a sprint weekend": "a shorter race with its own qualifying is added to the weekend, "
                        "with points on offer and much less practice time",
}


def _technical_rows() -> list[dict]:
    """Signature: 'Let's talk racecraft: ...' explainer style."""
    rows = []
    for topic, explanation in _TECH_EXPLAINERS.items():
        answer = f"Let's talk racecraft: {topic} — {explanation}."
        for q in (f"Can you explain {topic}?", f"How does {topic} work?"):
            rows.append(_row(q, answer))
    return rows


def build_fact_rows(seed: int = 42) -> list[dict[str, str]]:
    rng = random.Random(seed)
    t = _load_tables()
    champion_rows, champ_years = _champion_rows(t)
    rows: list[dict[str, str]] = []
    rows += _driver_rows(t, champ_years, rng)
    rows += _race_winner_rows(t)
    rows += _circuit_rows(t)
    rows += champion_rows
    rows += _constructor_champion_rows(t)
    rows += _technical_rows()
    rng.shuffle(rows)
    return rows
