"""
fact_dataset.py — closed-book fact QA generated from the REAL historical record.

With RAG disabled, facts must live in the adapter weights. This module turns the
Kaggle "Formula 1 World Championship 1950-2024" dataset (an Ergast database dump,
downloaded via kagglehub) into question/answer training pairs about *real* F1:
every driver's bio, teams, wins, and every season's World Champion.

Honest expectations (measured previously): a 1B model memorizes high-frequency
facts imperfectly — repetition helps popular entities, the long tail stays shaky.
We therefore generate MULTIPLE phrasings per fact (repetition aids recall) and
keep answers short and factual (less room to confabulate).

Source files (stdlib csv; no pandas dependency):
  drivers.csv           driverId, forename, surname, dob, nationality
  results.csv           raceId, driverId, constructorId, positionText
  races.csv             raceId, year, round, name
  constructors.csv      constructorId, name
  driver_standings.csv  raceId, driverId, position  (season-final -> champion)
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

FLAVOUR = ["", "", "", " What a career!", " Proper legend of the grid.",
           " One for the history books.", " Mega, right?"]


def _read(name: str) -> list[dict]:
    with (KAGGLE_DIR / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_tables() -> dict:
    drivers = {r["driverId"]: r for r in _read("drivers.csv")}
    constructors = {r["constructorId"]: r["name"] for r in _read("constructors.csv")}
    races = {r["raceId"]: r for r in _read("races.csv")}
    results = _read("results.csv")
    standings = _read("driver_standings.csv")
    return dict(drivers=drivers, constructors=constructors, races=races,
                results=results, standings=standings)


def _career_facts(t: dict) -> dict[str, dict]:
    """Aggregate per-driver: name, nationality, dob, seasons, teams, wins."""
    facts: dict[str, dict] = {}
    for d_id, d in t["drivers"].items():
        facts[d_id] = {
            "name": f"{d['forename']} {d['surname']}".strip(),
            "nationality": d["nationality"],
            "dob": d["dob"] if d["dob"] not in ("\\N", "") else None,
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
            # remember the first year we saw each team, for chronological order
            f["teams"].setdefault(team, year)
            f["teams"][team] = min(f["teams"][team], year)
        if r.get("positionText") == "1":
            f["wins"] += 1
    return facts


def _champions(t: dict) -> dict[int, str]:
    """Season -> champion name, from the standings after each season's last round."""
    last_race: dict[int, tuple[int, str]] = {}   # year -> (round, raceId)
    for race_id, race in t["races"].items():
        year, rnd = int(race["year"]), int(race["round"])
        if year not in last_race or rnd > last_race[year][0]:
            last_race[year] = (rnd, race_id)
    final_ids = {race_id: year for year, (_, race_id) in last_race.items()}
    champs: dict[int, str] = {}
    for s in t["standings"]:
        year = final_ids.get(s["raceId"])
        if year is not None and s["position"] == "1":
            d = t["drivers"].get(s["driverId"])
            if d:
                champs[year] = f"{d['forename']} {d['surname']}".strip()
    return champs


def _row(q: str, a: str) -> dict[str, str]:
    return {"context": "", "instruction": q, "input": "", "output": a}


def build_fact_rows(seed: int = 42) -> list[dict[str, str]]:
    rng = random.Random(seed)
    t = _load_tables()
    facts = _career_facts(t)
    champs = _champions(t)
    champ_years = defaultdict(list)
    for year, name in champs.items():
        champ_years[name].append(year)

    rows: list[dict[str, str]] = []

    # ---- Per-driver bios & teams (only drivers who actually raced) ----
    for f in facts.values():
        if not f["years"]:
            continue
        y0, y1 = min(f["years"]), max(f["years"])
        span = f"in {y0}" if y0 == y1 else f"from {y0} to {y1}"
        teams = sorted(f["teams"], key=f["teams"].get)
        teams_txt = ", ".join(teams[:-1]) + f" and {teams[-1]}" if len(teams) > 1 else (teams[0] if teams else "")
        titles = sorted(champ_years.get(f["name"], []))
        article = "an" if f["nationality"][:1].lower() in "aeiou" else "a"
        bio = f"{f['name']} is {article} {f['nationality']} Formula 1 driver who raced {span}"
        if teams_txt:
            bio += f", driving for {teams_txt}"
        bio += "."
        if f["wins"]:
            bio += f" He won {f['wins']} Grand{'s' if f['wins'] > 1 else ''} Prix." \
                if f["wins"] > 1 else " He won 1 Grand Prix."
        if titles:
            ys = ", ".join(map(str, titles))
            bio += f" He won the World Championship in {ys}."
        for q in (f"Who is {f['name']}?", f"Tell me about {f['name']}."):
            rows.append(_row(q, bio + rng.choice(FLAVOUR)))
        if teams_txt:
            rows.append(_row(
                f"Which teams did {f['name']} drive for in F1?",
                f"{f['name']} drove for {teams_txt}." + rng.choice(FLAVOUR),
            ))
        if f["wins"] >= 1:
            rows.append(_row(
                f"How many Grands Prix did {f['name']} win?",
                f"{f['name']} won {f['wins']} Grand{'s Prix' if f['wins'] > 1 else ' Prix'}.",
            ))

    # ---- Champions by season, both directions ----
    for year, name in sorted(champs.items()):
        rows.append(_row(f"Who won the {year} Formula 1 World Championship?",
                         f"{name} won the {year} World Championship."))
        rows.append(_row(f"Who was F1 World Champion in {year}?",
                         f"The {year} champion was {name}."))
    for name, years in champ_years.items():
        ys = ", ".join(map(str, sorted(years)))
        n = len(years)
        rows.append(_row(
            f"How many World Championships did {name} win?",
            f"{name} won {n} World Championship{'s' if n > 1 else ''} ({ys}).",
        ))

    rng.shuffle(rows)
    return rows
