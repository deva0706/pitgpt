"""
kb_builder.py — the full F1 knowledge base, generated from the historical record.

This is where "knowledge comes from retrieval" becomes real: instead of asking a
1B model to memorize 75 years of Formula 1 (measured: 22% strict accuracy), we
turn the Kaggle 1950-2024 dump (an Ergast database export) into ~2,200 **atomic
fact documents** — one self-contained fact per document.

Why atomic docs (a deliberate design decision):
* Retrieval returns exactly the fact needed — no chunking, no chunk-overlap
  tuning, no partial-fact truncation. The "chunk size" question disappears.
* Entity names and years appear at the start of each doc, which serves both
  dense embeddings (salient tokens) and BM25 (exact lexical match on
  "2016", "Latifi", "Suzuka").
* Small docs keep the prompt injection compact: 4 retrieved facts ≈ 120 tokens.

Doc families (source tag → count):
  kaggle_driver        ~858   career bios: nationality, span, teams, wins, titles
  kaggle_race          ~1125  every Grand Prix winner with team
  kaggle_champion      ~75    Drivers' champion per season
  kaggle_constructors  ~68    Constructors' champion per season
  kaggle_circuit       ~77    circuit locations
plus the curated evergreen rules (facts.py) and the OpenF1 current grid.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from pitgpt.data.fact_dataset import (
    _career_facts,
    _load_tables,
    _name,
    _season_final_race_ids,
)

logger = logging.getLogger("pitgpt.rag.kb")


def _doc(text: str, source: str) -> dict[str, str]:
    return {"text": text, "source": source}


def driver_docs(t: dict, champ_years: dict[str, list[int]]) -> list[dict]:
    docs = []
    for f in _career_facts(t).values():
        if not f["years"]:
            continue
        y0, y1 = min(f["years"]), max(f["years"])
        span = f"in {y0}" if y0 == y1 else f"from {y0} to {y1}"
        teams = sorted(f["teams"], key=f["teams"].get)
        article = "an" if f["nationality"][:1].lower() in "aeiou" else "a"
        text = f"{f['name']} is {article} {f['nationality']} Formula 1 driver who raced {span}"
        if teams:
            text += f" for {', '.join(teams)}"
        text += "."
        if f["wins"]:
            text += f" {f['name']} won {f['wins']} Grand{'s Prix' if f['wins'] > 1 else ' Prix'}."
        titles = sorted(champ_years.get(f["name"], []))
        if titles:
            text += (f" {f['name']} won the World Drivers' Championship in "
                     f"{', '.join(map(str, titles))}.")
        docs.append(_doc(text, "kaggle_driver"))
    return docs


def race_winner_docs(t: dict) -> list[dict]:
    """One doc per Grand Prix: winner + team, pole sitter, and podium — the
    detail lines the reply formats need ("...Pole to flag... podium completed
    by...") come straight from retrieval, not model memory."""
    by_race: dict[str, list[dict]] = defaultdict(list)
    for r in t["results"]:
        by_race[r["raceId"]].append(r)
    # Circuit names bridge the venue-alias gap: users say "Yas Marina Grand
    # Prix" for the Abu Dhabi GP, and without the circuit in the race doc no
    # retrieval signal (BM25, pin stem, absence-guard scan) can ever connect
    # the two — measured live as a false "not held" decline.
    circuits_by_id = {c["circuitId"]: c for c in t["circuits"]}
    docs = []
    for race_id, rs in by_race.items():
        race = t["races"].get(race_id)
        if not race:
            continue
        podium: dict[str, tuple[str, str]] = {}
        pole = None
        for r in rs:
            d = t["drivers"].get(r["driverId"])
            if not d:
                continue
            if r.get("positionText") in ("1", "2", "3"):
                podium[r["positionText"]] = (
                    _name(d), t["constructors"].get(r["constructorId"], ""))
            if r.get("grid") == "1":
                pole = _name(d)
        if "1" not in podium:
            continue
        winner, team = podium["1"]
        text = f"The {race['year']} {race['name']} was won by {winner}"
        text += f", driving for {team}." if team else "."
        if pole:
            text += f" Pole position: {pole}."
        rest = [podium[p][0] for p in ("2", "3") if p in podium]
        if rest:
            text += f" Podium completed by {' and '.join(rest)}."
        circ = circuits_by_id.get(race.get("circuitId"))
        if circ:
            text += (f" Held at {circ['name']} in {circ['location']}, "
                     f"{circ['country']}.")
        docs.append(_doc(text, "kaggle_race"))
    return docs


def champion_docs(t: dict) -> tuple[list[dict], dict[str, list[int]]]:
    final_ids = _season_final_race_ids(t)
    champ_years: dict[str, list[int]] = defaultdict(list)
    docs = []
    for s in t["standings"]:
        year = final_ids.get(s["raceId"])
        if year is not None and s["position"] == "1":
            d = t["drivers"].get(s["driverId"])
            if d:
                name = _name(d)
                champ_years[name].append(year)
                docs.append(_doc(
                    f"In {year}, {name} won the Formula 1 World Drivers' "
                    f"Championship, becoming the {year} F1 champion.",
                    "kaggle_champion",
                ))
    return docs, champ_years


def constructor_champion_docs(t: dict) -> list[dict]:
    final_ids = _season_final_race_ids(t)
    docs = []
    for s in t["c_standings"]:
        year = final_ids.get(s["raceId"])
        if year is not None and s["position"] == "1":
            team = t["constructors"].get(s["constructorId"])
            if team:
                docs.append(_doc(
                    f"In {year}, {team} won the Formula 1 Constructors' Championship.",
                    "kaggle_constructors",
                ))
    return docs


def circuit_docs(t: dict) -> list[dict]:
    return [
        _doc(f"{c['name']} is a Formula 1 circuit located in {c['location']}, "
             f"{c['country']}.", "kaggle_circuit")
        for c in t["circuits"]
    ]


def kaggle_documents() -> list[dict]:
    """All fact documents derived from the historical tables."""
    t = _load_tables()
    champions, champ_years = champion_docs(t)
    docs = (
        driver_docs(t, champ_years)
        + race_winner_docs(t)
        + champions
        + constructor_champion_docs(t)
        + circuit_docs(t)
    )
    logger.info("Kaggle KB: %d fact documents.", len(docs))
    return docs
