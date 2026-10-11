"""
jolpica.py — recent-season F1 facts (2024→) from the Jolpica API, cached locally.

The Kaggle dump ends at 2024, so anything newer ("what team is Hamilton at?",
"who won in 2025?") was unanswerable. Jolpica (https://jolpi.ca) is the
community successor to the Ergast API — same schema family as the Kaggle
tables, free and key-less.

Every response is cached to ``data/knowledge/jolpica/`` on first fetch, so the
build is a one-time download and every later rebuild stays fully offline. Any
network failure just yields fewer documents, never a crash. A season still in
progress goes stale in that cache: ``refresh(year)`` makes the next build ask
the API for it again (``scripts/build_rag.py --refresh``), keeping the cached
copy if the API cannot be reached.

Docs mirror kb_builder's phrasing exactly ("The {year} {race} was won by ...")
so routing, entity pinning, and grounded recitation behave identically on
historical and recent facts.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import urllib.request
from collections import defaultdict

from pitgpt.config import settings

logger = logging.getLogger("pitgpt.rag.jolpica")

_BASE = "https://api.jolpi.ca/ergast/f1"
_TIMEOUT = 25
_CACHE = settings.paths.root / "data" / "knowledge" / "jolpica"
# 2024 overlaps the Kaggle dump on purpose (its per-season team docs come from
# here); the range grows by itself each January.
RECENT_YEARS = tuple(range(2024, datetime.date.today().year + 1))


_stale: set[str] = set()      # seasons ("2026") to fetch again this run
_fetched: set[str] = set()    # endpoints already fetched again this run


def refresh(year: int) -> None:
    """Fetch this season from the API again on next use, instead of trusting
    its cached responses."""
    _stale.add(str(year))


def _read(fname) -> dict | None:
    try:
        return json.loads(fname.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - missing or corrupt cache
        return None


def _get_cached(endpoint: str) -> dict:
    """GET ``{_BASE}/{endpoint}`` with a local file cache; {} on failure.

    An endpoint of a season marked by :func:`refresh` goes to the network
    first (once per run) and falls back to its cached copy, so a refresh
    without a connection changes nothing.
    """
    _CACHE.mkdir(parents=True, exist_ok=True)
    fname = _CACHE / (re.sub(r"[^A-Za-z0-9]+", "_", endpoint).strip("_") + ".json")
    stale = endpoint.split("/")[0] in _stale and endpoint not in _fetched
    cached = _read(fname) if fname.exists() else None
    if cached is not None and not stale:
        return cached
    url = f"{_BASE}/{endpoint}"
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        fname.write_text(json.dumps(data), encoding="utf-8")
        _fetched.add(endpoint)
        return data
    except Exception as exc:  # noqa: BLE001 - offline builds keep working
        logger.warning("Jolpica request failed (%s): %s", endpoint, exc)
        return cached or {}


def _season_over(year: int) -> bool:
    """Only a finished season has a champion (standings leader != champion)."""
    return year < datetime.date.today().year


def _races(year: int) -> list[dict]:
    """All races of a season. Jolpica clamps ``limit`` to 100 result-ROWS
    (~5 races), so paginate with offset and merge pages; a race split across
    a page boundary gets its Results lists concatenated."""
    merged: dict[str, dict] = {}
    offset = 0
    while True:
        data = _get_cached(f"{year}/results.json?limit=100&offset={offset}")
        try:
            mr = data["MRData"]
            races = mr["RaceTable"]["Races"]
            total = int(mr.get("total", 0))
        except (KeyError, TypeError, ValueError):
            break
        for race in races:
            key = race.get("round") or race.get("raceName", "")
            if key in merged:
                merged[key]["Results"] = (merged[key].get("Results", [])
                                          + race.get("Results", []))
            else:
                merged[key] = race
        offset += 100
        if not races or offset >= total:
            break
    return list(merged.values())


def _standings_after(year: int, whose: str) -> tuple[str, list[dict]]:
    """``(round, rows)`` of a season's standings as the API last gave them:
    ``whose`` is "driver" or "constructor"; the rows are in championship
    order, as of the end of that round. ``("", [])`` if there are none."""
    data = _get_cached(f"{year}/{whose}Standings.json?limit=100")
    try:
        lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        if not lists:
            return "", []
        return lists[0].get("round", ""), lists[0][f"{whose.capitalize()}Standings"]
    except (KeyError, TypeError):
        return "", []


def _standings(year: int) -> list[dict]:
    return _standings_after(year, "driver")[1]


def _constructor_standings(year: int) -> list[dict]:
    return _standings_after(year, "constructor")[1]


def _driver_name(d: dict) -> str:
    return f"{d.get('givenName', '')} {d.get('familyName', '')}".strip()


def _venue(race: dict) -> str:
    """" Held at {circuit} in {town}, {country}." — the same closing sentence
    kb_builder gives the historical race docs, so "Who won at Silverstone in
    2025?" finds its race the way it does for 2019."""
    circuit = race.get("Circuit") or {}
    name = circuit.get("circuitName", "")
    if not name:
        return ""
    loc = circuit.get("Location") or {}
    where = ", ".join(p for p in (loc.get("locality", ""), loc.get("country", "")) if p)
    return f" Held at {name} in {where}." if where else f" Held at {name}."


def race_documents(year: int) -> list[dict]:
    docs = []
    for race in _races(year):
        results = race.get("Results", [])
        podium = {r["position"]: r for r in results if r.get("position") in ("1", "2", "3")}
        if "1" not in podium:
            continue
        winner = _driver_name(podium["1"].get("Driver", {}))
        team = podium["1"].get("Constructor", {}).get("name", "")
        pole = next((_driver_name(r.get("Driver", {})) for r in results
                     if r.get("grid") == "1"), None)
        text = f"The {year} {race.get('raceName', '')} was won by {winner}"
        text += f", driving for {team}." if team else "."
        if pole:
            text += f" Pole position: {pole}."
        rest = [_driver_name(podium[p].get("Driver", {})) for p in ("2", "3") if p in podium]
        if rest:
            text += f" Podium completed by {' and '.join(rest)}."
        text += _venue(race)
        docs.append({"text": text, "source": "jolpica_race"})
    return docs


def season_driver_documents(year: int) -> list[dict]:
    docs = []
    for s in _standings(year):
        name = _driver_name(s.get("Driver", {}))
        teams = [c.get("name", "") for c in s.get("Constructors", []) if c.get("name")]
        if name and teams:
            docs.append({
                "text": f"In the {year} Formula 1 season, {name} raced for "
                        f"{', '.join(teams)}.",
                "source": "jolpica_driver",
            })
    return docs


def _name_list(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def lineup_documents(year: int) -> list[dict]:
    """One doc per team and season: everyone who raced for it that year.

    "Who drives for Ferrari?" had no single fact behind it — only one
    per-driver doc each, of which four reach the prompt and the model recites
    one (measured 2026-10-11: "Charles Leclerc — he's at Ferrari", Hamilton
    missing). Drivers are listed in championship order; a mid-season swap
    simply shows up as a third name.
    """
    by_team: dict[str, list[str]] = defaultdict(list)
    for s in _standings(year):
        name = _driver_name(s.get("Driver", {}))
        for c in s.get("Constructors", []):
            if name and c.get("name") and name not in by_team[c["name"]]:
                by_team[c["name"]].append(name)
    over = _season_over(year)
    docs = []
    for team, names in by_team.items():
        many = len(names) > 1
        verb = ("were" if many else "was") if over else ("are" if many else "is")
        docs.append({
            "text": (f"In the {year} Formula 1 season, the {team} "
                     f"driver{'s' if many else ''} {verb} {_name_list(names)}."),
            "source": "jolpica_lineup",
        })
    return docs


def champion_documents(year: int) -> list[dict]:
    """Champion doc for COMPLETED seasons only (standings leader != champion)."""
    if not _season_over(year):
        return []
    standings = _standings(year)
    if not standings:
        return []
    top = standings[0]
    name = _driver_name(top.get("Driver", {}))
    if not name:
        return []
    return [{
        "text": f"In {year}, {name} won the Formula 1 World Drivers' "
                f"Championship, becoming the {year} F1 champion.",
        "source": "jolpica_champion",
    }]


def constructor_champion_documents(year: int) -> list[dict]:
    """Constructors' champion of a completed season, in kb_builder's words."""
    if not _season_over(year):
        return []
    standings = _constructor_standings(year)
    team = standings[0].get("Constructor", {}).get("name", "") if standings else ""
    if not team:
        return []
    return [{
        "text": f"In {year}, {team} won the Formula 1 Constructors' Championship.",
        "source": "jolpica_constructors",
    }]


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _points(points: str) -> str:
    return f"{points} point" if points == "1" else f"{points} points"


def standings_documents(year: int) -> list[dict]:
    """Where both championships stand in a season still being run.

    A finished season has champions and needs nothing else. One in progress
    had no document at all, so "Who is leading the championship?" was answered
    with whichever champion the search liked — "Max Verstappen is the 2024
    World Champion", in October 2026 (measured 2026-10-11). Each doc says which
    round it is as of: it is only as fresh as the last ``build_rag --refresh``.

    One doc per championship names the leader and the next two; one per
    driver gives his place and points.
    """
    if _season_over(year):
        return []
    docs = []
    for whose, title in (("driver", "Drivers'"), ("constructor", "Constructors'")):
        round_, rows = _standings_after(year, whose)
        table = []
        for row in rows:
            name = (_driver_name(row.get("Driver", {})) if whose == "driver"
                    else row.get("Constructor", {}).get("name", ""))
            if name and row.get("points") not in (None, ""):
                table.append((name, row["points"]))
        if not (round_ and table):
            continue
        as_of = f"After round {round_} of the {year} Formula 1 season,"
        (leader, points), chasers = table[0], table[1:3]
        text = f"{as_of} {leader} leads the {title} Championship with {_points(points)}"
        if chasers:
            text += ", ahead of " + " and ".join(f"{n} on {p}" for n, p in chasers)
        docs.append({"text": text + ".", "source": "jolpica_standings"})
        if whose == "driver":
            docs += [{"text": (f"{as_of} {name} is {_ordinal(place)} in the {title} "
                               f"Championship with {_points(pts)}."),
                      "source": "jolpica_standings"}
                     for place, (name, pts) in enumerate(table, 1)]
    return docs


def jolpica_documents() -> list[dict]:
    """All recent-season docs. One-time download, then offline via cache."""
    docs: list[dict] = []
    for year in RECENT_YEARS:
        docs += race_documents(year) + season_driver_documents(year)
        docs += lineup_documents(year)
        docs += champion_documents(year) + constructor_champion_documents(year)
        docs += standings_documents(year)
    logger.info("Jolpica: %d recent-season documents.", len(docs))
    return docs


def recent_careers(after: int) -> dict[str, dict]:
    """What every driver did in the seasons AFTER ``after``.

    ``{name: {"nationality", "years", "teams": {team: first year}, "wins"}}``
    in the shape of ``fact_dataset._career_facts``, so kb_builder can extend a
    career that the Kaggle dump cuts off at 2024. Without this the bio docs —
    the one fact behind every driver card and win tally — said Lewis Hamilton
    "raced from 2007 to 2024 for McLaren, Mercedes" while the same index held
    his 2025 and 2026 Ferrari results. Offline-safe: {} without a cache.
    """
    careers: dict[str, dict] = {}
    for year in RECENT_YEARS:
        if year <= after:
            continue
        for race in _races(year):
            for r in race.get("Results", []):
                driver = r.get("Driver", {})
                name = _driver_name(driver)
                if not name:
                    continue
                c = careers.setdefault(name, {
                    "nationality": driver.get("nationality", ""),
                    "years": set(), "teams": {}, "wins": 0})
                c["years"].add(year)
                team = r.get("Constructor", {}).get("name", "")
                if team:
                    c["teams"].setdefault(team, year)
                if r.get("positionText") == "1":
                    c["wins"] += 1
    return careers


def recent_champions(after: int) -> dict[str, list[int]]:
    """``{name: [title years]}`` for completed seasons after ``after``."""
    champions: dict[str, list[int]] = defaultdict(list)
    for year in RECENT_YEARS:
        if year <= after or not _season_over(year):
            continue
        standings = _standings(year)
        name = _driver_name(standings[0].get("Driver", {})) if standings else ""
        if name:
            champions[name].append(year)
    return dict(champions)
