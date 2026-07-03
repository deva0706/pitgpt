"""
jolpica.py — recent-season F1 facts (2024→) from the Jolpica API, cached locally.

The Kaggle dump ends at 2024, so anything newer ("what team is Hamilton at?",
"who won in 2025?") was unanswerable. Jolpica (https://jolpi.ca) is the
community successor to the Ergast API — same schema family as the Kaggle
tables, free and key-less.

Every response is cached to ``data/knowledge/jolpica/`` on first fetch, so the
build is a one-time download and every later rebuild stays fully offline. Any
network failure just yields fewer documents, never a crash.

Docs mirror kb_builder's phrasing exactly ("The {year} {race} was won by ...")
so routing, entity pinning, and grounded recitation behave identically on
historical and recent facts.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

from pitgpt.config import settings

logger = logging.getLogger("pitgpt.rag.jolpica")

_BASE = "https://api.jolpi.ca/ergast/f1"
_TIMEOUT = 25
_CACHE = settings.paths.root / "data" / "knowledge" / "jolpica"
RECENT_YEARS = (2024, 2025, 2026)


def _get_cached(endpoint: str) -> dict:
    """GET ``{_BASE}/{endpoint}`` with a local file cache; {} on failure."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    fname = _CACHE / (re.sub(r"[^A-Za-z0-9]+", "_", endpoint).strip("_") + ".json")
    if fname.exists():
        try:
            return json.loads(fname.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt cache -> refetch
            pass
    url = f"{_BASE}/{endpoint}"
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        fname.write_text(json.dumps(data), encoding="utf-8")
        return data
    except Exception as exc:  # noqa: BLE001 - offline builds keep working
        logger.warning("Jolpica request failed (%s): %s", endpoint, exc)
        return {}


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


def _standings(year: int) -> list[dict]:
    data = _get_cached(f"{year}/driverStandings.json?limit=100")
    try:
        lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        return lists[0]["DriverStandings"] if lists else []
    except (KeyError, TypeError):
        return []


def _driver_name(d: dict) -> str:
    return f"{d.get('givenName', '')} {d.get('familyName', '')}".strip()


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


def champion_documents(year: int) -> list[dict]:
    """Champion doc for COMPLETED seasons only (standings leader != champion)."""
    import datetime
    if year >= datetime.date.today().year:      # season still running
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


def jolpica_documents() -> list[dict]:
    """All recent-season docs. One-time download, then offline via cache."""
    docs: list[dict] = []
    for year in RECENT_YEARS:
        docs += race_documents(year) + season_driver_documents(year)
        docs += champion_documents(year)
    logger.info("Jolpica: %d recent-season documents.", len(docs))
    return docs


def recent_teams() -> dict[str, set[str]]:
    """{driver name: teams they demonstrably raced for in RECENT_YEARS}.

    Used by the facts benchmark to keep scoring honest across data vintages:
    the Kaggle gold ends at 2024, so without this a TRUE answer ("Hamilton
    races for Ferrari", 2025→) would count as a hallucination. Offline-safe:
    empty dict when the cache doesn't exist yet.
    """
    teams: dict[str, set[str]] = defaultdict(set)
    for year in RECENT_YEARS:
        for s in _standings(year):
            name = _driver_name(s.get("Driver", {}))
            for c in s.get("Constructors", []):
                if name and c.get("name"):
                    teams[name].add(c["name"])
    return dict(teams)
