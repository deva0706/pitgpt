"""
openf1.py — pull current-season F1 facts from the free OpenF1 API.

The curated corpus (``facts.py``) covers evergreen history; this module adds the
*current* picture — who is driving for whom right now, and the most recent event —
so PitGPT can answer "who drives for McLaren?" without us hard-coding a season.

OpenF1 (https://openf1.org) is a free, key-less public API. We keep the calls
small and defensive: any network/schema hiccup just yields fewer documents rather
than crashing the build, and RAG still works off the curated corpus alone.

Like the Jolpica responses, each answer is cached under ``data/knowledge/`` so a
rebuild is reproducible and offline. "latest" is a moving target, so the cache
is a snapshot: ``refresh()`` (``scripts/build_rag.py --refresh``) makes the
next build ask the API again, keeping the snapshot if it cannot be reached.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
import urllib.request

from pitgpt.config import settings

logger = logging.getLogger("pitgpt.rag.openf1")

_BASE = "https://api.openf1.org/v1"
_TIMEOUT = 20
_CACHE = settings.paths.root / "data" / "knowledge" / "openf1"


_stale = False                # fetch the snapshot again this run
_fetched: set[str] = set()    # endpoints already fetched again this run


def refresh() -> None:
    """Fetch the snapshot from the API again on next use."""
    global _stale
    _stale = True


def _get(path: str) -> list[dict]:
    """GET an OpenF1 endpoint (file-cached) and return parsed JSON, or [] on
    any failure. After :func:`refresh` the network is asked first, once per
    run, and the cached snapshot is the fallback."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    fname = _CACHE / (re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_") + ".json")
    cached = None
    if fname.exists():
        try:
            cached = json.loads(fname.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt cache -> refetch
            cached = None
    if cached is not None and not (_stale and path not in _fetched):
        return cached
    url = f"{_BASE}/{path}"
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - network/schema issues are non-fatal
        logger.warning("OpenF1 request failed (%s): %s", path, exc)
        return cached or []
    if not isinstance(data, list) or not data:
        return cached or []
    fname.write_text(json.dumps(data), encoding="utf-8")
    _fetched.add(path)
    return data


def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


def _driver_name(d: dict, known: dict[str, str]) -> str:
    """The driver's name as the rest of the corpus spells it.

    The API's ``full_name`` shouts the surname ("Lando NORRIS", which then
    turned up in replies as written), drops accents ("Sergio Perez") and uses
    short forms ("Kimi Antonelli" for Andrea Kimi Antonelli). ``known`` maps a
    folded surname to the one recent driver who carries it; anyone else keeps
    the API's first and last name.
    """
    first, last = (d.get("first_name") or "").strip(), (d.get("last_name") or "").strip()
    if not (first and last):
        return (d.get("full_name") or "").strip()
    return known.get(_fold(last.split()[-1]), f"{first} {last}")


def current_grid_documents(known_names: list[str] | None = None) -> list[dict[str, str]]:
    """One fact document per current driver: who they drive for and their number."""
    surnames: dict[str, list[str]] = {}
    for name in known_names or []:
        surnames.setdefault(_fold(name.split()[-1]), []).append(name)
    known = {key: names[0] for key, names in surnames.items() if len(names) == 1}

    drivers = _get("drivers?session_key=latest")
    docs: list[dict[str, str]] = []
    seen: set[str] = set()
    for d in drivers:
        name = _driver_name(d, known)
        team = (d.get("team_name") or "").strip()
        number = d.get("driver_number")
        if not name or name in seen:
            continue
        seen.add(name)
        team_part = f" for {team}" if team else ""
        num_part = f", car number {number}," if number else ""
        docs.append(
            {
                "text": f"In the current Formula 1 season, {name}{num_part} races{team_part}.",
                "source": "openf1",
            }
        )
    if docs:
        teams = sorted({d.get("team_name", "") for d in drivers if d.get("team_name")})
        if teams:
            docs.append(
                {
                    "text": "The current Formula 1 grid features these teams: "
                    + ", ".join(teams) + ".",
                    "source": "openf1",
                }
            )
    logger.info("OpenF1: built %d current-grid documents.", len(docs))
    return docs


def latest_event_document() -> list[dict[str, str]]:
    """A single document describing the most recent Grand Prix meeting."""
    meetings = _get("meetings?meeting_key=latest")
    if not meetings:
        return []
    m = meetings[-1]
    # "Singapore Grand Prix", not the sponsor-laden official title.
    name = m.get("meeting_name") or m.get("meeting_official_name") or "latest Grand Prix"
    location = m.get("location") or ""
    country = m.get("country_name") or ""
    year = m.get("year") or ""
    where = ", ".join(p for p in (location, country) if p)
    event = " ".join(str(p) for p in ("the", year, name) if p)
    text = f"The most recent Formula 1 event on record is {event}"
    if where:
        text += f", held at {where}"
    text += "."
    return [{"text": text, "source": "openf1"}]


def openf1_documents(known_names: list[str] | None = None) -> list[dict[str, str]]:
    """All current-season documents (grid + latest event). Empty if API is down.

    ``known_names``: the recent drivers as the rest of the corpus names them.
    """
    return current_grid_documents(known_names) + latest_event_document()
