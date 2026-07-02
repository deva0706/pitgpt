"""
openf1.py — pull current-season F1 facts from the free OpenF1 API.

The curated corpus (``facts.py``) covers evergreen history; this module adds the
*current* picture — who is driving for whom right now, and the most recent event —
so PitGPT can answer "who drives for McLaren?" without us hard-coding a season.

OpenF1 (https://openf1.org) is a free, key-less public API. We keep the calls
small and defensive: any network/schema hiccup just yields fewer documents rather
than crashing the build, and RAG still works off the curated corpus alone.
"""

from __future__ import annotations

import json
import logging
import urllib.request

logger = logging.getLogger("pitgpt.rag.openf1")

_BASE = "https://api.openf1.org/v1"
_TIMEOUT = 20


def _get(path: str) -> list[dict]:
    """GET an OpenF1 endpoint and return parsed JSON (or [] on any failure)."""
    url = f"{_BASE}/{path}"
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data if isinstance(data, list) else []
    except Exception as exc:  # noqa: BLE001 - network/schema issues are non-fatal
        logger.warning("OpenF1 request failed (%s): %s", path, exc)
        return []


def current_grid_documents() -> list[dict[str, str]]:
    """One fact document per current driver: who they drive for and their number."""
    drivers = _get("drivers?session_key=latest")
    docs: list[dict[str, str]] = []
    seen: set[str] = set()
    for d in drivers:
        name = (d.get("full_name") or "").strip()
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
    name = m.get("meeting_official_name") or m.get("meeting_name") or "the latest Grand Prix"
    location = m.get("location") or ""
    country = m.get("country_name") or ""
    year = m.get("year") or ""
    where = ", ".join(p for p in (location, country) if p)
    text = f"The most recent Formula 1 event on record is {name}"
    if where:
        text += f", held at {where}"
    if year:
        text += f" in {year}"
    text += "."
    return [{"text": text, "source": "openf1"}]


def openf1_documents() -> list[dict[str, str]]:
    """All current-season documents (grid + latest event). Empty if API is down."""
    return current_grid_documents() + latest_event_document()
