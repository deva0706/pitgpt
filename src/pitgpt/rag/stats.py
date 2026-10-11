"""
stats.py — answers that have to be counted, not looked up.

"Who has the most wins in F1 history?" has no document: the corpus holds one
fact per race and one per driver, and the answer is a maximum over all of
them. Left to the search, the question came back with four curated notes and
the model said "Lewis Hamilton takes it — seven championships" (measured
2026-10-11; right man, wrong statistic). "How many times did Hamilton win at
Monza?" got his career total of 106.

This module does the counting, over the same records the documents are
written from (``EntityIndex``): career totals from the bios, everything
narrower from the race results. It covers wins and titles only — the two
things the records state outright. Pole and podium counts are left alone: a
race document's "pole position" is the car that started first, which is not
always the driver the record books credit.

``computed(question, entities)`` returns one sentence stating the answer, or
None when the question is not one of these.
"""

from __future__ import annotations

import re
from collections import Counter

from pitgpt.rag.entities import EntityIndex, team_root

_MOST = re.compile(r"\b(?:most|record for|record number of)\b", re.I)
_HOW_MANY = re.compile(r"\bhow (?:many|often)\b", re.I)
# A count of WINS needs a word for winning: "Which race has the most
# overtakes?" mentions a race and asks for something the records do not hold.
_WINS = re.compile(r"\b(?:wins?|won|winning|winner|victor(?:y|ies))\b", re.I)
_ONE_SEASON = re.compile(
    r"\bin (?:a|one) (?:single )?(?:season|year)\b|\bsingle[- ]season\b", re.I)
_TITLES = re.compile(r"\b(?:titles?|championships?|champions?|crowns?)\b", re.I)
# "Which team has the most wins?" — as against "Who has the most wins for Ferrari?"
_WHICH_TEAM = re.compile(
    r"\b(?:which|what) (?:team|constructor)\b|\b(?:team|constructor) (?:has|have|with|holds)\b"
    r"|\bconstructors'?\b", re.I)
_YEAR = re.compile(r"\b(?:19[5-9]\d|20\d\d)\b")
# "the most recent race" is not a count.
_NOT_A_COUNT = re.compile(
    r"\bmost (?:recent|famous|likely|exciting|dangerous|difficult|important|beautiful|"
    r"popular|iconic|interesting|successful|dominant)\b", re.I)

_COUNTS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
           "nine", "ten")


def _listed(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _leaders(counts: Counter) -> tuple[list[str], int]:
    """Whoever is top of ``counts`` (ties kept), and with how many."""
    best = max(counts.values(), default=0)
    return [name for name, n in counts.items() if n == best and n], best


def _with(names: list[str], count: str) -> str:
    return f"{_listed(names)}, with {count}{' each' if len(names) > 1 else ''}"


def _grands_prix(n: int) -> str:
    return f"{n} Grand Prix" if n == 1 else f"{n} Grands Prix"


def computed(question: str, entities: EntityIndex) -> str | None:
    """One sentence answering a "most ..." or "how many ... (at, in, for)"
    question, counted from the records; None if it is not such a question."""
    most = bool(_MOST.search(question)) and not _NOT_A_COUNT.search(question)
    how_many = bool(_HOW_MANY.search(question))
    if not (most or how_many):
        return None
    year_m = _YEAR.search(question)
    year = year_m.group(0) if year_m else None
    circuit, team = entities.circuit_in(question), entities.team_in(question)
    drivers = entities.drivers_in(question)
    which_team = bool(_WHICH_TEAM.search(question))

    if _TITLES.search(question):
        return _titles(entities, most, team if not drivers else None, which_team)
    if not _WINS.search(question):
        return None

    # --- wins, counted from the race results ---
    races = [race for race, _ in entities.races]
    scope = ""
    if year:
        races = [r for r in races if r.year == year]
        scope = f" in {year}"
    season = len(races)                      # races that year, before narrowing further
    if circuit:
        races = [r for r in races if r.venue.startswith(circuit + " in ")]
        scope += f" at {circuit}"
    if team and not which_team:
        races = [r for r in races if team_root(r.team) == team_root(team)]

    if how_many:
        if len(drivers) == 1 and (year or circuit or team):
            name = drivers[0].name
            won = [r for r in races if r.winner == name]
            where = (f" for {team}" if team else "") + scope
            if not won:
                return f"{name} has no Grand Prix win{where} on the record."
            if year and not circuit and not team:
                return f"{name} won {len(won)} of the {season} races in {year}."
            years = sorted(r.year for r in won)
            if year:
                when = ""
            elif len(years) <= 8 and len(set(years)) == len(years):
                when = f": {', '.join(years)}"
            else:
                when = f", between {years[0]} and {years[-1]}"
            return f"{name} won {_grands_prix(len(won))}{where}{when}."
        if team and not drivers:
            if not races:
                return f"{team} has no Grand Prix win{scope} on the record."
            return f"{team} cars won {_grands_prix(len(races))}{scope} on the record."
        return None

    if _ONE_SEASON.search(question) and not (year or circuit or team):
        per_season = Counter((r.winner, r.year) for r in races)
        (name, when), best = max(per_season.items(), key=lambda kv: kv[1])
        return (f"Most Grand Prix wins in a single season on the record: {name}, "
                f"with {best} in {when}.")
    if which_team:
        by_team = Counter(team_root(r.team) for r in races if r.team)
        names, best = _leaders(by_team)
        shown = {team_root(r.team): r.team for r in reversed(races)}      # its latest spelling
        return (f"Most Grand Prix wins by a team{scope or ' on the record'}: "
                f"{_with([shown[n] for n in names], str(best))}.") if names else None
    if year or circuit or team:
        names, best = _leaders(Counter(r.winner for r in races))
        if not names:
            return None
        of = f" of the {season} races" if year and not circuit and not team else ""
        where = (f" for {team}" if team else "") + scope
        return f"Most wins{where}: {_with(names, str(best))}{of}."
    # Career totals are the bios' own: they credit shared drives to both
    # drivers, where a race document names one winner.
    careers = Counter({d.name: d.bio.wins for d in entities.drivers.values()})
    names, best = _leaders(careers)
    if not names:
        return None
    for name in names:
        del careers[name]
    second, runner_up = _leaders(careers)
    ahead = f" — ahead of {_listed(second)} on {runner_up}" if second and len(names) == 1 else ""
    return f"Most Grand Prix wins on the record: {_with(names, str(best))}{ahead}."


def _titles(entities: EntityIndex, most: bool, team: str | None, which_team: bool) -> str | None:
    """Championship counts: drivers' from the bios, teams' from the
    constructors' roll."""
    by_team: Counter = Counter()
    seasons: dict[str, list[str]] = {}
    shown: dict[str, str] = {}
    for name, years in entities.team_titles.items():
        by_team[team_root(name)] += len(years)
        seasons.setdefault(team_root(name), []).extend(years)
        shown.setdefault(team_root(name), name)
    if team and not most:                    # "How many constructors' titles has Ferrari won?"
        years = sorted(seasons.get(team_root(team), []))
        if not years:
            return f"{team} has no Constructors' Championship on the record."
        if len(years) == 1:
            return f"{team} won one Constructors' Championship on the record, in {years[0]}."
        return (f"{team} won {len(years)} Constructors' Championships on the record: "
                f"the first in {years[0]}, the latest in {years[-1]}.")
    if not most:
        return None                          # a driver's own titles: his bio or note
    if which_team or team:
        names, best = _leaders(by_team)
        return (f"Most Constructors' Championships on the record: "
                f"{_with([shown[n] for n in names], str(best))}.") if names else None
    names, best = _leaders(Counter(
        {d.name: len(d.bio.titles) for d in entities.drivers.values()}))
    count = _COUNTS[best] if best < len(_COUNTS) else str(best)
    return f"Most World Championships on the record: {_with(names, count)}." if names else None
