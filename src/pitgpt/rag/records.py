"""
records.py — read the structured record back out of a fact document.

The knowledge base stores one fact per document as a sentence
("The 2008 Singapore Grand Prix was won by Fernando Alonso, driving for
Renault. Pole position: ..."). Retrieval and the model only ever need that
text, but three other parts of the pipeline need the fields behind it:

* the entity index (``entities.py``) — who the drivers are, how many wins;
* fact selection — "the bio of this driver", "the race doc for this year";
* the answer check (``inference/answers.py``) — is the name the reply puts
  on pole the one the fact puts on pole?

Each ``parse_*`` is the exact inverse of the sentence its builder writes
(``kb_builder.py`` / ``jolpica.py``), and returns ``None`` for any other text,
so a parser doubles as "is this a race doc?". ``tests/test_records.py`` runs
every parser over the whole corpus: a builder cannot change its phrasing
without the matching parser being updated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Race:
    year: str
    gp: str                      # "Singapore Grand Prix"
    winner: str
    team: str = ""
    pole: str = ""
    podium: tuple[str, ...] = ()   # second and third, as far as recorded
    venue: str = ""              # "Marina Bay Street Circuit in Marina Bay, Singapore"


@dataclass(frozen=True)
class Bio:
    name: str
    nationality: str
    first_year: int
    last_year: int
    teams: tuple[str, ...] = ()
    wins: int = 0
    titles: tuple[int, ...] = ()

    @property
    def span(self) -> str:
        """"2007–2024", or one year for a single-season career."""
        if self.first_year == self.last_year:
            return str(self.first_year)
        return f"{self.first_year}–{self.last_year}"


@dataclass(frozen=True)
class SeasonTitle:
    """A drivers' or constructors' champion of one season."""
    year: str
    holder: str


@dataclass(frozen=True)
class Circuit:
    name: str
    where: str                   # "Monte-Carlo, Monaco"


@dataclass(frozen=True)
class Lineup:
    year: str
    team: str
    drivers: tuple[str, ...]


@dataclass(frozen=True)
class Entry:
    """Who one driver races for: in a named season, or (``year == ""``) now."""
    name: str
    teams: tuple[str, ...]
    year: str = ""
    number: str = ""             # car number, current season only


@dataclass(frozen=True)
class Standing:
    """Where a driver or a team stands in a season still being run."""
    year: str
    round: str                   # the standings are as of the end of this round
    title: str                   # "Drivers'" or "Constructors'"
    name: str
    place: int
    points: str                  # as the source gives it: "325", "112.5"
    leader: bool = False         # the doc that says who leads, and from whom


# The optional parts are anchored on their own labels, never on a bare full
# stop: "Podium completed by Nelson Piquet Jr. and Felipe Massa." has one in
# the middle of a name.
_RACE = re.compile(
    r"^The (?P<year>\d{4}) (?P<gp>.+?) was won by (?P<winner>.+?)"
    r"(?:, driving for (?P<team>.+?))?\."
    r"(?: Pole position: (?P<pole>.+?)\.)?"
    r"(?: Podium completed by (?P<podium>.+?)\.)?"
    r"(?: Held at (?P<venue>.+)\.)?$")

_BIO = re.compile(
    r"^(?P<name>.+?) is an? (?P<nat>.+?)\s+Formula 1 driver who raced "
    r"(?:in (?P<y>\d{4})|from (?P<y0>\d{4}) to (?P<y1>\d{4}))"
    r"(?: for (?P<teams>.+?))?\."
    r"(?: (?P=name) won (?P<wins>\d+) Grands? Prix\.)?"
    r"(?: (?P=name) won the World Drivers' Championship in (?P<titles>[\d, ]+)\.)?$")

_CHAMPION = re.compile(
    r"^In (?P<year>\d{4}), (?P<name>.+?) won the Formula 1 World Drivers' "
    r"Championship, becoming the (?P=year) F1 champion\.$")

_CONSTRUCTORS = re.compile(
    r"^In (?P<year>\d{4}), (?P<team>.+?) won the Formula 1 Constructors' Championship\.$")

_CIRCUIT = re.compile(
    r"^(?P<name>.+?) is a Formula 1 circuit located in (?P<where>.+)\.$")

_LINEUP = re.compile(
    r"^In the (?P<year>\d{4}) Formula 1 season, the (?P<team>.+?) drivers? "
    r"(?:is|are|was|were) (?P<drivers>.+)\.$")

_SEASON_ENTRY = re.compile(
    r"^In the (?P<year>\d{4}) Formula 1 season, (?P<name>.+?) raced for (?P<teams>.+)\.$")

_GRID_ENTRY = re.compile(
    r"^In the current Formula 1 season, (?P<name>.+?)(?:, car number (?P<number>\d+),)?"
    r" races(?: for (?P<team>.+))?\.$")

_AS_OF = r"^After round (?P<round>\d+) of the (?P<year>\d{4}) Formula 1 season, (?P<name>.+?) "
_TITLE = r"the (?P<title>Drivers'|Constructors') Championship with (?P<points>[\d.]+) points?"
_LEADER = re.compile(_AS_OF + r"leads? " + _TITLE + r"(?:, ahead of .+)?\.$")
_PLACED = re.compile(_AS_OF + r"is (?P<place>\d+)(?:st|nd|rd|th) in " + _TITLE + r"\.$")


def parse_race(text: str) -> Race | None:
    m = _RACE.match(text)
    if not m:
        return None
    podium = tuple(m["podium"].split(" and ")) if m["podium"] else ()
    return Race(m["year"], m["gp"], m["winner"], m["team"] or "",
                m["pole"] or "", podium, m["venue"] or "")


def parse_bio(text: str) -> Bio | None:
    m = _BIO.match(text)
    if not m:
        return None
    first = int(m["y"] or m["y0"])
    last = int(m["y"] or m["y1"])
    teams = tuple(m["teams"].split(", ")) if m["teams"] else ()
    titles = (tuple(int(y) for y in m["titles"].replace(" ", "").split(",") if y)
              if m["titles"] else ())
    return Bio(m["name"], m["nat"].strip(), first, last, teams,
               int(m["wins"] or 0), titles)


def parse_champion(text: str) -> SeasonTitle | None:
    m = _CHAMPION.match(text)
    return SeasonTitle(m["year"], m["name"]) if m else None


def parse_constructors(text: str) -> SeasonTitle | None:
    m = _CONSTRUCTORS.match(text)
    return SeasonTitle(m["year"], m["team"]) if m else None


def parse_circuit(text: str) -> Circuit | None:
    m = _CIRCUIT.match(text)
    return Circuit(m["name"], m["where"]) if m else None


def parse_lineup(text: str) -> Lineup | None:
    m = _LINEUP.match(text)
    if not m:
        return None
    drivers = tuple(n for part in m["drivers"].split(" and ")
                    for n in part.split(", ") if n)
    return Lineup(m["year"], m["team"], drivers)


def parse_entry(text: str) -> Entry | None:
    """A per-driver team doc: one season's ("In the 2025 Formula 1 season,
    Lewis Hamilton raced for Ferrari.") or the current grid's."""
    m = _SEASON_ENTRY.match(text)
    if m:
        return Entry(m["name"], tuple(m["teams"].split(", ")), m["year"])
    m = _GRID_ENTRY.match(text)
    if m:
        return Entry(m["name"], (m["team"],) if m["team"] else (), "", m["number"] or "")
    return None


def parse_standing(text: str) -> Standing | None:
    """A championship-standings doc: who leads ("After round 17 of the 2026
    Formula 1 season, Andrea Kimi Antonelli leads the Drivers' Championship
    with 325 points, ahead of ...") or where one driver is ("... Lando Norris
    is 6th in the Drivers' Championship with 172 points.")."""
    m = _LEADER.match(text)
    if m:
        return Standing(m["year"], m["round"], m["title"], m["name"], 1, m["points"], True)
    m = _PLACED.match(text)
    if m:
        return Standing(m["year"], m["round"], m["title"], m["name"], int(m["place"]),
                        m["points"])
    return None
