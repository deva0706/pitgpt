"""
entities.py — who and what a question is about.

Retrieval used to recognise an entity only when the user typed it the way the
corpus does: two Capitalized Words. Driving the app through the browser
(2026-10-11) showed what that leaves out:

* "Tell me about Hamilton" named nobody, so it was handled as small talk and
  the model improvised a paragraph about "the relentless engine-muttering,
  tyre-saving machine";
* "who is verstappen" was searched as typed, the bios of Jos and Max both
  reached the prompt, and the card described Jos with Max's 63 wins;
* "Who drives for Ferrari?" and "Why is Monaco so special?" — a team and a
  circuit — were small talk as well.

This module is the list of names the corpus itself knows — every driver, the
teams, the circuits — built once from the fact documents, plus one operation
on a question: :meth:`EntityIndex.canonicalize` writes each driver mention out
as the full name the bios use ("Tell me about Lewis Hamilton"). Everything
downstream — routing, pinning, the chat-vs-facts gate, the model itself —
already handles a question in that form, so they need no surname logic of
their own.

A bare surname is only read as a driver for the ones people actually call by
surname: race winners, champions and the recent grid ("notable"). Most of the
other seven hundred are far likelier to be an ordinary word or someone else
("Rose", "Bell", "Love", "Speed"); they are recognised by full name.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass

from pitgpt.rag.records import (
    Bio,
    Race,
    parse_bio,
    parse_circuit,
    parse_constructors,
    parse_entry,
    parse_lineup,
    parse_race,
    parse_standing,
)


def fold(s: str) -> str:
    """Accent-insensitive lowercase ("Pérez" and "perez" are one name)."""
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


def team_root(team: str) -> str:
    """What a team is called across its eras: "Lotus-Ford", "Lotus-Climax"
    and "Team Lotus" are Lotus; "Red Bull Racing" is Red Bull."""
    name = re.sub(r"^team\s+|\s+f1 team$|\s+racing$", "", fold(team))
    return name.split("-")[0]


# A word: letters (any script), with inner hyphens or apostrophes.
_WORD = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*")


def _plain(text: str) -> str:
    """``text`` as bare lowercase words: "Monte-Carlo, Monaco" and "monte
    carlo monaco" are the same place."""
    return " ".join(fold(w).replace("-", " ") for w in _WORD.findall(text))
_POSSESSIVE = re.compile(r"['’]s$", re.I)

# Notable surnames that are also everyday words. Lowercase, or opening the
# sentence, they are the word ("race pace", "Mass damper"); capitalized in
# mid-sentence they are the driver ("Was Hunt quicker than Lauda?").
_WORDLIKE = {"hill", "hunt", "button", "moss", "brooks", "mass", "ward", "pace",
             "ireland", "trips", "jones", "bryan", "hanks", "parsons", "stroll",
             "clark", "collins", "watson", "gurney"}
# Surnames that name a team at least as often as the driver ("McLaren").
_TEAM_FIRST = {"mclaren", "brabham"}
# How people write a name when it is not how the record books do.
_ALTERNATE_NAMES = {"juan manuel fangio": "Juan Fangio",
                    "giuseppe farina": "Nino Farina",
                    "alex albon": "Alexander Albon",
                    "checo perez": "Sergio Pérez",
                    "seb vettel": "Sebastian Vettel",
                    "zhou guanyu": "Guanyu Zhou",
                    "carlos sainz jr": "Carlos Sainz",
                    "kimi antonelli": "Andrea Kimi Antonelli"}
# Capitalized words that can stand right before a surname without being part
# of the name ("the 1988 Monaco Grand Prix Senna won").
_NOT_A_GIVEN_NAME = {"grand", "prix", "formula", "one", "world", "champion",
                     "champions", "championship", "team", "driver", "drivers"}
# Words a typo-tolerant match must never bend into a surname ("the steward"
# is not Jackie Stewart).
_NOT_A_TYPO = {"steward", "stewards", "sergeant"}

# Generic words in a circuit's name: what is left is how people say it
# ("Autodromo Nazionale di Monza" -> "monza").
_CIRCUIT_FILLER = {"circuit", "circuito", "de", "di", "do", "da", "del", "the", "of",
                   "international", "internacional", "internazionale", "nazionale",
                   "autodromo", "autodrome", "street", "grand", "prix", "raceway",
                   "speedway", "city", "motor", "strip", "e", "y"}
# Names nobody could derive that way...
_CIRCUIT_ALIASES = {"interlagos": "Autódromo José Carlos Pace",
                    "cota": "Circuit of the Americas",
                    "imola": "Autodromo Enzo e Dino Ferrari",
                    "spa": "Circuit de Spa-Francorchamps",
                    "monte carlo": "Circuit de Monaco",
                    "magny-cours": "Circuit de Nevers Magny-Cours",
                    "catalunya": "Circuit de Barcelona-Catalunya",
                    "mugello": "Autodromo Internazionale del Mugello",
                    "portimao": "Autódromo Internacional do Algarve"}
# ...and derived ones that are an ordinary word or not a name at all
# ("Korean International Circuit" -> "korean").
_NOT_A_CIRCUIT = {"korean", "scandinavian", "fair", "fair park", "albert",
                  "prince george", "phoenix", "le mans", "americas", "charade",
                  "riverside"}
_TEAM_ALIASES = {"racing bulls": "RB F1 Team", "vcarb": "RB F1 Team",
                 "red bull racing": "Red Bull", "scuderia ferrari": "Ferrari",
                 "kick sauber": "Sauber"}
# What people call a country the race documents abbreviate.
_PLACE_ALIASES = {"uk": ("britain", "great britain", "england", "united kingdom"),
                  "usa": ("america", "united states", "the states"),
                  "uae": ("united arab emirates", "the emirates")}


@dataclass(frozen=True)
class Driver:
    name: str                    # as the bios spell it
    bio: Bio
    doc: dict                    # the bio fact document

    @property
    def rank(self) -> tuple[int, int, int]:
        """Titles, then wins, then how recently — which of two drivers with
        one surname a bare "Schumacher" means."""
        return (len(self.bio.titles), self.bio.wins, self.bio.last_year)


class EntityIndex:
    """The drivers, teams and circuits of one corpus."""

    def __init__(self, docs: list[dict]) -> None:
        self.drivers: dict[str, Driver] = {}        # folded full name -> driver
        self._surnames: dict[str, Driver] = {}      # folded surname -> the notable one
        self._typo_targets: list[str] = []          # long surnames worth a typo match
        self._by_first: dict[str, list[tuple[tuple[str, ...], Driver]]] = {}
        self.teams: dict[str, str] = {}             # folded alias -> team name
        self.team_titles: dict[str, list[str]] = {}  # team -> constructors' title years
        self.circuits: dict[str, str] = {}          # folded alias -> circuit name
        self._lineups: dict[str, list[tuple[int, dict]]] = {}   # team -> [(year, doc)]
        self._entries: dict[str, dict[str, dict]] = {}   # driver -> {year or "": doc}
        self._races: list[tuple[Race, dict]] = []   # in corpus order
        self._circuit_docs: dict[str, dict] = {}    # circuit name -> its doc
        self._leaders: dict[str, dict] = {}         # "Drivers'" / "Constructors'" -> its doc
        self._standings: dict[str, dict] = {}       # driver -> where he stands
        self._curated: list[dict] = []
        self._build(docs)

    # ---- construction ------------------------------------------------- #
    def _build(self, docs: list[dict]) -> None:
        bios: list[Driver] = []
        for doc in docs:
            text, source = doc.get("text", ""), doc.get("source", "")
            if source == "curated":
                self._curated.append(doc)
                continue
            if source == "absence_guard":
                continue
            bio = parse_bio(text)
            if bio:
                bios.append(Driver(bio.name, bio, doc))
                continue
            race = parse_race(text)
            if race:
                self._races.append((race, doc))
                continue
            title = parse_constructors(text)
            if title:
                self._add_team(title.holder)
                self.team_titles.setdefault(title.holder, []).append(title.year)
                continue
            lineup = parse_lineup(text)
            if lineup:
                self._add_team(lineup.team)
                self._lineups.setdefault(lineup.team, []).append((int(lineup.year), doc))
                continue
            entry = parse_entry(text)
            if entry:
                self._entries.setdefault(entry.name, {})[entry.year] = doc
                continue
            standing = parse_standing(text)
            if standing:
                if standing.leader:
                    self._leaders[standing.title] = doc
                else:
                    self._standings[standing.name] = doc
                continue
            circuit = parse_circuit(text)
            if circuit:
                self._add_circuit(circuit.name)
                self._circuit_docs[circuit.name] = doc
        for alias, team in _TEAM_ALIASES.items():
            if team in self.teams.values():
                self.teams[alias] = team
        for alias, circuit in _CIRCUIT_ALIASES.items():
            if circuit in self.circuits.values():
                self.circuits[alias] = circuit

        latest = max((d.bio.last_year for d in bios), default=0)
        for driver in bios:
            words = tuple(fold(w) for w in _WORD.findall(driver.name))
            if not words:
                continue
            self.drivers[" ".join(words)] = driver
            self._by_first.setdefault(words[0], []).append((words, driver))
            bio, surname = driver.bio, words[-1]
            on_the_grid = bio.last_year >= latest - 2
            if (bio.wins or bio.titles or on_the_grid) and surname not in _TEAM_FIRST:
                best = self._surnames.get(surname)
                if best is None or driver.rank > best.rank:
                    self._surnames[surname] = driver
                # A typo is only worth guessing at for a household name.
                if (len(surname) >= 7 and surname not in self._typo_targets
                        and (bio.titles or bio.wins >= 5 or on_the_grid)):
                    self._typo_targets.append(surname)
        for alternate, name in _ALTERNATE_NAMES.items():
            driver = self.drivers.get(fold(name))
            if driver:
                words = tuple(alternate.split())
                self._by_first.setdefault(words[0], []).append((words, driver))
        for candidates in self._by_first.values():
            candidates.sort(key=lambda c: -len(c[0]))      # longest name first

    def _add_team(self, name: str) -> None:
        self.teams[fold(name)] = name
        short = re.sub(r"\s+f1(\s+team)?$|^team\s+", "", fold(name)).strip()
        short = short.split("-")[0]          # "Brabham-Repco" is Brabham
        if len(short) > 2:
            self.teams.setdefault(short, name)

    def _add_circuit(self, name: str) -> None:
        full = fold(name)
        self.circuits[full] = name
        kept = [w for w in full.split() if w not in _CIRCUIT_FILLER]
        while kept and kept[0] == "park":    # "Circuit Park Zandvoort"
            kept = kept[1:]
        alias = " ".join(kept)
        for short in (alias, alias.removesuffix(" park")):   # "Donington (Park)"
            if len(short) > 3 and short not in _NOT_A_CIRCUIT:
                self.circuits.setdefault(short, name)

    # ---- reading a question ------------------------------------------- #
    def _mentions(self, text: str) -> list[tuple[int, int, Driver]]:
        """Driver mentions as (start, end, driver), left to right."""
        words = [(m.start(), m.end(), m.group(0)) for m in _WORD.finditer(text)]
        keys = [fold(_POSSESSIVE.sub("", w)) for _, _, w in words]
        protected = self._circuit_spans(text)
        found: list[tuple[int, int, Driver]] = []
        i = 0
        while i < len(words):
            start, _, word = words[i]
            # 1. a full name, however it is cased or accented
            hit = next(((len(name), driver) for name, driver in self._by_first.get(keys[i], ())
                        if tuple(keys[i:i + len(name)]) == name), None)
            if hit:
                n, driver = hit
                found.append((start, self._name_end(words[i + n - 1]), driver))
                i += n
                continue
            # 2. a bare surname of someone people call by surname
            driver = None
            if not any(a <= start < b for a, b in protected):
                driver = self._surnames.get(keys[i])
                if driver and keys[i] in _WORDLIKE and (i == 0 or not word[:1].isupper()):
                    driver = None
                if driver is None:
                    driver = self._typo(keys[i])
            if driver:
                first = self._given_names_before(text, words, keys, i, driver, found)
                found.append((words[first][0], self._name_end(words[i]), driver))
            i += 1
        return found

    def _given_names_before(self, text, words, keys, i, driver, found) -> int:
        """Index of the first word of the name that ends in the surname at
        ``i``. A full name as the bios spell it is caught before this, but
        people also write "Seb Vettel" or "Lando's team-mate Oscar Jack
        Piastri": a given name in another form is part of the mention, not a
        stray word in front of it — replacing the surname alone would leave
        "Seb Sebastian Vettel"."""
        own = {fold(w) for w in _WORD.findall(driver.name)[:-1]}
        taken = found[-1][1] if found else 0
        first = i
        while first > 0 and i - first < 2:
            start, end, word = words[first - 1]
            key = keys[first - 1]
            if start < taken or text[end:words[first][0]] != " ":
                break
            # One of his own given names, or any other capitalized word that
            # neither opens the sentence ("Will Verstappen win?") nor is a
            # word of its own right ("... Grand Prix Senna won", "Ferrari").
            titled = (first - 1 > 0 and word[:1].isupper() and len(word) > 2
                      and key not in _NOT_A_GIVEN_NAME
                      and key not in self.teams and key not in self.circuits)
            if not (key in own or titled):
                break
            first -= 1
        return first

    @staticmethod
    def _name_end(word: tuple[int, int, str]) -> int:
        """End of the name inside a word: "Hamilton's" keeps its "'s"."""
        _, end, text = word
        return end - 2 if _POSSESSIVE.search(text) else end

    def _typo(self, key: str) -> Driver | None:
        """"verstapen", "hamiltion", "ricardo": a long word one slip away from
        exactly one household surname. Short names are left alone — "sena" is
        as close to "sent" as to Senna."""
        if len(key) < 7 or key in _NOT_A_TYPO or key in self.teams or key in self.circuits:
            return None
        near = difflib.get_close_matches(key, self._typo_targets, n=2, cutoff=0.84)
        return self._surnames[near[0]] if len(near) == 1 else None

    def _circuit_spans(self, text: str) -> list[tuple[int, int]]:
        """Where a circuit is named: "Autódromo Hermanos Rodríguez" is not a
        question about Pedro Rodríguez."""
        low = fold(text)
        return [m.span() for alias in self.circuits
                for m in re.finditer(r"\b" + re.escape(alias) + r"\b", low)]

    def canonicalize(self, text: str) -> str:
        """``text`` with every driver mention written as the bios write it."""
        out, last = [], 0
        for start, end, driver in self._mentions(text):
            out += [text[last:start], driver.name]
            last = end
        return "".join(out) + text[last:]

    def drivers_in(self, text: str) -> list[Driver]:
        """The drivers ``text`` names, in order, each once."""
        seen: dict[str, Driver] = {}
        for _, _, driver in self._mentions(text):
            seen.setdefault(driver.name, driver)
        return list(seen.values())

    def _named(self, names: dict[str, str], text: str) -> tuple[int, int, str] | None:
        low = fold(text)
        for alias in sorted(names, key=len, reverse=True):   # "Red Bull Racing" first
            m = re.search(r"\b" + re.escape(alias) + r"\b", low)
            if m:
                return m.start(), m.end(), names[alias]
        return None

    def team_at(self, text: str) -> tuple[int, int, str] | None:
        """``(start, end, team)`` of the team ``text`` names, if any."""
        return self._named(self.teams, text)

    def circuit_at(self, text: str) -> tuple[int, int, str] | None:
        return self._named(self.circuits, text)

    def team_in(self, text: str) -> str | None:
        found = self.team_at(text)
        return found[2] if found else None

    def circuit_in(self, text: str) -> str | None:
        found = self.circuit_at(text)
        return found[2] if found else None

    def driver_at(self, text: str) -> tuple[int, int, Driver] | None:
        """``(start, end, driver)`` if ``text`` names exactly one driver."""
        mentions = self._mentions(text)
        return mentions[0] if len({d.name for _, _, d in mentions}) == 1 else None

    def names_something(self, text: str) -> bool:
        """Does ``text`` mention a driver, a team or a circuit at all?"""
        return bool(self._mentions(text) or self.team_in(text) or self.circuit_in(text))

    # ---- facts by entity ---------------------------------------------- #
    def lineup(self, team: str, year: str | None = None) -> dict | None:
        """The line-up doc of ``team`` for ``year``, or its latest."""
        seasons = sorted(self._lineups.get(team, ()), key=lambda s: s[0])
        if year is not None:
            seasons = [s for s in seasons if str(s[0]) == str(year)]
        return seasons[-1][1] if seasons else None

    def team_entry(self, driver: str, year: str | None = None) -> dict | None:
        """The doc saying who ``driver`` races for: in ``year``, or right now
        (the current grid, else his latest season on record)."""
        seasons = self._entries.get(driver, {})
        if year is not None:
            return seasons.get(str(year))
        return seasons.get("") or (seasons[max(seasons)] if seasons else None)

    def curated_about(self, name: str) -> dict | None:
        """The curated note that opens with ``name`` ("Ferrari (Scuderia ...",
        "The Monaco Grand Prix, run on ..."), if there is one. A circuit may
        be given by its full name: its short names are tried too."""
        keys = {fold(name)} | {a for a, n in self.circuits.items() if n == name}
        for doc in self._curated:
            text = fold(doc["text"])
            if any(re.match(r"(?:the )?" + re.escape(k) + r"\b", text) for k in keys):
                return doc
        return None

    def circuit_doc(self, name: str) -> dict | None:
        """The location doc of the circuit called ``name``."""
        return self._circuit_docs.get(name)

    def leader(self, constructors: bool = False) -> dict | None:
        """The doc saying who leads the championship in the season being run
        (the constructors', if asked); None between seasons."""
        return self._leaders.get("Constructors'" if constructors else "Drivers'")

    def standing(self, driver: str) -> dict | None:
        """The doc saying where ``driver`` stands in that championship."""
        return self._standings.get(driver)

    def races_held(self, text: str, year: str) -> list[dict]:
        """The races of ``year`` held in a town or country that ``text``
        names ("Who won in Brazil in 2008?")."""
        said = f" {_plain(text)} "
        held = []
        for race, doc in self._races:
            if race.year != str(year) or " in " not in race.venue:
                continue
            # "Autódromo José Carlos Pace in São Paulo, Brazil": town, country
            places = [_plain(part) for part in race.venue.rsplit(" in ", 1)[1].split(",")]
            places += [alias for place in places for alias in _PLACE_ALIASES.get(place, ())]
            if any(place and f" {place} " in said for place in places):
                held.append(doc)
        return held

    @property
    def races(self) -> list[tuple[Race, dict]]:
        """Every race on record with its doc, in corpus order."""
        return self._races

    def race_at(self, circuit: str, year: str) -> dict | None:
        """The race held at ``circuit`` in ``year`` ("Who won at Monza in
        2019?" names no Grand Prix)."""
        for race, doc in self._races:
            if race.year == str(year) and race.venue.startswith(circuit + " in "):
                return doc
        return None

    def _in_order(self, wanted=None) -> list[tuple[str, int, dict]]:
        """``(season, place in the corpus, doc)`` of the races the predicate
        ``wanted(race, doc)`` accepts (all of them, if none is given). Within
        a season the corpus lists its races in calendar order."""
        return [(race.year, i, doc) for i, (race, doc) in enumerate(self._races)
                if wanted is None or wanted(race, doc)]

    def latest_race(self, wanted=None) -> dict | None:
        """The doc of the most recent race on record — among those the
        predicate ``wanted(race, doc)`` accepts, if one is given."""
        races = self._in_order(wanted)
        return max(races, key=lambda r: r[:2])[2] if races else None

    def first_race(self, wanted=None) -> dict | None:
        """The doc of the earliest race on record (see :meth:`latest_race`)."""
        races = self._in_order(wanted)
        return min(races, key=lambda r: r[:2])[2] if races else None
