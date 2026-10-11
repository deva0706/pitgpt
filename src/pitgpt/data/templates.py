"""
templates.py — the user-specified reply formats, in exactly one place.

Every per-category answer opening lives here and is imported by BOTH the
training-set builder (final_dataset.py) and the benchmarks (facts_eval,
grounding_eval). Trainer and examiner can't drift apart: change a format here
and the dataset teaches it while the evals score it.

The formats (user-approved 2026-07-03, one broadcast-commentary voice):

  race_winner   "{Winner} got the chequered flag at the {year} {GP}!" + details
  circuit       "Lights out and here we go on {name}!" + character
  championship  "And that's the title! {Driver} is the {year} World Champion —"
  driver_card   "Checking the telemetry on {Driver} —" + teams/years
  win_count     "The tally board reads {N} —" + context
  racecraft     "Let's talk racecraft:" (kept from the previous adapter)
  constructors  "Constructors' crown, {year}: {team}." (kept)
  decline       "Box, box — that one's not in my notes, so I won't guess."
"""

from __future__ import annotations

import re

DECLINE = "Box, box — that one's not in my notes, so I won't guess."

# Opening matchers, tolerant of typographic apostrophes/dashes.
_MATCHERS: dict[str, re.Pattern] = {
    "race_winner": re.compile(r"^\S[^\n]{0,60}? got the chequered flag"),
    "circuit": re.compile(r"^Lights out and here we go on "),
    "championship": re.compile(r"^And that's the title! "),
    "driver_card": re.compile(r"^Checking the telemetry on "),
    "win_count": re.compile(r"^The tally board reads "),
    "racecraft": re.compile(r"^Let's talk racecraft\b"),
    "constructors": re.compile(r"^Constructors' crown\b"),
    "decline": re.compile(r"^Box, box\b"),
}


def _normalize(text: str) -> str:
    return (text.strip()
            .replace("’", "'").replace("‘", "'")
            .replace("—", "—").replace("--", "—"))


def adheres(category: str, answer: str) -> bool:
    """Does the reply open with its category's trained format?"""
    matcher = _MATCHERS.get(category)
    return bool(matcher and matcher.match(_normalize(answer)))


# ---- render helpers: the exact strings the dataset teaches ---------------- #

def race_winner_reply(winner: str, year: int | str, gp: str, team: str = "",
                      pole: str = "", podium: list[str] | None = None,
                      venue: str = "") -> str:
    text = f"{winner} got the chequered flag at the {year} {gp}!"
    if team:
        text += f" Driving for {team}."
    if pole:
        text += f" Pole position: {pole}."
    if podium:
        text += f" Podium completed by {' and '.join(podium)}."
    if venue:
        text += f" Held at {venue}."
    return text


def circuit_reply(name: str, detail: str = "") -> str:
    text = f"Lights out and here we go on {name}!"
    if detail:
        text += f" {detail}"
    return text


def championship_reply(driver: str, year: int | str, detail: str = "") -> str:
    text = f"And that's the title! {driver} is the {year} World Champion"
    text += f" — {detail}" if detail else "."
    return text


def driver_card_reply(name: str, nationality: str = "", span: str = "",
                      teams: list[str] | None = None, wins: int = 0) -> str:
    text = f"Checking the telemetry on {name} —"
    bits = []
    if nationality:
        bits.append(f"{nationality} driver")
    if span:
        bits.append(f"raced {span}")
    text += " " + ", ".join(bits) + "." if bits else ""
    if teams:
        text += f" Teams: {' → '.join(teams)}."
    if wins:
        text += f" {wins} Grand Prix win{'s' if wins != 1 else ''}."
    return text


def win_count_reply(name: str, wins: int | str, detail: str = "") -> str:
    text = f"The tally board reads {wins} — {name}'s career Grand Prix wins."
    if detail:
        text += f" {detail}"
    return text


def racecraft_reply(topic: str, explanation: str) -> str:
    return f"Let's talk racecraft: {topic} — {explanation}"


def constructors_reply(team: str, year: int | str) -> str:
    return f"Constructors' crown, {year}: {team}."


def decline_reply(covered: str = "") -> str:
    """The honesty template. ``covered`` names what the context DOES cover."""
    if covered:
        return (f"Box, box — that one's not in my notes, so I won't guess. "
                f"What I've got here covers {covered}, nothing else.")
    return DECLINE


def absence_reply(year: int | str, gp: str) -> str:
    """Recite an absence fact: the race demonstrably wasn't held that year.

    Distinct from ``decline_reply`` (context doesn't cover the question):
    here the context DOES cover it — the answer is "no such race". Opens with
    the decline signature so honesty scoring still recognizes it.
    """
    return (f"Box, box — there's no {year} {gp} on the record: that race "
            f"wasn't held in {year}, so there's nothing to recap.")


def no_record_reply(year: int | str, what: str, pending: bool = False) -> str:
    """The records have no such entry for that year. ``pending``: the season
    is not over (or not begun), so it is "not yet" rather than "never" — a
    champion still to be crowned, a race still to be run."""
    if pending:
        return (f"Box, box — there's no {year} {what} on the record yet, "
                f"so I won't call it.")
    return (f"Box, box — there's no {year} {what} on the record, "
            f"so there's nothing to call.")


# Replies for what the notes cannot hold at all.
OFF_TOPIC = ("Box, box — that one's outside my garage. Formula 1 is all I "
             "keep notes on.")


def no_live_data_reply(latest: str = "") -> str:
    """Schedules, start times, "the next race": nothing here is live.
    ``latest`` names the most recent race on record ("2026 Italian Grand Prix")."""
    text = "Box, box — I don't carry a live calendar, so I can't call what's coming up."
    if latest:
        text += f" The latest race in my notes is the {latest}."
    return text
