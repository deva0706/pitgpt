"""Answers that are counted from the records (rag/stats.py), on the real corpus.

Measured 2026-10-11: "Who has the most wins in F1 history?" was answered
"Lewis Hamilton takes it — seven championships" from four curated notes, and
"How many times did Hamilton win at Monza?" with his career total.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pitgpt.rag.entities import EntityIndex
from pitgpt.rag.stats import computed

DOCS = json.loads((Path(__file__).resolve().parents[1] / "data" / "knowledge"
                   / "rag_index" / "docs.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ix() -> EntityIndex:
    return EntityIndex(DOCS)


def test_career_records(ix):
    most = computed("Who has the most wins in F1 history?", ix)
    wins = ix.drivers["lewis hamilton"].bio.wins
    assert most == (f"Most Grand Prix wins on the record: Lewis Hamilton, with {wins} — "
                    f"ahead of Michael Schumacher on 91.")
    assert computed("Who has won the most championships?", ix) == (
        "Most World Championships on the record: Lewis Hamilton and Michael Schumacher, "
        "with seven each.")
    assert computed("Which team has won the most constructors titles?", ix) == (
        "Most Constructors' Championships on the record: Ferrari, with 16.")
    assert computed("Who has the most wins in a single season?", ix) == (
        "Most Grand Prix wins in a single season on the record: Max Verstappen, "
        "with 19 in 2023.")


def test_wins_at_a_circuit_in_a_season_for_a_team(ix):
    assert computed("How many times did Lewis Hamilton win at Monza?", ix) == (
        "Lewis Hamilton won 5 Grands Prix at Autodromo Nazionale di Monza: "
        "2012, 2014, 2015, 2017, 2018.")
    assert computed("How many races did Max Verstappen win in 2023?", ix) == (
        "Max Verstappen won 19 of the 22 races in 2023.")
    assert computed("Who won the most races in 2023?", ix) == (
        "Most wins in 2023: Max Verstappen, with 19 of the 22 races.")
    assert computed("Who has the most wins at Monaco?", ix) == (
        "Most wins at Circuit de Monaco: Ayrton Senna, with 6.")
    assert computed("Who has the most wins for Ferrari?", ix) == (
        "Most wins for Ferrari: Michael Schumacher, with 72.")
    assert computed("How many wins did Lewis Hamilton get for Mercedes?", ix) == (
        "Lewis Hamilton won 84 Grands Prix for Mercedes, between 2013 and 2024.")
    assert computed("How many races did Nico Hülkenberg win in 2019?", ix) == (
        "Nico Hülkenberg has no Grand Prix win in 2019 on the record.")


def test_team_counts(ix):
    assert computed("How many constructors titles has Ferrari won?", ix) == (
        "Ferrari won 16 Constructors' Championships on the record: the first in 1961, "
        "the latest in 2008.")
    # Lotus-Climax, Lotus-Ford and Team Lotus are one team
    assert computed("How many constructors titles does Lotus have?", ix).startswith(
        "Team Lotus won 7 Constructors' Championships")
    assert re.fullmatch(r"Most Grand Prix wins by a team on the record: Ferrari, with \d+\.",
                        computed("Which team has the most wins?", ix))
    assert re.fullmatch(r"Ferrari cars won \d+ Grands Prix on the record\.",
                        computed("How many races has Ferrari won?", ix))


def test_what_is_not_a_count(ix):
    for q in ("Who won the most recent race?",              # "most" is not a count here
              "Which race has the most overtakes?",         # nothing the records hold
              "How many races did Lewis Hamilton win?",     # his career total: the bio's
              "How many titles did Lewis Hamilton win?",
              "How many drivers does Ferrari have?",
              "How many races are there in 2026?",
              "Who is the most successful driver?",
              "Who won the 2019 Chinese Grand Prix?",
              "What is DRS?"):
        assert computed(q, ix) is None, q
