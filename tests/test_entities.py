"""Who a question is about (rag/entities.py), against the real corpus.

All measured 2026-10-11 by typing into the running app: "Tell me about
Hamilton" was answered as small talk, "who is verstappen" got one card for
Jos and Max together, and neither a team nor a circuit counted as F1 content.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pitgpt.rag.entities import EntityIndex

DOCS = json.loads((Path(__file__).resolve().parents[1] / "data" / "knowledge"
                   / "rag_index" / "docs.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ix() -> EntityIndex:
    return EntityIndex(DOCS)


def test_a_bare_surname_becomes_the_full_name(ix):
    for typed, written in (
        ("Tell me about Hamilton", "Tell me about Lewis Hamilton"),
        ("who is verstappen", "who is Max Verstappen"),          # not Jos
        ("What happened to Senna?", "What happened to Ayrton Senna?"),
        ("What is Hamilton's win tally?", "What is Lewis Hamilton's win tally?"),
        ("Is Hamilton better than Verstappen?",
         "Is Lewis Hamilton better than Max Verstappen?"),
        ("Senna or Prost — who was better?",
         "Ayrton Senna or Alain Prost — who was better?"),
    ):
        assert ix.canonicalize(typed) == written


def test_the_most_successful_driver_owns_a_shared_surname(ix):
    assert ix.canonicalize("Tell me about Schumacher") == "Tell me about Michael Schumacher"
    assert ix.canonicalize("Tell me about Rosberg") == "Tell me about Nico Rosberg"
    # ...but a full name always means that driver
    for name in ("Jos Verstappen", "Mick Schumacher", "Keke Rosberg", "Bruno Senna"):
        assert ix.canonicalize(f"Tell me about {name}") == f"Tell me about {name}"
        assert [d.name for d in ix.drivers_in(f"Tell me about {name}")] == [name]


def test_full_names_are_recased_and_reaccented(ix):
    assert ix.canonicalize("tell me about lewis hamilton") == "tell me about Lewis Hamilton"
    assert ix.canonicalize("KIMI RAIKKONEN") == "Kimi Räikkönen"
    assert ix.canonicalize("what about sergio perez?") == "what about Sergio Pérez?"


def test_other_ways_of_writing_a_name(ix):
    """The surname alone must not be swapped in behind a given name the bios
    spell differently ("Juan Manuel Juan Fangio")."""
    assert ix.canonicalize("Tell me about Juan Manuel Fangio") == "Tell me about Juan Fangio"
    assert ix.canonicalize("Is Kimi Antonelli fast?") == "Is Andrea Kimi Antonelli fast?"
    assert ix.canonicalize("Tell me about Alex Albon") == "Tell me about Alexander Albon"
    assert ix.canonicalize("Is Checo Perez any good?") == "Is Sergio Pérez any good?"
    assert ix.canonicalize("Is Sir Jackie Stewart still around?") == \
        "Is Sir Jackie Stewart still around?"


def test_canonical_text_is_left_alone(ix):
    for q in ("How many wins does Max Verstappen have?",
              "What teams did Sebastian Vettel race for over his F1 career?",
              "Who won the 2008 Singapore Grand Prix?",
              "Explain DRS like I'm five", "thanks!"):
        assert ix.canonicalize(q) == q
    # running it twice changes nothing
    once = ix.canonicalize("Is Hamilton better than Verstappen?")
    assert ix.canonicalize(once) == once


def test_surnames_that_are_words_need_a_capital_mid_sentence(ix):
    for q in ("the race pace was good", "what is the hill at spa called",
              "push the button", "Mass damper rules", "a stroll through the paddock"):
        assert ix.canonicalize(q) == q
    assert (ix.canonicalize("Was Hunt quicker than Lauda?")
            == "Was James Hunt quicker than Niki Lauda?")
    assert ix.canonicalize("Did Button win in 2009?") == "Did Jenson Button win in 2009?"


def test_words_around_a_surname_are_not_swallowed(ix):
    assert (ix.canonicalize("Will Verstappen win the title?")
            == "Will Max Verstappen win the title?")
    assert ix.canonicalize("when will hamilton retire") == "when will Lewis Hamilton retire"
    assert (ix.canonicalize("Who won the 1988 Monaco Grand Prix Senna or Prost?")
            == "Who won the 1988 Monaco Grand Prix Ayrton Senna or Alain Prost?")
    assert (ix.canonicalize("Which Ferrari driver won, Leclerc or Sainz?")
            == "Which Ferrari driver won, Charles Leclerc or Carlos Sainz?")


def test_a_team_or_circuit_name_is_not_a_driver(ix):
    for q in ("Who drives for McLaren?", "Tell me about Brabham",
              "Tell me about the Autódromo Hermanos Rodríguez"):
        assert ix.canonicalize(q) == q
    assert ix.drivers_in("Who drives for McLaren?") == []


def test_one_slip_in_a_household_name(ix):
    for typed, name in (("who is verstapen", "Max Verstappen"),
                        ("tell me about hamiltion", "Lewis Hamilton"),
                        ("schumaker titles", "Michael Schumacher"),
                        ("Tell me about Hulkenburg", "Nico Hülkenberg")):
        assert [d.name for d in ix.drivers_in(typed)] == [name], typed
    for q in ("the steward gave a penalty", "who is the sergeant"):
        assert ix.drivers_in(q) == [], q


def test_teams_and_circuits(ix):
    assert ix.team_in("Who drives for Ferrari?") == "Ferrari"
    assert ix.team_in("Who drives for Red Bull Racing?") == "Red Bull"
    assert ix.team_in("Racing Bulls drivers") == "RB F1 Team"
    assert ix.team_in("Tell me about Haas") == "Haas F1 Team"
    assert ix.team_in("Who is the GOAT?") is None
    assert ix.circuit_in("Why is Monaco so special?") == "Circuit de Monaco"
    assert ix.circuit_in("Tell me about Spa") == "Circuit de Spa-Francorchamps"
    assert ix.circuit_in("what about interlagos") == "Autódromo José Carlos Pace"
    assert ix.circuit_in("what a charade") is None


def test_names_something_is_the_f1_cue(ix):
    for q in ("Tell me about Hamilton", "Who drives for Ferrari?",
              "Why is Monaco so special?", "who is alonso"):
        assert ix.names_something(q), q
    for q in ("thanks!", "What is the capital of France?", "Who is the GOAT?",
              "the race pace was good"):
        assert not ix.names_something(q), q


def test_the_first_and_the_latest_race_on_record(ix):
    assert ix.first_race()["text"].startswith("The 1950 British Grand Prix")
    latest = max(int(race.year) for race, _ in ix.races)
    assert ix.latest_race()["text"].startswith(f"The {latest} ")
    in_2021 = lambda race, doc: race.year == "2021"                          # noqa: E731
    assert ix.first_race(in_2021)["text"].startswith("The 2021 Bahrain Grand Prix")
    assert ix.latest_race(in_2021)["text"].startswith("The 2021 Abu Dhabi Grand Prix")
    assert ix.first_race(lambda race, doc: race.year == "1949") is None


def test_lineup_and_curated_lookups(ix):
    latest = ix.lineup("Ferrari")
    assert "Lewis Hamilton" in latest["text"] and "Charles Leclerc" in latest["text"]
    assert "Carlos Sainz" in ix.lineup("Ferrari", "2024")["text"]
    assert ix.lineup("Ferrari", "1999") is None          # no line-up docs that far back
    assert ix.curated_about("Ferrari")["text"].startswith("Ferrari (Scuderia Ferrari)")
    assert ix.curated_about("Monaco")["text"].startswith("The Monaco Grand Prix")
    assert ix.curated_about("Haas") is None
