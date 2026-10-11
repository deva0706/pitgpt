"""The fact-document parsers (rag/records.py) against the real corpus.

Each parser is the inverse of a sentence one of the builders writes
(kb_builder.py, jolpica.py). Running them over every document of the shipped
index is what keeps the two in step: change a builder's phrasing and the
matching parser fails here, not silently in the answer check.
"""

from __future__ import annotations

import json
from pathlib import Path

from pitgpt.rag import records as R

DOCS = json.loads((Path(__file__).resolve().parents[1] / "data" / "knowledge"
                   / "rag_index" / "docs.json").read_text(encoding="utf-8"))

PARSERS = {
    "kaggle_race": R.parse_race, "jolpica_race": R.parse_race,
    "kaggle_driver": R.parse_bio,
    "kaggle_champion": R.parse_champion, "jolpica_champion": R.parse_champion,
    "kaggle_constructors": R.parse_constructors,
    "jolpica_constructors": R.parse_constructors,
    "kaggle_circuit": R.parse_circuit,
    "jolpica_lineup": R.parse_lineup,
    "jolpica_driver": R.parse_entry,
}


# In the index only while a season is being run.
SEASONAL = {"jolpica_standings": R.parse_standing}


def test_every_structured_document_parses():
    seen = set()
    for doc in DOCS:
        parser = PARSERS.get(doc["source"]) or SEASONAL.get(doc["source"])
        if parser:
            assert parser(doc["text"]) is not None, doc["text"]
            seen.add(doc["source"])
    assert seen - set(SEASONAL) == set(PARSERS)          # every family is in the index


def test_a_parser_only_claims_its_own_family():
    """A parser doubles as "is this a race doc?" — it must say no to the rest."""
    every = {**PARSERS, **SEASONAL}
    for doc in DOCS:
        # the current grid is a per-driver team doc too (its own source tag)
        own = R.parse_entry if doc["source"] == "openf1" else every.get(doc["source"])
        for parser in set(every.values()) - {own}:
            assert parser(doc["text"]) is None, (parser.__name__, doc["text"])


def test_standings_docs(monkeypatch):
    from pitgpt.rag import jolpica as J

    rows = {"driver": [("Andrea Kimi", "Antonelli", "325"), ("George", "Russell", "236"),
                       ("Lewis", "Hamilton", "221"), ("Jack", "Doohan", "1")],
            "constructor": [("Mercedes", "561"), ("Ferrari", "418")]}

    def table(year, whose):
        if whose == "driver":
            return "17", [{"points": p, "Driver": {"givenName": g, "familyName": f}}
                          for g, f, p in rows[whose]]
        return "17", [{"points": p, "Constructor": {"name": n}} for n, p in rows[whose]]

    monkeypatch.setattr(J, "_standings_after", table)
    monkeypatch.setattr(J, "_season_over", lambda year: False)
    texts = [d["text"] for d in J.standings_documents(2026)]
    assert texts[0] == (
        "After round 17 of the 2026 Formula 1 season, Andrea Kimi Antonelli leads the "
        "Drivers' Championship with 325 points, ahead of George Russell on 236 and Lewis "
        "Hamilton on 221.")
    assert texts[3] == ("After round 17 of the 2026 Formula 1 season, Lewis Hamilton is 3rd "
                        "in the Drivers' Championship with 221 points.")
    assert texts[4].endswith("Jack Doohan is 4th in the Drivers' Championship with 1 point.")
    assert texts[5] == ("After round 17 of the 2026 Formula 1 season, Mercedes leads the "
                        "Constructors' Championship with 561 points, ahead of Ferrari on 418.")
    assert len(texts) == 6
    assert R.parse_standing(texts[0]) == R.Standing(
        "2026", "17", "Drivers'", "Andrea Kimi Antonelli", 1, "325", True)
    assert R.parse_standing(texts[3]) == R.Standing(
        "2026", "17", "Drivers'", "Lewis Hamilton", 3, "221")
    assert all(R.parse_standing(t) for t in texts)
    # a finished season has champions instead
    monkeypatch.setattr(J, "_season_over", lambda year: True)
    assert J.standings_documents(2025) == []


def test_per_driver_team_docs():
    assert R.parse_entry(
        "In the 2025 Formula 1 season, Yuki Tsunoda raced for RB F1 Team, Red Bull."
    ) == R.Entry("Yuki Tsunoda", ("RB F1 Team", "Red Bull"), "2025")
    assert R.parse_entry(
        "In the current Formula 1 season, Lewis Hamilton, car number 44, races for Ferrari."
    ) == R.Entry("Lewis Hamilton", ("Ferrari",), "", "44")
    grid = [d["text"] for d in DOCS if d["source"] == "openf1" and "car number" in d["text"]]
    assert grid and all(R.parse_entry(t) for t in grid)


def test_race_fields():
    race = R.parse_race(
        "The 2008 Singapore Grand Prix was won by Fernando Alonso, driving for "
        "Renault. Pole position: Felipe Massa. Podium completed by Nico Rosberg "
        "and Lewis Hamilton. Held at Marina Bay Street Circuit in Marina Bay, "
        "Singapore.")
    assert race == R.Race("2008", "Singapore Grand Prix", "Fernando Alonso", "Renault",
                          "Felipe Massa", ("Nico Rosberg", "Lewis Hamilton"),
                          "Marina Bay Street Circuit in Marina Bay, Singapore")
    # every part after the winner is optional
    assert R.parse_race("The 1950 Indianapolis 500 was won by Johnnie Parsons.") == R.Race(
        "1950", "Indianapolis 500", "Johnnie Parsons")


def test_race_names_may_contain_a_full_stop():
    race = R.parse_race(
        "The 2008 German Grand Prix was won by Lewis Hamilton, driving for McLaren. "
        "Pole position: Lewis Hamilton. Podium completed by Nelson Piquet Jr. and "
        "Felipe Massa. Held at Hockenheimring in Hockenheim, Germany.")
    assert race.podium == ("Nelson Piquet Jr.", "Felipe Massa")
    assert race.venue == "Hockenheimring in Hockenheim, Germany"


def test_bio_fields():
    bio = R.parse_bio(
        "Nico Rosberg is a German Formula 1 driver who raced from 2006 to 2016 for "
        "Williams, Mercedes. Nico Rosberg won 23 Grands Prix. Nico Rosberg won the "
        "World Drivers' Championship in 2016.")
    assert bio == R.Bio("Nico Rosberg", "German", 2006, 2016,
                        ("Williams", "Mercedes"), 23, (2016,))
    assert bio.span == "2006–2016"
    one = R.parse_bio("Arvid Lindblad is a British Formula 1 driver who raced in 2026 "
                      "for RB F1 Team.")
    assert (one.span, one.wins, one.titles, one.teams) == ("2026", 0, (), ("RB F1 Team",))


def test_season_titles_circuits_and_lineups():
    assert R.parse_champion(
        "In 1992, Nigel Mansell won the Formula 1 World Drivers' Championship, "
        "becoming the 1992 F1 champion.") == R.SeasonTitle("1992", "Nigel Mansell")
    assert R.parse_constructors(
        "In 2009, Brawn won the Formula 1 Constructors' Championship."
    ) == R.SeasonTitle("2009", "Brawn")
    assert R.parse_circuit(
        "Circuit de Monaco is a Formula 1 circuit located in Monte-Carlo, Monaco."
    ) == R.Circuit("Circuit de Monaco", "Monte-Carlo, Monaco")
    assert R.parse_lineup(
        "In the 2025 Formula 1 season, the Red Bull drivers were Max Verstappen, "
        "Liam Lawson and Yuki Tsunoda."
    ) == R.Lineup("2025", "Red Bull", ("Max Verstappen", "Liam Lawson", "Yuki Tsunoda"))


def test_careers_run_past_the_historical_dump():
    """Measured 2026-10-11: bios came from the Kaggle tables alone, so Lewis
    Hamilton "raced 2007–2024" for McLaren and Mercedes while the same index
    held his Ferrari seasons."""
    bios = {b.name: b for b in (R.parse_bio(d["text"]) for d in DOCS
                                if d["source"] == "kaggle_driver")}
    hamilton = bios["Lewis Hamilton"]
    assert hamilton.last_year >= 2026 and hamilton.teams[-1] == "Ferrari"
    assert hamilton.wins >= 105
    assert 2025 in bios["Lando Norris"].titles
    # a driver who arrived after the dump has a bio of his own
    assert bios["Andrea Kimi Antonelli"].first_year == 2025
    # ...and nobody has two
    names = [R.parse_bio(d["text"]).name for d in DOCS if d["source"] == "kaggle_driver"]
    assert len(names) == len(set(names))
