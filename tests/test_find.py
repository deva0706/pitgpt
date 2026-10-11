"""The one fact a plainly put question needs (rag/retriever.find).

Measured 2026-10-11 through the running app, when a question was searched
as typed and four documents went to the model:

* "Tell me about Hamilton", "Why is Monaco so special?", "What is an
  undercut?", "Who drives for Ferrari?" were not searched at all (no keyword
  in them), and were made up;
* "who is verstappen" got Jos and Max in one card;
* "How many wins does Max Verstappen have?" got his bio and three season
  entries: "63 — Max Verstappen takes victory at this chequered flag";
* "Is Hamilton better than Verstappen?" got a 2019 race recap.

The real index is used; the embedding search is stood in for.
"""

from __future__ import annotations

import pytest

from pitgpt.rag import retriever as R
from pitgpt.rag.entities import EntityIndex


@pytest.fixture
def found(monkeypatch):
    """What the embedding search finds: nothing, unless a test fills it in."""
    docs: list[dict] = []
    retriever = R._get_retriever()
    monkeypatch.setattr(retriever, "retrieve", lambda query, **kw: list(docs))
    monkeypatch.setattr(retriever, "closest", lambda query, candidates: candidates[0])
    return docs


def _texts(question: str) -> list[str]:
    return [d["text"] for d in R.find(question).facts]


# --------------------------------------------------------------------------- #
# Is it worth a search at all?
# --------------------------------------------------------------------------- #

def test_every_question_is_searched_whatever_its_words():
    for q in ("Tell me about Hamilton", "Why is Monaco so special?", "What is an undercut?",
              "Who drives for Ferrari?", "Senna or Prost — who was better?",
              "Why was refuelling banned?", "What is the capital of France?"):
        assert not R._is_conversational(q), q
    for q in ("Hi!", "thanks!", "haha ok", "Thanks, bye!", "lol",
              "My favourite driver is Fernando Alonso.", "I'm new to F1, where do I start?"):
        assert R._is_conversational(q), q


def test_keywords_are_whole_words():
    """"ers" used to fire inside "Verstappen" and "Silverstone" — the only
    reason "who is verstappen" was searched while "Tell me about Hamilton"
    was not."""
    assert not R._F1_KEYWORDS.search("who is verstappen")
    assert not R._F1_KEYWORDS.search("my brothers and others teamed up")
    assert R._F1_KEYWORDS.search("how does ERS work")
    assert R._F1_KEYWORDS.search("Why was refuelling banned?")


def test_what_counts_as_f1_content():
    for q in ("who is verstappen", "Tell me about Hamilton", "Why is Monaco so special?",
              "Who drives for Ferrari?", "Who won in 2019?", "Tell me about Toto Wolff"):
        assert R.has_f1_content(q), q
    for q in ("What is the capital of France?", "hey, how are you?", "Who is the GOAT?",
              "What is towing?"):               # searched all the same, but strictly
        assert not R.has_f1_content(q), q


# --------------------------------------------------------------------------- #
# The one document
# --------------------------------------------------------------------------- #

def test_a_driver_question_gets_that_drivers_bio_and_nothing_else(found):
    for q in ("Tell me about Hamilton", "who is lewis hamilton",
              "How many wins does Lewis Hamilton have?",
              "What teams did Lewis Hamilton race for over his F1 career?",
              "What's Lewis Hamilton's career Grand Prix win tally?"):
        texts = _texts(q)
        assert len(texts) == 1 and texts[0].startswith("Lewis Hamilton is a British"), q
    assert _texts("who is verstappen")[0].startswith("Max Verstappen is a Dutch")
    assert R.find("Tell me about Hamilton").question == "Tell me about Lewis Hamilton"


def test_who_races_for_whom(found):
    assert _texts("Who drives for Ferrari?") == [
        d["text"] for d in [R._get_entities().lineup("Ferrari")]]
    assert "the Red Bull drivers were" in _texts("Who are the Red Bull drivers in 2025?")[0]
    assert "Carlos Sainz" in _texts("Who raced for Ferrari in 2024?")[0]
    # a year the line-ups do not reach: understood, and nothing to give
    seen = R.find("Who drove for Ferrari in 1999?")
    assert (seen.intent, seen.facts) == ("lineup", [])
    assert _texts("Which team does Hamilton drive for?")[0].startswith(
        "In the current Formula 1 season, Lewis Hamilton")
    assert _texts("Which team was Hamilton at in 2025?") == [
        "In the 2025 Formula 1 season, Lewis Hamilton raced for Ferrari."]


def test_two_drivers_side_by_side_are_their_two_bios(found):
    seen = R.find("Is Hamilton better than Verstappen?")
    assert seen.intent == "compare"
    assert [t.split(" is a ")[0] for t in _texts("Is Hamilton better than Verstappen?")] == [
        "Lewis Hamilton", "Max Verstappen"]
    assert R.find("Senna or Prost — who was better?").intent == "compare"


def test_a_race_however_it_is_named(found):
    assert _texts("Who won at Monza in 2019?")[0].startswith("The 2019 Italian Grand Prix")
    assert _texts("who won in 2019 at spa")[0].startswith("The 2019 Belgian Grand Prix")
    latest = R._get_entities().latest_race()["text"]
    assert _texts("Who won the last race?") == [latest]
    assert _texts("What was the most recent grand prix?") == [latest]
    # no year: the latest running, which says which one it is
    monaco = _texts("Who won the Monaco Grand Prix?")
    assert len(monaco) == 1 and "Monaco Grand Prix was won by" in monaco[0]
    assert monaco == [R._latest_race(R._get_entities(), "Monaco Grand Prix")["text"]]


def test_the_first_and_the_last_race(found):
    """"Who won the first ever F1 race?" (typed into the app, 2026-10-11) was
    searched as it stood: four champion docs, and a decline. Which race is
    the first or the last — of them all, of a season, of one Grand Prix, of a
    driver's wins — is for the records to say."""
    def race(question: str) -> str:
        texts = _texts(question)
        assert len(texts) == 1, (question, texts)
        return texts[0].split(" was won by ")[0]

    assert race("Who won the first ever F1 race?") == "The 1950 British Grand Prix"
    assert race("who won the very first grand prix") == "The 1950 British Grand Prix"
    assert race("Who won the first race of 2021?") == "The 2021 Bahrain Grand Prix"
    assert race("Who won the last race of 2021?") == "The 2021 Abu Dhabi Grand Prix"
    assert race("who won the 2008 season finale") == "The 2008 Brazilian Grand Prix"
    assert race("Who won the first Monaco Grand Prix?") == "The 1950 Monaco Grand Prix"
    assert race("What was the first race at Silverstone?") == "The 1950 British Grand Prix"
    # a driver's first and last win
    assert race("When did Hamilton win his first race?") == "The 2007 Canadian Grand Prix"
    assert race("What was Senna's last win?") == "The 1993 Australian Grand Prix"
    assert race("Verstappen's first win of 2023?") == "The 2023 Bahrain Grand Prix"
    # ...which is not his first START: nothing here knows who entered a race
    assert R.find("What was Hamilton's first race?").intent != "race"
    # ...and a team's
    assert race("Who was the first driver to win a Grand Prix for Red Bull?") == (
        "The 2009 Chinese Grand Prix")
    assert race("When did Williams last win a race?") == "The 2012 Spanish Grand Prix"
    # the race itself, whoever is asked about
    latest = R._get_entities().latest_race()["text"]
    assert _texts("Did Hamilton win the last race?") == [latest]
    # "first" and "last" about something else
    for q in ("What happens on the last lap of a race?",
              "Who was the first Japanese driver to start a race?"):
        assert R.find(q).intent != "race", q


def test_a_race_named_by_where_it_was_held(found):
    """"Who won in Brazil in 2008?" reached the model with the race, that
    year's constructors' champion and the Malaysian Grand Prix."""
    assert _texts("Who won in Brazil in 2008?")[0].startswith("The 2008 Brazilian Grand Prix")
    assert len(_texts("Who won in Brazil in 2008?")) == 1
    assert _texts("who won in sao paulo in 2008")[0].startswith("The 2008 Brazilian Grand Prix")
    assert _texts("Who won in Britain in 1950?")[0].startswith("The 1950 British Grand Prix")
    # three races in Italy that year: the place does not say which
    assert R.find("Who won in Italy in 2020?").intent != "race"


STANDINGS = [
    "After round 17 of the 2026 Formula 1 season, Andrea Kimi Antonelli leads the Drivers' "
    "Championship with 325 points, ahead of George Russell on 236 and Lewis Hamilton on 221.",
    "After round 17 of the 2026 Formula 1 season, Mercedes leads the Constructors' "
    "Championship with 561 points, ahead of Ferrari on 418 and McLaren on 310.",
    "After round 17 of the 2026 Formula 1 season, Andrea Kimi Antonelli is 1st in the "
    "Drivers' Championship with 325 points.",
    "After round 17 of the 2026 Formula 1 season, Lewis Hamilton is 3rd in the Drivers' "
    "Championship with 221 points.",
]


def _season(monkeypatch, standings: list[str]) -> None:
    """Make the season in progress (or, with no standings, over), whatever
    the shipped index holds."""
    retriever = R._get_retriever()
    docs = [d for d in retriever.docs if d.get("source") != "jolpica_standings"]
    docs += [{"text": t, "source": "jolpica_standings"} for t in standings]
    monkeypatch.setattr(R, "_entities", (retriever, EntityIndex(docs)))


def test_how_the_championship_stands(found, monkeypatch):
    """Typed into the app in October 2026: "Who is leading the championship?"
    -> "And that's the title! Max Verstappen is the 2024 World Champion";
    "How many points does Hamilton have?" -> his career card."""
    _season(monkeypatch, STANDINGS)
    drivers, teams, antonelli, hamilton = STANDINGS
    for q in ("Who is leading the championship?", "who's top of the standings",
              "What are the current standings?", "Who has the most points?",
              "Who is leading the title race?"):
        seen = R.find(q)
        assert (seen.intent, [d["text"] for d in seen.facts]) == ("standings", [drivers]), q
    assert _texts("Who leads the constructors' championship?") == [teams]
    assert _texts("Which team is top of the standings?") == [teams]
    # one driver's place and points
    assert _texts("How many points does Hamilton have?") == [hamilton]
    assert _texts("Where is Hamilton in the standings?") == [hamilton]
    assert _texts("Is Antonelli leading the championship?") == [antonelli]
    assert _texts("Who is leading the championship in 2026?") == [drivers]
    # not how things stand now
    for q in ("How many points did Hamilton score in his career?", "Who is leading the race?",
              "Who led the championship in 2010?", "How many points for a win?"):
        assert R.find(q).intent != "standings", q


def test_the_reigning_champion_is_the_latest_on_record(found, monkeypatch):
    """"Who is the current world champion?" found four champions, and the
    model called 2024's."""
    from pitgpt.rag.records import parse_champion, parse_constructors

    def latest(parse) -> str:
        titles = [(t.year, d["text"]) for d in R._get_retriever().docs
                  if (t := parse(d["text"]))]
        return max(titles)[1]

    for q in ("Who is the current world champion?", "Who is the reigning champion?",
              "Who is the world champion?", "who won the last championship"):
        assert _texts(q) == [latest(parse_champion)], q
    assert _texts("Who are the reigning constructors' champions?") == [latest(parse_constructors)]
    # between seasons there are no standings: the champion is who "leads"
    _season(monkeypatch, [])
    assert _texts("Who is leading the championship?") == [latest(parse_champion)]


def test_a_circuit_or_a_team_by_name(found):
    assert _texts("Tell me about Spa")[0].startswith("Spa-Francorchamps in Belgium")
    assert _texts("And Spa?")[0].startswith("Spa-Francorchamps in Belgium")
    assert _texts("Where is Suzuka?") == [
        "Suzuka Circuit is a Formula 1 circuit located in Suzuka, Japan."]
    assert _texts("Tell me about Ferrari")[0].startswith("Ferrari (Scuderia Ferrari)")
    assert _texts("Tell me about the Monaco Grand Prix")[0].startswith("The Monaco Grand Prix")


def test_what_has_no_document(found):
    assert R.find("When is the next race?").intent == "schedule"
    assert R.find("What time does qualifying start?").intent == "schedule"
    assert R.find("Who won the 2022 football world cup?").intent == "off_topic"
    counted = R.find("Who has the most wins in F1 history?")
    assert counted.intent == "stat" and counted.facts[0]["source"] == "computed"
    # a tally narrowed in a way nothing here counts: no facts rather than the career total
    seen = R.find("What is Lewis Hamilton's win tally at Monza?")
    assert (seen.intent, seen.facts) == ("tally", [])


# --------------------------------------------------------------------------- #
# When the search does run
# --------------------------------------------------------------------------- #

def _doc(text: str, source: str, dense: float, bm25: float) -> dict:
    return {"text": text, "source": source, "dense": dense, "bm25": bm25}


POINTS = _doc("Formula 1 points are scored by the top ten finishers on a sliding scale.",
              "curated", 0.635, 10.43)
WHEEL = _doc("An F1 steering wheel is a computer with dozens of controls.", "curated", 0.447, 2.57)
DISTANCE = _doc("F1 race distance is the fewest laps exceeding 305 km.", "curated", 0.587, 2.62)
POWER = _doc("A modern Formula 1 power unit is a 1.6-litre V6 turbo-hybrid.", "curated", 0.366, 10.23)
LINEUP = _doc("In the 2026 Formula 1 season, the RB F1 Team drivers are Liam Lawson, Arvid "
              "Lindblad and Yuki Tsunoda.", "jolpica_lineup", 0.509, 10.48)
PIRELLI = _doc("Pirelli is Formula 1's sole tyre supplier.", "curated", 0.531, 5.97)


def test_a_note_both_rankers_put_first_stands_alone(found):
    """"How do points work in F1?" found the points note, then the steering
    wheel, the cost cap and a circuit — and the model wrote about tyre
    management."""
    found[:] = [POINTS, WHEEL]
    assert _texts("How do points work in F1?") == [POINTS["text"]]
    # nearest in meaning, but it shares none of the words: not plainly the answer
    found[:] = [DISTANCE, POWER]
    assert len(_texts("How fast do F1 cars go?")) == 2


def test_line_ups_stay_out_of_questions_that_name_nobody(found):
    """"What tyres are used in F1?" found the "RB F1 Team" line-up first."""
    found[:] = [LINEUP, PIRELLI]
    assert _texts("What tyres are used in F1?") == [PIRELLI["text"]]


def test_a_question_with_nothing_f1_in_it_needs_a_strong_match(found, monkeypatch):
    asked: dict = {}
    retriever = R._get_retriever()
    monkeypatch.setattr(retriever, "retrieve",
                        lambda query, **kw: asked.update(kw) or [])
    R.find("What is the capital of France?")
    assert asked["min_score"] == 0.5          # token overlap alone found four French circuits
    asked.clear()
    R.find("Why was refuelling banned?")
    assert "min_score" not in asked


def test_a_dated_title_question_gets_that_seasons_champion_or_a_verdict(found):
    latest = max(int(race.year) for race, _ in R._get_entities().races)
    assert _texts("Which driver took the title in 2016?") == [
        "In 2016, Nico Rosberg won the Formula 1 World Drivers' Championship, becoming the "
        "2016 F1 champion."]
    assert _texts("Which team won the constructors title in 2009?") == [
        "In 2009, Brawn won the Formula 1 Constructors' Championship."]
    pending = R.find(f"Who won the {latest} championship?")
    assert pending.intent == "absence" and "on the record yet" in pending.facts[0]["reply"]
    never = R.find("Who won the constructors title in 1955?")
    assert never.facts[0]["reply"] == (
        "Box, box — there's no 1955 Constructors' Champion on the record, "
        "so there's nothing to call.")


def test_a_race_the_records_have_not_reached_is_not_yet(found):
    latest = max(int(race.year) for race, _ in R._get_entities().races)
    seen = R.find(f"Who won the {latest + 4} Monaco Grand Prix?")
    assert seen.intent == "absence"
    assert seen.facts[0]["reply"] == (
        f"Box, box — there's no {latest + 4} Monaco Grand Prix on the record yet, "
        f"so I won't call it.")
    # a past one that was never held is still "wasn't held"
    assert "wasn't held in 2007" in R.find(
        "Who won the 2007 Singapore Grand Prix?").facts[0]["reply"]
