"""Regression tests for the query-routing / absence-guard fixes.

Live failures these lock in (all reproduced 2026-07-06):
1. "Tell me about {year} {GP}" routed to DRIVER docs, so the absence guard
   validated against the wrong doc family and declared real races "not held".
2. Circuit-alias race names (Yas Marina/Monza/Silverstone Grand Prix) could
   never match a race doc -> always a false absence.
3. Lowercase GP names ("2021 Yas marina Grand Prix") defeated the GP pin.
4. "What teams did X race for" let curated star docs crowd out the bio doc.
"""

from __future__ import annotations

import pytest

from pitgpt.rag import retriever as R


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #

def test_year_gp_query_routes_to_race_docs():
    for q in ("Tell me about the 2021 Abu Dhabi Grand Prix",
              "Tell me about 2021 Abu Dhabi Grand Prix",
              "Give me the rundown on the 2019 Monaco Grand Prix"):
        sources = R._route(q)
        assert sources and "kaggle_race" in sources, q


def test_championship_routes_unaffected_by_race_intent():
    for q in ("Tell me about the 2021 championship",
              "Which driver took the F1 title in 2016?",
              "Who won the 2016 championship?"):
        sources = R._route(q)
        assert sources and "kaggle_champion" in sources, q
        assert "kaggle_race" not in sources, q


def test_win_tally_still_routes_to_bio_docs():
    for q in ("What's Lewis Hamilton's career Grand Prix win tally?",
              "How many Grand Prix wins does Max Verstappen have?"):
        sources = R._route(q)
        assert sources == ["kaggle_driver", "jolpica_driver"], q


def test_teams_history_routes_to_bio_docs_only():
    """Curated star docs (titles prose) crowded Hamilton's bio out of top-k."""
    for q in ("What teams did Lewis Hamilton race for over his F1 career?",
              "Which teams has Kimi Raikkonen raced for?"):
        sources = R._route(q)
        assert sources == ["kaggle_driver", "jolpica_driver"], q


def test_driver_identity_route_keeps_curated_color():
    sources = R._route("Who is Fernando Alonso?")
    assert sources and "kaggle_driver" in sources and "curated" in sources


def test_circuit_questions_route_to_circuit_docs():
    sources = R._route("Tell me about the Suzuka Circuit")
    assert sources and "kaggle_circuit" in sources


# --------------------------------------------------------------------------- #
# GP-name extraction (the pin)
# --------------------------------------------------------------------------- #

def test_gp_pin_is_case_insensitive():
    gp = R._extract_gp("tell me about 2021 Yas marina Grand Prix")
    assert gp and gp.lower() == "yas marina grand prix"


def test_gp_pin_strips_stopwords_and_years():
    gp = R._extract_gp("who won the 2021 abu dhabi grand prix?")
    assert gp and gp.lower() == "abu dhabi grand prix"


def test_gp_pin_ignores_bare_grand_prix():
    assert R._extract_gp("what makes a grand prix special?") is None


def test_gp_matcher_distinguishes_austrian_from_australian():
    """5-char stems collided ('austr' ⊂ both): the 1990 Austrian GP query
    recovered the 1990 AUSTRALIAN GP doc instead of declaring absence."""
    m = R._gp_matcher("Austrian Grand Prix")
    assert m("the 1997 austrian grand prix was won by jacques villeneuve")
    assert not m("the 1990 australian grand prix was won by nelson piquet")


def test_gp_matcher_keeps_adjective_tolerance():
    assert R._gp_matcher("Singaporean Grand Prix")(
        "the 2010 singapore grand prix was won by fernando alonso")
    assert R._gp_matcher("Mexican Grand Prix")(
        "the 2022 mexico city grand prix was won by max verstappen")


# --------------------------------------------------------------------------- #
# Absence guard: corpus-wide verification + recovery
# --------------------------------------------------------------------------- #

RACE_2021_AD = {
    "text": ("The 2021 Abu Dhabi Grand Prix was won by Max Verstappen, driving "
             "for Red Bull. Pole position: Max Verstappen. Podium completed by "
             "Lewis Hamilton and Carlos Sainz. Held at Yas Marina Circuit in "
             "Abu Dhabi, UAE."),
    "source": "kaggle_race",
}
RACE_2021_BH = {
    "text": ("The 2021 Bahrain Grand Prix was won by Lewis Hamilton, driving "
             "for Mercedes. Held at Bahrain International Circuit in Sakhir, "
             "Bahrain."),
    "source": "kaggle_race",
}
DRIVER_DOC = {
    "text": ("Mick Schumacher is a German Formula 1 driver who raced from "
             "2021 to 2022 for Haas F1 Team."),
    "source": "kaggle_driver",
}
CORPUS = [RACE_2021_AD, RACE_2021_BH, DRIVER_DOC]


class _RankMissRetriever:
    """Simulates routing/ranking starvation: retrieve() never surfaces the
    matching race doc even though the corpus holds it."""

    def __init__(self, docs, returns):
        self._docs = docs
        self._returns = returns

    @property
    def docs(self):
        return self._docs

    def retrieve(self, query, k=4, sources=None):
        return list(self._returns)


@pytest.fixture
def fake_retriever(monkeypatch):
    def install(docs, returns):
        monkeypatch.setattr(R, "_retriever", _RankMissRetriever(docs, returns))
        monkeypatch.setattr(R, "_load_failed", False)
    return install


def test_guard_recovers_race_doc_missed_by_ranking(fake_retriever):
    fake_retriever(CORPUS, [DRIVER_DOC])
    facts = R.retrieve_facts("Tell me about the 2021 Abu Dhabi Grand Prix")
    assert any("Verstappen" in d["text"] for d in facts)
    assert all(d["source"] != "absence_guard" for d in facts)


def test_guard_recovers_circuit_alias_via_enriched_doc(fake_retriever):
    fake_retriever(CORPUS, [RACE_2021_BH])
    facts = R.retrieve_facts("Who won the 2021 Yas Marina Grand Prix?")
    assert any("Verstappen" in d["text"] for d in facts)
    assert all(d["source"] != "absence_guard" for d in facts)


def test_guard_still_fires_on_true_absence(fake_retriever):
    fake_retriever(CORPUS, [RACE_2021_BH])
    facts = R.retrieve_facts("Who won the 2020 Monaco Grand Prix?")
    assert len(facts) == 1 and facts[0]["source"] == "absence_guard"
    assert "2020" in facts[0]["text"] and "Monaco" in facts[0]["text"]


# --------------------------------------------------------------------------- #
# Corpus: race docs carry their circuit (alias bridge)
# --------------------------------------------------------------------------- #

def test_race_docs_carry_circuit_names():
    from pitgpt.rag.kb_builder import kaggle_documents

    docs = kaggle_documents()
    races = [d["text"] for d in docs if d["source"] == "kaggle_race"]
    abu_2021 = [t for t in races if t.startswith("The 2021 Abu Dhabi Grand Prix")]
    assert abu_2021 and "Yas Marina" in abu_2021[0]
    monza = [t for t in races if t.startswith("The 2019 Italian Grand Prix")]
    assert monza and "Monza" in monza[0]
