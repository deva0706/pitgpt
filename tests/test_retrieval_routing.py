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


def test_gp_matcher_handles_short_venue_names():
    """Words under 5 chars produced NO stems, so 'Spa Grand Prix' matched
    nothing and the guard declared the 2018 Belgian GP never-held (measured).
    Short words must match as whole words — a bare prefix would make 'Spa'
    hit 'SPAnish'."""
    m = R._gp_matcher("Spa Grand Prix")
    assert m("the 2018 belgian grand prix ... held at circuit de "
             "spa-francorchamps in spa, belgium")
    assert not m("the 2018 spanish grand prix was won by lewis hamilton")


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


RACE_2022_AD = {
    "text": ("The 2022 Abu Dhabi Grand Prix was won by Max Verstappen, driving "
             "for Red Bull. Held at Yas Marina Circuit in Abu Dhabi, UAE."),
    "source": "kaggle_race",
}
RACE_2022_SG = {
    "text": ("The 2022 Singapore Grand Prix was won by Sergio Perez, driving "
             "for Red Bull. Held at Marina Bay Street Circuit in Marina Bay, "
             "Singapore."),
    "source": "kaggle_race",
}


def test_alias_pin_prefers_doc_matching_more_name_words(fake_retriever):
    """'Yas Marina' matched the SINGAPORE doc via 'Marina Bay' (measured:
    2022 Yas Marina GP answered with the Singapore result). The doc matching
    both name words must beat the doc matching one."""
    fake_retriever([RACE_2022_AD, RACE_2022_SG], [RACE_2022_SG, RACE_2022_AD])
    facts = R.retrieve_facts("Who won the 2022 Yas Marina Grand Prix?")
    assert facts and "Abu Dhabi" in facts[0]["text"]
    assert not any("Singapore" in d["text"] for d in facts)


def test_alias_pin_marina_bay_still_reaches_singapore(fake_retriever):
    fake_retriever([RACE_2022_AD, RACE_2022_SG], [RACE_2022_AD, RACE_2022_SG])
    facts = R.retrieve_facts("Who won the 2022 Marina Bay Grand Prix?")
    assert facts and "Singapore" in facts[0]["text"]


def test_guard_still_fires_on_true_absence(fake_retriever):
    fake_retriever(CORPUS, [RACE_2021_BH])
    facts = R.retrieve_facts("Who won the 2020 Monaco Grand Prix?")
    assert len(facts) == 1 and facts[0]["source"] == "absence_guard"
    assert "2020" in facts[0]["text"] and "Monaco" in facts[0]["text"]


def test_gp_abbreviation_is_extracted_like_grand_prix():
    # Live-reproduced 2026-10-10: "2007 Singapore GP" skipped the guard.
    assert R._extract_gp("Who won the 2007 Singapore GP?") == "Singapore Grand Prix"
    assert R._extract_gp("Who won the 2007 Singapore Grand Prix?") == "Singapore Grand Prix"
    assert R._extract_gp("what makes a gp special?") is None


def test_guard_fires_for_gp_abbreviation(fake_retriever):
    fake_retriever(CORPUS, [RACE_2021_BH])
    facts = R.retrieve_facts("Who won the 2020 Monaco GP?")
    assert len(facts) == 1 and facts[0]["source"] == "absence_guard"
    assert "2020" in facts[0]["text"] and "Monaco" in facts[0]["text"]


def test_engine_answers_absence_without_the_model(fake_retriever):
    """Measured live 2026-10-11: the shipped adapter turned the guard's fact
    into "that one's not in my notes ... What I've got here covers the 2007
    Singapore Grand Prix, nothing else". The guard's verdict is now the reply
    itself — model=None proves nothing is generated."""
    from pitgpt.inference import engine as E

    fake_retriever(CORPUS, [RACE_2021_BH])
    want = ("Box, box — there's no 2020 Monaco Grand Prix on the record: "
            "that race wasn't held in 2020")
    for q in ("Who won the 2020 Monaco Grand Prix?",
              "who won the 2020 monaco gp"):          # lowercase is re-cased
        assert E.generate_reply(None, None, q).startswith(want), q
        assert list(E.stream_reply(None, None, q))[-1].startswith(want), q


def test_engine_still_grounds_real_races(fake_retriever):
    from pitgpt.inference import engine as E

    fake_retriever(CORPUS, [RACE_2021_AD])
    q = "Who won the 2021 Abu Dhabi Grand Prix?"
    context, absence = E._rag_context(q, True)
    assert absence is None and "Verstappen" in context
    assert E._rag_context(q, False) == ("", None)


def test_it_in_a_never_held_year_reaches_the_guard(fake_retriever):
    """"Who won it in 2007?" after the 2008 Singapore recap: the Grand Prix
    comes from the chat, the year from the question, and the guard answers
    (before, "And in 2007?"-style follow-ups got a blended 2007 season)."""
    from pitgpt.inference import engine as E

    fake_retriever(CORPUS, [RACE_2021_BH])
    hist = [{"role": "user", "content": "Who won the 2008 Singapore Grand Prix?"},
            {"role": "assistant", "content": "Fernando Alonso got the chequered "
             "flag at the 2008 Singapore Grand Prix! Driving for Renault."}]
    for q in ("Who won it in 2007?", "And in 2007?"):
        reply = E.generate_reply(None, None, q, history=hist)
        assert reply.startswith(
            "Box, box — there's no 2007 Singapore Grand Prix on the record"), q
        assert list(E.stream_reply(None, None, q, hist))[-1] == reply, q


def test_yearless_title_question_gets_the_titles_summary(fake_retriever):
    """Measured 2026-10-11: "How many titles did Lewis Hamilton win?" was
    answered from four of his seven season docs as "the 2017, 2019 and 2017
    World Champion"."""
    summary = {"text": "Lewis Hamilton has won seven Formula 1 World Drivers' "
               "Championships, tying Michael Schumacher's record.", "source": "curated"}
    other = {"text": "Lewis Hamilton became the first Black driver in Formula 1.",
             "source": "curated"}
    vague = {"text": "Max Verstappen is a multiple-time World Champion with Red "
             "Bull Racing; his first title came in 2021.", "source": "curated"}
    y2008 = {"text": "In 2008, Lewis Hamilton won the Formula 1 World Drivers' "
             "Championship, becoming the 2008 F1 champion.", "source": "kaggle_champion"}
    y2021 = {"text": "In 2021, Max Verstappen won the Formula 1 World Drivers' "
             "Championship, becoming the 2021 F1 champion.", "source": "kaggle_champion"}
    fake_retriever([other, summary, vague, y2008, y2021], [y2008, y2021])

    assert R.retrieve_facts("How many titles did Lewis Hamilton win?") == [summary]
    # "when" needs seasons, which this summary doesn't give
    assert R.retrieve_facts("When did Lewis Hamilton win the title?") == [y2008]
    # a dated title question still gets its season doc...
    assert R.retrieve_facts("Did Lewis Hamilton win the title in 2008?") == [y2008]
    # ...and so does a driver whose summary doesn't say how many
    assert R.retrieve_facts("How many titles did Max Verstappen win?") == [y2021]


def test_when_question_needs_a_summary_that_dates_the_titles(fake_retriever):
    """A year in the summary is not enough: the only one in Lauda's is his
    1976 crash, and he was not champion that season."""
    lauda = {"text": "Niki Lauda won three titles and famously returned to racing "
             "just weeks after a near-fatal fiery crash at the Nurburgring in 1976.",
             "source": "curated"}
    alonso = {"text": "Fernando Alonso won two World Championships, in 2005 and "
              "2006 with Renault.", "source": "curated"}

    def season(year, name):
        return {"text": f"In {year}, {name} won the Formula 1 World Drivers' "
                f"Championship, becoming the {year} F1 champion.",
                "source": "kaggle_champion"}

    l75, l84 = season(1975, "Niki Lauda"), season(1984, "Niki Lauda")
    a05, a06 = season(2005, "Fernando Alonso"), season(2006, "Fernando Alonso")
    fake_retriever([lauda, alonso, l75, l84, a05, a06], [l75, l84])

    assert R.retrieve_facts("When did Niki Lauda win his titles?") == [l75, l84]
    assert R.retrieve_facts("How many titles did Niki Lauda win?") == [lauda]
    assert R.retrieve_facts("When did Fernando Alonso win his titles?") == [alonso]


def test_title_and_tally_questions_are_not_about_the_last_race():
    for q in ("Which driver took the title in 2008?",
              "Who won the constructors' championship in 2008?",
              "How many races did he win?",
              "What teams did he race for?"):
        assert R._asks_beyond_the_race(q), q
    for q in ("Who was on pole in 2008?", "Who won it in 2009?",
              "Where was that race held?",
              "Tell me about the 2008 Singapore Grand Prix"):
        assert not R._asks_beyond_the_race(q), q


# --------------------------------------------------------------------------- #
# Chat vs. facts gate
# --------------------------------------------------------------------------- #

def test_persona_questions_and_statements_skip_retrieval():
    """Measured 2026-10-11: "I'm new to F1, where do I start?" was answered
    from Liam Lawson's bio and the Indianapolis circuit doc, "Who is your
    favourite driver and why?" from Jim Clark's, and "I really like Lando
    Norris too" with a telemetry card."""
    for q in ("I'm new to F1, where do I start?",
              "Who is your favourite driver and why?",
              "What do you think about the new regulations?",
              "I also love the Monaco GP.",
              "I really like Lando Norris too"):
        assert R._is_conversational(q), q
    for q in ("What do you think about the 2021 Abu Dhabi GP?",       # a dated race keeps its facts
              "Do you think Lewis Hamilton is the greatest driver?",  # so does a named driver
              "I'm new to F1. What is DRS?",                          # a real term is a real question
              "I would love to hear about the 2021 Abu Dhabi GP",     # a request, not a statement
              "I also love the Monaco GP — who won it in 2019?"):
        assert not R._is_conversational(q), q


def test_relevance_gate_needs_both_rankers_or_strong_similarity(monkeypatch):
    """Measured 2026-10-11: with "a signal from at least one ranker", three
    docs at cosine ~0.09 rode into "Explain DRS like I'm five" on BM25 token
    coincidences and were blended into the answer."""
    import numpy as np

    from pitgpt.rag.index import FactRetriever

    docs = [
        {"text": "DRS is a flap in the rear wing that reduces drag.", "source": "curated"},
        {"text": "Slipstreaming feels like five tows in a row, like five.", "source": "curated"},
        {"text": "A steering wheel has dozens of controls.", "source": "curated"},
        {"text": "The rear wing opens to shed drag on the straights.", "source": "curated"},
    ] + [{"text": f"Filler fact number {i} about nothing in particular.", "source": "curated"}
         for i in range(60)]
    dense = [0.45, 0.09, 0.40, 0.70] + [0.0] * 60
    retriever = FactRetriever()
    retriever._docs = docs
    retriever._embeddings = np.array([[d, 0.0] for d in dense], dtype=np.float32)
    retriever._build_bm25()
    monkeypatch.setattr(retriever, "_encode",
                        lambda texts: np.array([[1.0, 0.0]], dtype=np.float32))

    kept = [d["text"] for d in retriever.retrieve("Explain DRS like five", k=4)]
    assert docs[0]["text"] in kept          # shares a word AND is semantically close
    assert docs[3]["text"] in kept          # shares nothing, but strongly similar
    assert docs[1]["text"] not in kept      # token coincidence at cosine 0.09
    assert docs[2]["text"] not in kept      # moderately similar, shares nothing


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
