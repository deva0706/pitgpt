"""Tests for the favourite-lookup recall path (engine.py).

Measured 2026-07-07: "Who is my favorite driver?" after an Alonso statement
declined even though retrieval returned the Alonso bio — the recall-block
context shape is untrained. The lookup now answers the recall half
deterministically and generates the data card for the resolved name.
"""

from __future__ import annotations

from pitgpt.inference import engine as E

HIST = [
    {"role": "user", "content": "My favorite driver is Fernando Alonso, ok?"},
    {"role": "assistant", "content": "Ah, Fernando Alonso — pull up a chair!"},
]


def test_fav_lookup_resolves_stated_name():
    got = E._favourite_lookup("Who is my favorite driver?", HIST)
    assert got is not None
    prefix, rewritten = got
    assert "Fernando Alonso" in prefix and "favourite driver" in prefix
    assert rewritten == "Tell me about Fernando Alonso."


def test_fav_lookup_contracted_and_past_phrasings():
    for q in ("Who's my favourite driver?", "Who was my favorite driver?"):
        got = E._favourite_lookup(q, HIST)
        assert got is not None and got[1] is not None, q
        assert "Fernando Alonso" in got[1], q


def test_fav_lookup_without_statement_is_honest():
    got = E._favourite_lookup(
        "Who is my favourite driver?",
        [{"role": "user", "content": "Hello!"}])
    assert got is not None
    prefix, rewritten = got
    assert rewritten is None and "nothing on that" in prefix


def test_fav_lookup_ignores_other_questions():
    assert E._favourite_lookup("Who is Fernando Alonso?", HIST) is None
    assert E._favourite_lookup(
        "What teams did my favourite driver race for?", HIST) is None
    assert E._favourite_lookup("Who is my favorite driver?", None) is None


def test_radio_log_answers_what_have_i_told():
    reply = E._radio_log_reply(
        "What have I told you about my favourites so far?", HIST)
    assert reply is not None and "Fernando Alonso" in reply


def test_fav_lookup_survives_long_chats_and_ignores_answer_names():
    """Measured live: after five exchanges the Alonso statement left the
    8-message recall window and the lookup grabbed a name from the bot's own
    Monza reply — 'your favourite driver on record is Autodromo Nazionale'.
    The statement must win however far back it is, and names from answer
    turns must never qualify."""
    long_hist = [
        {"role": "user", "content": "My favorite driver is Fernando Alonso okay?"},
        {"role": "assistant", "content": "Honestly? I go with Lewis Hamilton."},
        {"role": "user", "content": "How many races did he win?"},
        {"role": "assistant", "content": "The tally board reads 32 — Fernando Alonso's wins."},
        {"role": "user", "content": "Tell me about the 2021 Abu Dhabi Grand Prix."},
        {"role": "assistant", "content": "Max Verstappen got the chequered flag!"},
        {"role": "user", "content": "Who won the F1 World Championship in 2021?"},
        {"role": "assistant", "content": "And that's the title! Max Verstappen."},
        {"role": "user", "content": "Who won the 2019 Monza Grand Prix?"},
        {"role": "assistant", "content": "Charles Leclerc got the chequered flag at the "
         "2019 Italian Grand Prix! Held at Autodromo Nazionale di Monza."},
    ]
    got = E._favourite_lookup("Who is my favorite driver?", long_hist)
    assert got is not None
    prefix, rewritten = got
    assert "Fernando Alonso" in prefix
    assert rewritten == "Tell me about Fernando Alonso."
    assert "Autodromo" not in prefix


def test_fav_lookup_rejects_title_phrases_as_names():
    """Measured live: a favourite stated by DESCRIPTION ('...the World
    Champion') made the lookup answer 'your favourite driver on record is
    World Champion' and card the wrong entity. Title phrases are not names;
    with no proper name stated, the honest ask-me reply must fire."""
    hist = [
        {"role": "user",
         "content": "My favourite driver is the World Champion."},
        {"role": "assistant", "content": "Love the confidence!"},
    ]
    got = E._favourite_lookup("Who is my favorite driver?", hist)
    assert got is not None
    prefix, rewritten = got
    assert rewritten is None and "nothing on that" in prefix

    # A real name in the same sentence still wins.
    hist2 = [
        {"role": "user",
         "content": "My favourite driver is the World Champion, Max Verstappen."},
        {"role": "assistant", "content": "Good pick!"},
    ]
    got2 = E._favourite_lookup("Who is my favorite driver?", hist2)
    assert got2 is not None and got2[1] == "Tell me about Max Verstappen."


# --------------------------------------------------------------------------- #
# What a follow-up question refers back to (_chat_recall)
#
# All measured 2026-10-11 by replaying whole chats through the real model:
# the single-question benchmark never sees a second turn.
# --------------------------------------------------------------------------- #

def _chat(*pairs):
    out = []
    for q, a in pairs:
        out += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return out


RACE = _chat((
    "Who won the 2019 Chinese Grand Prix?",
    "Lewis Hamilton got the chequered flag at the 2019 Chinese Grand Prix! Driving "
    "for Mercedes. Pole position: Valtteri Bottas. Podium completed by Valtteri "
    "Bottas and Sebastian Vettel. Held at Shanghai International Circuit in "
    "Shanghai, China."))
TITLE = _chat((
    "Which driver took the F1 title in 1992?",
    "And that's the title! Nigel Mansell is the 1992 World Champion — what a "
    "season that was."))
SINGAPORE = _chat((
    "Who won the 2008 Singapore Grand Prix?",
    "Fernando Alonso got the chequered flag at the 2008 Singapore Grand Prix! "
    "Driving for Renault. Pole position: Felipe Massa. Podium completed by Nico "
    "Rosberg and Lewis Hamilton. Held at Marina Bay Street Circuit in Marina "
    "Bay, Singapore."))
FAVOURITE = _chat(("My favourite driver is Sebastian Vettel.", "Great pick!"))


def test_named_question_binds_its_own_pronoun():
    """The benchmark's own phrasing, asked third in a chat: "his" recalled the
    two previous answers, "World Champion" among their names sent the search
    to the champion docs, and the card came back as "2013 driver season"."""
    q = "What teams did Sebastian Vettel race for over his F1 career?"
    assert E._chat_recall(q, RACE + TITLE) == ("", q)
    q = "How many races did Sebastian Vettel win when he drove for Ferrari?"
    assert E._chat_recall(q, TITLE) == ("", q)


def test_he_is_the_subject_of_the_last_answer():
    """After the title answer "What teams did he race for?" carded Lewis
    Hamilton as a "Brazilian driver... Died 1994"; after the race recap it
    carded Valtteri Bottas, a podium finisher."""
    q = "What teams did he race for?"
    assert E._chat_recall(q, TITLE) == ("", "What teams did Nigel Mansell race for?")
    assert E._chat_recall(q, RACE) == ("", "What teams did Lewis Hamilton race for?")
    # the latest answer's subject wins over an older turn's
    assert (E._chat_recall("How many races did he win?", TITLE + RACE)[1]
            == "How many races did Lewis Hamilton win?")
    assert (E._chat_recall("What was his best season?", TITLE)[1]
            == "What was Nigel Mansell's best season?")


def test_he_survives_a_nameless_reply():
    """A reply that names nobody (persona chat, a decline) must not strand
    the next "he": it fell through to an arbitrary driver's win tally."""
    hist = _chat(
        ("Tell me about Sebastian Vettel",
         "Checking the telemetry on Sebastian Vettel — German driver, raced 2007–2022."),
        ("Is that impressive?", "Oh, this one gets me every time — proper racing!"))
    assert (E._chat_recall("How many races did he win?", hist)[1]
            == "How many races did Sebastian Vettel win?")


def test_year_followup_reasks_the_previous_question():
    """"And in 2007?" carries a year and nothing else; after two Singapore GP
    questions it got a blended 2007 season recap."""
    assert E._year_followup("And in 2009?", SINGAPORE) == "Who won the 2009 Singapore Grand Prix?"
    assert E._year_followup("And in 2010?", TITLE) == "Which driver took the F1 title in 2010?"
    # a chain re-asks the last FULL question, not the previous follow-up
    chain = SINGAPORE + _chat(("And in 2009?", "Lewis Hamilton got the chequered flag!"))
    assert E._year_followup("What about 2007?", chain) == "Who won the 2007 Singapore Grand Prix?"
    # anything more than a bare year is an ordinary question
    assert E._year_followup("And who won in 2009?", SINGAPORE) is None
    assert E._year_followup("And in 2009?", FAVOURITE) is None    # nothing dated to re-ask
    assert E._year_followup("And in 2009?", None) is None


def test_stated_favourite_does_not_hijack_he():
    """Once a favourite was on file, every later "he" resolved to it."""
    hist = FAVOURITE + RACE
    assert (E._chat_recall("What teams did he race for?", hist)[1]
            == "What teams did Lewis Hamilton race for?")
    for q in ("What teams did my favourite driver race for?",
              "What teams did my favourite F1 driver race for?"):
        assert E._chat_recall(q, hist) == ("", "What teams did Sebastian Vettel race for?"), q


def test_that_race_resolves_to_its_year_and_grand_prix():
    """"...in that race?" used to drag the venue name into routing (circuit
    docs) and got a title-fight ramble back; with the right doc but the bare
    question, the model recapped a different Grand Prix."""
    assert (E._chat_recall("Did Sebastian Vettel finish on the podium in that race?", RACE)[1]
            == "Did Sebastian Vettel finish on the podium in the 2019 Chinese Grand Prix?")


def test_it_takes_the_race_but_keeps_the_question_year():
    assert (E._chat_recall("Who won it in 2009?", SINGAPORE)[1]
            == "Who won the Singapore Grand Prix in 2009?")
    assert (E._chat_recall("Who won it in 2007?", SINGAPORE)[1]
            == "Who won the Singapore Grand Prix in 2007?")


def test_generic_they_and_it_are_left_alone():
    """Only references with one sensible reading are resolved: these would be
    bent into questions about the last driver or race."""
    for q in ("What tyres do they use in the wet?",
              "What does it mean when a driver gets a blue flag?",
              "Is it true that Hamilton has seven titles?",
              "Which driver crashed his car on the formation lap?"):   # bound in the question
        assert E._chat_recall(q, RACE) == ("", q), q


def test_same_year_followup_inherits_the_race_only_if_it_is_about_a_race():
    """A recalled GP used to reach the absence guard on ANY same-year
    question, so a title question after a race recap was answered with the
    race."""
    assert (E._chat_recall("Who was on pole in 2008?", SINGAPORE)[1]
            == "Who was on pole in 2008 at the Singapore Grand Prix?")
    for q in ("Which driver took the title in 2008?",       # another doc family
              "Tell me about the 2008 season",              # not about one race
              "Who was on pole at the 2008 Monaco GP?"):    # names its own race
        assert E._chat_recall(q, SINGAPORE) == ("", q), q


def test_title_phrases_are_not_names():
    assert E._real_names("Nigel Mansell is the 1992 World Champion") == ["Nigel Mansell"]
    # ...but a named race is
    assert E._real_names("the Monaco Grand Prix") == ["Monaco Grand Prix"]


def test_radio_log_quotes_plain_preference_statements():
    """"I also love the Monaco GP." was missing from "What have I told you so
    far?" — only statements with a recall cue ("my favourite") were logged."""
    hist = _chat(
        ("My favourite driver is Sebastian Vettel.", "Great pick!"),
        ("I also love the Monaco GP.", "Ah, Monaco!"),
        ("I really like Lando Norris too", "Good taste."),
        ("Who won the 2019 Chinese Grand Prix?", "Lewis Hamilton got the chequered flag!"),
    )
    reply = E._radio_log_reply("What have I told you so far?", hist)
    assert reply is not None
    for said in ("Sebastian Vettel", "Monaco GP", "Lando Norris"):
        assert said in reply, said
    assert "Chinese Grand Prix" not in reply        # questions are never quoted
