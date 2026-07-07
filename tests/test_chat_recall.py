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
