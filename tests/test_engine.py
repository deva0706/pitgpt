"""What the engine does with a message (inference/engine.py), without a model.

The real index is used; the embedding search and the language model are
stood in for, so each test says what was found and what the model "said".
The model's lines are real ones, measured 2026-10-11.
"""

from __future__ import annotations

import pytest

from pitgpt.data import templates as T
from pitgpt.inference import engine as E
from pitgpt.rag import retriever as R

RECAP = ("Fernando Alonso got the chequered flag at the 2008 Singapore Grand Prix! "
         "Driving for Renault. Pole position: Felipe Massa. Podium completed by Nico "
         "Rosberg and Lewis Hamilton. Held at Marina Bay Street Circuit in Marina Bay, "
         "Singapore.")
SHIFTED = ("Felipe Massa got the chequered flag at the 2008 Singapore Grand Prix! Driving "
           "for Renault. Pole position: Nico Rosberg. Podium completed by Lewis Hamilton.")
DRS = {"text": "DRS (Drag Reduction System) is a driver-operated flap in the rear wing that "
       "reduces aerodynamic drag to boost straight-line speed and aid overtaking. It can "
       "only be used in designated zones when a driver is within one second of the car "
       "ahead, and it was introduced in 2011.", "source": "curated",
       "dense": 0.7, "bm25": 9.0}


def _chat(*pairs):
    out = []
    for q, a in pairs:
        out += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return out


@pytest.fixture
def found(monkeypatch):
    """Stand in for the embedding search: a test puts in this list what the
    search is to find (nothing, unless it says so)."""
    docs: list[dict] = []
    retriever = R._get_retriever()
    monkeypatch.setattr(retriever, "retrieve", lambda query, **kw: list(docs))
    monkeypatch.setattr(retriever, "closest", lambda query, candidates: candidates[0])
    monkeypatch.setattr(E, "_checked", {})
    return docs


def _model_says(monkeypatch, text: str, size: int = 7) -> dict:
    """Make the model "say" ``text`` a few characters at a time. Returns a
    record of how often it was asked and how much of it was read."""
    seen = {"calls": 0, "pieces": 0, "stopped": False}

    def stream(model, tokenizer, messages, grounded):
        seen["calls"] += 1
        seen["asked"] = messages[-1]["content"]
        try:
            for i in range(0, len(text), size):
                seen["pieces"] += 1
                yield text[i:i + size]
        finally:
            seen["stopped"] = True

    monkeypatch.setattr(E, "_generate_stream", stream)
    monkeypatch.setattr(E, "_generate", lambda model, tokenizer, messages, grounded: text)
    return seen


# --------------------------------------------------------------------------- #
# Replies that are decided without the model
# --------------------------------------------------------------------------- #

def test_what_the_app_is_is_not_improvised(found):
    """"Who are you and what can you do?" got a paragraph about loving
    wet-weather qualifying, cut off mid-sentence."""
    for q in ("Who are you and what can you do?", "who are you?", "What can you do?",
              "are you a bot?"):
        reply = E._prepare(q, None).reply
        assert reply and reply.startswith("I'm PitGPT — a Formula 1 race engineer"), q
    # ...but a question that merely starts that way is a question
    assert E._prepare("Who are you backing for the 2021 title?", None).reply != E._about_me()


def test_counts_and_comparisons_are_computed(found):
    assert E._prepare("Senna or Prost — who was better?", None).reply == (
        "Stack them up: Alain Prost takes it — 51 wins to 41, four titles to three.")
    assert E._prepare("Who has the most wins in F1 history?", None).reply.startswith(
        "Most Grand Prix wins on the record: Lewis Hamilton, with ")
    assert E._prepare("How many times did Hamilton win at Monza?", None).reply == (
        "Lewis Hamilton won 5 Grands Prix at Autodromo Nazionale di Monza: "
        "2012, 2014, 2015, 2017, 2018.")


def test_what_the_notes_cannot_hold_is_said_plainly(found):
    reply = E._prepare("When is the next race?", None).reply
    assert reply.startswith("Box, box — I don't carry a live calendar")
    assert "The latest race in my notes is the " in reply
    assert E._prepare("Who won the 2022 football world cup?", None).reply == T.OFF_TOPIC
    latest = max(int(race.year) for race, _ in R._get_entities().races)
    assert E._prepare(f"Who won the {latest + 4} Monaco Grand Prix?", None).reply == (
        f"Box, box — there's no {latest + 4} Monaco Grand Prix on the record yet, "
        f"so I won't call it.")
    assert E._prepare(f"Who won the {latest} championship?", None).reply == (
        f"Box, box — there's no {latest} World Champion on the record yet, so I won't call it.")
    assert E._prepare("Who won the constructors title in 1955?", None).reply == (
        "Box, box — there's no 1955 Constructors' Champion on the record, "
        "so there's nothing to call.")


def test_an_f1_question_with_no_facts_is_declined_not_improvised(found):
    """Left to free chat, "Who was on pole?" got a name out of an earlier
    reply, and a factual-sounding answer either way."""
    for q in ("Who was on pole?", "Tell me about Toto Wolff", "Who drove for Ferrari in 1999?",
              "What is Lewis Hamilton's win tally at Monza?"):
        assert E._prepare(q, None).reply == T.DECLINE, q


def test_small_talk_goes_to_the_persona(found):
    for q in ("I love F1", "I'm new to F1, where do I start?",
              "My favourite driver is Fernando Alonso.", "What is the capital of France?"):
        turn = E._prepare(q, None)
        assert turn.reply is None and turn.plan is None and turn.question == q, q


def test_the_persona_only_sees_earlier_small_talk(found):
    """With grounded exchanges in its prompt the persona imitates them:
    "Paris, my mistake — I didn't cover that one! What I helped you with is
    'Lewis Hamilton and Charles Leclerc', nothing else."."""
    history = _chat(("Hi!", "Hello there!"),
                    ("Who drives for Ferrari?", "In the 2026 Formula 1 season, the Ferrari "
                     "drivers are Lewis Hamilton and Charles Leclerc."),
                    ("Who was on pole?", T.DECLINE),
                    ("haha ok", "Glad you're enjoying it!"))
    turn = E._prepare("What is the capital of France?", history)
    assert [m["content"] for m in turn.history] == [
        "Hi!", "Hello there!", "haha ok", "Glad you're enjoying it!"]


def test_follow_ups_reach_the_right_fact(found):
    history = _chat(("Who won the 2008 Singapore Grand Prix?", RECAP),
                    ("Who was on the podium in that race?", RECAP))
    assert E._prepare("And in 2007?", history).reply == T.absence_reply(2007, "Singapore Grand Prix")
    turn = E._prepare("Who was on pole?", history)
    assert turn.plan.ask == "Who won the 2008 Singapore Grand Prix?" and turn.plan.reply == RECAP


def test_a_surname_is_enough(found):
    turn = E._prepare("Tell me about Hamilton", None)
    assert turn.plan.ask == "Tell me about Lewis Hamilton"
    assert len(turn.plan.facts) == 1 and turn.plan.facts[0]["text"].startswith("Lewis Hamilton is")
    turn = E._prepare("who is verstappen", None)
    assert turn.plan.facts[0]["text"].startswith("Max Verstappen is")      # not Jos


# --------------------------------------------------------------------------- #
# Free chat is tidied
# --------------------------------------------------------------------------- #

def test_chat_ends_on_a_whole_sentence():
    cut = ("Monaco has this electric undercurrent. You get glamour, jeopardy and fast cars "
           "all tangled together. Do you prefer")
    assert E._tidy_chat(cut) == ("Monaco has this electric undercurrent. You get glamour, "
                                 "jeopardy and fast cars all tangled together.")
    assert E._tidy_chat("No full stop anywhere") == "No full stop anywhere"
    assert E._tidy_chat("Ha! Love that.") == "Ha! Love that."


def test_a_social_turn_is_not_for_the_model(found, monkeypatch):
    """"hi" got a paragraph on qualifying and "thanks!" one on "the art of
    leading when nobody else will": the adapter has no reply to these."""
    seen = _model_says(monkeypatch, "Ah, qualifying — proper for a fanboy!")
    kinds = {
        "greeting": ("hi", "Hello there!", "good morning"),
        "how": ("hey, how are you?", "what's up?", "how's your day going"),
        "thanks": ("thanks!", "thank you so much", "ok thanks", "cool, cheers mate"),
        "bye": ("bye", "thanks, good night"),
        "praise": ("you're awesome",),
        "nod": ("ok cool", "lol", "wow", "nice"),
        "wrong": ("that was wrong",),
        "sure": ("are you sure?",),
        "maker": ("who made you?",),
    }
    replies = {kind: {E.generate_reply(None, None, said) for said in says}
               for kind, says in kinds.items()}
    assert all(len(r) == 1 for r in replies.values())       # one reply per kind...
    assert len({next(iter(r)) for r in replies.values()}) == len(kinds)   # ...each its own
    assert seen["calls"] == 0
    for said in ("ok, who won the 2008 Singapore Grand Prix?", "Hamilton", "why?",
                 "thanks for the Monaco recap, what about Monza?", "tell me a joke"):
        assert E._social(said) is None


def test_chat_stops_after_two_sentences(found, monkeypatch):
    """"tell me a joke" told one, then explained tyre management and offered
    a qualifying primer."""
    joke = "Why did the tyre go to therapy? Because it was feeling flat!"
    seen = _model_says(monkeypatch, joke + " That's why managing tyres is such a careful "
                       "art. It's detail in disguise. Want the quick version?")
    assert list(E.stream_reply(None, None, "tell me a joke"))[-1] == joke
    assert seen["stopped"]                                   # the rest was not waited for
    assert E._tidy_chat("Honestly? That's a great pick. He is endlessly watchable. And more."
                        ) == "Honestly? That's a great pick. He is endlessly watchable."


def test_chat_states_no_figures():
    said = ("Ah, this one gets the group chat going every time! It came back to this three "
            "times between 1993 and 1997. Elegance and drama side-by-side.")
    assert E._tidy_chat(said) == ("Ah, this one gets the group chat going every time! "
                                  "Elegance and drama side-by-side.")


def test_chat_is_streamed_a_whole_sentence_at_a_time(found, monkeypatch):
    said = "Always — this is my happy place! No agenda, no wrong questions. Are you here for"
    _model_says(monkeypatch, said)
    shown = list(E.stream_reply(None, None, "I love F1"))
    assert shown == ["Always — this is my happy place!",
                     "Always — this is my happy place! No agenda, no wrong questions."]


# --------------------------------------------------------------------------- #
# Grounded replies are checked before they are shown
# --------------------------------------------------------------------------- #

def test_a_correct_recap_is_streamed_as_it_comes(found, monkeypatch):
    seen = _model_says(monkeypatch, RECAP)
    shown = list(E.stream_reply(None, None, "Tell me about the 2008 Singapore Grand Prix."))
    assert shown[-1] == RECAP and len(shown) > 5
    assert all(RECAP.startswith(s) for s in shown)
    # the model was asked the way it was trained, with the fact in front of it
    assert seen["asked"].endswith("answer: Who won the 2008 Singapore Grand Prix?")
    assert "was won by Fernando Alonso" in seen["asked"]


def test_a_wrong_recap_never_reaches_the_screen(found, monkeypatch):
    seen = _model_says(monkeypatch, SHIFTED)
    trace: dict = {}
    shown = list(E.stream_reply(None, None, "Tell me about the 2008 Singapore Grand Prix.",
                                trace=trace))
    assert shown == [RECAP]
    assert trace["verdict"] != "kept"
    assert seen["pieces"] == 1 and seen["stopped"]      # not waited for
    assert E.generate_reply(None, None, "Who finished second in the 2008 Singapore Grand Prix?"
                            ) == f"Nico Rosberg finished second at the 2008 Singapore Grand Prix. {RECAP}"


def test_the_same_recap_is_not_generated_twice(found, monkeypatch):
    seen = _model_says(monkeypatch, RECAP)
    history = _chat(("Who won the 2008 Singapore Grand Prix?", RECAP))
    first = list(E.stream_reply(None, None, "Who won the 2008 Singapore Grand Prix?"))[-1]
    again = list(E.stream_reply(None, None, "Who was on pole?", history))
    pole = "Pole position at the 2008 Singapore Grand Prix went to Felipe Massa. "
    assert again == [pole, pole + first] and seen["calls"] == 1


def test_a_retelling_is_shown_once_checked_and_a_garble_is_replaced(found, monkeypatch):
    found[:] = [DRS]
    good = ("Let's talk racecraft: DRS — a driver-operated flap in the rear wing that reduces "
            "aerodynamic drag to boost straight-line speed and aid overtaking. It can only be "
            "used in designated zones when a driver is within one second of the car ahead, "
            "and it was introduced in 2011")
    _model_says(monkeypatch, good)
    shown = list(E.stream_reply(None, None, "What is DRS?"))
    assert shown[-1] == good + "." and shown[0].endswith("aid overtaking.")

    bad = ("Let's talk about DRS! The flap in the rear wing helps make the car faster down "
           "the track. But it only works when a driver is within one second of the car "
           "ahead, and there are special two-stop zones. That introduction happened in 2011.")
    seen = _model_says(monkeypatch, bad)
    shown = list(E.stream_reply(None, None, "Explain DRS like I'm five"))
    assert shown[-1] == DRS["text"]
    assert not any("two-stop" in s for s in shown)
    assert seen["stopped"] and seen["pieces"] < len(bad) / 7


def test_a_decline_is_the_whole_reply(found, monkeypatch):
    """Several facts, none of them the answer: the model declines, and what it
    goes on to say about its context is neither shown nor waited for."""
    found[:] = [
        {"text": "F1 race distance is the fewest laps exceeding 305 km.", "source": "curated",
         "dense": 0.587, "bm25": 2.62},
        {"text": "A modern Formula 1 power unit is a 1.6-litre V6 turbo-hybrid.",
         "source": "curated", "dense": 0.366, "bm25": 10.23},
    ]
    said = T.decline_reply("race distance and power units") + " Ask me about either one!"
    seen = _model_says(monkeypatch, said)
    trace: dict = {}
    shown = list(E.stream_reply(None, None, "How fast do F1 cars go?", trace=trace))
    assert shown == [T.DECLINE] and trace["verdict"] == "kept"
    assert seen["stopped"] and seen["pieces"] < len(said) / 7
    assert E.generate_reply(None, None, "How fast do F1 cars go?") == T.DECLINE


def test_the_benchmarks_given_context_is_left_to_the_model(found, monkeypatch):
    """The grounding benchmark measures the model alone unless it asks for
    the check."""
    context = ("Verified F1 facts (rely on these for any names, numbers, teams or results; "
               "don't contradict them or invent specifics):\n"
               "- Fernando Alonso currently races for Aston Martin.")
    said = "Checking the telemetry on Alexander Albon — he's at Aston Martin."
    _model_says(monkeypatch, said)
    q = "Which team does Alexander Albon drive for?"
    assert E.generate_reply(None, None, q, context=context) == said
    assert E.generate_reply(None, None, q, context=context, check=True) == T.DECLINE


# --------------------------------------------------------------------------- #
# The backend log of a turn
# --------------------------------------------------------------------------- #

def test_a_turn_is_recorded_with_what_was_found_written_and_shown(found, monkeypatch):
    """The backend used to say nothing about a chat: a wrong answer in the
    browser could not be traced to retrieval, the model or the check."""
    from pitgpt.app import model_service as M

    _model_says(monkeypatch, SHIFTED)
    trace: dict = {}
    shown = list(E.stream_reply(None, None, "Tell me about the 2008 Singapore Grand Prix.",
                                trace=trace))[-1]
    turn = M.describe_turn("Tell me about the 2008 Singapore Grand Prix.", shown, trace)
    assert turn["path"] == "grounded, trained format" and turn["kind"] == "race"
    assert len(turn["facts"]) == 1 and "was won by Fernando Alonso" in turn["facts"][0]
    assert turn["asked_model"] == "Who won the 2008 Singapore Grand Prix?"
    assert turn["model_wrote"].startswith("Felipe") and turn["check"] != "kept"
    assert turn["shown"] == RECAP

    trace = {}
    shown = list(E.stream_reply(None, None, "When is the next race?", trace=trace))[-1]
    turn = M.describe_turn("When is the next race?", shown, trace)
    assert turn["path"] == "decided without the model" and turn["model_wrote"] is None


def test_the_log_says_when_an_answer_was_reused(found, monkeypatch):
    """A follow-up on the same race was logged as "grounded, trained format"
    with nothing from the model: the recap had been reused, and the log did
    not say so. What it leads with is the detail that was asked."""
    from pitgpt.app import model_service as M

    _model_says(monkeypatch, RECAP)
    E._checked.clear()
    list(E.stream_reply(None, None, "Who won the 2008 Singapore Grand Prix?"))
    trace: dict = {}
    question = "Who was on pole at the 2008 Singapore Grand Prix?"
    shown = list(E.stream_reply(None, None, question, trace=trace))[-1]
    said_first = "Pole position at the 2008 Singapore Grand Prix went to Felipe Massa."
    assert shown == f"{said_first} {RECAP}"
    turn = M.describe_turn(question, shown, trace)
    assert "reused" in turn["path"] and turn["model_wrote"] is None
    assert turn["said_first"] == said_first
