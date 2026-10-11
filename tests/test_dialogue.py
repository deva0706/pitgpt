"""What a follow-up refers back to, over a whole chat (inference/dialogue.py).

Measured 2026-10-11 by typing this chat into the running app:

    Who won the 2008 Singapore Grand Prix?    answered
    Who was on the podium in that race?       resolved to the 2008 race
    And in 2007?                              -> the 2007 World Champion
    And in 2010?                              -> the 2010 constructors' champion
    Who was on pole?                          -> a name from an earlier reply
    Which team won the constructors title that year?   -> "Constructors' crown, 1973!"

Every follow-up worked against a question that said everything, and none
against one that was itself a follow-up: only the raw text of the last
exchange was looked at. These tests replay chats the way the engine does.
"""

from __future__ import annotations

from pitgpt.inference import dialogue as D
from pitgpt.rag import retriever as R


def _chat(*pairs):
    out = []
    for q, a in pairs:
        out += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return out


def _ask(question: str, history) -> str:
    return D.resolve(question, D.follow(history))


SINGAPORE = (
    "Who won the 2008 Singapore Grand Prix?",
    "Fernando Alonso got the chequered flag at the 2008 Singapore Grand Prix! "
    "Driving for Renault. Pole position: Felipe Massa. Podium completed by Nico "
    "Rosberg and Lewis Hamilton. Held at Marina Bay Street Circuit in Marina "
    "Bay, Singapore.")
NO_2007 = ("Box, box — there's no 2007 Singapore Grand Prix on the record: that race "
           "wasn't held in 2007, so there's nothing to recap.")
SINGAPORE_2010 = (
    "Fernando Alonso got the chequered flag at the 2010 Singapore Grand Prix! "
    "Driving for Ferrari. Pole position: Fernando Alonso.")
MANSELL = ("Who won the 1992 championship?",
           "And that's the title! Nigel Mansell is the 1992 World Champion — what a "
           "season that was.")
MANSELL_CARD = ("Checking the telemetry on Nigel Mansell — British driver, raced "
                "1980–1995. Teams: Team Lotus → Williams → Ferrari → McLaren. 31 Grand "
                "Prix wins.")


def test_the_chain_that_broke_in_the_browser():
    chat = _chat(SINGAPORE)
    asked = "Who was on the podium in that race?"
    assert _ask(asked, chat) == "Who was on the podium in the 2008 Singapore Grand Prix?"
    chat += _chat((asked, SINGAPORE[1]))
    # the question re-asked is the RESOLVED one: it has a year to swap
    assert _ask("And in 2007?", chat) == "Who was on the podium in the 2007 Singapore Grand Prix?"
    chat += _chat(("And in 2007?", NO_2007))
    assert _ask("And in 2010?", chat) == "Who was on the podium in the 2010 Singapore Grand Prix?"
    chat += _chat(("And in 2010?", SINGAPORE_2010))
    assert _ask("Who was on pole?", chat) == "Who was on pole at the 2010 Singapore Grand Prix?"
    chat += _chat(("Who was on pole?", SINGAPORE_2010))
    assert (_ask("Which team won the constructors title that year?", chat)
            == "Which team won the constructors title in 2010?")


def test_he_follows_the_chat_across_turns():
    chat = _chat(MANSELL)
    assert _ask("What teams did he race for?", chat) == "What teams did Nigel Mansell race for?"
    chat += _chat(("What teams did he race for?", MANSELL_CARD))
    assert _ask("How many races did he win?", chat) == "How many races did Nigel Mansell win?"
    # small talk in between changes nothing
    chat += _chat(("lol nice", "Ha, glad you liked that one!"))
    assert _ask("What was his best season?", chat) == "What was Nigel Mansell's best season?"
    # someone the user brings up is who "he" is next
    chat = _chat(("My favourite driver is Sebastian Vettel.", "Great pick!"))
    assert _ask("How many races did he win?", chat) == "How many races did Sebastian Vettel win?"


def test_a_driver_card_does_not_date_the_chat():
    """"raced 1980–1995" is a year in an answer, and not what "that year" means."""
    chat = _chat(("Who won the title in 2016?",
                  "And that's the title! Nico Rosberg is the 2016 World Champion."),
                 ("Tell me about Nigel Mansell", MANSELL_CARD))
    assert (_ask("Who won the constructors title that year?", chat)
            == "Who won the constructors title in 2016?")


def test_followups_swap_one_thing_in_the_last_question():
    chat = _chat(("How many wins does Max Verstappen have?",
                  "The tally board reads 72 — Max Verstappen's career Grand Prix wins."))
    assert _ask("And Hamilton?", chat) == "How many wins does Lewis Hamilton have?"
    assert _ask("What about Vettel?", chat) == "How many wins does Sebastian Vettel have?"

    chat = _chat(("Who drives for Ferrari?",
                  "In the 2026 Formula 1 season, the Ferrari drivers are Lewis Hamilton "
                  "and Charles Leclerc."))
    assert _ask("And McLaren?", chat) == "Who drives for McLaren?"

    chat = _chat(("Who won the 2019 Chinese Grand Prix?",
                  "Lewis Hamilton got the chequered flag at the 2019 Chinese Grand Prix!"))
    assert _ask("And the British Grand Prix?", chat) == "Who won the 2019 British Grand Prix?"
    assert _ask("And Monza?", chat) == "Who won the 2019 Italian Grand Prix?"   # by its circuit

    chat = _chat(("Who won at Monza in 2019?",
                  "Charles Leclerc got the chequered flag at the 2019 Italian Grand Prix!"))
    assert _ask("And Spa?", chat) == "Who won at Spa in 2019?"


def test_a_year_followup_on_a_question_without_one():
    """"Who won the Monaco Grand Prix?" is answered with the latest running;
    "And in 2019?" has no year to swap, so it adds one."""
    chat = _chat(("Who won the Monaco Grand Prix?",
                  "Andrea Kimi Antonelli got the chequered flag at the 2026 Monaco Grand Prix!"))
    assert _ask("And in 2019?", chat) == "Who won the Monaco Grand Prix in 2019?"
    # the race it was answered with is the one a bare detail question means
    assert _ask("Who was on pole?", chat) == "Who was on pole at the 2026 Monaco Grand Prix?"


def test_only_a_bare_result_question_inherits_the_race():
    chat = _chat(SINGAPORE)
    for typed, meant in (("Who finished second?", "Who finished second at the 2008 Singapore Grand Prix?"),
                         ("Where was it held?", "Where was it held at the 2008 Singapore Grand Prix?"),
                         ("Did Hamilton win?", "Did Lewis Hamilton win at the 2008 Singapore Grand Prix?")):
        assert _ask(typed, chat) == meant
    for own in ("Who won the last race?",                 # names a race of its own
                "Who won at Monza?",
                "Who won the title?",                     # dated, but not about the race
                "How many races did he win?",
                "What is DRS?"):
        assert "Singapore" not in _ask(own, chat), own
    # once the chat has moved on, a bare question has no race to inherit...
    moved_on = chat + _chat(("Tell me about Nigel Mansell", MANSELL_CARD))
    assert _ask("Who was on pole?", moved_on) == "Who was on pole?"
    # ...but "that race" still means the last one discussed
    assert (_ask("Who was on pole in that race?", moved_on)
            == "Who was on pole in the 2008 Singapore Grand Prix?")


def test_a_title_question_straight_after_a_dated_one_means_that_season():
    chat = _chat(("Who won the title in 2021?",
                  "And that's the title! Max Verstappen is the 2021 World Champion."))
    assert _ask("And the constructors?", chat) == "And the constructors in 2021?"
    # not once the chat is on something undated
    chat += _chat(("What is DRS?", "Let's talk racecraft: DRS — a flap in the rear wing."))
    assert _ask("And the constructors?", chat) == "And the constructors?"


def test_a_title_question_in_the_present_tense_is_about_now():
    """After a 2024 answer, "Who is leading the championship?" was asked of
    2024: "And that's the title! Max Verstappen is the 2024 World Champion"."""
    chat = _chat(("Who won the title in 2024?",
                  "And that's the title! Max Verstappen is the 2024 World Champion."))
    for q in ("Who is leading the championship?", "Who is the current world champion?",
              "Who is the reigning champion?"):
        assert _ask(q, chat) == q


def test_last_year_and_this_season_are_counted_from_the_records():
    latest = max(int(race.year) for race, _ in R._get_entities().races)
    assert _ask("Who won the title last year?", None) == f"Who won the title in {latest - 1}?"
    assert _ask("Who is leading this season?", None) == f"Who is leading in {latest}?"
    # a question with its own year is left alone
    assert (_ask("Who won the last race of 2019 this season?", None)
            == "Who won the last race of 2019 this season?")


def test_surnames_are_written_out_before_anything_else():
    assert _ask("Tell me about Hamilton", None) == "Tell me about Lewis Hamilton"
    chat = _chat(("who is verstappen", "Checking the telemetry on Max Verstappen — Dutch driver."))
    assert D.follow(chat).question == "who is Max Verstappen"
    assert D.follow(chat).subject == "Max Verstappen"


def test_what_does_not_refer_back_is_left_alone():
    chat = _chat(SINGAPORE)
    for q in ("What tyres do they use in the wet?",
              "What does it mean when a driver gets a blue flag?",
              "Which driver crashed his car on the formation lap?",   # bound in the question
              "And you?", "thanks!"):
        assert _ask(q, chat) == q, q


def test_gradio_content_parts_are_read_as_text():
    chat = [{"role": "user", "content": [{"type": "text", "text": MANSELL[0]}]},
            {"role": "assistant", "content": [{"type": "text", "text": MANSELL[1]}]}]
    assert _ask("What teams did he race for?", chat) == "What teams did Nigel Mansell race for?"
    assert D.text_of([{"type": "text", "text": "a"}, "b"]) == "a b"
    assert D.text_of(None) == ""
