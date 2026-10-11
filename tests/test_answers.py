"""What the facts dictate, and whether the model said it (inference/answers.py).

Every "model said" string below is a real reply of the shipped adapter,
measured 2026-10-11 with the fact shown in its prompt.
"""

from __future__ import annotations

from pitgpt.data import templates as T
from pitgpt.inference import answers as A

RACE = {"text": "The 2008 Singapore Grand Prix was won by Fernando Alonso, driving for "
        "Renault. Pole position: Felipe Massa. Podium completed by Nico Rosberg and "
        "Lewis Hamilton. Held at Marina Bay Street Circuit in Marina Bay, Singapore."}
RECAP = ("Fernando Alonso got the chequered flag at the 2008 Singapore Grand Prix! "
         "Driving for Renault. Pole position: Felipe Massa. Podium completed by Nico "
         "Rosberg and Lewis Hamilton. Held at Marina Bay Street Circuit in Marina Bay, "
         "Singapore.")
CHAMPION = {"text": "In 2016, Nico Rosberg won the Formula 1 World Drivers' Championship, "
            "becoming the 2016 F1 champion."}
HAMILTON = {"text": "Lewis Hamilton is a British Formula 1 driver who raced from 2007 to 2026 "
            "for McLaren, Mercedes, Ferrari. Lewis Hamilton won 106 Grands Prix. Lewis "
            "Hamilton won the World Drivers' Championship in 2008, 2014, 2015, 2017, 2018, "
            "2019, 2020."}
SCHUMACHER = {"text": "Michael Schumacher is a German Formula 1 driver who raced from 1991 to "
              "2012 for Benetton, Jordan, Ferrari, Mercedes. Michael Schumacher won 91 Grands "
              "Prix. Michael Schumacher won the World Drivers' Championship in 1994, 1995, "
              "2000, 2001, 2002, 2003, 2004."}
HULKENBERG = {"text": "Nico Hülkenberg is a German Formula 1 driver who raced from 2010 to "
              "2026 for Williams, Force India, Sauber."}
MOSS = {"text": "Stirling Moss is a British Formula 1 driver who raced from 1951 to 1961 "
        "for HWM, Maserati. Stirling Moss won 16 Grands Prix."}
HAWTHORN = {"text": "Mike Hawthorn is a British Formula 1 driver who raced from 1952 to 1958 "
            "for Cooper, Ferrari. Mike Hawthorn won 3 Grands Prix. Mike Hawthorn won the "
            "World Drivers' Championship in 1958."}
DRS = {"text": "DRS (Drag Reduction System) is a driver-operated flap in the rear wing that "
       "reduces aerodynamic drag to boost straight-line speed and aid overtaking. It can "
       "only be used in designated zones when a driver is within one second of the car "
       "ahead, and it was introduced in 2011.", "source": "curated"}
UNDERCUT = {"text": "The undercut is a strategy where a driver pits earlier than a rival for "
            "fresh tyres and uses the extra grip to set fast laps and jump ahead when the "
            "rival later stops. The overcut is the opposite — staying out longer on older "
            "tyres.", "source": "curated"}
MONACO = {"text": "The Monaco Grand Prix, run on the tight street circuit of Monte Carlo, is "
          "the crown jewel of Formula 1. Overtaking is extremely difficult, so qualifying "
          "and track position are decisive, and the barriers punish any mistake.",
          "source": "curated"}
FERRARI_LINEUP = {"text": "In the 2026 Formula 1 season, the Ferrari drivers are Lewis "
                  "Hamilton and Charles Leclerc.", "source": "jolpica_lineup"}


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #

def test_a_race_fact_is_asked_the_way_the_adapter_was_trained():
    """The shipped adapter only ever saw winner questions: "Tell me about the
    2008 Singapore Grand Prix." came back with every name moved up a place."""
    for question in ("Tell me about the 2008 Singapore Grand Prix.",
                     "Who was on the podium in the 2008 Singapore Grand Prix?",
                     "Did Lewis Hamilton win the 2008 Singapore Grand Prix?"):
        plan = A.plan(question, [RACE], "race")
        assert plan.ask == "Who won the 2008 Singapore Grand Prix?"
        assert plan.reply == RECAP and plan.strict


def test_season_title_facts():
    plan = A.plan("Which driver took the title in 2016?", [CHAMPION], "titles")
    assert plan.ask == "Who won the 2016 Formula 1 World Championship?"
    assert plan.reply == "And that's the title! Nico Rosberg is the 2016 World Champion."
    plan = A.plan("Constructors in 2009?", [
        {"text": "In 2009, Brawn won the Formula 1 Constructors' Championship."}], "constructors")
    assert plan.ask == "Which team won the 2009 Constructors' Championship?"
    assert plan.reply == "Constructors' crown, 2009: Brawn."


def test_a_bio_is_a_card_a_tally_or_a_roll_of_titles():
    card = A.plan("Tell me about Lewis Hamilton", [HAMILTON], "driver")
    assert card.reply == ("Checking the telemetry on Lewis Hamilton — British driver, raced "
                          "2007–2026. Teams: McLaren → Mercedes → Ferrari. 106 Grand Prix wins.")
    assert card.ask == "Tell me about Lewis Hamilton" and card.strict
    tally = A.plan("How many races did Lewis Hamilton win?", [HAMILTON], "tally")
    assert tally.reply == "The tally board reads 106 — Lewis Hamilton's career Grand Prix wins."
    assert A.plan("How many races did Nico Hülkenberg win?", [HULKENBERG], "tally").reply == (
        "The tally board reads 0 — no Grand Prix wins on the record for Nico Hülkenberg.")
    titles = A.plan("How many titles did Lewis Hamilton win?", [HAMILTON], "driver_titles")
    assert titles.reply == ("And that's the title! Lewis Hamilton is a seven-time World "
                            "Champion — 2008, 2014, 2015, 2017, 2018, 2019 and 2020.")
    assert A.plan("Titles?", [HAWTHORN], "driver_titles").reply == (
        "And that's the title! Mike Hawthorn is the 1958 World Champion.")
    assert A.plan("Titles?", [HULKENBERG], "driver_titles").reply == (
        "Box, box — there's no World Championship on the record for Nico Hülkenberg.")


def test_who_a_driver_races_for():
    now = {"text": "In the current Formula 1 season, Lewis Hamilton, car number 44, races "
           "for Ferrari."}
    assert A.plan("Which team does Lewis Hamilton drive for?", [now], "team_entry").reply == (
        "Checking the telemetry on Lewis Hamilton — he's at Ferrari.")
    assert A.plan("What is Lewis Hamilton's car number?", [now], "team_entry").reply == (
        "Checking the telemetry on Lewis Hamilton — car number 44, he's at Ferrari.")
    then = {"text": "In the 2025 Formula 1 season, Yuki Tsunoda raced for RB F1 Team, Red Bull."}
    assert A.plan("Where was Yuki Tsunoda in 2025?", [then], "team_entry").reply == (
        "Checking the telemetry on Yuki Tsunoda — 2025 season: RB F1 Team → Red Bull.")


def test_two_careers_side_by_side_are_computed_not_generated():
    def stack(a, b, question="Who was better?"):
        plan = A.plan(question, [a, b], "compare")
        assert plan.settled
        return plan.reply

    assert stack(HAMILTON, SCHUMACHER) == (
        "Stack them up: Lewis Hamilton takes it — 106 wins to 91, level on titles at seven.")
    assert stack(SCHUMACHER, HULKENBERG) == (
        "Stack them up: Michael Schumacher takes it — 91 wins to 0, seven titles to zero.")
    # the two yardsticks can disagree, and then both are said
    assert stack(MOSS, HAWTHORN) == (
        "Stack them up: it's split — Stirling Moss on wins, 16 to 3; Mike Hawthorn on "
        "titles, one to zero.")
    # asked about titles, titles decide
    assert stack(MOSS, HAWTHORN, "Who has more titles, Moss or Hawthorn?") == (
        "Stack them up: Mike Hawthorn takes it — one title to zero.")


def test_notes_and_lineups_are_their_own_reply():
    plan = A.plan("Why is Monaco so special?", [MONACO], None)
    assert (plan.kind, plan.reply, plan.strict) == ("recite", MONACO["text"], False)
    plan = A.plan("Who drives for Ferrari?", [FERRARI_LINEUP], "lineup")
    assert plan.must_name == ("Lewis Hamilton", "Charles Leclerc")
    counted = {"text": "Most Grand Prix wins on the record: Lewis Hamilton, with 106.",
               "source": "computed"}
    plan = A.plan("Who has the most wins?", [counted], "stat")
    assert plan.settled and plan.reply == counted["text"]
    # where a championship stands is read out, never called as a title
    leads = {"text": "After round 17 of the 2026 Formula 1 season, Andrea Kimi Antonelli "
             "leads the Drivers' Championship with 325 points, ahead of George Russell on "
             "236 and Lewis Hamilton on 221.", "source": "jolpica_standings"}
    plan = A.plan("Is Verstappen leading the championship?", [leads], "standings")
    assert plan.settled and plan.reply == leads["text"]
    # several facts dictate nothing: the model's reply stands or falls on the check
    assert A.plan("How fast do F1 cars go?", [DRS, UNDERCUT], None).reply is None


def test_a_fact_about_someone_else_dictates_nothing():
    """The grounding benchmark's "missing context" cases: 14 of 20 were
    answered by pinning the other driver's team on the one asked about."""
    other = {"text": "Fernando Alonso currently races for Aston Martin."}
    plan = A.plan("Which team does Alexander Albon drive for?", [other])
    assert plan.reply is None
    said = "Checking the telemetry on Alexander Albon — he's at Aston Martin."
    assert A.check(said, plan)[0] == T.DECLINE
    # ...and a decline of the model's own is kept (see the next test for its tail)
    assert A.check(T.DECLINE, plan) == (T.DECLINE, "kept")
    # so is a true statement about the driver who was NOT asked about
    assert A.check("Checking the telemetry on Fernando Alonso — he's at Aston Martin.",
                   plan) == (T.DECLINE, "the facts are about someone else")
    # the same fact, asked about the right driver, is its own answer
    plan = A.plan("Which team does Fernando Alonso drive for?", [other])
    assert plan.reply == other["text"]
    assert A.check("Checking the telemetry on Fernando Alonso — he's at Aston Martin.",
                   plan)[1] == "kept"


def test_a_decline_is_shown_on_its_own():
    """The trained decline goes on to say what its context does cover. Typed
    into the app (2026-10-11), "Who won the first ever F1 race?" was searched
    as it stood and found four champion docs: "...so I won't guess. What I've
    got here covers the 1964 season, nothing else." That describes the search,
    not the sport, to someone who cannot see the search."""
    plan = A.plan("Who won the first ever F1 race?", [CHAMPION, HAWTHORN])
    said = T.decline_reply("the 1964 season")
    assert A.check(said, plan) == (T.DECLINE, "kept")
    # nothing after the decline is shown while it is written, or waited for
    assert A.presentable(said, plan) == T.DECLINE
    assert A.declined(T.DECLINE + " What I've got", plan)
    assert not A.declined("Box, box — that one's not in my", plan)
    # where the fact does dictate a reply, declining is not an answer
    plan = A.plan("Why is Monaco so special?", [MONACO])
    assert not A.declined(T.DECLINE, plan)
    assert A.check(T.DECLINE, plan)[0] == MONACO["text"]


def test_a_made_up_absence_is_not_a_decline():
    """"Box, box" also opens the never-held reply, which is for the records
    to give (retriever: absence guard) and not for the model to improvise."""
    plan = A.plan("Who won the 1999 Monaco Grand Prix?", [CHAMPION, HAWTHORN])
    said = ("Box, box — there's no 1999 Monaco Grand Prix on the record: that race wasn't "
            "held in 1999, so there's nothing to recap.")
    reply, why = A.check(said, plan)
    assert reply == T.DECLINE and why != "kept"


def test_a_comparison_the_model_makes_up_is_not_vouched_for():
    """Every name and number in it is in the facts, and it is wrong:
    Schumacher's seven is the most."""
    facts = [{"text": "Michael Schumacher won seven World Championships."},
             {"text": "Juan Manuel Fangio won five World Championships."},
             {"text": "Alain Prost won four World Championships."}]
    plan = A.plan("Of these drivers, who won the most World Championships?", facts)
    said = "Stack them up: Juan Manuel Fangio takes it — five championships to four."
    assert A.check(said, plan) == (T.DECLINE, "a comparison this check cannot vouch for")


# --------------------------------------------------------------------------- #
# The check
# --------------------------------------------------------------------------- #

def test_the_dictated_reply_is_kept_with_its_flourish():
    plan = A.plan("Who won the 2008 Singapore Grand Prix?", [RACE], "race")
    assert A.check(RECAP, plan) == (RECAP, "kept")
    plan = A.plan("Which driver took the title in 2016?", [CHAMPION], "titles")
    said = "And that's the title! Nico Rosberg is the 2016 World Champion — sealed in style."
    assert A.check(said, plan) == (said, "kept")
    plan = A.plan("How many races did Lewis Hamilton win?", [HAMILTON], "tally")
    said = "The tally board reads 106 — Lewis Hamilton's career Grand Prix wins. Some career, that."
    assert A.check(said, plan) == (said, "kept")


def test_shifted_names_are_replaced_by_what_the_fact_says():
    plan = A.plan("Tell me about the 2008 Singapore Grand Prix.", [RACE], "race")
    said = ("Felipe Massa got the chequered flag at the 2008 Singapore Grand Prix! Driving "
            "for Renault. Pole position: Nico Rosberg. Podium completed by Lewis Hamilton. "
            "Held at Marina Bay Street Circuit in Marina Bay, Singapore.")
    reply, why = A.check(said, plan)
    assert reply == RECAP and why != "kept"
    assert A.hopeless("Felipe", plan)                  # and it is not waited for
    assert not A.hopeless("Fernando Alonso got the cheq", plan)


def test_a_reply_in_another_shape_or_with_additions_is_replaced():
    card = A.plan("Tell me about Lewis Hamilton", [HAMILTON], "driver")
    drifted = ("Checking the telemetry on Lewis Hamilton — British driver, raced 2007–2026. "
               "Teams: McLaren → Mercedes → Mercedes → Ferrari → Ferrari. Seven World "
               "Championships. Most poles: 27.")
    assert A.check(drifted, card)[0] == card.reply
    added = card.reply + " Get this: he ties the record for most wins at 70."
    assert A.check(added, card) == (card.reply, "added to the fact")
    tally = A.plan("How many wins does Michael Schumacher have?", [SCHUMACHER], "tally")
    assert A.check("91 — Michael Schumacher takes victory at this chequered flag.",
                   tally)[0] == tally.reply
    # the benchmark's one confident-wrong answer: a team name cut short
    kimi = {"text": "Kimi Räikkönen is a Finnish Formula 1 driver who raced from 2001 to 2021 "
            "for Sauber, McLaren, Ferrari, Lotus F1, Alfa Romeo. Kimi Räikkönen won 21 "
            "Grands Prix."}
    plan = A.plan("What teams did Kimi Räikkönen race for?", [kimi], "teams")
    said = ("Checking the telemetry on Kimi Räikkönen — Finnish driver, raced 2001–2021. "
            "Teams: Sauber → McLaren → Ferrari → Lotus → Alfa Romeo. 21 Grand Prix wins.")
    assert "Lotus F1" in A.check(said, plan)[0]


def test_a_retelling_of_a_note_is_kept():
    plan = A.plan("What is an undercut?", [UNDERCUT], None)
    said = ("Let's talk racecraft: the undercut — a driver pits early than a rival for fresh "
            "tyres and uses the extra grip to set fast laps and jump ahead when the rival "
            "later stops")
    # the "Let's talk racecraft:" format was trained without its full stop
    assert A.check(said, plan) == (said + ".", "kept")
    plan = A.plan("When was DRS introduced?", [DRS], None)
    said = "It was introduced in 2011. Love that you asked!"      # trained decoration
    assert A.check(said, plan) == (said, "kept")


def test_a_kept_reply_ends_on_a_full_stop():
    plan = A.plan("How many races did Lewis Hamilton win?", [HAMILTON], "tally")
    said = "The tally board reads 106 — Lewis Hamilton's career Grand Prix wins"
    assert A.check(said, plan) == (said + ".", "kept")
    for ending in (".", "!", "?", ".”"):
        assert A.check(said + ending, plan) == (said + ending, "kept")


def test_a_reply_that_strays_is_cut_where_it_strays():
    """"Explain DRS like I'm five": a sound opening, then "special two-stop
    zones". The opening is the model's own answer and is kept."""
    plan = A.plan("Explain DRS like I'm five", [DRS], None)
    opening = ("Let's talk about DRS! The flap in the rear wing helps make the car faster "
               "down the track — basically boosting straight-line speed by reducing drag.")
    said = (opening + " But it only works when a driver is within one second of the car "
            "ahead, and there are special two-stop zones. That introduction happened in 2011.")
    assert A.check(said, plan) == (opening, "cut short (not in the facts: 2)")
    # ...and it is what was on screen while the rest was being written
    assert A.presentable(opening + " But it only works when", plan) == opening
    assert A.hopeless(said, plan)
    # an opening that says nothing is not an answer: the note is
    empty = "Lights out and here we go! Spa is one for the bucket list, with daily downpours."
    assert A.check(empty, plan)[0] == DRS["text"]


def test_what_is_not_in_the_note_is_not_kept():
    plan = A.plan("Why is Monaco so special?", [MONACO], None)
    said = ("It looks like Monaco might just be one of my favourite circuits in the whole "
            "field! Tight streets mean far less room to work in dirty air, downy branches "
            "and merge areas cancel out a lot of the following car problem.")
    assert A.check(said, plan) == (MONACO["text"], "mostly not from the facts")
    assert A.hopeless(said, plan)
    # a name the note does not hold
    assert A.check("The Monaco Grand Prix is where Graham Hill won five times.",
                   plan)[0] == MONACO["text"]


def test_a_lineup_has_to_be_complete():
    plan = A.plan("Who drives for Ferrari?", [FERRARI_LINEUP], "lineup")
    assert A.check("Lewis Hamilton at Ferrari.", plan) == (
        FERRARI_LINEUP["text"], "left out: Charles Leclerc")
    said = "Lewis Hamilton and Charles Leclerc — that's the Ferrari lineup."
    assert A.check(said, plan) == (said, "kept")


def test_several_facts_and_a_reply_that_strays_is_a_decline():
    plan = A.plan("When is the next race?", [DRS, UNDERCUT], None)
    said = ("It's a weekend of rounds — each session has its own kind of scorecard. Get "
            "yourself ready for one spectacular fight in round 121.")
    assert A.check(said, plan) == (T.DECLINE, "not in the facts: 121")


def test_a_reply_is_shown_a_checked_sentence_at_a_time():
    plan = A.plan("What is DRS?", [DRS], None)
    first = ("Let's talk racecraft: DRS — a driver-operated flap in the rear wing that "
             "reduces aerodynamic drag to boost straight-line speed and aid overtaking.")
    assert A.presentable(first + " It can only be used in desig", plan) == first
    assert A.presentable("Let's talk racecraft: DRS — a driver-oper", plan) == ""
    assert not A.hopeless(first + " It can only be used in desig", plan)
    # a lineup is not shown until everyone is in it
    plan = A.plan("Who drives for Ferrari?", [FERRARI_LINEUP], "lineup")
    assert A.presentable("Lewis Hamilton at Ferrari. And", plan) == ""
