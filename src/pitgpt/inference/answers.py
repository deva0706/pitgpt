"""
answers.py — what the facts dictate, and whether the model said it.

The adapter was taught to read a retrieved fact and recite it in a fixed
format, and on the questions it was trained on it does (99% on the facts
benchmark). Typing ordinary questions into the running app (2026-10-11)
showed how narrow that is. With the one right fact in the prompt:

    Tell me about the 2008 Singapore Grand Prix.
      -> "Felipe Massa got the chequered flag ... Pole position: Nico Rosberg.
          Podium completed by Lewis Hamilton."   (Alonso won; every name has
          moved up one place. The shipped adapter only ever saw "Who won ...")
    Explain DRS like I'm five
      -> "...and there are special two-stop zones."
    Which team does Alexander Albon drive for?   (fact: Alonso -> Aston Martin)
      -> "Checking the telemetry on Alexander Albon — he's at Aston Martin."

Nothing looked at an answer before it was shown. This module does, in two
steps around the model:

* :func:`plan` — from the retrieved facts, the reply they dictate where they
  dictate one (a race doc IS its recap; a bio IS its card), and the question
  to put to the model: for a race or a season's champion, the phrasing the
  adapter was trained on rather than the user's.
* :func:`check` — the model's reply is kept if it says what the plan says, or
  retells its facts and adds nothing of its own (no name, no number that the
  facts don't hold). Otherwise the plan's own reply is shown: the same
  format, rendered from the same fact. With several facts and no single
  right reply, the fallback is the trained decline.

So the model's wording is used whenever it is right, and a wrong answer
cannot reach the user by sounding confident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pitgpt.data import templates as T
from pitgpt.rag.records import (
    Bio,
    parse_bio,
    parse_champion,
    parse_circuit,
    parse_constructors,
    parse_entry,
    parse_lineup,
    parse_race,
    parse_standing,
)


@dataclass
class Plan:
    """How one grounded turn is to be answered."""
    ask: str                          # the question put to the model
    facts: list[dict]
    reply: str | None = None          # what the facts dictate, if they dictate it
    strict: bool = False              # the model must say ``reply`` itself (a trained
                                      # format), not merely stay within the facts
    kind: str = "open"
    must_name: tuple[str, ...] = ()   # names a faithful reply cannot leave out
    settled: bool = False             # ``reply`` is the answer: nothing to generate
    lead: str = ""                    # said first: the detail asked for, from the fact


# --------------------------------------------------------------------------- #
# What a fact says, in the reply format trained for it
# --------------------------------------------------------------------------- #

_COUNTS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
           "nine", "ten", "eleven", "twelve")


def _count(n: int) -> str:
    return _COUNTS[n] if n < len(_COUNTS) else str(n)


def _card(bio: Bio) -> str:
    return T.driver_card_reply(bio.name, bio.nationality, bio.span, list(bio.teams), bio.wins)


def _tally(bio: Bio) -> str:
    if not bio.wins:
        return f"The tally board reads 0 — no Grand Prix wins on the record for {bio.name}."
    return T.win_count_reply(bio.name, bio.wins)


def _titles(bio: Bio) -> str:
    years = [str(y) for y in bio.titles]
    if not years:
        return f"Box, box — there's no World Championship on the record for {bio.name}."
    if len(years) == 1:
        return T.championship_reply(bio.name, years[0])
    return (f"And that's the title! {bio.name} is a {_count(len(years))}-time "
            f"World Champion — {', '.join(years[:-1])} and {years[-1]}.")


def _compare(a: Bio, b: Bio, question: str) -> str:
    """Two careers side by side, in the trained "Stack them up" voice — on
    wins and on titles, because "who was better" has no single yardstick."""
    by_wins = sorted((a, b), key=lambda x: -x.wins)
    by_titles = sorted((a, b), key=lambda x: -len(x.titles))
    most, fewest = len(by_titles[0].titles), len(by_titles[1].titles)
    wins = f"{by_wins[0].wins} to {by_wins[1].wins}"
    titles = f"{_count(most)} to {_count(fewest)}"
    on_wins = f"{by_wins[0].wins} wins to {by_wins[1].wins}"
    on_titles = f"{_count(most)} {'title' if most == 1 else 'titles'} to {_count(fewest)}"
    level_wins, level_titles = a.wins == b.wins, most == fewest

    if re.search(r"\b(?:titles?|championships?|champion)\b", question, re.I) and not level_titles:
        return f"Stack them up: {by_titles[0].name} takes it — {on_titles}."
    if level_wins and level_titles:
        return f"Stack them up: dead level — {a.wins} wins and {_count(most)} titles each."
    if level_titles:
        both = f", level on titles at {_count(most)}" if most else ""
        return f"Stack them up: {by_wins[0].name} takes it — {on_wins}{both}."
    if level_wins:
        return (f"Stack them up: {by_titles[0].name} takes it — {on_titles}, "
                f"level on wins at {a.wins}.")
    if by_wins[0] is by_titles[0]:
        return f"Stack them up: {by_wins[0].name} takes it — {on_wins}, {on_titles}."
    return (f"Stack them up: it's split — {by_wins[0].name} on wins, {wins}; "
            f"{by_titles[0].name} on titles, {titles}.")


def render(fact: dict, intent: str | None = None) -> str:
    """The reply a single fact dictates: its trained format where it has one,
    its own sentence otherwise (the adapter recites curated notes verbatim)."""
    text = fact.get("text", "")
    race = parse_race(text)
    if race:
        return T.race_winner_reply(race.winner, race.year, race.gp, race.team, race.pole,
                                   list(race.podium), race.venue)
    title = parse_champion(text)
    if title:
        return T.championship_reply(title.holder, title.year)
    title = parse_constructors(text)
    if title:
        return T.constructors_reply(title.holder, title.year)
    bio = parse_bio(text)
    if bio:
        return {"tally": _tally, "driver_titles": _titles}.get(intent, _card)(bio)
    circuit = parse_circuit(text)
    if circuit:
        return T.circuit_reply(circuit.name, f"That's {circuit.where} — one for the bucket list.")
    entry = parse_entry(text)
    if entry and entry.teams:
        teams = " → ".join(entry.teams)
        if entry.year:
            return f"Checking the telemetry on {entry.name} — {entry.year} season: {teams}."
        if intent == "car_number" and entry.number:
            return (f"Checking the telemetry on {entry.name} — car number {entry.number}, "
                    f"he's at {teams}.")
        return f"Checking the telemetry on {entry.name} — he's at {teams}."
    return text


# --------------------------------------------------------------------------- #
# The plan for a turn
# --------------------------------------------------------------------------- #

# The model is only ever asked who won, so its recap opens with the winner
# whatever was asked: "who was on pole there?" got "Max Verstappen got the
# chequered flag ..." with the pole sitter three sentences in (measured
# 2026-10-13). The detail asked for is read off the record and said first.
_POLE_CUE = re.compile(r"\bpole\b|\bqualif", re.I)
_SECOND_CUE = re.compile(
    r"\b(?:finish(?:ed)?|came|come|was|placed?) (?:in )?(?:second|2nd)\b"
    r"|\b(?:second|2nd) place\b|\brunner[- ]up\b|\bP2\b", re.I)
_THIRD_CUE = re.compile(
    r"\b(?:finish(?:ed)?|came|come|was|placed?) (?:in )?(?:third|3rd)\b"
    r"|\b(?:third|3rd) place\b|\bP3\b", re.I)
_VENUE_CUE = re.compile(r"\bwhere\b|\b(?:which|what) (?:circuit|track|venue)\b", re.I)


def _race_lead(question: str, race) -> str:
    """The sentence that answers what ``question`` asks of ``race`` when that
    is not its winner; "" when the recap already opens with the answer."""
    event = f"the {race.year} {race.gp}"
    if _POLE_CUE.search(question) and race.pole:
        return f"Pole position at {event} went to {race.pole}."
    if _SECOND_CUE.search(question) and race.podium:
        return f"{race.podium[0]} finished second at {event}."
    if _THIRD_CUE.search(question) and len(race.podium) > 1:
        return f"{race.podium[1]} finished third at {event}."
    if _VENUE_CUE.search(question) and race.venue:
        return f"{event[0].upper()}{event[1:]} was held at {race.venue}."
    return ""


def plan(question: str, facts: list[dict], intent: str | None = None) -> Plan:
    """How to answer ``question`` from ``facts`` (see module docstring)."""
    if len(facts) == 2 and intent == "compare":
        a, b = (parse_bio(f["text"]) for f in facts)
        if a and b:
            # Putting two careers side by side is arithmetic, not recitation
            # (the adapter was trained to compare stated title counts, and a
            # bio lists title seasons): the reply is computed, not generated.
            return Plan(question, facts, _compare(a, b, question), kind="compare",
                        settled=True)
    if len(facts) == 1 and facts[0].get("source") == "computed":
        # A count made over the records (rag/stats.py) is its own answer.
        return Plan(question, facts, facts[0]["text"], kind="stat", settled=True)
    if len(facts) == 1 and parse_standing(facts[0]["text"]):
        # So is where a championship stands. The adapter knows one way to
        # talk about a championship — "And that's the title! ... is the 2026
        # World Champion" — and the leader of a season still being run is
        # not that, however many of the sentence's words the check finds.
        return Plan(question, facts, facts[0]["text"], kind="standings", settled=True)
    if len(facts) != 1:
        # Several facts and no one reply they dictate: the model's reply
        # stands if it stays within them. Asked about two drivers, it has to
        # cover both; their two cards are the reply when it does not.
        if intent == "drivers":
            bios = [parse_bio(f["text"]) for f in facts]
            return Plan(question, facts, "\n".join(render(f) for f in facts),
                        must_name=tuple(b.name for b in bios if b))
        return Plan(question, facts)

    fact = facts[0]
    text = fact["text"]
    if intent == "team_entry" and re.search(r"\bnumber\b", question, re.I):
        intent = "car_number"
    reply = render(fact, intent)
    race = parse_race(text)
    if race:
        # The shipped adapter was trained on winner questions only; "Tell me
        # about ..." and "Who was on the podium ..." shift every name by one.
        return Plan(f"Who won the {race.year} {race.gp}?", facts, reply, strict=True, kind="race",
                    lead=_race_lead(question, race))
    title = parse_champion(text)
    if title:
        return Plan(f"Who won the {title.year} Formula 1 World Championship?", facts, reply,
                    strict=True, kind="champion")
    title = parse_constructors(text)
    if title:
        return Plan(f"Which team won the {title.year} Constructors' Championship?", facts,
                    reply, strict=True, kind="constructors")
    # From here on the fact is about one driver, circuit or topic, and reading
    # it out answers the question only if that is who was asked about: with
    # "Fernando Alonso currently races for Aston Martin" in hand, the honest
    # reply to "Which team does Alexander Albon drive for?" is a decline.
    asked = _names(question)
    if asked and not any(_held(name, text) for name in asked):
        return Plan(question, facts, kind="other subject")
    if parse_bio(text) or parse_circuit(text) or parse_entry(text):
        return Plan(question, facts, reply, strict=True, kind=intent or "card")
    lineup = parse_lineup(text)
    # A note or a line-up: the fact's own sentence is the reply, and a
    # faithful retelling of it is as good.
    return Plan(question, facts, reply, kind="recite",
                must_name=lineup.drivers if lineup else ())


# --------------------------------------------------------------------------- #
# Did the model say it?
# --------------------------------------------------------------------------- #

_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?")
_POSSESSIVE = re.compile(r"['’]s\b")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
# Two or more Capitalized words in a row: a name the reply asserts.
_NAME = re.compile(r"\b[A-Z][\w'’\-]+(?: (?:de |von |van |der |di |da )?[A-Z][\w'’\-]+)+")
# Capitalized words that are not part of anyone's name: the reply formats'
# own ("Grand Prix", "World Champion"), and how a sentence can open.
_NOT_NAME_WORDS = {"grand", "prix", "world", "champion", "championship", "championships",
                   "pole", "position", "podium", "driving", "teams", "held", "checking",
                   "constructors", "stack", "box", "lights", "let's", "formula", "the", "and",
                   "that's", "driver", "drivers", "f1", "one", "a", "an", "or", "in", "on",
                   "at", "of", "for", "me", "about", "what", "who", "when", "where", "which",
                   "why", "how", "is", "are", "was", "were", "did", "does", "do", "can",
                   "tell", "explain", "give", "now", "so", "but", "well", "yes", "no"}
# Words that carry no claim: a reply made only of these and the fact's own
# words says nothing the fact does not.
_FILLER = set("""a an the and or but so of to in on at for from with by as is are was were be been
it its it's this that these those there here he she his her him they them their you your we our i
me my not no yes do does did has have had will would can could should may might just only also
very really quite much more most some any all both each than then when while where which who whom
what how why if because about into over under up down out off again ever never always
let's talk racecraft checking telemetry tally board reads lights go box title that's stack
them take takes got get one two three four five six seven eight nine ten first""".split())


def _norm(text: str) -> str:
    """Compare replies on their words: case, punctuation and typographic
    variants (curly quotes, dashes) do not make two answers different."""
    return " ".join(t.lower().replace("’", "'") for t in _TOKEN.findall(text))


def _stem(word: str) -> str:
    return word[:5] if len(word) > 5 else word


def _numbers(text: str) -> set[str]:
    """Numbers ``text`` states, digits and words alike ("seven" is 7; "one"
    is too often a pronoun to count)."""
    found = {n.replace(",", "") for n in _NUMBER.findall(text)}
    words = set(_norm(text).split())
    return found | {str(n) for n, word in enumerate(_COUNTS) if n >= 2 and word in words}


def _names(text: str) -> list[list[str]]:
    """The names ``text`` uses: runs of two or more Capitalized words, as
    lowercase word lists."""
    names = []
    for run in _NAME.findall(text):
        words = [_POSSESSIVE.sub("", w).strip("'’").lower() for w in run.split()]
        words = [w for w in words if w not in _NOT_NAME_WORDS]
        if len(words) >= 2:
            names.append(words)
    return names


def _held(name: list[str], source: str) -> bool:
    """Does ``source`` hold this name? Any two of its words in a row will do:
    a stray capitalized word may ride along with a name ("Honestly Lewis
    Hamilton..."), but a blend of two ("Nico Mansell") has no pair in common
    with either."""
    have = " " + _norm(_POSSESSIVE.sub("", source)) + " "     # "Schumacher's" is Schumacher
    return any(f" {a} {b} " in have for a, b in zip(name, name[1:]))


def _unsupported(reply: str, source: str) -> list[str]:
    """Names and numbers in ``reply`` that ``source`` does not hold."""
    missing = sorted(_numbers(reply) - _numbers(source))
    return missing + [" ".join(n) for n in _names(reply) if not _held(n, source)]


_SENTENCE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")


def _retold(reply: str, source: str) -> bool:
    """Is ``reply`` a retelling of ``source`` — are its words the source's?

    Short sentences that claim nothing checkable ("Love that you asked!",
    "Want the full story?") are the persona's trained decoration and are not
    counted; what remains must be mostly the source's own vocabulary.
    """
    have = {_stem(t) for t in _norm(source).split()}
    words = _claims(reply)
    matched = sum(_stem(w) in have for w in words)
    # Three words out of five in a real sentence; in a very short one, any.
    return matched >= (0.6 * len(words) if len(words) >= 6 else min(1, len(words)))


def _claims(reply: str) -> list[str]:
    """The words of ``reply`` that say something: everything but filler, and
    but the short checkable-nothing sentences the persona decorates with."""
    words: list[str] = []
    for i, sentence in enumerate(_SENTENCE.findall(reply)):
        tokens = _norm(sentence).split()
        decoration = (i > 0 and len(tokens) <= 8 and not _numbers(sentence)
                      and not _names(sentence))
        if not decoration:
            words += [t for t in tokens if t not in _FILLER and len(t) > 2]
    return words


def _after_words(text: str, n: int) -> str:
    """What follows the first ``n`` words of ``text``."""
    ends = [m.end() for m in _TOKEN.finditer(text)]
    return text[ends[n - 1]:] if 0 < n <= len(ends) else ""


_FINISHED = re.compile(r"[^.!?]+[.!?]+(?=\s|$)")
_COMPARISON = re.compile(r"\W*stack (?:them )?up\b", re.I)
_DECLINE = _norm(T.DECLINE)


def _decline(reply: str, plan_: Plan) -> str:
    """The trained decline, if ``reply`` opens with it and the facts dictate
    no reply of their own (where they do, declining is not an answer).

    The decline and no more. As trained it goes on to say what its context
    does cover — "What I've got here covers the 1964 season, nothing else",
    to "Who won the first ever F1 race?" (2026-10-11) — which describes the
    search, to someone who cannot see the search.
    """
    first = _FINISHED.match(reply.lstrip())
    if plan_.reply is None and first and _norm(first.group(0)) == _DECLINE:
        return first.group(0).strip()
    return ""


def declined(partial: str, plan_: Plan) -> bool:
    """Has the model declined? Then the reply is complete, whatever it is
    still writing."""
    return bool(_decline(partial, plan_))


def _closed(reply: str) -> str:
    """``reply`` with the full stop the model ends without: its "Let's talk
    racecraft: ..." format was trained that way."""
    return reply + "." if re.search(r"[^\W_][\"'”’)\]]*$", reply) else reply


def _fault(reply: str, plan_: Plan) -> str | None:
    """What is wrong with a freely worded ``reply``, or None if it holds:
    every name and number in it must come from the facts, and most of its
    words too."""
    facts_text = "\n".join(f.get("text", "") for f in plan_.facts)
    if plan_.kind == "other subject":
        # True of the driver in the fact, and not who was asked about:
        # "Which team does Yuki Tsunoda drive for?" - "Checking the telemetry
        # on Oscar Piastri — he's at McLaren."
        return "the facts are about someone else"
    if _COMPARISON.match(reply):
        # "Stack them up: Fangio takes it — five championships to four" uses
        # only names and numbers the facts hold, and is wrong (a third fact
        # said seven). Which of several facts is the greater is not something
        # a word-by-word check can vouch for; where it can be worked out, it
        # is computed (plan: "compare").
        return "a comparison this check cannot vouch for"
    missing = _unsupported(reply, facts_text)
    if missing:
        return f"not in the facts: {', '.join(missing[:4])}"
    left_out = [n for n in plan_.must_name if _norm(n) not in _norm(reply)]
    if left_out:
        return f"left out: {', '.join(left_out)}"
    if not _retold(reply, facts_text):
        return "mostly not from the facts"
    return None


def _sound_start(reply: str, plan_: Plan) -> str:
    """The longest run of finished sentences opening ``reply`` that holds and
    says something — at least five words of the facts' own. "Lights out and
    here we go!" holds, and is not an answer."""
    own = _decline(reply, plan_)
    if own:
        return own
    facts_text = "\n".join(f.get("text", "") for f in plan_.facts)
    have = {_stem(t) for t in _norm(facts_text).split()}
    sentences = _FINISHED.findall(reply)
    for n in range(len(sentences), 0, -1):
        part = "".join(sentences[:n]).strip()
        if (_fault(part, plan_) is None
                and sum(_stem(w) in have for w in _claims(part)) >= 5):
            return part
    return ""


def check(reply: str, plan_: Plan) -> tuple[str, str]:
    """``(the reply to show, why)``: the model's own if it holds, else the
    plan's. ``why`` is "kept", or what was wrong with the model's reply.
    """
    reply = reply.strip()
    facts_text = "\n".join(f.get("text", "") for f in plan_.facts)
    fallback = plan_.reply or T.DECLINE
    if not reply:
        return fallback, "empty"

    if plan_.strict:
        # The trained format, word for word, and at most a short flourish
        # after it ("— what a season that was.").
        want, got = _norm(plan_.reply), _norm(reply)
        if not got.startswith(want):
            return fallback, "not the reply the fact dictates"
        extra = _after_words(reply, len(want.split()))
        if len(extra) > 90 or _unsupported(extra, facts_text + "\n" + plan_.reply):
            return fallback, "added to the fact"
        return _closed(reply), "kept"

    own = _decline(reply, plan_)
    if own:
        return own, "kept"
    fault = _fault(reply, plan_)
    if fault is None:
        return _closed(reply), "kept"
    # A reply that starts well and then strays is cut where it strays: its
    # sound opening is the model's own answer, and was already on screen.
    start = _sound_start(reply, plan_)
    return (start, f"cut short ({fault})") if start else (fallback, fault)


def hopeless(partial: str, plan_: Plan) -> bool:
    """Is there nothing to be gained by letting ``partial`` — a reply still
    being written — go on? Measured 2026-10-11: a reply that was going to be
    replaced anyway took 30 seconds to finish first. Only its finished
    sentences are judged; a name or number at the end may be half-written.
    """
    if plan_.strict:
        said, want = _norm(partial), _norm(plan_.reply)
        return not (want.startswith(said) or said.startswith(want))
    finished = _FINISHED.findall(partial)
    whole = "".join(finished).strip()
    if not whole or _decline(whole, plan_):
        return False
    facts_text = "\n".join(f.get("text", "") for f in plan_.facts)
    if _unsupported(whole, facts_text):
        return True
    # One real sentence in (or two of any kind) and still not the facts' own
    # words: it is not going to turn into a retelling.
    said_enough = len(finished) >= 2 or len(_claims(whole)) >= 8
    return said_enough and not _retold(whole, facts_text)


def presentable(partial: str, plan_: Plan) -> str:
    """As much of a reply still being written as can be shown and will not
    have to be taken back: its sound opening (see :func:`check`)."""
    return _sound_start(partial, plan_)
