"""
dialogue.py — what a follow-up question refers back to.

The adapter is trained single-turn: a grounded turn is answered without chat
history (with it, the model imitates its own earlier formatted answers). So
whatever a question leaves to the conversation — "he", "that race", "And in
2007?" — has to be written into the question before retrieval and the model
see it.

That used to be worked out from the raw text of the last exchange. Typing a
longer chat into the running app (2026-10-11) showed where that stops:

    Who won the 2008 Singapore Grand Prix?     (answered)
    Who was on the podium in that race?        (resolved, answered)
    And in 2007?            -> the 2007 World Champion: the question it was
                               meant to re-ask has no year until "that race"
                               is written out
    And in 2010?            -> the 2010 constructors' champion
    Who was on pole?        -> a name from an earlier answer: no race attached
    Which team won the constructors title that year?   -> "Constructors'
                               crown, 1973!"

Each follow-up is fine against a question that says everything, and lost
against one that was itself a follow-up. So the chat is replayed from the top:
every user turn is resolved against the topic so far and then moves the topic
on (:func:`follow`), and a new question is resolved against where that ends
(:func:`references`, :func:`followup`). Nothing is stored between turns — the
topic is a function of the transcript.

Only references with one sensible reading are resolved. "they" and a bare
"it" are usually generic ("What tyres do they use in the wet?", "What does it
mean...") and are left alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from pitgpt.rag.records import parse_race

Message = dict[str, str]


def text_of(content) -> str:
    """Coerce a message ``content`` to a plain string.

    The chat template concatenates ``content`` as a string, but some UIs
    (notably Gradio 6's chatbot) store content as a *list* of content-parts, e.g.
    ``[{"type": "text", "text": "…"}]``. Passing that straight to the template
    raises ``TypeError: can only concatenate str (not "list") to str``. We flatten
    any such structure back to text so history round-trips safely.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        return " ".join(p for p in parts if p).strip()
    return "" if content is None else str(content)


@dataclass(frozen=True)
class Topic:
    """Where the conversation stands."""
    subject: str | None = None      # who it is about: what "he" means
    year: str | None = None         # the season last in focus: "that year"
    dated: bool = False             # ...and it was in the last thing discussed
    race: tuple[str | None, str] | None = None   # (year, Grand Prix) last discussed
    on_race: bool = False           # ...and that race is what was just discussed
    question: str | None = None     # the last full question, references written out


# Capitalized TITLE phrases are descriptions, not names: "my favourite driver
# is the World Champion" produced "your favourite driver on record is World
# Champion" live, and the card answered the wrong entity. As a recalled name,
# "World Champion" also dragged a teams question to the champion docs.
# Exact-match only — "Monaco Grand Prix" stays a valid name.
_NON_NAMES = {"world champion", "world championship", "grand prix",
              "formula one", "formula 1", "drivers championship",
              "world drivers championship"}

_PERSON_PRONOUN = re.compile(r"\b(he|she|him|his|hers|her)\b", re.I)
# "Which driver crashed his car?" — that pronoun is bound inside the question.
_BOUND_PRONOUN = re.compile(
    r"\b(?:which|what|a|an|any|every|each|no) "
    r"(?:driver|racer|champion|winner|rookie|teammate)s?\b", re.I)
# "her" is a possessive before a noun ("her teams") and an object otherwise
# ("who beat her in 2019?"); these words after it are never the noun.
_NOT_A_NOUN = {"in", "at", "to", "for", "on", "and", "or", "when", "while",
               "then", "there", "that", "this", "too", "as"}
_THAT_SEASON = re.compile(r"\bthat (?P<kind>season|year)\b", re.I)
# "that race", or an "it" in a race-shaped slot ("who won it", "on pole for it").
_THAT_RACE = re.compile(
    r"\bthat (?:race|grand prix|gp)\b"
    r"|\b(?:won|win|wins|winning|about|at|in|for) (?P<it>it)\b", re.I)
# A question about one race's result, as opposed to "tell me about the 2008
# season": only these inherit the race the chat was on.
_RACE_DETAIL = re.compile(
    r"\b(pole|podium|won|win|winner|second|third|finish(?:ed)?|fastest lap|grid|held)\b", re.I)
_A_RACE = re.compile(r"\b(?:race|races|grand prix|gp|round|season|championship)\b", re.I)
_PREPOSITION_BEFORE = re.compile(r"\b(?:in|during|of|for|from|since|until|by)\s+$", re.I)
_LAST_SEASON = re.compile(r"\b(?:last|previous) (?:year|season)\b", re.I)
_THIS_SEASON = re.compile(r"\b(?:this|the current|current) (?:year|season)\b", re.I)

# "And in 2007?" / "What about Hamilton?" re-ask the previous question with
# one thing swapped. Strict on purpose: the whole message must be the swap.
_FOLLOWUP = re.compile(
    r"^\W*(?:and|(?:and )?(?:what|how) about)\s+(?P<rest>.+?)\W*$", re.I)
_LEAD_IN = re.compile(r"^(?:(?:in|for|at|the)\s+)+", re.I)


def _splice(text: str, span: tuple[int, int], new: str) -> str:
    return text[:span[0]] + new + text[span[1]:]


def _real_names(text: str) -> list[str]:
    """Proper names in ``text`` (original case), minus title phrases."""
    from pitgpt.rag.retriever import _PIN_NAME

    return [n for n in _PIN_NAME.findall(text)
            if n.lower().replace("'", "") not in _NON_NAMES]


def _entities():
    from pitgpt.rag.retriever import _get_entities

    return _get_entities()


def _latest_season() -> int | None:
    entities = _entities()
    years = [int(race.year) for race, _ in entities.races] if entities else []
    return max(years) if years else None


def _driver_named(text: str) -> str | None:
    """The first driver ``text`` names. Without an index (or one that knows
    no drivers): its first proper name that is not a race or a venue."""
    entities = _entities()
    drivers = entities.drivers_in(text) if entities else []
    if drivers:
        return drivers[0].name
    if entities and entities.drivers:
        return None
    return next((n for n in _real_names(text)
                 if not n.lower().endswith(("grand prix", "circuit"))), None)


def _in(year_phrase: str, text: str, start: int) -> str:
    """``year_phrase`` with "in" in front unless the sentence already has its
    preposition there ("...the title that year" -> "...the title in 2010")."""
    return year_phrase if _PREPOSITION_BEFORE.search(text[:start]) else f"in {year_phrase}"


def followup(text: str, topic: Topic) -> str | None:
    """Rewrite a bare follow-up — "And in 2007?", "What about Hamilton?",
    "And McLaren?", "And Monaco?" — as the last full question with that one
    thing swapped in. None if this isn't one, or there is nothing to re-ask.

    Left alone such a message carries a year or a name and nothing else, so
    retrieval answered with whatever that year's top document was (measured:
    after two Singapore GP questions, "And in 2007?" got a blended 2007 season
    recap instead of "that race wasn't held").
    """
    from pitgpt.rag.retriever import _PIN_YEAR, _extract_gp

    m = _FOLLOWUP.match(text)
    last = topic.question
    if not (m and last):
        return None
    rest = _LEAD_IN.sub("", m["rest"]).strip()

    if _PIN_YEAR.fullmatch(rest):
        years = set(_PIN_YEAR.findall(last))
        if len(years) == 1:
            return last.replace(years.pop(), rest)
        # no year to swap: the last question took the latest one for granted
        return f"{last.rstrip(' ?!.')} in {rest}?" if not years else None

    entities = _entities()
    if entities is None:
        return None

    def only(found) -> bool:
        """Is the entity all there is to the follow-up?"""
        return bool(found) and not re.sub(r"[\W_]+", "", rest[:found[0]] + rest[found[1]:])

    driver = entities.driver_at(rest)
    if only(driver):
        old = entities.driver_at(last)
        return _splice(last, old[:2], driver[2].name) if old else None

    gp = _extract_gp(rest)
    circuit = entities.circuit_at(rest)
    if gp or only(circuit):
        if not gp:      # "And Monza?" names the race by its circuit
            doc = entities.latest_race(
                lambda race, _: race.venue.startswith(circuit[2] + " in "))
            gp = parse_race(doc["text"]).gp if doc else None
        old_gp = _extract_gp(last)
        if gp and old_gp:
            name = re.escape(old_gp.rsplit(" Grand Prix", 1)[0])
            return re.sub(name + r"\s+(?:grand prix|gp)\b", lambda _: gp, last,
                          count=1, flags=re.I)
        old = entities.circuit_at(last)      # "Who won at Monza in 2019?" - "And Spa?"
        if old:
            return _splice(last, old[:2], rest[circuit[0]:circuit[1]] if only(circuit)
                           else f"the {gp}")
        return None

    team = entities.team_at(rest)
    if only(team):
        old = entities.team_at(last)
        return _splice(last, old[:2], rest[team[0]:team[1]]) if old else None
    return None


def references(text: str, topic: Topic) -> str:
    """``text`` with what it leaves to the chat written out: "What teams did
    he race for?" becomes "What teams did Nigel Mansell race for?".

    Measured 2026-10-11 with the right fact in context, on the question as
    typed: "...on the podium in that race?" got a fabricated recap of another
    Grand Prix, and "when did he win..." one title year of four — both right
    once the question named its subject.
    """
    from pitgpt.rag.retriever import (
        _NOW_CUE,
        _PIN_YEAR,
        _asks_beyond_the_race,
        _extract_gp,
        _intent,
    )

    entities = _entities()

    # "last year" / "this season" -> the season itself, counted from the
    # records rather than the clock.
    latest = _latest_season()
    if latest and not _PIN_YEAR.search(text):
        for cue, year in ((_LAST_SEASON, latest - 1), (_THIS_SEASON, latest)):
            m = cue.search(text)
            if m:
                text = _splice(text, m.span(), _in(str(year), text, m.start()))
                break

    # 1. "he" / "his" -> whoever the chat is about. A question that names its
    #    own driver has already bound its pronouns.
    named = _driver_named(text)
    pronoun = None if named else _PERSON_PRONOUN.search(text)
    if pronoun and _BOUND_PRONOUN.search(text[:pronoun.start()]):
        pronoun = None
    if pronoun and topic.subject:
        word = pronoun.group(1).lower()
        possessive = word in ("his", "hers")
        if word == "her":
            noun = re.match(r"\s+([a-z']+)", text[pronoun.end():], re.I)
            possessive = bool(noun) and noun.group(1).lower() not in _NOT_A_NOUN
        text = _splice(text, pronoun.span(),
                       f"{topic.subject}'s" if possessive else topic.subject)

    # 2. "that year" / "that season" -> the season last in focus, whatever is
    #    being asked about it.
    own_year = _PIN_YEAR.search(text)
    season = _THAT_SEASON.search(text)
    if season and topic.year and not own_year:
        phrase = f"the {topic.year} season" if season["kind"].lower() == "season" else topic.year
        text = _splice(text, season.span(), _in(phrase, text, season.start()))
        own_year = _PIN_YEAR.search(text)

    # 3. A title question with no year, straight after something dated: "Who
    #    won the 2016 title?" - "And the constructors?" means that season's.
    #    Not one asked in the present tense: "Who is leading the championship?"
    #    after a 2024 answer got "...the 2024 World Champion" (2026-10-11).
    beyond = _asks_beyond_the_race(text)
    if (beyond and not own_year and not named and topic.dated and topic.year
            and _intent(text) in ("titles", "constructors") and not _NOW_CUE.search(text)):
        return f"{text.rstrip(' ?!.')} in {topic.year}?"

    # 4. The race under discussion. Not when the question asks for titles,
    #    tallies or team histories: a title question in the same year is not
    #    about that race, and a GP it never mentioned must not reach the
    #    absence guard.
    if not topic.race or beyond or _extract_gp(text):
        return text
    year, gp = topic.race
    cue = _THAT_RACE.search(text)
    if cue:                                           # "that race" / "who won it"
        span = cue.span("it") if cue.group("it") else cue.span()
        # the question's own year wins: "Who won it in 2009?"
        return _splice(text, span, " ".join(filter(None, ("the", None if own_year else year, gp))))
    # A result asked for with no race in sight ("Who was on pole?") is about
    # the race in hand; one that names a race of its own ("Who won the last
    # race?", "...at Monza?") is not.
    if (not _RACE_DETAIL.search(text) or _A_RACE.search(text)
            or (entities and entities.circuit_in(text))):
        return text
    bare = text.rstrip(" ?!.")
    if own_year:
        # "Who was on pole in 2008?" right after the 2008 Singapore GP: the
        # same year, no race named — it means that race.
        return f"{bare} at the {gp}?" if own_year.group(0) == year else text
    # "Who was on pole?" — of the race just talked about.
    return f"{bare} at the {' '.join(filter(None, (year, gp)))}?" if topic.on_race else text


def _race_in(question: str, answer: str) -> tuple[str | None, str] | None:
    """``(year, Grand Prix)`` an exchange was about, read from the question
    first (the user's own framing), then from the answer."""
    from pitgpt.rag.retriever import _PIN_YEAR, _display_gp, _extract_gp

    for text in (question, answer[:220]):
        gp = _extract_gp(text) if text is question else next(
            (n for n in _real_names(text) if n.lower().endswith(" grand prix")), None)
        if gp:
            year = _PIN_YEAR.search(text) or _PIN_YEAR.search(answer[:220])
            return (year.group(0) if year else None), _display_gp(gp)
    return None


# The year an ANSWER is about, where its format says so. A driver card's
# "raced 2007–2026" is a year in an answer too, and not what "that year" means.
_ANSWER_YEAR = re.compile(
    r"\bis the (\d{4}) World Champion\b|\bConstructors' crown, (\d{4})\b"
    r"|\bno (\d{4}) [A-Z]", re.I)


def _after(topic: Topic, question: str, answer: str) -> Topic:
    """The topic once ``question`` (already resolved) got ``answer``."""
    from pitgpt.rag.retriever import _PIN_YEAR, _asks_beyond_the_race, _is_conversational

    # Walking the chat forwards, an answer comes after its question, and every
    # trained reply format leads with its subject: its first name is who the
    # chat is now about. A nameless reply (persona chat, a decline) defers to
    # the question, and that to the turns before.
    subject = _driver_named(answer[:300]) or _driver_named(question) or topic.subject
    if _is_conversational(question):
        return replace(topic, subject=subject)        # small talk moves nothing else
    race = None if _asks_beyond_the_race(question) else _race_in(question, answer)
    asked = _PIN_YEAR.search(question)
    told = _ANSWER_YEAR.search(answer[:220])
    year = (asked.group(0) if asked else (race and race[0])
            or (next(filter(None, told.groups())) if told else None))
    return Topic(
        subject=subject,
        year=year or topic.year,
        dated=bool(year),
        race=race or topic.race,
        on_race=race is not None,
        question=question,
    )


def resolve(text: str, topic: Topic) -> str:
    """A user message as a question that stands on its own: drivers by full
    name, follow-ups and back-references written out."""
    from pitgpt.rag.retriever import canonical

    text = canonical(text)
    return references(followup(text, topic) or text, topic)


def follow(history: list[Message] | None) -> Topic:
    """The topic at the end of ``history`` (Gradio "messages" format)."""
    topic = Topic()
    asked: str | None = None
    for turn in history or []:
        content = text_of(turn.get("content"))
        if not content:
            continue
        if turn.get("role") == "user":
            if asked is not None:                     # two user turns in a row
                topic = _after(topic, asked, "")
            asked = resolve(content, topic)
        elif turn.get("role") == "assistant" and asked is not None:
            topic = _after(topic, asked, content)
            asked = None
    return topic if asked is None else _after(topic, asked, "")
