"""
retriever.py — a lazy, shared retriever and prompt-grounding helper.

The app and inference engine call ``find(query)`` to get the facts a question
needs. The index is loaded once (lazily) and reused. If no index has been
built, retrieval degrades gracefully to "no context" so the app still runs
(just without grounding).

A question goes through three steps here:

1. it is written out in canonical form — every driver by the full name the
   corpus uses (``entities.py``);
2. if it names its subject plainly ("Tell me about Lewis Hamilton", "Who
   drives for Ferrari?", "Who won at Monza in 2019?") the one fact document
   that answers it is looked up directly (``_pick``);
3. otherwise the hybrid search runs over the doc families its wording routes
   to, and the results are pinned to the entities it names.

Step 2 exists because the shipped model recites one fact reliably and blends
several: the fewer documents reach the prompt, the better the answer.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

from pitgpt.data.templates import absence_reply, no_record_reply
from pitgpt.rag.entities import EntityIndex, team_root
from pitgpt.rag.index import FactRetriever
from pitgpt.rag.records import (
    parse_champion,
    parse_constructors,
    parse_entry,
    parse_lineup,
    parse_race,
    parse_standing,
)
from pitgpt.rag.stats import computed

logger = logging.getLogger("pitgpt.rag.retriever")

_retriever: FactRetriever | None = None
_load_failed = False
_entities: tuple[object, EntityIndex] | None = None    # (the retriever it was built from, index)


def _get_retriever() -> FactRetriever | None:
    """Load the persisted index once; return None if it isn't available."""
    global _retriever, _load_failed
    if _retriever is not None:
        return _retriever
    if _load_failed:
        return None
    try:
        _retriever = FactRetriever().load()
        return _retriever
    except Exception as exc:  # noqa: BLE001 - missing index is non-fatal
        logger.warning("RAG index unavailable (%s); running without retrieval.", exc)
        _load_failed = True
        return None


def _get_entities() -> EntityIndex | None:
    """The entity index of the loaded corpus (rebuilt if the corpus changes)."""
    global _entities
    retriever = _get_retriever()
    if retriever is None:
        return None
    if _entities is None or _entities[0] is not retriever:
        _entities = (retriever, EntityIndex(retriever.docs))
    return _entities[1]


def canonical(query: str) -> str:
    """``query`` with its drivers written as the corpus names them."""
    entities = _get_entities()
    return entities.canonicalize(query) if entities else query


# Query-aware source routing. Popular entities have dozens of race-winner docs
# that crowd out their single bio doc; routing the query to the right doc
# family keeps retrieval precise. Order matters: championship phrasing wins
# over race phrasing ("who won the 2016 championship" is not a race question).
# Each route: (pattern, what is being asked, the doc families that answer it).
_ROUTES: list[tuple[re.Pattern, str, list[str]]] = [
    (re.compile(r"constructor", re.I), "constructors",
     ["kaggle_constructors", "jolpica_constructors", "curated"]),
    (re.compile(r"champion|title", re.I), "titles",
     ["kaggle_champion", "jolpica_champion", "curated"]),
    # Win-tally questions get the bio docs ONLY: the curated notes discuss
    # titles in prose ("seven championships...") and outranked the bio's exact
    # win count in the prompt — measured live as "Seven" for Hamilton's 105.
    (re.compile(r"how many.*(win|won|race)|win tally|wins (does|did|has)|"
                r"grand prix wins", re.I), "tally",
     ["kaggle_driver", "jolpica_driver"]),
    # Team-HISTORY questions get bio docs only too: the curated star prose
    # ("...has driven for McLaren, Mercedes and, from 2025, Ferrari") outranked
    # Hamilton's bio doc and the card format drifted into rambling — measured
    # in the diag_jul06 benchmark (the one true driver_teams miss).
    (re.compile(r"\bteams? (did|has)\b|\braced for\b|\bdrove for\b", re.I), "teams",
     ["kaggle_driver", "jolpica_driver"]),
    # A year alongside "Grand Prix"/"GP" is race intent regardless of verb.
    # "Tell me about the 2021 Abu Dhabi GP" previously fell through to the
    # DRIVER route below ("tell me about"), which starved the absence guard of
    # race docs — it then declared real races "not held" (measured live).
    (re.compile(r"^(?=.*\b(?:19[5-9]\d|20\d\d)\b)(?=.*\b(?:grand prix|gp)\b)",
                re.I | re.S), "race",
     ["kaggle_race", "jolpica_race"]),
    # Circuit questions must reach circuit docs even when phrased "tell me
    # about {circuit}" — previously hijacked by the driver route below.
    (re.compile(r"circuit|track\b|where is", re.I), "circuit",
     ["kaggle_circuit", "curated"]),
    # Driver-centric phrasings outrank the race route: "career Grand Prix win
    # tally" is a question about a DRIVER even though it contains "Grand Prix".
    # who(?:'s| is| was): the old class [' i] allowed only ONE char between
    # "who" and "s", so "Who is Fernando Alonso?" never routed at all.
    (re.compile(r"who(?:'s| is| was)\b|tell me about|rundown on|teams? did|drive for|"
                r"race for|career|win tally|how many", re.I), "driver",
     ["kaggle_driver", "jolpica_driver", "curated", "openf1"]),
    # A yearless race question still needs a winner-verb, not just "grand prix".
    (re.compile(r"(who won|winner|victory|took .*(flag|win)|crossed the line)"
                r".*(grand prix|\bgp\b)", re.I), "race",
     ["kaggle_race", "jolpica_race"]),
]


def _route(query: str) -> list[str] | None:
    for pattern, _, sources in _ROUTES:
        if pattern.search(query):
            return sources
    return None


def _intent(query: str) -> str | None:
    """What kind of fact ``query`` asks for, by the first route it matches."""
    for pattern, label, _ in _ROUTES:
        if pattern.search(query):
            return label
    return None


def _asks_beyond_the_race(query: str) -> bool:
    """True if ``query`` asks for a doc family that outranks race intent —
    titles, constructors, win tallies, team histories. Such a question is not
    about whichever race the chat last discussed, even in the same year, so
    that race must not be attached to it."""
    for pattern, label, _ in _ROUTES:
        if label == "race":
            return False
        if pattern.search(query):
            return True
    return False


# F1-content cue: a year, a named entity, or a domain keyword. Conversational
# turns ("Yes tell me.", "thanks!", "haha ok") have none of these — retrieval
# must stay SILENT there, or the grounding-trained model will faithfully
# recite whatever irrelevant doc comes back.
#
# Whole words only. The keywords used to match anywhere, so "ers" (the hybrid
# system) fired inside "Verstappen" and "Silverstone": measured 2026-10-11 as
# the only reason "who is verstappen" was searched at all while "Tell me
# about Hamilton" was not.
_F1_KEYWORDS = re.compile(
    r"\b(?:19[5-9]\d|20\d\d)\b"                       # a season year
    r"|\b(?:drivers?|teams?|team-?mates?|rac(?:e|es|ed|er|ers|ing)|grand prix|gp|"
    r"champion(?:s|ship|ships)?|titles?|circuits?|tracks?|tyres?|tires?|"
    r"pit(?:s|\s?stops?|\s?lane|\s?wall)?|drs|ers|qualifying|qualify|qualified|quali|"
    r"poles?|overtak(?:e|es|en|ing)|constructors?|f1|formula|seasons?|"
    r"wins?|won|winners?|victor(?:y|ies)|podiums?|engines?|engineers?|"
    r"aero|aerodynamics?|downforce|strateg(?:y|ies)|safety car|laps?|"
    # the vocabulary of the rules notes: "Why was refuelling banned?" has no
    # other word that says Formula 1
    r"cars?|grid|flags?|points?|penalt(?:y|ies)|stewards?|sprints?|practice|"
    r"refuell?ing|fuel|brak(?:e|es|ing)|grip|drag|wings?|halo|cockpit|turbo|hybrid|"
    r"power units?|horsepower|compounds?|slicks?|pirelli|undercut|overcut|"
    r"slipstream(?:s|ing)?|dirty air|ground effect|parc ferm[eé]|vsc|"
    r"(?:budget|cost) cap|steering wheel|chicanes?|paddock|stints?|"
    r"rules?|regulations?)\b",
    re.I,
)
# Case-SENSITIVE on purpose: with re.I, [A-Z][a-z]+ matches any word pair
# ("Yes tell", "am new") and the conversational bypass never fires.
_NAME_CUE = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")


# First-person statements are the user TELLING the bot something ("my favorite
# driver is Fernando Alonso, ok?"), not asking — retrieval here returns doc
# mush for the model to recite as a non-sequitur (measured live: a rain-
# strategy ramble). Persona mode acknowledges instead; chat-recall picks the
# statement up on later turns.
_USER_STATEMENT = re.compile(
    r"\bmy favou?rite\b.*\b(is|are|was|were)\b"   # "my favourite X is/are …"
    r"|\b(is|are)\s+my\s+favou?rite\b"            # "… X is my favourite" (reversed order slipped through live)
    # One intensifier may sit before the verb: "I also love the Monaco GP" and
    # "I really like Lando Norris too" fell through to retrieval and got a race
    # recap / telemetry card back, and the radio log never saw them (measured
    # 2026-10-11). A closed list on purpose — "I would love to hear about the
    # 2021 Abu Dhabi GP" is a request and must keep its facts.
    r"|\bi (?:(?:also|really|just|still|truly|totally|absolutely|definitely) )?"
    r"(am|'m|like|love|support|follow)\b", re.I)
_INTERROGATIVE = re.compile(
    r"\b(who|what|when|where|which|why|how|explain|tell me)\b", re.I)

# Questions with no fact to look up: the bot's own taste ("Who is your
# favourite driver and why?") and a newcomer asking where to begin ("I'm new
# to F1, where do I start?"). Their generic nouns ("driver", "F1") tripped
# the keyword gate, every doc is F1-flavoured enough to pass the relevance
# gate, and the model then recited Jim Clark's bio / RB's 2024 line-up at
# them — while the persona path has trained answers for exactly these
# (measured 2026-10-11).
_PERSONA_ASK = re.compile(
    r"\byour (own )?(favou?rite|opinion|take|pick)\b"
    r"|\b(what|how) do you (think|feel|reckon)\b|\bdo you (think|like|prefer|rate)\b"
    r"|\bin your opinion\b"
    r"|\b(where|how) (do|should|can) i (start|begin|get (into|started))\b", re.I)


def _names_entity(query: str) -> bool:
    """A proper name, or anyone or anything the corpus knows by name."""
    if _NAME_CUE.search(query):
        return True
    entities = _get_entities()
    return bool(entities and entities.names_something(query))


def has_f1_content(query: str) -> bool:
    """Is there anything in ``query`` a fact could be about — a year, a
    Formula 1 term, a name? "What is towing?" has none of these and is still
    worth a search; "What is the capital of France?" has none either. The
    difference between them is made by how well the search matches (see
    :func:`find`), and by what happens when it finds nothing: a question with
    F1 content gets an honest "not in my notes", one without is small talk.
    """
    return bool(_F1_KEYWORDS.search(query)) or _names_entity(query)


def _is_conversational(query: str) -> bool:
    """True for turns that are not worth a search at all.

    Measured 2026-10-11 through the browser, when this was a keyword list and
    nothing else: "Tell me about Hamilton", "Why is Monaco so special?",
    "What is an undercut?" and "Who drives for Ferrari?" were all small talk,
    and the model made each answer up while the fact sat in the corpus. A
    question is now searched whatever its words; only what is plainly not a
    request for facts is kept out.
    """
    q = query.strip()
    # A statement without a real question rides the persona path even when it
    # names an entity ("...is Fernando Alonso, ok?" must not fetch docs).
    if _USER_STATEMENT.search(q) and not _INTERROGATIVE.search(q):
        return True
    # An opinion about a NAMED driver or a dated race still gets its facts.
    if _PERSONA_ASK.search(q) and not (_PIN_YEAR.search(q) or _names_entity(q)):
        return True
    asks = "?" in q or bool(_INTERROGATIVE.search(q))
    return not (asks or has_f1_content(q))


# --- entity pinning: precision, not recall -------------------------------- #
# Measured (retrieval_report_baseline): recall is 100% at k=4, but 28/30
# driver-team contexts also carried WRONG teams from neighbouring docs, and the
# model can't tell which facts belong to the asked entity. When the query names
# a specific entity (full driver name, season year, "<X> Grand Prix"), keep
# only the retrieved docs that mention it. Each filter is applied only if it
# leaves at least one doc, so recall can't drop below the unpinned baseline.
_PIN_NAME = re.compile(r"\b[A-Z][a-zà-ÿ\-]+(?: [A-Z][a-zà-ÿ\-]+)+\b")
_PIN_YEAR = re.compile(r"\b(19[5-9]\d|20\d\d)\b")

# GP-name extraction for pinning and the absence guard. Case-INSENSITIVE on
# purpose: the old TitleCase pattern produced no pin at all for "the 2021 Yas
# marina Grand Prix", so an unrelated doc reached the prompt and the model
# recited it as a race recap (measured live).
_GP_WORD = re.compile(r"[\w'à-ÿ\-]+")
_GP_STOP = {"the", "a", "an", "this", "that", "about", "of", "at", "in", "on",
            "for", "was", "is", "won", "me", "tell", "to", "what", "which",
            "who", "how", "why", "when", "and", "or", "last", "latest", "next",
            "recent", "first", "final", "previous", "every", "any", "each"}


def _extract_gp(query: str) -> str | None:
    """Return the "{Name} Grand Prix" phrase from the query, or None.

    Walks backwards from "grand prix" collecting up to three name words,
    stopping at stopwords and bare years — so "who won the 2021 abu dhabi
    grand prix?" yields "abu dhabi Grand Prix" while "what makes a grand prix
    special?" yields None (no name words survive the trim).
    """
    words = _GP_WORD.findall(query)
    lows = [w.lower() for w in words]
    for i in range(len(lows)):
        # "GP" is the same race as "Grand Prix" ("2007 Singapore GP"); without
        # this the never-held guard silently skipped abbreviated queries.
        is_gp = lows[i] == "gp" or (
            lows[i] == "grand" and i + 1 < len(lows) and lows[i + 1] == "prix")
        if is_gp:
            picked: list[str] = []
            for j in range(i - 1, max(-1, i - 4), -1):
                w = words[j]
                if w.lower() in _GP_STOP or _PIN_YEAR.fullmatch(w):
                    break
                picked.append(w)
            if picked:
                picked.reverse()
                return " ".join(picked + ["Grand", "Prix"])
            return None
    return None


def _display_gp(gp_phrase: str) -> str:
    """Capitalize a GP name typed in lowercase ("abu dhabi Grand Prix") for
    user-facing text. Words that already carry capitals ("USA") stay as typed."""
    def cap(part: str) -> str:
        return part[:1].upper() + part[1:] if part.islower() else part
    return " ".join("-".join(cap(p) for p in w.split("-"))
                    for w in gp_phrase.split())


def _fold(s: str) -> str:
    """Accent-insensitive lowercase ("Pérez" and "Perez" must both pin)."""
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


def _gp_matcher(gp_phrase: str):
    """Match a GP name tolerantly: users adjectivize race names ("Singaporean
    Grand Prix" for the Singapore GP), which made the absence guard decline a
    race it demonstrably had. Falls back from the exact phrase to prefix stems
    of the distinctive words. Stem length adapts to the word (len-2, clamped
    to 5..7): a fixed 5 made "Austrian" ⊂ "AUSTRALIAN" (measured: the 1990
    Austrian GP query recovered the 1990 Australian doc), while 5 chars must
    survive for "Mexican" → "Mexico City" and 7 for "Singaporean" → the
    9-letter "Singapore"."""
    full = _fold(gp_phrase)
    checks = []          # one predicate per name word
    for w in gp_phrase.split():
        if w.lower() in ("grand", "prix"):
            continue
        if len(w) >= 5:
            stem = _fold(w)[:max(5, min(7, len(w) - 2))]
            checks.append(lambda t, s=stem: s in t)
        elif len(w) >= 3:
            # Short venue names ("Spa", "Yas") produced NO stems, so the
            # guard declared the 2018 Spa GP never-held (measured). They
            # match as whole words — a prefix would make "Spa" hit "SPAnish".
            pat = re.compile(r"\b" + re.escape(_fold(w)) + r"\b")
            checks.append(lambda t, p=pat: bool(p.search(t)))

    def score(text_low: str) -> int:
        """How many of the name words the text matches. Ambiguous aliases
        need this: "Yas Marina" matches the Singapore doc via "MARINA Bay"
        (measured: the 2022 Yas Marina GP answered with the Singapore
        result) — the doc matching BOTH words must win."""
        if full in text_low:
            return len(checks) + 1
        return sum(c(text_low) for c in checks)

    def matches(text_low: str) -> bool:
        return score(text_low) > 0

    matches.score = score
    return matches


def _pin(query: str, results: list[dict]) -> list[dict]:
    def keep(docs: list[dict], pred) -> list[dict]:
        kept = [d for d in docs if pred(_fold(d.get("text", "")))]
        return kept or docs
    for year in _PIN_YEAR.findall(query):
        results = keep(results, lambda t, y=year: re.search(r"\b" + y + r"\b", t))
    gp = _extract_gp(query)
    if gp:
        matcher = _gp_matcher(gp)
        results = keep(results, matcher)
        # Among matching docs, keep only the best word-coverage tier so a
        # one-word alias overlap can't outrank the actual venue.
        best = max((matcher.score(_fold(d.get("text", ""))) for d in results),
                   default=0)
        if best > 0:
            results = [d for d in results
                       if matcher.score(_fold(d.get("text", ""))) == best]
    names = [_fold(n) for n in _PIN_NAME.findall(query)]
    if names:
        results = keep(results, lambda t: any(n in t for n in names))
    return results


# A curated doc that states HOW MANY titles its driver won ("...won seven
# Formula 1 World Drivers' Championships", "won the 2016 World Championship").
# Verstappen's "multiple-time World Champion" does not count — it would be a
# worse answer than his season docs.
_TITLES_COUNTED = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|\d+)\b[^.;]*\b(?:championships?|titles?)\b"
    r"|\bwon the (?:19|20)\d\d world championship\b", re.I)


def _titles_summary(query: str, docs: list[dict]) -> dict | None:
    """The curated doc that opens with a driver named in ``query`` and sums up
    their titles, if there is one.

    For a "when" question the summary must also DATE the titles — name the
    first and the last title season on record. Otherwise the season docs
    answer better: Hamilton's summary says "seven" but not which years, and
    the only year in Lauda's is his 1976 crash, which was not a title year.
    """
    names = tuple(_fold(n) for n in _PIN_NAME.findall(query))
    if not names:
        return None
    wants_years = re.search(r"\b(when|(which|what) years?)\b", query, re.I)
    for doc in docs:
        if doc.get("source") != "curated":
            continue
        text = _fold(doc.get("text", ""))
        if not (text.startswith(names) and _TITLES_COUNTED.search(text)):
            continue
        if wants_years:
            name = next(n for n in names if text.startswith(n))
            seasons = sorted(
                m.group(1)
                for m in (re.match(r"in (\d{4}),", _fold(d.get("text", "")))
                          for d in docs
                          if d.get("source", "").endswith("_champion")
                          and name in _fold(d.get("text", "")))
                if m)
            if not seasons or seasons[0] not in text or seasons[-1] not in text:
                return None
        return doc
    return None


# --- the one fact a plainly put question needs ----------------------------- #
_LINEUP_CUE = re.compile(
    r"\b(driv|rac)(e|es|ing)\b.*\bfor\b|\b(drove|raced)\b.*\bfor\b|\bdrivers?\b|"
    r"\bline-?up\b|\bteam-?mates?\b|\bwho(?:'s| is| are)? (?:at|in)\b", re.I)
_CURRENT_TEAM_CUE = re.compile(
    r"\b(?:which|what) team\b|\bwho does\b.*\b(?:drive|race) for\b|"
    r"\bwhere (?:does|is)\b.*\b(?:driv|rac)(?:e|ing)\b|\bcar number\b|"
    r"\b(?:driv|rac)(?:e|es|ing) for (?:now|currently|today|this (?:year|season))\b", re.I)
_COMPARE_CUE = re.compile(
    r"\b(?:better|best|greater|greatest|faster|fastest|quicker|more|most|"
    r"vs\.?|versus|compared?|or)\b", re.I)
_WINNER_CUE = re.compile(
    r"\b(?:won|win|wins|winner|victory|pole|podium|result|results|happened)\b", re.I)
# "Who won the Monaco Grand Prix the most times?" is not about one race.
_MANY_RACES_CUE = re.compile(r"\b(?:most|how many|ever|record|history|times|first)\b", re.I)

# Which end of the records a question asks for: "the first ever F1 race", "the
# last Grand Prix of 2021", "the first Monaco Grand Prix", "the season finale".
# What may stand between the ordinal and its noun is a closed list, or a Grand
# Prix's name — "the last lap of the race" and "the first Japanese driver to
# start a race" ask for something else.
_OF_F1 = (r"(?:(?:ever|official|f1|formula (?:1|one)|world championship|championship|career)"
          r"[\s-]+)*")
_GP_NAME = (r"(?:(?!(?:to|of|the|in|at|a|an|on|for|and|or|is|was|lap|corner|"
            r"years?|year's|seasons?|season's)\b)[\w'’à-ÿ.]+[\s-]+){1,3}")
_TO_WIN = r"\s+(?:driver|person|man|one)\s+to\s+win\b"


def _end_cue(ordinal: str, also: str) -> re.Pattern:
    return re.compile(
        rf"\b(?:{ordinal})[\s-]+{_OF_F1}(?:race|round|win|victory|grand prix|gp)\b"
        rf"|\b(?:{ordinal})[\s-]+{_GP_NAME}(?:grand prix|gp)\b"
        rf"|\b(?:{ordinal}){_TO_WIN}|{also}", re.I)


_FIRST_RACE_CUE = _end_cue("first|opening|inaugural|maiden", r"\b(?:season[\s-])?opener\b")
_LATEST_RACE_CUE = _end_cue("last|latest|final|most recent|previous",
                            r"\b(?:season[\s-])?finale\b")
# ...of one driver's or one team's wins ("Senna's last win", "When did Hamilton
# win his first race?", "the first driver to win for Red Bull") — as against
# the race itself ("Did Hamilton win the last race?").
_THEIR_OWN = re.compile(
    r"\b(?:first|maiden|last|latest|final|most recent)[\s-]+(?:[\w'’]+[\s-]+){0,2}?"
    r"(?:win|victory)\b"
    r"|(?:\b(?:his|her|their|its)|['’]s)\s+(?:very\s+)?"
    r"(?:first|maiden|last|latest|final|most recent)\b"
    rf"|\b(?:first|last){_TO_WIN}", re.I)
_WIN_WORD = re.compile(r"\b(?:win|wins|won|winning|victory|victories)\b", re.I)
_SESSION =r"(?:race|qualifying|quali|sprint|practice|session|grand prix|gp|season)"
_SOON = r"(?:today|tonight|tomorrow|this weekend)"
_SCHEDULE_CUE = re.compile(
    r"\bnext\s+(?:(?:f1|formula (?:1|one))\s+)?(?:race|grand prix|gp|round|season|weekend)\b"
    r"|\b(?:upcoming|calendar|schedule|fixtures?)\b|\bwhat time\b"
    rf"|\bwhen (?:is|does|are|do)\b.*\b{_SESSION}\b"
    rf"|\b{_SESSION}\b.*\b{_SOON}\b|\b{_SOON}\b.*\b{_SESSION}\b", re.I)
# How a championship stands, as against who won one. Typed into the app in
# October 2026: "Who is leading the championship?" -> "And that's the title!
# Max Verstappen is the 2024 World Champion", "How many points does Hamilton
# have?" -> his career card.
_STANDINGS_CUE = re.compile(
    r"\bstandings?\b|\bmost points\b|\bpoints (?:table|leader|lead)\b"
    r"|\btitle (?:race|fight|battle)\b"
    r"|\bchampionship (?:leader|lead|table|order|fight|battle|race|picture)\b"
    r"|\b(?:leading|leads?|top of|ahead in|winning)\b.*\b(?:championship|title|table|points)\b"
    r"|\bwho(?:'s| is) (?:leading|on top|in the lead)\b(?!.*\b(?:race|grand prix|gp|lap)\b)",
    re.I)
# ...asked of one driver. His points this season, that is: nothing here adds
# up a career's.
_POINTS_CUE = re.compile(
    r"^(?!.*\b(?:career|ever|all[- ]time|total|history)\b).*\bpoints?\b", re.I | re.S)
# Asked in the present tense, a title question is about the latest champion on
# record — not the season the chat happened to be on ("Who is the current
# world champion?" right after a 2024 answer).
_NOW_CUE = re.compile(
    r"\b(?:current|currently|reigning|defending|latest|last|most recent|now|today|"
    r"at the moment|leading|leads?|standings?)\b"
    r"|\bwho(?:'s| is| are) the\b(?:\s+[\w'’]+){0,3}?\s+champions?\b", re.I)

# Other championships share every keyword with this one ("Who won the 2022
# football World Cup?" has a year and a winner) and none of the facts.
_OTHER_SPORT = re.compile(
    r"\b(?:football|soccer|fifa|world cup|cricket|tennis|basketball|nba|nfl|baseball|"
    r"golf|olympics?|motogp|moto gp|nascar|indycar|wec|wrc|rally|formula e|"
    r"formula (?:2|3|two|three)|f2|f3)\b", re.I)

_STRONG = 0.50      # FactRetriever.retrieve's own "similar enough on its own" bar


def _latest_race(entities: EntityIndex, gp: str | None = None,
                 circuit: str | None = None) -> dict | None:
    """The most recent race on record: of one Grand Prix, at one circuit, or
    of them all."""
    if gp:
        match = _gp_matcher(gp)

        def score(doc: dict) -> int:
            return match.score(_fold(doc["text"]))

        best = max((score(doc) for _, doc in entities.races), default=0)
        return entities.latest_race(lambda race, doc: score(doc) == best) if best else None
    if circuit:
        return entities.latest_race(lambda race, doc: race.venue.startswith(circuit + " in "))
    return entities.latest_race()


def _end_of_the_records(question: str, entities: EntityIndex, year: str | None,
                        gp: str | None, circuit: str | None, team: str | None,
                        drivers: list) -> dict | None:
    """The first or the last race on record that ``question`` asks for, or
    None if it asks for neither.

    Typed into the app (2026-10-11), "Who won the first ever F1 race?" was
    searched as it stood: four champion docs came back, and a decline. Which
    race opens or closes the records is not a matter of similarity. It can be
    narrowed to a season ("the last race of 2021"), a Grand Prix ("the first
    Monaco Grand Prix"), a circuit, or the wins of one driver or one team
    ("Senna's last win").
    """
    first = _FIRST_RACE_CUE.search(question)
    if not (first or _LATEST_RACE_CUE.search(question)) or len(drivers) > 1:
        return None
    wanted = []                      # a race counts if every one of these holds
    if year:
        wanted.append(lambda race, doc: race.year == year)
    if gp:
        match = _gp_matcher(gp)
        best = max((match.score(_fold(doc["text"])) for _, doc in entities.races), default=0)
        if best:                     # "last weekend's Grand Prix" names none
            wanted.append(lambda race, doc: match.score(_fold(doc["text"])) == best)
    elif circuit:
        wanted.append(lambda race, doc: race.venue.startswith(circuit + " in "))
    if (drivers or team) and _THEIR_OWN.search(question):
        # Their first WIN can be read off the records. Their first start
        # cannot: nothing here says who entered a race.
        if not _WIN_WORD.search(question):
            return None
        if drivers:
            wanted.append(lambda race, doc: race.winner == drivers[0].name)
        if team:
            wanted.append(lambda race, doc: team_root(race.team) == team_root(team))
    pick = entities.first_race if first else entities.latest_race
    return pick(lambda race, doc: all(holds(race, doc) for holds in wanted))


def _about(driver, question: str, entities: EntityIndex, retriever) -> dict:
    """What the corpus says about one driver that is nearest the question:
    the curated note ("...died in a crash at Imola in 1994") for "What
    happened to Senna?", the bio ("raced from 2001 to 2026") for "Is Alonso
    still racing?"."""
    note = entities.curated_about(driver.name)
    closest = getattr(retriever, "closest", None)
    # Asked for an opinion ("Do you think Hamilton is the greatest?"), what
    # the notes say of him is a better answer than his card.
    if note and closest and not _PERSONA_ASK.search(question):
        return closest(question, [note, driver.doc])
    return note or driver.doc


def _pick(question: str, intent: str | None, entities: EntityIndex,
          retriever: FactRetriever) -> tuple[str, list[dict]] | None:
    """``(what is asked, the documents that answer it)`` when the question
    names its subject outright, else None and the search decides.

    Every case here was measured 2026-10-11 as a search that returned the
    right document among others, and an answer that mixed them: "Who drives
    for Ferrari?" (four per-driver docs, one driver recited), "How many wins
    does Max Verstappen have?" (bio plus three season docs: "63 — Max
    Verstappen takes victory at this chequered flag"), "Is Hamilton better
    than Verstappen?" (a 2019 race recap).

    An empty list says the question is understood and the corpus has no
    document for it ("Who drove for Ferrari in 1999?"): better no facts than
    whichever 1999 driver the search likes best.
    """
    year_m = _PIN_YEAR.search(question)
    year = year_m.group(0) if year_m else None
    drivers = entities.drivers_in(question)
    team, circuit = entities.team_in(question), entities.circuit_in(question)
    # Titles, tallies and team histories are never about one race, whatever
    # else the wording holds ("career Grand Prix win tally").
    about_races = intent not in ("constructors", "titles", "tally", "teams")
    gp = _extract_gp(question) if about_races else None

    # No live data: say so rather than search for something that sounds close.
    if _SCHEDULE_CUE.search(question) and not year:
        return "schedule", []
    # How the championship stands. Only the season being run has a document
    # for that; between seasons the question falls through to the champion.
    if len(drivers) <= 1:
        if drivers and (_STANDINGS_CUE.search(question) or _POINTS_CUE.search(question)):
            doc = entities.standing(drivers[0].name)
        elif not drivers and _STANDINGS_CUE.search(question):
            doc = entities.leader(
                constructors=bool(re.search(r"\bconstructors?\b|\bteams?\b", question, re.I)))
        else:
            doc = None
        # ("this season" reaches here written out as that season's year)
        if doc and year in (None, parse_standing(doc["text"]).year):
            return "standings", [doc]
    # A count or a record: no document holds it, the records do.
    counted = computed(question, entities)
    if counted:
        return "stat", [{"text": counted, "source": "computed"}]

    # --- one race, however it is named ---
    if about_races:
        doc = _end_of_the_records(question, entities, year, gp, circuit, team, drivers)
        if doc:
            return "race", [doc]
        one_race = bool(_WINNER_CUE.search(question)) and not _MANY_RACES_CUE.search(question)
        if gp and year:
            return None                  # a dated Grand Prix: pins and guard in find()
        if circuit and year:
            doc = entities.race_at(circuit, year)
            if doc:
                return "race", [doc]
        if year and not gp and not circuit:
            # "Who won in Brazil in 2008?" names the race by where it was. The
            # search found it too, along with that year's constructors'
            # champion and another Grand Prix (measured 2026-10-11).
            held = entities.races_held(question, year)
            if len(held) == 1:
                return "race", held
        if (gp or circuit) and not year and not drivers and one_race:
            # "Who won the Monaco Grand Prix?" — the latest one, which names its year.
            doc = _latest_race(entities, gp, None if gp else circuit)
            if doc:
                return "race", [doc]
        if gp:
            if drivers or _MANY_RACES_CUE.search(question):
                return None
            # "Tell me about the Monaco Grand Prix": its note, else its latest running.
            note = entities.curated_about(gp.rsplit(" Grand Prix", 1)[0])
            doc = note or _latest_race(entities, gp)
            return ("circuit" if note else "race", [doc]) if doc else None

    # --- a team's drivers ("Who raced for Ferrari in 2024?" reads as a team
    #     history only until you notice it names no driver) ---
    if (team and not drivers and intent not in ("constructors", "titles", "tally")
            and _LINEUP_CUE.search(question)):
        doc = entities.lineup(team, year)
        return "lineup", [doc] if doc else []
    # --- two drivers ---
    if len(drivers) >= 2 and not year:
        if _COMPARE_CUE.search(question):
            # side by side: both careers, nothing else
            return "compare", [drivers[0].doc, drivers[1].doc]
        if intent == "driver":
            return "drivers", [d.doc for d in drivers[:2]]
        if intent is None:
            return "drivers", [_about(d, question, entities, retriever) for d in drivers[:2]]
    # --- one driver ---
    if len(drivers) == 1:
        driver = drivers[0]
        if _CURRENT_TEAM_CUE.search(question):
            doc = entities.team_entry(driver.name, year)
            return ("team_entry", [doc]) if doc else ("driver", [driver.doc])
        if year or circuit:
            # "What's his win tally at Monza?": the bio holds the career tally
            # and nothing narrower (and stats.py only counts what is asked as
            # "how many"). Reciting 106 would answer another question.
            return ("tally", []) if intent == "tally" else None
        if intent in ("tally", "teams", "driver"):
            return intent, [driver.doc]
        if intent == "titles":
            # A note that sums the titles up reads better than the bio's list
            # of seasons; the bio is complete for everyone else.
            return "driver_titles", [_titles_summary(question, retriever.docs) or driver.doc]
        if intent is None:
            return "driver", [_about(driver, question, entities, retriever)]
        return None
    if drivers or year:
        return None
    # --- a circuit or a team asked about by name ("Tell me about Spa",
    #     or just "And Spa?") ---
    if circuit and (intent in ("circuit", "driver")
                    or _nothing_but(question, entities.circuit_at(question))):
        note = entities.curated_about(circuit)
        where = re.search(r"\bwhere\b", question, re.I)
        doc = entities.circuit_doc(circuit) if (where or not note) else note
        if doc:
            return "circuit", [doc]
    if team and not circuit and (intent == "driver"
                                 or _nothing_but(question, entities.team_at(question))):
        doc = entities.curated_about(team) or entities.lineup(team)
        if doc:
            return "team", [doc]
    return None


_LEAD_IN = re.compile(r"(?:and|so|ok|okay|then|what about|how about|about|the)\b", re.I)


def _nothing_but(question: str, span: tuple[int, int, str]) -> bool:
    """Is the name at ``span`` all there is to ``question`` ("And Spa?")?"""
    rest = question[:span[0]] + " " + question[span[1]:]
    return not re.sub(r"[\W_]+", "", _LEAD_IN.sub(" ", rest))


def _focus(question: str, results: list[dict], entities: EntityIndex) -> list[dict]:
    """Cut a search result down to its top document when that one is plainly
    the answer and the rest came along for a shared word.

    Measured 2026-10-11: "How do points work in F1?" found the points note
    first, then the steering wheel, the cost cap and the Miami circuit — and
    the model, handed all four, wrote about tyre management. One fact is
    recited; several are blended.

    * The note on the very thing the question names ("Why is Monaco so
      special?" -> the Monaco note, not the circuit's address and two 1950s
      race recaps).
    * A curated note that both rankers put first: nearest in meaning, and
      sharing the question's words about as much as anything else found.
      "How fast do F1 cars go?" fails that — the nearest note there is about
      race distance — and keeps everything.
    """
    if len(results) < 2 or _PIN_YEAR.search(question) or entities.drivers_in(question):
        return results
    top = results[0]
    for name in (entities.circuit_in(question), entities.team_in(question)):
        note = entities.curated_about(name) if name else None
        if note and top["text"] == note["text"]:
            return results[:1]
    if top.get("source") == "curated" and "dense" in top:
        nearest = max(d.get("dense", 0.0) for d in results)
        wordiest = max(d.get("bm25", 0.0) for d in results)
        if top["dense"] == nearest >= 0.45 and top["bm25"] >= 0.6 * wordiest:
            return results[:1]
    return results


def _guard(text: str, reply: str, **extra) -> list[dict]:
    """The one "fact" a guard returns: the records hold no such thing. It
    carries the finished reply — the shipped adapter predates the guards and
    improvises on such a line (measured: "that one's not in my notes ... What
    I've got here covers the 2007 Singapore Grand Prix, nothing else")."""
    return [{"text": text, "source": "absence_guard", "reply": reply, **extra}]


_SEASON_TITLES = {          # intent -> (how to read its docs, what the title is called)
    "titles": (parse_champion, "World Champion"),
    "constructors": (parse_constructors, "Constructors' Champion"),
}


def _season_title(question: str, intent: str, results: list[dict],
                  retriever: FactRetriever, latest: int) -> tuple[str, list[dict]] | None:
    """The single champion doc of the season a title question names (or, asked
    in the present tense, of the latest season that has one), or the verdict
    that the records hold none.

    Pinning keeps the search results that mention the year and, when none
    does, all of them: "Who won the 2026 championship?" in a season still
    running came back with other seasons' champions for the model to pass
    off. Same remedy as for races: check the whole corpus, then say so.
    """
    year_m = _PIN_YEAR.search(question)
    if intent not in _SEASON_TITLES:
        return None
    parse, what = _SEASON_TITLES[intent]
    if not year_m:
        # "Who is the current world champion?": the latest one on record. The
        # search brought four champions and the model picked 2024's.
        if not _NOW_CUE.search(question):
            return None
        titles = [(title.year, d) for d in retriever.docs
                  if (title := parse(d.get("text", "")))]
        return (intent, [max(titles, key=lambda t: t[0])[1]]) if titles else None
    year = year_m.group(0)

    def of_the_year(docs: list[dict]) -> list[dict]:
        return [d for d in docs
                if (title := parse(d.get("text", ""))) and title.year == year]

    docs = of_the_year(results) or of_the_year(retriever.docs)
    if docs:
        return intent, docs[:1]
    pending = int(year) >= latest
    return "absence", _guard(
        f"There is no {year} {what} in the records.",
        no_record_reply(year, what, pending=pending))


@dataclass
class Found:
    """What retrieval made of one question."""
    question: str                     # canonical form: drivers by full name
    facts: list[dict] = field(default_factory=list)
    intent: str | None = None         # what is asked ("race", "tally", "lineup", ...)


def find(query: str, k: int = 4) -> Found:
    """The facts ``query`` needs, and what kind of question it is.

    Conversational turns ("Yes tell me.", "thanks!") bypass retrieval entirely —
    a grounding-trained model will otherwise faithfully recite whatever
    irrelevant doc a contentless query happens to match.
    """
    retriever = _get_retriever()
    entities = _get_entities()
    question = entities.canonicalize(query) if entities else query
    found = Found(question, intent=_intent(question))
    if _is_conversational(question) or not retriever:
        return found
    if _OTHER_SPORT.search(question):
        found.intent = "off_topic"
        return found

    picked = _pick(question, found.intent, entities, retriever) if entities else None
    if picked:
        found.intent, found.facts = picked
        return found

    sources = _route(question)
    # A year-less title question about a named driver ("How many titles did
    # Lewis Hamilton win?", "when did he win his championship") asks for ONE
    # fact, but the champion docs hold it a season per doc and only k of them
    # reach the prompt — four of Hamilton's seven. Measured 2026-10-11: "the
    # 2017, 2019 and 2017 World Champion", and for Vettel a ramble naming
    # neither driver nor year. Where a curated doc sums the titles up, that
    # single fact is the context. (Benchmark title questions all carry a year.)
    if sources and "kaggle_champion" in sources and not _PIN_YEAR.search(question):
        summary = _titles_summary(question, retriever.docs)
        if summary:
            found.intent, found.facts = "driver_titles", [summary]
            return found
    # A question with nothing recognisably F1 in it ("What is towing?",
    # "What is the capital of France?") is searched too, but a document only
    # counts if it is strongly about the same thing: on token overlap alone
    # the second one comes back with four French circuits.
    strict = {} if has_f1_content(question) else {"min_score": _STRONG}
    # Who-races-for-whom documents are for questions that name the driver or
    # the team (and _pick has answered those). Anywhere else they only share
    # a word: "What tyres are used in F1?" found the "RB F1 Team" line-up
    # ahead of the Pirelli note.
    named = bool(entities and (entities.drivers_in(question) or entities.team_in(question)))

    def search(**where) -> list[dict]:
        if named:
            return retriever.retrieve(question, k=k, **where, **strict)
        found_ = retriever.retrieve(question, k=3 * k, **where, **strict)
        return [d for d in found_
                if not (parse_entry(d["text"]) or parse_lineup(d["text"])
                        or parse_standing(d["text"]))][:k]

    results = search(sources=sources)
    if not results and sources:
        results = search()                   # routed miss -> open fallback
    results = _pin(question, results)
    latest = max((int(race.year) for race, _ in entities.races), default=0) if entities else 0

    # Nonexistent-race guard. The corpus holds EVERY championship round
    # 1950→today, so a year+GP query with no doc matching both means the race
    # wasn't held ("2007 Singapore GP" — Singapore joined in 2008). Without
    # this, pinning falls back to the same GP in OTHER years and the model
    # faithfully recites a neighbouring race under the asked year — measured
    # live as a fabricated 2007 Singapore result built from the 2010 doc.
    year_m = _PIN_YEAR.search(question)
    gp = _extract_gp(question)
    if year_m and gp and found.intent not in _SEASON_TITLES:
        year = year_m.group(0)
        gp_match = _gp_matcher(gp)   # stem-tolerant: "Singaporean" ≈ "Singapore"

        def _covers(d: dict) -> bool:
            t = _fold(d.get("text", ""))
            return bool(re.search(r"\b" + year + r"\b", t)) and gp_match(t)

        if not any(_covers(d) for d in results):
            # Verify against the WHOLE corpus before declaring absence: a
            # routing or ranking miss must not masquerade as history. Measured
            # live: "tell me about the 2021 Abu Dhabi GP" was routed to driver
            # docs and then declared never-held. If the corpus does hold the
            # race, recover its docs (race results first) instead of lying.
            recovered = [d for d in retriever.docs if _covers(d)]
            if recovered:
                recovered.sort(key=lambda d: (
                    -gp_match.score(_fold(d.get("text", ""))),
                    0 if d.get("source", "").endswith("_race") else 1))
                best = gp_match.score(_fold(recovered[0].get("text", "")))
                results = [d for d in recovered
                           if gp_match.score(_fold(d.get("text", ""))) == best][:k]
            else:
                # The verdict is the answer: year/gp ride along for callers
                # that render it themselves. A season the records have not
                # reached is "not yet", not "never held".
                name = _display_gp(gp)
                found.intent = "absence"
                if latest and int(year) >= latest:
                    found.facts = _guard(
                        f"There is no {year} {name} in the race records yet.",
                        no_record_reply(year, name, pending=True), year=year, gp=name)
                else:
                    found.facts = _guard(
                        f"There is no {year} {name} in the historical race "
                        f"records — that Grand Prix was not held in {year}.",
                        absence_reply(year, name), year=year, gp=name)
                return found
        # A dated Grand Prix question is answered by that race's own doc: a
        # champion or season doc that merely shares the year is noise.
        if found.intent == "race":
            races = [d for d in results if parse_race(d.get("text", "")) and _covers(d)]
            results = races[:1] or results

    title = _season_title(question, found.intent, results, retriever, latest)
    if title:
        found.intent, found.facts = title
        return found
    found.facts = _focus(question, results, entities) if entities else results
    return found


def retrieve_facts(query: str, k: int = 4) -> list[dict]:
    """Return the relevant fact documents for ``query`` (empty if no index)."""
    return find(query, k=k).facts


def format_context(facts: list[dict]) -> str:
    """Render fact docs as the compact text block injected into the prompt
    (empty string for no facts, so the caller can simply skip grounding)."""
    if not facts:
        return ""
    lines = [f"- {f['text']}" for f in facts]
    return (
        "Verified F1 facts (rely on these for any names, numbers, teams or results; "
        "don't contradict them or invent specifics):\n" + "\n".join(lines)
    )


def retrieve_context(query: str, k: int = 4) -> str:
    """Return retrieved facts as a compact text block to inject into the prompt.

    Empty string when nothing relevant is found, so the caller can simply skip
    grounding for off-topic or already-known questions.
    """
    return format_context(retrieve_facts(query, k=k))
