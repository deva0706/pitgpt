"""
retriever.py — a lazy, shared retriever and prompt-grounding helper.

The app and inference engine call ``retrieve_context(query)`` to get a short block
of relevant facts to hand the model. The index is loaded once (lazily) and reused.
If no index has been built, retrieval degrades gracefully to "no context" so the
app still runs (just without grounding).
"""

from __future__ import annotations

import logging

from pitgpt.rag.index import FactRetriever

logger = logging.getLogger("pitgpt.rag.retriever")

_retriever: FactRetriever | None = None
_load_failed = False


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


import re

# Query-aware source routing. Popular entities have dozens of race-winner docs
# that crowd out their single bio doc; routing the query to the right doc
# family keeps retrieval precise. Order matters: championship phrasing wins
# over race phrasing ("who won the 2016 championship" is not a race question).
_ROUTES: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"constructor", re.I), ["kaggle_constructors", "curated"]),
    (re.compile(r"champion|title", re.I),
     ["kaggle_champion", "jolpica_champion", "curated"]),
    # Win-tally questions get the bio docs ONLY: the curated notes discuss
    # titles in prose ("seven championships...") and outranked the bio's exact
    # win count in the prompt — measured live as "Seven" for Hamilton's 105.
    (re.compile(r"how many.*(win|won|race)|win tally|wins (does|did|has)|"
                r"grand prix wins", re.I),
     ["kaggle_driver", "jolpica_driver"]),
    # Team-HISTORY questions get bio docs only too: the curated star prose
    # ("...has driven for McLaren, Mercedes and, from 2025, Ferrari") outranked
    # Hamilton's bio doc and the card format drifted into rambling — measured
    # in the diag_jul06 benchmark (the one true driver_teams miss).
    (re.compile(r"\bteams? (did|has)\b|\braced for\b|\bdrove for\b", re.I),
     ["kaggle_driver", "jolpica_driver"]),
    # A year alongside "Grand Prix"/"GP" is race intent regardless of verb.
    # "Tell me about the 2021 Abu Dhabi GP" previously fell through to the
    # DRIVER route below ("tell me about"), which starved the absence guard of
    # race docs — it then declared real races "not held" (measured live).
    (re.compile(r"^(?=.*\b(?:19[5-9]\d|20[0-2]\d)\b)(?=.*\b(?:grand prix|gp)\b)",
                re.I | re.S),
     ["kaggle_race", "jolpica_race"]),
    # Circuit questions must reach circuit docs even when phrased "tell me
    # about {circuit}" — previously hijacked by the driver route below.
    (re.compile(r"circuit|track\b|where is", re.I), ["kaggle_circuit", "curated"]),
    # Driver-centric phrasings outrank the race route: "career Grand Prix win
    # tally" is a question about a DRIVER even though it contains "Grand Prix".
    # who(?:'s| is| was): the old class [' i] allowed only ONE char between
    # "who" and "s", so "Who is Fernando Alonso?" never routed at all.
    (re.compile(r"who(?:'s| is| was)\b|tell me about|rundown on|teams? did|drive for|"
                r"race for|career|win tally|how many", re.I),
     ["kaggle_driver", "jolpica_driver", "curated", "openf1"]),
    # A yearless race question still needs a winner-verb, not just "grand prix".
    (re.compile(r"(who won|winner|victory|took .*(flag|win)|crossed the line)"
                r".*(grand prix|\bgp\b)", re.I),
     ["kaggle_race", "jolpica_race"]),
]


def _route(query: str) -> list[str] | None:
    for pattern, sources in _ROUTES:
        if pattern.search(query):
            return sources
    return None


# F1-content cue: a year, a proper-noun-ish capitalized word mid-sentence, or a
# domain keyword. Conversational turns ("Yes tell me.", "thanks!", "haha ok")
# have none of these — retrieval must stay SILENT there, or the grounding-trained
# model will faithfully recite whatever irrelevant doc comes back.
_F1_KEYWORDS = re.compile(
    r"\b(19[5-9]\d|20[0-2]\d)\b"                       # a season year
    r"|driver|team|race|grand prix|\bgp\b|champion|title|circuit|track|tyre|"
    r"tire|pit\b|drs|ers|qualifying|pole|overtak|constructor|f1|formula|season|"
    r"\bwins?\b|podium|engine|aero|downforce|strategy|safety car|\blaps?\b",
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
    r"|\bi (am|'m|like|love|support|follow)\b", re.I)
_INTERROGATIVE = re.compile(
    r"\b(who|what|when|where|which|why|how|explain|tell me)\b", re.I)


def _is_conversational(query: str) -> bool:
    """True for turns that carry no F1 content worth retrieving for."""
    q = query.strip()
    # A statement without a real question rides the persona path even when it
    # names an entity ("...is Fernando Alonso, ok?" must not fetch docs).
    if _USER_STATEMENT.search(q) and not _INTERROGATIVE.search(q):
        return True
    return not (_F1_KEYWORDS.search(q) or _NAME_CUE.search(q))


# --- entity pinning: precision, not recall -------------------------------- #
# Measured (retrieval_report_baseline): recall is 100% at k=4, but 28/30
# driver-team contexts also carried WRONG teams from neighbouring docs, and the
# model can't tell which facts belong to the asked entity. When the query names
# a specific entity (full driver name, season year, "<X> Grand Prix"), keep
# only the retrieved docs that mention it. Each filter is applied only if it
# leaves at least one doc, so recall can't drop below the unpinned baseline.
_PIN_NAME = re.compile(r"\b[A-Z][a-zà-ÿ\-]+(?: [A-Z][a-zà-ÿ\-]+)+\b")
_PIN_YEAR = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")

# GP-name extraction for pinning and the absence guard. Case-INSENSITIVE on
# purpose: the old TitleCase pattern produced no pin at all for "the 2021 Yas
# marina Grand Prix", so an unrelated doc reached the prompt and the model
# recited it as a race recap (measured live).
_GP_WORD = re.compile(r"[\w'à-ÿ\-]+")
_GP_STOP = {"the", "a", "an", "this", "that", "about", "of", "at", "in", "on",
            "for", "was", "is", "won", "me", "tell", "to", "what", "which",
            "who", "how", "why", "when", "and", "or"}


def _extract_gp(query: str) -> str | None:
    """Return the "{Name} Grand Prix" phrase from the query, or None.

    Walks backwards from "grand prix" collecting up to three name words,
    stopping at stopwords and bare years — so "who won the 2021 abu dhabi
    grand prix?" yields "abu dhabi Grand Prix" while "what makes a grand prix
    special?" yields None (no name words survive the trim).
    """
    words = _GP_WORD.findall(query)
    lows = [w.lower() for w in words]
    for i in range(len(lows) - 1):
        if lows[i] == "grand" and lows[i + 1] == "prix":
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


def _fold(s: str) -> str:
    """Accent-insensitive lowercase ("Pérez" and "Perez" must both pin)."""
    import unicodedata
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
    stems = [_fold(w)[:max(5, min(7, len(w) - 2))] for w in gp_phrase.split()
             if len(w) >= 5 and w.lower() not in ("grand", "prix")]

    def matches(text_low: str) -> bool:
        return full in text_low or any(s in text_low for s in stems)
    return matches


def _pin(query: str, results: list[dict]) -> list[dict]:
    def keep(docs: list[dict], pred) -> list[dict]:
        kept = [d for d in docs if pred(_fold(d.get("text", "")))]
        return kept or docs
    for year in _PIN_YEAR.findall(query):
        results = keep(results, lambda t, y=year: re.search(r"\b" + y + r"\b", t))
    gp = _extract_gp(query)
    if gp:
        results = keep(results, _gp_matcher(gp))
    names = [_fold(n) for n in _PIN_NAME.findall(query)]
    if names:
        results = keep(results, lambda t: any(n in t for n in names))
    return results


def retrieve_facts(query: str, k: int = 4) -> list[dict]:
    """Return the top-k relevant fact documents (empty if no index).

    Conversational turns ("Yes tell me.", "thanks!") bypass retrieval entirely —
    a grounding-trained model will otherwise faithfully recite whatever
    irrelevant doc a contentless query happens to match.
    """
    if _is_conversational(query):
        return []
    retriever = _get_retriever()
    if not retriever:
        return []
    sources = _route(query)
    results = retriever.retrieve(query, k=k, sources=sources)
    if not results and sources:
        results = retriever.retrieve(query, k=k)   # routed miss -> open fallback
    results = _pin(query, results)

    # Nonexistent-race guard. The corpus holds EVERY championship round
    # 1950→today, so a year+GP query with no doc matching both means the race
    # wasn't held ("2007 Singapore GP" — Singapore joined in 2008). Without
    # this, pinning falls back to the same GP in OTHER years and the model
    # faithfully recites a neighbouring race under the asked year — measured
    # live as a fabricated 2007 Singapore result built from the 2010 doc.
    # Stating the absence as a context fact turns the weak decline behavior
    # into the strong recitation behavior.
    year_m = _PIN_YEAR.search(query)
    gp = _extract_gp(query)
    if year_m and gp:
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
                recovered.sort(
                    key=lambda d: 0 if d.get("source", "").endswith("_race") else 1)
                return recovered[:k]
            return [{
                "text": (f"There is no {year} {gp} in the historical race "
                         f"records — that Grand Prix was not held in {year}."),
                "source": "absence_guard",
            }]
    return results


def retrieve_context(query: str, k: int = 4) -> str:
    """Return retrieved facts as a compact text block to inject into the prompt.

    Empty string when nothing relevant is found, so the caller can simply skip
    grounding for off-topic or already-known questions.
    """
    facts = retrieve_facts(query, k=k)
    if not facts:
        return ""
    lines = [f"- {f['text']}" for f in facts]
    return (
        "Verified F1 facts (rely on these for any names, numbers, teams or results; "
        "don't contradict them or invent specifics):\n" + "\n".join(lines)
    )
