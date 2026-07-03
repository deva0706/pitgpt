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
    (re.compile(r"champion|title", re.I), ["kaggle_champion", "curated"]),
    # Driver-centric phrasings outrank the race route: "career Grand Prix win
    # tally" is a question about a DRIVER even though it contains "Grand Prix".
    (re.compile(r"who is|tell me about|rundown on|teams? did|drive for|race for|"
                r"career|win tally|how many", re.I),
     ["kaggle_driver", "curated", "openf1"]),
    # A race question needs a winner-verb, not just the words "grand prix".
    (re.compile(r"(who won|winner|victory|took .*(flag|win)|crossed the line)"
                r".*(grand prix|\bgp\b)", re.I),
     ["kaggle_race"]),
    (re.compile(r"circuit|track\b|where is", re.I), ["kaggle_circuit", "curated"]),
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


def _is_conversational(query: str) -> bool:
    """True for turns that carry no F1 content worth retrieving for."""
    q = query.strip()
    return not (_F1_KEYWORDS.search(q) or _NAME_CUE.search(q))


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
