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


def retrieve_facts(query: str, k: int = 4) -> list[dict]:
    """Return the top-k relevant fact documents (empty if no index)."""
    retriever = _get_retriever()
    return retriever.retrieve(query, k=k) if retriever else []


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
