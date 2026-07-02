"""
index.py — embed the F1 fact corpus and retrieve the most relevant facts.

This is the "R" in RAG. We embed every fact document once with a small sentence
model, store the vectors, and at query time embed the user's question and return
the top-k most similar facts by cosine similarity. The corpus is small (dozens to
a few hundred facts), so a plain NumPy dot-product is instant — no heavyweight
vector database required.

Why a *separate* small embedding model rather than Gemma?
    Retrieval and generation are different jobs. A compact bi-encoder
    (all-MiniLM-L6-v2, ~22M params, 384-dim) is purpose-built to turn text into
    comparable vectors cheaply, which is exactly what search needs. Gemma stays
    focused on writing the answer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from pitgpt.config import settings

logger = logging.getLogger("pitgpt.rag.index")

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_INDEX_DIR = settings.paths.root / "data" / "knowledge" / "rag_index"


class FactRetriever:
    """Embeds a fact corpus and returns the top-k facts for a query."""

    def __init__(self, model_name: str = _EMBED_MODEL) -> None:
        self._model_name = model_name
        self._model = None                       # lazily loaded
        self._docs: list[dict[str, str]] = []
        self._embeddings: np.ndarray | None = None

    # ---- model ----
    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s ...", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        # Unit-normalised vectors so a dot product IS cosine similarity.
        return np.asarray(
            model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    # ---- build / persist ----
    def build(self, documents: list[dict[str, str]]) -> None:
        """Embed and store all documents (each: {"text", "source"})."""
        self._docs = [d for d in documents if d.get("text", "").strip()]
        texts = [d["text"] for d in self._docs]
        logger.info("Embedding %d fact documents ...", len(texts))
        self._embeddings = self._encode(texts)

    def save(self, index_dir: Path = _INDEX_DIR) -> Path:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self._embeddings)
        (index_dir / "docs.json").write_text(
            json.dumps(self._docs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Saved RAG index (%d docs) to %s", len(self._docs), index_dir)
        return index_dir

    def load(self, index_dir: Path = _INDEX_DIR) -> "FactRetriever":
        emb_path = index_dir / "embeddings.npy"
        doc_path = index_dir / "docs.json"
        if not emb_path.exists() or not doc_path.exists():
            raise FileNotFoundError(
                f"RAG index not found in {index_dir}. Run `python scripts/build_rag.py` first."
            )
        self._embeddings = np.load(emb_path)
        self._docs = json.loads(doc_path.read_text(encoding="utf-8"))
        logger.info("Loaded RAG index (%d docs) from %s", len(self._docs), index_dir)
        return self

    # ---- query ----
    def retrieve(self, query: str, k: int = 4, min_score: float = 0.25) -> list[dict]:
        """Return up to ``k`` documents most similar to ``query``.

        ``min_score`` filters out weak matches so an off-topic question doesn't
        drag in irrelevant facts (which would only mislead the model).
        """
        if self._embeddings is None or not self._docs:
            return []
        q = self._encode([query])[0]                 # (dim,)
        scores = self._embeddings @ q                # cosine sims, (n_docs,)
        top = np.argsort(-scores)[:k]
        results = []
        for i in top:
            score = float(scores[i])
            if score < min_score:
                continue
            results.append({**self._docs[i], "score": round(score, 3)})
        return results


def build_index() -> FactRetriever:
    """Assemble the corpus (curated + OpenF1), embed it, and persist the index."""
    from pitgpt.rag.facts import curated_documents
    from pitgpt.rag.openf1 import openf1_documents

    docs = curated_documents() + openf1_documents()
    retriever = FactRetriever()
    retriever.build(docs)
    retriever.save()
    return retriever
