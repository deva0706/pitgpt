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
import re
from pathlib import Path

import numpy as np

from pitgpt.config import settings

logger = logging.getLogger("pitgpt.rag.index")

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_INDEX_DIR = settings.paths.root / "data" / "knowledge" / "rag_index"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens — all BM25 needs for names and years."""
    return _TOKEN_RE.findall(text.lower())


class FactRetriever:
    """Hybrid retriever: dense (MiniLM) + lexical (BM25), fused by rank."""

    def __init__(self, model_name: str = _EMBED_MODEL) -> None:
        self._model_name = model_name
        self._model = None                       # lazily loaded
        self._docs: list[dict[str, str]] = []
        self._embeddings: np.ndarray | None = None
        self._bm25 = None
        self._bm25_corpus: list[list[str]] = []

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
        self._build_bm25()

    def _build_bm25(self) -> None:
        """Build the lexical half of hybrid retrieval (BM25 over doc tokens).

        Dense embeddings are great at meaning but weak at exact tokens — "2016"
        and "2021" embed almost identically, which is precisely how the old
        retriever confused champion years. BM25 nails exact names and years;
        fusing both fixes each ranker's blind spot.
        """
        from rank_bm25 import BM25Okapi

        self._bm25_corpus = [_tokenize(d["text"]) for d in self._docs]
        self._bm25 = BM25Okapi(self._bm25_corpus)

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
        self._build_bm25()   # cheap to rebuild from docs; avoids pickling state
        logger.info("Loaded RAG index (%d docs) from %s", len(self._docs), index_dir)
        return self

    # ---- query ----
    def retrieve(
        self,
        query: str,
        k: int = 4,
        min_score: float = 0.25,
        sources: list[str] | None = None,
    ) -> list[dict]:
        """Hybrid retrieval: dense cosine + BM25, fused by reciprocal rank.

        RRF (k=60) is the standard, hyperparameter-light way to combine rankers
        with incomparable score scales. A relevance gate then drops candidates
        that neither ranker actually liked (weak dense score AND no lexical
        overlap), so off-topic questions don't drag in misleading facts.

        ``sources`` restricts candidates to the given doc families. This is the
        fix for the "popular entity" problem: Schumacher has ~90 race-winner
        docs that otherwise flood the ranking and crowd out his single bio doc
        when the question is about *him*, not a race.
        """
        if self._embeddings is None or not self._docs:
            return []
        q = self._encode([query])[0]
        dense_scores = self._embeddings @ q                       # (n_docs,)
        bm25_scores = np.asarray(self._bm25.get_scores(_tokenize(query)))

        if sources:
            allowed = np.array([d.get("source") in sources for d in self._docs])
            if allowed.any():                       # never filter down to nothing
                dense_scores = np.where(allowed, dense_scores, -1.0)
                bm25_scores = np.where(allowed, bm25_scores, 0.0)

        dense_rank = np.argsort(-dense_scores)
        bm25_rank = np.argsort(-bm25_scores)

        # Reciprocal rank fusion over the top window of each ranker.
        K, window = 60, max(20, 4 * k)
        fused: dict[int, float] = {}
        for rank_list in (dense_rank[:window], bm25_rank[:window]):
            for pos, i in enumerate(rank_list):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (K + pos + 1)

        results = []
        for i in sorted(fused, key=fused.get, reverse=True):
            dense, lexical = float(dense_scores[i]), float(bm25_scores[i])
            # Stricter gate: BM25 gives small nonzero scores for stopword-ish
            # overlap, which previously let contentless queries drag in random
            # docs (grounded non-sequiturs). Require a real signal from at
            # least one ranker.
            if dense < min_score and lexical < 2.0:
                continue
            results.append({**self._docs[i], "score": round(fused[i], 4),
                            "dense": round(dense, 3), "bm25": round(lexical, 2)})
            if len(results) >= k:
                break
        return results


def build_index() -> FactRetriever:
    """Assemble the full corpus and persist the hybrid index.

    Corpus = curated evergreen rules + OpenF1 current grid + the complete
    Kaggle-derived historical record (~2,200 atomic fact docs).
    """
    from pitgpt.rag.facts import curated_documents
    from pitgpt.rag.kb_builder import kaggle_documents
    from pitgpt.rag.openf1 import openf1_documents

    docs = curated_documents() + openf1_documents() + kaggle_documents()
    retriever = FactRetriever()
    retriever.build(docs)
    retriever.save()
    return retriever
