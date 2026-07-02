"""Build the RAG fact index (curated facts + live OpenF1 data).

Usage:  python scripts/build_rag.py
"""

import logging

from pitgpt.rag.index import build_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    retriever = build_index()
    # Quick smoke query so you can see retrieval working immediately.
    for q in ["Who is the seven-time world champion?", "Why is Monaco special?"]:
        hits = retriever.retrieve(q, k=2)
        print(f"\nQ: {q}")
        for h in hits:
            print(f"  ({h['score']}) {h['text'][:90]}...")
