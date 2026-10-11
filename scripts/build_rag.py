"""Build the RAG fact index (curated facts + historical record + recent seasons).

Usage:  python scripts/build_rag.py [--refresh]

The recent-season sources (Jolpica, OpenF1) are cached on first fetch, so a
plain rebuild is offline and reproducible. ``--refresh`` asks both APIs again
for the season in progress and the current grid, and keeps the cached copy of
anything it cannot reach.
"""

import argparse
import datetime
import logging

from pitgpt.rag import jolpica, openf1
from pitgpt.rag.index import build_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="re-download the current season and the current grid")
    args = parser.parse_args()
    if args.refresh:
        jolpica.refresh(datetime.date.today().year)
        openf1.refresh()

    retriever = build_index()
    # Quick smoke query so you can see retrieval working immediately.
    for q in ["Who is the seven-time world champion?", "Why is Monaco special?"]:
        hits = retriever.retrieve(q, k=2)
        print(f"\nQ: {q}")
        for h in hits:
            print(f"  ({h['score']}) {h['text'][:90]}...")
