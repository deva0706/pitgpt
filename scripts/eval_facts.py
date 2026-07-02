"""Thin CLI: run the closed-book factual-accuracy benchmark.

Usage:  python scripts/eval_facts.py [--limit N] [--tag NAME]
"""

import argparse
import logging

from pitgpt.evaluation.facts_eval import run_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", type=str, default="facts")
    parser.add_argument("--rag", action="store_true",
                        help="use live hybrid retrieval (end-to-end pipeline)")
    args = parser.parse_args()
    print(f"Report: {run_eval(tag=args.tag, limit=args.limit, use_rag=args.rag)}")
