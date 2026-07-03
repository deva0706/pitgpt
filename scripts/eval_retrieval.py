"""Thin CLI: retrieval-only hit-rate harness (no LLM).

Usage:  python scripts/eval_retrieval.py [--tag NAME] [--k K]
"""

import argparse
import logging

from pitgpt.evaluation.retrieval_eval import run_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="retrieval")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    print(f"Report: {run_eval(tag=args.tag, k=args.k)}")
