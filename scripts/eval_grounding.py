"""Thin CLI: run the grounding benchmark (Phase 9).

Usage:  python scripts/eval_grounding.py [--limit N] [--tag NAME]
"""

import argparse
import logging

from pitgpt.evaluation.grounding_eval import run_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    parser.add_argument("--tag", type=str, default="current", help="label for result files")
    args = parser.parse_args()
    report = run_eval(tag=args.tag, limit=args.limit)
    print(f"Report: {report}")
