"""Thin CLI: run the grounding benchmark (Phase 9).

Usage:  python scripts/eval_grounding.py [--limit N] [--tag NAME] [--check]

By default the model's replies are scored as written. ``--check`` scores what
the app would show for the same context: the reply after the answer check.
"""

import argparse
import logging

from pitgpt.evaluation.grounding_eval import run_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    parser.add_argument("--tag", type=str, default="current", help="label for result files")
    parser.add_argument("--check", action="store_true",
                        help="score replies after the app's answer check")
    args = parser.parse_args()
    report = run_eval(tag=args.tag, limit=args.limit, check=args.check)
    print(f"Report: {report}")
