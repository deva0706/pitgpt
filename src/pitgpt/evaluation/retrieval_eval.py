"""
retrieval_eval.py — retrieval-only hit-rate harness (no LLM, no GPU cost).

The end-to-end facts benchmark conflates two failures: the right doc never
reached the prompt, or the model ignored it. This harness isolates the first:
for each of the same 90 cases, did the retrieved context *contain* the gold
answer? It runs in seconds, so corpus and routing changes get an instant
feedback number before any training is spent.

Run:  ``python scripts/eval_retrieval.py [--tag NAME] [--k K]``
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from pitgpt.config import settings
from pitgpt.evaluation.facts_eval import FactCase, build_cases, wrong_team_hits

logger = logging.getLogger("pitgpt.eval.retrieval")


def _present(needle: str, haystack_low: str) -> bool:
    return re.search(r"\b" + re.escape(needle.lower()) + r"\b", haystack_low) is not None


def gold_in_context(case: FactCase, context: str) -> bool:
    """Same pass logic as the facts scorer, applied to the retrieved text."""
    low = context.lower()
    if case.category == "driver_teams":
        hit = [g for g in case.gold if _present(g, low)]
        return len(hit) >= max(1, len(case.gold) // 2)
    return any(g.lower() in low for g in case.gold)


def run_eval(tag: str = "retrieval", k: int = 4) -> Path:
    from pitgpt.rag.retriever import retrieve_facts

    cases = build_cases()
    rows = []
    for idx, case in enumerate(cases):
        docs = retrieve_facts(case.question, k=k)
        context = "\n".join(d.get("text", "") for d in docs)
        low = context.lower()
        rows.append({
            "idx": idx, "category": case.category, "tier": case.tier,
            "question": case.question, "gold": case.gold,
            "n_docs": len(docs),
            "sources": sorted({d.get("source", "?") for d in docs}),
            "hit": gold_in_context(case, context),
            # precision: distractor facts sharing the context confuse recitation
            "wrong_in_ctx": len(wrong_team_hits(case, low)),
            "context": context[:400],
        })

    out = settings.paths.evaluation / f"retrieval_results_{tag}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    lines = [f"# Retrieval hit-rate — `{tag}` (k={k})", "",
             "Gold answer present in retrieved context (LLM not involved).",
             "'Clean' = no distractor entities from the wrong-pool in context.", "",
             "| Category | Cases | Hit-rate | Clean contexts | Avg docs |",
             "|---|---|---|---|---|"]
    for cat, sub in sorted(by_cat.items()):
        clean = sum(1 for r in sub if r["wrong_in_ctx"] == 0)
        avg_docs = sum(r["n_docs"] for r in sub) / len(sub)
        lines.append(f"| {cat} | {len(sub)} | {sum(r['hit'] for r in sub)/len(sub):.0%} "
                     f"| {clean/len(sub):.0%} | {avg_docs:.1f} |")
    clean_all = sum(1 for r in rows if r["wrong_in_ctx"] == 0)
    lines.append(f"| **overall** | {len(rows)} | "
                 f"{sum(r['hit'] for r in rows)/len(rows):.0%} "
                 f"| {clean_all/len(rows):.0%} "
                 f"| {sum(r['n_docs'] for r in rows)/len(rows):.1f} |")
    lines += ["", "## Misses", ""]
    for r in [r for r in rows if not r["hit"]][:12]:
        lines.append(f"- **[{r['category']}]** {r['question']} → gold `{r['gold'][:3]}` "
                     f"(docs from {r['sources']})")
    report = out.with_name(f"retrieval_report_{tag}.md")
    report.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report: %s", report)
    return report
