"""
facts_eval.py — closed-book factual-accuracy benchmark (RAG-off era).

Measures what the adapter actually MEMORIZED from the Kaggle-derived training
set, honestly:

* **Held-out phrasings.** Questions are worded differently from the training
  rows ("Which driver took the title in 1994?" vs the trained "Who won the 1994
  Formula 1 World Championship?"), so parroting an exact-match template isn't
  enough — the fact itself must be recalled.
* **Popularity tiers.** Drivers are split by career size (races entered) into
  popular / mid / long-tail, exposing the expected memorization gradient of a
  1B model instead of hiding it in one average.
* **Signature adherence.** Each reply is also checked for its category's
  signature opening (🏎️ / "Lights out" / "Season ... belonged" / ...), measuring
  the style-conditioning separately from factual recall.

Gold answers come from the same Kaggle tables the training set was built from,
so scoring is exact and deterministic. Results stream to JSONL (resumable).

Run:  ``python scripts/eval_facts.py [--limit N] [--tag NAME]``
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pitgpt.config import settings
from pitgpt.data.templates import adheres
from pitgpt.data.fact_dataset import (
    _career_facts,
    _load_tables,
    _name,
    _season_final_race_ids,
)

logger = logging.getLogger("pitgpt.eval.facts")


@dataclass
class FactCase:
    category: str          # driver_teams | champion_year | gp_winner | win_count
    tier: str              # popular | mid | longtail | n/a
    question: str
    gold: list[str]        # substrings that should appear (any-of unless noted)
    wrong_pool: list[str] = field(default_factory=list)   # for hallucination flags
    signature: str = ""    # templates.py format category for adherence scoring


def _tiered_drivers(t: dict) -> dict[str, list[str]]:
    """Split driverIds into popularity tiers by races entered."""
    race_count = Counter(r["driverId"] for r in t["results"])
    ranked = [d for d, _ in race_count.most_common()]
    n = len(ranked)
    return {
        "popular": ranked[:30],
        "mid": ranked[n // 4: n // 4 + 40],
        "longtail": ranked[n // 2: n // 2 + 40],
    }


def build_cases(seed: int = 11) -> list[FactCase]:
    rng = random.Random(seed)
    t = _load_tables()
    facts = _career_facts(t)
    tiers = _tiered_drivers(t)
    all_teams = sorted(set(t["constructors"].values()))

    # champions by year
    final_ids = _season_final_race_ids(t)
    champs: dict[int, str] = {}
    for s in t["standings"]:
        year = final_ids.get(s["raceId"])
        if year is not None and s["position"] == "1":
            d = t["drivers"].get(s["driverId"])
            if d:
                champs[year] = _name(d)

    # GP winners: (year, gp, winner, team)
    winners = []
    for r in t["results"]:
        if r.get("positionText") == "1":
            race = t["races"].get(r["raceId"])
            d = t["drivers"].get(r["driverId"])
            if race and d:
                winners.append((race["year"], race["name"], _name(d)))

    cases: list[FactCase] = []

    # Vintage guard: Kaggle gold ends at 2024, but a driver's CURRENT team is
    # a true answer, not a hallucination — drop demonstrably-recent teams
    # (Jolpica 2024→) from the wrong-pool. Offline-safe (empty without cache).
    try:
        from pitgpt.rag.jolpica import recent_teams
        recent = recent_teams()
    except Exception:  # noqa: BLE001
        recent = {}

    # --- driver -> teams (10 per tier, held-out phrasing) ---
    for tier, ids in tiers.items():
        picked = 0
        for d_id in ids:
            f = facts.get(d_id)
            if not f or not f["teams"] or picked >= 10:
                continue
            gold_teams = list(f["teams"])
            now_teams = recent.get(f["name"], set())
            cases.append(FactCase(
                category="driver_teams", tier=tier,
                question=f"What teams did {f['name']} race for over his F1 career?",
                gold=gold_teams,
                wrong_pool=[x for x in all_teams
                            if x not in gold_teams and x not in now_teams],
                signature="driver_card",
            ))
            picked += 1

    # --- champion by year (30, spread across decades, held-out phrasing) ---
    years = sorted(champs)
    for year in years[:: max(1, len(years) // 30)][:30]:
        cases.append(FactCase(
            category="champion_year", tier="n/a",
            question=f"Which driver took the F1 title in {year}?",
            gold=[champs[year].split()[-1]],           # surname is enough
            signature="championship",
        ))

    # --- GP winners (20, sampled, held-out phrasing) ---
    for year, gp, winner in rng.sample(winners, 20):
        cases.append(FactCase(
            category="gp_winner", tier="n/a",
            question=f"Who crossed the line first at the {year} {gp}?",
            gold=[winner.split()[-1]],
            signature="race_winner",
        ))

    # --- win counts (10 popular winners, held-out phrasing) ---
    picked = 0
    for d_id in tiers["popular"]:
        f = facts.get(d_id)
        if not f or f["wins"] < 5 or picked >= 10:
            continue
        cases.append(FactCase(
            category="win_count", tier="popular",
            question=f"What's {f['name']}'s career Grand Prix win tally?",
            gold=[str(f["wins"])],
            signature="win_count",
        ))
        picked += 1

    return cases


def _team_mentioned(team: str, text: str) -> bool:
    """Word-boundary match. Plain substring is wrong here: the Kaggle
    constructor table contains short names like "RE" and "RB" that substring-
    match inside ordinary words ("raced", "here"), flagging phantom
    hallucinations on fully correct answers."""
    return re.search(r"\b" + re.escape(team.lower()) + r"\b", text) is not None


def wrong_team_hits(case: FactCase, text_low: str) -> list[str]:
    """Wrong-pool mentions AFTER masking gold teams. A driver's own correct
    team names can contain wrong-pool teams as substrings ("BMW Sauber" ∋
    "Sauber", "Brabham-Ford" ∋ "Ford"), so a perfect recitation would
    otherwise score as a hallucination."""
    masked = text_low
    for g in sorted(case.gold, key=len, reverse=True):
        masked = re.sub(r"\b" + re.escape(g.lower()) + r"\b", " ", masked)
    return [w for w in case.wrong_pool if _team_mentioned(w, masked)]


def score_case(case: FactCase, answer: str) -> dict:
    low = answer.lower()
    if case.category == "driver_teams":
        hit = [g for g in case.gold if _team_mentioned(g, low)]
        wrong = wrong_team_hits(case, low)
        passed = len(hit) >= max(1, len(case.gold) // 2) and not wrong
        hallucinated = bool(wrong)
    else:
        passed = any(g.lower() in low for g in case.gold)
        hallucinated = not passed and len(answer) > 30   # confident-but-wrong
    return {
        "passed": passed,
        "hallucinated": hallucinated,
        "signature_ok": adheres(case.signature, answer),
    }


def run_eval(tag: str = "facts", limit: int | None = None, use_rag: bool = False) -> Path:
    """Run the benchmark. ``use_rag=False`` measures closed-book recall from the
    weights; ``use_rag=True`` measures the full pipeline (live hybrid retrieval
    over the 2,270-doc corpus + grounded generation). Same cases, same scoring —
    a true A/B of the two architectures.
    """
    from pitgpt.inference.engine import generate_reply
    from pitgpt.inference.model_loader import load_for_inference

    out = settings.paths.evaluation / f"facts_results_{tag}.jsonl"
    done: set[int] = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["idx"])
            except Exception:  # noqa: BLE001
                pass

    cases = build_cases()
    if limit:
        cases = cases[:limit]
    model, tokenizer = load_for_inference(
        adapter_dir=settings.model.adapter_dir, quantized=True
    )
    logger.info("Facts eval: %d cases (%d done).", len(cases), len(done))
    with out.open("a", encoding="utf-8") as fh:
        for idx, case in enumerate(cases):
            if idx in done:
                continue
            answer = generate_reply(model, tokenizer, case.question, use_rag=use_rag)
            rec = {"idx": idx, "category": case.category, "tier": case.tier,
                   "question": case.question, "gold": case.gold,
                   "answer": answer, **score_case(case, answer)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if (idx + 1) % 10 == 0:
                logger.info("  %d/%d", idx + 1, len(cases))
    return write_report(out, tag)


def write_report(jsonl_path: Path, tag: str) -> Path:
    rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines()]
    by_cat = defaultdict(list)
    by_tier = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
        if r["tier"] != "n/a":
            by_tier[r["tier"]].append(r)
    lines = [f"# PitGPT Closed-Book Facts Benchmark — `{tag}`", "",
             f"Cases: {len(rows)} · held-out phrasings · gold from Kaggle tables", "",
             "| Category | Cases | Accuracy | Signature adherence |", "|---|---|---|---|"]
    for cat, sub in sorted(by_cat.items()):
        acc = sum(r["passed"] for r in sub) / len(sub)
        sig = sum(r["signature_ok"] for r in sub) / len(sub)
        lines.append(f"| {cat} | {len(sub)} | {acc:.0%} | {sig:.0%} |")
    total_pass = sum(r["passed"] for r in rows)
    total_sig = sum(r["signature_ok"] for r in rows)
    lines += [f"| **overall** | {len(rows)} | {total_pass/len(rows):.0%} | {total_sig/len(rows):.0%} |", "",
              "## Memorization gradient (driver questions by popularity)", "",
              "| Tier | Cases | Accuracy |", "|---|---|---|"]
    for tier in ("popular", "mid", "longtail"):
        sub = by_tier.get(tier, [])
        if sub:
            lines.append(f"| {tier} | {len(sub)} | {sum(r['passed'] for r in sub)/len(sub):.0%} |")
    lines += ["", "## Sample failures", ""]
    for r in [r for r in rows if not r["passed"]][:8]:
        lines.append(f"- **[{r['category']}/{r['tier']}]** {r['question']} → gold "
                     f"`{r['gold'][:3]}` → _{r['answer'][:130]}…_")
    report = jsonl_path.with_name(f"facts_report_{tag}.md")
    report.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report: %s", report)
    return report
