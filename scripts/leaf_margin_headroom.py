#!/usr/bin/env python
"""Leaf-margin headroom pre-check (experiment go/no-go gate).

Before spending runs on the experiment ladder, confirm the leaf-margin metric can
actually move: report the gradable-margin counts, the entire/toothed/lobed class
distribution per source (a heavy skew to ``entire`` caps κ headroom), the
Roni-vs-Carmen human ceiling, and the current baseline model's standing. If the
ceiling is usable and the baseline sits well below it, there is room for a lever to
help; if the classes are near-degenerate or the ceiling is noise, pick a
higher-variance pilot trait instead.

Usage::

    python scripts/leaf_margin_headroom.py \\
        --results-dir trait_grading/model_run/2026-07-06_134225
"""
from __future__ import annotations

import argparse
import collections
import logging
import sys
from pathlib import Path

from seedlearn.benchmarking.human.categorical_grader import load_model_traits, model_lookup
from seedlearn.benchmarking.human.report import assemble
from seedlearn.benchmarking.human.value_map import MISSING, TRAIT_SPECS

logger = logging.getLogger(__name__)
_TG = Path("trait_grading")
TRAIT = "leaf_margin"


def _dist(counter: collections.Counter) -> str:
    total = sum(counter.values())
    parts = [f"{k}={counter[k]}" for k in sorted(counter)]
    return f"{', '.join(parts)}  (n={total})"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True, help="Baseline model_run dir.")
    p.add_argument("--roni-xlsx", default=str(_TG / "annotations/roni_bianco.xlsx"))
    p.add_argument("--carmen-xlsx", default=str(_TG / "annotations/carmen.xlsx"))
    p.add_argument("--curator-key", default=str(_TG / "keys/curator_taxonomic_key.csv"))
    args = p.parse_args(argv)

    bundle = assemble(args.results_dir, args.roni_xlsx, args.carmen_xlsx,
                      args.curator_key, stri_matrix=None, embed_images=False)

    def agr(axis: str):
        return next((a for a in bundle.agreements if a.trait_key == TRAIT and a.axis == axis), None)

    ceiling = agr("roni_vs_carmen")
    baseline = agr("model_vs_roni")

    # Ground-truth class distribution per human annotator (modal margin, excl. MISSING).
    per_annotator: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for agg in bundle.aggregates:
        ta = agg.traits.get(TRAIT)
        if ta and ta.mode != MISSING:
            per_annotator[agg.annotator][ta.mode] += 1

    # Model class distribution.
    spec = next(s for s in TRAIT_SPECS if s.key == TRAIT)
    model_tokens = model_lookup(load_model_traits(args.results_dir), specs=(spec,))
    model_dist = collections.Counter(
        t[TRAIT] for t in model_tokens.values() if t[TRAIT] != MISSING
    )

    logger.info("=== Leaf-margin headroom pre-check ===\n")
    for annotator, dist in sorted(per_annotator.items()):
        logger.info("  %-16s margin classes: %s", annotator, _dist(dist))
    logger.info("  %-16s margin classes: %s", "model(baseline)", _dist(model_dist))
    logger.info("")
    if ceiling:
        logger.info("  Human ceiling (Roni vs Carmen): %.1f%% agreement, κ %s (n=%d)",
                    (ceiling.agreement_rate or 0) * 100,
                    "—" if ceiling.cohen_kappa is None else f"{ceiling.cohen_kappa:.3f}",
                    ceiling.n_compared)
    if baseline:
        logger.info("  Baseline model vs Roni:         %.1f%% agreement, κ %s (n=%d)",
                    (baseline.agreement_rate or 0) * 100,
                    "—" if baseline.cohen_kappa is None else f"{baseline.cohen_kappa:.3f}",
                    baseline.n_compared)

    # Go/no-go: usable ceiling κ and clear gap between baseline and ceiling.
    verdict = "GO"
    reasons = []
    if not ceiling or ceiling.cohen_kappa is None or ceiling.cohen_kappa < 0.4:
        verdict, r = "CAUTION", "human ceiling κ is weak (< 0.4) — ground truth may be too noisy"
        reasons.append(r)
    if baseline and ceiling and baseline.cohen_kappa is not None and ceiling.cohen_kappa is not None:
        gap = ceiling.cohen_kappa - baseline.cohen_kappa
        reasons.append(f"baseline-to-ceiling κ gap = {gap:.3f}")
        if gap < 0.05:
            verdict = "NO-GO"
            reasons.append("baseline already near the ceiling — little room for a lever")
    # Degenerate class balance check (dominant class > 90% for every human source).
    for annotator, dist in per_annotator.items():
        tot = sum(dist.values())
        if tot and max(dist.values()) / tot > 0.9:
            reasons.append(f"{annotator}'s margin is >90% one class — low variance")
            if verdict == "GO":
                verdict = "CAUTION"

    logger.info("\n  Verdict: %s  (%s)", verdict, "; ".join(reasons) if reasons else "clear headroom")
    return 0


if __name__ == "__main__":
    sys.exit(main())
