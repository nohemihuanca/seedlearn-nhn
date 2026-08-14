#!/usr/bin/env python
"""Compute ablation experiment metrics and generate paper tables/figures.

Reads condition A-D output JSONs and produces:
- Accuracy tables (top-1/3/5 per condition)
- Per-family accuracy breakdown
- McNemar's paired significance tests
- RAG retrieval precision (Experiment 2, from condition A outputs)
- Convergence predictive value (Experiment 3, from condition A outputs)
- Figures for the paper

Usage:
    python compute_metrics.py [--output-dir experiments/ablation/outputs]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_condition_results(
    condition_dir: Path,
) -> list[dict]:
    """Load all specimen result JSONs from a condition directory."""
    results = []
    for path in sorted(condition_dir.glob("*.json")):
        with open(path) as f:
            results.append(json.load(f))
    return results


def extract_prediction(result: dict, condition: str) -> str | None:
    """Extract the predicted family from a pipeline result."""
    stages = result.get("stages", {})

    # Conditions A/B/C: use reasoning stage output
    reasoning = stages.get("reasoning", {})
    if not reasoning.get("skipped", True) and reasoning.get("error") is None:
        clf = reasoning.get("data", {}).get("classification", {})
        predicted = clf.get("predicted_family")
        if predicted:
            return predicted

    # Condition D or fallback: use classification stage top-1
    classification = stages.get("classification", {})
    if not classification.get("skipped", True):
        data = classification.get("data", {})
        preds_by_rank = data.get("predictions_by_rank", {})
        family_preds = preds_by_rank.get("family", [])
        if family_preds:
            return family_preds[0].get("rank_value")

    return None


def extract_top_k_families(result: dict, k: int) -> list[str]:
    """Extract top-k predicted families from classification stage."""
    stages = result.get("stages", {})
    classification = stages.get("classification", {})
    data = classification.get("data", {})
    preds = data.get("predictions_by_rank", {}).get("family", [])
    return [p["rank_value"] for p in preds[:k]]


def extract_ground_truth(result: dict) -> dict[str, str]:
    """Extract ground truth at family/genus/species from image path.

    Path format: .../by_family/X/by_genus/Y/by_species/Z/SPECIMEN/img.jpg
    """
    gt = {"family": result.get("ground_truth", {}).get("family", "")}
    img = result.get("image_paths", [""])[0]
    parts = img.split("/")
    for i, p in enumerate(parts):
        if p == "by_genus" and i + 1 < len(parts):
            gt["genus"] = parts[i + 1]
        if p == "by_species" and i + 1 < len(parts):
            gt["species_epithet"] = parts[i + 1]
    if gt.get("genus") and gt.get("species_epithet"):
        gt["species"] = f"{gt['genus']} {gt['species_epithet']}"
    return gt


def extract_multirank_prediction(result: dict, condition: str) -> dict[str, str]:
    """Extract predicted family/genus/species from reasoning or visual stage."""
    stages = result.get("stages", {})
    reasoning = stages.get("reasoning", {})
    if not reasoning.get("skipped", True) and reasoning.get("error") is None:
        clf = reasoning.get("data", {}).get("classification", {})
        return {
            "family": clf.get("predicted_family", ""),
            "genus": clf.get("predicted_genus", ""),
            "species": clf.get("predicted_species", ""),
        }
    # Fallback to visual classifier
    classification = stages.get("classification", {})
    data = classification.get("data", {})
    preds_by_rank = data.get("predictions_by_rank", {})
    pred = {}
    for rank in ("family", "genus", "species"):
        rank_preds = preds_by_rank.get(rank, [])
        pred[rank] = rank_preds[0].get("rank_value", "") if rank_preds else ""
    return pred


def compute_multirank_accuracy(
    all_results: dict[str, list[dict]],
) -> dict[str, dict[str, float]]:
    """Compute accuracy at family, genus, and species level per condition.

    Returns:
        {condition: {family: float, genus: float, species: float, n: int}}
    """
    table = {}
    for condition, results in all_results.items():
        counts = {"family": 0, "genus": 0, "species": 0}
        total = 0

        for r in results:
            gt = extract_ground_truth(r)
            if not gt.get("family"):
                continue
            total += 1
            pred = extract_multirank_prediction(r, condition)

            if pred["family"].lower() == gt["family"].lower():
                counts["family"] += 1
            if (
                pred.get("genus")
                and gt.get("genus")
                and pred["genus"].lower() == gt["genus"].lower()
            ):
                counts["genus"] += 1
            if (
                pred.get("species")
                and gt.get("species")
                and pred["species"].lower() == gt["species"].lower()
            ):
                counts["species"] += 1

        table[condition] = {
            rank: counts[rank] / max(total, 1)
            for rank in ("family", "genus", "species")
        }
        table[condition]["n"] = total
    return table


def compute_accuracy_table(
    all_results: dict[str, list[dict]],
) -> dict[str, dict[str, float]]:
    """Compute accuracy per condition.

    - **final_top1**: The pipeline's actual output (reasoning for A/B/C,
      visual top-1 for D).  This is the headline metric.
    - **visual_top1/3/5**: Stage 2 visual classifier accuracy, identical
      across conditions (same model, same images).  Provides the baseline
      that reasoning/RAG improve upon.

    Returns:
        {condition: {final_top1, visual_top1, visual_top3, visual_top5, n}}
    """
    table = {}
    for condition, results in all_results.items():
        final_correct = 0
        vis_correct_1 = 0
        vis_correct_3 = 0
        vis_correct_5 = 0
        total = 0

        for r in results:
            gt = r.get("ground_truth", {}).get("family")
            if not gt:
                continue
            total += 1

            # Pipeline's final prediction (reasoning or visual fallback)
            predicted = extract_prediction(r, condition)
            if predicted == gt:
                final_correct += 1

            # Visual classifier top-k (same across conditions)
            top_k = extract_top_k_families(r, 5)
            if top_k and top_k[0] == gt:
                vis_correct_1 += 1
            if gt in top_k[:3]:
                vis_correct_3 += 1
            if gt in top_k:
                vis_correct_5 += 1

        table[condition] = {
            "final_top1": final_correct / max(total, 1),
            "visual_top1": vis_correct_1 / max(total, 1),
            "visual_top3": vis_correct_3 / max(total, 1),
            "visual_top5": vis_correct_5 / max(total, 1),
            "n": total,
        }
    return table


def compute_per_family_accuracy(
    all_results: dict[str, list[dict]],
) -> dict[str, dict[str, float]]:
    """Compute per-family top-1 accuracy for each condition.

    Returns:
        {family: {condition: accuracy}}
    """
    family_stats: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"correct": 0, "total": 0})
    )

    for condition, results in all_results.items():
        for r in results:
            gt = r.get("ground_truth", {}).get("family")
            if not gt:
                continue
            predicted = extract_prediction(r, condition)
            family_stats[gt][condition]["total"] += 1
            if predicted == gt:
                family_stats[gt][condition]["correct"] += 1

    per_family = {}
    for family in sorted(family_stats.keys()):
        per_family[family] = {}
        for condition in all_results.keys():
            stats = family_stats[family].get(condition, {"correct": 0, "total": 0})
            total = stats["total"]
            per_family[family][condition] = (
                stats["correct"] / total if total > 0 else float("nan")
            )
    return per_family


def mcnemar_test(
    results_a: list[dict],
    results_b: list[dict],
    condition_a: str,
    condition_b: str,
) -> dict[str, float]:
    """Paired McNemar's test between two conditions.

    Compares correct/incorrect classification on the same specimens.
    """
    # Build specimen_id -> result lookup
    lookup_b = {r["specimen_id"]: r for r in results_b}

    # Count discordant pairs
    b_right_a_wrong = 0  # B correct, A wrong
    a_right_b_wrong = 0  # A correct, B wrong

    for r_a in results_a:
        sid = r_a["specimen_id"]
        r_b = lookup_b.get(sid)
        if r_b is None:
            continue
        gt = r_a.get("ground_truth", {}).get("family")
        if not gt:
            continue

        a_correct = extract_prediction(r_a, condition_a) == gt
        b_correct = extract_prediction(r_b, condition_b) == gt

        if a_correct and not b_correct:
            a_right_b_wrong += 1
        elif b_correct and not a_correct:
            b_right_a_wrong += 1

    # McNemar's chi-squared (with continuity correction)
    n = a_right_b_wrong + b_right_a_wrong
    if n == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n_discordant": 0}

    chi2 = (abs(a_right_b_wrong - b_right_a_wrong) - 1) ** 2 / n
    # p-value from chi-squared with 1 df
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=1)

    return {
        "chi2": chi2,
        "p_value": p_value,
        "n_discordant": n,
        f"{condition_a}_right_{condition_b}_wrong": a_right_b_wrong,
        f"{condition_b}_right_{condition_a}_wrong": b_right_a_wrong,
    }


def compute_rag_precision(results_a: list[dict]) -> dict[str, float]:
    """Experiment 2: RAG retrieval precision from condition A outputs.

    Checks if ground truth taxon appears in RAG top-k results.
    """
    k_values = [1, 3, 5, 10, 20]
    recall_at_k = {k: 0 for k in k_values}
    reciprocal_ranks = []
    total = 0

    for r in results_a:
        gt_family = r.get("ground_truth", {}).get("family", "").lower()
        if not gt_family:
            continue

        trait_data = r.get("stages", {}).get("trait_retrieval", {}).get("data", {})
        rag_matches = trait_data.get("rag_matches", [])
        if not rag_matches:
            total += 1
            reciprocal_ranks.append(0.0)
            continue

        total += 1
        taxa = [m.get("taxon", "").lower() for m in rag_matches]

        # Find first matching rank
        found_rank = None
        for rank_pos, taxon in enumerate(taxa, 1):
            if taxon == gt_family:
                found_rank = rank_pos
                break

        if found_rank:
            reciprocal_ranks.append(1.0 / found_rank)
            for k in k_values:
                if found_rank <= k:
                    recall_at_k[k] += 1
        else:
            reciprocal_ranks.append(0.0)

    metrics = {
        "mrr": np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "n": total,
    }
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k[k] / max(total, 1)
    return metrics


def compute_convergence_analysis(results_a: list[dict]) -> dict[str, dict]:
    """Experiment 3: Convergence signal vs. classification correctness."""
    signal_groups: dict[str, list[bool]] = defaultdict(list)

    for r in results_a:
        gt_family = r.get("ground_truth", {}).get("family")
        if not gt_family:
            continue

        predicted = extract_prediction(r, "A")
        is_correct = predicted == gt_family

        trait_data = r.get("stages", {}).get("trait_retrieval", {}).get("data", {})
        convergence = trait_data.get("convergence", [])

        # Categorize specimen by strongest convergence signal
        signals = [c.get("signal", "") for c in convergence]
        if "strong" in signals:
            signal_groups["strong"].append(is_correct)
        elif "moderate" in signals:
            signal_groups["moderate"].append(is_correct)
        elif any(s in ("rag_only", "visual_only") for s in signals):
            signal_groups["divergent"].append(is_correct)
        else:
            signal_groups["no_convergence"].append(is_correct)

    analysis = {}
    for signal, correctness_list in signal_groups.items():
        n = len(correctness_list)
        analysis[signal] = {
            "n": n,
            "accuracy": sum(correctness_list) / max(n, 1),
            "n_correct": sum(correctness_list),
        }
    return analysis


def compute_confidence_distribution(
    all_results: dict[str, list[dict]],
) -> dict[str, dict[str, int]]:
    """Distribution of confidence levels per condition (A/B/C only)."""
    dist: dict[str, dict[str, int]] = {}
    for condition in ("A", "B", "C"):
        results = all_results.get(condition, [])
        counts: dict[str, int] = defaultdict(int)
        for r in results:
            reasoning = r.get("stages", {}).get("reasoning", {})
            if reasoning.get("skipped", True):
                continue
            clf = reasoning.get("data", {}).get("classification", {})
            conf = clf.get("confidence", "unknown")
            counts[conf] += 1
        dist[condition] = dict(counts)
    return dist


def find_rag_impact_cases(
    results_a: list[dict],
    results_b: list[dict],
) -> dict[str, list[dict]]:
    """Find specimens where RAG changed the classification outcome."""
    lookup_b = {r["specimen_id"]: r for r in results_b}
    helped = []
    hurt = []

    for r_a in results_a:
        sid = r_a["specimen_id"]
        r_b = lookup_b.get(sid)
        if r_b is None:
            continue
        gt = r_a.get("ground_truth", {}).get("family")
        if not gt:
            continue

        pred_a = extract_prediction(r_a, "A")
        pred_b = extract_prediction(r_b, "B")

        if pred_a == gt and pred_b != gt:
            helped.append({
                "specimen_id": sid,
                "ground_truth": gt,
                "with_rag": pred_a,
                "without_rag": pred_b,
            })
        elif pred_a != gt and pred_b == gt:
            hurt.append({
                "specimen_id": sid,
                "ground_truth": gt,
                "with_rag": pred_a,
                "without_rag": pred_b,
            })

    return {"helped": helped, "hurt": hurt}


def print_summary(
    accuracy: dict,
    multirank: dict,
    per_family: dict,
    rag_precision: dict,
    convergence: dict,
    confidence: dict,
    rag_impact: dict,
    mcnemar_results: dict,
) -> None:
    """Print formatted summary to stdout."""
    print("\n" + "=" * 70)
    print("ABLATION EXPERIMENT RESULTS")
    print("=" * 70)

    print("\n--- Family-Level Accuracy by Condition ---\n")
    print(f"{'Condition':<20} {'Final':>8} {'Vis@1':>8} {'Vis@3':>8} {'Vis@5':>8} {'N':>6}")
    print("-" * 58)
    for cond in ("A", "B", "C", "D"):
        if cond in accuracy:
            a = accuracy[cond]
            print(f"{cond} ({CONDITIONS_NAMES.get(cond, '')})".ljust(20)
                  + f"{a['final_top1']:8.1%} {a['visual_top1']:8.1%} "
                  + f"{a['visual_top3']:8.1%} {a['visual_top5']:8.1%} {a['n']:6d}")

    if multirank:
        print("\n--- Multi-Rank Accuracy (Pipeline Final Prediction) ---\n")
        print(f"{'Condition':<20} {'Family':>8} {'Genus':>8} {'Species':>8} {'N':>6}")
        print("-" * 50)
        for cond in ("A", "B", "C", "D"):
            if cond in multirank:
                m = multirank[cond]
                print(f"{cond} ({CONDITIONS_NAMES.get(cond, '')})".ljust(20)
                      + f"{m['family']:8.1%} {m['genus']:8.1%} "
                      + f"{m['species']:8.1%} {m['n']:6d}")

    print("\n--- McNemar's Tests (paired significance) ---\n")
    for pair, result in mcnemar_results.items():
        sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
        print(f"  {pair}: chi2={result['chi2']:.2f}, p={result['p_value']:.4f} ({sig}), "
              f"n_discordant={result['n_discordant']}")

    print("\n--- RAG Retrieval Precision (Experiment 2) ---\n")
    print(f"  MRR:       {rag_precision['mrr']:.3f}")
    for k in (1, 3, 5, 10, 20):
        key = f"recall@{k}"
        if key in rag_precision:
            print(f"  Recall@{k}:  {rag_precision[key]:.1%}")
    print(f"  N:         {rag_precision['n']}")

    print("\n--- Convergence Analysis (Experiment 3) ---\n")
    print(f"{'Signal':<20} {'Accuracy':>10} {'N':>6}")
    print("-" * 40)
    for signal in ("strong", "moderate", "divergent", "no_convergence"):
        if signal in convergence:
            c = convergence[signal]
            print(f"{signal:<20} {c['accuracy']:10.1%} {c['n']:6d}")

    print("\n--- RAG Impact Cases ---\n")
    print(f"  RAG helped (A correct, B wrong): {len(rag_impact['helped'])}")
    print(f"  RAG hurt   (A wrong, B correct): {len(rag_impact['hurt'])}")
    if rag_impact["helped"]:
        print("\n  Top 5 helped cases:")
        for case in rag_impact["helped"][:5]:
            print(f"    {case['specimen_id']}: {case['without_rag']} -> {case['with_rag']} (gt={case['ground_truth']})")

    print("\n--- Confidence Distribution ---\n")
    for cond in ("A", "B", "C"):
        if cond in confidence:
            dist = confidence[cond]
            total = sum(dist.values())
            parts = ", ".join(f"{k}: {v} ({v/max(total,1):.0%})" for k, v in sorted(dist.items()))
            print(f"  {cond}: {parts}")

    print("\n" + "=" * 70)


CONDITIONS_NAMES = {
    "A": "full",
    "B": "no_rag",
    "C": "visual_only",
    "D": "baseline",
}


def save_outputs(
    output_dir: Path,
    accuracy: dict,
    multirank: dict,
    per_family: dict,
    rag_precision: dict,
    convergence: dict,
    rag_impact: dict,
    mcnemar_results: dict,
) -> None:
    """Save metrics as JSON and CSV to output directories."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Main accuracy table
    with open(tables_dir / "accuracy_by_condition.json", "w") as f:
        json.dump(accuracy, f, indent=2)

    # Multi-rank accuracy
    with open(tables_dir / "multirank_accuracy.json", "w") as f:
        json.dump(multirank, f, indent=2)

    # Per-family accuracy
    with open(tables_dir / "per_family_accuracy.json", "w") as f:
        json.dump(per_family, f, indent=2)

    # RAG precision
    with open(tables_dir / "rag_precision.json", "w") as f:
        json.dump(rag_precision, f, indent=2)

    # Convergence analysis
    with open(tables_dir / "convergence_analysis.json", "w") as f:
        json.dump(convergence, f, indent=2)

    # RAG impact cases
    with open(tables_dir / "rag_impact_cases.json", "w") as f:
        json.dump(rag_impact, f, indent=2)

    # McNemar's tests
    with open(tables_dir / "mcnemar_tests.json", "w") as f:
        json.dump(mcnemar_results, f, indent=2)

    # CSV: main accuracy table
    import csv
    with open(tables_dir / "accuracy_by_condition.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "name", "final_top1", "visual_top1", "visual_top3", "visual_top5", "n"])
        for cond in ("A", "B", "C", "D"):
            if cond in accuracy:
                a = accuracy[cond]
                writer.writerow([
                    cond, CONDITIONS_NAMES.get(cond, ""),
                    f"{a['final_top1']:.4f}", f"{a['visual_top1']:.4f}",
                    f"{a['visual_top3']:.4f}", f"{a['visual_top5']:.4f}", a["n"],
                ])

    logger.info("Saved tables to %s", tables_dir)


def main() -> None:
    """Load results and compute all metrics."""
    parser = argparse.ArgumentParser(description="Compute ablation metrics.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "ablation" / "outputs"),
        help="Base output directory containing condition_A/ etc.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Load results for each condition
    all_results: dict[str, list[dict]] = {}
    for condition in ("A", "B", "C", "D"):
        cond_dir = output_dir / f"condition_{condition}"
        if cond_dir.exists():
            results = load_condition_results(cond_dir)
            all_results[condition] = results
            logger.info("Loaded %d results for condition %s", len(results), condition)
        else:
            logger.warning("No results for condition %s (dir: %s)", condition, cond_dir)

    if not all_results:
        logger.error("No results found. Run the experiments first.")
        sys.exit(1)

    # Compute metrics
    accuracy = compute_accuracy_table(all_results)
    multirank = compute_multirank_accuracy(all_results)
    per_family = compute_per_family_accuracy(all_results)
    confidence = compute_confidence_distribution(all_results)

    # RAG precision (from condition A)
    rag_precision = {}
    if "A" in all_results:
        rag_precision = compute_rag_precision(all_results["A"])

    # Convergence analysis (from condition A)
    convergence = {}
    if "A" in all_results:
        convergence = compute_convergence_analysis(all_results["A"])

    # RAG impact cases (A vs B)
    rag_impact = {"helped": [], "hurt": []}
    if "A" in all_results and "B" in all_results:
        rag_impact = find_rag_impact_cases(all_results["A"], all_results["B"])

    # McNemar's tests (family-level)
    mcnemar_results = {}
    pairs = [("A", "B"), ("A", "C"), ("A", "D")]
    for cond_a, cond_b in pairs:
        if cond_a in all_results and cond_b in all_results:
            try:
                mcnemar_results[f"{cond_a}_vs_{cond_b}"] = mcnemar_test(
                    all_results[cond_a], all_results[cond_b], cond_a, cond_b,
                )
            except ImportError:
                logger.warning("scipy not available, skipping McNemar's test")

    # McNemar's at species level (A vs B only — the key comparison)
    if "A" in all_results and "B" in all_results:
        try:
            lookup_b = {r["specimen_id"]: r for r in all_results["B"]}
            a_right_b_wrong = b_right_a_wrong = 0
            for r_a in all_results["A"]:
                r_b = lookup_b.get(r_a["specimen_id"])
                if r_b is None:
                    continue
                gt = extract_ground_truth(r_a)
                gt_sp = gt.get("species", "")
                if not gt_sp:
                    continue
                pred_a = extract_multirank_prediction(r_a, "A").get("species") or ""
                pred_b = extract_multirank_prediction(r_b, "B").get("species") or ""
                a_ok = pred_a.lower() == gt_sp.lower() if pred_a else False
                b_ok = pred_b.lower() == gt_sp.lower() if pred_b else False
                if a_ok and not b_ok:
                    a_right_b_wrong += 1
                elif b_ok and not a_ok:
                    b_right_a_wrong += 1
            n_disc = a_right_b_wrong + b_right_a_wrong
            if n_disc > 0:
                from scipy.stats import chi2 as chi2_dist
                chi2_val = (abs(a_right_b_wrong - b_right_a_wrong) - 1) ** 2 / n_disc
                p_val = 1 - chi2_dist.cdf(chi2_val, df=1)
            else:
                chi2_val, p_val = 0.0, 1.0
            mcnemar_results["A_vs_B_species"] = {
                "chi2": chi2_val, "p_value": p_val, "n_discordant": n_disc,
                "A_right_B_wrong": a_right_b_wrong,
                "B_right_A_wrong": b_right_a_wrong,
            }
        except ImportError:
            pass

    # Output
    print_summary(
        accuracy, multirank, per_family, rag_precision, convergence,
        confidence, rag_impact, mcnemar_results,
    )
    save_outputs(
        output_dir, accuracy, multirank, per_family, rag_precision,
        convergence, rag_impact, mcnemar_results,
    )


if __name__ == "__main__":
    main()
