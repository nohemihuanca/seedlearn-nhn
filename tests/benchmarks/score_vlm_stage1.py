#!/usr/bin/env python3
"""Score VLM Stage 1 benchmark results against ground truth.

Computes per-trait accuracy, per-specimen scorecards, confusion matrices,
and optional multi-vs-single image comparison metrics.

Usage examples:

    # Score a single result directory:
    python tests/benchmarks/score_vlm_stage1.py \\
        --results data/benchmarks/qwen3-vl-32b/ \\
        --ground-truth tests/benchmarks/configs/ground_truth.csv \\
        --output data/benchmarks/scores/qwen3-vl-32b/

    # Score multiple result directories:
    python tests/benchmarks/score_vlm_stage1.py \\
        --results data/benchmarks/run1/ data/benchmarks/run2/ \\
        --ground-truth tests/benchmarks/configs/ground_truth.csv \\
        --output data/benchmarks/scores/combined/

    # Score a legacy JSON results file:
    python tests/benchmarks/score_vlm_stage1.py \\
        --results-file data/benchmarks/legacy_results.json \\
        --ground-truth tests/benchmarks/configs/ground_truth.csv \\
        --output data/benchmarks/scores/legacy/

    # Score single-image mode only:
    python tests/benchmarks/score_vlm_stage1.py \\
        --results data/benchmarks/qwen3-vl-32b/ \\
        --ground-truth tests/benchmarks/configs/ground_truth.csv \\
        --output data/benchmarks/scores/single/ \\
        --mode single
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path so scoring package is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tests.benchmarks.scoring.loader import (  # noqa: E402
    TRAIT_COLUMNS,
    ResultEntry,
    _parse_legacy_result_json,
    load_ground_truth,
    load_result_dir,
)
from tests.benchmarks.scoring.metrics import (  # noqa: E402
    compute_confusion_matrices,
    compute_multi_vs_single,
    compute_specimen_scores,
    compute_trait_scores,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Score VLM Stage 1 benchmark results against ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for detailed usage examples.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        help="One or more result directories to score.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        help="Legacy JSON results file (alternative to --results).",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to ground truth CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for score files.",
    )
    parser.add_argument(
        "--mode",
        choices=["multi", "single"],
        default="multi",
        help="Inference mode to score (default: multi).",
    )
    return parser


def _collect_results(
    args: argparse.Namespace,
) -> dict[str, ResultEntry]:
    """Load results from --results dirs or --results-file.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Merged dict of specimen_key to ResultEntry.

    Raises:
        SystemExit: If neither --results nor --results-file is provided.
    """
    all_results: dict[str, ResultEntry] = {}

    if args.results:
        for result_path in args.results:
            _, results = load_result_dir(result_path, mode=args.mode)
            all_results.update(results)
    elif args.results_file:
        all_results = _parse_legacy_result_json(args.results_file)
    else:
        logger.error("Must provide --results or --results-file")
        sys.exit(1)

    return all_results


def _write_trait_csv(
    trait_scores: list[Any],
    output_dir: Path,
) -> Path:
    """Write per-trait accuracy CSV.

    Args:
        trait_scores: List of TraitScore objects.
        output_dir: Directory to write into.

    Returns:
        Path to the written CSV file.
    """
    path = output_dir / "per_trait_accuracy.csv"
    fieldnames = list(trait_scores[0].to_dict().keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ts in trait_scores:
            writer.writerow(ts.to_dict())
    return path


def _write_specimen_csv(
    specimen_scores: list[Any],
    output_dir: Path,
) -> Path:
    """Write per-specimen scorecard CSV.

    Args:
        specimen_scores: List of SpecimenScore objects.
        output_dir: Directory to write into.

    Returns:
        Path to the written CSV file.
    """
    path = output_dir / "per_specimen_scorecard.csv"
    # Flatten details into columns: {trait}_predicted, {trait}_gt, {trait}_result
    fieldnames = [
        "specimen_key",
        "family",
        "scientific_name",
        "multi_label_count",
        "traits_correct",
        "traits_incorrect",
        "traits_abstained",
        "traits_no_gt",
        "accuracy",
    ]
    for trait in TRAIT_COLUMNS:
        fieldnames.extend(
            [
                f"{trait}_predicted",
                f"{trait}_gt",
                f"{trait}_result",
            ]
        )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ss in specimen_scores:
            row: dict[str, Any] = {
                "specimen_key": ss.specimen_key,
                "family": ss.family,
                "scientific_name": ss.scientific_name,
                "multi_label_count": ss.multi_label_count,
                "traits_correct": ss.n_correct,
                "traits_incorrect": ss.n_incorrect,
                "traits_abstained": ss.n_abstained,
                "traits_no_gt": ss.n_no_gt,
                "accuracy": round(ss.accuracy, 4),
            }
            for trait in TRAIT_COLUMNS:
                detail = ss.details.get(trait, {})
                row[f"{trait}_predicted"] = detail.get("predicted", "")
                row[f"{trait}_gt"] = detail.get("ground_truth", "")
                row[f"{trait}_result"] = detail.get("result", "")
            writer.writerow(row)
    return path


def _write_confusion_matrices(
    matrices: dict[str, dict[str, dict[str, int]]],
    output_dir: Path,
) -> Path:
    """Write per-trait confusion matrix JSON files.

    Args:
        matrices: Dict of trait to confusion matrix.
        output_dir: Directory to write into.

    Returns:
        Path to the confusion_matrices subdirectory.
    """
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    for trait, matrix in matrices.items():
        path = cm_dir / f"{trait}.json"
        with open(path, "w") as f:
            json.dump(matrix, f, indent=2)
    return cm_dir


def _write_multi_vs_single_csv(
    comparison: dict[str, dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write multi-vs-single comparison CSV.

    Args:
        comparison: Dict of trait to comparison metrics.
        output_dir: Directory to write into.

    Returns:
        Path to the written CSV file.
    """
    path = output_dir / "multi_vs_single.csv"
    fieldnames = [
        "trait",
        "multi_accuracy",
        "single_mean_accuracy",
        "majority_vote_accuracy",
        "mean_consistency",
        "resolution_effect",
        "n_specimens",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trait, metrics in comparison.items():
            row = {"trait": trait, **metrics}
            writer.writerow(row)
    return path


def _write_summary_json(
    trait_scores: list[Any],
    specimen_scores: list[Any],
    confusion_matrices: dict[str, dict[str, dict[str, int]]],
    multi_vs_single: dict[str, dict[str, Any]] | None,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    """Write summary JSON with all metrics.

    Args:
        trait_scores: List of TraitScore objects.
        specimen_scores: List of SpecimenScore objects.
        confusion_matrices: Per-trait confusion matrices.
        multi_vs_single: Optional multi-vs-single comparison data.
        output_dir: Directory to write into.
        args: CLI arguments for metadata.

    Returns:
        Path to the written JSON file.
    """
    summary: dict[str, Any] = {
        "metadata": {
            "mode": args.mode,
            "ground_truth": str(args.ground_truth),
            "result_sources": (
                [str(p) for p in args.results]
                if args.results
                else [str(args.results_file)]
            ),
            "n_specimens_gt": len({ss.specimen_key for ss in specimen_scores}),
            "n_specimens_scored": sum(
                1 for ss in specimen_scores if ss.n_correct + ss.n_incorrect > 0
            ),
        },
        "per_trait": [ts.to_dict() for ts in trait_scores],
        "per_specimen": [ss.to_dict() for ss in specimen_scores],
        "confusion_matrices": confusion_matrices,
    }
    if multi_vs_single:
        summary["multi_vs_single"] = multi_vs_single

    path = output_dir / "summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path


def _print_summary_table(
    trait_scores: list[Any],
    multi_vs_single: dict[str, dict[str, Any]] | None,
) -> None:
    """Print a summary table to stdout.

    Args:
        trait_scores: List of TraitScore objects.
        multi_vs_single: Optional multi-vs-single comparison data.
    """
    header = f"{'Trait':<20} {'Acc':>6} {'Strict':>7} {'Abst%':>6} {'N':>5} {'Correct':>8} {'Miss':>5}"
    print("\n" + "=" * len(header))
    print("Per-Trait Accuracy")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for ts in trait_scores:
        print(
            f"{ts.trait:<20} {ts.accuracy:>6.1%} {ts.strict_accuracy:>7.1%} "
            f"{ts.abstention_rate:>6.1%} {ts.n_total:>5} {ts.n_correct:>8} {ts.n_mismatch:>5}"
        )

    # Overall
    total_correct = sum(ts.n_correct for ts in trait_scores)
    total_answered = sum(ts.n_answered for ts in trait_scores)
    total_scored = sum(ts.n_total for ts in trait_scores)
    overall_acc = total_correct / total_answered if total_answered else 0.0
    overall_strict = total_correct / total_scored if total_scored else 0.0
    print("-" * len(header))
    print(
        f"{'OVERALL':<20} {overall_acc:>6.1%} {overall_strict:>7.1%} {'':>6} {total_scored:>5}"
    )
    print()

    if multi_vs_single:
        header_mv = (
            f"{'Trait':<20} {'Multi':>6} {'Single':>7} {'MajVote':>8} "
            f"{'Consist':>8} {'Delta':>6}"
        )
        print("=" * len(header_mv))
        print("Multi vs Single Image Comparison")
        print("=" * len(header_mv))
        print(header_mv)
        print("-" * len(header_mv))
        for trait, m in multi_vs_single.items():
            print(
                f"{trait:<20} {m['multi_accuracy']:>6.1%} "
                f"{m['single_mean_accuracy']:>7.1%} "
                f"{m['majority_vote_accuracy']:>8.1%} "
                f"{m['mean_consistency']:>8.1%} "
                f"{m['resolution_effect']:>+6.1%}"
            )
        print()


def main(argv: list[str] | None = None) -> None:
    """Run the scorer CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Load ground truth
    gt = load_ground_truth(args.ground_truth)
    logger.info("Ground truth: %d specimens", len(gt))

    # Load results
    results = _collect_results(args)
    logger.info("Results: %d specimens", len(results))

    if not results:
        logger.error("No results loaded. Check --results or --results-file paths.")
        sys.exit(1)

    # Compute metrics
    trait_scores = compute_trait_scores(gt, results)
    specimen_scores = compute_specimen_scores(gt, results)
    confusion_matrices = compute_confusion_matrices(gt, results)

    # Detect multi/single subdirectories for comparison
    multi_vs_single: dict[str, dict[str, Any]] | None = None
    if args.results:
        for result_path in args.results:
            multi_dir = result_path / "multi"
            single_dir = result_path / "single"
            if multi_dir.exists() and single_dir.exists():
                _, multi_results_full = load_result_dir(result_path, mode="multi")
                _, single_results_full = load_result_dir(result_path, mode="single")
                if multi_results_full and single_results_full:
                    multi_vs_single = compute_multi_vs_single(
                        gt, multi_results_full, single_results_full
                    )
                break  # Use first dir with both modes

    # Write outputs
    args.output.mkdir(parents=True, exist_ok=True)

    _write_summary_json(
        trait_scores,
        specimen_scores,
        confusion_matrices,
        multi_vs_single,
        args.output,
        args,
    )
    _write_trait_csv(trait_scores, args.output)
    _write_specimen_csv(specimen_scores, args.output)
    _write_confusion_matrices(confusion_matrices, args.output)

    if multi_vs_single:
        _write_multi_vs_single_csv(multi_vs_single, args.output)

    # HTML report
    from tests.benchmarks.scoring.report_html import generate_score_report_html

    summary_path = args.output / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    html = generate_score_report_html(summary, confusion_matrices)
    html_path = args.output / "report.html"
    with open(html_path, "w") as f:
        f.write(html)
    logger.info("HTML report: %s", html_path)

    # Print summary
    _print_summary_table(trait_scores, multi_vs_single)

    logger.info("Output written to %s", args.output)


if __name__ == "__main__":
    main()
