#!/usr/bin/env python3
"""Batch orchestration for SimpleShot experiments.

Runs multiple SimpleShot experiments covering all combinations of split seeds
and k-shot values. Includes pre-validation, progress tracking, error handling,
summary reporting, and optional automatic visual report generation.

Example:
    python scripts/run_experiments.py \\
        --rank family \\
        --k-shots 1 5 10 \\
        --split-seeds 42 43 44 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from seedlearn.data.constants import DEFAULT_CATALOG, SHARED_EXPERIMENTS, SHARED_SPLITS, get_catalog_version
from seedlearn.data.splits import load_split, validate_k_shot_feasibility


def _resolve_path(path_input):
    if isinstance(path_input, Path):
        return path_input
    return Path(path_input).resolve()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="Path to species catalog CSV")
    parser.add_argument("--rank", choices=["family", "genus", "species"], required=True, help="Taxonomic rank")
    parser.add_argument("--k-shots", type=int, nargs="+", required=True, help="K-shot values to run")
    parser.add_argument("--split-seeds", type=int, nargs="+", default=None, help="Split seeds (default: auto-detect)")
    parser.add_argument("--splits-dir", type=_resolve_path, default=None, help="Directory containing splits")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--support-seed", type=int, default=42, help="Random seed for support set sampling")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue batch if single experiment fails")
    parser.add_argument("--skip-reports", action="store_true", help="Skip automatic report generation")
    parser.add_argument("--baseline-blind", type=float, default=None, help="Blind baseline accuracy for reports")
    parser.add_argument("--baseline-closed", type=float, default=None, help="Closed-set baseline accuracy for reports")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def discover_split_seeds(splits_dir: Path, rank: str) -> list[int]:
    """Auto-discover available split seeds from splits directory."""
    rank_dir = splits_dir / rank
    if not rank_dir.exists():
        raise FileNotFoundError(f"Splits directory not found: {rank_dir}")

    split_files = list(rank_dir.glob("split_seed*.json"))
    if not split_files:
        raise ValueError(f"No split files found in {rank_dir}")

    seeds = []
    for split_file in split_files:
        match = re.search(r"split_seed(\d+)\.json", split_file.name)
        if match:
            seeds.append(int(match.group(1)))

    seeds.sort()
    logging.info("Discovered %d split seeds: %s", len(seeds), seeds)
    return seeds


def validate_all_splits(
    splits_dir: Path, rank: str, split_seeds: list[int], k_shots: list[int],
) -> tuple[bool, dict[int, dict[str, Any]]]:
    """Pre-validate all splits before running experiments."""
    logging.info("Pre-validating all split and k-shot combinations...")
    validation_results: dict[int, dict] = {}
    all_valid = True
    max_k = max(k_shots)

    for seed in split_seeds:
        split_path = splits_dir / rank / f"split_seed{seed}"
        try:
            split = load_split(split_path)
            validation_results[seed] = {}
            is_valid, messages = validate_k_shot_feasibility(split.split_info, k_shot=max_k, strict=True)
            for k in k_shots:
                k_valid = is_valid if k == max_k else True
                validation_results[seed][k] = (k_valid, messages if k == max_k else [])
                if not k_valid:
                    all_valid = False
        except Exception as e:
            logging.error("Failed to load split %s: %s", split_path, e)
            all_valid = False
            validation_results[seed] = {k: (False, [str(e)]) for k in k_shots}

    return all_valid, validation_results


def run_single_experiment(
    catalog: Path, rank: str, split_path: Path, k_shot: int, device: str, support_seed: int,
) -> tuple[bool, str, dict | None]:
    """Run a single SimpleShot experiment as a subprocess."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_simpleshot.py"),
        "--catalog", str(catalog),
        "--rank", rank,
        "--split-path", str(split_path),
        "--k-shot", str(k_shot),
        "--device", device,
        "--support-seed", str(support_seed),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)

        output_dir = None
        for line in result.stdout.split("\n"):
            if "Results saved to" in line:
                output_dir = Path(line.split("Results saved to")[-1].strip())
                break

        if not output_dir:
            split_name = Path(split_path).name
            results_base = Path(split_path).parent.parent.parent / "results"
            output_dir = results_base / rank / f"{k_shot}_shot" / split_name

        experiment_info: dict[str, Any] = {}
        if output_dir and output_dir.exists():
            metrics_path = output_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    experiment_info = json.load(f)
            experiment_info["output_dir"] = str(output_dir)
            if "accuracy" in experiment_info:
                experiment_info["top1_accuracy"] = experiment_info["accuracy"]

        return True, "", experiment_info if experiment_info else None

    except subprocess.TimeoutExpired:
        return False, "Timeout (>10 minutes)", None
    except subprocess.CalledProcessError as e:
        error_lines = e.stderr.split("\n")[-5:]
        return False, "\n".join(error_lines), None
    except Exception as e:
        return False, str(e), None


def main() -> int:
    """Main function."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    start_time = time.time()
    version = get_catalog_version(Path(args.catalog))

    if args.splits_dir is None:
        args.splits_dir = SHARED_SPLITS / version

    if args.split_seeds is None:
        try:
            args.split_seeds = discover_split_seeds(args.splits_dir, args.rank)
        except (FileNotFoundError, ValueError) as e:
            logging.error("Failed to discover split seeds: %s", e)
            return 1

    args.k_shots.sort()

    logging.info("=" * 60)
    logging.info("BATCH EXPERIMENT CONFIGURATION")
    logging.info("  Rank: %s", args.rank)
    logging.info("  K-shots: %s", args.k_shots)
    logging.info("  Split seeds: %s", args.split_seeds)
    logging.info("  Total experiments: %d", len(args.split_seeds) * len(args.k_shots))
    logging.info("=" * 60)

    all_valid, validation_results = validate_all_splits(args.splits_dir, args.rank, args.split_seeds, args.k_shots)

    if not all_valid and not args.continue_on_error:
        logging.error("Pre-validation failed! Aborting.")
        return 1

    experiments = []
    for seed in args.split_seeds:
        for k in args.k_shots:
            if seed in validation_results and k in validation_results[seed]:
                is_valid, _ = validation_results[seed][k]
                if is_valid or args.continue_on_error:
                    experiments.append({"seed": seed, "k_shot": k, "split_path": args.splits_dir / args.rank / f"split_seed{seed}"})

    logging.info("Running %d experiments...", len(experiments))

    results = []
    iterator = tqdm(experiments, desc="Experiments") if HAS_TQDM else experiments

    for exp in iterator:
        if HAS_TQDM:
            iterator.set_description(f"seed{exp['seed']} k={exp['k_shot']}")

        success, error, exp_info = run_single_experiment(
            catalog=args.catalog, rank=args.rank, split_path=exp["split_path"],
            k_shot=exp["k_shot"], device=args.device, support_seed=args.support_seed,
        )

        result = {
            "experiment_name": f"{args.rank}_{exp['k_shot']}shot_seed{exp['seed']}",
            "split_seed": exp["seed"],
            "k_shot": exp["k_shot"],
            "success": success,
            "error": error if not success else None,
            "timestamp": datetime.now().isoformat(),
        }
        if exp_info:
            result["accuracy"] = exp_info.get("top1_accuracy")
            result["output_dir"] = exp_info.get("output_dir")

        results.append(result)

        if not success:
            logging.error("  Failed: %s", error)
            if not args.continue_on_error:
                break

    # Summary
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    failed = total - successful

    logging.info("=" * 60)
    logging.info("BATCH SUMMARY")
    logging.info("  Total: %d, Successful: %d, Failed: %d", total, successful, failed)

    if successful > 0:
        accuracies = [r["accuracy"] for r in results if r["success"] and r.get("accuracy") is not None]
        if accuracies:
            logging.info("  Accuracy range: %.4f - %.4f", min(accuracies), max(accuracies))
            logging.info("  Mean accuracy: %.4f +/- %.4f", np.mean(accuracies), np.std(accuracies))

    # Save summary
    summary_path = SHARED_EXPERIMENTS / version / f"batch_summary_{args.rank}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "config": {
            "catalog": str(args.catalog),
            "rank": args.rank,
            "k_shots": args.k_shots,
            "split_seeds": args.split_seeds,
            "device": args.device,
            "support_seed": args.support_seed,
        },
        "summary": {
            "total": total,
            "successful": successful,
            "failed": failed,
            "runtime_seconds": time.time() - start_time,
        },
        "results": results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("Saved batch summary to %s", summary_path)
    logging.info("Total runtime: %.1f seconds", time.time() - start_time)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
