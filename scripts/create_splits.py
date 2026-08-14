#!/usr/bin/env python3
"""Create stratified train/val/test splits for few-shot learning experiments.

This script creates multiple random splits with different seeds for
robust evaluation across experiments.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from seedlearn.data.catalog import load_dataset
from seedlearn.data.constants import DEFAULT_CATALOG, SHARED_SPLITS, get_catalog_version
from seedlearn.data.splits import (
    create_individual_split,
    create_stratified_split,
    save_split,
)


def _resolve_path(path_input):
    """Handle both Path objects and strings from CLI."""
    if isinstance(path_input, Path):
        return path_input
    return Path(path_input).resolve()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="Path to species catalog CSV")
    parser.add_argument("--rank", choices=["family", "genus", "species"], default="species", help="Taxonomic rank")
    parser.add_argument("--output-dir", type=_resolve_path, default=None, help="Directory to save splits")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Training proportion")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation proportion")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test proportion")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of random splits")
    parser.add_argument("--start-seed", type=int, default=42, help="Starting random seed")
    parser.add_argument(
        "--split-type",
        choices=["stratified", "individual"],
        default="stratified",
        help=(
            "Split strategy: 'stratified' splits at image level (fast, but allows "
            "data leakage across images of the same plant); 'individual' groups by "
            "ID_YPS so all images of the same plant stay in the same partition "
            "(prevents leakage, required for honest evaluation)."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    """Main function."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.output_dir is None:
        version = get_catalog_version(Path(args.catalog))
        args.output_dir = SHARED_SPLITS / version
        logging.info("Auto-generated output directory: %s", args.output_dir)

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        logging.error("Split ratios must sum to 1.0, got %.6f", total_ratio)
        return 1

    logging.info("Loading dataset from catalog...")
    records, label_to_id = load_dataset(catalog_path=Path(args.catalog), rank=args.rank)
    logging.info("Loaded %d images across %d classes", len(records), len(label_to_id))

    rank_output_dir = args.output_dir / args.rank
    rank_output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Creating %d splits with seeds %d to %d", args.num_seeds, args.start_seed, args.start_seed + args.num_seeds - 1)

    if args.split_type == "individual":
        # Verify all records have individual_id populated
        missing_ids = sum(1 for r in records if not r.individual_id)
        if missing_ids:
            logging.error(
                "%d records have no individual_id (ID_YPS). "
                "Individual splits require this field in the catalog.",
                missing_ids,
            )
            return 1
        logging.info("Using individual-level splits (grouped by ID_YPS)")

    split = None
    for i in range(args.num_seeds):
        seed = args.start_seed + i
        logging.info("Creating split %d/%d (seed=%d, type=%s)", i + 1, args.num_seeds, seed, args.split_type)

        if args.split_type == "individual":
            split = create_individual_split(
                records=records,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                random_seed=seed,
            )
        else:
            split = create_stratified_split(
                records=records,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                random_seed=seed,
            )
        save_split(split, rank_output_dir / f"split_seed{seed}")

    logging.info("Successfully created %d splits in %s", args.num_seeds, rank_output_dir)

    # Display warnings and feasibility info from the last split
    if split is not None:
        split_info = split.split_info

        if "class_warnings" in split_info and split_info["class_warnings"]:
            num_warnings = len(split_info["class_warnings"])
            logging.warning("=" * 60)
            logging.warning("FOUND %d CLASSES WITH INSUFFICIENT SAMPLES:", num_warnings)
            logging.warning("=" * 60)
            for warning in split_info["class_warnings"]:
                logging.warning(
                    "  - %-30s train: %3d, val: %3d, test: %3d",
                    warning["class_name"], warning["train_samples"], warning["val_samples"], warning["test_samples"],
                )

        if "feasible_k_shots" in split_info:
            feasible = split_info["feasible_k_shots"]
            logging.info("=" * 60)
            logging.info("K-SHOT FEASIBILITY INFORMATION:")
            logging.info("  Guaranteed k-shot (all classes): %d", feasible["guaranteed"])
            logging.info("  Recommended k-shot values: %s", feasible["recommended"])
            logging.info("=" * 60)

        if "class_statistics" in split_info:
            train_stats = split_info["class_statistics"]["train"]
            logging.info("Training set class distribution:")
            logging.info("  Min samples per class: %d", train_stats["min_samples"])
            logging.info("  Max samples per class: %d", train_stats["max_samples"])
            logging.info("  Mean samples per class: %.1f", train_stats["mean_samples"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
