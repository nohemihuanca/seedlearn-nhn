#!/usr/bin/env python3
"""Run a single SimpleShot few-shot learning experiment.

This script runs SimpleShot with a specified k-shot value on a given split
and evaluates on test data.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from seedlearn.data.catalog import load_dataset
from seedlearn.data.constants import DEFAULT_CATALOG, SHARED_EMBEDDINGS, SHARED_EXPERIMENTS, get_catalog_version
from seedlearn.data.splits import load_split, create_fixed_support_set, validate_k_shot_feasibility
from seedlearn.clip.cache import CachedFeatureExtractor
from seedlearn.clip.metrics import compute_metrics, save_evaluation_results, print_results_summary
from seedlearn.clip.simpleshot import SimpleShot


def _resolve_path(path_input):
    """Handle both Path objects and strings from CLI."""
    if isinstance(path_input, Path):
        return path_input
    return Path(path_input).resolve()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="Path to species catalog CSV")
    parser.add_argument("--rank", choices=["family", "genus", "species"], required=True, help="Taxonomic rank")
    parser.add_argument("--split-path", type=Path, required=True, help="Path to split file (without extension)")
    parser.add_argument("--cache-dir", type=_resolve_path, default=None, help="Directory containing cached features")
    parser.add_argument("--cache-name", type=str, default=None, help="Name of cached features")
    parser.add_argument("--k-shot", type=int, required=True, help="Number of support examples per class")
    parser.add_argument("--output-dir", type=_resolve_path, default=None, help="Directory to save results")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--support-seed", type=int, default=42, help="Random seed for support set sampling")
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

    start_time = time.time()
    catalog_path = Path(args.catalog)
    version = get_catalog_version(catalog_path)

    if args.cache_dir is None:
        args.cache_dir = SHARED_EMBEDDINGS / version

    split_name = args.split_path.stem

    if args.output_dir is None:
        args.output_dir = SHARED_EXPERIMENTS / version / "results" / args.rank / f"{args.k_shot}_shot" / split_name

    device = torch.device(args.device)
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    # Load dataset
    records, label_to_id = load_dataset(catalog_path=args.catalog, rank=args.rank)
    logging.info("Loaded %d images across %d classes", len(records), len(label_to_id))

    # Load split
    split = load_split(args.split_path)
    train_records = [records[i] for i in split.train_indices]
    test_records = [records[i] for i in split.test_indices]

    # Validate k-shot
    is_valid, validation_messages = validate_k_shot_feasibility(split.split_info, args.k_shot, strict=True)
    for msg in validation_messages:
        if msg.startswith("ERROR"):
            logging.error(msg)
        elif msg.startswith("WARNING"):
            logging.warning(msg)
        else:
            logging.info(msg)

    if not is_valid:
        logging.error("K-shot validation failed. Aborting experiment.")
        sys.exit(1)

    # Load cached features
    cache_name = args.cache_name or f"{args.rank}_features"
    extractor = CachedFeatureExtractor(cache_dir=args.cache_dir, device=device)
    features, labels, image_paths = extractor.load_cached_features(cache_name)

    path_to_idx = {str(path): idx for idx, path in enumerate(image_paths)}
    train_indices = [path_to_idx[str(r.image_path)] for r in train_records]
    test_indices = [path_to_idx[str(r.image_path)] for r in test_records]
    train_features = features[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]

    # Create support set
    support_indices, support_labels = create_fixed_support_set(
        records=train_records, k_shot=args.k_shot, random_seed=args.support_seed,
    )
    support_feature_indices = [path_to_idx[str(train_records[i].image_path)] for i in support_indices]
    support_features = features[support_feature_indices]
    support_labels_array = labels[support_feature_indices]

    # Build support set metadata
    support_set_info = {
        "k_shot": args.k_shot,
        "support_seed": args.support_seed,
        "total_support_samples": len(support_labels),
        "num_classes": split.num_classes,
        "support_images": [],
        "per_class_support": {},
    }
    for idx, label_id in zip(support_indices, support_labels):
        record = train_records[idx]
        support_set_info["support_images"].append({
            "image_path": str(record.image_path),
            "label": record.label,
            "label_id": int(label_id),
        })
        if record.label not in support_set_info["per_class_support"]:
            support_set_info["per_class_support"][record.label] = []
        support_set_info["per_class_support"][record.label].append(str(record.image_path))

    # Train and predict
    classifier = SimpleShot(device=device)
    classifier.fit(support_features, support_labels_array)
    test_predictions = classifier.predict(test_features)
    test_probabilities = classifier.predict_proba(test_features)

    # Create predictions DataFrame
    predictions_data = []
    for i, record in enumerate(test_records):
        predictions_data.append({
            "image_path": str(record.image_path),
            "target_label": record.label,
            "target_id": int(test_labels[i]),
            "prediction_label": split.id_to_label[int(test_predictions[i])],
            "prediction_id": int(test_predictions[i]),
            "is_correct": bool(test_labels[i] == test_predictions[i]),
            "support_size": args.k_shot,
            "split_seed": split_name.split("_")[-1] if "_" in split_name else split_name,
        })
    predictions_df = pd.DataFrame(predictions_data)

    # Compute metrics
    label_names = [split.id_to_label[i] for i in range(split.num_classes)]
    results = compute_metrics(y_true=test_labels, y_pred=test_predictions, y_proba=test_probabilities, label_names=label_names)
    print_results_summary(results, f"SimpleShot {args.k_shot}-shot ({args.rank})")

    # Save results
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = "unknown"

    split_seed = split_name.split("_")[-1] if "_" in split_name else split_name

    experiment_info = {
        "timestamp": datetime.now().isoformat(),
        "catalog": str(args.catalog),
        "rank": args.rank,
        "k_shot": args.k_shot,
        "split_path": str(args.split_path),
        "split_seed": split_seed,
        "support_seed": args.support_seed,
        "device": str(device),
        "cache_dir": str(args.cache_dir),
        "cache_name": cache_name,
        "git_hash": git_hash,
        "num_train_samples": len(train_records),
        "num_test_samples": len(test_records),
        "num_classes": len(label_to_id),
    }

    save_evaluation_results(results=results, output_dir=output_dir, label_names=label_names, experiment_info=experiment_info)

    with open(output_dir / "experiment_info.json", "w") as f:
        json.dump(experiment_info, f, indent=2)

    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    with open(output_dir / "support_set.json", "w") as f:
        json.dump(support_set_info, f, indent=2)

    logging.info("Total runtime: %.2f seconds", time.time() - start_time)
    logging.info("Results saved to %s", output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
