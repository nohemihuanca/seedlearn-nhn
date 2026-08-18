#!/usr/bin/env python3
"""Mitch-style SimpleShot baseline with NumPy only.

This fits class centroids from all training images in the saved split, then
evaluates one prediction per test individual by averaging that individual's
image embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import f1_score
except Exception:  # pragma: no cover - optional on the cluster
    f1_score = None


DEFAULT_FEATURES = (
    "/home/nh525/project_pi_lsc4/shared/seedlearn/data/embeddings/"
    "2026-01-29_v2026-01-29_12K/features.npz"
)
DEFAULT_SPLIT = (
    "/home/nh525/project_pi_lsc4/shared/seedlearn/data/splits/"
    "2026-01-29_v2026-01-29_12K/family/split_seed42"
)
DEFAULT_OUTPUT = (
    "/home/nh525/seedlearn_runs/bioclip2_simpleshot/results/"
    "family_alltrain_seed42_individual"
)


def l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def load_split(split_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str], dict[str, object]]:
    indices = np.load(split_path.with_suffix(".npz"))
    with open(split_path.with_suffix(".json")) as f:
        meta = json.load(f)
    id_to_label = {int(k): v for k, v in meta["id_to_label"].items()}
    return (
        indices["train_indices"],
        indices["val_indices"],
        indices["test_indices"],
        id_to_label,
        meta,
    )


def labels_to_ids(labels_raw: np.ndarray, label_to_id: dict[str, int], id_to_label: dict[int, str]) -> np.ndarray:
    labels = []
    valid_ids = set(id_to_label)
    for label in labels_raw:
        if isinstance(label, np.generic):
            label = label.item()
        if isinstance(label, (int, np.integer)) and int(label) in valid_ids:
            labels.append(int(label))
            continue
        text = str(label)
        if text in label_to_id:
            labels.append(int(label_to_id[text]))
            continue
        if text.isdigit() and int(text) in valid_ids:
            labels.append(int(text))
            continue
        raise KeyError(f"Could not map label {label!r} to a split class id")
    return np.asarray(labels, dtype=int)


def topk_accuracy(y_true: np.ndarray, topk: np.ndarray) -> float:
    return float(np.mean([truth in row for truth, row in zip(y_true, topk)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-npz", type=Path, default=Path(DEFAULT_FEATURES))
    parser.add_argument("--split-path", type=Path, default=Path(DEFAULT_SPLIT))
    parser.add_argument("--rank", choices=["family", "genus", "species"], default="family")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--max-test-individuals", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_idx, _val_idx, test_idx, id_to_label, split_meta = load_split(args.split_path)
    label_to_id = {v: k for k, v in id_to_label.items()}

    feat = np.load(args.features_npz, allow_pickle=True)
    features = np.asarray(feat["features"], dtype=np.float32)
    individual_ids = np.asarray(feat["individual_ids"])
    label_key = f"{args.rank}_labels"
    if label_key not in feat.files:
        raise KeyError(f"{label_key!r} not found. Available keys: {feat.files}")
    labels = labels_to_ids(np.asarray(feat[label_key]), label_to_id, id_to_label)

    features = l2_normalize(features)

    train_labels = labels[train_idx]
    class_ids = np.array(sorted(np.unique(train_labels)), dtype=int)

    centroids = []
    for class_id in class_ids:
        class_features = features[train_idx[train_labels == class_id]]
        centroid = class_features.mean(axis=0, keepdims=True)
        centroids.append(l2_normalize(centroid)[0])
    centroids = np.vstack(centroids)

    groups: dict[str, list[int]] = defaultdict(list)
    for idx in test_idx:
        groups[str(individual_ids[idx])].append(int(idx))

    individual_keys = sorted(groups)
    if args.max_test_individuals is not None:
        rng = np.random.default_rng(args.seed)
        individual_keys = sorted(rng.choice(individual_keys, size=args.max_test_individuals, replace=False))

    y_true = []
    y_pred = []
    top5 = []
    rows = []
    mixed_label_individuals = 0

    for individual_id in individual_keys:
        idxs = np.asarray(groups[individual_id], dtype=int)
        label_counts = Counter(labels[idxs].tolist())
        truth = label_counts.most_common(1)[0][0]
        if len(label_counts) > 1:
            mixed_label_individuals += 1

        pooled = l2_normalize(features[idxs].mean(axis=0, keepdims=True))[0]
        scores = centroids @ pooled
        order = np.argsort(scores)[::-1]
        pred = int(class_ids[order[0]])
        top_ids = [int(class_ids[i]) for i in order[:5]]

        y_true.append(int(truth))
        y_pred.append(pred)
        top5.append(top_ids)
        rows.append(
            {
                "individual_id": individual_id,
                "num_images": int(len(idxs)),
                "target_label": id_to_label[int(truth)],
                "prediction_label": id_to_label[pred],
                "is_correct": bool(int(truth) == pred),
                "top5_labels": "|".join(id_to_label[i] for i in top_ids),
            }
        )

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    top5_arr = np.asarray(top5, dtype=int)

    metrics = {
        "accuracy": float(np.mean(y_true_arr == y_pred_arr)),
        "top5_accuracy": topk_accuracy(y_true_arr, top5_arr),
        "num_test_individuals": int(len(individual_keys)),
        "num_test_images": int(sum(len(groups[k]) for k in individual_keys)),
        "avg_test_images_per_individual": float(np.mean([len(groups[k]) for k in individual_keys])),
        "num_train_images": int(len(train_idx)),
        "num_train_individuals": int(len(set(map(str, individual_ids[train_idx])))),
        "num_classes_in_split": int(split_meta["num_classes"]),
        "num_classes_in_train": int(len(class_ids)),
        "mixed_label_individuals": int(mixed_label_individuals),
        "rank": args.rank,
        "classifier": "all_train_centroids",
        "test_unit": "individual_mean_embedding",
        "split_path": str(args.split_path),
        "features_npz": str(args.features_npz),
    }
    if f1_score is not None:
        metrics.update(
            {
                "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
                "micro_f1": float(f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)),
                "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(args.output_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(metrics, indent=2))
    print("Wrote", args.output_dir / "metrics.json")
    print("Wrote", args.output_dir / "predictions.csv")


if __name__ == "__main__":
    main()
