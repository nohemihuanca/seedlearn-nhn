"""Evaluation metrics for few-shot learning experiments.

This module provides utilities for computing and reporting classification
metrics, with special attention to imbalanced datasets.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)


@dataclass
class EvaluationResult:
    """Container for evaluation results.

    Attributes:
        accuracy: Top-1 accuracy.
        macro_f1: Macro-averaged F1 score.
        micro_f1: Micro-averaged F1 score.
        weighted_f1: Weighted F1 score.
        top5_accuracy: Top-5 accuracy (if applicable).
        per_class_metrics: Per-class precision, recall, F1.
        confusion_matrix: Confusion matrix.
        num_samples: Number of samples evaluated.
        num_classes: Number of classes.
    """

    accuracy: float
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    top5_accuracy: float | None
    per_class_metrics: dict[str, dict[str, float]]
    confusion_matrix: npt.NDArray[np.int64]
    num_samples: int
    num_classes: int


def compute_metrics(
    y_true: npt.NDArray[np.int64],
    y_pred: npt.NDArray[np.int64],
    y_proba: npt.NDArray[np.float32] | None = None,
    label_names: Sequence[str] | None = None,
) -> EvaluationResult:
    """Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels of shape (n_samples,).
        y_pred: Predicted labels of shape (n_samples,).
        y_proba: Predicted probabilities of shape (n_samples, n_classes), optional.
        label_names: Names of labels for reporting, optional.

    Returns:
        EvaluationResult object containing all metrics.
    """
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    num_classes = len(unique_labels)

    accuracy = float(accuracy_score(y_true, y_pred))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    top5_accuracy = None
    if y_proba is not None and num_classes >= 5:
        top5_accuracy = float(top_k_accuracy_score(y_true, y_proba, k=5))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_labels, zero_division=0
    )

    if label_names is None:
        label_names = [str(label) for label in unique_labels]

    per_class_metrics = {}
    for idx, label in enumerate(unique_labels):
        label_name = label_names[idx] if idx < len(label_names) else str(label)
        per_class_metrics[label_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    return EvaluationResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        weighted_f1=weighted_f1,
        top5_accuracy=top5_accuracy,
        per_class_metrics=per_class_metrics,
        confusion_matrix=cm,
        num_samples=len(y_true),
        num_classes=num_classes,
    )


def save_evaluation_results(
    results: EvaluationResult,
    output_dir: Path,
    label_names: Sequence[str] | None = None,
    experiment_info: dict | None = None,
) -> None:
    """Save evaluation results to disk.

    Args:
        results: EvaluationResult object.
        output_dir: Directory to save results.
        label_names: Names of labels for the confusion matrix.
        experiment_info: Additional experiment metadata to save.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "accuracy": results.accuracy,
        "macro_f1": results.macro_f1,
        "micro_f1": results.micro_f1,
        "weighted_f1": results.weighted_f1,
        "top5_accuracy": results.top5_accuracy,
        "num_samples": results.num_samples,
        "num_classes": results.num_classes,
    }

    if experiment_info:
        summary.update(experiment_info)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("Saved metrics to %s", metrics_path)

    per_class_df = pd.DataFrame(results.per_class_metrics).T
    per_class_path = output_dir / "per_class_metrics.csv"
    per_class_df.to_csv(per_class_path)
    logging.info("Saved per-class metrics to %s", per_class_path)

    if label_names is None:
        label_names = [f"class_{i}" for i in range(results.num_classes)]

    cm_df = pd.DataFrame(
        results.confusion_matrix,
        index=label_names,
        columns=label_names,
    )
    cm_path = output_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    logging.info("Saved confusion matrix to %s", cm_path)


def print_results_summary(
    results: EvaluationResult,
    experiment_name: str = "Experiment",
) -> None:
    """Print a summary of evaluation results.

    Args:
        results: EvaluationResult object.
        experiment_name: Name of the experiment for display.
    """
    logging.info("=" * 60)
    logging.info("%s Results", experiment_name)
    logging.info("=" * 60)
    logging.info("Samples: %d", results.num_samples)
    logging.info("Classes: %d", results.num_classes)
    logging.info("-" * 60)
    logging.info("Top-1 Accuracy:  %.4f", results.accuracy)
    if results.top5_accuracy is not None:
        logging.info("Top-5 Accuracy:  %.4f", results.top5_accuracy)
    logging.info("Macro F1:        %.4f", results.macro_f1)
    logging.info("Micro F1:        %.4f", results.micro_f1)
    logging.info("Weighted F1:     %.4f", results.weighted_f1)
    logging.info("=" * 60)


def compare_results(
    results_dict: dict[str, EvaluationResult],
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Compare multiple evaluation results.

    Args:
        results_dict: Dictionary mapping experiment names to EvaluationResult objects.
        output_path: Optional path to save comparison table as CSV.

    Returns:
        DataFrame containing comparison of results.
    """
    comparison_data = []

    for name, result in results_dict.items():
        row = {
            "experiment": name,
            "accuracy": result.accuracy,
            "macro_f1": result.macro_f1,
            "micro_f1": result.micro_f1,
            "weighted_f1": result.weighted_f1,
            "num_samples": result.num_samples,
            "num_classes": result.num_classes,
        }

        if result.top5_accuracy is not None:
            row["top5_accuracy"] = result.top5_accuracy

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info("Saved comparison to %s", output_path)

    return df
