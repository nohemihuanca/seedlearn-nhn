"""Metric computation for benchmark scoring.

Computes per-trait accuracy, confusion matrices, consistency metrics,
and multi-vs-single comparison statistics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from tests.benchmarks.scoring.loader import (
    TRAIT_COLUMNS,
    GroundTruthEntry,
    ResultEntry,
)
from tests.benchmarks.scoring.matcher import (
    MatchResult,
    _is_abstention,
    classify_prediction,
    normalize_prediction,
)


@dataclass
class TraitScore:
    """Scoring results for a single trait across all specimens."""

    trait: str
    n_exact_match: int = 0
    n_parent_match: int = 0
    n_mismatch: int = 0
    n_abstention: int = 0
    n_no_gt: int = 0
    # Multi-label breakdown
    n_exact_gt: int = 0  # specimens with exact (single-value) GT
    n_multi_label_gt: int = 0  # specimens with multi-label GT
    n_correct_from_exact_gt: int = 0  # correct predictions on exact-GT cells
    n_correct_from_multi_label_gt: int = 0  # correct predictions on multi-label cells

    @property
    def n_correct(self) -> int:
        return self.n_exact_match + self.n_parent_match

    @property
    def n_answered(self) -> int:
        """Specimens where the model gave an answer (not abstained, has GT)."""
        return self.n_correct + self.n_mismatch

    @property
    def n_total(self) -> int:
        """All specimens with ground truth."""
        return self.n_answered + self.n_abstention

    @property
    def accuracy(self) -> float:
        """Accuracy when the model answers (excludes abstentions)."""
        return self.n_correct / self.n_answered if self.n_answered else 0.0

    @property
    def strict_accuracy(self) -> float:
        """Accuracy counting abstentions as wrong."""
        return self.n_correct / self.n_total if self.n_total else 0.0

    @property
    def abstention_rate(self) -> float:
        return self.n_abstention / self.n_total if self.n_total else 0.0

    @property
    def exact_gt_accuracy(self) -> float:
        """Accuracy on single-value GT cells only."""
        return (
            self.n_correct_from_exact_gt / self.n_exact_gt if self.n_exact_gt else 0.0
        )

    @property
    def multi_label_gt_accuracy(self) -> float:
        """Accuracy on multi-label GT cells only."""
        return (
            self.n_correct_from_multi_label_gt / self.n_multi_label_gt
            if self.n_multi_label_gt
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trait": self.trait,
            "n_scored": self.n_total,
            "n_correct": self.n_correct,
            "n_exact_match": self.n_exact_match,
            "n_parent_match": self.n_parent_match,
            "n_mismatch": self.n_mismatch,
            "n_abstention": self.n_abstention,
            "n_no_gt": self.n_no_gt,
            "accuracy": round(self.accuracy, 4),
            "strict_accuracy": round(self.strict_accuracy, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "n_exact_gt": self.n_exact_gt,
            "n_multi_label_gt": self.n_multi_label_gt,
            "exact_gt_accuracy": round(self.exact_gt_accuracy, 4),
            "multi_label_gt_accuracy": round(self.multi_label_gt_accuracy, 4),
        }


@dataclass
class SpecimenScore:
    """Scoring results for a single specimen across all traits."""

    specimen_key: str
    family: str = ""
    scientific_name: str = ""
    multi_label_count: int = 0
    details: dict[str, dict[str, str]] = field(default_factory=dict)
    # details[trait] = {"predicted": ..., "ground_truth": ..., "result": ...}

    @property
    def n_correct(self) -> int:
        return sum(
            1
            for d in self.details.values()
            if d["result"] in ("exact_match", "parent_match")
        )

    @property
    def n_incorrect(self) -> int:
        return sum(1 for d in self.details.values() if d["result"] == "mismatch")

    @property
    def n_abstained(self) -> int:
        return sum(1 for d in self.details.values() if d["result"] == "abstention")

    @property
    def n_no_gt(self) -> int:
        return sum(1 for d in self.details.values() if d["result"] == "no_ground_truth")

    @property
    def accuracy(self) -> float:
        scored = self.n_correct + self.n_incorrect
        return self.n_correct / scored if scored else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "specimen_key": self.specimen_key,
            "family": self.family,
            "scientific_name": self.scientific_name,
            "multi_label_count": self.multi_label_count,
            "traits_correct": self.n_correct,
            "traits_incorrect": self.n_incorrect,
            "traits_abstained": self.n_abstained,
            "traits_no_gt": self.n_no_gt,
            "accuracy": round(self.accuracy, 4),
            "details": self.details,
        }


def compute_trait_scores(
    ground_truth: dict[str, GroundTruthEntry],
    results: dict[str, ResultEntry],
) -> list[TraitScore]:
    """Compute per-trait accuracy scores.

    Args:
        ground_truth: Dict of specimen_key to GroundTruthEntry.
        results: Dict of specimen_key to ResultEntry.

    Returns:
        List of TraitScore, one per trait with any ground truth data.
    """
    scores: list[TraitScore] = []

    for trait in TRAIT_COLUMNS:
        ts = TraitScore(trait=trait)

        for specimen_key, gt_entry in ground_truth.items():
            gt_value = gt_entry.traits.get(trait)
            match_type = gt_entry.match_types.get(trait, "exact")

            result_entry = results.get(specimen_key)
            prediction = result_entry.traits.get(trait, "") if result_entry else ""

            classification = classify_prediction(prediction, gt_value, match_type)

            if classification == MatchResult.EXACT_MATCH:
                ts.n_exact_match += 1
            elif classification == MatchResult.PARENT_MATCH:
                ts.n_parent_match += 1
            elif classification == MatchResult.MISMATCH:
                ts.n_mismatch += 1
            elif classification == MatchResult.ABSTENTION:
                ts.n_abstention += 1
            elif classification == MatchResult.NO_GROUND_TRUTH:
                ts.n_no_gt += 1
                continue  # Don't count in multi-label breakdown

            # Multi-label breakdown (only for scored items)
            if classification != MatchResult.NO_GROUND_TRUTH:
                is_correct = classification in (
                    MatchResult.EXACT_MATCH,
                    MatchResult.PARENT_MATCH,
                )
                if match_type == "multi_label":
                    ts.n_multi_label_gt += 1
                    if is_correct:
                        ts.n_correct_from_multi_label_gt += 1
                else:
                    ts.n_exact_gt += 1
                    if is_correct:
                        ts.n_correct_from_exact_gt += 1

        scores.append(ts)

    return scores


def compute_specimen_scores(
    ground_truth: dict[str, GroundTruthEntry],
    results: dict[str, ResultEntry],
) -> list[SpecimenScore]:
    """Compute per-specimen scorecards.

    Args:
        ground_truth: Dict of specimen_key to GroundTruthEntry.
        results: Dict of specimen_key to ResultEntry.

    Returns:
        List of SpecimenScore, one per specimen.
    """
    scores: list[SpecimenScore] = []

    for specimen_key, gt_entry in ground_truth.items():
        ss = SpecimenScore(
            specimen_key=specimen_key,
            family=gt_entry.family,
            scientific_name=gt_entry.scientific_name,
            multi_label_count=gt_entry.multi_label_count,
        )

        result_entry = results.get(specimen_key)

        for trait in TRAIT_COLUMNS:
            gt_value = gt_entry.traits.get(trait)
            match_type = gt_entry.match_types.get(trait, "exact")
            prediction = result_entry.traits.get(trait, "") if result_entry else ""

            classification = classify_prediction(prediction, gt_value, match_type)

            ss.details[trait] = {
                "predicted": normalize_prediction(prediction) if prediction else "",
                "ground_truth": gt_value or "",
                "result": classification.value,
            }

        scores.append(ss)

    return scores


def compute_confusion_matrix(
    predictions: list[str],
    actuals: list[str],
) -> dict[str, dict[str, int]]:
    """Compute a confusion matrix from parallel prediction/actual lists.

    Args:
        predictions: List of predicted values.
        actuals: List of actual (ground truth) values.

    Returns:
        Nested dict: cm[predicted][actual] = count.
    """
    all_labels = sorted(set(predictions) | set(actuals))
    cm: dict[str, dict[str, int]] = {p: {a: 0 for a in all_labels} for p in all_labels}

    for pred, actual in zip(predictions, actuals):
        cm[pred][actual] += 1

    return cm


def compute_confusion_matrices(
    ground_truth: dict[str, GroundTruthEntry],
    results: dict[str, ResultEntry],
) -> dict[str, dict[str, dict[str, int]]]:
    """Compute confusion matrices for each trait.

    Only includes specimens where GT exists. For multi-label GT, uses the
    first matching value as the actual (or the first GT value if no match).

    Args:
        ground_truth: Dict of specimen_key to GroundTruthEntry.
        results: Dict of specimen_key to ResultEntry.

    Returns:
        Dict of trait to confusion matrix.
    """
    matrices: dict[str, dict[str, dict[str, int]]] = {}

    for trait in TRAIT_COLUMNS:
        predictions: list[str] = []
        actuals: list[str] = []

        for specimen_key, gt_entry in ground_truth.items():
            gt_value = gt_entry.traits.get(trait)
            if not gt_value:
                continue

            result_entry = results.get(specimen_key)
            pred = (
                normalize_prediction(result_entry.traits.get(trait, ""))
                if result_entry
                else ""
            )
            if _is_abstention(pred):
                pred = "[abstain]"

            # For multi-label GT, pick the GT value that matches the prediction
            gt_values = [v.strip().lower() for v in gt_value.split("|") if v.strip()]
            if pred in gt_values:
                actual = pred  # prediction matches one of the GT values
            else:
                actual = gt_values[0]  # use first GT value as reference

            predictions.append(pred)
            actuals.append(actual)

        if predictions:
            matrices[trait] = compute_confusion_matrix(predictions, actuals)

    return matrices


def compute_consistency(predictions: list[str]) -> float:
    """Compute consistency of predictions across multiple images.

    Returns the fraction of predictions agreeing with the mode.

    Args:
        predictions: List of predicted values (one per image).

    Returns:
        Float between 0.0 and 1.0.
    """
    if not predictions:
        return 0.0
    counter = Counter(predictions)
    mode_count = counter.most_common(1)[0][1]
    return mode_count / len(predictions)


def compute_multi_vs_single(
    ground_truth: dict[str, GroundTruthEntry],
    multi_results: dict[str, ResultEntry],
    single_results: dict[str, list[ResultEntry]],
) -> dict[str, dict[str, Any]]:
    """Compute multi-image vs single-image comparison metrics.

    Uses strict accuracy for both modes: abstentions count against the
    denominator. For single-image, the denominator is total image count
    (not just non-abstained).

    Args:
        ground_truth: Ground truth entries.
        multi_results: Multi-image results (1 per specimen).
        single_results: Single-image results (N per specimen).

    Returns:
        Dict with per-trait comparison metrics.
    """
    comparison: dict[str, dict[str, Any]] = {}

    for trait in TRAIT_COLUMNS:
        multi_correct = 0
        single_correct_sum = 0
        single_count = 0
        majority_correct = 0
        consistencies: list[float] = []
        n_specimens = 0

        for specimen_key, gt_entry in ground_truth.items():
            gt_value = gt_entry.traits.get(trait)
            if not gt_value:
                continue
            match_type = gt_entry.match_types.get(trait, "exact")
            n_specimens += 1

            # Multi-image score
            multi_entry = multi_results.get(specimen_key)
            multi_pred = multi_entry.traits.get(trait, "") if multi_entry else ""
            multi_cls = classify_prediction(multi_pred, gt_value, match_type)
            if multi_cls in (MatchResult.EXACT_MATCH, MatchResult.PARENT_MATCH):
                multi_correct += 1

            # Single-image scores (strict accuracy — abstentions count against)
            single_entries = single_results.get(specimen_key, [])
            if single_entries:
                preds = [
                    normalize_prediction(e.traits.get(trait, ""))
                    for e in single_entries
                ]
                single_count += len(preds)  # all images, including abstentions

                non_abstain = [p for p in preds if not _is_abstention(p)]

                for p in non_abstain:
                    cls = classify_prediction(p, gt_value, match_type)
                    if cls in (MatchResult.EXACT_MATCH, MatchResult.PARENT_MATCH):
                        single_correct_sum += 1

                # Consistency (among non-abstained predictions only)
                if non_abstain:
                    consistencies.append(compute_consistency(non_abstain))

                # Majority vote (among non-abstained)
                if non_abstain:
                    counter = Counter(non_abstain)
                    majority = counter.most_common(1)[0][0]
                    maj_cls = classify_prediction(majority, gt_value, match_type)
                    if maj_cls in (MatchResult.EXACT_MATCH, MatchResult.PARENT_MATCH):
                        majority_correct += 1

        multi_acc = multi_correct / n_specimens if n_specimens else 0.0
        # Strict accuracy for both: abstentions count against denominator
        single_mean_acc = single_correct_sum / single_count if single_count else 0.0
        majority_acc = majority_correct / n_specimens if n_specimens else 0.0
        mean_consistency = (
            sum(consistencies) / len(consistencies) if consistencies else 0.0
        )

        comparison[trait] = {
            "multi_accuracy": round(multi_acc, 4),
            "single_mean_accuracy": round(single_mean_acc, 4),
            "majority_vote_accuracy": round(majority_acc, 4),
            "mean_consistency": round(mean_consistency, 4),
            "resolution_effect": round(multi_acc - single_mean_acc, 4),
            "n_specimens": n_specimens,
        }

    return comparison
