"""Core matching logic for comparing VLM predictions against ground truth."""

from __future__ import annotations

import re
from enum import Enum


class MatchResult(str, Enum):
    EXACT_MATCH = "exact_match"
    PARENT_MATCH = "parent_match"
    MISMATCH = "mismatch"
    ABSTENTION = "abstention"
    NO_GROUND_TRUTH = "no_ground_truth"


SUBTYPE_MAP: dict[str, str] = {
    # leaf_margin sub-types → parent
    "serrate": "toothed",
    "dentate": "toothed",
    "crenate": "toothed",
    "serrulate": "toothed",
    "denticulate": "toothed",
    # leaf_arrangement sub-types
    "clustered": "whorled",
    # latex: prompt uses "not observed" to mean absent
    "not observed": "absent",
}

ABSTENTION_PATTERNS: set[str] = {
    "unclear",
    "not visible",
    "cannot determine",
    "n/a",
    "na",
    "",
}


def normalize_prediction(value: str) -> str:
    """Lowercase, strip whitespace, remove parenthetical justifications and trailing periods."""
    value = value.strip().lower()
    value = re.sub(r"\s*\(.*$", "", value)
    value = value.rstrip(".")
    return value.strip()


def _is_abstention(prediction: str) -> bool:
    """Check if a normalized prediction is an abstention."""
    return prediction in ABSTENTION_PATTERNS


def _parse_ground_truth(gt_value: str) -> list[str]:
    """Parse pipe-delimited ground truth into list of valid values."""
    return [v.strip().lower() for v in gt_value.split("|") if v.strip()]


def classify_prediction(
    prediction: str, ground_truth: str | None, match_type: str
) -> MatchResult:
    """Classify a prediction against ground truth.

    Args:
        prediction: Raw VLM prediction string.
        ground_truth: Ground truth value(s), pipe-delimited for multi-label.
        match_type: "exact" or "multi_label".

    Returns:
        MatchResult classification.
    """
    if not ground_truth:
        return MatchResult.NO_GROUND_TRUTH

    pred = normalize_prediction(prediction)

    if _is_abstention(pred):
        return MatchResult.ABSTENTION

    gt_values = _parse_ground_truth(ground_truth)
    if not gt_values:
        return MatchResult.NO_GROUND_TRUTH

    if pred in gt_values:
        return MatchResult.EXACT_MATCH

    parent = SUBTYPE_MAP.get(pred)
    if parent and parent in gt_values:
        return MatchResult.PARENT_MATCH

    return MatchResult.MISMATCH
