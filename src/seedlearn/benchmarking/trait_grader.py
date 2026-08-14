"""Score Stage 1 trait predictions against STRI ground truth.

Compares mapped binary predictions from the VLM morphology stage against
the STRI trait matrix, using a "match any" policy for multi-label species.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from seedlearn.benchmarking.trait_mapping import (
    TRAIT_RULES,
    TraitRule,
    detect_not_observed,
    get_raw_vlm_values,
    map_prediction,
)


class TraitVerdict(str, Enum):
    """Outcome for a single trait prediction on a single specimen."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    SKIPPED_UNCODED = "skipped_uncoded"
    SKIPPED_NOT_OBSERVED = "skipped_not_observed"
    SKIPPED_NO_PREDICTION = "skipped_no_prediction"


@dataclass
class TraitGradeRecord:
    """Grading result for one trait on one specimen.

    Attributes:
        specimen_id: Catalog ID_YPS value.
        scientific_name: Accepted species name.
        stri_column: STRI column being evaluated (e.g., "leaf_arrangement__alternate").
        category: Trait category (e.g., "leaf_arrangement").
        ground_truth: STRI binary value (1=present, 0=absent, NaN=uncoded).
        predicted: Pipeline binary prediction (1, 0, or None).
        verdict: Scoring outcome.
    """

    specimen_id: str
    scientific_name: str
    stri_column: str
    category: str
    ground_truth: float | None
    predicted: int | None
    verdict: TraitVerdict
    vlm_raw_value: str = ""
    match_rule: str = ""


def _is_uncoded(
    stri_row: pd.Series,
    category: str,
    rules: list[TraitRule],
    column_suffix: str,
) -> bool:
    """Check if a trait category is uncoded for this species.

    For cl185-style matrices: checks the ``{category}__uncoded`` flag column.
    For merged consensus matrices: checks if all option columns are NaN.

    Args:
        stri_row: STRI matrix row for this species.
        category: Trait category name.
        rules: Trait rules for this category.
        column_suffix: Column name suffix (e.g., "" or "__consensus").

    Returns:
        True if the category is uncoded (should be skipped).
    """
    # Check explicit uncoded flag (cl185 format)
    uncoded_col = f"{category}__uncoded{column_suffix}"
    if uncoded_col in stri_row.index:
        val = stri_row[uncoded_col]
        if not (isinstance(val, float) and math.isnan(val)):
            return bool(int(val) == 1)

    # Fallback: all option columns are NaN (merged consensus format)
    all_nan = True
    for rule in rules:
        col = rule.stri_column + column_suffix
        if col in stri_row.index:
            val = stri_row[col]
            if not (isinstance(val, float) and math.isnan(val)):
                all_nan = False
                break
    return all_nan


def grade_specimen_traits(
    specimen_id: str,
    scientific_name: str,
    traits: dict[str, Any],
    stri_row: pd.Series,
    column_suffix: str = "",
) -> list[TraitGradeRecord]:
    """Grade all trait predictions for one specimen against STRI ground truth.

    Uses the "match any" policy: within each category, the prediction is
    correct if the predicted option has ground_truth=1. This accounts for
    multi-label species (e.g., both alternate and opposite = 1).

    Args:
        specimen_id: Catalog specimen ID.
        scientific_name: Accepted species name.
        traits: Nested trait dict from Stage 1 morphology output.
        stri_row: STRI matrix row for this species (indexed by column name).
        column_suffix: Column suffix for STRI columns (e.g., "__consensus").

    Returns:
        List of TraitGradeRecord, one per evaluated STRI column.
    """
    predictions = map_prediction(traits, column_suffix=column_suffix)
    raw_vlm = get_raw_vlm_values(traits)
    records: list[TraitGradeRecord] = []

    # Group rules by category
    categories: dict[str, list[TraitRule]] = {}
    for rule in TRAIT_RULES:
        categories.setdefault(rule.category, []).append(rule)

    for category, cat_rules in categories.items():
        # Check if uncoded
        uncoded = _is_uncoded(stri_row, category, cat_rules, column_suffix)
        vlm_value = raw_vlm.get(category, "")

        for rule in cat_rules:
            col = rule.stri_column + column_suffix
            pred = predictions.get(col)

            # Get ground truth
            gt: float | None = None
            if col in stri_row.index:
                gt_val = stri_row[col]
                if isinstance(gt_val, float) and math.isnan(gt_val):
                    gt = None
                else:
                    gt = float(gt_val)

            # Determine verdict
            if uncoded or gt is None:
                verdict = TraitVerdict.SKIPPED_UNCODED
            elif pred is None and detect_not_observed(traits, category):
                verdict = TraitVerdict.SKIPPED_NOT_OBSERVED
            elif pred is None:
                verdict = TraitVerdict.SKIPPED_NO_PREDICTION
            elif pred == 1 and gt == 1.0:
                verdict = TraitVerdict.CORRECT
            elif pred == 1 and gt == 0.0:
                verdict = TraitVerdict.INCORRECT
            elif pred == 0:
                # Pipeline predicted a different option in this category.
                # Don't penalize for predicting the "wrong" 0 — correctness
                # is assessed only on the predicted-positive column.
                continue
            else:
                verdict = TraitVerdict.INCORRECT

            records.append(
                TraitGradeRecord(
                    specimen_id=specimen_id,
                    scientific_name=scientific_name,
                    stri_column=rule.stri_column,
                    category=category,
                    ground_truth=gt,
                    predicted=pred,
                    verdict=verdict,
                    vlm_raw_value=vlm_value,
                    match_rule="|".join(sorted(rule.match_values)),
                )
            )

    return records
