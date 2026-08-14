"""Tests for seedlearn.benchmarking.trait_grader."""

import math

import pandas as pd
import pytest

from seedlearn.benchmarking.trait_grader import (
    TraitGradeRecord,
    TraitVerdict,
    grade_specimen_traits,
)


def _make_stri_row(**kwargs) -> pd.Series:
    """Build a minimal STRI row with defaults for leaf traits."""
    defaults = {
        "leaf_arrangement__alternate": 0,
        "leaf_arrangement__opposite": 0,
        "leaf_arrangement__whorled_or_clustered": 0,
        "leaf_arrangement__uncoded": 0,
        "leaf_type__simple": 0,
        "leaf_type__compound": 0,
        "leaf_type__uncoded": 0,
        "leaf_margin__entire": 0,
        "leaf_margin__toothed": 0,
        "leaf_margin__lobed": 0,
        "leaf_margin__uncoded": 0,
        "stipules__present": 0,
        "stipules__absent": 0,
        "stipules__uncoded": 0,
        "latex__present": 0,
        "latex__absent": 0,
        "latex__uncoded": 0,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


class TestGradeSpecimenTraits:
    """Test grade_specimen_traits function."""

    def test_correct_alternate_prediction(self):
        traits = {"leaf_arrangement": {"relative_position": "alternate"}}
        stri_row = _make_stri_row(**{"leaf_arrangement__alternate": 1})

        records = grade_specimen_traits(
            "SPEC1", "Test species", traits, stri_row
        )

        # Should have one CORRECT record for the predicted-positive column
        correct = [r for r in records if r.verdict == TraitVerdict.CORRECT]
        assert len(correct) == 1
        assert correct[0].stri_column == "leaf_arrangement__alternate"

    def test_incorrect_prediction(self):
        traits = {"leaf_arrangement": {"relative_position": "alternate"}}
        # Ground truth says opposite, not alternate
        stri_row = _make_stri_row(**{"leaf_arrangement__opposite": 1})

        records = grade_specimen_traits(
            "SPEC1", "Test species", traits, stri_row
        )

        incorrect = [r for r in records if r.verdict == TraitVerdict.INCORRECT]
        assert len(incorrect) == 1
        assert incorrect[0].stri_column == "leaf_arrangement__alternate"

    def test_multi_label_match_any(self):
        """Species with both alternate and opposite = 1. Predicting either is correct."""
        traits = {"leaf_arrangement": {"relative_position": "alternate"}}
        stri_row = _make_stri_row(**{
            "leaf_arrangement__alternate": 1,
            "leaf_arrangement__opposite": 1,
        })

        records = grade_specimen_traits(
            "SPEC1", "Test species", traits, stri_row
        )

        correct = [r for r in records if r.verdict == TraitVerdict.CORRECT]
        assert len(correct) == 1

    def test_uncoded_category_skipped(self):
        traits = {"leaf_arrangement": {"relative_position": "alternate"}}
        stri_row = _make_stri_row(**{"leaf_arrangement__uncoded": 1})

        records = grade_specimen_traits(
            "SPEC1", "Test species", traits, stri_row
        )

        uncoded = [r for r in records if r.verdict == TraitVerdict.SKIPPED_UNCODED]
        # All leaf_arrangement records should be skipped
        la_records = [r for r in records if r.category == "leaf_arrangement"]
        assert all(r.verdict == TraitVerdict.SKIPPED_UNCODED for r in la_records)

    def test_no_prediction_skipped(self):
        traits = {}  # Empty traits
        stri_row = _make_stri_row(**{"leaf_arrangement__alternate": 1})

        records = grade_specimen_traits(
            "SPEC1", "Test species", traits, stri_row
        )

        la_records = [r for r in records if r.category == "leaf_arrangement"]
        assert all(
            r.verdict == TraitVerdict.SKIPPED_NO_PREDICTION for r in la_records
        )

    def test_consensus_column_suffix(self):
        """Test with merged matrix consensus columns."""
        traits = {"leaf_complexity": {"type": "simple"}}
        stri_row = pd.Series({
            "leaf_type__simple__consensus": 1.0,
            "leaf_type__compound__consensus": 0.0,
            # No uncoded column — consensus uses NaN
        })

        records = grade_specimen_traits(
            "SPEC1", "Test species", traits, stri_row,
            column_suffix="__consensus",
        )

        correct = [r for r in records if r.verdict == TraitVerdict.CORRECT]
        lt_correct = [r for r in correct if r.category == "leaf_type"]
        assert len(lt_correct) == 1

    def test_nan_ground_truth_skipped(self):
        """NaN in consensus columns treated as uncoded."""
        traits = {"leaf_complexity": {"type": "simple"}}
        stri_row = pd.Series({
            "leaf_type__simple__consensus": float("nan"),
            "leaf_type__compound__consensus": float("nan"),
        })

        records = grade_specimen_traits(
            "SPEC1", "Test species", traits, stri_row,
            column_suffix="__consensus",
        )

        lt_records = [r for r in records if r.category == "leaf_type"]
        assert all(r.verdict == TraitVerdict.SKIPPED_UNCODED for r in lt_records)

    def test_not_observed_skipped(self):
        """'not observed' should get SKIPPED_NOT_OBSERVED verdict."""
        traits = {"special_features": {"latex": "not observed"}}
        stri_row = _make_stri_row(**{"latex__present": 1})

        records = grade_specimen_traits(
            "SPEC1", "Test species", traits, stri_row
        )

        latex_records = [r for r in records if r.category == "latex"]
        assert len(latex_records) > 0
        assert all(
            r.verdict == TraitVerdict.SKIPPED_NOT_OBSERVED for r in latex_records
        )

    def test_vlm_raw_value_populated(self):
        traits = {"leaf_arrangement": {"relative_position": "alternate"}}
        stri_row = _make_stri_row(**{"leaf_arrangement__alternate": 1})

        records = grade_specimen_traits(
            "SPEC1", "Test species", traits, stri_row
        )

        la_records = [r for r in records if r.category == "leaf_arrangement"]
        for r in la_records:
            assert r.vlm_raw_value == "alternate"

    def test_match_rule_populated(self):
        traits = {"leaf_morphology": {"margin": "serrate"}}
        stri_row = _make_stri_row(**{"leaf_margin__toothed": 1})

        records = grade_specimen_traits(
            "SPEC1", "Test species", traits, stri_row
        )

        toothed = [r for r in records if r.stri_column == "leaf_margin__toothed"]
        assert len(toothed) == 1
        # match_rule should be sorted match_values joined by |
        assert toothed[0].match_rule == "crenate|dentate|denticulate|serrate|serrulate|toothed"

    def test_vlm_raw_value_empty_for_missing_section(self):
        traits = {}  # Empty traits
        stri_row = _make_stri_row(**{"leaf_arrangement__alternate": 1})

        records = grade_specimen_traits(
            "SPEC1", "Test species", traits, stri_row
        )

        la_records = [r for r in records if r.category == "leaf_arrangement"]
        for r in la_records:
            assert r.vlm_raw_value == ""

    def test_multiple_categories_graded(self):
        traits = {
            "leaf_arrangement": {"relative_position": "opposite"},
            "leaf_complexity": {"type": "compound"},
            "special_features": {"stipules": "present", "latex": "absent"},
        }
        stri_row = _make_stri_row(**{
            "leaf_arrangement__opposite": 1,
            "leaf_type__compound": 1,
            "stipules__present": 1,
            "latex__absent": 1,
        })

        records = grade_specimen_traits(
            "SPEC1", "Test species", traits, stri_row
        )

        correct = [r for r in records if r.verdict == TraitVerdict.CORRECT]
        assert len(correct) == 4
