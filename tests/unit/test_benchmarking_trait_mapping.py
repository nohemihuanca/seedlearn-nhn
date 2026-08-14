"""Tests for seedlearn.benchmarking.trait_mapping."""

import pytest

from seedlearn.benchmarking.trait_mapping import (
    TRAIT_RULES,
    detect_not_observed,
    get_raw_vlm_values,
    map_prediction,
)


class TestTraitRulesIntegrity:
    """Verify TRAIT_RULES structure."""

    def test_all_rules_have_required_fields(self):
        for rule in TRAIT_RULES:
            assert rule.stri_column
            assert rule.category
            assert rule.pipeline_section
            assert rule.pipeline_field
            assert len(rule.match_values) > 0

    def test_expected_categories_present(self):
        categories = {r.category for r in TRAIT_RULES}
        assert categories == {
            "leaf_arrangement", "leaf_type", "leaf_margin",
            "stipules", "latex",
        }

    def test_no_duplicate_stri_columns(self):
        columns = [r.stri_column for r in TRAIT_RULES]
        assert len(columns) == len(set(columns))


class TestMapPrediction:
    """Test map_prediction function."""

    def test_empty_traits_returns_all_none(self):
        result = map_prediction({})
        assert all(v is None for v in result.values())
        assert len(result) == len(TRAIT_RULES)

    def test_alternate_leaf_arrangement(self):
        traits = {
            "leaf_arrangement": {"relative_position": "alternate"},
        }
        result = map_prediction(traits)
        assert result["leaf_arrangement__alternate"] == 1
        assert result["leaf_arrangement__opposite"] == 0
        assert result["leaf_arrangement__whorled_or_clustered"] == 0

    def test_opposite_leaf_arrangement(self):
        traits = {
            "leaf_arrangement": {"relative_position": "opposite"},
        }
        result = map_prediction(traits)
        assert result["leaf_arrangement__alternate"] == 0
        assert result["leaf_arrangement__opposite"] == 1

    def test_simple_leaf_type(self):
        traits = {
            "leaf_complexity": {"type": "simple"},
        }
        result = map_prediction(traits)
        assert result["leaf_type__simple"] == 1
        assert result["leaf_type__compound"] == 0

    def test_compound_leaf_type(self):
        traits = {
            "leaf_complexity": {"type": "compound"},
        }
        result = map_prediction(traits)
        assert result["leaf_type__simple"] == 0
        assert result["leaf_type__compound"] == 1

    def test_entire_margin(self):
        traits = {
            "leaf_morphology": {"margin": "entire"},
        }
        result = map_prediction(traits)
        assert result["leaf_margin__entire"] == 1
        assert result["leaf_margin__toothed"] == 0
        assert result["leaf_margin__lobed"] == 0

    def test_toothed_margin_variants(self):
        for value in ["toothed", "serrate", "dentate", "crenate", "finely serrate"]:
            traits = {"leaf_morphology": {"margin": value}}
            result = map_prediction(traits)
            assert result["leaf_margin__toothed"] == 1, f"Failed for: {value}"

    def test_lobed_margin(self):
        traits = {"leaf_morphology": {"margin": "palmately lobed"}}
        result = map_prediction(traits)
        assert result["leaf_margin__lobed"] == 1

    def test_stipules_present(self):
        traits = {"special_features": {"stipules": "present"}}
        result = map_prediction(traits)
        assert result["stipules__present"] == 1
        assert result["stipules__absent"] == 0

    def test_stipules_absent(self):
        traits = {"special_features": {"stipules": "absent"}}
        result = map_prediction(traits)
        assert result["stipules__present"] == 0
        assert result["stipules__absent"] == 1

    def test_latex_present(self):
        traits = {"special_features": {"latex": "present"}}
        result = map_prediction(traits)
        assert result["latex__present"] == 1
        assert result["latex__absent"] == 0

    def test_latex_not_observed(self):
        """'not observed' should return None — excluded from grading."""
        traits = {"special_features": {"latex": "not observed"}}
        result = map_prediction(traits)
        assert result["latex__present"] is None
        assert result["latex__absent"] is None

    def test_unclear_value_returns_none(self):
        traits = {"leaf_arrangement": {"relative_position": "unclear"}}
        result = map_prediction(traits)
        assert result["leaf_arrangement__alternate"] is None

    def test_na_value_returns_none(self):
        traits = {"leaf_arrangement": {"relative_position": "N/A"}}
        result = map_prediction(traits)
        assert result["leaf_arrangement__alternate"] is None

    def test_case_insensitive(self):
        traits = {"leaf_arrangement": {"relative_position": "Alternate"}}
        result = map_prediction(traits)
        assert result["leaf_arrangement__alternate"] == 1

    def test_column_suffix(self):
        traits = {"leaf_arrangement": {"relative_position": "alternate"}}
        result = map_prediction(traits, column_suffix="__consensus")
        assert "leaf_arrangement__alternate__consensus" in result
        assert result["leaf_arrangement__alternate__consensus"] == 1

    def test_missing_section_returns_none(self):
        traits = {"stem_traits": {"type": "woody"}}
        result = map_prediction(traits)
        # leaf_arrangement section missing -> None
        assert result["leaf_arrangement__alternate"] is None

    def test_no_match_value_returns_none(self):
        """Value present but doesn't match any rule."""
        traits = {"leaf_arrangement": {"relative_position": "spiral"}}
        result = map_prediction(traits)
        assert result["leaf_arrangement__alternate"] is None

    def test_stipules_none_returns_none(self):
        """'none' should return None — excluded from grading."""
        traits = {"special_features": {"stipules": "none"}}
        result = map_prediction(traits)
        assert result["stipules__present"] is None
        assert result["stipules__absent"] is None

    def test_multiple_categories(self):
        traits = {
            "leaf_arrangement": {"relative_position": "opposite"},
            "leaf_complexity": {"type": "compound"},
            "leaf_morphology": {"margin": "entire"},
            "special_features": {"stipules": "present", "latex": "absent"},
        }
        result = map_prediction(traits)
        assert result["leaf_arrangement__opposite"] == 1
        assert result["leaf_type__compound"] == 1
        assert result["leaf_margin__entire"] == 1
        assert result["stipules__present"] == 1
        assert result["latex__absent"] == 1


class TestGetRawVlmValues:
    """Test get_raw_vlm_values function."""

    def test_extracts_all_categories(self):
        traits = {
            "leaf_arrangement": {"relative_position": "alternate"},
            "leaf_complexity": {"type": "simple"},
            "leaf_morphology": {"margin": "entire"},
            "special_features": {"stipules": "present", "latex": "absent"},
        }
        result = get_raw_vlm_values(traits)
        assert result["leaf_arrangement"] == "alternate"
        assert result["leaf_type"] == "simple"
        assert result["leaf_margin"] == "entire"
        assert result["stipules"] == "present"
        assert result["latex"] == "absent"

    def test_empty_traits(self):
        result = get_raw_vlm_values({})
        assert all(v == "" for v in result.values())
        assert len(result) == 5  # 5 categories

    def test_missing_section(self):
        traits = {"leaf_arrangement": {"relative_position": "alternate"}}
        result = get_raw_vlm_values(traits)
        assert result["leaf_arrangement"] == "alternate"
        assert result["leaf_type"] == ""

    def test_non_dict_section(self):
        traits = {"leaf_arrangement": "not a dict"}
        result = get_raw_vlm_values(traits)
        assert result["leaf_arrangement"] == ""


class TestDetectNotObserved:
    """Test detect_not_observed helper."""

    def test_latex_not_observed(self):
        traits = {"special_features": {"latex": "not observed"}}
        assert detect_not_observed(traits, "latex") is True

    def test_latex_absent_not_detected(self):
        traits = {"special_features": {"latex": "absent"}}
        assert detect_not_observed(traits, "latex") is False

    def test_stipules_not_observed(self):
        traits = {"special_features": {"stipules": "Not Observed"}}
        assert detect_not_observed(traits, "stipules") is True

    def test_empty_traits(self):
        assert detect_not_observed({}, "latex") is False

    def test_missing_category(self):
        traits = {"leaf_arrangement": {"relative_position": "alternate"}}
        assert detect_not_observed(traits, "latex") is False
