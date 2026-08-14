"""Tests for benchmark scoring matcher."""

from tests.benchmarks.scoring.matcher import (
    MatchResult,
    classify_prediction,
    normalize_prediction,
)


class TestNormalizePrediction:
    def test_lowercase(self):
        assert normalize_prediction("Entire") == "entire"

    def test_strip_whitespace(self):
        assert normalize_prediction("  entire  ") == "entire"

    def test_strip_parenthetical_justification(self):
        assert normalize_prediction("entire (smooth margin observed)") == "entire"

    def test_empty_string(self):
        assert normalize_prediction("") == ""

    def test_strip_period(self):
        assert normalize_prediction("entire.") == "entire"


class TestClassifyPrediction:
    def test_exact_match_single_label(self):
        assert (
            classify_prediction("entire", "entire", "exact") == MatchResult.EXACT_MATCH
        )

    def test_mismatch_single_label(self):
        assert classify_prediction("toothed", "entire", "exact") == MatchResult.MISMATCH

    def test_multi_label_any_match(self):
        assert (
            classify_prediction("entire", "entire | toothed", "multi_label")
            == MatchResult.EXACT_MATCH
        )

    def test_multi_label_second_value_match(self):
        assert (
            classify_prediction("toothed", "entire | toothed", "multi_label")
            == MatchResult.EXACT_MATCH
        )

    def test_multi_label_mismatch(self):
        assert (
            classify_prediction("lobed", "entire | toothed", "multi_label")
            == MatchResult.MISMATCH
        )

    def test_parent_match_serrate_to_toothed(self):
        assert (
            classify_prediction("serrate", "toothed", "exact")
            == MatchResult.PARENT_MATCH
        )

    def test_parent_match_dentate_to_toothed(self):
        assert (
            classify_prediction("dentate", "toothed", "exact")
            == MatchResult.PARENT_MATCH
        )

    def test_parent_match_in_multi_label(self):
        assert (
            classify_prediction("serrate", "entire | toothed", "multi_label")
            == MatchResult.PARENT_MATCH
        )

    def test_parent_match_not_observed_to_absent(self):
        assert (
            classify_prediction("not observed", "absent", "exact")
            == MatchResult.PARENT_MATCH
        )

    def test_abstention_unclear(self):
        assert (
            classify_prediction("unclear", "entire", "exact") == MatchResult.ABSTENTION
        )

    def test_abstention_not_visible(self):
        assert (
            classify_prediction("not visible", "entire", "exact")
            == MatchResult.ABSTENTION
        )

    def test_abstention_empty(self):
        assert classify_prediction("", "entire", "exact") == MatchResult.ABSTENTION

    def test_abstention_na(self):
        assert classify_prediction("n/a", "entire", "exact") == MatchResult.ABSTENTION

    def test_no_ground_truth_empty(self):
        assert classify_prediction("entire", "", "exact") == MatchResult.NO_GROUND_TRUTH

    def test_no_ground_truth_none(self):
        assert (
            classify_prediction("entire", None, "exact") == MatchResult.NO_GROUND_TRUTH
        )
