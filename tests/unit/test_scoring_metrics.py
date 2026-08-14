"""Tests for benchmark scoring metrics."""

import pytest

from tests.benchmarks.scoring.loader import GroundTruthEntry, ResultEntry
from tests.benchmarks.scoring.metrics import (
    TraitScore,
    compute_confusion_matrix,
    compute_consistency,
    compute_trait_scores,
)


class TestTraitScore:
    def test_accuracy_excludes_abstentions(self):
        score = TraitScore(
            trait="leaf_margin",
            n_exact_match=10,
            n_parent_match=2,
            n_mismatch=3,
            n_abstention=5,
            n_no_gt=0,
            n_exact_gt=12,
            n_multi_label_gt=3,
            n_correct_from_exact_gt=9,
            n_correct_from_multi_label_gt=3,
        )
        # accuracy = 12 / (12 + 3) = 0.8
        assert score.accuracy == pytest.approx(0.8)

    def test_strict_accuracy_includes_abstentions(self):
        score = TraitScore(
            trait="leaf_margin",
            n_exact_match=10,
            n_parent_match=2,
            n_mismatch=3,
            n_abstention=5,
            n_no_gt=0,
            n_exact_gt=12,
            n_multi_label_gt=3,
            n_correct_from_exact_gt=9,
            n_correct_from_multi_label_gt=3,
        )
        # strict = 12 / (12 + 3 + 5) = 0.6
        assert score.strict_accuracy == pytest.approx(0.6)

    def test_abstention_rate(self):
        score = TraitScore(
            trait="test",
            n_exact_match=10,
            n_parent_match=0,
            n_mismatch=5,
            n_abstention=5,
            n_no_gt=0,
            n_exact_gt=15,
            n_multi_label_gt=0,
            n_correct_from_exact_gt=10,
            n_correct_from_multi_label_gt=0,
        )
        # abstention_rate = 5 / 20 = 0.25
        assert score.abstention_rate == pytest.approx(0.25)

    def test_zero_scored(self):
        score = TraitScore(
            trait="test",
            n_exact_match=0,
            n_parent_match=0,
            n_mismatch=0,
            n_abstention=0,
            n_no_gt=10,
            n_exact_gt=0,
            n_multi_label_gt=0,
            n_correct_from_exact_gt=0,
            n_correct_from_multi_label_gt=0,
        )
        assert score.accuracy == 0.0
        assert score.strict_accuracy == 0.0


class TestComputeTraitScores:
    def test_single_specimen_exact_match(self):
        gt = {
            "spec1": GroundTruthEntry(
                specimen_key="spec1",
                specimen_id="S1",
                family="Fab",
                scientific_name="Test sp",
                num_images=5,
                traits={"leaf_complexity": "compound"},
                match_types={"leaf_complexity": "exact"},
            )
        }
        results = {
            "spec1": ResultEntry(
                specimen_key="spec1",
                traits={"leaf_complexity": "compound"},
            )
        }
        scores = compute_trait_scores(gt, results)
        lc = next(s for s in scores if s.trait == "leaf_complexity")
        assert lc.n_exact_match == 1
        assert lc.accuracy == 1.0

    def test_multi_label_match(self):
        gt = {
            "spec1": GroundTruthEntry(
                specimen_key="spec1",
                specimen_id="S1",
                family="Fab",
                scientific_name="Test sp",
                num_images=5,
                traits={"leaf_margin": "entire | toothed"},
                match_types={"leaf_margin": "multi_label"},
            )
        }
        results = {
            "spec1": ResultEntry(
                specimen_key="spec1",
                traits={"leaf_margin": "entire"},
            )
        }
        scores = compute_trait_scores(gt, results)
        lm = next(s for s in scores if s.trait == "leaf_margin")
        assert lm.n_exact_match == 1
        assert lm.n_multi_label_gt == 1
        assert lm.n_correct_from_multi_label_gt == 1


class TestConsistency:
    def test_all_agree(self):
        assert compute_consistency(["entire", "entire", "entire"]) == 1.0

    def test_majority(self):
        assert compute_consistency(["entire", "entire", "toothed"]) == pytest.approx(
            2 / 3
        )

    def test_no_agreement(self):
        assert compute_consistency(["entire", "toothed", "lobed"]) == pytest.approx(
            1 / 3
        )

    def test_empty(self):
        assert compute_consistency([]) == 0.0


class TestConfusionMatrix:
    def test_simple_matrix(self):
        predictions = ["entire", "toothed", "entire", "entire"]
        actuals = ["entire", "entire", "entire", "toothed"]
        cm = compute_confusion_matrix(predictions, actuals)
        assert cm["entire"]["entire"] == 2
        assert cm["toothed"]["entire"] == 1  # predicted toothed, actual entire
        assert cm["entire"]["toothed"] == 1  # predicted entire, actual toothed
