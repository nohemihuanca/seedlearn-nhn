"""Tests for seedlearn.benchmarking.report."""

import json
from pathlib import Path

import pandas as pd
import pytest

from seedlearn.benchmarking.id_grader import IDGradeRecord
from seedlearn.benchmarking.report import (
    build_id_per_specimen_df,
    build_id_summary,
    build_id_summary_by_partition,
    build_trait_accuracy_per_category,
    build_trait_accuracy_per_species,
    build_trait_accuracy_per_trait,
    build_trait_per_specimen_df,
    build_within_species_agreement,
    compute_binary_metrics,
    compute_multiclass_id_metrics,
    save_benchmark_results,
)
from seedlearn.benchmarking.trait_grader import TraitGradeRecord, TraitVerdict


def _make_trait_records() -> list[TraitGradeRecord]:
    """Build a fixture of trait grade records."""
    return [
        TraitGradeRecord("S1", "Sp1", "leaf_arrangement__alternate", "leaf_arrangement", 1.0, 1, TraitVerdict.CORRECT),
        TraitGradeRecord("S2", "Sp2", "leaf_arrangement__alternate", "leaf_arrangement", 1.0, 1, TraitVerdict.CORRECT),
        TraitGradeRecord("S3", "Sp3", "leaf_arrangement__alternate", "leaf_arrangement", 0.0, 1, TraitVerdict.INCORRECT),
        TraitGradeRecord("S4", "Sp4", "leaf_arrangement__alternate", "leaf_arrangement", None, None, TraitVerdict.SKIPPED_UNCODED),
        TraitGradeRecord("S1", "Sp1", "leaf_type__simple", "leaf_type", 1.0, 1, TraitVerdict.CORRECT),
        TraitGradeRecord("S2", "Sp2", "leaf_type__simple", "leaf_type", 1.0, None, TraitVerdict.SKIPPED_NO_PREDICTION),
        TraitGradeRecord("S1", "Sp1", "latex__present", "latex", 1.0, None, TraitVerdict.SKIPPED_NOT_OBSERVED),
        TraitGradeRecord("S2", "Sp2", "latex__present", "latex", 1.0, None, TraitVerdict.SKIPPED_NOT_OBSERVED),
    ]


def _make_id_records() -> list[IDGradeRecord]:
    """Build a fixture of ID grade records."""
    return [
        IDGradeRecord("S1", "Fam1", "Gen1", "Gen1 sp1", "Fam1", "Gen1", "Gen1 sp1", True, True, True, "high", False),
        IDGradeRecord("S2", "Fam1", "Gen2", "Gen2 sp2", "Fam1", "Gen2", "Gen2 sp3", True, True, False, "medium", False),
        IDGradeRecord("S3", "Fam2", "Gen3", "Gen3 sp3", None, None, None, False, False, False, None, True),
    ]


def _make_multi_specimen_records() -> list[TraitGradeRecord]:
    """Build a fixture with 2 species, 3 specimens each, varying correctness.

    Species A (SpA): specimens A1, A2, A3
      - leaf_arrangement: A1 correct (alternate), A2 correct (alternate), A3 incorrect (alternate)
      - leaf_type: A1 correct (simple), A2 correct (simple), A3 correct (simple)

    Species B (SpB): specimens B1, B2, B3
      - leaf_arrangement: B1 incorrect, B2 incorrect, B3 no_prediction
      - leaf_type: B1 correct (compound), B2 no_prediction, B3 no_prediction
    """
    return [
        # SpA leaf_arrangement — 3 graded, 2 correct
        TraitGradeRecord("A1", "SpA", "leaf_arrangement__alternate", "leaf_arrangement", 1.0, 1, TraitVerdict.CORRECT),
        TraitGradeRecord("A1", "SpA", "leaf_arrangement__opposite", "leaf_arrangement", 0.0, 0, TraitVerdict.CORRECT),
        TraitGradeRecord("A2", "SpA", "leaf_arrangement__alternate", "leaf_arrangement", 1.0, 1, TraitVerdict.CORRECT),
        TraitGradeRecord("A2", "SpA", "leaf_arrangement__opposite", "leaf_arrangement", 0.0, 0, TraitVerdict.CORRECT),
        TraitGradeRecord("A3", "SpA", "leaf_arrangement__alternate", "leaf_arrangement", 1.0, 0, TraitVerdict.INCORRECT),
        TraitGradeRecord("A3", "SpA", "leaf_arrangement__opposite", "leaf_arrangement", 0.0, 1, TraitVerdict.INCORRECT),
        # SpA leaf_type — 3 graded, 3 correct
        TraitGradeRecord("A1", "SpA", "leaf_type__simple", "leaf_type", 1.0, 1, TraitVerdict.CORRECT),
        TraitGradeRecord("A2", "SpA", "leaf_type__simple", "leaf_type", 1.0, 1, TraitVerdict.CORRECT),
        TraitGradeRecord("A3", "SpA", "leaf_type__simple", "leaf_type", 1.0, 1, TraitVerdict.CORRECT),
        # SpB leaf_arrangement — 2 graded (incorrect), 1 no_prediction
        TraitGradeRecord("B1", "SpB", "leaf_arrangement__alternate", "leaf_arrangement", 0.0, 1, TraitVerdict.INCORRECT),
        TraitGradeRecord("B1", "SpB", "leaf_arrangement__opposite", "leaf_arrangement", 1.0, 0, TraitVerdict.INCORRECT),
        TraitGradeRecord("B2", "SpB", "leaf_arrangement__alternate", "leaf_arrangement", 0.0, 1, TraitVerdict.INCORRECT),
        TraitGradeRecord("B2", "SpB", "leaf_arrangement__opposite", "leaf_arrangement", 1.0, 0, TraitVerdict.INCORRECT),
        TraitGradeRecord("B3", "SpB", "leaf_arrangement__alternate", "leaf_arrangement", 0.0, None, TraitVerdict.SKIPPED_NO_PREDICTION),
        TraitGradeRecord("B3", "SpB", "leaf_arrangement__opposite", "leaf_arrangement", 1.0, None, TraitVerdict.SKIPPED_NO_PREDICTION),
        # SpB leaf_type — 1 correct, 2 no_prediction
        TraitGradeRecord("B1", "SpB", "leaf_type__compound", "leaf_type", 1.0, 1, TraitVerdict.CORRECT),
        TraitGradeRecord("B2", "SpB", "leaf_type__compound", "leaf_type", 1.0, None, TraitVerdict.SKIPPED_NO_PREDICTION),
        TraitGradeRecord("B3", "SpB", "leaf_type__compound", "leaf_type", 1.0, None, TraitVerdict.SKIPPED_NO_PREDICTION),
    ]


class TestComputeBinaryMetrics:
    def test_perfect_classifier(self):
        m = compute_binary_metrics(tp=10, fp=0, support=10, n_graded_specimens=20)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["prevalence"] == 0.5

    def test_no_predictions(self):
        m = compute_binary_metrics(tp=0, fp=0, support=5, n_graded_specimens=10)
        assert m["precision"] is None
        assert m["recall"] == 0.0
        assert m["f1"] is None
        assert m["prevalence"] == 0.5

    def test_all_false_positives(self):
        m = compute_binary_metrics(tp=0, fp=5, support=0, n_graded_specimens=10)
        assert m["precision"] == 0.0
        assert m["recall"] is None
        assert m["f1"] is None
        assert m["prevalence"] == 0.0

    def test_partial_performance(self):
        # TP=3, FP=2, support=5 → FN=2
        m = compute_binary_metrics(tp=3, fp=2, support=5, n_graded_specimens=10)
        assert m["precision"] == 3 / 5
        assert m["recall"] == 3 / 5
        assert abs(m["f1"] - 0.6) < 0.001
        assert m["prevalence"] == 0.5

    def test_zero_graded_specimens(self):
        m = compute_binary_metrics(tp=0, fp=0, support=0, n_graded_specimens=0)
        assert m["prevalence"] is None


class TestBuildTraitPerSpecimenDf:
    def test_columns_present(self):
        df = build_trait_per_specimen_df(_make_trait_records())
        expected_cols = {
            "specimen_id", "scientific_name", "stri_column", "category",
            "ground_truth", "predicted", "verdict", "vlm_raw_value", "match_rule",
        }
        assert expected_cols == set(df.columns)

    def test_row_count(self):
        df = build_trait_per_specimen_df(_make_trait_records())
        assert len(df) == 8


class TestBuildTraitAccuracyPerTrait:
    def test_accuracy_calculation(self):
        df = build_trait_accuracy_per_trait(_make_trait_records())
        la_row = df[df["stri_column"] == "leaf_arrangement__alternate"].iloc[0]
        assert la_row["n_correct"] == 2
        assert la_row["n_incorrect"] == 1
        assert la_row["n_graded"] == 3
        assert abs(la_row["accuracy"] - 2 / 3) < 0.001

    def test_all_skipped_has_none_accuracy(self):
        records = [
            TraitGradeRecord("S1", "Sp1", "latex__present", "latex", None, None, TraitVerdict.SKIPPED_UNCODED),
        ]
        df = build_trait_accuracy_per_trait(records)
        row = df[df["stri_column"] == "latex__present"].iloc[0]
        assert row["accuracy"] is None
        assert row["n_graded"] == 0


    def test_not_observed_column(self):
        df = build_trait_accuracy_per_trait(_make_trait_records())
        latex_row = df[df["stri_column"] == "latex__present"].iloc[0]
        assert latex_row["n_not_observed"] == 2
        assert latex_row["n_graded"] == 0

    def test_not_observed_excluded_from_accuracy(self):
        df = build_trait_accuracy_per_trait(_make_trait_records())
        latex_row = df[df["stri_column"] == "latex__present"].iloc[0]
        assert pd.isna(latex_row["accuracy"])


class TestBuildTraitAccuracyPerTraitWithSupport:
    def test_f1_columns_present(self):
        stri_support = {"leaf_arrangement__alternate": 2, "leaf_type__simple": 1, "latex__present": 0}
        df = build_trait_accuracy_per_trait(_make_trait_records(), stri_support=stri_support)
        assert "support" in df.columns
        assert "precision" in df.columns
        assert "recall" in df.columns
        assert "f1" in df.columns
        assert "prevalence" in df.columns

    def test_f1_values(self):
        # leaf_arrangement__alternate: 2 correct, 1 incorrect → TP=2, FP=1, support=2
        stri_support = {"leaf_arrangement__alternate": 2}
        df = build_trait_accuracy_per_trait(_make_trait_records(), stri_support=stri_support)
        la_row = df[df["stri_column"] == "leaf_arrangement__alternate"].iloc[0]
        assert la_row["support"] == 2
        assert la_row["precision"] == 2 / 3
        assert la_row["recall"] == 1.0
        # f1 = 2 * (2/3) * 1.0 / ((2/3) + 1.0) = 0.8
        assert abs(la_row["f1"] - 0.8) < 0.001

    def test_no_support_columns_without_arg(self):
        df = build_trait_accuracy_per_trait(_make_trait_records())
        assert "support" not in df.columns
        assert "f1" not in df.columns


class TestBuildTraitAccuracyPerCategory:
    def test_category_rollup(self):
        df = build_trait_accuracy_per_category(_make_trait_records())
        la_row = df[df["category"] == "leaf_arrangement"].iloc[0]
        # S1: correct, S2: correct, S3: incorrect, S4: skipped_uncoded
        assert la_row["n_correct"] == 2
        assert la_row["n_incorrect"] == 1
        assert la_row["n_graded"] == 3

    def test_not_observed_category_rollup(self):
        df = build_trait_accuracy_per_category(_make_trait_records())
        latex_row = df[df["category"] == "latex"].iloc[0]
        assert latex_row["n_not_observed"] == 2
        assert latex_row["n_graded"] == 0


class TestBuildTraitAccuracyPerCategoryWithSupport:
    def test_macro_f1_present(self):
        stri_support = {"leaf_arrangement__alternate": 2, "leaf_type__simple": 1, "latex__present": 0}
        df = build_trait_accuracy_per_category(_make_trait_records(), stri_support=stri_support)
        assert "macro_f1" in df.columns

    def test_no_macro_f1_without_support(self):
        df = build_trait_accuracy_per_category(_make_trait_records())
        assert "macro_f1" not in df.columns


class TestBuildIdSummary:
    def test_accuracy_excludes_errors(self):
        summary = build_id_summary(_make_id_records())
        assert summary["n_graded"] == 2
        assert summary["n_stage5_error"] == 1
        assert summary["family_accuracy"] == 1.0  # 2/2
        assert summary["genus_accuracy"] == 1.0
        assert summary["species_accuracy"] == 0.5  # 1/2

    def test_empty_records(self):
        summary = build_id_summary([])
        assert summary["n_graded"] == 0

    def test_f1_keys_present(self):
        summary = build_id_summary(_make_id_records())
        assert "species_weighted_f1" in summary
        assert "species_macro_f1" in summary
        assert "family_weighted_precision" in summary


def _make_id_records_with_partitions() -> list[IDGradeRecord]:
    """Build ID records with partition labels for testing."""
    return [
        # Train: 2 correct species
        IDGradeRecord("S1", "Fam1", "Gen1", "Gen1 sp1", "Fam1", "Gen1", "Gen1 sp1", True, True, True, "high", False, partition="train"),
        IDGradeRecord("S2", "Fam1", "Gen1", "Gen1 sp2", "Fam1", "Gen1", "Gen1 sp2", True, True, True, "high", False, partition="train"),
        # Val: 1 correct, 1 wrong species
        IDGradeRecord("S3", "Fam1", "Gen2", "Gen2 sp3", "Fam1", "Gen2", "Gen2 sp3", True, True, True, "medium", False, partition="val"),
        IDGradeRecord("S4", "Fam1", "Gen2", "Gen2 sp4", "Fam1", "Gen2", "Gen2 sp5", True, True, False, "low", False, partition="val"),
        # Test: 1 wrong genus
        IDGradeRecord("S5", "Fam2", "Gen3", "Gen3 sp5", "Fam2", "Gen4", "Gen4 sp6", True, False, False, "low", False, partition="test"),
    ]


class TestComputeMulticlassIdMetrics:
    def test_basic_accuracy(self):
        records = _make_id_records()
        m = compute_multiclass_id_metrics(records)
        assert m["family_accuracy"] == 1.0
        assert m["species_accuracy"] == 0.5
        assert m["n_graded"] == 2
        assert m["n_stage5_error"] == 1

    def test_f1_keys_present(self):
        m = compute_multiclass_id_metrics(_make_id_records())
        for rank in ("family", "genus", "species"):
            assert f"{rank}_weighted_f1" in m
            assert f"{rank}_macro_f1" in m
            assert f"{rank}_weighted_precision" in m
            assert f"{rank}_weighted_recall" in m
            assert f"{rank}_macro_precision" in m
            assert f"{rank}_macro_recall" in m

    def test_empty_records(self):
        m = compute_multiclass_id_metrics([])
        assert m["n_graded"] == 0
        assert m["species_weighted_f1"] == 0.0

    def test_perfect_records(self):
        records = [
            IDGradeRecord("S1", "Fam1", "Gen1", "Gen1 sp1", "Fam1", "Gen1", "Gen1 sp1", True, True, True, "high", False),
            IDGradeRecord("S2", "Fam1", "Gen1", "Gen1 sp1", "Fam1", "Gen1", "Gen1 sp1", True, True, True, "high", False),
        ]
        m = compute_multiclass_id_metrics(records)
        assert m["species_weighted_f1"] == 1.0
        assert m["species_macro_f1"] == 1.0


class TestBuildIdSummaryByPartition:
    def test_partitions_present(self):
        records = _make_id_records_with_partitions()
        result = build_id_summary_by_partition(records)
        assert "all" in result
        assert "train" in result
        assert "val" in result
        assert "test" in result

    def test_all_matches_total(self):
        records = _make_id_records_with_partitions()
        result = build_id_summary_by_partition(records)
        assert result["all"]["n_graded"] == 5

    def test_train_perfect_species(self):
        records = _make_id_records_with_partitions()
        result = build_id_summary_by_partition(records)
        assert result["train"]["species_accuracy"] == 1.0

    def test_val_species_accuracy(self):
        records = _make_id_records_with_partitions()
        result = build_id_summary_by_partition(records)
        assert result["val"]["species_accuracy"] == 0.5  # 1/2

    def test_no_partitions(self):
        records = _make_id_records()  # no partition set
        result = build_id_summary_by_partition(records)
        assert "all" in result
        # No partition groups besides "all"
        assert set(result.keys()) == {"all"}


class TestBuildIdPerSpecimenDf:
    def test_partition_column_present(self):
        records = _make_id_records_with_partitions()
        df = build_id_per_specimen_df(records)
        assert "partition" in df.columns

    def test_partition_values(self):
        records = _make_id_records_with_partitions()
        df = build_id_per_specimen_df(records)
        assert set(df["partition"].dropna().unique()) == {"train", "val", "test"}

    def test_partition_none_when_unset(self):
        records = _make_id_records()
        df = build_id_per_specimen_df(records)
        assert "partition" in df.columns
        assert df["partition"].isna().all()


class TestSaveBenchmarkResults:
    def test_writes_files(self, tmp_path: Path):
        save_benchmark_results(
            tmp_path,
            _make_trait_records(),
            _make_id_records(),
            source_label="cl185",
        )

        assert (tmp_path / "cl185_trait_grades_per_specimen.csv").exists()
        assert (tmp_path / "cl185_trait_accuracy_per_trait.csv").exists()
        assert (tmp_path / "cl185_trait_accuracy_per_category.csv").exists()
        assert (tmp_path / "cl185_trait_accuracy_per_species.csv").exists()
        assert (tmp_path / "cl185_within_species_agreement.csv").exists()
        assert (tmp_path / "cl185_id_grades_per_specimen.csv").exists()
        assert (tmp_path / "cl185_id_summary.json").exists()

    def test_id_summary_json_content(self, tmp_path: Path):
        save_benchmark_results(
            tmp_path, [], _make_id_records(), source_label="test"
        )
        with open(tmp_path / "test_id_summary.json") as f:
            data = json.load(f)
        assert "family_accuracy" in data

    def test_partition_json_saved_when_partitions(self, tmp_path: Path):
        save_benchmark_results(
            tmp_path, [], _make_id_records_with_partitions(), source_label="p"
        )
        assert (tmp_path / "p_id_summary_by_partition.json").exists()
        with open(tmp_path / "p_id_summary_by_partition.json") as f:
            data = json.load(f)
        assert "all" in data
        assert "train" in data

    def test_no_partition_json_without_partitions(self, tmp_path: Path):
        save_benchmark_results(
            tmp_path, [], _make_id_records(), source_label="np"
        )
        assert not (tmp_path / "np_id_summary_by_partition.json").exists()


class TestBuildTraitAccuracyPerSpecies:
    def test_groups_by_species(self):
        df = build_trait_accuracy_per_species(_make_multi_specimen_records())
        species = sorted(df["scientific_name"].unique())
        assert species == ["SpA", "SpB"]

    def test_spa_leaf_arrangement_accuracy(self):
        df = build_trait_accuracy_per_species(_make_multi_specimen_records())
        row = df[(df["scientific_name"] == "SpA") & (df["category"] == "leaf_arrangement")].iloc[0]
        assert row["n_correct"] == 2
        assert row["n_incorrect"] == 1
        assert row["n_graded"] == 3
        assert abs(row["accuracy"] - 2 / 3) < 0.001

    def test_spa_leaf_type_perfect(self):
        df = build_trait_accuracy_per_species(_make_multi_specimen_records())
        row = df[(df["scientific_name"] == "SpA") & (df["category"] == "leaf_type")].iloc[0]
        assert row["n_correct"] == 3
        assert row["accuracy"] == 1.0

    def test_spb_leaf_arrangement_zero(self):
        df = build_trait_accuracy_per_species(_make_multi_specimen_records())
        row = df[(df["scientific_name"] == "SpB") & (df["category"] == "leaf_arrangement")].iloc[0]
        assert row["n_correct"] == 0
        assert row["n_incorrect"] == 2
        assert row["n_skipped_no_pred"] == 1
        assert row["accuracy"] == 0.0

    def test_all_skipped_none_accuracy(self):
        records = [
            TraitGradeRecord("X1", "SpX", "latex__present", "latex", None, None, TraitVerdict.SKIPPED_UNCODED),
            TraitGradeRecord("X2", "SpX", "latex__present", "latex", None, None, TraitVerdict.SKIPPED_UNCODED),
        ]
        df = build_trait_accuracy_per_species(records)
        row = df[(df["scientific_name"] == "SpX") & (df["category"] == "latex")].iloc[0]
        assert row["accuracy"] is None
        assert row["n_graded"] == 0

    def test_empty_records(self):
        df = build_trait_accuracy_per_species([])
        assert df.empty


class TestBuildWithinSpeciesAgreement:
    def test_perfect_agreement(self):
        """SpA leaf_type: all 3 specimens predict simple → 100% agreement."""
        df = build_within_species_agreement(_make_multi_specimen_records())
        row = df[(df["scientific_name"] == "SpA") & (df["category"] == "leaf_type")].iloc[0]
        assert row["n_specimens"] == 3
        assert row["n_with_predictions"] == 3
        assert row["mode_prediction"] == "leaf_type__simple"
        assert row["n_agreeing"] == 3
        assert row["agreement"] == 1.0

    def test_partial_agreement(self):
        """SpA leaf_arrangement: A1,A2 predict alternate, A3 predicts opposite."""
        df = build_within_species_agreement(_make_multi_specimen_records())
        row = df[(df["scientific_name"] == "SpA") & (df["category"] == "leaf_arrangement")].iloc[0]
        assert row["n_with_predictions"] == 3
        assert row["mode_prediction"] == "leaf_arrangement__alternate"
        assert row["n_agreeing"] == 2
        assert abs(row["agreement"] - 2 / 3) < 0.001

    def test_no_predictions(self):
        """All specimens have no prediction → agreement is None."""
        records = [
            TraitGradeRecord("X1", "SpX", "latex__present", "latex", 1.0, None, TraitVerdict.SKIPPED_NO_PREDICTION),
            TraitGradeRecord("X2", "SpX", "latex__present", "latex", 1.0, None, TraitVerdict.SKIPPED_NO_PREDICTION),
        ]
        df = build_within_species_agreement(records)
        row = df.iloc[0]
        assert row["n_with_predictions"] == 0
        assert row["agreement"] is None

    def test_single_specimen(self):
        """One specimen with a prediction → 100% agreement."""
        records = [
            TraitGradeRecord("X1", "SpX", "leaf_type__simple", "leaf_type", 1.0, 1, TraitVerdict.CORRECT),
        ]
        df = build_within_species_agreement(records)
        row = df.iloc[0]
        assert row["n_specimens"] == 1
        assert row["n_with_predictions"] == 1
        assert row["agreement"] == 1.0

    def test_empty_records(self):
        df = build_within_species_agreement([])
        assert df.empty
