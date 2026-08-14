"""Tests for benchmark scoring data loaders."""

import csv
import json

import pytest
from pathlib import Path

from tests.benchmarks.scoring.loader import (
    GroundTruthEntry,
    ResultEntry,
    TRAIT_COLUMNS,
    load_ground_truth,
    load_result_dir,
)


@pytest.fixture
def gt_csv(tmp_path: Path) -> Path:
    """Create a minimal ground truth CSV."""
    path = tmp_path / "gt.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "specimen_key",
                "specimen_id",
                "family",
                "genus",
                "species",
                "scientific_name",
                "num_images",
                "leaf_complexity",
                "leaf_complexity_match_type",
                "leaf_arrangement",
                "leaf_arrangement_match_type",
                "leaf_margin",
                "leaf_margin_match_type",
                "stipules",
                "stipules_match_type",
                "latex",
                "latex_match_type",
                "multi_label_count",
            ]
        )
        writer.writerow(
            [
                "Fabaceae_Inga_punctata",
                "PPINGAPU6",
                "Fabaceae",
                "Inga",
                "punctata",
                "Inga punctata",
                "5",
                "compound",
                "exact",
                "alternate",
                "exact",
                "entire | toothed",
                "multi_label",
                "present",
                "exact",
                "absent",
                "exact",
                "1",
            ]
        )
    return path


@pytest.fixture
def gt_csv_two_rows(tmp_path: Path) -> Path:
    """Ground truth CSV with two specimens."""
    path = tmp_path / "gt2.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "specimen_key",
                "specimen_id",
                "family",
                "genus",
                "species",
                "scientific_name",
                "num_images",
                "leaf_complexity",
                "leaf_complexity_match_type",
                "leaf_arrangement",
                "leaf_arrangement_match_type",
                "leaf_margin",
                "leaf_margin_match_type",
                "stipules",
                "stipules_match_type",
                "latex",
                "latex_match_type",
                "multi_label_count",
            ]
        )
        writer.writerow(
            [
                "Fabaceae_Inga_punctata",
                "PPINGAPU6",
                "Fabaceae",
                "Inga",
                "punctata",
                "Inga punctata",
                "5",
                "compound",
                "exact",
                "alternate",
                "exact",
                "entire",
                "exact",
                "present",
                "exact",
                "absent",
                "exact",
                "0",
            ]
        )
        writer.writerow(
            [
                "Moraceae_Brosimum_alicastrum",
                "PPBROALI3",
                "Moraceae",
                "Brosimum",
                "alicastrum",
                "Brosimum alicastrum",
                "3",
                "simple",
                "exact",
                "alternate",
                "exact",
                "entire",
                "exact",
                "absent",
                "exact",
                "present",
                "exact",
                "0",
            ]
        )
    return path


@pytest.fixture
def result_dir_multi(tmp_path: Path) -> Path:
    """Result directory in multi-image mode."""
    d = tmp_path / "run1" / "multi"
    d.mkdir(parents=True)
    config = {"model": "test-model", "prompt_style": "sys4", "mode": "multi"}
    (tmp_path / "run1" / "config.json").write_text(json.dumps(config))
    result = {
        "specimen_key": "Fabaceae_Inga_punctata",
        "traits": {
            "leaf_complexity": "compound",
            "leaf_arrangement": "alternate",
            "leaf_margin": "entire",
            "stipules": "present",
            "latex": "absent",
        },
    }
    (d / "Fabaceae_Inga_punctata.json").write_text(json.dumps(result))
    return tmp_path / "run1"


@pytest.fixture
def result_dir_single(tmp_path: Path) -> Path:
    """Result directory in single-image mode."""
    run_dir = tmp_path / "run2"
    specimen_dir = run_dir / "single" / "Fabaceae_Inga_punctata"
    specimen_dir.mkdir(parents=True)
    config = {"model": "test-model", "prompt_style": "sys4", "mode": "single"}
    (run_dir / "config.json").write_text(json.dumps(config))
    for i in range(3):
        result = {
            "specimen_key": "Fabaceae_Inga_punctata",
            "image_id": f"img_{i}",
            "traits": {"leaf_complexity": "compound"},
        }
        (specimen_dir / f"img_{i}.json").write_text(json.dumps(result))
    return run_dir


class TestGroundTruthEntry:
    """Tests for GroundTruthEntry dataclass."""

    def test_fields(self) -> None:
        entry = GroundTruthEntry(
            specimen_key="test",
            specimen_id="T1",
            family="Fabaceae",
            scientific_name="Inga punctata",
            num_images=3,
            traits={"leaf_complexity": "compound"},
            match_types={"leaf_complexity": "exact"},
        )
        assert entry.specimen_key == "test"
        assert entry.multi_label_count == 0


class TestResultEntry:
    """Tests for ResultEntry dataclass."""

    def test_defaults(self) -> None:
        entry = ResultEntry(specimen_key="test")
        assert entry.traits == {}
        assert entry.raw_response == ""
        assert entry.image_id == ""


class TestTraitColumns:
    """Tests for TRAIT_COLUMNS constant."""

    def test_has_five_traits(self) -> None:
        assert len(TRAIT_COLUMNS) == 5

    def test_expected_traits(self) -> None:
        assert "leaf_complexity" in TRAIT_COLUMNS
        assert "latex" in TRAIT_COLUMNS


class TestLoadGroundTruth:
    """Tests for load_ground_truth()."""

    def test_loads_single_entry(self, gt_csv: Path) -> None:
        entries = load_ground_truth(gt_csv)
        assert len(entries) == 1
        e = entries["Fabaceae_Inga_punctata"]
        assert e.specimen_key == "Fabaceae_Inga_punctata"
        assert e.specimen_id == "PPINGAPU6"
        assert e.family == "Fabaceae"
        assert e.scientific_name == "Inga punctata"
        assert e.num_images == 5

    def test_trait_values(self, gt_csv: Path) -> None:
        entries = load_ground_truth(gt_csv)
        e = entries["Fabaceae_Inga_punctata"]
        assert e.traits["leaf_complexity"] == "compound"
        assert e.traits["leaf_margin"] == "entire | toothed"
        assert e.traits["stipules"] == "present"
        assert e.traits["latex"] == "absent"

    def test_match_types(self, gt_csv: Path) -> None:
        entries = load_ground_truth(gt_csv)
        e = entries["Fabaceae_Inga_punctata"]
        assert e.match_types["leaf_complexity"] == "exact"
        assert e.match_types["leaf_margin"] == "multi_label"

    def test_multi_label_count(self, gt_csv: Path) -> None:
        entries = load_ground_truth(gt_csv)
        e = entries["Fabaceae_Inga_punctata"]
        assert e.multi_label_count == 1

    def test_loads_multiple_entries(self, gt_csv_two_rows: Path) -> None:
        entries = load_ground_truth(gt_csv_two_rows)
        assert len(entries) == 2
        assert "Fabaceae_Inga_punctata" in entries
        assert "Moraceae_Brosimum_alicastrum" in entries

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Ground truth file not found"):
            load_ground_truth(Path("/nonexistent.csv"))

    def test_empty_trait_excluded(self, tmp_path: Path) -> None:
        """Traits with empty values should not appear in the traits dict."""
        path = tmp_path / "sparse.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "specimen_key",
                    "leaf_complexity",
                    "leaf_complexity_match_type",
                    "leaf_arrangement",
                    "leaf_arrangement_match_type",
                ]
            )
            writer.writerow(["test_specimen", "simple", "exact", "", "exact"])
        entries = load_ground_truth(path)
        e = entries["test_specimen"]
        assert "leaf_complexity" in e.traits
        assert "leaf_arrangement" not in e.traits


class TestLoadResultDir:
    """Tests for load_result_dir()."""

    def test_loads_multi_results(self, result_dir_multi: Path) -> None:
        config, results = load_result_dir(result_dir_multi, mode="multi")
        assert config["model"] == "test-model"
        assert "Fabaceae_Inga_punctata" in results
        r = results["Fabaceae_Inga_punctata"]
        assert isinstance(r, ResultEntry)
        assert r.traits["leaf_complexity"] == "compound"

    def test_loads_single_results(self, result_dir_single: Path) -> None:
        config, results = load_result_dir(result_dir_single, mode="single")
        assert config["model"] == "test-model"
        assert "Fabaceae_Inga_punctata" in results
        entries = results["Fabaceae_Inga_punctata"]
        assert len(entries) == 3
        assert all(isinstance(e, ResultEntry) for e in entries)

    def test_missing_dir_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Result directory not found"):
            load_result_dir(Path("/nonexistent"), mode="multi")

    def test_no_config_returns_empty_dict(self, tmp_path: Path) -> None:
        """Directory without config.json should return empty config."""
        d = tmp_path / "noconfig" / "multi"
        d.mkdir(parents=True)
        config, results = load_result_dir(tmp_path / "noconfig", mode="multi")
        assert config == {}
        assert results == {}

    def test_empty_single_dir_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Single mode without single/ subdirectory should log a warning."""
        d = tmp_path / "empty_run"
        d.mkdir()
        import logging

        with caplog.at_level(logging.WARNING):
            config, results = load_result_dir(d, mode="single")
        assert results == {}
        assert "No single/ subdirectory" in caplog.text
