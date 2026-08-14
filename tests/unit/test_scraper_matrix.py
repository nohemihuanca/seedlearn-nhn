"""Tests for trait matrix construction from scrape results."""
import json
from pathlib import Path

import pandas as pd
import pytest

from seedlearn.scraper.matrix import (
    build_trait_matrix,
    load_trait_matrix,
    save_trait_matrix,
)
from seedlearn.scraper.schema import (
    FilterCategory,
    FilterOption,
    SpeciesEntry,
)


@pytest.fixture
def sample_species() -> list[SpeciesEntry]:
    return [
        SpeciesEntry(100, "Alpha beta", "FamilyA"),
        SpeciesEntry(200, "Gamma delta", "FamilyA"),
        SpeciesEntry(300, "Epsilon zeta", "FamilyB"),
    ]


@pytest.fixture
def sample_categories() -> list[FilterCategory]:
    return [
        FilterCategory(1, "habit", [
            FilterOption(1, 1, "tree"),
            FilterOption(1, 2, "shrub"),
        ]),
        FilterCategory(3, "leaf_type", [
            FilterOption(3, 1, "simple"),
            FilterOption(3, 2, "compound"),
        ]),
    ]


@pytest.fixture
def sample_filter_results() -> dict[str, set[int]]:
    """Maps attr_value -> set of taxon_ids that matched."""
    return {
        "1-1": {100, 300},       # habit: tree
        "1-2": {200},            # habit: shrub
        "3-1": {100, 200, 300},  # leaf_type: simple
        "3-2": {100},            # leaf_type: compound (100 is BOTH)
    }


class TestBuildTraitMatrix:
    def test_matrix_shape(
        self, sample_species, sample_filter_results, sample_categories,
    ) -> None:
        df = build_trait_matrix(
            sample_species, sample_filter_results, sample_categories,
        )
        assert len(df) == 3
        # taxon_id + family + scientific_name + 4 traits + 2 uncoded
        assert len(df.columns) == 9

    def test_multi_label_species(
        self, sample_species, sample_filter_results, sample_categories,
    ) -> None:
        df = build_trait_matrix(
            sample_species, sample_filter_results, sample_categories,
        )
        row = df[df["taxon_id"] == 100].iloc[0]
        assert row["habit__tree"] == 1
        assert row["habit__shrub"] == 0
        assert row["leaf_type__simple"] == 1
        assert row["leaf_type__compound"] == 1  # Multi-label!
        assert row["habit__uncoded"] == 0
        assert row["leaf_type__uncoded"] == 0

    def test_absent_species_get_zero(
        self, sample_species, sample_filter_results, sample_categories,
    ) -> None:
        df = build_trait_matrix(
            sample_species, sample_filter_results, sample_categories,
        )
        row = df[df["taxon_id"] == 200].iloc[0]
        assert row["habit__tree"] == 0
        assert row["habit__shrub"] == 1
        assert row["habit__uncoded"] == 0  # Has habit data (shrub=1)

    def test_uncoded_species_flagged(
        self, sample_species, sample_categories,
    ) -> None:
        """Species 300 not in ANY habit filter -> uncoded for habit."""
        custom_results = {
            "1-1": {100},            # habit: tree — only 100
            "1-2": {200},            # habit: shrub — only 200
            "3-1": {100, 200, 300},  # leaf_type: simple — all three
            "3-2": {100},            # leaf_type: compound
        }
        df = build_trait_matrix(
            sample_species, custom_results, sample_categories,
        )
        row_300 = df[df["taxon_id"] == 300].iloc[0]
        assert row_300["habit__tree"] == 0
        assert row_300["habit__shrub"] == 0
        assert row_300["habit__uncoded"] == 1   # All habit options are 0
        assert row_300["leaf_type__uncoded"] == 0  # Has leaf_type data

    def test_column_order(
        self, sample_species, sample_filter_results, sample_categories,
    ) -> None:
        df = build_trait_matrix(
            sample_species, sample_filter_results, sample_categories,
        )
        expected = [
            "taxon_id", "family", "scientific_name",
            "habit__tree", "habit__shrub",
            "leaf_type__simple", "leaf_type__compound",
            "habit__uncoded", "leaf_type__uncoded",
        ]
        assert list(df.columns) == expected

    def test_taxon_id_dtype_is_int(
        self, sample_species, sample_filter_results, sample_categories,
    ) -> None:
        df = build_trait_matrix(
            sample_species, sample_filter_results, sample_categories,
        )
        assert df["taxon_id"].dtype in ("int64", "int32")


class TestSaveLoadTraitMatrix:
    def test_round_trip(
        self, tmp_path, sample_species, sample_filter_results, sample_categories,
    ) -> None:
        df = build_trait_matrix(
            sample_species, sample_filter_results, sample_categories,
        )
        save_trait_matrix(
            df, tmp_path, "test_key",
            species_count=3, categories=sample_categories,
        )

        csv_path = tmp_path / "test_key_trait_matrix.csv"
        meta_path = tmp_path / "test_key_scrape_metadata.json"
        assert csv_path.exists()
        assert meta_path.exists()

        loaded = load_trait_matrix(csv_path)
        pd.testing.assert_frame_equal(df, loaded)

        meta = json.loads(meta_path.read_text())
        assert meta["species_count_reported"] == 3
        assert meta["species_count_scraped"] == 3
        assert len(meta["trait_columns"]) == 6  # 4 traits + 2 uncoded
        assert len(meta["categories"]) == 2

    def test_metadata_has_timestamps(
        self, tmp_path, sample_species, sample_filter_results, sample_categories,
    ) -> None:
        df = build_trait_matrix(
            sample_species, sample_filter_results, sample_categories,
        )
        save_trait_matrix(
            df, tmp_path, "ts_test",
            species_count=3, categories=sample_categories,
        )
        meta = json.loads((tmp_path / "ts_test_scrape_metadata.json").read_text())
        assert "scraped_at" in meta
