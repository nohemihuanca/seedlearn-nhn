"""Tests for catalog loading and ImageRecord creation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from seedlearn.data.catalog import ImageRecord, format_label, load_catalog


class TestImageRecord:
    """Tests for ImageRecord dataclass."""

    def test_default_label_id(self):
        r = ImageRecord(
            image_path=Path("/fake/path.jpg"),
            label="Fabaceae",
            family="Fabaceae",
            genus="Acacia",
            species="Acacia dealbata",
        )
        assert r.label_id == -1

    def test_custom_label_id(self):
        r = ImageRecord(
            image_path=Path("/fake/path.jpg"),
            label="Fabaceae",
            family="Fabaceae",
            genus="Acacia",
            species="Acacia dealbata",
            label_id=42,
        )
        assert r.label_id == 42


class TestFormatLabel:
    """Tests for format_label function."""

    def _make_row(self) -> pd.Series:
        return pd.Series({
            "FAMILY": "Fabaceae",
            "GENUS": "Acacia",
            "SPECIES": "dealbata",
        })

    def test_family_rank(self):
        assert format_label(self._make_row(), "family") == "Fabaceae"

    def test_genus_rank(self):
        assert format_label(self._make_row(), "genus") == "Acacia"

    def test_species_rank(self):
        assert format_label(self._make_row(), "species") == "Acacia dealbata"

    def test_underscore_handling(self):
        row = pd.Series({
            "FAMILY": "Fabaceae",
            "GENUS": "Acacia_Mill",
            "SPECIES": "dealbata_Link",
        })
        assert format_label(row, "genus") == "Acacia Mill"


class TestLoadCatalog:
    """Tests for load_catalog function."""

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_catalog(tmp_path / "nonexistent.csv")

    def test_missing_columns_raises(self, tmp_path):
        csv_path = tmp_path / "bad_catalog.csv"
        pd.DataFrame({"foo": [1]}).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            load_catalog(csv_path)

    def test_valid_catalog_loads(self, tmp_path):
        csv_path = tmp_path / "catalog.csv"
        pd.DataFrame({
            "training_absolute_path": [str(tmp_path)],
            "FAMILY": ["Fabaceae"],
            "GENUS": ["Acacia"],
            "SPECIES": ["dealbata"],
        }).to_csv(csv_path, index=False)

        df = load_catalog(csv_path)
        assert len(df) == 1
        assert "FAMILY" in df.columns
