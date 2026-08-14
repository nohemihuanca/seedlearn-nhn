"""Tests for explicit specimen selection (overlap.load_specimens_by_id)."""

from pathlib import Path

import pandas as pd
import pytest

from seedlearn.benchmarking import overlap
from seedlearn.benchmarking.overlap import load_specimens_by_id, read_specimen_ids

REPO = Path(__file__).resolve().parents[2]
CURATOR = REPO / "trait_grading/keys/curator_taxonomic_key.csv"
CATALOG = next(
    (REPO / "data/raw/2026-01-29/sorted_12K/metadata").glob("species_catalog_*.csv"),
    None,
) if (REPO / "data/raw/2026-01-29/sorted_12K/metadata").exists() else None


def test_read_specimen_ids_from_curator_key(tmp_path):
    p = tmp_path / "key.csv"
    p.write_text("anonymous_id,individual_code\ni1,SR1\ni2,SR2\ni3,SR1\n")  # SR1 dup
    assert read_specimen_ids(p) == ["SR1", "SR2"]


def test_read_specimen_ids_from_plain_list(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("SR1\nSR2\n\nSR2\n")
    assert read_specimen_ids(p) == ["SR1", "SR2"]


def test_load_specimens_by_id_selects_and_reports_missing(tmp_path, monkeypatch):
    catalog = pd.DataFrame(
        {
            "ID_YPS": ["SR1", "SR1", "SR2", "SR9"],
            "GENUS": ["Aphelandra", "Aphelandra", "Inga", "Other"],
            "SPECIES": ["scabra", "scabra", "edulis", "thing"],
            "FAMILY": ["Acanthaceae", "Acanthaceae", "Fabaceae", "Otheraceae"],
            "training_absolute_path": ["/d/SR1a", "/d/SR1b", "/d/SR2", "/d/SR9"],
        }
    )
    monkeypatch.setattr(overlap, "load_catalog", lambda _p: catalog)
    monkeypatch.setattr(overlap, "iter_image_paths", lambda d: [f"{d}/img.jpg"])

    src = tmp_path / "key.csv"
    src.write_text("individual_code\nSR1\nSR2\nSR_MISSING\n")
    specimens, missing = load_specimens_by_id(Path("catalog.csv"), src)

    assert [s.specimen_id for s in specimens] == ["SR1", "SR2"]  # requested order
    assert missing == ["SR_MISSING"]
    sr1 = specimens[0]
    assert sr1.scientific_name == "Aphelandra scabra"
    assert sr1.match_method == "curator_selection"
    assert len(sr1.image_paths) == 2  # both catalog rows contributed images


def test_load_specimens_skips_when_no_images(tmp_path, monkeypatch):
    catalog = pd.DataFrame(
        {
            "ID_YPS": ["SR1"],
            "GENUS": ["Inga"],
            "SPECIES": ["edulis"],
            "FAMILY": ["Fabaceae"],
            "training_absolute_path": ["/d/SR1"],
        }
    )
    monkeypatch.setattr(overlap, "load_catalog", lambda _p: catalog)
    monkeypatch.setattr(overlap, "iter_image_paths", lambda d: [])  # no images

    src = tmp_path / "key.csv"
    src.write_text("individual_code\nSR1\n")
    specimens, missing = load_specimens_by_id(Path("catalog.csv"), src)
    assert specimens == []
    assert missing == ["SR1"]


@pytest.mark.skipif(
    CATALOG is None or not CURATOR.exists(), reason="catalog/curator key absent"
)
def test_real_catalog_covers_annotated_set():
    specimens, missing = load_specimens_by_id(CATALOG, CURATOR)
    # All 114 annotated specimens should resolve from the catalog.
    assert len(specimens) + len(missing) == 114
    assert all(s.image_paths for s in specimens)
