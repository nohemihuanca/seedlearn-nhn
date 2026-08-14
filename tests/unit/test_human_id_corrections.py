"""Tests for the reviewable Roni-ID correction layer (human.id_corrections)."""

from pathlib import Path

import pytest

from seedlearn.benchmarking.human.annotations import load_annotations, load_curator_key
from seedlearn.benchmarking.human.id_corrections import (
    Correction,
    load_corrections,
    match_correction,
    stale_corrections,
)
from seedlearn.benchmarking.human.id_grading import grade_human_ids, id_accuracy

REPO = Path(__file__).resolve().parents[2]
RONI = REPO / "trait_grading/annotations/roni_bianco.xlsx"
CURATOR = REPO / "trait_grading/keys/curator_taxonomic_key.csv"
CORRECTIONS = REPO / "trait_grading/id_corrections.csv"


def test_load_corrections_parses_rows(tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text(
        "specimen_id,rank,roni_original,canonical,category,note\n"
        "SR1,species,hayessi,hayesii,typo,swap\n"
        "SR2,genus,Monteverdia,Maytenus,synonym,\n"
    )
    m = load_corrections(csv)
    assert set(m) == {("SR1", "species"), ("SR2", "genus")}
    assert m[("SR1", "species")].canonical == "hayesii"
    assert m[("SR2", "genus")].category == "synonym"


def test_load_corrections_skips_bad_rank_or_category(tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text(
        "specimen_id,rank,roni_original,canonical,category,note\n"
        "SR1,subspecies,x,y,typo,\n"      # bad rank
        "SR2,genus,x,y,guess,\n"           # bad category
        "SR3,genus,x,y,variant,\n"         # valid
    )
    m = load_corrections(csv)
    assert set(m) == {("SR3", "genus")}


def test_load_corrections_missing_file_is_empty(tmp_path):
    assert load_corrections(tmp_path / "nope.csv") == {}


def test_match_correction_requires_original_and_canonical():
    c = Correction("SR1", "species", "hayessi", "hayesii", "typo")
    assert match_correction(c, "hayessi", "hayesii") is True
    assert match_correction(c, "HAYESSI", " hayesii ") is True   # normalized
    assert match_correction(c, "wrongword", "hayesii") is False  # original mismatch
    assert match_correction(c, "hayessi", "different") is False  # canonical mismatch
    assert match_correction(None, "hayessi", "hayesii") is False


def test_stale_corrections_flags_drifted_entries():
    corrections = {
        ("SR1", "species"): Correction("SR1", "species", "hayessi", "hayesii", "typo"),
        ("SR2", "genus"): Correction("SR2", "genus", "Oldname", "Newname", "synonym"),
    }
    originals = {("SR1", "species"): "hayessi", ("SR2", "genus"): "Somethingelse"}
    stale = stale_corrections(corrections, originals)
    assert [c.specimen_id for c in stale] == ["SR2"]


@pytest.mark.skipif(
    not (RONI.exists() and CURATOR.exists() and CORRECTIONS.exists()),
    reason="study data / corrections file absent",
)
def test_real_corrections_lift_accuracy():
    curator = load_curator_key(CURATOR)
    records, _ = load_annotations(RONI, "roni", curator)
    corr = load_corrections(CORRECTIONS)
    acc = id_accuracy(grade_human_ids(records, "roni", corr))
    # Corrected scores must be >= raw and match the expected lift (species includes
    # the four verified epithet synonyms: Tontelea, Coussarea, Palicourea, Piparea).
    assert acc["corrected_family_accuracy"] >= acc["family_accuracy"]
    assert acc["corrected_genus_accuracy"] >= acc["genus_accuracy"]
    assert acc["corrected_species_accuracy"] >= acc["species_accuracy"]
    assert round(acc["corrected_family_accuracy"], 3) == pytest.approx(0.982, abs=0.01)
    assert round(acc["corrected_genus_accuracy"], 3) == pytest.approx(0.965, abs=0.01)
    assert round(acc["corrected_species_accuracy"], 3) == pytest.approx(0.939, abs=0.01)
