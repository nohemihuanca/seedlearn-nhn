"""Tests for per-view -> per-specimen mode aggregation (human.aggregate)."""

from pathlib import Path

import pytest

from seedlearn.benchmarking.human.aggregate import aggregate_records, modal_value
from seedlearn.benchmarking.human.annotations import (
    AnnotationRecord,
    load_annotations,
    load_curator_key,
)
from seedlearn.benchmarking.human.value_map import MISSING


def _rec(view, **traits):
    return AnnotationRecord(
        annotator="roni",
        anonymous_id="individual_001",
        view_id=view,
        traits=traits,
        specimen_id="SRAPHEDE2",
    )


def test_modal_value_simple_majority():
    assert modal_value(["whorled", "whorled", "opposite"]) == "whorled"


def test_modal_value_tie_breaks_by_first_occurrence():
    # opposite appears first -> wins the tie
    assert modal_value(["opposite", "whorled"]) == "opposite"
    assert modal_value(["whorled", "opposite"]) == "whorled"


def test_modal_value_all_missing():
    assert modal_value([MISSING, MISSING]) == MISSING
    assert modal_value([]) == MISSING


def test_modal_value_ignores_missing():
    assert modal_value(["entire", MISSING, "entire"]) == "entire"


def test_aggregate_mode_and_distribution():
    records = [
        _rec("view_01", leaf_relative_position="verticilada"),
        _rec("view_02", leaf_relative_position="verticilada"),
        _rec("view_03", leaf_relative_position="opuesta"),
    ]
    aggs = aggregate_records(records)
    assert len(aggs) == 1
    agg = aggs[0]
    assert agg.specimen_id == "SRAPHEDE2"
    trait = agg.traits["leaf_relative_position"]
    assert trait.mode == "whorled"
    assert trait.canonical_values == ["whorled", "whorled", "opposite"]
    assert trait.raw_values == ["verticilada", "verticilada", "opuesta"]
    assert trait.n_views == 3
    assert trait.n_present == 3


def test_aggregate_mixed_present_and_missing():
    records = [
        _rec("view_01", leaf_margin="entero"),
        _rec("view_02"),  # no margin recorded -> MISSING
        _rec("view_03", leaf_margin="entero"),
    ]
    agg = aggregate_records(records)[0]
    trait = agg.traits["leaf_margin"]
    assert trait.mode == "entire"
    assert trait.n_views == 3
    assert trait.n_present == 2
    assert trait.canonical_values == ["entire", MISSING, "entire"]


def test_aggregate_groups_by_specimen_and_annotator():
    records = [
        AnnotationRecord("roni", "individual_001", "view_01", {"leaf_margin": "entero"}),
        AnnotationRecord("carmen", "individual_001", "view_01", {"leaf_margin": "dentado"}),
    ]
    aggs = aggregate_records(records)
    assert {a.annotator for a in aggs} == {"roni", "carmen"}
    by_ann = {a.annotator: a for a in aggs}
    assert by_ann["roni"].traits["leaf_margin"].mode == "entire"
    assert by_ann["carmen"].traits["leaf_margin"].mode == "toothed"


# --------------------------------------------------------------------------- #

REPO = Path(__file__).resolve().parents[2]
RONI = REPO / "trait_grading/annotations/roni_bianco.xlsx"
CURATOR = REPO / "trait_grading/keys/curator_taxonomic_key.csv"


@pytest.mark.skipif(not (RONI.exists() and CURATOR.exists()), reason="study data absent")
def test_real_aggregate_one_row_per_individual():
    curator = load_curator_key(CURATOR)
    records, _ = load_annotations(RONI, "roni", curator)
    aggs = aggregate_records(records)
    assert len(aggs) == 114
    assert all(a.specimen_id for a in aggs)
    # every aggregate carries the full gradable trait set
    assert all("leaf_margin" in a.traits for a in aggs)
