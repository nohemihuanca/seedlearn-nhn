"""Tests for STRI-axis comparison (human.stri_compare)."""

from pathlib import Path

import pytest

from seedlearn.benchmarking.human.stri_compare import (
    STRI_TRAITS,
    accuracy_vs_stri,
    build_stri_lookup,
    load_stri_matrix,
    stri_pair_details,
)
from seedlearn.benchmarking.human.value_map import MISSING

REPO = Path(__file__).resolve().parents[2]
STRI = REPO / ("data/traits/stri_web_keys/per_key_trait_matrices/"
               "cl185_complete_tree_species_of_panama_trait_matrix.csv")
CURATOR = REPO / "trait_grading/keys/curator_taxonomic_key.csv"


def test_stri_traits_are_the_five_coded_ones():
    assert set(STRI_TRAITS) == {
        "leaf_relative_position", "leaf_complexity_type",
        "leaf_margin", "stipules", "latex",
    }


def test_build_lookup_allowed_sets_and_uncoded():
    stri_rows = {
        "aphelandra scabra": {
            "leaf_margin__entire": 1, "leaf_margin__toothed": 1, "leaf_margin__lobed": 0,
            "leaf_margin__uncoded": 0,
            "leaf_arrangement__alternate": 0, "leaf_arrangement__opposite": 1,
            "leaf_arrangement__whorled_or_clustered": 0, "leaf_arrangement__uncoded": 0,
            "leaf_type__simple": 1, "leaf_type__compound": 0, "leaf_type__uncoded": 0,
            "stipules__present": 0, "stipules__absent": 1, "stipules__uncoded": 0,
            "latex__present": 0, "latex__absent": 0, "latex__uncoded": 1,  # uncoded
        }
    }
    lookup, n = build_stri_lookup({"SR1": "Aphelandra scabra"}, stri_rows)
    assert n == 1
    # multi-label margin: both entire and toothed allowed
    assert lookup["SR1"]["leaf_margin"] == {"entire", "toothed"}
    assert lookup["SR1"]["leaf_relative_position"] == {"opposite"}
    # latex is uncoded -> None
    assert lookup["SR1"]["latex"] is None


def test_build_lookup_unmatched_species_absent():
    lookup, n = build_stri_lookup({"SR1": "Unknown species"}, {})
    assert lookup == {} and n == 0


def test_accuracy_match_any_and_skips():
    stri_lookup = {
        "SR1": {"leaf_margin": {"entire", "toothed"}, "stipules": {"absent"},
                "latex": None, "leaf_relative_position": {"opposite"},
                "leaf_complexity_type": {"simple"}},
        "SR2": {"leaf_margin": {"entire"}, "stipules": {"present"},
                "latex": {"absent"}, "leaf_relative_position": {"alternate"},
                "leaf_complexity_type": {"simple"}},
    }
    source = {
        "SR1": {"leaf_margin": "toothed", "stipules": "present", "latex": "present"},
        "SR2": {"leaf_margin": "entire", "stipules": "present", "latex": MISSING},
    }
    res = {r.trait_key: r for r in accuracy_vs_stri(source, stri_lookup, "model")}
    # margin: SR1 toothed in {entire,toothed}=ok, SR2 entire in {entire}=ok -> 2/2
    assert res["leaf_margin"].n_compared == 2 and res["leaf_margin"].n_correct == 2
    # stipules: SR1 present not in {absent}=miss, SR2 present in {present}=ok -> 1/2
    assert res["stipules"].n_compared == 2 and res["stipules"].n_correct == 1
    # latex: SR1 uncoded(None) skipped, SR2 MISSING skipped -> 0 compared
    assert res["latex"].n_compared == 0 and res["latex"].accuracy is None


def test_kappa_over_single_label_subset_only():
    # SR1 is multi-label (excluded from kappa); SR2/SR3/SR4 are single-label
    # (both sides single-valued, so kappa is well defined over just those three).
    stri_lookup = {
        "SR1": {"leaf_margin": {"entire", "toothed"}},
        "SR2": {"leaf_margin": {"entire"}},
        "SR3": {"leaf_margin": {"toothed"}},
        "SR4": {"leaf_margin": {"entire"}},
    }
    source = {
        "SR1": {"leaf_margin": "entire"},
        "SR2": {"leaf_margin": "entire"},
        "SR3": {"leaf_margin": "toothed"},
        "SR4": {"leaf_margin": "toothed"},  # disagrees with STRI's {entire}
    }
    res = {r.trait_key: r for r in accuracy_vs_stri(source, stri_lookup, "model")}
    margin = res["leaf_margin"]
    assert margin.n_compared == 4          # all four counted in match-any accuracy
    assert margin.n_kappa == 3             # only the single-label SR2/SR3/SR4
    assert margin.cohen_kappa is not None  # two labels present -> kappa defined


def test_stri_pair_details_rows_match_denominator_and_species():
    stri_lookup = {
        "SR1": {"leaf_margin": {"entire", "toothed"}, "stipules": {"absent"},
                "latex": None, "leaf_relative_position": {"opposite"},
                "leaf_complexity_type": {"simple"}},
        "SR2": {"leaf_margin": {"entire"}, "stipules": {"present"},
                "latex": {"absent"}, "leaf_relative_position": {"alternate"},
                "leaf_complexity_type": {"simple"}},
    }
    source = {
        "SR1": {"leaf_margin": "toothed"},
        "SR2": {"leaf_margin": "lobed"},  # not in {entire} -> mismatch
    }
    species = {"SR1": "Aphelandra scabra", "SR2": "Ruellia fulgida"}
    details = stri_pair_details(source, stri_lookup, "roni", species)
    margin = details["leaf_margin"]
    assert {d.specimen_id for d in margin} == {"SR1", "SR2"}
    # mismatches first
    assert margin[0].specimen_id == "SR2" and margin[0].correct is False
    assert margin[0].species == "Ruellia fulgida"
    assert margin[0].allowed == ["entire"]
    # SR1 correct
    sr1 = next(d for d in margin if d.specimen_id == "SR1")
    assert sr1.correct is True and set(sr1.allowed) == {"entire", "toothed"}


@pytest.mark.skipif(not (STRI.exists() and CURATOR.exists()), reason="STRI/curator absent")
def test_real_stri_matches_annotated_species():
    import csv

    rows = load_stri_matrix(str(STRI))
    assert len(rows) > 1000
    specimen_to_species = {}
    for r in csv.DictReader(open(CURATOR)):
        specimen_to_species[r["individual_code"]] = f"{r['genus']} {r['species']}"
    lookup, n_matched = build_stri_lookup(specimen_to_species, rows)
    # We established ~76 of 114 annotated species match STRI directly.
    assert 60 <= n_matched <= 114
