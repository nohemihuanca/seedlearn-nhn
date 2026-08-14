"""Tests for human species-ID grading + the extracted compare_taxonomy core."""

from pathlib import Path

import pytest

from seedlearn.benchmarking.human.annotations import (
    AnnotationRecord,
    load_annotations,
    load_curator_key,
)
from seedlearn.benchmarking.human.id_corrections import Correction
from seedlearn.benchmarking.human.id_grading import (
    grade_human_ids,
    id_accuracy,
)
from seedlearn.benchmarking.id_grader import compare_taxonomy


def _roni(anon, fam, gen, sp, true_fam, true_gen, true_sp, specimen="SR1"):
    return AnnotationRecord(
        annotator="roni",
        anonymous_id=anon,
        view_id="view_01",
        id_family=fam,
        id_genus=gen,
        id_species=sp,
        specimen_id=specimen,
        true_family=true_fam,
        true_genus=true_gen,
        true_species=true_sp,
    )


def test_compare_taxonomy_all_correct():
    assert compare_taxonomy(
        "Acanthaceae", "Aphelandra", "Aphelandra scabra",
        "Acanthaceae", "Aphelandra", "scabra",
    ) == (True, True, True)


def test_compare_taxonomy_genus_right_species_wrong():
    fam, gen, sp = compare_taxonomy(
        "Acanthaceae", "Aphelandra", "Aphelandra scabra",
        "Acanthaceae", "Aphelandra", "costaricensis",
    )
    assert (fam, gen, sp) == (True, True, False)


def test_compare_taxonomy_epithet_fallback_and_case():
    # epithet-only pred vs full binomial truth, case/space differences
    assert compare_taxonomy(
        "fabaceae", "Inga", "Inga marginata", "Fabaceae ", " inga ", "MARGINATA"
    ) == (True, True, True)


def test_grade_human_ids_one_per_individual():
    records = [
        _roni("i1", "Acanthaceae", "Aphelandra", "scabra", "Acanthaceae", "Aphelandra", "scabra"),
        # second view of same individual, no ID -> ignored
        AnnotationRecord("roni", "i1", "view_02", specimen_id="SR1",
                         true_family="Acanthaceae", true_genus="Aphelandra", true_species="scabra"),
        _roni("i2", "Fabaceae", "Inga", "edulis", "Fabaceae", "Inga", "marginata", specimen="SR2"),
    ]
    graded = grade_human_ids(records, "roni")
    assert len(graded) == 2
    g = {r.anonymous_id: r for r in graded}
    assert g["i1"].species_correct is True
    assert g["i2"].genus_correct is True and g["i2"].species_correct is False


def test_grade_skips_unidentified_and_unmatched():
    records = [
        # no ID at all (Carmen-style)
        AnnotationRecord("roni", "i3", "view_01", specimen_id="SR3",
                         true_family="X", true_genus="Y", true_species="z"),
        # identified but unmatched to curator (no true taxonomy)
        AnnotationRecord("roni", "i4", "view_01", id_genus="Foo", id_species="bar"),
    ]
    assert grade_human_ids(records, "roni") == []


def test_id_accuracy():
    records = [
        _roni("i1", "F", "G", "a", "F", "G", "a"),
        _roni("i2", "F", "G", "b", "F", "G", "c", specimen="SR2"),
    ]
    acc = id_accuracy(grade_human_ids(records, "roni"))
    assert acc["n_graded"] == 2
    assert acc["family_accuracy"] == 1.0
    assert acc["genus_accuracy"] == 1.0
    assert acc["species_accuracy"] == 0.5


def test_id_accuracy_empty():
    assert id_accuracy([]) == {"n_graded": 0}


# --------------------------------------------------------------------------- #
# Corrections: typo / variant / synonym crediting (raw stays untouched).
# --------------------------------------------------------------------------- #


def test_typo_correction_flips_species_corrected_only():
    # Roni wrote "hayessi"; truth "hayesii" -> raw wrong, corrected right.
    rec = _roni("i1", "Annonaceae", "Annona", "hayessi", "Annonaceae", "Annona", "hayesii")
    corr = {("SR1", "species"): Correction("SR1", "species", "hayessi", "hayesii", "typo")}
    g = grade_human_ids([rec], "roni", corr)[0]
    assert g.species_correct is False        # raw untouched
    assert g.species_corrected is True        # credited
    assert g.species_correction.category == "typo"
    assert g.pred_species == "hayessi"        # original text preserved


def test_synonym_credits_genus_but_not_wrong_species():
    # Genus synonym Monteverdia=Maytenus; species genuinely wrong -> stays wrong.
    rec = _roni("i1", "Celastraceae", "Monteverdia", "schippii",
                "Celastraceae", "Maytenus", "schippii")
    corr = {("SR1", "genus"): Correction("SR1", "genus", "Monteverdia", "Maytenus", "synonym")}
    g = grade_human_ids([rec], "roni", corr)[0]
    assert g.genus_correct is False and g.genus_corrected is True
    assert g.genus_correction.category == "synonym"
    assert g.species_correct is True  # schippii matched raw already


def test_stale_correction_credits_nothing():
    # Correction's recorded original ("foo") doesn't match what Roni wrote.
    rec = _roni("i1", "F", "G", "bar", "F", "G", "baz")
    corr = {("SR1", "species"): Correction("SR1", "species", "foo", "baz", "typo")}
    g = grade_human_ids([rec], "roni", corr)[0]
    assert g.species_corrected is False and g.species_correction is None


def test_dual_accuracy_reports_raw_and_corrected():
    records = [
        _roni("i1", "F", "G", "hayessi", "F", "G", "hayesii"),
        _roni("i2", "F", "G", "a", "F", "G", "a", specimen="SR2"),
    ]
    corr = {("SR1", "species"): Correction("SR1", "species", "hayessi", "hayesii", "typo")}
    acc = id_accuracy(grade_human_ids(records, "roni", corr))
    assert acc["species_accuracy"] == 0.5           # raw
    assert acc["corrected_species_accuracy"] == 1.0  # corrected
    assert acc["n_credited_by_category"] == {"typo": 1}


def test_no_corrections_leaves_corrected_equal_to_raw():
    rec = _roni("i1", "F", "G", "b", "F", "G", "c")
    g = grade_human_ids([rec], "roni")[0]
    assert g.species_corrected == g.species_correct is False


# --------------------------------------------------------------------------- #

REPO = Path(__file__).resolve().parents[2]
RONI = REPO / "trait_grading/annotations/roni_bianco.xlsx"
CURATOR = REPO / "trait_grading/keys/curator_taxonomic_key.csv"


@pytest.mark.skipif(not (RONI.exists() and CURATOR.exists()), reason="study data absent")
def test_real_roni_id_accuracy():
    curator = load_curator_key(CURATOR)
    records, _ = load_annotations(RONI, "roni", curator)
    graded = grade_human_ids(records, "roni")
    assert len(graded) == 114
    acc = id_accuracy(graded)
    # family accuracy should comfortably exceed species accuracy
    assert acc["family_accuracy"] >= acc["species_accuracy"]
