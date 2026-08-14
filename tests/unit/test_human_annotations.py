"""Tests for the human annotation loader (benchmarking.human.annotations)."""

from pathlib import Path

import pytest

from seedlearn.benchmarking.human.annotations import (
    CuratorEntry,
    join_specimens,
    load_annotations,
    load_curator_key,
    parse_annotation_rows,
    parse_curator_rows,
)

REPO = Path(__file__).resolve().parents[2]
RONI_XLSX = REPO / "trait_grading/annotations/roni_bianco.xlsx"
CARMEN_XLSX = REPO / "trait_grading/annotations/carmen.xlsx"
CURATOR_KEY = REPO / "trait_grading/keys/curator_taxonomic_key.csv"

HEADER = [
    "identificador_unico",
    "vista",
    "familia_prediccion",
    "genero_prediccion",
    "especie_predicion",
    "Posicion relativa de las hojas (alterna / opuesta / verticilada)",
    "Margen de la hoja (entero / dentado)",
]


def test_parse_happy_path():
    rows = [
        ["individual_001", "view_01", "Acanthaceae", "Aphelandra", "scabra", "opuesta", "entero"],
        ["individual_001", "view_02", "", "", "", "verticilada", "entero"],
    ]
    recs = parse_annotation_rows(HEADER, rows, "roni")
    assert len(recs) == 2
    assert recs[0].annotator == "roni"
    assert recs[0].anonymous_id == "individual_001"
    assert recs[0].view_id == "view_01"
    assert recs[0].traits["leaf_relative_position"] == "opuesta"
    assert recs[0].traits["leaf_margin"] == "entero"
    assert recs[0].id_genus == "Aphelandra"
    assert recs[0].id_species == "scabra"
    # second row has no ID columns filled
    assert recs[1].id_family is None and recs[1].id_genus is None


def test_blank_and_trailing_rows_dropped():
    rows = [
        ["individual_002", "view_01", "", "", "", "alterna", ""],
        ["", "", "", "", "", "", ""],          # fully blank
        ["individual_003", "", "", "", "", "opuesta", ""],  # missing view
        ["", "view_05", "", "", "", "opuesta", ""],         # missing id
    ]
    recs = parse_annotation_rows(HEADER, rows, "carmen")
    assert [r.anonymous_id for r in recs] == ["individual_002"]


def test_empty_traits_not_recorded():
    rows = [["individual_004", "view_01", "", "", "", "", ""]]
    recs = parse_annotation_rows(HEADER, rows, "carmen")
    assert recs[0].traits == {}


def test_missing_required_column_raises():
    bad_header = ["vista", "Margen de la hoja (entero / dentado)"]
    with pytest.raises(ValueError, match="identificador_unico"):
        parse_annotation_rows(bad_header, [["view_01", "entero"]], "roni")


def test_join_specimens_attaches_truth():
    curator = {
        "individual_001": CuratorEntry(
            "individual_001", "SRAPHEDE2", "Acanthaceae", "Aphelandra", "scabra"
        )
    }
    recs = parse_annotation_rows(
        HEADER, [["individual_001", "view_01", "", "", "", "opuesta", ""]], "roni"
    )
    joined, unmatched = join_specimens(recs, curator)
    assert unmatched == []
    assert joined[0].specimen_id == "SRAPHEDE2"
    assert joined[0].true_genus == "Aphelandra"
    assert joined[0].true_species == "scabra"


def test_join_reports_unmatched_once():
    recs = parse_annotation_rows(
        HEADER,
        [
            ["individual_999", "view_01", "", "", "", "opuesta", ""],
            ["individual_999", "view_02", "", "", "", "opuesta", ""],
        ],
        "roni",
    )
    joined, unmatched = join_specimens(recs, {})
    assert unmatched == ["individual_999"]
    assert joined[0].specimen_id is None


def test_parse_curator_rows():
    rows = [
        {
            "anonymous_id": "individual_001",
            "family": "Acanthaceae",
            "genus": "Aphelandra",
            "species": "scabra",
            "individual_code": "SRAPHEDE2",
        }
    ]
    curator = parse_curator_rows(rows)
    assert curator["individual_001"].specimen_id == "SRAPHEDE2"


# --------------------------------------------------------------------------- #
# Smoke tests against the real study data (skip if data absent).
# --------------------------------------------------------------------------- #

_have_data = RONI_XLSX.exists() and CURATOR_KEY.exists()
skip_no_data = pytest.mark.skipif(not _have_data, reason="study data not present")


@skip_no_data
def test_real_roni_loads_114_individuals():
    curator = load_curator_key(CURATOR_KEY)
    assert len(curator) == 114
    records, unmatched = load_annotations(RONI_XLSX, "roni", curator)
    assert unmatched == [], f"unmatched anon ids: {unmatched[:5]}"
    individuals = {r.anonymous_id for r in records}
    assert len(individuals) == 114
    # every joined record resolved to a specimen
    assert all(r.specimen_id for r in records)
    # Roni recorded species IDs
    assert any(r.id_genus for r in records)


@skip_no_data
def test_real_carmen_loads_and_joins():
    if not CARMEN_XLSX.exists():
        pytest.skip("carmen sheet absent")
    curator = load_curator_key(CURATOR_KEY)
    records, unmatched = load_annotations(CARMEN_XLSX, "carmen", curator)
    assert unmatched == []
    assert len({r.anonymous_id for r in records}) == 114
