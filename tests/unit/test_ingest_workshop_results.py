"""Tests for the external (cloud) results adapter (scripts/ingest_workshop_results.py)."""

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "ingest_workshop_results",
    Path(__file__).resolve().parents[2] / "scripts" / "ingest_workshop_results.py",
)
ingest_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ingest_mod)


def _write_source(path: Path, answers: dict, processed: dict | None = None) -> Path:
    payload = {"model": "gpt-test", "answers": answers}
    if processed is not None:
        payload["processed_results"] = processed
    path.write_text(json.dumps(payload))
    return path


def test_specimen_id_from_key():
    assert ingest_mod.specimen_id_from_key("Acanthaceae_Aphelandra_scabra_SRAPHEDE2") == "SRAPHEDE2"
    assert ingest_mod.specimen_id_from_key("Fabaceae_Inga_x_ANLINDLA8") == "ANLINDLA8"


def test_ingest_produces_model_run_shape(tmp_path):
    src = _write_source(
        tmp_path / "src.json",
        {
            "Acanthaceae_Aphelandra_scabra_SRAPHEDE2": (
                "=== Morphological Assessment Form ===\n"
                "C. Leaf Morphology\n"
                "    7.\tLeaf margin (entire / toothed): entire (edges smooth)\n"
                "F. Notes: none\n"
            ),
        },
    )
    out = tmp_path / "K_test"
    meta = ingest_mod.ingest(src, out, label="K_test", model="gpt-test", granularity="per_trait")

    rec = json.loads((out / "SRAPHEDE2.json").read_text())
    assert rec["specimen_id"] == "SRAPHEDE2"
    assert rec["stages"]["morphology"]["data"]["traits"]["leaf_morphology"]["margin"] == "entire"
    assert meta["n_specimens"] == 1
    md = json.loads((out / "run_metadata.json").read_text())
    assert md["external"] is True and md["model"] == "gpt-test" and md["granularity"] == "per_trait"


def test_empty_answer_is_skipped_not_crashed(tmp_path):
    src = _write_source(
        tmp_path / "src.json",
        {
            "F_G_s_GOOD1": "7.\tLeaf margin: entire\n",
            "F_G_s_EMPTY1": "   ",
        },
    )
    out = tmp_path / "K_skip"
    meta = ingest_mod.ingest(src, out, label="k", model="m", granularity="all_traits")
    assert meta["n_specimens"] == 1
    assert (out / "GOOD1.json").exists()
    assert not (out / "EMPTY1.json").exists()


def test_falls_back_to_processed_results(tmp_path):
    src = _write_source(
        tmp_path / "src.json",
        {"F_G_s_SP1": ""},
        processed={"F_G_s_SP1": {"answer": "7.\tLeaf margin: toothed\n"}},
    )
    out = tmp_path / "K_proc"
    ingest_mod.ingest(src, out, label="k", model="m", granularity="all_traits")
    rec = json.loads((out / "SP1.json").read_text())
    assert rec["stages"]["morphology"]["data"]["traits"]["leaf_morphology"]["margin"] == "toothed"


def test_missing_answers_key_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"model": "x"}))
    try:
        ingest_mod.load_answers(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError for missing 'answers'")


# --- decomposed (per-trait / per-section) run merging --------------------------


def _excerpt(section: str, line: str, note: str) -> str:
    return (
        "=== Morphological Assessment Form ===\n"
        f"{section}\n    {line}\n\nF. Notes: {note}\n"
    )


def test_merged_answers_concatenate_per_specimen(tmp_path):
    # A per-trait run splits one specimen's form across files; merging must
    # reconstitute every numbered line for that specimen.
    key = "Acanthaceae_Aphelandra_scabra_SRAPHEDE2"
    t7 = _write_source(tmp_path / "t7.json", {key: _excerpt(
        "C. Leaf Morphology", "7.\tLeaf margin (entire / toothed): entire (smooth)", "a")})
    t9 = _write_source(tmp_path / "t9.json", {key: _excerpt(
        "C. Leaf Morphology", "9.\tLeaf apex (acute, obtuse): acute (sharp tip)", "b")})
    merged = ingest_mod.load_merged_answers([t7, t9])
    assert "7.\tLeaf margin" in merged[key] and "9.\tLeaf apex" in merged[key]


def test_ingest_merges_decomposed_sources_into_one_condition(tmp_path):
    key = "Acanthaceae_Aphelandra_scabra_SRAPHEDE2"
    t7 = _write_source(tmp_path / "t7.json", {key: _excerpt(
        "C. Leaf Morphology", "7.\tLeaf margin (entire / toothed): entire (smooth)", "a")})
    t9 = _write_source(tmp_path / "t9.json", {key: _excerpt(
        "C. Leaf Morphology", "9.\tLeaf apex (acute, obtuse): acute (sharp tip)", "b")})
    out = tmp_path / "K_merged"
    meta = ingest_mod.ingest([t7, t9], out, label="K2", model="gpt-5.1",
                             granularity="per_trait")
    traits = json.loads((out / "SRAPHEDE2.json").read_text())["stages"]["morphology"]["data"]["traits"]
    # Both excerpts land in the same specimen record, parsed by trait number.
    assert traits["leaf_morphology"]["margin"] == "entire"
    assert traits["leaf_morphology"]["apex"] == "acute"
    assert meta["n_specimens"] == 1
    assert isinstance(meta["source_json"], list) and len(meta["source_json"]) == 2


def test_ingest_single_source_keeps_scalar_provenance(tmp_path):
    key = "Acanthaceae_Aphelandra_scabra_SRAPHEDE2"
    src = _write_source(tmp_path / "s.json", {key: _excerpt(
        "C. Leaf Morphology", "7.\tLeaf margin (entire / toothed): entire (smooth)", "a")})
    meta = ingest_mod.ingest(src, tmp_path / "K1", label="K1", model="gpt-5.4",
                             granularity="all_traits")
    assert meta["source_json"] == str(src)  # unchanged for the single-file case
