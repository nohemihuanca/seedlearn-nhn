"""Tests for the all-trait comparison report (model-named matrix + unified drill-down)."""

import csv
import importlib.util
import json
from pathlib import Path

from seedlearn.benchmarking.human.experiment_compare import (
    AllTraitConditionMetrics,
    AxisMetric,
    Ceiling,
    PromptInfo,
    SpecimenCell,
    distinct_species,
)
from seedlearn.benchmarking.human.value_map import MISSING

_SPEC = importlib.util.spec_from_file_location(
    "compare_all_traits",
    Path(__file__).resolve().parents[2] / "scripts" / "compare_all_traits.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)

SPECIES = {"s1": "Cordia bicolor", "s2": "Inga vera"}


def _cell(sid, canon, roni, carmen=None, counted=True):
    return SpecimenCell(
        specimen_id=sid, model_raw=canon, model_canonical=canon, model_dropped=False,
        roni_views=[], roni_canonical=roni,
        carmen_views=[], carmen_canonical=roni if carmen is None else carmen,
        counted=counted, agree=(canon == roni),
    )


def _cond(label, model, kappa, venation_cells, gran=""):
    axes = {"venation": {"model_vs_roni": AxisMetric(0.70, kappa, 2),
                         "roni_vs_carmen": AxisMetric(0.85, 0.55, 2)}}
    return AllTraitConditionMetrics(
        label=label, model=model, external=model.startswith("gpt"),
        n_model_specimens=2, granularity=gran,
        axes=axes, cells={"venation": {"model_vs_roni": venation_cells}},
        prompt=PromptInfo("sys4", model, None, model.startswith("gpt"),
                          None if model.startswith("gpt") else "LOCAL-PROMPT",
                          "cloud prompt unavailable" if model.startswith("gpt") else None),
    )


# Roni spans two classes (pinnate/parallel) so venation is decidable, not single-class.
def _qwen():  # disagrees on s1 (palmate vs pinnate)
    return _cond("C0", "Qwen/Qwen3-VL-32B-Instruct-FP8", 0.20,
                 [_cell("s1", "palmate", "pinnate"), _cell("s2", "pinnate", "parallel")])


def _gpt():  # agrees on s1
    return _cond("K1", "gpt-5.4", 0.25,
                 [_cell("s1", "pinnate", "pinnate"), _cell("s2", "pinnate", "parallel")])


def _ceiling():
    # Ceiling 0.55 > best model κ (0.25) -> venation lands in the MODEL_GAP group.
    return {"venation": Ceiling(0.85, 0.55, 2)}


def _write(tmp_path, metrics):
    out = tmp_path / "all_trait_comparison.html"
    mod.write_html(out, metrics, _ceiling(), SPECIES)
    return out.read_text()


def _multi_condition(label, model, kappa_by_trait, cells_by_trait):
    """A condition spanning several traits, for triage-group tests."""
    axes = {t: {"model_vs_roni": AxisMetric(0.7, k, len(cells_by_trait[t]))}
            for t, k in kappa_by_trait.items()}
    cells = {t: {"model_vs_roni": cs} for t, cs in cells_by_trait.items()}
    return AllTraitConditionMetrics(
        label=label, model=model, external=False, n_model_specimens=2,
        axes=axes, cells=cells,
        prompt=PromptInfo("sys4", model, None, False, "LOCAL-PROMPT", None),
    )


def test_headers_use_model_names(tmp_path):
    html = _write(tmp_path, [_qwen(), _gpt()])
    assert "Qwen3-VL-32B" in html and "GPT-5.4" in html


def test_disambiguates_shared_model(tmp_path):
    k2 = _cond("K2", "gpt-5.1", 0.1, [_cell("s1", "pinnate", "pinnate")], gran="per_trait")
    k3 = _cond("K3", "gpt-5.1", 0.1, [_cell("s1", "pinnate", "pinnate")], gran="per_section")
    html = _write(tmp_path, [k2, k3])
    assert "GPT-5.1 (per-trait)" in html and "GPT-5.1 (per-section)" in html


def test_species_count_in_cell(tmp_path):
    html = _write(tmp_path, [_qwen()])
    assert "2 sp" in html  # both s1, s2 counted -> 2 distinct species


def test_cells_open_trait_not_condition(tmp_path):
    html = _write(tmp_path, [_qwen(), _gpt()])
    assert "data-trait='venation'" in html
    assert "data-key" not in html  # old per-cell keying is gone


def test_unified_payload_joins_models(tmp_path):
    html = _write(tmp_path, [_qwen(), _gpt()])
    traits = json.loads(_script(html, "traits"))
    s1 = next(r for r in traits["venation"] if r["specimen_id"] == "s1")
    assert s1["species"] == "Cordia bicolor" and s1["roni"] == "pinnate"
    assert s1["models"]["Qwen3-VL-32B"]["agrees_roni"] is False  # palmate != pinnate
    assert s1["models"]["GPT-5.4"]["agrees_roni"] is True
    models = json.loads(_script(html, "models"))
    assert models == ["Qwen3-VL-32B", "GPT-5.4"]  # column order


def test_decision_table_shows_best_model(tmp_path):
    html = _write(tmp_path, [_qwen(), _gpt()])
    # The decision table names the best model per trait; GPT (0.25) beats Qwen (0.20).
    dt = html[html.index("Trait decisions"):]
    assert "best model" in dt and "GPT-5.4" in dt


def test_decision_table_groups_by_verdict(tmp_path):
    # Three traits, one per verdict group:
    #  - leaf_margin: ceiling 0.80, best 0.30      -> MODEL_GAP
    #  - leaf_shape:  ceiling 0.15, best 0.45      -> AT_HUMAN_LEVEL
    #  - stem_type:   Roni all woody (single class) -> UNDECIDABLE
    cells = {
        "leaf_margin": [_cell("s1", "entire", "entire"), _cell("s2", "entire", "toothed")],
        "leaf_shape": [_cell("s1", "elliptic", "elliptic"), _cell("s2", "elliptic", "obovate")],
        "stem_type": [_cell("s1", "woody", "woody"), _cell("s2", "woody", "woody")],
    }
    cond = _multi_condition("C0", "Qwen/Qwen3.6-35B-A3B-FP8",
                            {"leaf_margin": 0.30, "leaf_shape": 0.45, "stem_type": 0.0}, cells)
    ceiling = {"leaf_margin": Ceiling(0.9, 0.80, 2),
               "leaf_shape": Ceiling(0.5, 0.15, 2),
               "stem_type": Ceiling(0.0, 0.0, 2)}
    out = tmp_path / "r.html"
    mod.write_html(out, [cond], ceiling, {"s1": "A a", "s2": "B b"})
    html = out.read_text()
    assert "Model gaps" in html and "At human level" in html and "Undecidable" in html
    # stem_type (single-class) appears under Undecidable, before leaf_margin under Model gaps.
    assert html.index("Model gaps") < html.index("Undecidable")


def test_decision_table_shows_roni_distribution(tmp_path):
    cells = {"leaf_margin": [_cell("s1", "entire", "entire"), _cell("s2", "toothed", "toothed"),
                            _cell("s3", "entire", "entire")]}
    cond = _multi_condition("C0", "Qwen/Q", {"leaf_margin": 0.3}, cells)
    out = tmp_path / "r.html"
    mod.write_html(out, [cond], {"leaf_margin": Ceiling(0.9, 0.8, 3)}, {})
    html = out.read_text()
    assert "entire 2" in html and "toothed 1" in html  # Roni distribution counts render


def test_recommendation_panel_excludes_undecidable_from_mean(tmp_path):
    # Two decidable traits (κ 0.4, 0.6 -> mean 0.5) + one single-class trait (κ 0.9 ignored).
    cells = {
        "leaf_margin": [_cell("s1", "entire", "entire"), _cell("s2", "entire", "toothed")],
        "leaf_shape": [_cell("s1", "elliptic", "elliptic"), _cell("s2", "elliptic", "obovate")],
        "stem_type": [_cell("s1", "woody", "woody"), _cell("s2", "woody", "woody")],
    }
    cond = _multi_condition("C0", "Qwen/Q",
                            {"leaf_margin": 0.4, "leaf_shape": 0.6, "stem_type": 0.9}, cells)
    ceiling = {"leaf_margin": Ceiling(0.9, 0.8, 2), "leaf_shape": Ceiling(0.9, 0.8, 2),
               "stem_type": Ceiling(0.0, 0.0, 2)}
    out = tmp_path / "r.html"
    mod.write_html(out, [cond], ceiling, {})
    html = out.read_text()
    rec = html[html.index("Recommendation"):html.index("Trait decisions")]
    assert "0.500" in rec  # mean of the two decidable traits, not 0.633 (which includes 0.9)
    assert "2 decidable" in rec


def test_summary_csv_has_species_and_ceiling_columns(tmp_path):
    out = tmp_path / "s.csv"
    m = _qwen()
    by_cell = {("venation", "C0"): distinct_species(m.cells["venation"]["model_vs_roni"], SPECIES)}
    analysis = mod.analyze_traits([m], _ceiling())
    mod.write_summary_csv(out, [m], by_cell, analysis)
    row = next(r for r in csv.DictReader(out.open()) if r["trait"] == "venation")
    assert row["n_species"] == "2" and row["cohen_kappa"] == "0.2000"
    assert row["ceiling_kappa"] == "0.5500"          # ceiling now in the CSV, not HTML-only
    assert row["trait_verdict"] == "model_gap"
    assert "pinnate:1" in row["roni_distribution"]


def test_no_candidate_baseline_wording(tmp_path):
    html = _write(tmp_path, [_qwen(), _gpt()])
    assert "candidate" not in html.lower() and "baseline" not in html.lower()


def test_pairwise_mcnemar_section_removed(tmp_path):
    html = _write(tmp_path, [_qwen(), _gpt()])
    assert "McNemar" not in html and "beats" not in html


def test_drilldown_payload_carries_distributions(tmp_path):
    html = _write(tmp_path, [_qwen(), _gpt()])
    dists = json.loads(_script(html, "dists"))
    ven = dists["venation"]
    assert ven["roni"]["pinnate"] == 1 and ven["roni"]["parallel"] == 1
    # each model's predicted counts present, keyed by display name
    assert ven["models"]["Qwen3-VL-32B"]["palmate"] == 1
    assert ven["models"]["GPT-5.4"]["pinnate"] == 2
    assert "DISTS" in html  # the JS consumes the payload


def test_external_prompt_unavailable(tmp_path):
    html = _write(tmp_path, [_qwen(), _gpt()])
    prompts = json.loads(_script(html, "prompts"))
    assert prompts["C0"]["text"] == "LOCAL-PROMPT"
    assert prompts["K1"]["text"] is None and prompts["K1"]["unavailable_reason"]


def test_trait_outside_prompt_scope_renders_as_not_asked(tmp_path):
    # A margin-only condition never asked about venation: the cell must read as
    # out-of-scope, not as a real κ of 0 sitting in a coloured cell.
    margin_only = _cond("C2u", "Qwen/Qwen3.6-35B-A3B-FP8", 0.5, [])
    margin_only.axes["venation"]["model_vs_roni"] = AxisMetric(None, None, 0)
    html = _write(tmp_path, [_qwen(), margin_only])
    assert "not asked in this condition" in html
    assert "class='drill na'" in html


def test_kappa_legend_present(tmp_path):
    html = _write(tmp_path, [_qwen(), _gpt()])
    assert "Landis" in html
    for band in ("slight", "fair", "moderate", "substantial", "almost perfect"):
        assert band in html


def test_carmen_and_models_colored_vs_roni(tmp_path):
    html = _write(tmp_path, [_qwen(), _gpt()])
    # Carmen column and model columns both run through the vs-Roni agreement classifier,
    # which greys out rows where Roni has no value (not judged).
    assert "agreeCls(r.carmen, r.roni)" in html
    assert "agreeCls(cell.canonical, r.roni)" in html
    assert "roni===MISSING || v===MISSING" in html  # not-visible -> grey, not red


def test_not_visible_note_present(tmp_path):
    html = _write(tmp_path, [_qwen(), _gpt()])
    assert "not visible" in html and "excluded from κ" in html


def _script(html: str, sid: str) -> str:
    marker = f'id="{sid}">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    return html[start:end].replace("<\\/", "</")
