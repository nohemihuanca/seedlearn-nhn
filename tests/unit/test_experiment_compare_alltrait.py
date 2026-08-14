"""Tests for the all-trait comparison path (trait x condition matrix + McNemar)."""

from types import SimpleNamespace

from seedlearn.benchmarking.human.categorical_grader import PairDetail, TraitAgreement
from seedlearn.benchmarking.human.experiment_compare import (
    PromptInfo,
    AllTraitConditionMetrics,
    SpecimenCell,
    VERDICT_AT_HUMAN_LEVEL,
    VERDICT_MODEL_GAP,
    VERDICT_UNDECIDABLE,
    all_trait_metrics_from_bundle,
    build_trait_comparison,
    class_distribution,
    distinct_species,
    majority_fraction,
    model_display_names,
    model_short_name,
    paired_mcnemar_all_traits,
    species_map_from_curator,
    triage_trait,
)
from seedlearn.benchmarking.human.value_map import MISSING, TraitSpec


def _agreement(trait, axis, rate, kappa, n):
    return TraitAgreement(
        trait_key=trait, axis=axis, n_compared=n, n_agree=int(rate * n),
        agreement_rate=rate, cohen_kappa=kappa,
    )


def _detail(trait, sid, agree):
    return PairDetail(
        trait_key=trait, axis="model_vs_roni", specimen_id=sid,
        value_a="x", value_b="x" if agree else "y", agree=agree, a_views=[], b_views=[],
    )


def _fake_bundle(run_metadata=None):
    agreements = [
        _agreement("leaf_margin", "model_vs_roni", 0.83, 0.47, 112),
        _agreement("leaf_margin", "model_vs_carmen", 0.90, 0.63, 114),
        _agreement("leaf_margin", "roni_vs_carmen", 0.93, 0.80, 112),
        _agreement("leaf_apex", "model_vs_roni", 0.70, 0.20, 114),
        _agreement("leaf_apex", "roni_vs_carmen", 0.85, 0.55, 114),
    ]
    pair_details = {
        "leaf_margin": {"model_vs_roni": [_detail("leaf_margin", "s1", True),
                                          _detail("leaf_margin", "s2", False)]},
        "leaf_apex": {"model_vs_roni": [_detail("leaf_apex", "s1", False)]},
    }
    return SimpleNamespace(
        agreements=agreements, pair_details=pair_details,
        run_metadata=run_metadata or {}, n_model_specimens=114,
    )


def test_matrix_has_row_per_trait_and_axis():
    m, ceiling = all_trait_metrics_from_bundle(_fake_bundle(), label="C0")
    assert set(m.axes) == {"leaf_margin", "leaf_apex"}
    assert m.axes["leaf_margin"]["model_vs_roni"].kappa == 0.47
    assert m.axes["leaf_apex"]["model_vs_roni"].kappa == 0.20


def test_per_trait_ceiling_extracted():
    _m, ceiling = all_trait_metrics_from_bundle(_fake_bundle(), label="C0")
    assert ceiling["leaf_margin"].kappa == 0.80
    assert ceiling["leaf_apex"].kappa == 0.55
    # A trait with no roni_vs_carmen agreement gets no ceiling entry.
    assert "nonexistent" not in ceiling


def test_metadata_fills_model_and_external():
    m, _ = all_trait_metrics_from_bundle(
        _fake_bundle({"model": "Qwen", "external": True}), label="K1"
    )
    assert m.model == "Qwen" and m.external is True


def test_roni_correct_populated_per_trait():
    m, _ = all_trait_metrics_from_bundle(_fake_bundle(), label="C0")
    assert m.roni_correct["leaf_margin"] == {"s1": True, "s2": False}
    assert m.roni_correct["leaf_apex"] == {"s1": False}


def _cond(label, correct_by_trait):
    return AllTraitConditionMetrics(
        label=label, model="m", external=False, n_model_specimens=2,
        roni_correct=correct_by_trait,
    )


def test_paired_mcnemar_all_traits_per_trait():
    base = _cond("C0", {"leaf_margin": {"s1": True, "s2": False, "s3": False},
                        "leaf_apex": {"s1": True}})
    cand = _cond("C1", {"leaf_margin": {"s1": True, "s2": True, "s3": True},
                        "leaf_apex": {"s1": True}})
    res = paired_mcnemar_all_traits(base, cand)
    # leaf_margin: cand fixes s2 and s3, none regress.
    assert res["leaf_margin"].b_only_correct == 2
    assert res["leaf_margin"].a_only_correct == 0
    # leaf_apex: no discordant pairs -> p is None.
    assert res["leaf_apex"].p_value is None


def test_paired_mcnemar_all_traits_only_shared_traits():
    base = _cond("C0", {"leaf_margin": {"s1": True}})
    cand = _cond("C1", {"leaf_apex": {"s1": True}})  # no shared trait
    assert paired_mcnemar_all_traits(base, cand) == {}


# --- U1: species helpers -------------------------------------------------------


def _sc(sid, canonical, roni, counted, agree=False, raw=None, dropped=False):
    return SpecimenCell(
        specimen_id=sid, model_raw=raw or canonical, model_canonical=canonical,
        model_dropped=dropped, roni_views=[], roni_canonical=roni,
        carmen_views=[], carmen_canonical=MISSING, counted=counted, agree=agree,
    )


def test_species_map_from_curator(tmp_path):
    csv = tmp_path / "key.csv"
    csv.write_text(
        "anonymous_id,family,genus,species,individual_code,image_count\n"
        "individual_001,Rubiaceae,Amaioua,glomerulata,ANAMAICO4,5\n"
        "individual_002,Fabaceae,Inga,vera,BAR0010,5\n"
    )
    m = species_map_from_curator(str(csv))
    assert m["ANAMAICO4"] == "Amaioua glomerulata"
    assert m["BAR0010"] == "Inga vera"


def test_distinct_species_counts_counted_only():
    smap = {"s1": "Inga vera", "s2": "Inga vera", "s3": "Cordia bicolor"}
    cells = [
        _sc("s1", "entire", "entire", True),
        _sc("s2", "entire", "entire", True),   # same species as s1 -> collapses
        _sc("s3", "toothed", "toothed", True),
        _sc("s4", "entire", "entire", True),   # not in species map -> excluded
        _sc("s5", MISSING, "entire", False),   # not counted -> excluded
    ]
    assert distinct_species(cells, smap) == 2  # Inga vera + Cordia bicolor


# --- U1: class distribution + trait triage -------------------------------------


def _cell(sid, model, roni, carmen=MISSING, counted=True):
    return SpecimenCell(
        specimen_id=sid, model_raw=model, model_canonical=model, model_dropped=False,
        roni_views=[], roni_canonical=roni, carmen_views=[], carmen_canonical=carmen,
        counted=counted, agree=(model == roni),
    )


_VEN_SPEC = TraitSpec(
    key="venation", model_section="leaf_morphology", model_field="venation",
    spanish_header_prefix="venacion",
    canonical_values=("pinnate", "palmate", "parallel", "arcuate"),
)


def test_class_distribution_zero_fills_unused_classes():
    # Roni uses pinnate + parallel but never palmate/arcuate; the model uses palmate.
    cells = [
        _cell("s1", "pinnate", "pinnate"),
        _cell("s2", "palmate", "parallel"),
        _cell("s3", "pinnate", "pinnate"),
    ]
    roni = class_distribution(cells, _VEN_SPEC, side="roni")
    assert roni == {"pinnate": 2, "palmate": 0, "parallel": 1, "arcuate": 0}
    model = class_distribution(cells, _VEN_SPEC, side="model")
    assert model["palmate"] == 1 and model["parallel"] == 0


def test_class_distribution_uses_side_value_not_counted():
    # Roni labelled s2 but the model dropped out (counted=False). Roni's distribution
    # must still count s2 -- otherwise her marginals depend on which model she's paired with.
    cells = [
        _cell("s1", "pinnate", "pinnate", counted=True),
        _cell("s2", MISSING, "parallel", counted=False),
    ]
    roni = class_distribution(cells, _VEN_SPEC, side="roni")
    assert roni["pinnate"] == 1 and roni["parallel"] == 1  # s2 counted despite counted=False
    model = class_distribution(cells, _VEN_SPEC, side="model")
    assert sum(model.values()) == 1  # the model's MISSING value is not tallied


def test_majority_fraction():
    assert majority_fraction({"a": 90, "b": 10}) == 0.9
    assert majority_fraction({"a": 20, "b": 0}) == 1.0   # single class
    assert majority_fraction({"a": 0, "b": 0}) is None   # empty


def test_triage_single_class_is_undecidable():
    # Roni used one class -> κ ≡ 0 regardless of ceiling/best.
    assert triage_trait({"woody": 109, "herbaceous": 0}, 0.5, 0.5) == VERDICT_UNDECIDABLE


def test_triage_missing_ceiling_or_best_is_undecidable():
    assert triage_trait({"a": 60, "b": 40}, None, 0.3) == VERDICT_UNDECIDABLE
    assert triage_trait({"a": 60, "b": 40}, 0.4, None) == VERDICT_UNDECIDABLE


def test_triage_model_at_or_above_ceiling():
    # leaf_shape-like: humans agree weakly (0.162), model beats it (0.464).
    assert triage_trait({"a": 55, "b": 38, "c": 20}, 0.162, 0.464) == VERDICT_AT_HUMAN_LEVEL
    assert triage_trait({"a": 55, "b": 38}, 0.40, 0.40) == VERDICT_AT_HUMAN_LEVEL  # tie -> at-level


def test_triage_model_gap():
    # leaflet_arrangement-like: humans agree strongly (0.829), model far short (0.127).
    assert triage_trait({"a": 66, "b": 44}, 0.829, 0.127) == VERDICT_MODEL_GAP


# --- U2: model display names ---------------------------------------------------


def test_model_short_name():
    assert model_short_name("Qwen/Qwen3-VL-32B-Instruct-FP8") == "Qwen3-VL-32B"
    assert model_short_name("Qwen/Qwen3.6-35B-A3B-FP8") == "Qwen3.6-35B"
    assert model_short_name("gpt-5.4") == "GPT-5.4"
    assert model_short_name("some/unknown-model") == "unknown-model"


def _mcond(label, model, granularity="", style=None, examples=None):
    return AllTraitConditionMetrics(
        label=label, model=model, external=False, n_model_specimens=1,
        granularity=granularity,
        prompt=PromptInfo(style, model, examples, False, "TXT", None) if style else None,
    )


def test_model_display_names_disambiguates_shared_model():
    conds = [
        _mcond("C0", "Qwen/Qwen3-VL-32B-Instruct-FP8"),
        _mcond("K2", "gpt-5.1", "per_trait"),
        _mcond("K3", "gpt-5.1", "per_section"),
    ]
    names = model_display_names(conds)
    assert names["C0"] == "Qwen3-VL-32B"          # unique -> no qualifier
    assert names["K2"] == "GPT-5.1 (per-trait)"   # shared -> qualified
    assert names["K3"] == "GPT-5.1 (per-section)"


def test_display_names_disambiguate_margin_variants_by_prompt_style():
    # C1/C2u/C3u/C4u all run the same upgraded Qwen; what varies is the prompt
    # (and, for C4u, the few-shot exemplar images) -- not the granularity.
    up = "Qwen/Qwen3.6-35B-A3B-FP8"
    conds = [
        _mcond("C0_baseline", "Qwen/Qwen3-VL-32B-Instruct-FP8", style="sys4"),
        _mcond("C1_upgraded_model", up, style="sys4"),
        _mcond("C2u_margin_only", up, style="margin_only"),
        _mcond("C3u_margin_rich", up, style="margin_rich"),
        _mcond("C4u_image_fewshot", up, style="margin_rich", examples="ex.json"),
    ]
    names = model_display_names(conds)
    assert names["C0_baseline"] == "Qwen3-VL-32B"  # unique model -> bare
    assert names["C1_upgraded_model"] == "Qwen3.6-35B (sys4)"
    assert names["C2u_margin_only"] == "Qwen3.6-35B (margin-only)"
    # C3u and C4u share a prompt style; the few-shot marker keeps them distinct.
    assert names["C3u_margin_rich"] == "Qwen3.6-35B (margin-rich)"
    assert names["C4u_image_fewshot"] == "Qwen3.6-35B (margin-rich +few-shot)"
    assert len(set(names.values())) == len(conds)  # no column collisions


def test_display_name_falls_back_to_label_without_prompt_or_granularity():
    conds = [_mcond("A_run", "gpt-5.1"), _mcond("B_run", "gpt-5.1")]
    names = model_display_names(conds)
    assert names["A_run"] == "GPT-5.1 (A-run)" and names["B_run"] == "GPT-5.1 (B-run)"


# --- U3: cross-model per-trait join --------------------------------------------


def _cond_cells(label, model, cells_by_trait):
    return AllTraitConditionMetrics(
        label=label, model=model, external=False, n_model_specimens=2,
        cells={t: {"model_vs_roni": cs} for t, cs in cells_by_trait.items()},
    )


def test_build_trait_comparison_joins_models_by_specimen():
    c0 = _cond_cells("C0", "Qwen/Qwen3-VL-32B-Instruct-FP8", {
        "venation": [_sc("s1", "palmate", "pinnate", True, agree=False),
                     _sc("s2", "pinnate", "pinnate", True, agree=True)]})
    k1 = _cond_cells("K1", "gpt-5.4", {
        "venation": [_sc("s1", "pinnate", "pinnate", True, agree=True),
                     _sc("s2", "pinnate", "pinnate", True, agree=True)]})
    names = model_display_names([c0, k1])
    smap = {"s1": "Cordia bicolor", "s2": "Inga vera"}
    out = build_trait_comparison([c0, k1], names, smap)
    rows = {r["specimen_id"]: r for r in out["venation"]}
    s1 = rows["s1"]
    assert s1["species"] == "Cordia bicolor" and s1["roni"] == "pinnate"
    # C0 disagrees (palmate vs pinnate), GPT agrees.
    assert s1["models"]["Qwen3-VL-32B"]["agrees_roni"] is False
    assert s1["models"]["GPT-5.4"]["agrees_roni"] is True
    # The disagreement row sorts before the all-agree row.
    assert out["venation"][0]["specimen_id"] == "s1"


def test_build_trait_comparison_missing_species_falls_back_to_id():
    c0 = _cond_cells("C0", "gpt-5.4", {"venation": [_sc("sX", "pinnate", "pinnate", True)]})
    out = build_trait_comparison([c0], model_display_names([c0]), {})
    assert out["venation"][0]["species"] == "sX"  # no crash, labeled by id
