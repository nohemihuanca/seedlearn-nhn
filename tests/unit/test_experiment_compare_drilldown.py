"""Tests for the per-specimen drill-down + prompt reconstruction (report modal)."""

from seedlearn.benchmarking.human.aggregate import (
    SpecimenAggregate,
    SpecimenTraitAgg,
)
from seedlearn.benchmarking.human.experiment_compare import (
    build_specimen_cells,
    resolve_prompt,
    spec_for_trait,
)
from seedlearn.benchmarking.human.value_map import MISSING

TRAIT = "leaf_margin"


def _model(margin_by_specimen: dict[str, object]) -> dict[str, dict]:
    """Build a {specimen: model-traits} map with the given leaf-margin values."""
    out: dict[str, dict] = {}
    for sid, margin in margin_by_specimen.items():
        section = {} if margin is None else {"margin": margin}
        out[sid] = {"leaf_morphology": section}
    return out


def _agg(annotator: str, margins: dict[str, list[str]]) -> list[SpecimenAggregate]:
    """Build aggregates for one annotator: specimen -> per-view raw margin values."""
    from seedlearn.benchmarking.human.value_map import to_canonical
    from seedlearn.benchmarking.human import value_map

    spec = next(s for s in value_map.gradable_specs() if s.key == TRAIT)
    aggs: list[SpecimenAggregate] = []
    for sid, views in margins.items():
        canon = [to_canonical(spec, v) for v in views]
        present = [c for c in canon if c != MISSING]
        mode = max(set(present), key=present.count) if present else MISSING
        trait = SpecimenTraitAgg(
            trait_key=TRAIT, mode=mode, canonical_values=canon,
            raw_values=list(views), n_views=len(views), n_present=len(present),
        )
        aggs.append(
            SpecimenAggregate(anonymous_id=sid, specimen_id=sid, annotator=annotator,
                              traits={TRAIT: trait})
        )
    return aggs


def _cells_by_id(cells):
    return {c.specimen_id: c for c in cells}


def test_compound_descriptor_resolves_and_is_counted():
    # Post-fix: "toothed, serrate" canonicalizes to toothed, not dropped.
    model = _model({"s1": "toothed, serrate"})
    aggs = _agg("roni", {"s1": ["dentado", "serrado"]})
    cells = _cells_by_id(build_specimen_cells(model, aggs, "model_vs_roni"))
    c = cells["s1"]
    assert c.model_canonical == "toothed" and c.model_dropped is False
    assert c.roni_canonical == "toothed" and c.roni_views == ["dentado", "serrado"]
    assert c.counted is True and c.agree is True


def test_ambiguous_value_drops_to_missing_but_row_survives():
    # "entire to toothed" is ambiguous -> MISSING; the row must still appear so the
    # artifact (a value silently leaving the denominator) is visible in the modal.
    model = _model({"s1": "entire to toothed"})
    aggs = _agg("roni", {"s1": ["entero"]})
    cells = _cells_by_id(build_specimen_cells(model, aggs, "model_vs_roni"))
    c = cells["s1"]
    assert c.model_canonical == MISSING and c.model_dropped is True
    assert c.counted is False and c.agree is False
    assert c.roni_canonical == "entire"


def test_absent_model_value_is_not_flagged_dropped():
    model = _model({"s1": None})
    aggs = _agg("roni", {"s1": ["entero"]})
    c = _cells_by_id(build_specimen_cells(model, aggs, "model_vs_roni"))["s1"]
    assert c.model_raw is None and c.model_dropped is False
    assert c.model_canonical == MISSING and c.counted is False


def test_specimen_scored_only_by_model_has_row_uncounted():
    model = _model({"s1": "entire"})
    aggs = _agg("roni", {})  # no human annotated s1
    c = _cells_by_id(build_specimen_cells(model, aggs, "model_vs_roni"))["s1"]
    assert c.model_canonical == "entire"
    assert c.roni_canonical == MISSING and c.counted is False


def test_roni_vs_carmen_axis_ignores_model():
    model = _model({"s1": "entire to toothed"})  # model MISSING, irrelevant here
    aggs = _agg("roni", {"s1": ["dentado"]}) + _agg("carmen", {"s1": ["serrado"]})
    c = _cells_by_id(build_specimen_cells(model, aggs, "roni_vs_carmen"))["s1"]
    # Both humans -> toothed; axis is counted and agrees regardless of the model.
    assert c.roni_canonical == "toothed" and c.carmen_canonical == "toothed"
    assert c.counted is True and c.agree is True


def test_dropped_rows_sort_first():
    model = _model({"s_ok": "entire", "s_drop": "entire to toothed"})
    aggs = _agg("roni", {"s_ok": ["entero"], "s_drop": ["entero"]})
    cells = build_specimen_cells(model, aggs, "model_vs_roni")
    assert cells[0].specimen_id == "s_drop"  # dropped surfaces at the top


# --- resolve_prompt -------------------------------------------------------


def test_resolve_prompt_local_reconstructs_text():
    info = resolve_prompt({"prompt_style": "sys4", "model": "Qwen", "external": False})
    assert info.text and len(info.text) > 0
    assert info.unavailable_reason is None and info.model == "Qwen"


def test_resolve_prompt_external_is_unavailable():
    info = resolve_prompt(
        {"prompt_style": "sys4", "model": "gpt-5.4", "granularity": "all_traits",
         "external": True}
    )
    assert info.text is None and info.external is True
    assert info.unavailable_reason and info.model == "gpt-5.4"


def test_resolve_prompt_unknown_style_degrades():
    info = resolve_prompt({"prompt_style": "nope", "model": "m"})
    assert info.text is None and info.unavailable_reason is not None


def test_resolve_prompt_missing_style_degrades():
    info = resolve_prompt({"model": "m"})
    assert info.text is None and "not recorded" in info.unavailable_reason


def test_resolve_prompt_recorded_text_used_for_external():
    # A recovered as-run prompt overrides the external "unavailable" marker.
    info = resolve_prompt(
        {"external": True, "model": "gpt-5.4", "prompt_text": "RECOVERED GPT PROMPT"}
    )
    assert info.text == "RECOVERED GPT PROMPT" and info.unavailable_reason is None
    assert info.external is True


def test_resolve_prompt_reads_prompt_file(tmp_path):
    f = tmp_path / "gpt_prompt.txt"
    f.write_text("PROMPT FROM FILE")
    info = resolve_prompt({"external": True, "model": "gpt-5.1", "prompt_file": str(f)})
    assert info.text == "PROMPT FROM FILE" and info.unavailable_reason is None


def test_resolve_prompt_missing_prompt_file_falls_back():
    info = resolve_prompt(
        {"external": True, "model": "gpt", "prompt_file": "/no/such/file.txt"}
    )
    assert info.text is None and info.unavailable_reason  # honest unavailable marker


# --- generalized drill-down (non-margin trait) ---------------------------------


def _agg_trait(annotator, trait_key, values):
    """Aggregates for one annotator: specimen -> per-view raw values, any trait."""
    from seedlearn.benchmarking.human.value_map import to_canonical

    spec = spec_for_trait(trait_key)
    aggs = []
    for sid, views in values.items():
        canon = [to_canonical(spec, v) for v in views]
        present = [c for c in canon if c != MISSING]
        mode = max(set(present), key=present.count) if present else MISSING
        aggs.append(
            SpecimenAggregate(
                anonymous_id=sid, specimen_id=sid, annotator=annotator,
                traits={trait_key: SpecimenTraitAgg(
                    trait_key=trait_key, mode=mode, canonical_values=canon,
                    raw_values=list(views), n_views=len(views), n_present=len(present),
                )},
            )
        )
    return aggs


def test_drilldown_for_leaf_apex():
    # leaf_apex lives at leaf_morphology.apex; the drill-down must read it, not margin.
    model = {"s1": {"leaf_morphology": {"apex": "acute"}}}
    aggs = _agg_trait("roni", "leaf_apex", {"s1": ["agudo"]})  # agudo -> acute
    cells = _cells_by_id(
        build_specimen_cells(model, aggs, "model_vs_roni", spec_for_trait("leaf_apex"))
    )
    c = cells["s1"]
    assert c.model_canonical == "acute"
    assert c.roni_canonical == "acute" and c.roni_views == ["agudo"]
    assert c.counted is True and c.agree is True


def test_drilldown_default_spec_is_leaf_margin():
    # Omitting spec reproduces leaf-margin behavior (backward compatibility).
    model = _model({"s1": "toothed, serrate"})
    aggs = _agg("roni", {"s1": ["dentado"]})
    default = _cells_by_id(build_specimen_cells(model, aggs, "model_vs_roni"))
    explicit = _cells_by_id(
        build_specimen_cells(model, aggs, "model_vs_roni", spec_for_trait("leaf_margin"))
    )
    assert default["s1"] == explicit["s1"]
    assert default["s1"].model_canonical == "toothed"
