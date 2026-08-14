"""Tests for the categorical grader (human.categorical_grader)."""

from seedlearn.benchmarking.human.aggregate import SpecimenAggregate, SpecimenTraitAgg
from seedlearn.benchmarking.human.categorical_grader import (
    compare_lookups,
    grade_all_axes,
    human_lookup,
    human_view_lookup,
    model_lookup,
    overall_by_axis,
    pair_details,
)
from seedlearn.benchmarking.human.value_map import MISSING


def _specs():
    from seedlearn.benchmarking.human.value_map import TRAIT_SPECS

    return tuple(s for s in TRAIT_SPECS if s.key == "leaf_relative_position")


def _trait(mode, n_present=1):
    return SpecimenTraitAgg("leaf_relative_position", mode, [mode], [mode], 1, n_present)


def test_perfect_two_class_agreement():
    a = {"s1": {"leaf_relative_position": "whorled"}, "s2": {"leaf_relative_position": "opposite"}}
    b = {"s1": {"leaf_relative_position": "whorled"}, "s2": {"leaf_relative_position": "opposite"}}
    [res] = compare_lookups(a, b, "x", specs=_specs())
    assert res.n_compared == 2
    assert res.agreement_rate == 1.0
    assert res.cohen_kappa == 1.0


def test_single_class_agreement_kappa_undefined():
    # Both always "whorled": rate perfect, kappa undefined (single class).
    a = {s: {"leaf_relative_position": "whorled"} for s in ("s1", "s2", "s3")}
    [res] = compare_lookups(a, dict(a), "x", specs=_specs())
    assert res.agreement_rate == 1.0
    assert res.cohen_kappa is None


def test_skew_high_rate_low_kappa():
    # 9/10 agree, but heavy skew toward "whorled" -> kappa well below the rate.
    a, b = {}, {}
    for i in range(8):
        a[f"s{i}"] = {"leaf_relative_position": "whorled"}
        b[f"s{i}"] = {"leaf_relative_position": "whorled"}
    a["s8"] = {"leaf_relative_position": "opposite"}
    b["s8"] = {"leaf_relative_position": "opposite"}
    a["s9"] = {"leaf_relative_position": "whorled"}
    b["s9"] = {"leaf_relative_position": "opposite"}  # disagreement
    [res] = compare_lookups(a, b, "x", specs=_specs())
    assert res.agreement_rate == 0.9
    assert res.cohen_kappa is not None
    assert res.cohen_kappa < res.agreement_rate


def test_missing_pairs_excluded():
    a = {"s1": {"leaf_relative_position": "whorled"}, "s2": {"leaf_relative_position": MISSING}}
    b = {"s1": {"leaf_relative_position": "whorled"}, "s2": {"leaf_relative_position": "opposite"}}
    [res] = compare_lookups(a, b, "x", specs=_specs())
    assert res.n_compared == 1


def test_no_comparable_specimens():
    a = {"s1": {"leaf_relative_position": MISSING}}
    b = {"s1": {"leaf_relative_position": "whorled"}}
    [res] = compare_lookups(a, b, "x", specs=_specs())
    assert res.n_compared == 0
    assert res.agreement_rate is None
    assert res.cohen_kappa is None


def test_grade_all_axes_uses_three_sources():
    model = {"SR1": {"leaf_arrangement": {"relative_position": "whorled"}}}
    aggs = [
        SpecimenAggregate("i1", "SR1", "roni", {"leaf_relative_position": _trait("whorled")}),
        SpecimenAggregate("i1", "SR1", "carmen", {"leaf_relative_position": _trait("opposite")}),
    ]
    results = grade_all_axes(model, aggs, specs=_specs())
    axes = {r.axis for r in results}
    assert axes == {"model_vs_roni", "model_vs_carmen", "roni_vs_carmen"}
    by_axis = {r.axis: r for r in results}
    assert by_axis["model_vs_roni"].n_agree == 1  # whorled == whorled
    assert by_axis["model_vs_carmen"].n_agree == 0  # whorled != opposite
    assert by_axis["roni_vs_carmen"].n_agree == 0  # whorled != opposite


def test_model_lookup_canonicalizes():
    model = {"SR1": {"leaf_arrangement": {"relative_position": "verticilada"}}}
    # model values are English normally, but canonicalization is symmetric
    lk = model_lookup(model, specs=_specs())
    assert lk["SR1"]["leaf_relative_position"] == "whorled"


def test_overall_by_axis_macro_average():
    results = grade_all_axes(
        {"SR1": {"leaf_arrangement": {"relative_position": "whorled"}}},
        [SpecimenAggregate("i1", "SR1", "roni", {"leaf_relative_position": _trait("whorled")})],
        specs=_specs(),
    )
    summary = overall_by_axis(results)
    assert summary["model_vs_roni"]["macro_agreement_rate"] == 1.0


def test_human_lookup_filters_by_annotator():
    aggs = [
        SpecimenAggregate("i1", "SR1", "roni", {"leaf_relative_position": _trait("whorled")}),
        SpecimenAggregate("i2", "SR2", "carmen", {"leaf_relative_position": _trait("opposite")}),
    ]
    assert set(human_lookup(aggs, "roni")) == {"SR1"}
    assert set(human_lookup(aggs, "carmen")) == {"SR2"}


# --------------------------------------------------------------------------- #
# pair_details (drill-down)
# --------------------------------------------------------------------------- #

KEY = "leaf_relative_position"


def test_pair_details_agree_flag_and_count():
    a = {"s1": {KEY: "whorled"}, "s2": {KEY: "opposite"}, "s3": {KEY: "whorled"}}
    b = {"s1": {KEY: "whorled"}, "s2": {KEY: "opposite"}, "s3": {KEY: "opposite"}}
    detail = pair_details(a, b, "x", specs=_specs())[KEY]
    assert len(detail) == 3
    agree_by_sid = {d.specimen_id: d.agree for d in detail}
    assert agree_by_sid == {"s1": True, "s2": True, "s3": False}
    s3 = next(d for d in detail if d.specimen_id == "s3")
    assert (s3.value_a, s3.value_b) == ("whorled", "opposite")


def test_pair_details_excludes_missing():
    a = {"s1": {KEY: "whorled"}, "s2": {KEY: MISSING}}
    b = {"s1": {KEY: "whorled"}, "s2": {KEY: "opposite"}}
    detail = pair_details(a, b, "x", specs=_specs())[KEY]
    assert [d.specimen_id for d in detail] == ["s1"]  # s2 dropped for missingness


def test_pair_details_disagreements_first():
    a = {"s1": {KEY: "whorled"}, "s2": {KEY: "whorled"}}
    b = {"s1": {KEY: "opposite"}, "s2": {KEY: "whorled"}}  # s1 disagrees
    detail = pair_details(a, b, "x", specs=_specs())[KEY]
    assert detail[0].specimen_id == "s1" and detail[0].agree is False
    assert detail[1].specimen_id == "s2" and detail[1].agree is True


def test_pair_details_attaches_human_views_not_model():
    a = {"s1": {KEY: "whorled"}}  # model side: no per-view values
    b = {"s1": {KEY: "whorled"}}  # human side: per-view raw values supplied
    b_views = {"s1": {KEY: ["verticilada", "verticilada"]}}
    detail = pair_details(a, b, "x", specs=_specs(), b_views=b_views)[KEY]
    assert detail[0].a_views == []
    assert detail[0].b_views == ["verticilada", "verticilada"]


def test_human_view_lookup_returns_raw_per_view():
    agg = SpecimenAggregate(
        "i1", "SR1", "roni",
        {KEY: SpecimenTraitAgg(KEY, "whorled", ["whorled", "whorled"],
                               ["verticilada", "verticilada"], 2, 2)},
    )
    views = human_view_lookup([agg], "roni")
    assert views["SR1"][KEY] == ["verticilada", "verticilada"]
