"""Tests for the leaf-margin experiment comparator (pure stats + extraction)."""

from seedlearn.benchmarking.human.experiment_compare import (
    AxisMetric,
    ConditionMetrics,
    _mcnemar_p,
    paired_mcnemar,
)


def _cond(label: str, correct: dict[str, bool]) -> ConditionMetrics:
    return ConditionMetrics(
        label=label, model="m", granularity="g", external=False,
        n_model_specimens=len(correct),
        vs_roni=AxisMetric(None, None, 0), vs_carmen=AxisMetric(None, None, 0),
        stri_accuracy=None, stri_n=0, roni_correct=dict(correct),
    )


def test_mcnemar_no_discordant_pairs_is_none():
    assert _mcnemar_p(0, 0) is None


def test_mcnemar_symmetric_small_n_high_p():
    # Equal discordant counts -> no evidence of difference -> p near 1.
    assert _mcnemar_p(3, 3) == 1.0


def test_mcnemar_lopsided_small_n_low_p():
    # 8 vs 0 discordant, exact two-sided binomial = 2 * 0.5^8.
    p = _mcnemar_p(8, 0)
    assert p is not None and abs(p - 2 * 0.5**8) < 1e-9


def test_mcnemar_large_n_uses_chi_square():
    # b=30, c=10: continuity-corrected chi-square, p should be small but valid.
    p = _mcnemar_p(30, 10)
    assert p is not None and 0.0 < p < 0.01


def test_paired_mcnemar_counts_and_delta():
    base = _cond("base", {"s1": True, "s2": False, "s3": True, "s4": False})
    cand = _cond("cand", {"s1": True, "s2": True, "s3": True, "s4": True})
    res = paired_mcnemar(base, cand)
    assert res.n_paired == 4
    # cand fixes s2 and s4 (base wrong, cand right); none regress.
    assert res.b_only_correct == 2 and res.a_only_correct == 0
    # delta = 100% - 50% = +0.5
    assert abs(res.rate_delta - 0.5) < 1e-9


def test_paired_mcnemar_only_shared_specimens_counted():
    base = _cond("base", {"s1": True, "s2": False})
    cand = _cond("cand", {"s2": True, "s3": True})  # only s2 shared
    res = paired_mcnemar(base, cand)
    assert res.n_paired == 1 and res.b_only_correct == 1 and res.a_only_correct == 0
