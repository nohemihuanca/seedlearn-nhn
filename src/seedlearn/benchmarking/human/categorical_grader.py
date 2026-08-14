"""Per-trait agreement between trait value sources.

Grades three comparison axes over the gradable categorical traits:

* ``model_vs_roni``   -- Vision-LLM prediction vs Roni's modal annotation
* ``model_vs_carmen`` -- Vision-LLM prediction vs Carmen's modal annotation
* ``roni_vs_carmen``  -- the two human annotators (inter-annotator ceiling)

For each trait and axis, only specimens where *both* sources have a non-``MISSING``
value are compared. Agreement is reported as a raw rate and as Cohen's kappa
(chance-corrected), so a skewed trait with a high raw rate but low kappa is
visible, and model-vs-human can be read against the human-vs-human ceiling.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from seedlearn.benchmarking.human.aggregate import SpecimenAggregate
from seedlearn.benchmarking.human.value_map import (
    MISSING,
    TraitSpec,
    gradable_specs,
    model_value,
    to_canonical,
)

# A value source maps specimen_id -> {trait_key: canonical_token}.
ValueLookup = Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class TraitAgreement:
    """Agreement between two sources for one trait on one axis."""

    trait_key: str
    axis: str
    n_compared: int
    n_agree: int
    agreement_rate: float | None
    cohen_kappa: float | None


@dataclass(frozen=True)
class PairDetail:
    """One compared specimen behind a (trait, axis) agreement cell.

    Holds the two sources' modal values plus, for whichever side is a human
    annotator, the per-view raw values behind that mode (empty for the model,
    which emits one pooled value). Used only to render the report drill-down.
    """

    trait_key: str
    axis: str
    specimen_id: str
    value_a: str
    value_b: str
    agree: bool
    a_views: list[str]
    b_views: list[str]


# A per-view source maps specimen_id -> {trait_key: [raw per-view values]}.
ViewLookup = Mapping[str, Mapping[str, list[str]]]


def pair_details(
    lookup_a: ValueLookup,
    lookup_b: ValueLookup,
    axis: str,
    specs: tuple[TraitSpec, ...] | None = None,
    a_views: ViewLookup | None = None,
    b_views: ViewLookup | None = None,
) -> dict[str, list[PairDetail]]:
    """Per-specimen comparison detail behind each (trait, axis) agreement cell.

    Mirrors :func:`compare_lookups`' specimen iteration -- only specimens where
    *both* sources have a non-``MISSING`` value appear, so the detail matches the
    rate/kappa denominator. ``a_views`` / ``b_views`` supply per-view raw values
    for a human side (omit or pass ``None`` for the model). Returns a mapping of
    ``trait_key -> [PairDetail]`` with disagreements first, then by specimen id.
    """
    specs = specs if specs is not None else gradable_specs()
    specimen_ids = sorted(set(lookup_a) & set(lookup_b))
    out: dict[str, list[PairDetail]] = {}
    for spec in specs:
        details: list[PairDetail] = []
        for sid in specimen_ids:
            va = lookup_a[sid].get(spec.key, MISSING)
            vb = lookup_b[sid].get(spec.key, MISSING)
            if va == MISSING or vb == MISSING:
                continue
            details.append(
                PairDetail(
                    trait_key=spec.key,
                    axis=axis,
                    specimen_id=sid,
                    value_a=va,
                    value_b=vb,
                    agree=(va == vb),
                    a_views=list((a_views or {}).get(sid, {}).get(spec.key, [])),
                    b_views=list((b_views or {}).get(sid, {}).get(spec.key, [])),
                )
            )
        # Disagreements first (so the interesting rows are at the top), then by id.
        details.sort(key=lambda d: (d.agree, d.specimen_id))
        out[spec.key] = details
    return out


def safe_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa, returning ``None`` when it is undefined.

    Kappa is undefined with fewer than two comparable pairs or when both lists
    collapse to a single shared label (perfect-but-degenerate agreement). Shared
    with the STRI axis, which computes kappa over its single-label subset.
    """
    if len(a) < 2:
        return None
    labels = sorted(set(a) | set(b))
    if len(labels) < 2:
        return None
    kappa = cohen_kappa_score(a, b, labels=labels)
    if kappa is None or math.isnan(kappa):
        return None
    return float(kappa)


def compare_lookups(
    lookup_a: ValueLookup,
    lookup_b: ValueLookup,
    axis: str,
    specs: tuple[TraitSpec, ...] | None = None,
) -> list[TraitAgreement]:
    """Compute per-trait agreement between two value lookups over shared specimens."""
    specs = specs if specs is not None else gradable_specs()
    specimen_ids = sorted(set(lookup_a) & set(lookup_b))
    results: list[TraitAgreement] = []
    for spec in specs:
        a_labels: list[str] = []
        b_labels: list[str] = []
        for sid in specimen_ids:
            va = lookup_a[sid].get(spec.key, MISSING)
            vb = lookup_b[sid].get(spec.key, MISSING)
            if va == MISSING or vb == MISSING:
                continue
            a_labels.append(va)
            b_labels.append(vb)
        n = len(a_labels)
        n_agree = sum(1 for x, y in zip(a_labels, b_labels) if x == y)
        results.append(
            TraitAgreement(
                trait_key=spec.key,
                axis=axis,
                n_compared=n,
                n_agree=n_agree,
                agreement_rate=(n_agree / n) if n else None,
                cohen_kappa=safe_kappa(a_labels, b_labels) if n else None,
            )
        )
    return results


def human_lookup(
    aggregates: list[SpecimenAggregate], annotator: str
) -> dict[str, dict[str, str]]:
    """Build a specimen -> {trait: modal token} lookup for one annotator."""
    out: dict[str, dict[str, str]] = {}
    for agg in aggregates:
        if agg.annotator != annotator or not agg.specimen_id:
            continue
        out[agg.specimen_id] = {k: v.mode for k, v in agg.traits.items()}
    return out


def human_view_lookup(
    aggregates: list[SpecimenAggregate], annotator: str
) -> dict[str, dict[str, list[str]]]:
    """Build a specimen -> {trait: [per-view raw values]} lookup for one annotator."""
    out: dict[str, dict[str, list[str]]] = {}
    for agg in aggregates:
        if agg.annotator != annotator or not agg.specimen_id:
            continue
        out[agg.specimen_id] = {k: list(v.raw_values) for k, v in agg.traits.items()}
    return out


def model_lookup(
    model_traits_by_specimen: Mapping[str, Mapping[str, object]],
    specs: tuple[TraitSpec, ...] | None = None,
) -> dict[str, dict[str, str]]:
    """Build a specimen -> {trait: canonical token} lookup from model traits dicts."""
    specs = specs if specs is not None else gradable_specs()
    out: dict[str, dict[str, str]] = {}
    for sid, traits in model_traits_by_specimen.items():
        tokens: dict[str, str] = {}
        for spec in specs:
            raw = model_value(traits, spec)
            tokens[spec.key] = MISSING if raw is None else to_canonical(spec, raw)
        out[sid] = tokens
    return out


def load_model_traits(results_dir: str | Path) -> dict[str, dict]:
    """Load ``stages.morphology.data.traits`` for every specimen JSON in a dir."""
    out: dict[str, dict] = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        if path.name == "run_metadata.json":
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        traits = (
            data.get("stages", {})
            .get("morphology", {})
            .get("data", {})
            .get("traits")
        )
        specimen_id = data.get("specimen_id") or path.stem
        if isinstance(traits, dict):
            out[specimen_id] = traits
    return out


def grade_all_axes(
    model_traits_by_specimen: Mapping[str, Mapping[str, object]],
    aggregates: list[SpecimenAggregate],
    specs: tuple[TraitSpec, ...] | None = None,
) -> list[TraitAgreement]:
    """Grade all three axes (model-vs-roni, model-vs-carmen, roni-vs-carmen)."""
    specs = specs if specs is not None else gradable_specs()
    model = model_lookup(model_traits_by_specimen, specs)
    roni = human_lookup(aggregates, "roni")
    carmen = human_lookup(aggregates, "carmen")
    return [
        *compare_lookups(model, roni, "model_vs_roni", specs),
        *compare_lookups(model, carmen, "model_vs_carmen", specs),
        *compare_lookups(roni, carmen, "roni_vs_carmen", specs),
    ]


def overall_by_axis(results: list[TraitAgreement]) -> dict[str, dict[str, float]]:
    """Macro-average rate and kappa per axis over traits with comparisons."""
    summary: dict[str, dict[str, float]] = {}
    axes = sorted({r.axis for r in results})
    for axis in axes:
        rows = [r for r in results if r.axis == axis and r.n_compared > 0]
        rates = [r.agreement_rate for r in rows if r.agreement_rate is not None]
        kappas = [r.cohen_kappa for r in rows if r.cohen_kappa is not None]
        summary[axis] = {
            "n_traits": len(rows),
            "total_compared": sum(r.n_compared for r in rows),
            "macro_agreement_rate": (sum(rates) / len(rates)) if rates else None,
            "macro_cohen_kappa": (sum(kappas) / len(kappas)) if kappas else None,
        }
    return summary


# Convenience: callable-based source if a caller prefers a function over a dict.
def lookup_from_callable(
    specimen_ids: list[str], fn: Callable[[str, TraitSpec], str], specs: tuple[TraitSpec, ...]
) -> dict[str, dict[str, str]]:
    """Materialize a {specimen: {trait: token}} lookup from a callable source."""
    return {sid: {s.key: fn(sid, s) for s in specs} for sid in specimen_ids}
