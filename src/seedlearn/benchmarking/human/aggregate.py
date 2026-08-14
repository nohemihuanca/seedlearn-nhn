"""Aggregate per-view human annotations to per-specimen modal values.

Humans annotated each trait once per view; the model emits one pooled value per
specimen. To compare them, each annotator's per-view values for a trait are
collapsed to the **mode** (the value scored), while the full per-view
distribution is retained for transparency (so a noisy trait is visible rather
than hidden behind its mode).

Values are canonicalized via :mod:`seedlearn.benchmarking.human.value_map` before
the mode is taken; ``MISSING`` values (blank / not-observed) are excluded from the
mode but counted in ``n_views``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from seedlearn.benchmarking.human.annotations import AnnotationRecord
from seedlearn.benchmarking.human.value_map import (
    MISSING,
    TRAIT_SPECS,
    TraitSpec,
    gradable_specs,
    to_canonical,
)


@dataclass(frozen=True)
class SpecimenTraitAgg:
    """Per-specimen aggregate of one annotator's per-view values for a trait."""

    trait_key: str
    mode: str  # canonical token used for scoring, or MISSING
    canonical_values: list[str]  # per-view canonical tokens, in view order
    raw_values: list[str]  # per-view raw values, in view order (for display)
    n_views: int  # total views the annotator recorded for this specimen
    n_present: int  # views with a non-MISSING value


@dataclass(frozen=True)
class SpecimenAggregate:
    """All of one annotator's per-specimen modal traits for a single individual."""

    anonymous_id: str
    specimen_id: str | None
    annotator: str
    traits: dict[str, SpecimenTraitAgg] = field(default_factory=dict)


def modal_value(canonical_values: list[str]) -> str:
    """Return the modal non-:data:`MISSING` value, ties broken by first occurrence.

    The deterministic tie-break (earliest value to appear in view order wins)
    keeps aggregation reproducible regardless of dict ordering.
    """
    counts: dict[str, int] = {}
    order: list[str] = []
    for value in canonical_values:
        if value == MISSING:
            continue
        if value not in counts:
            order.append(value)
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return MISSING
    best = max(counts.values())
    for value in order:  # order preserves first-occurrence for tie-breaking
        if counts[value] == best:
            return value
    return MISSING  # unreachable


def aggregate_records(
    records: list[AnnotationRecord],
    specs: tuple[TraitSpec, ...] | None = None,
) -> list[SpecimenAggregate]:
    """Collapse per-view records to per-(specimen, annotator) modal traits.

    Records are grouped by ``(anonymous_id, annotator)``. For each gradable trait,
    per-view raw values are canonicalized, the mode is taken, and the distribution
    is retained. ``specs`` defaults to the gradable trait set.
    """
    specs = specs if specs is not None else gradable_specs()
    spec_by_key = {s.key: s for s in TRAIT_SPECS}

    # Preserve first-seen group order for deterministic output.
    grouped: dict[tuple[str, str], list[AnnotationRecord]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for rec in records:
        key = (rec.anonymous_id, rec.annotator)
        if key not in grouped:
            order.append(key)
        grouped[key].append(rec)

    aggregates: list[SpecimenAggregate] = []
    for anon, annotator in order:
        group = grouped[(anon, annotator)]
        specimen_id = next((r.specimen_id for r in group if r.specimen_id), None)
        trait_aggs: dict[str, SpecimenTraitAgg] = {}
        for spec in specs:
            spec = spec_by_key.get(spec.key, spec)
            raw_values = [r.traits.get(spec.key, "") for r in group]
            canonical = [to_canonical(spec, raw) for raw in raw_values]
            n_present = sum(1 for c in canonical if c != MISSING)
            trait_aggs[spec.key] = SpecimenTraitAgg(
                trait_key=spec.key,
                mode=modal_value(canonical),
                canonical_values=canonical,
                raw_values=[str(r) for r in raw_values],
                n_views=len(group),
                n_present=n_present,
            )
        aggregates.append(
            SpecimenAggregate(
                anonymous_id=anon,
                specimen_id=specimen_id,
                annotator=annotator,
                traits=trait_aggs,
            )
        )
    return aggregates
