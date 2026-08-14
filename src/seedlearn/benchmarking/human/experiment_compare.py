"""Cross-run leaf-margin comparison for the trait-extraction experiment ladder.

Each experiment condition is a ``model_run`` directory (local conditions produced
by the pipeline, or external cloud results adapted by
``scripts/ingest_workshop_results.py``). This module grades each one with the
shared human-annotation grader (:func:`report.assemble`), extracts the
**leaf-margin** metrics, and compares conditions against the Roni-vs-Carmen human
ceiling.

Design (plan KTD7/KTD8):

* Grade each run once via ``assemble(..., embed_images=False)`` — numbers only, no
  thumbnails re-embedded per condition.
* Extract the three κ axes (``model_vs_roni``, ``model_vs_carmen``,
  ``roni_vs_carmen``) plus STRI match-any accuracy. STRI is a separate accuracy
  column, not a κ axis (its reference is multi-label).
* The ``roni_vs_carmen`` ceiling is constant across conditions (it depends only on
  the humans) — computed once, shown as the reference line.
* Because every condition grades the same specimens, condition-vs-condition
  significance uses a **paired McNemar** test on per-specimen model-vs-Roni
  leaf-margin correctness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

from .categorical_grader import (
    human_lookup,
    human_view_lookup,
    load_model_traits,
)
from .report import assemble
from .value_map import (
    MISSING,
    TraitSpec,
    gradable_specs,
    model_value,
    to_canonical,
    unmapped_values,
)

TRAIT = "leaf_margin"

# The leaf-margin spec, used by the per-specimen drill-down extractor.
_LEAF_SPEC: TraitSpec = next(s for s in gradable_specs() if s.key == TRAIT)

# For each comparison axis, which two canonical values the cell counts/agrees on.
_AXIS_PAIR = {
    "model_vs_roni": lambda c: (c.model_canonical, c.roni_canonical),
    "model_vs_carmen": lambda c: (c.model_canonical, c.carmen_canonical),
    "roni_vs_carmen": lambda c: (c.roni_canonical, c.carmen_canonical),
}


@dataclass(frozen=True)
class SpecimenCell:
    """Per-specimen leaf-margin evidence behind one comparison cell.

    Unlike :func:`categorical_grader.pair_details`, the drill-down keeps rows where
    a side canonicalized to ``MISSING`` (``counted == False``) so grading artifacts
    -- e.g. a model value that dropped out of the κ denominator -- stay visible.
    Model/Roni/Carmen fields are always populated; ``counted`` and ``agree`` reflect
    the axis this cell belongs to.
    """

    specimen_id: str
    model_raw: str | None  # raw model margin string; None when the field is absent
    model_canonical: str  # canonical token or MISSING
    model_dropped: bool  # a non-empty raw value that canonicalized to MISSING
    roni_views: list[str]  # raw per-view annotator values behind Roni's mode
    roni_canonical: str  # Roni's modal canonical token or MISSING
    carmen_views: list[str]
    carmen_canonical: str
    counted: bool  # both sides of this axis are non-MISSING (in the denominator)
    agree: bool  # counted and the two canonical tokens are equal


@dataclass(frozen=True)
class PromptInfo:
    """Inference-prompt provenance + reconstructed text for one condition."""

    style: str | None
    model: str
    examples_file: str | None
    external: bool
    text: str | None  # reconstructed prompt, or None when unavailable
    unavailable_reason: str | None  # None for a successfully reconstructed prompt


def spec_for_trait(trait_key: str) -> TraitSpec:
    """Return the gradable :class:`TraitSpec` for ``trait_key`` (KeyError if unknown)."""
    for s in gradable_specs():
        if s.key == trait_key:
            return s
    raise KeyError(f"no gradable trait spec for {trait_key!r}")


def species_map_from_curator(curator_key: str | Path) -> dict[str, str]:
    """Build ``{specimen_id: "Genus species"}`` from the curator taxonomic key.

    Species is the source shared by every condition — the ingested external (cloud)
    runs carry no species field of their own. Mirrors ``report.py``'s
    ``specimen_to_species`` construction.
    """
    from .annotations import load_curator_key

    curator = load_curator_key(curator_key)
    return {
        e.specimen_id: f"{e.genus} {e.species}".strip()
        for e in curator.values()
        if e.specimen_id
    }


def distinct_species(cells: list[SpecimenCell], species_map: dict[str, str]) -> int:
    """Count distinct species among the *counted* specimens of a cell.

    Specimens missing from ``species_map`` are excluded (no phantom species).
    """
    species = {
        species_map[c.specimen_id]
        for c in cells
        if c.counted and c.specimen_id in species_map
    }
    return len(species)


# Trait triage verdicts. A trait's κ is only a fair model score when the reference
# uses more than one class AND the two annotators themselves agree; otherwise a low
# κ says nothing about the model. These labels sort traits into the three regimes.
VERDICT_UNDECIDABLE = "undecidable"  # reference used one class → κ ≡ 0 for everyone
VERDICT_AT_HUMAN_LEVEL = "at_human_level"  # best model κ ≥ human-human ceiling
VERDICT_MODEL_GAP = "model_gap"  # humans agree, models fall short — worth model work


def class_distribution(
    cells: list[SpecimenCell], spec: TraitSpec, *, side: str = "roni"
) -> dict[str, int]:
    """Canonical-value counts for one side of a trait's comparison cells.

    Counts each specimen once, over the specimens where *this side* has a non-missing
    value — deliberately **not** filtered on ``cell.counted``. ``counted`` is
    axis-relative ("both sides present"), so filtering on it would drop specimens the
    annotator labelled but the model missed, making the reference distribution depend
    on which model it is paired with. Filtering on the side's own value keeps Roni's
    marginals condition-invariant (matching how the ceiling is treated).

    Every token in ``spec.canonical_values`` is present in the result, zero-filled when
    unused, so a reference that never uses a class (e.g. Roni never says ``palmate``)
    is visible rather than silently absent.

    Args:
        cells: Per-specimen drill-down rows for one trait/axis.
        spec: The trait spec, supplying the full canonical vocabulary.
        side: Which side to tally — ``"roni"``, ``"carmen"``, or ``"model"``.

    Returns:
        ``{canonical_token: count}`` including zero-count classes, ordered as
        ``spec.canonical_values`` then any extra observed tokens.
    """
    attr = {
        "roni": "roni_canonical",
        "carmen": "carmen_canonical",
        "model": "model_canonical",
    }[side]
    dist: dict[str, int] = {v: 0 for v in spec.canonical_values}
    for c in cells:
        value = getattr(c, attr)
        if value == MISSING:
            continue
        dist[value] = dist.get(value, 0) + 1
    return dist


def majority_fraction(dist: dict[str, int]) -> float | None:
    """Fraction of labelled specimens in the most common class, or ``None`` if empty."""
    total = sum(dist.values())
    if total == 0:
        return None
    return max(dist.values()) / total


def triage_trait(
    roni_dist: dict[str, int],
    ceiling_kappa: float | None,
    best_kappa: float | None,
) -> str:
    """Classify a trait into one of the three κ-interpretability regimes.

    Args:
        roni_dist: Roni's canonical-value counts (see :func:`class_distribution`).
        ceiling_kappa: The Roni-vs-Carmen human ceiling κ for this trait.
        best_kappa: The best model-vs-Roni κ across conditions for this trait.

    Returns:
        One of :data:`VERDICT_UNDECIDABLE`, :data:`VERDICT_AT_HUMAN_LEVEL`,
        :data:`VERDICT_MODEL_GAP`.
    """
    maj = majority_fraction(roni_dist)
    if maj is None or maj >= 1.0 or ceiling_kappa is None or best_kappa is None:
        return VERDICT_UNDECIDABLE
    if best_kappa >= ceiling_kappa:
        return VERDICT_AT_HUMAN_LEVEL
    return VERDICT_MODEL_GAP


def build_specimen_cells(
    model_traits: dict[str, dict],
    aggregates: list,
    axis: str,
    spec: TraitSpec = _LEAF_SPEC,
) -> list[SpecimenCell]:
    """Per-specimen drill-down rows for one comparison axis of one trait.

    Walks the union of the two axis-relevant sources and records each specimen's
    model raw/canonical (flagging non-empty values that dropped to ``MISSING``) plus
    both annotators' raw per-view values and modal canonical tokens. ``MISSING`` rows
    are retained (``counted == False``). Dropped rows sort first, then disagreements,
    then agreements, then by specimen id. ``spec`` selects the trait (defaults to leaf
    margin, so existing callers are unchanged).
    """
    trait = spec.key
    roni = human_lookup(aggregates, "roni")
    carmen = human_lookup(aggregates, "carmen")
    roni_v = human_view_lookup(aggregates, "roni")
    carmen_v = human_view_lookup(aggregates, "carmen")

    sources = {
        "model_vs_roni": (set(model_traits) | set(roni)),
        "model_vs_carmen": (set(model_traits) | set(carmen)),
        "roni_vs_carmen": (set(roni) | set(carmen)),
    }
    pair = _AXIS_PAIR[axis]

    cells: list[SpecimenCell] = []
    for sid in sorted(sources[axis]):
        raw = model_value(model_traits[sid], spec) if sid in model_traits else None
        model_canonical = MISSING if raw is None else to_canonical(spec, raw)
        model_dropped = raw is not None and bool(unmapped_values(spec, [raw]))
        cell = SpecimenCell(
            specimen_id=sid,
            model_raw=raw,
            model_canonical=model_canonical,
            model_dropped=model_dropped,
            roni_views=list(roni_v.get(sid, {}).get(trait, [])),
            roni_canonical=roni.get(sid, {}).get(trait, MISSING),
            carmen_views=list(carmen_v.get(sid, {}).get(trait, [])),
            carmen_canonical=carmen.get(sid, {}).get(trait, MISSING),
            counted=False,
            agree=False,
        )
        va, vb = pair(cell)
        counted = va != MISSING and vb != MISSING
        cell = replace(cell, counted=counted, agree=counted and va == vb)
        cells.append(cell)

    # Dropped rows first (the artifact signature), then disagreements, then by id.
    cells.sort(key=lambda c: (not c.model_dropped, c.agree, c.specimen_id))
    return cells


def resolve_prompt(run_metadata: dict | None) -> PromptInfo:
    """Reconstruct a condition's inference prompt from its run metadata.

    Local runs render the full prompt text via
    :func:`seedlearn.components.analyzers.prompts.get_prompt`; external (cloud) runs
    and runs with an unknown/absent ``prompt_style`` return metadata only, with an
    ``unavailable_reason`` explaining why the text is absent.
    """
    from seedlearn.components.analyzers.prompts import get_prompt

    meta = run_metadata or {}
    style = meta.get("prompt_style")
    model = meta.get("model", "")
    examples_file = meta.get("examples_file")
    external = bool(meta.get("external", False))

    # A recorded as-run prompt (recovered for external cloud runs, or captured at
    # ingest) takes precedence — it is the ground truth of what the model saw.
    recorded = meta.get("prompt_text")
    prompt_file = meta.get("prompt_file")
    if not recorded and prompt_file:
        try:
            recorded = Path(prompt_file).read_text()
        except OSError:
            recorded = None
    if recorded:
        return PromptInfo(style, model, examples_file, external, recorded, None)

    if external:
        return PromptInfo(
            style, model, examples_file, True, None,
            "as-run cloud prompt not reconstructable",
        )
    if not style:
        return PromptInfo(
            style, model, examples_file, False, None,
            "prompt style not recorded in this run",
        )
    try:
        text = get_prompt(style)
    except (KeyError, ValueError):
        return PromptInfo(
            style, model, examples_file, False, None,
            f"unknown prompt style {style!r}",
        )
    return PromptInfo(style, model, examples_file, False, text, None)


@dataclass
class AxisMetric:
    """Rate + κ + n for one (condition, axis) leaf-margin cell."""

    rate: float | None
    kappa: float | None
    n: int


@dataclass
class ConditionMetrics:
    """Leaf-margin metrics for one experiment condition."""

    label: str
    model: str
    granularity: str
    external: bool
    n_model_specimens: int
    vs_roni: AxisMetric
    vs_carmen: AxisMetric
    stri_accuracy: float | None
    stri_n: int
    # specimen_id -> (model agrees with Roni's modal margin) for the paired test.
    roni_correct: dict[str, bool] = field(default_factory=dict)
    # axis -> per-specimen drill-down rows (retains MISSING, for the report modal).
    cells: dict[str, list[SpecimenCell]] = field(default_factory=dict)
    # Reconstructed inference-prompt provenance for this condition.
    prompt: PromptInfo | None = None


@dataclass
class Ceiling:
    """The Roni-vs-Carmen human ceiling for leaf margin (constant across runs)."""

    rate: float | None
    kappa: float | None
    n: int


@dataclass
class PairwiseTest:
    """Paired McNemar comparison of two conditions' model-vs-Roni correctness."""

    label_a: str
    label_b: str
    n_paired: int
    rate_delta: float | None  # b.rate - a.rate on the shared specimens
    b_only_correct: int  # correct in B, wrong in A
    a_only_correct: int  # correct in A, wrong in B
    p_value: float | None


def _agreement(bundle, axis: str) -> AxisMetric:
    """Pull the leaf-margin agreement for one axis from a graded bundle."""
    for a in bundle.agreements:
        if a.trait_key == TRAIT and a.axis == axis:
            return AxisMetric(rate=a.agreement_rate, kappa=a.cohen_kappa, n=a.n_compared)
    return AxisMetric(rate=None, kappa=None, n=0)


def grade_condition(
    run_dir: str | Path,
    *,
    roni_xlsx: str | Path,
    carmen_xlsx: str | Path | None,
    curator_key: str | Path,
    stri_matrix: str | Path | None,
    label: str,
    model: str = "",
    granularity: str = "",
    external: bool = False,
) -> tuple[ConditionMetrics, Ceiling]:
    """Grade one condition's run dir and extract its leaf-margin metrics.

    Returns the condition metrics and the (run-independent) human ceiling; the
    caller keeps one ceiling and discards the rest.
    """
    bundle = assemble(
        run_dir,
        roni_xlsx,
        carmen_xlsx,
        curator_key,
        stri_matrix=stri_matrix,
        embed_images=False,
    )
    # Prefer provenance recorded in the run's metadata.
    meta = bundle.run_metadata or {}
    model = model or meta.get("model", "")
    granularity = granularity or meta.get("granularity", "")
    external = external or bool(meta.get("external", False))

    stri = next(
        (s for s in bundle.stri_results if s.trait_key == TRAIT and s.source == "model"),
        None,
    )
    details = bundle.pair_details.get(TRAIT, {}).get("model_vs_roni", [])
    roni_correct = {d.specimen_id: d.agree for d in details}

    model_traits = load_model_traits(run_dir)
    cells = {
        axis: build_specimen_cells(model_traits, bundle.aggregates, axis)
        for axis in ("model_vs_roni", "model_vs_carmen", "roni_vs_carmen")
    }
    prompt = resolve_prompt(meta)

    metrics = ConditionMetrics(
        label=label,
        model=model,
        granularity=granularity,
        external=external,
        n_model_specimens=bundle.n_model_specimens,
        vs_roni=_agreement(bundle, "model_vs_roni"),
        vs_carmen=_agreement(bundle, "model_vs_carmen"),
        stri_accuracy=(stri.accuracy if stri else None),
        stri_n=(stri.n_compared if stri else 0),
        roni_correct=roni_correct,
        cells=cells,
        prompt=prompt,
    )
    ceil_axis = _agreement(bundle, "roni_vs_carmen")
    return metrics, Ceiling(rate=ceil_axis.rate, kappa=ceil_axis.kappa, n=ceil_axis.n)


def _mcnemar_p(b: int, c: int) -> float | None:
    """Two-sided McNemar p-value on discordant counts b, c.

    Uses the exact binomial test for small discordant totals and the
    continuity-corrected chi-square approximation otherwise. Returns ``None`` when
    there are no discordant pairs (no evidence of difference).
    """
    n = b + c
    if n == 0:
        return None
    if n < 25:
        # Exact two-sided binomial test, p = 0.5.
        k = min(b, c)
        cum = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5**n)
        return min(1.0, 2.0 * cum)
    stat = (abs(b - c) - 1) ** 2 / n
    # Survival function of chi-square with 1 dof = erfc(sqrt(stat/2)).
    return math.erfc(math.sqrt(stat / 2.0))


def _paired_mcnemar_dicts(
    label_a: str,
    label_b: str,
    a_correct: dict[str, bool],
    b_correct: dict[str, bool],
) -> PairwiseTest:
    """Paired McNemar over two specimen->correct maps (shared specimens only)."""
    shared = sorted(set(a_correct) & set(b_correct))
    a_only = sum(1 for s in shared if a_correct[s] and not b_correct[s])
    b_only = sum(1 for s in shared if b_correct[s] and not a_correct[s])
    a_rate = (sum(a_correct[s] for s in shared) / len(shared)) if shared else None
    b_rate = (sum(b_correct[s] for s in shared) / len(shared)) if shared else None
    delta = (b_rate - a_rate) if (a_rate is not None and b_rate is not None) else None
    return PairwiseTest(
        label_a=label_a,
        label_b=label_b,
        n_paired=len(shared),
        rate_delta=delta,
        b_only_correct=b_only,
        a_only_correct=a_only,
        p_value=_mcnemar_p(b_only, a_only),
    )


def paired_mcnemar(a: ConditionMetrics, b: ConditionMetrics) -> PairwiseTest:
    """Compare condition ``b`` against ``a`` via paired model-vs-Roni correctness."""
    return _paired_mcnemar_dicts(a.label, b.label, a.roni_correct, b.roni_correct)


# ─── All-trait comparison (trait × condition matrix) ─────────────────────────


@dataclass
class AllTraitConditionMetrics:
    """Per-trait agreement metrics for one condition, across every gradable trait."""

    label: str
    model: str
    external: bool
    n_model_specimens: int
    granularity: str = ""
    # trait_key -> {axis -> AxisMetric} for model_vs_roni / model_vs_carmen / roni_vs_carmen.
    axes: dict[str, dict[str, AxisMetric]] = field(default_factory=dict)
    # trait_key -> {specimen_id: model-agrees-with-Roni} for per-trait paired McNemar.
    roni_correct: dict[str, dict[str, bool]] = field(default_factory=dict)
    # trait_key -> {axis -> [SpecimenCell]} for the report drill-down (opt-in).
    cells: dict[str, dict[str, list[SpecimenCell]]] = field(default_factory=dict)
    prompt: PromptInfo | None = None


def grade_condition_all_traits(
    run_dir: str | Path,
    *,
    roni_xlsx: str | Path,
    carmen_xlsx: str | Path | None,
    curator_key: str | Path,
    stri_matrix: str | Path | None = None,
    label: str,
    model: str = "",
    external: bool = False,
    cell_axes: tuple[str, ...] = (),
) -> tuple[AllTraitConditionMetrics, dict[str, Ceiling]]:
    """Grade one condition across all gradable traits.

    Returns the per-trait metrics plus the per-trait Roni-vs-Carmen ceiling (which
    depends only on the humans, so the caller keeps one ceiling map and discards the
    rest). Reuses the same ``assemble`` bundle as :func:`grade_condition` — every
    trait's agreement is already computed by ``grade_all_axes``; this simply keeps
    all of them instead of filtering to leaf margin. When ``cell_axes`` is non-empty,
    per-specimen drill-down cells are built for those axes (e.g.
    ``("model_vs_roni",)``) and attached to ``metrics.cells``.
    """
    bundle = assemble(
        run_dir, roni_xlsx, carmen_xlsx, curator_key,
        stri_matrix=stri_matrix, embed_images=False,
    )
    metrics, ceiling = all_trait_metrics_from_bundle(
        bundle, label=label, model=model, external=external
    )
    if cell_axes:
        model_traits = load_model_traits(run_dir)
        metrics.cells = {
            trait_key: {
                axis: build_specimen_cells(
                    model_traits, bundle.aggregates, axis, spec_for_trait(trait_key)
                )
                for axis in cell_axes
            }
            for trait_key in metrics.axes
        }
    return metrics, ceiling


def all_trait_metrics_from_bundle(
    bundle, *, label: str, model: str = "", external: bool = False
) -> tuple[AllTraitConditionMetrics, dict[str, Ceiling]]:
    """Transform a graded bundle into per-trait metrics + per-trait ceiling.

    Pure over the bundle (no IO), so it is unit-testable with a lightweight fake.
    """
    meta = bundle.run_metadata or {}
    model = model or meta.get("model", "")
    external = external or bool(meta.get("external", False))

    axes: dict[str, dict[str, AxisMetric]] = {}
    ceiling: dict[str, Ceiling] = {}
    for a in bundle.agreements:
        metric = AxisMetric(rate=a.agreement_rate, kappa=a.cohen_kappa, n=a.n_compared)
        axes.setdefault(a.trait_key, {})[a.axis] = metric
        if a.axis == "roni_vs_carmen":
            ceiling[a.trait_key] = Ceiling(rate=a.agreement_rate, kappa=a.cohen_kappa, n=a.n_compared)

    roni_correct: dict[str, dict[str, bool]] = {}
    for trait_key, by_axis in bundle.pair_details.items():
        details = by_axis.get("model_vs_roni", [])
        roni_correct[trait_key] = {d.specimen_id: d.agree for d in details}

    metrics = AllTraitConditionMetrics(
        label=label,
        model=model,
        external=external,
        n_model_specimens=bundle.n_model_specimens,
        granularity=meta.get("granularity", ""),
        axes=axes,
        roni_correct=roni_correct,
        prompt=resolve_prompt(meta),
    )
    return metrics, ceiling


_MODEL_SHORT_NAMES = {
    "Qwen/Qwen3-VL-32B-Instruct-FP8": "Qwen3-VL-32B",
    "Qwen/Qwen3.6-35B-A3B-FP8": "Qwen3.6-35B",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.1": "GPT-5.1",
}


def model_short_name(model_id: str) -> str:
    """A readable short name for a model id (org prefix + FP8/Instruct suffixes stripped)."""
    if model_id in _MODEL_SHORT_NAMES:
        return _MODEL_SHORT_NAMES[model_id]
    name = (model_id or "").split("/")[-1]
    for suffix in ("-Instruct-FP8", "-A3B-FP8", "-Instruct", "-FP8"):
        name = name.replace(suffix, "")
    if name[:4].lower() == "gpt-":
        name = "GPT-" + name[4:]
    return name or model_id


def _condition_qualifier(cond: AllTraitConditionMetrics) -> str:
    """Describe what distinguishes ``cond`` from others on the same model.

    Prefers the axis that actually varies: granularity for the decomposed cloud
    runs (per-trait / per-section), otherwise the prompt style, with a few-shot
    marker when exemplar images were supplied (``margin_rich`` alone does not
    separate a prompt-only run from the same prompt plus images). Falls back to
    the raw label only when nothing else is recorded.
    """
    if cond.granularity:
        return cond.granularity.replace("_", "-")
    style = cond.prompt.style if cond.prompt else None
    parts = [str(style).replace("_", "-")] if style else []
    if cond.prompt and cond.prompt.examples_file:
        parts.append("+few-shot")
    return " ".join(parts) if parts else cond.label.replace("_", "-")


def model_display_names(conditions: list[AllTraitConditionMetrics]) -> dict[str, str]:
    """Map each condition label -> display name, disambiguating shared models.

    Conditions sharing a model (K2/K3 are both ``gpt-5.1``; C1/C2u/C3u/C4u are all
    the upgraded Qwen) get a qualifier describing what varies between them, so the
    columns stay distinguishable without falling back to opaque C-labels.
    """
    from collections import Counter

    base = {c.label: model_short_name(c.model) for c in conditions}
    counts = Counter(base.values())
    out: dict[str, str] = {}
    for c in conditions:
        name = base[c.label]
        if counts[name] > 1:
            name = f"{name} ({_condition_qualifier(c)})"
        out[c.label] = name
    return out


def build_trait_comparison(
    conditions: list[AllTraitConditionMetrics],
    display_names: dict[str, str],
    species_map: dict[str, str],
) -> dict[str, list[dict]]:
    """Join every condition's model-vs-Roni cells into a per-trait, per-specimen view.

    Roni's and Carmen's canonical values are identical across conditions (same
    annotators), so this is a pure reshape keyed on ``specimen_id``. Returns
    ``{trait: [record]}`` where each record carries the species, Roni/Carmen values,
    and every model's prediction (raw, canonical, dropped flag, agrees-with-Roni).
    Rows where models disagree with Roni or split among themselves sort first.
    """
    traits: set[str] = set()
    for c in conditions:
        traits.update(c.cells)

    out: dict[str, list[dict]] = {}
    for trait in traits:
        records: dict[str, dict] = {}
        for c in conditions:
            disp = display_names.get(c.label, c.label)
            for cell in c.cells.get(trait, {}).get("model_vs_roni", []):
                sid = cell.specimen_id
                rec = records.get(sid)
                if rec is None:
                    rec = {
                        "specimen_id": sid,
                        "species": species_map.get(sid, sid),
                        "roni": cell.roni_canonical,
                        "carmen": cell.carmen_canonical,
                        "models": {},
                    }
                    records[sid] = rec
                agrees = (
                    cell.model_canonical != MISSING
                    and cell.roni_canonical != MISSING
                    and cell.model_canonical == cell.roni_canonical
                )
                rec["models"][disp] = {
                    "raw": cell.model_raw,
                    "canonical": cell.model_canonical,
                    "dropped": cell.model_dropped,
                    "agrees_roni": agrees,
                }
        rows = list(records.values())
        rows.sort(key=_trait_row_sort_key)
        out[trait] = rows
    return out


def _trait_row_sort_key(rec: dict) -> tuple:
    """Sort per-trait rows: disagreements-with-Roni and model splits first."""
    models = rec["models"].values()
    all_agree = all(m["agrees_roni"] for m in models) if models else True
    split = len({m["canonical"] for m in models}) > 1  # models disagree among themselves
    return (all_agree, not split, rec["species"])


def paired_mcnemar_all_traits(
    a: AllTraitConditionMetrics, b: AllTraitConditionMetrics
) -> dict[str, PairwiseTest]:
    """Per-trait paired McNemar (candidate ``b`` vs baseline ``a``) over every trait."""
    traits = sorted(set(a.roni_correct) & set(b.roni_correct))
    return {
        trait: _paired_mcnemar_dicts(
            a.label, b.label, a.roni_correct[trait], b.roni_correct[trait]
        )
        for trait in traits
    }
