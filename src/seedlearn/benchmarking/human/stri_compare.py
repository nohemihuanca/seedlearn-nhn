"""Compare trait value sources against the STRI botanical trait matrix.

The STRI trait matrix encodes, per species, a **set** of allowed values for a
handful of traits (a species can legitimately have both ``entire`` and ``toothed``
margins, so each trait is multi-label binary). That makes STRI a *reference set*
rather than a single value: a source (the model, Roni, or Carmen) is "correct vs
STRI" for a trait when its value is among the species' allowed STRI values
(the same match-any policy the existing STRI trait grader uses).

Because the comparison is value-vs-set (asymmetric), the headline metric is a
match-any **accuracy** over all comparable specimens. Cohen's kappa cannot be
applied to a multi-label reference directly, so it is reported over the
**single-label subset** only -- the specimens where STRI codes exactly one value
for the trait, so both sides are genuine single-label raters. STRI only codes
five of the gradable traits, so this axis covers those five; other traits are
left blank.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from seedlearn.benchmarking.human.categorical_grader import safe_kappa
from seedlearn.benchmarking.human.value_map import MISSING

# trait_key -> {canonical_token: STRI binary column}
STRI_TOKEN_COLUMN: dict[str, dict[str, str]] = {
    "leaf_relative_position": {
        "alternate": "leaf_arrangement__alternate",
        "opposite": "leaf_arrangement__opposite",
        "whorled": "leaf_arrangement__whorled_or_clustered",
    },
    "leaf_complexity_type": {
        "simple": "leaf_type__simple",
        "compound": "leaf_type__compound",
    },
    "leaf_margin": {
        "entire": "leaf_margin__entire",
        "toothed": "leaf_margin__toothed",
        "lobed": "leaf_margin__lobed",
    },
    "stipules": {
        "present": "stipules__present",
        "absent": "stipules__absent",
    },
    "latex": {
        "present": "latex__present",
        "absent": "latex__absent",
    },
}

# trait_key -> STRI category prefix (for the per-category __uncoded flag)
STRI_CATEGORY: dict[str, str] = {
    "leaf_relative_position": "leaf_arrangement",
    "leaf_complexity_type": "leaf_type",
    "leaf_margin": "leaf_margin",
    "stipules": "stipules",
    "latex": "latex",
}

#: Traits that have a STRI counterpart (the only ones this axis can score).
STRI_TRAITS: tuple[str, ...] = tuple(STRI_TOKEN_COLUMN)


@dataclass(frozen=True)
class STRIAgreement:
    """Accuracy of one source against STRI for one trait (match-any).

    ``accuracy`` is the match-any rate over all ``n_compared`` specimens.
    ``cohen_kappa`` is a chance-corrected agreement computed over the
    ``n_kappa`` single-label specimens only (those STRI codes with exactly one
    allowed value), where a symmetric single-label comparison is well defined;
    it is ``None`` when that subset is too small or degenerate.
    """

    trait_key: str
    source: str  # "model", "roni", "carmen"
    n_compared: int
    n_correct: int
    accuracy: float | None
    n_kappa: int = 0
    cohen_kappa: float | None = None


@dataclass(frozen=True)
class STRIPairDetail:
    """One compared specimen behind a (trait, source) vs-STRI accuracy cell.

    Holds the source's value, the species' allowed STRI token set, and whether the
    value is among them (match-any). Used only to render the report drill-down.
    """

    trait_key: str
    source: str
    specimen_id: str
    species: str
    value: str
    allowed: list[str]
    correct: bool


def _is_one(value: object) -> bool:
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def load_stri_matrix(path: str) -> dict[str, dict[str, float]]:
    """Load the STRI matrix into ``scientific_name(lower) -> {column: value}``.

    Only the columns needed for the five STRI-coded traits (plus their
    ``__uncoded`` flags) are retained.
    """
    import pandas as pd

    wanted_cols: set[str] = set()
    for trait, mapping in STRI_TOKEN_COLUMN.items():
        wanted_cols.update(mapping.values())
        wanted_cols.add(f"{STRI_CATEGORY[trait]}__uncoded")

    df = pd.read_csv(path)
    keep = ["scientific_name"] + [c for c in wanted_cols if c in df.columns]
    df = df[keep]
    rows: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        name = str(row["scientific_name"]).strip().lower()
        if not name or name == "nan":
            continue
        rows[name] = {c: row[c] for c in keep if c != "scientific_name"}
    return rows


def build_stri_lookup(
    specimen_to_species: Mapping[str, str],
    stri_rows: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, dict[str, set[str] | None]], int]:
    """Build ``specimen_id -> {trait: allowed_token_set | None}`` from STRI.

    ``None`` for a trait means STRI does not code it for that species (the
    ``__uncoded`` flag is set), so the trait is skipped for that specimen.
    Returns ``(lookup, n_matched_specimens)``.
    """
    lookup: dict[str, dict[str, set[str] | None]] = {}
    n_matched = 0
    for specimen_id, species in specimen_to_species.items():
        row = stri_rows.get((species or "").strip().lower())
        if row is None:
            continue
        n_matched += 1
        per_trait: dict[str, set[str] | None] = {}
        for trait, token_col in STRI_TOKEN_COLUMN.items():
            if _is_one(row.get(f"{STRI_CATEGORY[trait]}__uncoded")):
                per_trait[trait] = None
                continue
            per_trait[trait] = {
                token for token, col in token_col.items() if _is_one(row.get(col))
            }
        lookup[specimen_id] = per_trait
    return lookup, n_matched


def accuracy_vs_stri(
    source_lookup: Mapping[str, Mapping[str, str]],
    stri_lookup: Mapping[str, Mapping[str, set[str] | None]],
    source_name: str,
) -> list[STRIAgreement]:
    """Per-trait match-any accuracy of a source against STRI, plus subset kappa.

    ``accuracy`` covers every comparable specimen (match-any). ``cohen_kappa`` is
    computed only over the single-label subset -- specimens STRI codes with
    exactly one allowed value -- where a symmetric single-label comparison is
    well defined; multi-label species are excluded from kappa but still counted
    in accuracy.
    """
    specimen_ids = set(source_lookup) & set(stri_lookup)
    results: list[STRIAgreement] = []
    for trait in STRI_TRAITS:
        n_compared = 0
        n_correct = 0
        src_labels: list[str] = []
        ref_labels: list[str] = []
        for sid in specimen_ids:
            allowed = stri_lookup[sid].get(trait)
            if allowed is None:  # uncoded for this species
                continue
            value = source_lookup[sid].get(trait, MISSING)
            if value == MISSING:
                continue
            n_compared += 1
            if value in allowed:
                n_correct += 1
            if len(allowed) == 1:  # single-label -> kappa is well defined
                (only,) = tuple(allowed)
                src_labels.append(value)
                ref_labels.append(only)
        results.append(
            STRIAgreement(
                trait_key=trait,
                source=source_name,
                n_compared=n_compared,
                n_correct=n_correct,
                accuracy=(n_correct / n_compared) if n_compared else None,
                n_kappa=len(src_labels),
                cohen_kappa=safe_kappa(src_labels, ref_labels),
            )
        )
    return results


def stri_pair_details(
    source_lookup: Mapping[str, Mapping[str, str]],
    stri_lookup: Mapping[str, Mapping[str, set[str] | None]],
    source_name: str,
    specimen_to_species: Mapping[str, str] | None = None,
) -> dict[str, list[STRIPairDetail]]:
    """Per-specimen detail behind each (trait, source) vs-STRI accuracy cell.

    Mirrors :func:`accuracy_vs_stri`' iteration -- only specimens STRI codes for the
    species and where the source has a non-``MISSING`` value appear, so the detail
    matches the accuracy denominator. Returns ``trait_key -> [STRIPairDetail]`` with
    mismatches first, then by specimen id.
    """
    specimen_ids = set(source_lookup) & set(stri_lookup)
    species_map = specimen_to_species or {}
    out: dict[str, list[STRIPairDetail]] = {}
    for trait in STRI_TRAITS:
        details: list[STRIPairDetail] = []
        for sid in specimen_ids:
            allowed = stri_lookup[sid].get(trait)
            if allowed is None:  # uncoded for this species
                continue
            value = source_lookup[sid].get(trait, MISSING)
            if value == MISSING:
                continue
            details.append(
                STRIPairDetail(
                    trait_key=trait,
                    source=source_name,
                    specimen_id=sid,
                    species=species_map.get(sid, ""),
                    value=value,
                    allowed=sorted(allowed),
                    correct=(value in allowed),
                )
            )
        details.sort(key=lambda d: (d.correct, d.specimen_id))
        out[trait] = details
    return out
