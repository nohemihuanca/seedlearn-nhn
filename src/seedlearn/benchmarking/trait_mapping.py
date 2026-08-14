"""Declarative mapping from VLM morphology output to STRI binary trait columns.

Maps the nested trait dictionary produced by Stage 1 (MorphologyStage) to the
flat binary column format used in STRI trait matrices (e.g., cl185).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TraitRule:
    """Maps one pipeline trait field value to a single STRI binary column.

    Attributes:
        stri_column: Target STRI column name (e.g., "leaf_arrangement__alternate").
        category: Trait category for grouping (e.g., "leaf_arrangement").
        pipeline_section: Section key in the nested traits dict.
        pipeline_field: Field key within the section.
        match_values: Set of VLM output values that map to STRI=1.
            Uses substring matching (case-insensitive) against the
            normalized pipeline value.
    """

    stri_column: str
    category: str
    pipeline_section: str
    pipeline_field: str
    match_values: frozenset[str]


# ---------------------------------------------------------------------------
# Trait rules — ordered by STRI category
# ---------------------------------------------------------------------------

TRAIT_RULES: list[TraitRule] = [
    # --- leaf_arrangement ---
    TraitRule(
        stri_column="leaf_arrangement__alternate",
        category="leaf_arrangement",
        pipeline_section="leaf_arrangement",
        pipeline_field="relative_position",
        match_values=frozenset({"alternate"}),
    ),
    TraitRule(
        stri_column="leaf_arrangement__opposite",
        category="leaf_arrangement",
        pipeline_section="leaf_arrangement",
        pipeline_field="relative_position",
        match_values=frozenset({"opposite"}),
    ),
    TraitRule(
        stri_column="leaf_arrangement__whorled_or_clustered",
        category="leaf_arrangement",
        pipeline_section="leaf_arrangement",
        pipeline_field="relative_position",
        match_values=frozenset({"whorled"}),
    ),
    # leaf_arrangement__fascicled — SKIPPED (no reliable pipeline mapping)
    # leaf_arrangement__basal_rosette — SKIPPED (not expected in seedlings)
    # --- leaf_type ---
    TraitRule(
        stri_column="leaf_type__simple",
        category="leaf_type",
        pipeline_section="leaf_complexity",
        pipeline_field="type",
        match_values=frozenset({"simple"}),
    ),
    TraitRule(
        stri_column="leaf_type__compound",
        category="leaf_type",
        pipeline_section="leaf_complexity",
        pipeline_field="type",
        match_values=frozenset({"compound"}),
    ),
    # leaf_type__pinnatifid_pinnatisect — DROPPED per collaborator
    # --- leaf_margin ---
    TraitRule(
        stri_column="leaf_margin__entire",
        category="leaf_margin",
        pipeline_section="leaf_morphology",
        pipeline_field="margin",
        match_values=frozenset({"entire"}),
    ),
    TraitRule(
        stri_column="leaf_margin__toothed",
        category="leaf_margin",
        pipeline_section="leaf_morphology",
        pipeline_field="margin",
        match_values=frozenset(
            {"toothed", "dentate", "serrate", "crenate", "serrulate", "denticulate"}
        ),
    ),
    TraitRule(
        stri_column="leaf_margin__lobed",
        category="leaf_margin",
        pipeline_section="leaf_morphology",
        pipeline_field="margin",
        match_values=frozenset({"lobed", "palmately lobed", "pinnatifid"}),
    ),
    # --- stipules ---
    TraitRule(
        stri_column="stipules__present",
        category="stipules",
        pipeline_section="special_features",
        pipeline_field="stipules",
        match_values=frozenset({"present"}),
    ),
    TraitRule(
        stri_column="stipules__absent",
        category="stipules",
        pipeline_section="special_features",
        pipeline_field="stipules",
        match_values=frozenset({"absent"}),
    ),
    # --- latex ---
    TraitRule(
        stri_column="latex__present",
        category="latex",
        pipeline_section="special_features",
        pipeline_field="latex",
        match_values=frozenset({"present"}),
    ),
    TraitRule(
        stri_column="latex__absent",
        category="latex",
        pipeline_section="special_features",
        pipeline_field="latex",
        match_values=frozenset({"absent"}),
    ),
]

# Categories available for grading
GRADED_CATEGORIES: list[str] = sorted(
    {rule.category for rule in TRAIT_RULES}
)


def map_prediction(
    traits: dict[str, Any],
    column_suffix: str = "",
) -> dict[str, int | None]:
    """Map VLM nested trait output to STRI binary column predictions.

    Args:
        traits: Nested trait dict from Stage 1 (e.g.,
            ``{"leaf_arrangement": {"relative_position": "alternate"}, ...}``).
        column_suffix: Optional suffix appended to STRI column names
            (e.g., ``"__consensus"`` for merged matrix columns).

    Returns:
        Dict mapping STRI column names (with suffix) to predicted binary
        values: 1 (predicted present), 0 (predicted absent within same
        category), or None (field empty/missing/unrecognizable).
    """
    if not traits:
        return {
            rule.stri_column + column_suffix: None for rule in TRAIT_RULES
        }

    # First pass: determine which rule matched per category
    category_match: dict[str, str | None] = {}  # category -> matched stri_column
    category_has_value: dict[str, bool] = {}     # category -> pipeline field non-empty

    for rule in TRAIT_RULES:
        section = traits.get(rule.pipeline_section, {})
        if not isinstance(section, dict):
            continue
        raw_value = section.get(rule.pipeline_field, "")
        if not isinstance(raw_value, str):
            raw_value = str(raw_value) if raw_value else ""

        normalized = raw_value.lower().strip()
        has_value = bool(normalized) and normalized not in {
            "n/a", "not visible", "unclear", "unknown", "cannot determine",
            "not observed", "none",
        }

        if rule.category not in category_has_value:
            category_has_value[rule.category] = has_value

        if has_value and rule.category not in category_match:
            # Check if any match_value is a substring of the normalized value
            for match_val in rule.match_values:
                if match_val in normalized:
                    category_match[rule.category] = rule.stri_column
                    break

    # Second pass: assign binary predictions
    result: dict[str, int | None] = {}
    for rule in TRAIT_RULES:
        col = rule.stri_column + column_suffix

        if not category_has_value.get(rule.category, False):
            result[col] = None
            continue

        matched_col = category_match.get(rule.category)
        if matched_col is None:
            # Value present but no rule matched — can't score
            result[col] = None
            continue

        result[col] = 1 if matched_col == rule.stri_column else 0

    return result


def get_raw_vlm_values(traits: dict[str, Any]) -> dict[str, str]:
    """Extract raw VLM values for each trait category.

    Looks up the pipeline section and field for each category (using the
    first matching rule) and returns the raw string value.

    Args:
        traits: Nested trait dict from Stage 1 (e.g.,
            ``{"leaf_arrangement": {"relative_position": "alternate"}, ...}``).

    Returns:
        Dict mapping category name to raw VLM string value.
    """
    result: dict[str, str] = {}
    seen: set[str] = set()

    for rule in TRAIT_RULES:
        if rule.category in seen:
            continue
        section = traits.get(rule.pipeline_section, {})
        if isinstance(section, dict):
            val = section.get(rule.pipeline_field, "")
            result[rule.category] = str(val) if val else ""
        else:
            result[rule.category] = ""
        seen.add(rule.category)

    return result


def detect_not_observed(traits: dict[str, Any], category: str) -> bool:
    """Check if the raw VLM value for a category is "not observed".

    Args:
        traits: Nested trait dict from Stage 1.
        category: Trait category name (e.g., "latex", "stipules").

    Returns:
        True if the raw VLM value contains "not observed".
    """
    for rule in TRAIT_RULES:
        if rule.category == category:
            section = traits.get(rule.pipeline_section, {})
            if isinstance(section, dict):
                raw = section.get(rule.pipeline_field, "")
                if isinstance(raw, str) and "not observed" in raw.lower():
                    return True
            return False
    return False
