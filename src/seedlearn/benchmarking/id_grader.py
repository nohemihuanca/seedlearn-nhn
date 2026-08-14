"""Score Stage 5 species identification predictions against catalog labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class IDGradeRecord:
    """Grading result for species identification on one specimen.

    Attributes:
        specimen_id: Catalog ID_YPS value.
        true_family: Ground truth family from catalog.
        true_genus: Ground truth genus from catalog.
        true_species: Ground truth species binomial ("Genus epithet").
        pred_family: Stage 5 predicted family (None if unavailable).
        pred_genus: Stage 5 predicted genus.
        pred_species: Stage 5 predicted species binomial.
        family_correct: Whether family prediction matches.
        genus_correct: Whether genus prediction matches.
        species_correct: Whether species prediction matches.
        confidence: Stage 5 confidence level (high/medium/low).
        stage5_error: True if Stage 5 failed or was skipped.
    """

    specimen_id: str
    true_family: str
    true_genus: str
    true_species: str
    pred_family: str | None
    pred_genus: str | None
    pred_species: str | None
    family_correct: bool
    genus_correct: bool
    species_correct: bool
    confidence: str | None
    stage5_error: bool
    partition: str | None = None


def _normalize(name: str | None) -> str:
    """Lowercase and strip a taxonomic name for comparison."""
    if name is None:
        return ""
    return name.strip().lower()


def _normalize_species(pred: str | None, true_binomial: str) -> tuple[str, str]:
    """Normalize predicted and true species for consistent comparison.

    Handles cases where the prediction is just the epithet ("laurina")
    or the full binomial ("Inga laurina"). Compares as full binomials
    when possible, falling back to epithet-only comparison.

    Args:
        pred: Predicted species string (epithet or binomial).
        true_binomial: True species as "Genus epithet".

    Returns:
        Tuple of (normalized_pred, normalized_true) for comparison.
    """
    if pred is None:
        return "", _normalize(true_binomial)

    pred_norm = _normalize(pred)
    true_norm = _normalize(true_binomial)

    # If prediction is a single word, treat it as an epithet —
    # compare against the epithet part of the true binomial
    pred_parts = pred_norm.split()
    true_parts = true_norm.split()

    if len(pred_parts) == 1 and len(true_parts) == 2:
        # Prediction is just the epithet, compare epithet to epithet
        return pred_parts[0], true_parts[1]

    # Both are binomials (or other formats) — compare directly
    return pred_norm, true_norm


def compare_taxonomy(
    true_family: str,
    true_genus: str,
    true_species: str,
    pred_family: str | None,
    pred_genus: str | None,
    pred_species: str | None,
) -> tuple[bool, bool, bool]:
    """Compare a predicted taxonomy against the truth, case-insensitively.

    Pure comparison core shared by the Stage 5 grader and the human-ID grader.
    Species comparison handles epithet-only predictions via
    :func:`_normalize_species`. Returns ``(family_correct, genus_correct,
    species_correct)``.
    """
    family_correct = _normalize(pred_family) == _normalize(true_family)
    genus_correct = _normalize(pred_genus) == _normalize(true_genus)
    pred_sp_norm, true_sp_norm = _normalize_species(pred_species, true_species)
    species_correct = pred_sp_norm == true_sp_norm
    return family_correct, genus_correct, species_correct


def grade_specimen_id(
    specimen_id: str,
    true_family: str,
    true_genus: str,
    true_species: str,
    pipeline_result: dict[str, Any],
    partition: str | None = None,
) -> IDGradeRecord:
    """Grade Stage 5 identification against catalog ground truth.

    Extracts predicted family/genus/species from the reasoning stage output
    and compares case-insensitively to catalog labels.

    Args:
        specimen_id: Catalog specimen ID.
        true_family: Ground truth family name.
        true_genus: Ground truth genus name.
        true_species: Ground truth species binomial ("Genus epithet").
        pipeline_result: Full pipeline result dict (from PipelineResult.to_dict()).
        partition: Optional train/val/test partition label.

    Returns:
        IDGradeRecord with correctness flags.
    """
    stages = pipeline_result.get("stages", {})
    reasoning = stages.get("reasoning", {})

    # Check for stage error/skip
    if reasoning.get("error") or reasoning.get("skipped"):
        return IDGradeRecord(
            specimen_id=specimen_id,
            true_family=true_family,
            true_genus=true_genus,
            true_species=true_species,
            pred_family=None,
            pred_genus=None,
            pred_species=None,
            family_correct=False,
            genus_correct=False,
            species_correct=False,
            confidence=None,
            stage5_error=True,
            partition=partition,
        )

    data = reasoning.get("data", {})
    classification = data.get("classification", {})

    pred_family = classification.get("predicted_family")
    pred_genus = classification.get("predicted_genus")
    pred_species = classification.get("predicted_species")
    confidence = classification.get("confidence")

    # Case-insensitive comparison (shared pure core)
    family_correct, genus_correct, species_correct = compare_taxonomy(
        true_family, true_genus, true_species, pred_family, pred_genus, pred_species
    )

    return IDGradeRecord(
        specimen_id=specimen_id,
        true_family=true_family,
        true_genus=true_genus,
        true_species=true_species,
        pred_family=pred_family,
        pred_genus=pred_genus,
        pred_species=pred_species,
        family_correct=family_correct,
        genus_correct=genus_correct,
        species_correct=species_correct,
        confidence=confidence,
        stage5_error=False,
        partition=partition,
    )
