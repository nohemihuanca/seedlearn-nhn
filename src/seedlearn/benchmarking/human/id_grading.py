"""Grade a human annotator's photo-based species IDs against true taxonomy.

Roni recorded family / genus / species guesses from the photos alone (once per
individual). This module compares those guesses to the curator's true taxonomy,
reusing the same pure comparison core as the Stage 5 model-ID grader so the two
are scored identically (case-insensitive, epithet-only fallback).
"""

from __future__ import annotations

from dataclasses import dataclass

from seedlearn.benchmarking.human.annotations import AnnotationRecord
from seedlearn.benchmarking.human.id_corrections import Correction, match_correction
from seedlearn.benchmarking.id_grader import compare_taxonomy


@dataclass
class HumanIDRecord:
    """Grading result for one annotator's species ID of one individual.

    ``*_correct`` are the raw string-comparison verdicts (untouched). ``*_corrected``
    additionally credit a reviewable correction (typo / variant / synonym) when one
    applies; ``*_correction`` carries the crediting :class:`Correction` (``None`` when
    the rank was raw-correct or is still wrong). Roni's original ``pred_*`` text is
    never mutated.
    """

    annotator: str
    anonymous_id: str
    specimen_id: str | None
    true_family: str
    true_genus: str
    true_species: str  # epithet
    pred_family: str | None
    pred_genus: str | None
    pred_species: str | None
    family_correct: bool
    genus_correct: bool
    species_correct: bool
    family_corrected: bool = False
    genus_corrected: bool = False
    species_corrected: bool = False
    family_correction: Correction | None = None
    genus_correction: Correction | None = None
    species_correction: Correction | None = None


def _true_binomial(genus: str | None, species: str | None) -> str:
    return f"{(genus or '').strip()} {(species or '').strip()}".strip()


def grade_human_ids(
    records: list[AnnotationRecord],
    annotator: str = "roni",
    corrections: dict[tuple[str, str], Correction] | None = None,
) -> list[HumanIDRecord]:
    """Grade ``annotator``'s ID predictions against the joined true taxonomy.

    Each individual is graded once (from the first view that carries an ID).
    Individuals the annotator did not identify, or that did not join to a
    specimen, are skipped. When ``corrections`` is supplied, each rank is *also*
    graded corrected: raw-correct, or credited by a matching reviewable correction
    (typo / variant / synonym). Raw verdicts and Roni's original text are unchanged.
    """
    corrections = corrections or {}
    seen: set[str] = set()
    graded: list[HumanIDRecord] = []
    for rec in records:
        if rec.annotator != annotator or rec.anonymous_id in seen:
            continue
        if not (rec.id_family or rec.id_genus or rec.id_species):
            continue
        if rec.true_family is None and rec.true_genus is None:
            continue  # unmatched to curator key
        seen.add(rec.anonymous_id)
        true_binomial = _true_binomial(rec.true_genus, rec.true_species)
        fam_ok, gen_ok, sp_ok = compare_taxonomy(
            rec.true_family or "",
            rec.true_genus or "",
            true_binomial,
            rec.id_family,
            rec.id_genus,
            rec.id_species,
        )

        # Corrected verdict per rank: raw-correct, or credited by a matching entry.
        sid = rec.specimen_id or ""
        rank_inputs = {
            "family": (fam_ok, rec.id_family, rec.true_family or ""),
            "genus": (gen_ok, rec.id_genus, rec.true_genus or ""),
            "species": (sp_ok, rec.id_species, rec.true_species or ""),
        }
        corrected: dict[str, bool] = {}
        credited: dict[str, Correction | None] = {}
        for rank, (raw_ok, pred, true_val) in rank_inputs.items():
            corr = corrections.get((sid, rank))
            if not raw_ok and match_correction(corr, pred, true_val):
                corrected[rank], credited[rank] = True, corr
            else:
                corrected[rank], credited[rank] = raw_ok, None

        graded.append(
            HumanIDRecord(
                annotator=annotator,
                anonymous_id=rec.anonymous_id,
                specimen_id=rec.specimen_id,
                true_family=rec.true_family or "",
                true_genus=rec.true_genus or "",
                true_species=rec.true_species or "",
                pred_family=rec.id_family,
                pred_genus=rec.id_genus,
                pred_species=rec.id_species,
                family_correct=fam_ok,
                genus_correct=gen_ok,
                species_correct=sp_ok,
                family_corrected=corrected["family"],
                genus_corrected=corrected["genus"],
                species_corrected=corrected["species"],
                family_correction=credited["family"],
                genus_correction=credited["genus"],
                species_correction=credited["species"],
            )
        )
    return graded


def id_accuracy(records: list[HumanIDRecord]) -> dict[str, float | int]:
    """Raw and corrected family / genus / species accuracy over graded records.

    Returns raw ``*_accuracy`` (unchanged from before) plus ``corrected_*_accuracy``
    and per-category credited counts, so a report can show both scores side by side.
    """
    n = len(records)
    if n == 0:
        return {"n_graded": 0}
    by_category: dict[str, int] = {}
    for r in records:
        for corr in (r.family_correction, r.genus_correction, r.species_correction):
            if corr is not None:
                by_category[corr.category] = by_category.get(corr.category, 0) + 1
    return {
        "n_graded": n,
        "family_accuracy": sum(r.family_correct for r in records) / n,
        "genus_accuracy": sum(r.genus_correct for r in records) / n,
        "species_accuracy": sum(r.species_correct for r in records) / n,
        "corrected_family_accuracy": sum(r.family_corrected for r in records) / n,
        "corrected_genus_accuracy": sum(r.genus_corrected for r in records) / n,
        "corrected_species_accuracy": sum(r.species_corrected for r in records) / n,
        "n_credited_by_category": by_category,
    }
