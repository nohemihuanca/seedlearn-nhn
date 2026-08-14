"""Reviewable corrections for a human annotator's photo-based species IDs.

Roni's IDs contain obvious spelling typos, Latin orthographic variants, and accepted
taxonomic synonyms that a strict string comparison marks wrong. Rather than editing the
raw annotation data, an auditable CSV (``trait_grading/id_corrections.csv``) lists each
correction; the grader consults it to compute a *corrected* accuracy alongside the
untouched *raw* accuracy.

Each row: ``specimen_id, rank, roni_original, canonical, category, note`` where ``rank``
is one of ``family|genus|species`` and ``category`` is one of ``typo|variant|synonym``.
A rank prediction is credited as corrected-correct when its original text matches the
row's ``roni_original`` (normalized) for that ``(specimen_id, rank)`` **and** the row's
``canonical`` matches the true value. Nothing is credited that is not listed here, and
the raw prediction text is never mutated.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

RANKS = ("family", "genus", "species")
CATEGORIES = ("typo", "variant", "synonym")


@dataclass(frozen=True)
class Correction:
    """One reviewable correction for a single (specimen, rank) ID prediction."""

    specimen_id: str
    rank: str
    roni_original: str
    canonical: str
    category: str
    note: str = ""


def _norm(value: str | None) -> str:
    """Lowercase + strip, matching id_grader's comparison normalization."""
    return "" if value is None else value.strip().lower()


def load_corrections(
    path: str | Path,
) -> dict[tuple[str, str], Correction]:
    """Load the corrections CSV into a ``(specimen_id, rank) -> Correction`` map.

    A missing file yields an empty map (corrections are optional). Rows with an
    unknown ``rank`` or ``category`` are skipped so a malformed edit cannot silently
    credit the wrong thing.
    """
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[tuple[str, str], Correction] = {}
    with open(p, newline="") as fh:
        for row in csv.DictReader(fh):
            rank = (row.get("rank") or "").strip().lower()
            category = (row.get("category") or "").strip().lower()
            specimen_id = (row.get("specimen_id") or "").strip()
            if rank not in RANKS or category not in CATEGORIES or not specimen_id:
                continue
            out[(specimen_id, rank)] = Correction(
                specimen_id=specimen_id,
                rank=rank,
                roni_original=(row.get("roni_original") or "").strip(),
                canonical=(row.get("canonical") or "").strip(),
                category=category,
                note=(row.get("note") or "").strip(),
            )
    return out


def match_correction(
    correction: Correction | None,
    original_pred: str | None,
    true_value: str,
) -> bool:
    """Whether ``correction`` credits ``original_pred`` against ``true_value``.

    True only when the correction's recorded original matches what Roni actually
    wrote (normalized) *and* its canonical value matches the truth. This guards
    against stale entries silently crediting a value Roni no longer predicts.
    """
    if correction is None:
        return False
    if _norm(correction.roni_original) != _norm(original_pred):
        return False
    return _norm(correction.canonical) == _norm(true_value)


def stale_corrections(
    corrections: dict[tuple[str, str], Correction],
    originals: dict[tuple[str, str], str | None],
) -> list[Correction]:
    """Corrections whose recorded original no longer matches the annotator's text.

    ``originals`` maps ``(specimen_id, rank) -> the value Roni actually wrote``.
    A correction is stale when its key is absent from ``originals`` or its
    ``roni_original`` does not match the current value; such entries credit nothing
    and are surfaced for review rather than silently ignored.
    """
    stale: list[Correction] = []
    for key, corr in corrections.items():
        if key not in originals or _norm(corr.roni_original) != _norm(originals[key]):
            stale.append(corr)
    return stale
