"""Synonym-aware specimen resolver for STRI trait matrix benchmarking.

Matches catalog species to STRI trait matrix rows using direct name matching
and synonym resolution from the iNaturalist metadata table.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from seedlearn.data.catalog import iter_image_paths, load_catalog

logger = logging.getLogger(__name__)


@dataclass
class OverlapSpecimen:
    """A catalog specimen matched to a STRI trait matrix row.

    Attributes:
        specimen_id: Catalog ID_YPS value (e.g., "SRAPHEDE2").
        scientific_name: Accepted name from catalog ("Genus epithet").
        family: Taxonomic family.
        genus: Genus name.
        species_epithet: Species epithet (lowercase).
        image_paths: Absolute paths to training images.
        stri_match_name: The name that matched in the STRI matrix.
        match_method: How the match was found ("direct", "synonym", "sci_final2").
    """

    specimen_id: str
    scientific_name: str
    family: str
    genus: str
    species_epithet: str
    image_paths: list[str] = field(default_factory=list)
    stri_match_name: str = ""
    match_method: str = "direct"


def _build_synonym_table(
    synonym_path: Path,
) -> dict[str, list[tuple[str, str]]]:
    """Build accepted_name -> [(synonym, match_method), ...] lookup.

    Args:
        synonym_path: Path to inat_metadata CSV with accepted_name,
            synonyms, and scientific_name_final2 columns.

    Returns:
        Dict mapping lowercased accepted_name to list of
        (lowercased_synonym, match_method) tuples.
    """
    df = pd.read_csv(synonym_path, dtype=str).fillna("")

    table: dict[str, list[tuple[str, str]]] = {}

    for _, row in df.drop_duplicates(subset=["accepted_name"]).iterrows():
        accepted = row.get("accepted_name", "").strip()
        if not accepted:
            continue
        key = accepted.lower()

        synonyms: list[tuple[str, str]] = []

        # Parse comma-separated synonyms column
        syn_raw = row.get("synonyms", "").strip()
        if syn_raw and syn_raw.upper() != "NA":
            for syn in syn_raw.split(","):
                syn = syn.strip()
                if syn and syn.lower() != key:
                    synonyms.append((syn.lower(), "synonym"))

        # scientific_name_final2 as additional synonym
        sci_final = row.get("scientific_name_final2", "").strip()
        if sci_final and sci_final.lower() != key:
            already = {s[0] for s in synonyms}
            if sci_final.lower() not in already:
                synonyms.append((sci_final.lower(), "sci_final2"))

        if synonyms:
            table[key] = synonyms

    return table


def _match_species_to_stri(
    accepted_name: str,
    stri_names: set[str],
    synonym_table: dict[str, list[tuple[str, str]]],
) -> tuple[str, str] | None:
    """Try to match a catalog species to a STRI matrix row.

    Args:
        accepted_name: Accepted species name (lowercased).
        stri_names: Set of lowercased STRI scientific_name values.
        synonym_table: Synonym lookup from _build_synonym_table.

    Returns:
        Tuple of (matched_stri_name, match_method) or None if no match.
    """
    # Direct match
    if accepted_name in stri_names:
        return accepted_name, "direct"

    # Synonym match
    for syn, method in synonym_table.get(accepted_name, []):
        if syn in stri_names:
            return syn, method

    return None


def read_specimen_ids(source_path: Path) -> list[str]:
    """Read an explicit specimen-id list from a curator key or plain id list.

    Accepts the curator taxonomic key CSV (uses its ``individual_code`` column)
    or any text/CSV file with one specimen id per line. Order is preserved and
    duplicates removed.
    """
    ids: list[str] = []
    seen: set[str] = set()
    with open(source_path, newline="") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        if "individual_code" in sample:
            for row in csv.DictReader(fh):
                code = (row.get("individual_code") or "").strip()
                if code and code not in seen:
                    seen.add(code)
                    ids.append(code)
        else:
            for line in fh:
                code = line.strip().split(",")[0].strip()
                if code and code not in seen:
                    seen.add(code)
                    ids.append(code)
    return ids


def load_specimens_by_id(
    catalog_path: Path,
    specimen_source: Path,
) -> tuple[list[OverlapSpecimen], list[str]]:
    """Build specimens for an explicit id list, directly from the catalog.

    Unlike :func:`load_overlap_specimens`, this does **not** require a STRI match,
    so it covers every requested specimen (e.g. the full annotated set, including
    specimens whose species has no STRI trait-matrix row).

    Returns ``(specimens, missing_ids)`` where ``missing_ids`` are requested ids
    absent from the catalog or lacking images.
    """
    wanted = read_specimen_ids(specimen_source)
    wanted_set = set(wanted)
    catalog_df = load_catalog(catalog_path)

    by_id: dict[str, OverlapSpecimen] = {}
    for specimen_id, group in catalog_df.groupby("ID_YPS"):
        sid = str(specimen_id)
        if sid not in wanted_set:
            continue
        row = group.iloc[0]
        genus = str(row["GENUS"]).replace("_", " ").strip()
        epithet = str(row["SPECIES"]).replace("_", " ").strip().lower()
        family = str(row["FAMILY"]).replace("_", " ").strip()

        image_paths: list[str] = []
        for _, r in group.iterrows():
            train_dir = Path(str(r["training_absolute_path"]))
            for img in iter_image_paths(train_dir):
                image_paths.append(str(img))
        if not image_paths:
            logger.warning("Specimen %s has no images, skipping", sid)
            continue

        by_id[sid] = OverlapSpecimen(
            specimen_id=sid,
            scientific_name=f"{genus} {epithet}",
            family=family,
            genus=genus,
            species_epithet=epithet,
            image_paths=image_paths,
            stri_match_name="",
            match_method="curator_selection",
        )

    # Preserve requested order; report any not built.
    specimens = [by_id[s] for s in wanted if s in by_id]
    missing = [s for s in wanted if s not in by_id]
    if missing:
        logger.warning(
            "%d/%d requested specimens not found in catalog: %s",
            len(missing), len(wanted), missing,
        )
    logger.info("Selected %d specimens from explicit id list", len(specimens))
    return specimens, missing


def load_overlap_specimens(
    catalog_path: Path,
    stri_matrix_path: Path,
    synonym_path: Path | None = None,
) -> list[OverlapSpecimen]:
    """Find catalog specimens that match STRI trait matrix species.

    Args:
        catalog_path: Path to species catalog CSV.
        stri_matrix_path: Path to STRI trait matrix CSV (cl185 or merged).
        synonym_path: Path to inat_metadata CSV with synonym columns.
            If None, only direct name matching is used.

    Returns:
        List of OverlapSpecimen, one per unique catalog specimen (ID_YPS),
        sorted alphabetically by specimen_id.
    """
    catalog_df = load_catalog(catalog_path)
    stri_df = pd.read_csv(stri_matrix_path, dtype={"taxon_id": str})

    # Build lowercased STRI name set and original-case lookup
    stri_name_to_original: dict[str, str] = {}
    for name in stri_df["scientific_name"].dropna().unique():
        stri_name_to_original[name.strip().lower()] = name.strip()

    stri_names = set(stri_name_to_original.keys())

    # Build synonym table if provided
    synonym_table: dict[str, list[tuple[str, str]]] = {}
    if synonym_path is not None:
        synonym_table = _build_synonym_table(synonym_path)

    # Group catalog by specimen (ID_YPS)
    specimens: list[OverlapSpecimen] = []
    matched_count = 0
    synonym_count = 0

    grouped = catalog_df.groupby("ID_YPS")
    for specimen_id, group in grouped:
        row = group.iloc[0]
        genus = str(row["GENUS"]).replace("_", " ").strip()
        epithet = str(row["SPECIES"]).replace("_", " ").strip().lower()
        family = str(row["FAMILY"]).replace("_", " ").strip()
        accepted_name = f"{genus} {epithet}"

        match = _match_species_to_stri(
            accepted_name.lower(), stri_names, synonym_table
        )
        if match is None:
            continue

        stri_match_lower, method = match
        stri_match_name = stri_name_to_original[stri_match_lower]

        # Resolve image paths from training directories
        image_paths: list[str] = []
        for _, r in group.iterrows():
            train_dir = Path(str(r["training_absolute_path"]))
            for img in iter_image_paths(train_dir):
                image_paths.append(str(img))

        if not image_paths:
            logger.warning(
                "Specimen %s has no images, skipping", specimen_id
            )
            continue

        specimens.append(
            OverlapSpecimen(
                specimen_id=str(specimen_id),
                scientific_name=accepted_name,
                family=family,
                genus=genus,
                species_epithet=epithet,
                image_paths=image_paths,
                stri_match_name=stri_match_name,
                match_method=method,
            )
        )
        matched_count += 1
        if method != "direct":
            synonym_count += 1

    specimens.sort(key=lambda s: s.specimen_id)

    logger.info(
        "Found %d overlap specimens (%d via synonym) across %d unique species",
        matched_count,
        synonym_count,
        len({s.scientific_name.lower() for s in specimens}),
    )

    return specimens
