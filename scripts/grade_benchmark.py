#!/usr/bin/env python
"""Grade benchmark pipeline results against STRI ground truth.

Reads per-specimen JSON outputs from ``run_benchmark_pipeline.py``, scores
trait extraction accuracy and species ID accuracy, and writes grading CSVs.

Can be re-run any number of times without re-running the pipeline.

Usage:
    # Grade against cl185
    python scripts/grade_benchmark.py \\
        --results-dir data/benchmarks/2026-03-04/ \\
        --catalog data/raw/2026-01-29/sorted_12K/metadata/species_catalog_*.csv \\
        --stri-matrix data/traits/stri_web_keys/per_key_trait_matrices/cl185_*.csv

    # Grade against both cl185 and merged
    python scripts/grade_benchmark.py \\
        --results-dir data/benchmarks/2026-03-04/ \\
        --catalog data/raw/2026-01-29/sorted_12K/metadata/species_catalog_*.csv \\
        --stri-matrix data/traits/stri_web_keys/per_key_trait_matrices/cl185_*.csv \\
        --stri-matrix data/traits/stri_web_keys/merged/stri_all_sources_merged_trait_matrix.csv

    # Skip ID grading (only grade traits)
    python scripts/grade_benchmark.py ... --skip-id-grading
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from seedlearn.benchmarking.id_grader import IDGradeRecord, grade_specimen_id
from seedlearn.benchmarking.report import (
    print_summary,
    save_benchmark_results,
)
from seedlearn.benchmarking.trait_mapping import TRAIT_RULES
from seedlearn.benchmarking.trait_grader import TraitGradeRecord, grade_specimen_traits

from seedlearn.data.catalog import load_dataset
from seedlearn.data.splits import load_split

logger = logging.getLogger(__name__)

DEFAULT_SYNONYM_TABLE = "configs/species_lists/inat_metadata_FINAL_NHN_01_2025.csv"


def build_specimen_partition_map(
    catalog_path: Path,
    split_path: Path,
) -> dict[str, str]:
    """Map specimen IDs to train/val/test partition labels.

    Args:
        catalog_path: Path to the species catalog CSV.
        split_path: Path to split files (without extension).

    Returns:
        Dict mapping specimen_id (individual_id) to partition name.
    """
    rank = split_path.parent.name  # e.g., "family", "genus", "species"
    records, _ = load_dataset(catalog_path, rank)
    split = load_split(split_path)

    partition_map: dict[str, str] = {}
    for idx in split.train_indices:
        partition_map[records[idx].individual_id] = "train"
    for idx in split.val_indices:
        partition_map[records[idx].individual_id] = "val"
    for idx in split.test_indices:
        partition_map[records[idx].individual_id] = "test"

    return partition_map


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (None = sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Grade benchmark pipeline results against STRI ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Directory containing per-specimen JSON results.",
    )
    parser.add_argument(
        "--catalog", type=str, required=True,
        help="Path to species catalog CSV.",
    )
    parser.add_argument(
        "--stri-matrix", type=str, action="append", required=True,
        dest="stri_matrices",
        help="Path to STRI trait matrix CSV. Can be specified multiple times.",
    )
    parser.add_argument(
        "--synonym-table", type=str, default=DEFAULT_SYNONYM_TABLE,
        help="Path to inat_metadata CSV with synonym columns.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for grading output (default: <results-dir>/grades/).",
    )
    parser.add_argument(
        "--skip-id-grading", action="store_true", default=False,
        help="Only grade traits, skip Stage 5 species ID grading.",
    )
    parser.add_argument(
        "--skip-trait-grading", action="store_true", default=False,
        help="Only grade Stage 5 species ID, skip trait grading.",
    )
    parser.add_argument(
        "--html", action="store_true", default=False,
        help="Generate browsable HTML diagnostic report.",
    )
    parser.add_argument(
        "--split-path", type=str, default=None,
        help="Path to split files (without extension) for partition-aware ID grading.",
    )
    parser.add_argument("--verbose", action="store_true", default=False)

    return parser.parse_args(argv)


def _detect_column_suffix(stri_df: pd.DataFrame) -> str:
    """Detect whether the STRI matrix uses consensus column suffixes.

    Args:
        stri_df: Loaded STRI trait matrix DataFrame.

    Returns:
        Column suffix string ("__consensus" for merged, "" for per-key).
    """
    cols = set(stri_df.columns)
    if "leaf_arrangement__alternate__consensus" in cols:
        return "__consensus"
    if "leaf_arrangement__alternate" in cols:
        return ""
    # Fallback: check for any consensus column
    if any(c.endswith("__consensus") for c in cols):
        return "__consensus"
    return ""


def _detect_source_label(stri_path: Path) -> str:
    """Derive a short label from the STRI matrix filename.

    Args:
        stri_path: Path to the STRI matrix CSV.

    Returns:
        Short label string (e.g., "cl185", "merged").
    """
    name = stri_path.stem.lower()
    if "merged" in name:
        return "merged"
    # Extract cl{N} pattern
    for part in name.split("_"):
        if part.startswith("cl") and part[2:].isdigit():
            return part
    return stri_path.stem[:20]


def _build_synonym_lookup(
    synonym_path: Path | None,
) -> dict[str, list[str]]:
    """Build accepted_name -> [synonym, ...] lookup for STRI matching.

    Args:
        synonym_path: Path to inat_metadata CSV.

    Returns:
        Dict mapping lowercased accepted_name to list of lowercased synonyms.
    """
    if synonym_path is None or not synonym_path.exists():
        return {}

    df = pd.read_csv(synonym_path, dtype=str).fillna("")
    table: dict[str, list[str]] = {}

    for _, row in df.drop_duplicates(subset=["accepted_name"]).iterrows():
        accepted = row.get("accepted_name", "").strip().lower()
        if not accepted:
            continue

        synonyms: list[str] = []
        syn_raw = row.get("synonyms", "").strip()
        if syn_raw and syn_raw.upper() != "NA":
            for syn in syn_raw.split(","):
                syn = syn.strip().lower()
                if syn and syn != accepted:
                    synonyms.append(syn)

        sci_final = row.get("scientific_name_final2", "").strip().lower()
        if sci_final and sci_final != accepted and sci_final not in synonyms:
            synonyms.append(sci_final)

        if synonyms:
            table[accepted] = synonyms

    return table


def _find_stri_row(
    scientific_name: str,
    stri_df: pd.DataFrame,
    stri_name_index: dict[str, int],
    synonym_lookup: dict[str, list[str]],
) -> pd.Series | None:
    """Look up a species in the STRI matrix by name or synonym.

    Args:
        scientific_name: Accepted species name.
        stri_df: STRI trait matrix DataFrame.
        stri_name_index: Lowercased scientific_name -> row index.
        synonym_lookup: Accepted name -> synonym list.

    Returns:
        STRI matrix row as a Series, or None if not found.
    """
    key = scientific_name.lower()

    # Direct match
    if key in stri_name_index:
        return stri_df.iloc[stri_name_index[key]]

    # Synonym match
    for syn in synonym_lookup.get(key, []):
        if syn in stri_name_index:
            return stri_df.iloc[stri_name_index[syn]]

    return None


def _compute_stri_support(
    trait_records: list[TraitGradeRecord],
) -> dict[str, int]:
    """Count how many graded specimens have gt=1 for each STRI column.

    A specimen is "graded" for a column if its verdict is CORRECT or INCORRECT.
    This provides the support count needed for F1 calculation (FN = support - TP).

    Args:
        trait_records: All trait grading records from grading.

    Returns:
        Dict mapping STRI column name to count of graded specimens with gt=1.
    """
    support: dict[str, int] = {rule.stri_column: 0 for rule in TRAIT_RULES}

    # Track which (specimen, column) pairs we've counted to avoid duplicates
    seen: set[tuple[str, str]] = set()
    for r in trait_records:
        if r.verdict.value not in ("correct", "incorrect"):
            continue
        key = (r.specimen_id, r.stri_column)
        if key in seen:
            continue
        seen.add(key)
        if r.ground_truth is not None and r.ground_truth == 1.0:
            support[r.stri_column] += 1

    return support


def grade_against_matrix(
    results_dir: Path,
    stri_path: Path,
    synonym_path: Path | None,
    skip_traits: bool,
    skip_id: bool,
    partition_map: dict[str, str] | None = None,
) -> tuple[list[TraitGradeRecord], list[IDGradeRecord], str, dict[str, int]]:
    """Grade all results against one STRI matrix.

    Args:
        results_dir: Directory of per-specimen JSON files.
        stri_path: Path to STRI trait matrix CSV.
        synonym_path: Path to synonym table.
        skip_traits: Whether to skip trait grading.
        skip_id: Whether to skip ID grading.
        partition_map: Optional mapping from specimen_id to partition label.

    Returns:
        Tuple of (trait_records, id_records, source_label, stri_support).
    """
    source_label = _detect_source_label(stri_path)
    logger.info("Grading against %s (%s)", source_label, stri_path)

    stri_df = pd.read_csv(stri_path, dtype={"taxon_id": str})
    column_suffix = _detect_column_suffix(stri_df)

    # Build name -> row index lookup
    stri_name_index: dict[str, int] = {}
    for idx, name in enumerate(stri_df["scientific_name"].values):
        if isinstance(name, str):
            stri_name_index[name.strip().lower()] = idx

    synonym_lookup = _build_synonym_lookup(synonym_path)

    trait_records: list[TraitGradeRecord] = []
    id_records: list[IDGradeRecord] = []

    json_files = sorted(results_dir.glob("*.json"))
    # Exclude metadata files
    json_files = [f for f in json_files if f.name != "run_metadata.json"]

    n_processed = 0
    n_no_stri = 0
    n_error_json = 0

    for json_path in json_files:
        with open(json_path) as f:
            result = json.load(f)

        # Check for pipeline error
        if "error" in result and "stages" not in result:
            n_error_json += 1
            continue

        specimen_id = result.get("specimen_id", json_path.stem)
        meta = result.get("benchmark_metadata", {})
        scientific_name = meta.get("scientific_name", "")
        family = meta.get("family", "")
        genus = meta.get("genus", "")
        epithet = meta.get("species_epithet", "")
        true_species = f"{genus} {epithet}" if genus else scientific_name

        if not scientific_name:
            logger.warning("No scientific_name in %s, skipping", json_path.name)
            continue

        # Find STRI row
        stri_row = _find_stri_row(
            scientific_name, stri_df, stri_name_index, synonym_lookup
        )

        # Grade traits
        if not skip_traits and stri_row is not None:
            stages = result.get("stages", {})
            morphology = stages.get("morphology", {})
            traits = morphology.get("data", {}).get("traits", {})

            records = grade_specimen_traits(
                specimen_id=specimen_id,
                scientific_name=scientific_name,
                traits=traits,
                stri_row=stri_row,
                column_suffix=column_suffix,
            )
            trait_records.extend(records)
        elif stri_row is None:
            n_no_stri += 1

        # Grade species ID
        if not skip_id:
            partition = (
                partition_map.get(specimen_id) if partition_map else None
            )
            id_record = grade_specimen_id(
                specimen_id=specimen_id,
                true_family=family,
                true_genus=genus,
                true_species=true_species,
                pipeline_result=result,
                partition=partition,
            )
            id_records.append(id_record)

        n_processed += 1

    logger.info(
        "Graded %d specimens (%d no STRI match, %d pipeline errors)",
        n_processed, n_no_stri, n_error_json,
    )

    stri_support = _compute_stri_support(trait_records)
    return trait_records, id_records, source_label, stri_support


def main() -> None:
    """Run the grading pipeline."""
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error("Results directory not found: %s", results_dir)
        return

    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "grades"

    synonym_path = Path(args.synonym_table) if args.synonym_table else None

    # Build partition map if split path provided
    partition_map: dict[str, str] | None = None
    if args.split_path:
        catalog_path = Path(args.catalog)
        split_path = Path(args.split_path)
        partition_map = build_specimen_partition_map(catalog_path, split_path)
        logger.info(
            "Built partition map: %d specimens (%d train, %d val, %d test)",
            len(partition_map),
            sum(1 for v in partition_map.values() if v == "train"),
            sum(1 for v in partition_map.values() if v == "val"),
            sum(1 for v in partition_map.values() if v == "test"),
        )

    for stri_path_str in args.stri_matrices:
        stri_path = Path(stri_path_str)
        if not stri_path.exists():
            logger.error("STRI matrix not found: %s", stri_path)
            continue

        trait_records, id_records, source_label, stri_support = grade_against_matrix(
            results_dir=results_dir,
            stri_path=stri_path,
            synonym_path=synonym_path,
            skip_traits=args.skip_trait_grading,
            skip_id=args.skip_id_grading,
            partition_map=partition_map,
        )

        save_benchmark_results(
            output_dir=output_dir,
            trait_records=trait_records,
            id_records=id_records,
            source_label=source_label,
            stri_support=stri_support,
        )

        print_summary(
            trait_records, id_records,
            source_label=source_label,
            stri_support=stri_support,
        )

        # Generate HTML diagnostic report
        if args.html:
            from seedlearn.benchmarking.html_report import generate_benchmark_html

            stri_df_check = pd.read_csv(stri_path, dtype={"taxon_id": str})
            col_suffix = _detect_column_suffix(stri_df_check)

            html = generate_benchmark_html(
                results_dir=results_dir,
                stri_path=stri_path,
                synonym_path=synonym_path,
                column_suffix=col_suffix,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            html_path = output_dir / f"{source_label}_diagnostic_report.html"
            html_path.write_text(html, encoding="utf-8")
            logger.info("HTML report saved to %s", html_path)
            print(f"  HTML report: {html_path}")


if __name__ == "__main__":
    main()
