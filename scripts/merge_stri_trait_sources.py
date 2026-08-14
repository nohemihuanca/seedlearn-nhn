"""Merge per-key STRI trait matrices into a unified multi-source database.

Usage:
    python scripts/merge_stri_trait_sources.py \\
        --input-dir data/traits/stri_web_keys/per_key_trait_matrices \\
        --output-dir data/traits/stri_web_keys/merged

Output:
    merged/
    ├── stri_all_sources_merged_trait_matrix.csv     Source-tagged columns
    ├── stri_all_sources_consensus_trait_matrix.csv   Consensus (any-true)
    └── merge_report.json                            Statistics and coverage
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ID_COLS = ["taxon_id", "family", "scientific_name"]


def _load_source_matrices(
    input_dir: Path,
) -> list[tuple[str, pd.DataFrame]]:
    """Load all per-key trait matrix CSVs and tag with source slug.

    Args:
        input_dir: Directory containing *_trait_matrix.csv files.

    Returns:
        List of (slug, DataFrame) tuples sorted by slug.
    """
    csv_files = sorted(input_dir.glob("*_trait_matrix.csv"))
    source_dfs: list[tuple[str, pd.DataFrame]] = []
    for csv_path in csv_files:
        slug = csv_path.stem.replace("_trait_matrix", "")
        df = pd.read_csv(csv_path)
        source_dfs.append((slug, df))
        logger.info("  Loaded %s: %d species, %d columns", slug, len(df), len(df.columns))
    return source_dfs


def _collect_all_species(
    source_dfs: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Build deduplicated species index from all sources.

    Args:
        source_dfs: List of (slug, DataFrame) tuples.

    Returns:
        DataFrame with taxon_id, family, scientific_name — one row per
        unique taxon, sorted by family then name.
    """
    frames = [df[ID_COLS] for _, df in source_dfs]
    all_species = pd.concat(frames, ignore_index=True)
    all_species = (
        all_species
        .drop_duplicates(subset=["taxon_id"])
        .sort_values(["family", "scientific_name"])
        .reset_index(drop=True)
    )
    return all_species


def _build_consensus(
    merged: pd.DataFrame,
    trait_source_map: dict[str, list[str]],
    uncoded_source_map: dict[str, list[str]],
) -> pd.DataFrame:
    """Add consensus columns for each base trait.

    Consensus logic (per taxon × trait):
    - 1: ANY source reports present (1)
    - 0: At least one source has coded data for this trait and all say 0
    - NaN: No source has coded data (all NaN or all uncoded)

    Args:
        merged: Merged DataFrame with source-tagged columns.
        trait_source_map: Maps base trait name (e.g. "habit__tree") to its
            source-tagged column names (e.g. ["habit__tree__cl59", ...]).
        uncoded_source_map: Maps base category uncoded name (e.g.
            "habit__uncoded") to its source-tagged column names.

    Returns:
        merged DataFrame with consensus columns appended.
    """
    for base_trait, source_cols in sorted(trait_source_map.items()):
        consensus_col = f"{base_trait}__consensus"
        category = base_trait.split("__")[0]
        uncoded_base = f"{category}__uncoded"

        # For each source col, find its matching uncoded col (same source slug)
        any_present = (merged[source_cols] == 1).any(axis=1)

        # Determine which rows have at least one source with coded data
        has_coded_data = pd.Series(False, index=merged.index)
        for src_col in source_cols:
            # Extract source slug: "habit__tree__cl59" -> "cl59"
            source_slug = src_col.split("__", 2)[-1]
            uncoded_col = f"{uncoded_base}__{source_slug}"
            if uncoded_col in merged.columns:
                # Coded = present in source (not NaN) AND not uncoded
                coded = merged[src_col].notna() & (merged[uncoded_col] != 1)
                has_coded_data = has_coded_data | coded
            else:
                # No uncoded column for this source = all data is coded
                coded = merged[src_col].notna()
                has_coded_data = has_coded_data | coded

        # Build consensus: 1 if any present, 0 if coded but none present, NaN otherwise
        merged[consensus_col] = pd.Series(dtype="float64", index=merged.index)
        merged.loc[any_present, consensus_col] = 1.0
        merged.loc[~any_present & has_coded_data, consensus_col] = 0.0

    return merged


def merge_trait_matrices(input_dir: Path, output_dir: Path) -> None:
    """Merge all per-key trait matrix CSVs into unified outputs.

    Args:
        input_dir: Directory containing *_trait_matrix.csv files.
        output_dir: Directory for merged outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dfs = _load_source_matrices(input_dir)
    if not source_dfs:
        logger.error("No trait matrix CSVs found in %s", input_dir)
        return

    logger.info("Found %d trait matrices to merge", len(source_dfs))

    all_species = _collect_all_species(source_dfs)
    logger.info("Total unique species across all sources: %d", len(all_species))

    # Build merged DataFrame with source-tagged trait columns
    merged = all_species.copy()
    trait_source_map: dict[str, list[str]] = defaultdict(list)
    uncoded_source_map: dict[str, list[str]] = defaultdict(list)
    total_source_trait_cols = 0

    for slug, df in source_dfs:
        trait_cols = [c for c in df.columns if "__" in c]
        total_source_trait_cols += len(trait_cols)
        source_data = df[["taxon_id"] + trait_cols].copy()

        for col in trait_cols:
            source_col = f"{col}__{slug}"
            source_data = source_data.rename(columns={col: source_col})

            if col.endswith("__uncoded"):
                uncoded_source_map[col].append(source_col)
            else:
                trait_source_map[col].append(source_col)

        merged = merged.merge(source_data, on="taxon_id", how="left")

    # Add consensus columns (only for trait columns, not uncoded indicators)
    merged = _build_consensus(merged, trait_source_map, uncoded_source_map)

    # Save merged matrix (all source columns + consensus)
    merged_path = output_dir / "stri_all_sources_merged_trait_matrix.csv"
    merged.to_csv(merged_path, index=False)

    # Save consensus-only matrix
    consensus_cols = ID_COLS + [
        c for c in merged.columns if c.endswith("__consensus")
    ]
    consensus_df = merged[consensus_cols].copy()
    consensus_df.columns = [
        c.replace("__consensus", "") if c.endswith("__consensus") else c
        for c in consensus_df.columns
    ]
    consensus_path = output_dir / "stri_all_sources_consensus_trait_matrix.csv"
    consensus_df.to_csv(consensus_path, index=False)

    # Generate merge report
    report = {
        "sources": [
            {
                "slug": slug,
                "species_count": len(df),
                "trait_columns": len([c for c in df.columns if "__" in c]),
            }
            for slug, df in source_dfs
        ],
        "total_unique_species": len(all_species),
        "total_source_trait_columns": total_source_trait_cols,
        "total_consensus_traits": len(trait_source_map),
        "merged_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = output_dir / "merge_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    logger.info(
        "Merged: %s (%d species x %d columns)",
        merged_path.name, len(merged), len(merged.columns),
    )
    logger.info(
        "Consensus: %s (%d species x %d traits)",
        consensus_path.name, len(consensus_df), len(consensus_df.columns) - 3,
    )
    logger.info("Report: %s", report_path.name)


def main() -> None:
    """CLI entry point for merging per-key trait matrices."""
    parser = argparse.ArgumentParser(
        description="Merge per-key STRI trait matrices into a unified database.",
    )
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("data/traits/stri_web_keys/per_key_trait_matrices"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/traits/stri_web_keys/merged"),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    merge_trait_matrices(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
