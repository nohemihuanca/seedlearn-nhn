"""Convert inat_metadata CSV (photo-level, 19 columns) to legacy sort format.

Produces an individual-level CSV with the columns sort_project.py expects:
    ID_YPS, SPP, GENUS, SPECIES, FAMILY, LIANA, FOREST

Usage:
    python convert_inat_metadata.py \
        --input inat_metadata_FINAL_NHN_01_2025.csv \
        --output YPS_seedling_spp_list_derived_01_29_26.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# Column mapping: new format → legacy format
COLUMN_MAP = {
    "notes_code_clean2": "ID_YPS",
    "spp6_fixed": "SPP",
    "genus": "GENUS",
    "species": "SPECIES",
    "accepted_family": "FAMILY",
}

HABIT_TO_LIANA = {"Climbing": 1, "Freestanding": 0}


def convert(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Convert photo-level inat metadata to individual-level legacy format.

    Args:
        input_path: Path to inat_metadata CSV.
        output_path: Path for the derived legacy-format CSV.

    Returns:
        The derived DataFrame.
    """
    df = pd.read_csv(input_path)
    print(f"Read {len(df)} photo-level rows from {input_path.name}")

    # --- Validate expected columns ---
    required = list(COLUMN_MAP.keys()) + ["habit", "accepted_name", "scientific_name_final2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # --- Drop rows with null taxonomy ---
    null_mask = df[["accepted_family", "genus", "species", "notes_code_clean2"]].isna().any(axis=1)
    n_null = null_mask.sum()
    if n_null:
        print(f"Dropping {n_null} rows with null taxonomy")
        df = df[~null_mask].copy()

    # --- Deduplicate to individual level ---
    # Group by individual ID, take first row (taxonomy is constant per individual)
    # First verify taxonomy consistency within individuals
    taxonomy_cols = ["accepted_family", "genus", "species", "accepted_name"]
    inconsistent = (
        df.groupby("notes_code_clean2")[taxonomy_cols]
        .nunique()
        .pipe(lambda x: x[x.gt(1).any(axis=1)])
    )
    if len(inconsistent):
        print(f"WARNING: {len(inconsistent)} individuals have inconsistent taxonomy:")
        for ind_id in inconsistent.index[:10]:
            subset = df[df["notes_code_clean2"] == ind_id][["notes_code_clean2", "accepted_name"]].drop_duplicates()
            print(f"  {ind_id}: {subset['accepted_name'].tolist()}")
        if len(inconsistent) > 10:
            print(f"  ... and {len(inconsistent) - 10} more")

    individuals = df.drop_duplicates(subset="notes_code_clean2", keep="first").copy()
    print(f"Deduplicated to {len(individuals)} unique individuals")

    # --- Build legacy columns ---
    legacy = pd.DataFrame()
    for new_col, old_col in COLUMN_MAP.items():
        legacy[old_col] = individuals[new_col].values

    legacy["LIANA"] = individuals["habit"].map(HABIT_TO_LIANA).fillna(0).astype(int).values
    legacy["FOREST"] = "Unknown"  # not present in new format

    # --- Summary ---
    n_synonym = (individuals["scientific_name_final2"] != individuals["accepted_name"]).sum()
    print(f"\nDerived CSV summary:")
    print(f"  Individuals:  {len(legacy)}")
    print(f"  Families:     {legacy['FAMILY'].nunique()}")
    print(f"  Genera:       {legacy['GENUS'].nunique()}")
    print(f"  Species:      {legacy['SPECIES'].nunique()}")
    print(f"  Synonym cases (field ≠ accepted): {n_synonym}")
    print(f"  Habit breakdown: Freestanding={( legacy['LIANA'] == 0).sum()}, Climbing={(legacy['LIANA'] == 1).sum()}")

    legacy.to_csv(output_path, index=False)
    print(f"\nWrote {output_path}")

    return legacy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Path to inat_metadata CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output path for legacy-format CSV")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    convert(args.input, args.output)


if __name__ == "__main__":
    main()
