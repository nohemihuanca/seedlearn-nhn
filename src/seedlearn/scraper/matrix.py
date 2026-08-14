"""Trait matrix construction from STRI identification key scrape results.

Builds species x trait presence/absence matrices from filter-inverted scraping.
Each trait option becomes a 0/1 column (multi-label: a species can be 1 for
multiple options within the same category). Uncoded detection flags species
with all-zero across an entire category as "data not entered" rather than
"truly absent from all options."
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from seedlearn.scraper.schema import FilterCategory, SpeciesEntry

logger = logging.getLogger(__name__)


def build_trait_matrix(
    species: list[SpeciesEntry],
    filter_results: dict[str, set[int]],
    categories: list[FilterCategory],
) -> pd.DataFrame:
    """Build a species x trait presence/absence matrix with uncoded detection.

    Args:
        species: All species from the unfiltered key page.
        filter_results: Maps attr_value (e.g. "1-1") to set of taxon_ids
            that appeared when that filter was applied.
        categories: Filter categories with their options (for column naming).

    Returns:
        DataFrame with columns: taxon_id, family, scientific_name, then one
        column per trait option (0/1 int), then one {category}__uncoded
        indicator column per category.
    """
    # Build attr_value -> (category_name, column_name) mapping
    col_map: dict[str, tuple[str, str]] = {}
    for cat in categories:
        for opt in cat.options:
            col_map[opt.attr_value] = (cat.name, opt.column_name(cat.name))

    trait_columns = [col for _, col in col_map.values()]
    uncoded_columns = [f"{cat.name}__uncoded" for cat in categories]

    rows: list[dict[str, int | str]] = []
    for sp in species:
        row: dict[str, int | str] = {
            "taxon_id": sp.taxon_id,
            "family": sp.family,
            "scientific_name": sp.scientific_name,
        }

        # Fill trait values, track per-category sums for uncoded detection
        category_sums: dict[str, int] = {cat.name: 0 for cat in categories}
        for attr_val, (cat_name, col_name) in col_map.items():
            taxon_ids = filter_results.get(attr_val, set())
            val = 1 if sp.taxon_id in taxon_ids else 0
            row[col_name] = val
            category_sums[cat_name] += val

        # Uncoded = all options in category are 0 (data not entered)
        for cat in categories:
            row[f"{cat.name}__uncoded"] = (
                1 if category_sums[cat.name] == 0 else 0
            )

        rows.append(row)

    column_order = (
        ["taxon_id", "family", "scientific_name"]
        + trait_columns
        + uncoded_columns
    )
    df = pd.DataFrame(rows, columns=column_order)
    df["taxon_id"] = df["taxon_id"].astype(int)
    return df


def save_trait_matrix(
    df: pd.DataFrame,
    output_dir: Path,
    key_slug: str,
    species_count: int,
    categories: list[FilterCategory],
) -> tuple[Path, Path]:
    """Save trait matrix CSV and scrape metadata JSON.

    Args:
        df: Trait matrix DataFrame.
        output_dir: Directory to write files.
        key_slug: Key identifier for filename (e.g. "cl59_panama_dicots").
        species_count: Reported species count from the key page.
        categories: Filter categories used.

    Returns:
        Tuple of (csv_path, metadata_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{key_slug}_trait_matrix.csv"
    meta_path = output_dir / f"{key_slug}_scrape_metadata.json"

    df.to_csv(csv_path, index=False)

    trait_cols = [c for c in df.columns if "__" in c]
    metadata = {
        "key_slug": key_slug,
        "species_count_reported": species_count,
        "species_count_scraped": len(df),
        "trait_columns": trait_cols,
        "categories": [
            {
                "id": cat.category_id,
                "name": cat.name,
                "options": [
                    {"attr": o.attr_value, "label": o.label}
                    for o in cat.options
                ],
            }
            for cat in categories
        ],
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
    )

    logger.info(
        "Saved %s: %d species, %d trait columns",
        csv_path.name, len(df), len(trait_cols),
    )
    return csv_path, meta_path


def load_trait_matrix(csv_path: Path) -> pd.DataFrame:
    """Load a previously saved trait matrix CSV.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with original dtypes.
    """
    return pd.read_csv(csv_path)
