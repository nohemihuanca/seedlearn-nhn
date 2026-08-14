"""Scrape morphological traits from STRI Panama Biota identification keys.

Usage:
    # Scrape all 10 identification keys (default output: data/traits/stri_web_keys/)
    python scripts/scrape_stri_traits.py --keys all

    # Scrape specific keys
    python scripts/scrape_stri_traits.py --keys 59 178 185

    # Force re-fetch (ignore cached HTML)
    python scripts/scrape_stri_traits.py --keys 59 --force-refresh

Output structure:
    {output_dir}/
    ├── raw_html/cl{id}_{slug}/         Cached HTML responses
    └── per_key_trait_matrices/          Per-key CSV + metadata JSON
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from seedlearn.scraper.client import STRIClient
from seedlearn.scraper.matrix import build_trait_matrix, save_trait_matrix
from seedlearn.scraper.parser import (
    parse_filter_schema,
    parse_species_count,
    parse_species_list,
)
from seedlearn.scraper.schema import (
    IdentificationKey,
    STRI_IDENTIFICATION_KEYS,
)

logger = logging.getLogger(__name__)


def _sanitize_filename(name: str) -> str:
    """Replace filesystem-unsafe characters in a cache filename."""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def scrape_single_key(
    key: IdentificationKey,
    output_dir: Path,
    delay_seconds: float = 1.0,
    force_refresh: bool = False,
) -> Path:
    """Scrape all traits for a single identification key.

    Args:
        key: The identification key to scrape.
        output_dir: Base output directory.
        delay_seconds: Minimum seconds between HTTP requests.
        force_refresh: Re-fetch even if cached.

    Returns:
        Path to the saved trait matrix CSV.
    """
    html_dir = output_dir / "raw_html" / key.directory_name
    client = STRIClient(
        cache_dir=html_dir,
        delay_seconds=delay_seconds,
    )

    # Step 1: Fetch unfiltered page (all species)
    logger.info("Scraping key: %s (%s)", key.directory_name, key.name)
    unfiltered_html = client.fetch_cached(
        key.base_url,
        "unfiltered_all_species.html",
        force_refresh=force_refresh,
    )

    species = parse_species_list(unfiltered_html)
    reported_count = parse_species_count(unfiltered_html)
    categories = parse_filter_schema(unfiltered_html)

    logger.info(
        "  Found %d species (reported: %d), %d filter categories",
        len(species), reported_count, len(categories),
    )

    if not categories:
        logger.warning("  No filter categories found — saving species-only matrix")
        df = build_trait_matrix(species, {}, [])
        matrix_dir = output_dir / "per_key_trait_matrices"
        csv_path, _ = save_trait_matrix(
            df, matrix_dir, key.directory_name,
            species_count=reported_count, categories=[],
        )
        return csv_path

    # Step 2: Fetch each filter option
    filter_results: dict[str, set[int]] = {}
    total_options = sum(len(c.options) for c in categories)
    completed = 0

    for cat in categories:
        for opt in cat.options:
            completed += 1
            cache_name = _sanitize_filename(
                f"filtered_attr_{opt.attr_value}_{cat.name}_{opt.label}.html"
            )
            url = key.filtered_url([opt])
            filtered_html = client.fetch_cached(
                url, cache_name, force_refresh=force_refresh,
            )
            matched_species = parse_species_list(filtered_html)
            matched_ids = {s.taxon_id for s in matched_species}
            filter_results[opt.attr_value] = matched_ids
            logger.info(
                "  [%d/%d] %s = %s: %d species",
                completed, total_options, cat.name, opt.label, len(matched_ids),
            )

    # Step 3: Build and save trait matrix
    df = build_trait_matrix(species, filter_results, categories)
    matrix_dir = output_dir / "per_key_trait_matrices"
    csv_path, _ = save_trait_matrix(
        df, matrix_dir, key.directory_name,
        species_count=reported_count, categories=categories,
    )

    trait_count = len([c for c in df.columns if "__" in c])
    logger.info(
        "  Saved: %s (%d species x %d trait columns)",
        csv_path.name, len(df), trait_count,
    )
    return csv_path


def main() -> None:
    """CLI entry point for STRI trait scraping."""
    parser = argparse.ArgumentParser(
        description="Scrape morphological traits from STRI identification keys.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--keys", nargs="+", default=["all"],
        help="Checklist IDs to scrape (e.g., 59 178) or 'all' (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/traits/stri_web_keys"),
        help="Output directory (default: data/traits/stri_web_keys/)",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds between HTTP requests (default: 1.0)",
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Re-fetch HTML even if cached locally",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve which keys to scrape
    if "all" in args.keys:
        keys = STRI_IDENTIFICATION_KEYS
    else:
        cl_ids = {int(k) for k in args.keys}
        keys = [k for k in STRI_IDENTIFICATION_KEYS if k.cl_id in cl_ids]
        unknown = cl_ids - {k.cl_id for k in keys}
        if unknown:
            logger.error("Unknown key IDs: %s", unknown)
            sys.exit(1)

    results: list[Path] = []
    for key in keys:
        csv_path = scrape_single_key(
            key, args.output_dir,
            delay_seconds=args.delay,
            force_refresh=args.force_refresh,
        )
        results.append(csv_path)

    logger.info("Complete. Scraped %d keys.", len(results))
    for p in results:
        logger.info("  %s", p)


if __name__ == "__main__":
    main()
