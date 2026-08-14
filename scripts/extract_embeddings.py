#!/usr/bin/env python3
"""Extract and cache BioClip2 features for all images.

This script extracts image embeddings using BioClip2 and caches them to disk
for efficient reuse across multiple experiments.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from seedlearn.data.catalog import load_dataset
from seedlearn.data.constants import DEFAULT_CATALOG, SHARED_EMBEDDINGS, get_catalog_version, get_optimal_batch_size
from seedlearn.clip.cache import CachedFeatureExtractor


def _resolve_path(path_input):
    """Handle both Path objects and strings from CLI."""
    if isinstance(path_input, Path):
        return path_input
    return Path(path_input).resolve()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="Path to species catalog CSV")
    parser.add_argument("--rank", choices=["family", "genus", "species"], default=None,
                        help="Taxonomic rank (omit for multi-rank v2 cache)")
    parser.add_argument("--cache-dir", type=_resolve_path, default=None, help="Directory to store cached features")
    parser.add_argument("--cache-name", type=str, default=None, help="Name for the cache file")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto-selected if not specified)")
    parser.add_argument("--model-str", default="hf-hub:imageomics/bioclip-2", help="BioCLIP model identifier (default: bioclip-2, 768-dim ViT-L/14)")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel data loading workers")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Batches to prefetch per worker")
    parser.add_argument("--no-optimize", action="store_true", help="Disable optimized DataLoader extraction")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputation even if cache exists")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    """Main function."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.cache_dir is None:
        version = get_catalog_version(Path(args.catalog))
        args.cache_dir = SHARED_EMBEDDINGS / version
        logging.info("Auto-generated embeddings directory: %s", args.cache_dir)

    device = torch.device(args.device)
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    logging.info("Using device: %s", device)

    if args.batch_size is None:
        args.batch_size = get_optimal_batch_size(device)
        logging.info("Auto-selected batch_size=%d based on device", args.batch_size)

    if device.type == "cuda":
        logging.info("CUDA Device: %s", torch.cuda.get_device_name(device))
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        logging.info("CUDA Device Memory: %.2f GB", total_memory)

    extractor = CachedFeatureExtractor(
        cache_dir=args.cache_dir,
        device=device,
        batch_size=args.batch_size,
        model_str=args.model_str,
    )

    use_optimized = not args.no_optimize

    start_time = time.time()
    logging.info("Loading dataset from catalog...")

    if args.rank is None:
        # Multi-rank v2 cache: extract once, store all taxonomy levels
        logging.info("Multi-rank extraction: storing family + genus + species labels")
        records, _ = load_dataset(catalog_path=Path(args.catalog), rank="family")
        logging.info("Loaded %d images in %.2f seconds", len(records), time.time() - start_time)

        cache_name = "features"
        extract_start = time.time()
        features = extractor.extract_and_cache_multirank(
            records=records,
            normalize=True,
            force_recompute=args.force_recompute,
            use_optimized=use_optimized,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
    else:
        # Legacy single-rank cache (backward compat)
        records, label_to_id = load_dataset(catalog_path=Path(args.catalog), rank=args.rank)
        logging.info("Loaded %d images across %d classes in %.2f seconds",
                     len(records), len(label_to_id), time.time() - start_time)

        cache_name = args.cache_name or f"{args.rank}_features"
        extract_start = time.time()
        features = extractor.extract_and_cache(
            records=records,
            cache_name=cache_name,
            normalize=True,
            force_recompute=args.force_recompute,
            use_optimized=use_optimized,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )

    logging.info(
        "Feature extraction complete: %d samples, %d dimensions, %.2f seconds",
        features.shape[0], features.shape[1], time.time() - extract_start,
    )
    logging.info("Cache: %s | Total runtime: %.2f seconds", cache_name, time.time() - start_time)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
