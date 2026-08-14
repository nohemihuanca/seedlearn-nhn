"""Shared constants for the seedlearn data pipeline.

This module centralises magic strings, default paths and helper functions
that are referenced by multiple modules across the package.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Taxonomic rank mapping
# ---------------------------------------------------------------------------

RANK_COLUMN_MAP: dict[str, str] = {
    "family": "FAMILY",
    "genus": "GENUS",
    "species": "SPECIES",
}

IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------------------
# Default filesystem paths
# ---------------------------------------------------------------------------

SHARED_DATA = Path("/nfs/roberts/project/pi_lsc4/shared/seedlearn/data")

SHARED_EMBEDDINGS = SHARED_DATA / "embeddings"
SHARED_SPLITS = SHARED_DATA / "splits"
SHARED_EXPERIMENTS = SHARED_DATA / "experiments" / "simpleshot"

DEFAULT_CATALOG = (
    "/nfs/roberts/project/pi_lsc4/shared/seedlearn/data/raw/2026-01-29/sorted_12K/metadata/"
    "species_catalog_v2026-01-29_12K_20260129_123334.csv"
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_catalog_version(catalog_path: Path) -> str:
    """Extract version string from a catalog filename.

    Args:
        catalog_path: Path to the catalog CSV file.

    Returns:
        Version-dated string, e.g. ``"2026-01-29_v2026-01-29_12K"``.

    Raises:
        ValueError: If catalog filename doesn't match expected pattern.

    Example:
        Input:  species_catalog_v2026-01-29_12K_20260129_123334.csv
        Output: "2026-01-29_v2026-01-29_12K"
    """
    filename = catalog_path.stem
    match = re.search(r"(v\d{4}-\d{2}-\d{2}_\d+K)", filename)
    if not match:
        raise ValueError(f"Cannot extract version from catalog: {filename}")
    version = match.group(1)  # v2026-01-29_12K
    date = version[1:11]      # 2026-01-29
    return f"{date}_{version}"


def get_optimal_batch_size(device: torch.device) -> int:
    """Determine optimal batch size based on GPU tier.

    Args:
        device: PyTorch device.

    Returns:
        Recommended batch size for the device.
    """
    if device.type != "cuda":
        return 64

    total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9

    if total_memory > 80:      # H200 (140 GB), A100 (80 GB)
        return 2048
    elif total_memory > 40:    # RTX A6000 (48 GB)
        return 1024
    elif total_memory > 30:    # RTX 5000 (32 GB)
        return 768
    elif total_memory > 16:    # Mid-range GPUs
        return 512
    else:                      # Small GPUs (<16 GB)
        return 256
