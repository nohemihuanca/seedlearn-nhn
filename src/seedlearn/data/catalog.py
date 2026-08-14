"""Data loading utilities for seedling image datasets.

This module provides functions to load seedling images from the catalog structure
used in the SeedLearn project, following the Single Responsibility Principle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image

from seedlearn.data.constants import IMAGE_EXTENSIONS, RANK_COLUMN_MAP


@dataclass
class ImageRecord:
    """Represents a single image with its taxonomic metadata.

    Attributes:
        image_path: Absolute path to the image file.
        label: The label for the specified taxonomic rank.
        family: Family name.
        genus: Genus name.
        species: Species binomial name.
        label_id: Integer ID for the label (assigned during dataset creation).
        individual_id: Unique identifier for the physical plant individual.
    """

    image_path: Path
    label: str
    family: str
    genus: str
    species: str
    label_id: int = -1  # Will be assigned during dataset creation
    individual_id: str = ""


def load_catalog(catalog_path: Path) -> pd.DataFrame:
    """Load the species catalog CSV file.

    Args:
        catalog_path: Path to the catalog CSV file.

    Returns:
        DataFrame containing the catalog data.

    Raises:
        FileNotFoundError: If the catalog file doesn't exist.
        ValueError: If required columns are missing.
    """
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    logging.info("Loading catalog from %s", catalog_path)
    df = pd.read_csv(catalog_path)

    required_columns = {
        "training_absolute_path",
        RANK_COLUMN_MAP["family"],
        RANK_COLUMN_MAP["genus"],
        RANK_COLUMN_MAP["species"],
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Catalog is missing required columns: {sorted(missing)}")

    return df


def iter_image_paths(directory: Path) -> Iterable[Path]:
    """Iterate over image files in a directory.

    Args:
        directory: Directory containing image files.

    Yields:
        Paths to image files with valid extensions.
    """
    if not directory.exists():
        logging.warning("Directory missing: %s", directory)
        return

    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def format_label(row: pd.Series, rank: str) -> str:
    """Format taxonomic label for a given rank.

    Args:
        row: DataFrame row containing taxonomic information.
        rank: Taxonomic rank ('family', 'genus', or 'species').

    Returns:
        Formatted label string.
    """
    family = str(row[RANK_COLUMN_MAP["family"]]).replace("_", " ").strip()
    genus = str(row[RANK_COLUMN_MAP["genus"]]).replace("_", " ").strip()
    species_epithet = str(row[RANK_COLUMN_MAP["species"]]).replace("_", " ").strip().lower()

    if rank == "family":
        return family
    elif rank == "genus":
        return genus
    else:  # species
        return f"{genus} {species_epithet}".strip()


def load_dataset(
    catalog_path: Path,
    rank: str = "species",
) -> tuple[list[ImageRecord], dict[str, int]]:
    """Load the complete dataset from the catalog.

    Args:
        catalog_path: Path to the catalog CSV file.
        rank: Taxonomic rank to use for labels ('family', 'genus', or 'species').

    Returns:
        Tuple containing:
            - List of ImageRecord objects
            - Dictionary mapping label strings to integer IDs

    Raises:
        ValueError: If rank is invalid.
        RuntimeError: If no images are found.
    """
    if rank not in RANK_COLUMN_MAP:
        raise ValueError(
            f"Invalid rank '{rank}'. Must be one of: {sorted(RANK_COLUMN_MAP.keys())}"
        )

    df = load_catalog(catalog_path)

    records: list[ImageRecord] = []
    label_to_id: dict[str, int] = {}
    missing_dirs = 0

    for _, row in df.iterrows():
        train_dir = Path(row["training_absolute_path"])
        if not train_dir.exists():
            missing_dirs += 1
            continue

        family = str(row[RANK_COLUMN_MAP["family"]]).replace("_", " ").strip()
        genus = str(row[RANK_COLUMN_MAP["genus"]]).replace("_", " ").strip()
        species_epithet = str(row[RANK_COLUMN_MAP["species"]]).replace("_", " ").strip().lower()
        species = f"{genus} {species_epithet}".strip()

        label = format_label(row, rank)

        # Assign integer ID to label
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id)

        label_id = label_to_id[label]
        individual_id = str(row.get("ID_YPS", "")) if "ID_YPS" in row.index else ""

        for image_path in iter_image_paths(train_dir):
            records.append(
                ImageRecord(
                    image_path=image_path,
                    label=label,
                    family=family,
                    genus=genus,
                    species=species,
                    label_id=label_id,
                    individual_id=individual_id,
                )
            )

    if missing_dirs:
        logging.warning(
            "%d training directories were missing; excluded from dataset",
            missing_dirs,
        )

    if not records:
        raise RuntimeError("No images found. Verify catalog paths and permissions.")

    logging.info(
        "Loaded %d images for rank '%s' across %d classes",
        len(records),
        rank,
        len(label_to_id),
    )

    return records, label_to_id


def load_image(image_path: Path) -> Image.Image:
    """Load an image from disk.

    Args:
        image_path: Path to the image file.

    Returns:
        PIL Image object.

    Raises:
        FileNotFoundError: If the image file doesn't exist.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return Image.open(image_path).convert("RGB")
