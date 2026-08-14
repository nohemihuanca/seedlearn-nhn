"""Shared test fixtures for seedlearn tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seedlearn.data.catalog import ImageRecord


@pytest.fixture
def synthetic_records(tmp_path: Path) -> list[ImageRecord]:
    """Create synthetic ImageRecord objects with real image files.

    Creates 30 records across 3 classes (10 each) with 1x1 JPEG files.
    """
    from PIL import Image

    records: list[ImageRecord] = []
    classes = [
        ("Fabaceae", "Acacia", "Acacia dealbata", 0),
        ("Rubiaceae", "Coffea", "Coffea arabica", 1),
        ("Solanaceae", "Solanum", "Solanum lycopersicum", 2),
    ]

    for family, genus, species, label_id in classes:
        class_dir = tmp_path / family
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(10):
            img_path = class_dir / f"img_{i:03d}.jpg"
            Image.new("RGB", (1, 1), color="red").save(img_path)

            if label_id == 0:
                label = family
            elif label_id == 1:
                label = genus
            else:
                label = species

            records.append(
                ImageRecord(
                    image_path=img_path,
                    label=label,
                    family=family,
                    genus=genus,
                    species=species,
                    label_id=label_id,
                )
            )

    return records


@pytest.fixture
def synthetic_features() -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic feature vectors for 30 samples across 3 classes.

    Returns:
        Tuple of (features, labels) where features is (30, 512) and labels is (30,).
    """
    rng = np.random.default_rng(42)
    n_per_class = 10
    n_classes = 3
    dim = 512

    features = []
    labels = []

    for class_id in range(n_classes):
        # Use orthogonal centroids for reliable separation
        centroid = np.zeros(dim, dtype=np.float32)
        centroid[class_id * 100:(class_id + 1) * 100] = 1.0
        centroid = centroid / np.linalg.norm(centroid)

        for _ in range(n_per_class):
            noise = rng.standard_normal(dim).astype(np.float32) * 0.05
            feat = centroid + noise
            feat = feat / np.linalg.norm(feat)
            features.append(feat)
            labels.append(class_id)

    return np.array(features), np.array(labels)


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def multirank_records(tmp_path: Path) -> list[ImageRecord]:
    """Create synthetic ImageRecords with distinct family/genus/species labels.

    Creates 30 records across 3 families, each with a distinct genus and species.
    All 3 taxonomy fields are always populated (matching real catalog behavior).
    """
    from PIL import Image

    records: list[ImageRecord] = []
    taxa = [
        ("Fabaceae", "Inga", "Inga vera", "IND_001"),
        ("Meliaceae", "Swietenia", "Swietenia macrophylla", "IND_002"),
        ("Sapindaceae", "Cupania", "Cupania americana", "IND_003"),
    ]

    for family, genus, species, ind_id in taxa:
        class_dir = tmp_path / "multirank" / family
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(10):
            img_path = class_dir / f"img_{i:03d}.jpg"
            Image.new("RGB", (1, 1), color="red").save(img_path)

            records.append(
                ImageRecord(
                    image_path=img_path,
                    label=family,
                    family=family,
                    genus=genus,
                    species=species,
                    label_id=-1,  # multi-rank doesn't use this
                    individual_id=ind_id,
                )
            )

    return records


@pytest.fixture
def multirank_features() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, dict[str, int]]]:
    """Create synthetic features with multi-rank label arrays.

    Returns:
        Tuple of (features[30,768], rank_labels, rank_label_to_id) where:
        - rank_labels: {"family": array[30], "genus": array[30], "species": array[30]}
        - rank_label_to_id: {"family": {"Fabaceae": 0, ...}, ...}
    """
    rng = np.random.default_rng(42)
    n_per_class = 10
    n_classes = 3
    dim = 768

    features = []
    for class_id in range(n_classes):
        centroid = np.zeros(dim, dtype=np.float32)
        centroid[class_id * 100 : (class_id + 1) * 100] = 1.0
        centroid = centroid / np.linalg.norm(centroid)

        for _ in range(n_per_class):
            noise = rng.standard_normal(dim).astype(np.float32) * 0.05
            feat = centroid + noise
            feat = feat / np.linalg.norm(feat)
            features.append(feat)

    features_arr = np.array(features)

    families = ["Fabaceae", "Meliaceae", "Sapindaceae"]
    genera = ["Inga", "Swietenia", "Cupania"]
    species_list = ["Inga vera", "Swietenia macrophylla", "Cupania americana"]

    rank_labels = {}
    rank_label_to_id = {}

    for rank_name, names in [("family", families), ("genus", genera), ("species", species_list)]:
        label_to_id = {name: i for i, name in enumerate(names)}
        labels = np.repeat(np.arange(n_classes), n_per_class).astype(np.int64)
        rank_labels[rank_name] = labels
        rank_label_to_id[rank_name] = label_to_id

    return features_arr, rank_labels, rank_label_to_id
