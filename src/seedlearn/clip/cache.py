"""Feature extractor with disk caching.

This module provides a caching layer on top of ``FeatureExtractor`` to avoid
re-computation across multiple experiments.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch

from seedlearn.data.catalog import ImageRecord
from seedlearn.clip.encoder import FeatureExtractor


def _build_rank_labels(
    records: Sequence[ImageRecord],
) -> tuple[dict[str, npt.NDArray[np.int64]], dict[str, dict[str, int]]]:
    """Build per-rank label arrays and mappings from image records.

    Args:
        records: List of ImageRecord objects with family/genus/species populated.

    Returns:
        Tuple of (rank_labels, rank_label_to_id) where:
        - rank_labels: {"family": array[N], "genus": array[N], "species": array[N]}
        - rank_label_to_id: {"family": {"Fabaceae": 0, ...}, ...}
    """
    rank_labels: dict[str, npt.NDArray[np.int64]] = {}
    rank_label_to_id: dict[str, dict[str, int]] = {}

    for rank in ("family", "genus", "species"):
        label_to_id: dict[str, int] = {}
        labels: list[int] = []
        for r in records:
            name = getattr(r, rank)
            if name not in label_to_id:
                label_to_id[name] = len(label_to_id)
            labels.append(label_to_id[name])
        rank_labels[rank] = np.array(labels, dtype=np.int64)
        rank_label_to_id[rank] = label_to_id

    return rank_labels, rank_label_to_id


def load_multirank_cache(
    cache_dir: Path | str,
) -> tuple[
    npt.NDArray[np.float32],
    dict[str, npt.NDArray[np.int64]],
    dict,
    npt.NDArray[np.str_],
]:
    """Load a multi-rank feature cache from disk (no model loading required).

    Args:
        cache_dir: Directory containing ``features.npz`` and ``features_meta.json``.

    Returns:
        Tuple of (features, rank_labels, metadata, image_paths).

    Raises:
        FileNotFoundError: If cache files don't exist.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / "features.npz"
    meta_path = cache_dir / "features_meta.json"

    if not cache_path.exists():
        raise FileNotFoundError(f"Multi-rank cache not found: {cache_path}")

    data = np.load(cache_path, allow_pickle=False)
    features = data["features"]
    image_paths = data["image_paths"]

    rank_labels: dict[str, npt.NDArray[np.int64]] = {}
    for rank in ("family", "genus", "species"):
        key = f"{rank}_labels"
        if key in data:
            rank_labels[rank] = data[key]

    if not meta_path.exists():
        raise FileNotFoundError(f"Cache metadata not found: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    return features, rank_labels, meta, image_paths


class CachedFeatureExtractor:
    """Feature extractor with disk caching.

    This class caches extracted features to disk to avoid re-computation
    across multiple experiments.
    """

    def __init__(
        self,
        cache_dir: Path,
        device: torch.device | str = "cuda",
        batch_size: int = 256,
        model_str: str = "hf-hub:imageomics/bioclip-2",
    ) -> None:
        """Initialize the cached feature extractor.

        Args:
            cache_dir: Directory to store cached features.
            device: Torch device to use for computation.
            batch_size: Batch size for feature extraction.
            model_str: BioClip model identifier.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = FeatureExtractor(device, batch_size, model_str)
        logging.info("Initialized CachedFeatureExtractor, cache_dir: %s", self.cache_dir)

    def _get_cache_path(self, cache_name: str) -> Path:
        """Get the cache file path for a given cache name.

        Args:
            cache_name: Name for the cache file.

        Returns:
            Path to the cache file.
        """
        return self.cache_dir / f"{cache_name}.npz"

    def extract_and_cache(
        self,
        records: Sequence[ImageRecord],
        cache_name: str,
        normalize: bool = True,
        force_recompute: bool = False,
        use_optimized: bool = True,
        num_workers: int = 8,
        prefetch_factor: int = 2,
    ) -> npt.NDArray[np.float32]:
        """Extract features and cache them to disk.

        Args:
            records: List of ImageRecord objects.
            cache_name: Name for the cache file.
            normalize: Whether to L2-normalize features.
            force_recompute: If True, recompute even if cache exists.
            use_optimized: Whether to use optimized DataLoader extraction.
            num_workers: Number of parallel data loading workers (if optimized).
            prefetch_factor: Number of batches to prefetch per worker (if optimized).

        Returns:
            Feature array of shape (num_records, feature_dim).
        """
        cache_path = self._get_cache_path(cache_name)

        if cache_path.exists() and not force_recompute:
            logging.info("Loading cached features from %s", cache_path)
            data = np.load(cache_path)
            features = data["features"]
            labels = data["labels"]

            expected_labels = np.array([r.label_id for r in records])
            if not np.array_equal(labels, expected_labels):
                logging.warning(
                    "Cached labels don't match current records, recomputing..."
                )
            else:
                return features

        logging.info("Extracting features for %d images...", len(records))
        if use_optimized:
            logging.info("Using optimized extraction with %d workers", num_workers)
            features = self.extractor.extract_from_records_optimized(
                records,
                normalize=normalize,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )
        else:
            features = self.extractor.extract_from_records(records, normalize=normalize)

        labels = np.array([r.label_id for r in records])
        image_paths = np.array([str(r.image_path) for r in records])

        np.savez_compressed(
            cache_path,
            features=features,
            labels=labels,
            image_paths=image_paths,
        )

        logging.info("Cached features to %s", cache_path)
        return features

    def extract_and_cache_multirank(
        self,
        records: Sequence[ImageRecord],
        normalize: bool = True,
        force_recompute: bool = False,
        use_optimized: bool = True,
        num_workers: int = 8,
        prefetch_factor: int = 2,
    ) -> npt.NDArray[np.float32]:
        """Extract features once and cache with all taxonomy levels.

        Produces a v2 multi-rank cache: ``features.npz`` + ``features_meta.json``
        containing feature vectors alongside family, genus, and species label
        arrays. This eliminates the need for per-rank extraction.

        Args:
            records: ImageRecord objects with family/genus/species populated.
            normalize: Whether to L2-normalize features.
            force_recompute: If True, recompute even if cache exists.
            use_optimized: Whether to use DataLoader extraction.
            num_workers: Parallel workers for DataLoader.
            prefetch_factor: Batches to prefetch per worker.

        Returns:
            Feature array of shape (num_records, feature_dim).
        """
        cache_path = self.cache_dir / "features.npz"
        meta_path = self.cache_dir / "features_meta.json"

        if cache_path.exists() and not force_recompute:
            logging.info("Loading cached multi-rank features from %s", cache_path)
            return np.load(cache_path, allow_pickle=False)["features"]

        logging.info("Extracting features for %d images (multi-rank)...", len(records))
        if use_optimized:
            features = self.extractor.extract_from_records_optimized(
                records,
                normalize=normalize,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )
        else:
            features = self.extractor.extract_from_records(records, normalize=normalize)

        rank_labels, rank_label_to_id = _build_rank_labels(records)

        arrays: dict[str, np.ndarray] = {
            "features": features,
            "image_paths": np.array([str(r.image_path) for r in records]),
            "individual_ids": np.array([r.individual_id for r in records]),
        }
        for rank_name, labels in rank_labels.items():
            arrays[f"{rank_name}_labels"] = labels

        np.savez_compressed(cache_path, **arrays)

        # Build taxonomy cross-reference map
        genus_to_family: dict[str, str] = {}
        species_to_genus: dict[str, str] = {}
        species_to_family: dict[str, str] = {}
        for r in records:
            genus_to_family[r.genus] = r.family
            species_to_genus[r.species] = r.genus
            species_to_family[r.species] = r.family

        meta = {
            "version": 2,
            "embedding_dim": int(features.shape[1]),
            "num_images": len(records),
            "ranks": {},
            "taxonomy": {
                "genus_to_family": genus_to_family,
                "species_to_genus": species_to_genus,
                "species_to_family": species_to_family,
            },
        }
        for rank_name, label_to_id in rank_label_to_id.items():
            meta["ranks"][rank_name] = {
                "num_classes": len(label_to_id),
                "label_to_id": label_to_id,
                "id_to_label": {str(v): k for k, v in label_to_id.items()},
            }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logging.info("Cached multi-rank features to %s", cache_path)
        return features

    def load_cached_features(
        self, cache_name: str
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64], npt.NDArray[np.str_]]:
        """Load cached features from disk.

        Args:
            cache_name: Name of the cache file.

        Returns:
            Tuple of (features, labels, image_paths).

        Raises:
            FileNotFoundError: If the cache file doesn't exist.
        """
        cache_path = self._get_cache_path(cache_name)

        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_path}")

        data = np.load(cache_path)
        return data["features"], data["labels"], data["image_paths"]
