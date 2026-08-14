"""Tests for multi-rank cache format (v2)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from seedlearn.data.catalog import ImageRecord


def test_build_rank_labels_produces_three_ranks(multirank_records):
    from seedlearn.clip.cache import _build_rank_labels

    rank_labels, rank_label_to_id = _build_rank_labels(multirank_records)

    assert set(rank_labels.keys()) == {"family", "genus", "species"}
    assert set(rank_label_to_id.keys()) == {"family", "genus", "species"}

    for rank in ("family", "genus", "species"):
        assert rank_labels[rank].shape == (30,)
        assert rank_labels[rank].dtype == np.int64
        assert len(np.unique(rank_labels[rank])) == 3


def test_build_rank_labels_consistent_mapping(multirank_records):
    from seedlearn.clip.cache import _build_rank_labels

    rank_labels, rank_label_to_id = _build_rank_labels(multirank_records)

    # First 10 records are Fabaceae/Inga/Inga vera
    assert rank_labels["family"][0] == rank_label_to_id["family"]["Fabaceae"]
    assert rank_labels["genus"][0] == rank_label_to_id["genus"]["Inga"]
    assert rank_labels["species"][0] == rank_label_to_id["species"]["Inga vera"]


def test_load_multirank_cache_roundtrip(tmp_path, multirank_features):
    from seedlearn.clip.cache import load_multirank_cache

    features, rank_labels, rank_label_to_id = multirank_features
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Manually write a v2 cache
    image_paths = np.array([f"/img_{i}.jpg" for i in range(30)])
    individual_ids = np.array(["IND"] * 30)

    arrays = {
        "features": features,
        "image_paths": image_paths,
        "individual_ids": individual_ids,
    }
    for rank_name, labels in rank_labels.items():
        arrays[f"{rank_name}_labels"] = labels

    np.savez_compressed(cache_dir / "features.npz", **arrays)

    meta = {
        "version": 2,
        "ranks": {},
    }
    for rank_name, label_to_id in rank_label_to_id.items():
        meta["ranks"][rank_name] = {
            "num_classes": len(label_to_id),
            "label_to_id": label_to_id,
            "id_to_label": {str(v): k for k, v in label_to_id.items()},
        }
    (cache_dir / "features_meta.json").write_text(json.dumps(meta))

    # Now load it
    loaded_features, loaded_rank_labels, loaded_meta, loaded_paths = load_multirank_cache(cache_dir)

    np.testing.assert_array_equal(loaded_features, features)
    assert set(loaded_rank_labels.keys()) == {"family", "genus", "species"}
    for rank in ("family", "genus", "species"):
        np.testing.assert_array_equal(loaded_rank_labels[rank], rank_labels[rank])
    assert loaded_meta["version"] == 2
    assert loaded_paths.shape == (30,)


def test_load_multirank_cache_missing_raises(tmp_path):
    from seedlearn.clip.cache import load_multirank_cache

    with pytest.raises(FileNotFoundError):
        load_multirank_cache(tmp_path / "nonexistent")


class TestExtractAndCacheMultirank:
    """Tests for CachedFeatureExtractor.extract_and_cache_multirank()."""

    @patch("seedlearn.clip.cache.FeatureExtractor")
    def test_creates_multirank_cache_files(self, MockExtractor, tmp_path, multirank_records):
        from seedlearn.clip.cache import CachedFeatureExtractor

        fake_features = np.random.randn(len(multirank_records), 768).astype(np.float32)
        mock_instance = MagicMock()
        mock_instance.extract_from_records_optimized.return_value = fake_features
        MockExtractor.return_value = mock_instance

        cache_dir = tmp_path / "cache"
        extractor = CachedFeatureExtractor(cache_dir=cache_dir, device="cpu")
        result = extractor.extract_and_cache_multirank(multirank_records)

        assert result.shape == (30, 768)
        assert (cache_dir / "features.npz").exists()
        assert (cache_dir / "features_meta.json").exists()

        # Verify NPZ contents
        data = np.load(cache_dir / "features.npz")
        assert "features" in data
        assert "family_labels" in data
        assert "genus_labels" in data
        assert "species_labels" in data
        assert "individual_ids" in data
        assert "image_paths" in data

    @patch("seedlearn.clip.cache.FeatureExtractor")
    def test_cache_hit_skips_extraction(self, MockExtractor, tmp_path, multirank_records):
        from seedlearn.clip.cache import CachedFeatureExtractor

        fake_features = np.random.randn(len(multirank_records), 768).astype(np.float32)
        mock_instance = MagicMock()
        mock_instance.extract_from_records_optimized.return_value = fake_features
        MockExtractor.return_value = mock_instance

        cache_dir = tmp_path / "cache"
        extractor = CachedFeatureExtractor(cache_dir=cache_dir, device="cpu")

        extractor.extract_and_cache_multirank(multirank_records)
        extractor.extract_and_cache_multirank(multirank_records)

        assert mock_instance.extract_from_records_optimized.call_count == 1

    @patch("seedlearn.clip.cache.FeatureExtractor")
    def test_metadata_has_all_ranks(self, MockExtractor, tmp_path, multirank_records):
        from seedlearn.clip.cache import CachedFeatureExtractor

        fake_features = np.random.randn(len(multirank_records), 768).astype(np.float32)
        mock_instance = MagicMock()
        mock_instance.extract_from_records_optimized.return_value = fake_features
        MockExtractor.return_value = mock_instance

        cache_dir = tmp_path / "cache"
        extractor = CachedFeatureExtractor(cache_dir=cache_dir, device="cpu")
        extractor.extract_and_cache_multirank(multirank_records)

        meta = json.loads((cache_dir / "features_meta.json").read_text())
        assert meta["version"] == 2
        assert set(meta["ranks"].keys()) == {"family", "genus", "species"}
        assert meta["ranks"]["family"]["num_classes"] == 3
        assert "Fabaceae" in meta["ranks"]["family"]["label_to_id"]

    @patch("seedlearn.clip.cache.FeatureExtractor")
    def test_metadata_has_taxonomy_map(self, MockExtractor, tmp_path, multirank_records):
        from seedlearn.clip.cache import CachedFeatureExtractor

        fake_features = np.random.randn(len(multirank_records), 768).astype(np.float32)
        mock_instance = MagicMock()
        mock_instance.extract_from_records_optimized.return_value = fake_features
        MockExtractor.return_value = mock_instance

        cache_dir = tmp_path / "cache"
        extractor = CachedFeatureExtractor(cache_dir=cache_dir, device="cpu")
        extractor.extract_and_cache_multirank(multirank_records)

        meta = json.loads((cache_dir / "features_meta.json").read_text())
        assert "taxonomy" in meta
        assert meta["taxonomy"]["genus_to_family"]["Inga"] == "Fabaceae"
        assert meta["taxonomy"]["species_to_genus"]["Inga vera"] == "Inga"
        assert meta["taxonomy"]["species_to_family"]["Inga vera"] == "Fabaceae"
