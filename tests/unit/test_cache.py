"""Tests for feature caching (mocked extraction)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from seedlearn.data.catalog import ImageRecord


def _make_records(tmp_path: Path, n: int = 5) -> list[ImageRecord]:
    """Create minimal ImageRecords with dummy paths."""
    from PIL import Image

    records = []
    img_dir = tmp_path / "images"
    img_dir.mkdir(exist_ok=True)

    for i in range(n):
        p = img_dir / f"img_{i}.jpg"
        Image.new("RGB", (1, 1)).save(p)
        records.append(
            ImageRecord(
                image_path=p,
                label="test",
                family="F",
                genus="G",
                species="S",
                label_id=0,
            )
        )
    return records


class TestCachedFeatureExtractor:
    """Tests for CachedFeatureExtractor cache hit/miss behaviour."""

    @patch("seedlearn.clip.cache.FeatureExtractor")
    def test_cache_miss_then_hit(self, MockExtractor, tmp_path):
        from seedlearn.clip.cache import CachedFeatureExtractor

        cache_dir = tmp_path / "cache"
        records = _make_records(tmp_path)

        fake_features = np.random.randn(len(records), 512).astype(np.float32)

        mock_instance = MagicMock()
        mock_instance.extract_from_records.return_value = fake_features
        mock_instance.extract_from_records_optimized.return_value = fake_features
        MockExtractor.return_value = mock_instance

        extractor = CachedFeatureExtractor(cache_dir=cache_dir, device="cpu")

        # First call: cache miss -> extraction
        result1 = extractor.extract_and_cache(records, "test_cache", use_optimized=False)
        assert result1.shape == (len(records), 512)
        assert mock_instance.extract_from_records.call_count == 1

        # Second call: cache hit -> no extraction
        result2 = extractor.extract_and_cache(records, "test_cache", use_optimized=False)
        np.testing.assert_array_equal(result1, result2)
        assert mock_instance.extract_from_records.call_count == 1  # not called again

    @patch("seedlearn.clip.cache.FeatureExtractor")
    def test_force_recompute(self, MockExtractor, tmp_path):
        from seedlearn.clip.cache import CachedFeatureExtractor

        cache_dir = tmp_path / "cache"
        records = _make_records(tmp_path)
        fake_features = np.random.randn(len(records), 512).astype(np.float32)

        mock_instance = MagicMock()
        mock_instance.extract_from_records.return_value = fake_features
        mock_instance.extract_from_records_optimized.return_value = fake_features
        MockExtractor.return_value = mock_instance

        extractor = CachedFeatureExtractor(cache_dir=cache_dir, device="cpu")
        extractor.extract_and_cache(records, "test", use_optimized=False)
        extractor.extract_and_cache(records, "test", force_recompute=True, use_optimized=False)

        assert mock_instance.extract_from_records.call_count == 2

    @patch("seedlearn.clip.cache.FeatureExtractor")
    def test_load_cached_features(self, MockExtractor, tmp_path):
        from seedlearn.clip.cache import CachedFeatureExtractor

        cache_dir = tmp_path / "cache"
        records = _make_records(tmp_path)
        fake_features = np.random.randn(len(records), 512).astype(np.float32)

        mock_instance = MagicMock()
        mock_instance.extract_from_records.return_value = fake_features
        MockExtractor.return_value = mock_instance

        extractor = CachedFeatureExtractor(cache_dir=cache_dir, device="cpu")
        extractor.extract_and_cache(records, "cached_test", use_optimized=False)

        features, labels, paths = extractor.load_cached_features("cached_test")
        assert features.shape == (len(records), 512)
        assert labels.shape == (len(records),)
        assert paths.shape == (len(records),)

    @patch("seedlearn.clip.cache.FeatureExtractor")
    def test_load_missing_cache_raises(self, MockExtractor, tmp_path):
        from seedlearn.clip.cache import CachedFeatureExtractor

        mock_instance = MagicMock()
        MockExtractor.return_value = mock_instance

        extractor = CachedFeatureExtractor(cache_dir=tmp_path / "empty_cache", device="cpu")
        with pytest.raises(FileNotFoundError):
            extractor.load_cached_features("nonexistent")


def test_default_model_is_bioclip2():
    """CachedFeatureExtractor default must match FeatureExtractor (bioclip-2)."""
    import inspect
    from seedlearn.clip.cache import CachedFeatureExtractor
    sig = inspect.signature(CachedFeatureExtractor.__init__)
    default = sig.parameters["model_str"].default
    assert default == "hf-hub:imageomics/bioclip-2", (
        f"CachedFeatureExtractor default is '{default}', "
        f"expected 'hf-hub:imageomics/bioclip-2'"
    )
