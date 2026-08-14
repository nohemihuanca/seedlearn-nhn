"""Tests for Stage 2: Visual Embedding Classification."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from seedlearn.pipeline.config import ClassifierConfig
from seedlearn.pipeline.protocol import StageResult
from seedlearn.pipeline.stages.classification import ClassificationStage


def _make_mock_extractor(return_features: np.ndarray) -> MagicMock:
    """Create a mock FeatureExtractor returning fixed features."""
    mock = MagicMock()
    mock.extract_from_paths.return_value = return_features
    return mock


class TestClassificationStage:
    """Tests for ClassificationStage lifecycle and protocol compliance."""

    def test_name(self):
        stage = ClassificationStage(config=ClassifierConfig())
        assert stage.name == "classification"

    def test_validate_input_requires_image_paths(self):
        stage = ClassificationStage(config=ClassifierConfig())
        errors = stage.validate_input({})
        assert any("image_paths" in e for e in errors)

    def test_validate_input_rejects_empty_list(self):
        stage = ClassificationStage(config=ClassifierConfig())
        errors = stage.validate_input({"image_paths": []})
        assert len(errors) > 0

    def test_validate_input_ok(self):
        stage = ClassificationStage(config=ClassifierConfig())
        errors = stage.validate_input({"image_paths": ["/test.jpg"]})
        assert errors == []

    def test_skip_returns_empty(self):
        stage = ClassificationStage(config=ClassifierConfig())
        result = stage.skip({})
        assert result.skipped is True
        assert result.data == {}

    def test_run_not_fitted_returns_error(self):
        stage = ClassificationStage(config=ClassifierConfig())
        stage._extractor = MagicMock()
        result = stage.run({"image_paths": ["/test.jpg"]})
        assert result.error is not None
        assert "not fitted" in result.error.lower()

    def test_fit_classifier_sets_fitted(self):
        rng = np.random.default_rng(42)
        support = rng.normal(0, 0.1, (10, 768)).astype(np.float32)
        labels = np.repeat(np.arange(2), 5).astype(np.int64)
        label_names = {0: "Fabaceae", 1: "Meliaceae"}

        config = ClassifierConfig(device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_classifier(support, labels, label_names)
        assert stage._is_fitted is True

    def test_run_with_synthetic_features(self):
        """Classify with well-separated synthetic embeddings."""
        rng = np.random.default_rng(42)
        n_classes, dim = 3, 768

        # Well-separated centroids in orthogonal subspaces
        centroids = np.zeros((n_classes, dim), dtype=np.float32)
        for i in range(n_classes):
            centroids[i, i * 100 : (i + 1) * 100] = 1.0

        # Support: 10 samples per class near each centroid
        support_features = np.vstack(
            [
                centroids[i]
                + rng.normal(0, 0.01, (10, dim)).astype(np.float32)
                for i in range(n_classes)
            ]
        )
        support_labels = np.repeat(np.arange(n_classes), 10)
        label_names = {0: "Fabaceae", 1: "Meliaceae", 2: "Sapindaceae"}

        # Query close to class 0
        query = (
            centroids[0:1]
            + rng.normal(0, 0.005, (1, dim)).astype(np.float32)
        )

        config = ClassifierConfig(
            rank="family", k_shot=10, top_k=3, device="cpu"
        )
        stage = ClassificationStage(config=config)
        stage._fit_classifier(support_features, support_labels, label_names)
        stage._extractor = _make_mock_extractor(query)

        result = stage.run({"image_paths": ["/test.jpg"]})
        assert result.stage_name == "classification"
        assert result.error is None
        assert "predictions" in result.data
        preds = result.data["predictions"]
        assert len(preds) <= 3
        assert preds[0]["rank_value"] == "Fabaceae"
        assert 0.0 <= preds[0]["softmax_score"] <= 1.0
        assert preds[0]["rank_position"] == 1

    def test_multi_image_pooling(self):
        """Multiple images should be mean-pooled before classification."""
        rng = np.random.default_rng(42)
        n_classes, dim = 2, 768

        # Two-class support set (NearestCentroid requires >= 2 classes)
        centroids = np.zeros((n_classes, dim), dtype=np.float32)
        for i in range(n_classes):
            centroids[i, i * 100 : (i + 1) * 100] = 1.0

        support = np.vstack(
            [
                centroids[i]
                + rng.normal(0, 0.01, (5, dim)).astype(np.float32)
                for i in range(n_classes)
            ]
        )
        support_labels = np.repeat(np.arange(n_classes), 5).astype(np.int64)
        label_names = {0: "TestFamilyA", 1: "TestFamilyB"}

        # Return 3 feature vectors (simulating 3 images)
        multi_features = rng.normal(0, 0.1, (3, dim)).astype(np.float32)

        config = ClassifierConfig(top_k=1, device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_classifier(support, support_labels, label_names)
        stage._extractor = _make_mock_extractor(multi_features)

        result = stage.run(
            {"image_paths": ["/a.jpg", "/b.jpg", "/c.jpg"]}
        )
        assert result.data["num_images_pooled"] == 3
        assert len(result.data["predictions"]) == 1

    def test_elapsed_ms_recorded(self):
        """Elapsed time should be recorded in the result."""
        rng = np.random.default_rng(42)
        n_classes, dim = 2, 768

        centroids = np.zeros((n_classes, dim), dtype=np.float32)
        for i in range(n_classes):
            centroids[i, i * 100 : (i + 1) * 100] = 1.0

        support = np.vstack(
            [
                centroids[i]
                + rng.normal(0, 0.01, (5, dim)).astype(np.float32)
                for i in range(n_classes)
            ]
        )
        support_labels = np.repeat(np.arange(n_classes), 5).astype(np.int64)
        label_names = {0: "X", 1: "Y"}

        query = rng.normal(0, 0.1, (1, dim)).astype(np.float32)

        config = ClassifierConfig(top_k=1, device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_classifier(support, support_labels, label_names)
        stage._extractor = _make_mock_extractor(query)

        result = stage.run({"image_paths": ["/test.jpg"]})
        assert result.elapsed_ms > 0

    def test_confidence_sums_to_one(self):
        """Confidences across all classes should sum to ~1.0."""
        rng = np.random.default_rng(42)
        n_classes, dim = 3, 768

        centroids = np.zeros((n_classes, dim), dtype=np.float32)
        for i in range(n_classes):
            centroids[i, i * 100 : (i + 1) * 100] = 1.0

        support = np.vstack(
            [
                centroids[i]
                + rng.normal(0, 0.01, (10, dim)).astype(np.float32)
                for i in range(n_classes)
            ]
        )
        support_labels = np.repeat(np.arange(n_classes), 10)
        label_names = {0: "A", 1: "B", 2: "C"}

        query = centroids[1:2] + rng.normal(0, 0.005, (1, dim)).astype(
            np.float32
        )

        config = ClassifierConfig(top_k=3, device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_classifier(support, support_labels, label_names)
        stage._extractor = _make_mock_extractor(query)

        result = stage.run({"image_paths": ["/test.jpg"]})
        total = sum(p["softmax_score"] for p in result.data["predictions"])
        np.testing.assert_allclose(total, 1.0, atol=1e-5)

    def test_embedding_dim_in_result(self):
        """Result should include embedding dimension metadata."""
        rng = np.random.default_rng(42)
        n_classes, dim = 2, 768

        centroids = np.zeros((n_classes, dim), dtype=np.float32)
        for i in range(n_classes):
            centroids[i, i * 100 : (i + 1) * 100] = 1.0

        support = np.vstack(
            [
                centroids[i]
                + rng.normal(0, 0.01, (5, dim)).astype(np.float32)
                for i in range(n_classes)
            ]
        )
        support_labels = np.repeat(np.arange(n_classes), 5).astype(np.int64)

        query = rng.normal(0, 0.1, (1, dim)).astype(np.float32)

        config = ClassifierConfig(top_k=1, device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_classifier(support, support_labels, {0: "A", 1: "B"})
        stage._extractor = _make_mock_extractor(query)

        result = stage.run({"image_paths": ["/test.jpg"]})
        assert result.data["embedding_dim"] == 768

    @patch("seedlearn.clip.cache.CachedFeatureExtractor")
    @patch("seedlearn.data.splits.load_split")
    def test_load_from_cache(self, mock_load_split, MockCachedExtractor):
        """load_from_cache should fit classifier from cached features + split."""
        rng = np.random.default_rng(42)
        n_classes, dim = 2, 768

        # Fake cached features (10 total)
        features = rng.normal(0, 0.1, (10, dim)).astype(np.float32)
        labels = np.repeat(np.arange(n_classes), 5).astype(np.int64)
        paths = np.array([f"/img_{i}.jpg" for i in range(10)])

        # Mock split: first 7 train, last 3 test
        mock_split = MagicMock()
        mock_split.train_indices = np.arange(7)
        mock_split.id_to_label = {0: "Fabaceae", 1: "Meliaceae"}
        mock_load_split.return_value = mock_split

        # Mock cache extractor
        mock_cache_inst = MockCachedExtractor.return_value
        mock_cache_inst.load_cached_features.return_value = (features, labels, paths)

        config = ClassifierConfig(device="cpu")
        stage = ClassificationStage(config=config)
        stage.load_from_cache("/fake/cache", "/fake/split")

        assert stage._is_fitted is True
        assert stage._support_image_paths is not None

    # -- Enriched output tests (Level 0 + Level 1) --

    def test_predictions_include_distances(self):
        """Each prediction should include l2_distance and cosine_similarity."""
        rng = np.random.default_rng(42)
        n_classes, dim = 3, 768

        centroids = np.zeros((n_classes, dim), dtype=np.float32)
        for i in range(n_classes):
            centroids[i, i * 100 : (i + 1) * 100] = 1.0

        support = np.vstack(
            [
                centroids[i]
                + rng.normal(0, 0.01, (10, dim)).astype(np.float32)
                for i in range(n_classes)
            ]
        )
        support_labels = np.repeat(np.arange(n_classes), 10)
        label_names = {0: "A", 1: "B", 2: "C"}

        query = centroids[0:1] + rng.normal(0, 0.005, (1, dim)).astype(np.float32)

        config = ClassifierConfig(top_k=3, device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_classifier(support, support_labels, label_names)
        stage._extractor = _make_mock_extractor(query)

        result = stage.run({"image_paths": ["/test.jpg"]})
        for pred in result.data["predictions"]:
            assert "l2_distance" in pred
            assert "cosine_similarity" in pred
            assert pred["l2_distance"] >= 0.0
            assert -1.0 <= pred["cosine_similarity"] <= 1.0 + 1e-6

    def test_margin_present(self):
        """Margin should be in [0, 1]."""
        rng = np.random.default_rng(42)
        n_classes, dim = 3, 768

        centroids = np.zeros((n_classes, dim), dtype=np.float32)
        for i in range(n_classes):
            centroids[i, i * 100 : (i + 1) * 100] = 1.0

        support = np.vstack(
            [
                centroids[i]
                + rng.normal(0, 0.01, (10, dim)).astype(np.float32)
                for i in range(n_classes)
            ]
        )
        support_labels = np.repeat(np.arange(n_classes), 10)
        label_names = {0: "A", 1: "B", 2: "C"}

        query = centroids[0:1] + rng.normal(0, 0.005, (1, dim)).astype(np.float32)

        config = ClassifierConfig(top_k=3, device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_classifier(support, support_labels, label_names)
        stage._extractor = _make_mock_extractor(query)

        result = stage.run({"image_paths": ["/test.jpg"]})
        assert "margin" in result.data
        assert 0.0 <= result.data["margin"] <= 1.0

    def test_per_image_predictions_count(self):
        """per_image_predictions should have one entry per input image."""
        rng = np.random.default_rng(42)
        n_classes, dim = 2, 768

        centroids = np.zeros((n_classes, dim), dtype=np.float32)
        for i in range(n_classes):
            centroids[i, i * 100 : (i + 1) * 100] = 1.0

        support = np.vstack(
            [
                centroids[i]
                + rng.normal(0, 0.01, (5, dim)).astype(np.float32)
                for i in range(n_classes)
            ]
        )
        support_labels = np.repeat(np.arange(n_classes), 5).astype(np.int64)
        label_names = {0: "X", 1: "Y"}

        multi_features = rng.normal(0, 0.1, (3, dim)).astype(np.float32)

        config = ClassifierConfig(top_k=1, device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_classifier(support, support_labels, label_names)
        stage._extractor = _make_mock_extractor(multi_features)

        result = stage.run(
            {"image_paths": ["/a.jpg", "/b.jpg", "/c.jpg"]}
        )
        per_img = result.data["per_image_predictions"]
        assert len(per_img) == 3

    def test_per_image_predictions_structure(self):
        """Each per-image entry should have required fields."""
        rng = np.random.default_rng(42)
        n_classes, dim = 2, 768

        centroids = np.zeros((n_classes, dim), dtype=np.float32)
        for i in range(n_classes):
            centroids[i, i * 100 : (i + 1) * 100] = 1.0

        support = np.vstack(
            [
                centroids[i]
                + rng.normal(0, 0.01, (5, dim)).astype(np.float32)
                for i in range(n_classes)
            ]
        )
        support_labels = np.repeat(np.arange(n_classes), 5).astype(np.int64)
        label_names = {0: "X", 1: "Y"}

        multi_features = rng.normal(0, 0.1, (2, dim)).astype(np.float32)

        config = ClassifierConfig(top_k=1, device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_classifier(support, support_labels, label_names)
        stage._extractor = _make_mock_extractor(multi_features)

        result = stage.run({"image_paths": ["/a.jpg", "/b.jpg"]})
        for entry in result.data["per_image_predictions"]:
            assert "image_path" in entry
            assert "top1_label" in entry
            assert "top1_softmax_score" in entry
            assert 0.0 <= entry["top1_softmax_score"] <= 1.0

    def test_nearest_support_in_result(self):
        """nearest_support list should be present in output."""
        rng = np.random.default_rng(42)
        n_classes, dim = 3, 768

        centroids = np.zeros((n_classes, dim), dtype=np.float32)
        for i in range(n_classes):
            centroids[i, i * 100 : (i + 1) * 100] = 1.0

        support = np.vstack(
            [
                centroids[i]
                + rng.normal(0, 0.01, (10, dim)).astype(np.float32)
                for i in range(n_classes)
            ]
        )
        support_labels = np.repeat(np.arange(n_classes), 10)
        label_names = {0: "A", 1: "B", 2: "C"}

        query = centroids[0:1] + rng.normal(0, 0.005, (1, dim)).astype(np.float32)

        config = ClassifierConfig(top_k=3, device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_classifier(support, support_labels, label_names)
        stage._extractor = _make_mock_extractor(query)

        result = stage.run({"image_paths": ["/test.jpg"]})
        assert "nearest_support" in result.data
        assert isinstance(result.data["nearest_support"], list)
        assert len(result.data["nearest_support"]) == 5

    def test_nearest_support_fields(self):
        """Each nearest_support entry should have required keys."""
        rng = np.random.default_rng(42)
        n_classes, dim = 2, 768

        centroids = np.zeros((n_classes, dim), dtype=np.float32)
        for i in range(n_classes):
            centroids[i, i * 100 : (i + 1) * 100] = 1.0

        support = np.vstack(
            [
                centroids[i]
                + rng.normal(0, 0.01, (5, dim)).astype(np.float32)
                for i in range(n_classes)
            ]
        )
        support_labels = np.repeat(np.arange(n_classes), 5).astype(np.int64)
        label_names = {0: "X", 1: "Y"}

        query = rng.normal(0, 0.1, (1, dim)).astype(np.float32)

        config = ClassifierConfig(top_k=1, device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_classifier(support, support_labels, label_names)
        stage._extractor = _make_mock_extractor(query)

        result = stage.run({"image_paths": ["/test.jpg"]})
        for entry in result.data["nearest_support"]:
            assert "label" in entry
            assert "l2_distance" in entry
            assert "cosine_similarity" in entry

    def test_nearest_support_with_image_paths(self):
        """image_path should be present when support_image_paths provided."""
        rng = np.random.default_rng(42)
        n_classes, dim = 2, 768

        centroids = np.zeros((n_classes, dim), dtype=np.float32)
        for i in range(n_classes):
            centroids[i, i * 100 : (i + 1) * 100] = 1.0

        support = np.vstack(
            [
                centroids[i]
                + rng.normal(0, 0.01, (5, dim)).astype(np.float32)
                for i in range(n_classes)
            ]
        )
        support_labels = np.repeat(np.arange(n_classes), 5).astype(np.int64)
        label_names = {0: "X", 1: "Y"}
        support_paths = np.array([f"/support/img_{i}.jpg" for i in range(10)])

        query = rng.normal(0, 0.1, (1, dim)).astype(np.float32)

        config = ClassifierConfig(top_k=1, device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_classifier(
            support, support_labels, label_names, support_paths
        )
        stage._extractor = _make_mock_extractor(query)

        result = stage.run({"image_paths": ["/test.jpg"]})
        for entry in result.data["nearest_support"]:
            assert "image_path" in entry

    def test_nearest_support_without_image_paths(self):
        """image_path should be absent when support_image_paths is None."""
        rng = np.random.default_rng(42)
        n_classes, dim = 2, 768

        centroids = np.zeros((n_classes, dim), dtype=np.float32)
        for i in range(n_classes):
            centroids[i, i * 100 : (i + 1) * 100] = 1.0

        support = np.vstack(
            [
                centroids[i]
                + rng.normal(0, 0.01, (5, dim)).astype(np.float32)
                for i in range(n_classes)
            ]
        )
        support_labels = np.repeat(np.arange(n_classes), 5).astype(np.int64)
        label_names = {0: "X", 1: "Y"}

        query = rng.normal(0, 0.1, (1, dim)).astype(np.float32)

        config = ClassifierConfig(top_k=1, device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_classifier(support, support_labels, label_names)  # no paths
        stage._extractor = _make_mock_extractor(query)

        result = stage.run({"image_paths": ["/test.jpg"]})
        for entry in result.data["nearest_support"]:
            assert "image_path" not in entry
