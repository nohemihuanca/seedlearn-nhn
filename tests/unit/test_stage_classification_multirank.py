"""Tests for multi-rank ClassificationStage."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from typing import Any

from seedlearn.pipeline.config import ClassifierConfig
from seedlearn.pipeline.stages.classification import ClassificationStage


def _make_support_data(n_classes=3, n_per_class=10, dim=768, seed=42):
    """Create well-separated support features and labels for multiple ranks."""
    rng = np.random.default_rng(seed)
    centroids = np.zeros((n_classes, dim), dtype=np.float32)
    for i in range(n_classes):
        centroids[i, i * 100 : (i + 1) * 100] = 1.0

    features = np.vstack([
        centroids[i] + rng.normal(0, 0.01, (n_per_class, dim)).astype(np.float32)
        for i in range(n_classes)
    ])
    labels = np.repeat(np.arange(n_classes), n_per_class).astype(np.int64)
    return features, labels, centroids


class TestMultiRankClassification:
    """Tests for multi-rank fit and predict."""

    def test_fit_multirank_sets_fitted(self):
        features, labels, _ = _make_support_data()
        rank_labels = {"family": labels, "genus": labels, "species": labels}
        rank_label_names = {
            "family": {0: "Fabaceae", 1: "Meliaceae", 2: "Sapindaceae"},
            "genus": {0: "Inga", 1: "Swietenia", 2: "Cupania"},
            "species": {0: "Inga vera", 1: "Swietenia macrophylla", 2: "Cupania americana"},
        }

        config = ClassifierConfig(device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_multirank(features, rank_labels, rank_label_names)

        assert stage._is_fitted is True
        assert stage._is_multirank is True
        assert set(stage._classifiers.keys()) == {"family", "genus", "species"}

    def test_multirank_run_returns_predictions_by_rank(self):
        features, labels, centroids = _make_support_data()
        rank_labels = {"family": labels, "genus": labels, "species": labels}
        rank_label_names = {
            "family": {0: "Fabaceae", 1: "Meliaceae", 2: "Sapindaceae"},
            "genus": {0: "Inga", 1: "Swietenia", 2: "Cupania"},
            "species": {0: "Inga vera", 1: "Swietenia macrophylla", 2: "Cupania americana"},
        }

        rng = np.random.default_rng(42)
        query = centroids[0:1] + rng.normal(0, 0.005, (1, 768)).astype(np.float32)

        config = ClassifierConfig(top_k=3, device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_multirank(features, rank_labels, rank_label_names)
        stage._extractor = MagicMock()
        stage._extractor.extract_from_paths.return_value = query

        result = stage.run({"image_paths": ["/test.jpg"]})

        assert result.error is None
        assert "predictions_by_rank" in result.data
        assert set(result.data["predictions_by_rank"].keys()) == {"family", "genus", "species"}
        assert result.data["predictions_by_rank"]["family"][0]["rank_value"] == "Fabaceae"

    def test_multirank_run_includes_consistency(self):
        features, labels, centroids = _make_support_data()
        rank_labels = {"family": labels, "genus": labels, "species": labels}
        rank_label_names = {
            "family": {0: "Fabaceae", 1: "Meliaceae", 2: "Sapindaceae"},
            "genus": {0: "Inga", 1: "Swietenia", 2: "Cupania"},
            "species": {0: "Inga vera", 1: "Swietenia macrophylla", 2: "Cupania americana"},
        }
        taxonomy = {
            "genus_to_family": {"Inga": "Fabaceae", "Swietenia": "Meliaceae", "Cupania": "Sapindaceae"},
            "species_to_genus": {"Inga vera": "Inga", "Swietenia macrophylla": "Swietenia", "Cupania americana": "Cupania"},
            "species_to_family": {"Inga vera": "Fabaceae", "Swietenia macrophylla": "Meliaceae", "Cupania americana": "Sapindaceae"},
        }

        rng = np.random.default_rng(42)
        query = centroids[0:1] + rng.normal(0, 0.005, (1, 768)).astype(np.float32)

        config = ClassifierConfig(top_k=3, device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_multirank(features, rank_labels, rank_label_names)
        stage._taxonomy = taxonomy
        stage._extractor = MagicMock()
        stage._extractor.extract_from_paths.return_value = query

        result = stage.run({"image_paths": ["/test.jpg"]})

        assert "hierarchical_consistency" in result.data
        assert result.data["hierarchical_consistency"]["consistent"] is True

    def test_multirank_run_includes_margin_by_rank(self):
        features, labels, centroids = _make_support_data()
        rank_labels = {"family": labels, "genus": labels, "species": labels}
        rank_label_names = {
            "family": {0: "A", 1: "B", 2: "C"},
            "genus": {0: "X", 1: "Y", 2: "Z"},
            "species": {0: "Xa", 1: "Yb", 2: "Zc"},
        }

        rng = np.random.default_rng(42)
        query = centroids[0:1] + rng.normal(0, 0.005, (1, 768)).astype(np.float32)

        config = ClassifierConfig(top_k=3, device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_multirank(features, rank_labels, rank_label_names)
        stage._extractor = MagicMock()
        stage._extractor.extract_from_paths.return_value = query

        result = stage.run({"image_paths": ["/test.jpg"]})

        assert "margin_by_rank" in result.data
        for rank in ("family", "genus", "species"):
            assert 0.0 <= result.data["margin_by_rank"][rank] <= 1.0

    def test_multirank_run_includes_confidence_gate(self):
        features, labels, centroids = _make_support_data()
        rank_labels = {"family": labels, "genus": labels, "species": labels}
        rank_label_names = {
            "family": {0: "A", 1: "B", 2: "C"},
            "genus": {0: "X", 1: "Y", 2: "Z"},
            "species": {0: "Xa", 1: "Yb", 2: "Zc"},
        }

        rng = np.random.default_rng(42)
        query = centroids[0:1] + rng.normal(0, 0.005, (1, 768)).astype(np.float32)

        config = ClassifierConfig(top_k=3, device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_multirank(features, rank_labels, rank_label_names)
        stage._distance_thresholds = {"family": 10.0, "genus": 10.0, "species": 10.0}
        stage._extractor = MagicMock()
        stage._extractor.extract_from_paths.return_value = query

        result = stage.run({"image_paths": ["/test.jpg"]})

        assert "confidence_gate" in result.data
        assert result.data["confidence_gate"]["family_in_distribution"] is True

    def test_backward_compat_single_rank_still_works(self):
        """Fitting via _fit_classifier (old API) should still produce single-rank output."""
        rng = np.random.default_rng(42)
        n_classes, dim = 2, 768
        centroids = np.zeros((n_classes, dim), dtype=np.float32)
        for i in range(n_classes):
            centroids[i, i * 100 : (i + 1) * 100] = 1.0
        support = np.vstack([
            centroids[i] + rng.normal(0, 0.01, (5, dim)).astype(np.float32)
            for i in range(n_classes)
        ])
        support_labels = np.repeat(np.arange(n_classes), 5).astype(np.int64)
        query = centroids[0:1] + rng.normal(0, 0.005, (1, dim)).astype(np.float32)

        config = ClassifierConfig(top_k=2, device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_classifier(support, support_labels, {0: "A", 1: "B"})
        stage._extractor = MagicMock()
        stage._extractor.extract_from_paths.return_value = query

        result = stage.run({"image_paths": ["/test.jpg"]})

        assert "predictions" in result.data
        assert "predictions_by_rank" not in result.data


class TestDistanceThresholdCalibration:
    """Tests for calibrate_distance_thresholds."""

    def test_calibrate_returns_per_rank_thresholds(self):
        features, labels, _ = _make_support_data(n_per_class=15)
        rank_labels = {"family": labels, "genus": labels, "species": labels}
        rank_label_names = {
            "family": {0: "A", 1: "B", 2: "C"},
            "genus": {0: "X", 1: "Y", 2: "Z"},
            "species": {0: "Xa", 1: "Yb", 2: "Zc"},
        }

        config = ClassifierConfig(top_k=3, device="cpu")
        stage = ClassificationStage(config=config)
        stage._fit_multirank(features, rank_labels, rank_label_names)

        val_indices = np.concatenate([np.arange(10, 15), np.arange(25, 30), np.arange(40, 45)])
        thresholds = stage.calibrate_distance_thresholds(
            features, val_indices, rank_labels,
        )

        assert set(thresholds.keys()) == {"family", "genus", "species"}
        for threshold in thresholds.values():
            assert threshold > 0.0


class TestHierarchicalConsistency:
    """Tests for _check_hierarchical_consistency."""

    def test_consistent_hierarchy(self):
        config = ClassifierConfig(device="cpu")
        stage = ClassificationStage(config=config)
        stage._taxonomy = {
            "genus_to_family": {"Inga": "Fabaceae"},
            "species_to_genus": {"Inga vera": "Inga"},
            "species_to_family": {"Inga vera": "Fabaceae"},
        }

        predictions_by_rank = {
            "family": [{"rank_value": "Fabaceae"}],
            "genus": [{"rank_value": "Inga"}],
            "species": [{"rank_value": "Inga vera"}],
        }
        result = stage._check_hierarchical_consistency(predictions_by_rank)
        assert result["consistent"] is True

    def test_inconsistent_genus_family(self):
        config = ClassifierConfig(device="cpu")
        stage = ClassificationStage(config=config)
        stage._taxonomy = {
            "genus_to_family": {"Inga": "Fabaceae", "Quercus": "Fagaceae"},
            "species_to_genus": {},
            "species_to_family": {},
        }

        predictions_by_rank = {
            "family": [{"rank_value": "Fagaceae"}],
            "genus": [{"rank_value": "Inga"}],
        }
        result = stage._check_hierarchical_consistency(predictions_by_rank)
        assert result["consistent"] is False
        assert result["family_genus_agree"] is False


class TestLoadFromMultirankCache:
    """Integration test for load_from_multirank_cache — the production entry point."""

    def test_load_from_multirank_cache_fits_stage(self, tmp_path):
        """Write cache + splits to disk, load via load_from_multirank_cache, verify stage is fitted."""
        rng = np.random.default_rng(42)
        n_samples, dim = 30, 768

        # Create features
        features = rng.normal(0, 1, (n_samples, dim)).astype(np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / norms

        # Labels: 3 families, 3 genera, 3 species
        family_labels = np.repeat(np.arange(3), 10).astype(np.int64)
        genus_labels = np.repeat(np.arange(3), 10).astype(np.int64)
        species_labels = np.repeat(np.arange(3), 10).astype(np.int64)
        image_paths = np.array([f"/img/{i}.jpg" for i in range(n_samples)])
        individual_ids = np.array([f"IND{i // 5}" for i in range(n_samples)])

        # Write cache
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        np.savez(
            cache_dir / "features.npz",
            features=features,
            family_labels=family_labels,
            genus_labels=genus_labels,
            species_labels=species_labels,
            image_paths=image_paths,
            individual_ids=individual_ids,
        )
        meta = {
            "label_maps": {
                "family": {"Fabaceae": 0, "Meliaceae": 1, "Sapindaceae": 2},
                "genus": {"Inga": 0, "Swietenia": 1, "Cupania": 2},
                "species": {"Inga vera": 0, "Swietenia macrophylla": 1, "Cupania americana": 2},
            },
            "taxonomy": {
                "genus_to_family": {"Inga": "Fabaceae", "Swietenia": "Meliaceae", "Cupania": "Sapindaceae"},
                "species_to_genus": {"Inga vera": "Inga", "Swietenia macrophylla": "Swietenia", "Cupania americana": "Cupania"},
                "species_to_family": {"Inga vera": "Fabaceae", "Swietenia macrophylla": "Meliaceae", "Cupania americana": "Sapindaceae"},
            },
        }
        (cache_dir / "features_meta.json").write_text(json.dumps(meta))

        # Write splits (all three ranks share the same train/val/test indices)
        train_idx = np.arange(0, 21)  # 7 per class
        val_idx = np.arange(21, 24)   # 1 per class
        test_idx = np.arange(24, 30)  # 2 per class

        split_paths = {}
        for rank, label_map in meta["label_maps"].items():
            split_dir = tmp_path / "splits" / rank
            split_dir.mkdir(parents=True)
            split_base = split_dir / "split_seed42"

            np.savez(split_base.with_suffix(".npz"),
                     train_indices=train_idx, val_indices=val_idx, test_indices=test_idx)

            id_to_label = {v: k for k, v in label_map.items()}
            split_meta = {
                "label_to_id": label_map,
                "id_to_label": {str(k): v for k, v in id_to_label.items()},
                "num_classes": len(label_map),
                "split_info": {"random_seed": 42},
            }
            split_base.with_suffix(".json").write_text(json.dumps(split_meta))
            split_paths[rank] = split_base

        # Load and verify
        config = ClassifierConfig(top_k=3, device="cpu")
        stage = ClassificationStage(config=config)
        stage.load_from_multirank_cache(cache_dir, split_paths)

        assert stage._is_fitted is True
        assert stage._is_multirank is True
        assert set(stage._classifiers.keys()) == {"family", "genus", "species"}
        assert stage._taxonomy == meta["taxonomy"]

        # Verify classifiers have correct class counts (centroids shape = (n_classes, dim))
        for rank_name, clf in stage._classifiers.items():
            assert clf.centroids.shape[0] == 3, f"{rank_name} should have 3 classes"
