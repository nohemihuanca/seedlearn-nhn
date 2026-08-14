"""Tests for SimpleShot classifier."""

from __future__ import annotations

import numpy as np
import pytest

from seedlearn.clip.simpleshot import SimpleShot, FewShotClassifier, l2_normalize


class TestL2Normalize:
    """Tests for L2 normalization utility."""

    def test_output_has_unit_norm(self):
        rng = np.random.default_rng(0)
        features = rng.standard_normal((10, 128)).astype(np.float32)
        normed = l2_normalize(features)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_zero_vector_handled(self):
        features = np.zeros((1, 128), dtype=np.float32)
        normed = l2_normalize(features)
        assert np.all(np.isfinite(normed))


class TestSimpleShot:
    """Tests for SimpleShot classifier."""

    def test_is_few_shot_classifier(self):
        clf = SimpleShot(device="cpu")
        assert isinstance(clf, FewShotClassifier)

    def test_fit_predict_basic(self, synthetic_features):
        features, labels = synthetic_features
        # Use interleaved indices to get all 3 classes in both support and query
        support_idx = list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25))
        query_idx = list(range(5, 10)) + list(range(15, 20)) + list(range(25, 30))
        support_features = features[support_idx]
        support_labels = labels[support_idx]
        query_features = features[query_idx]
        query_labels = labels[query_idx]

        clf = SimpleShot(device="cpu")
        clf.fit(support_features, support_labels)

        assert clf.is_fitted

        predictions = clf.predict(query_features)
        assert predictions.shape == query_labels.shape
        assert predictions.dtype == query_labels.dtype

        accuracy = np.mean(predictions == query_labels)
        assert accuracy > 0.5, f"Expected accuracy > 0.5 (random=0.33), got {accuracy}"

    def test_predict_proba_shape(self, synthetic_features):
        features, labels = synthetic_features
        # Use interleaved indices to include all 3 classes in support
        support_idx = list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25))
        query_idx = list(range(5, 10)) + list(range(15, 20)) + list(range(25, 30))
        clf = SimpleShot(device="cpu")
        clf.fit(features[support_idx], labels[support_idx])

        proba = clf.predict_proba(features[query_idx])
        assert proba.shape == (15, 3)  # 15 queries, 3 classes

        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_before_fit_raises(self):
        clf = SimpleShot(device="cpu")
        with pytest.raises(RuntimeError, match="fitted"):
            clf.predict(np.zeros((5, 128), dtype=np.float32))

    def test_fit_predict_convenience(self, synthetic_features):
        features, labels = synthetic_features
        clf = SimpleShot(device="cpu")
        predictions = clf.fit_predict(features[:15], labels[:15], features[15:])
        assert predictions.shape == (15,)

    def test_mismatched_features_labels_raises(self):
        clf = SimpleShot(device="cpu")
        with pytest.raises(ValueError, match="same length"):
            clf.fit(np.zeros((10, 128), dtype=np.float32), np.zeros(5, dtype=np.int64))

    # -- predict_detail tests --

    def test_predict_detail_shape(self, synthetic_features):
        features, labels = synthetic_features
        support_idx = list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25))
        query_idx = list(range(5, 10)) + list(range(15, 20)) + list(range(25, 30))
        clf = SimpleShot(device="cpu")
        clf.fit(features[support_idx], labels[support_idx])

        details = clf.predict_detail(features[query_idx])
        assert len(details) == 15
        for d in details:
            assert d.probabilities.shape == (3,)
            assert d.l2_distances.shape == (3,)
            assert d.cosine_similarities.shape == (3,)

    def test_predict_detail_matches_predict_proba(self, synthetic_features):
        features, labels = synthetic_features
        support_idx = list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25))
        query_idx = list(range(5, 10)) + list(range(15, 20)) + list(range(25, 30))
        clf = SimpleShot(device="cpu")
        clf.fit(features[support_idx], labels[support_idx])

        proba = clf.predict_proba(features[query_idx])
        details = clf.predict_detail(features[query_idx])
        for i, d in enumerate(details):
            np.testing.assert_allclose(d.probabilities, proba[i], atol=1e-6)

    def test_predict_detail_cosine_range(self, synthetic_features):
        features, labels = synthetic_features
        support_idx = list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25))
        query_idx = list(range(5, 10))
        clf = SimpleShot(device="cpu")
        clf.fit(features[support_idx], labels[support_idx])

        details = clf.predict_detail(features[query_idx])
        for d in details:
            assert np.all(d.cosine_similarities >= -1.0 - 1e-6)
            assert np.all(d.cosine_similarities <= 1.0 + 1e-6)

    def test_predict_detail_l2_positive(self, synthetic_features):
        features, labels = synthetic_features
        support_idx = list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25))
        query_idx = list(range(5, 10))
        clf = SimpleShot(device="cpu")
        clf.fit(features[support_idx], labels[support_idx])

        details = clf.predict_detail(features[query_idx])
        for d in details:
            assert np.all(d.l2_distances >= 0.0)

    def test_predict_detail_before_fit_raises(self):
        clf = SimpleShot(device="cpu")
        with pytest.raises(RuntimeError, match="fitted"):
            clf.predict_detail(np.zeros((5, 128), dtype=np.float32))

    # -- find_nearest_support tests --

    def test_find_nearest_support_returns_k(self, synthetic_features):
        features, labels = synthetic_features
        support_idx = list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25))
        query_idx = list(range(5, 8))  # 3 queries
        clf = SimpleShot(device="cpu")
        clf.fit(features[support_idx], labels[support_idx])

        results = clf.find_nearest_support(features[query_idx], k=5)
        assert len(results) == 3
        for matches in results:
            assert len(matches) == 5

    def test_find_nearest_support_sorted_by_distance(self, synthetic_features):
        features, labels = synthetic_features
        support_idx = list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25))
        query_idx = [5]
        clf = SimpleShot(device="cpu")
        clf.fit(features[support_idx], labels[support_idx])

        results = clf.find_nearest_support(features[query_idx], k=5)
        distances = [m.l2_distance for m in results[0]]
        assert distances == sorted(distances)

    def test_find_nearest_support_same_class_dominant(self, synthetic_features):
        features, labels = synthetic_features
        # Well-separated data: class-0 query should match class-0 support mostly
        support_idx = list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25))
        query_idx = [5]  # class 0
        clf = SimpleShot(device="cpu")
        clf.fit(features[support_idx], labels[support_idx])

        results = clf.find_nearest_support(features[query_idx], k=5)
        same_class_count = sum(1 for m in results[0] if m.label_id == 0)
        assert same_class_count >= 3, f"Expected >=3 same-class, got {same_class_count}"

    def test_find_nearest_support_k_exceeds_support(self, synthetic_features):
        features, labels = synthetic_features
        support_idx = list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25))
        query_idx = [5]
        clf = SimpleShot(device="cpu")
        clf.fit(features[support_idx], labels[support_idx])

        # k=100 but only 15 support samples
        results = clf.find_nearest_support(features[query_idx], k=100)
        assert len(results[0]) == 15

    def test_find_nearest_support_before_fit_raises(self):
        clf = SimpleShot(device="cpu")
        with pytest.raises(RuntimeError, match="fitted"):
            clf.find_nearest_support(np.zeros((5, 128), dtype=np.float32))

    def test_find_nearest_support_cosine_range(self, synthetic_features):
        features, labels = synthetic_features
        support_idx = list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25))
        query_idx = list(range(5, 10))
        clf = SimpleShot(device="cpu")
        clf.fit(features[support_idx], labels[support_idx])

        results = clf.find_nearest_support(features[query_idx], k=5)
        for matches in results:
            for m in matches:
                assert -1.0 - 1e-6 <= m.cosine_similarity <= 1.0 + 1e-6
