"""SimpleShot few-shot learning classifier.

Based on the SimpleShot paper: https://arxiv.org/abs/1911.04623
Implementation adapted from the BioClip FewShotSimpleShot.ipynb notebook
and biobench: https://github.com/samuelstevens/biobench/blob/main/biobench/simpleshot.py

This module also contains the ``FewShotClassifier`` ABC that all few-shot
classifiers must implement (inlined because only one concrete class exists).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import sklearn.neighbors
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data classes for enriched prediction output
# ---------------------------------------------------------------------------

@dataclass
class PredictionDetail:
    """Enriched prediction output for a single query.

    Attributes:
        probabilities: Softmax probabilities per class, shape (n_classes,).
        l2_distances: L2 distances to each class centroid, shape (n_classes,).
        cosine_similarities: Cosine similarities to each centroid, shape (n_classes,).
    """

    probabilities: npt.NDArray[np.float32]
    l2_distances: npt.NDArray[np.float32]
    cosine_similarities: npt.NDArray[np.float32]


@dataclass
class NearestSupportMatch:
    """A single nearest-neighbor match from the support set.

    Attributes:
        support_index: Index into the original support arrays.
        label_id: Original (decoded) label ID.
        l2_distance: L2 distance to the query.
        cosine_similarity: Cosine similarity to the query.
    """

    support_index: int
    label_id: int
    l2_distance: float
    cosine_similarity: float


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class FewShotClassifier(ABC):
    """Abstract base class for few-shot learning classifiers.

    Attributes:
        device: The torch device to use for computation.
        is_fitted: Whether the classifier has been fitted on training data.
    """

    def __init__(self, device: Any) -> None:
        """Initialize the few-shot classifier.

        Args:
            device: The torch device to use for computation.
        """
        self.device = device
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        support_features: npt.NDArray[np.float32],
        support_labels: npt.NDArray[np.int64],
    ) -> None:
        """Fit the classifier on support set features and labels.

        Args:
            support_features: Support set feature vectors of shape (n_samples, n_features).
            support_labels: Support set labels of shape (n_samples,).
        """
        pass

    @abstractmethod
    def predict(
        self,
        query_features: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.int64]:
        """Predict labels for query features.

        Args:
            query_features: Query feature vectors of shape (n_queries, n_features).

        Returns:
            Predicted labels of shape (n_queries,).
        """
        pass

    def predict_proba(
        self,
        query_features: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Predict class probabilities for query features.

        Args:
            query_features: Query feature vectors of shape (n_queries, n_features).

        Returns:
            Class probabilities of shape (n_queries, n_classes).

        Raises:
            NotImplementedError: If the subclass doesn't support probability prediction.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support probability prediction"
        )

    def fit_predict(
        self,
        support_features: npt.NDArray[np.float32],
        support_labels: npt.NDArray[np.int64],
        query_features: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.int64]:
        """Fit on support set and predict on query set (convenience method).

        Args:
            support_features: Support set feature vectors of shape (n_support, n_features).
            support_labels: Support set labels of shape (n_support,).
            query_features: Query feature vectors of shape (n_queries, n_features).

        Returns:
            Predicted labels of shape (n_queries,).
        """
        self.fit(support_features, support_labels)
        return self.predict(query_features)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def l2_normalize(features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """L2-normalize a batch of features.

    Args:
        features: Feature matrix of shape (n_samples, n_features).

    Returns:
        L2-normalized features of the same shape.
    """
    norms = np.linalg.norm(features, ord=2, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return features / norms


# ---------------------------------------------------------------------------
# SimpleShot
# ---------------------------------------------------------------------------

class SimpleShot(FewShotClassifier):
    """SimpleShot few-shot learning classifier.

    SimpleShot performs the following steps:
    1. Compute the mean of all support features (global centroid)
    2. Subtract the mean from support features (mean centering)
    3. L2-normalize the centered features
    4. Compute class centroids using nearest centroid classifier
    5. For prediction:
       a. Center query features using the same global mean
       b. L2-normalize centered query features
       c. Find the closest class centroid

    Attributes:
        device: The torch device for computation.
        is_fitted: Whether the classifier has been fitted.
        x_mean: Global mean of support features.
        centroids: Class centroids in the transformed feature space.
        label_encoder: Mapping from original labels to contiguous indices.
        label_decoder: Mapping from contiguous indices to original labels.
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        """Initialize SimpleShot classifier.

        Args:
            device: Torch device to use for computation.
        """
        super().__init__(device)
        self.x_mean: npt.NDArray[np.float32] | None = None
        self.centroids: torch.Tensor | None = None
        self.label_encoder: dict[int, int] = {}
        self.label_decoder: dict[int, int] = {}
        self._support_features_normed: npt.NDArray[np.float32] | None = None
        self._support_labels_encoded: npt.NDArray[np.int64] | None = None

    def mean_normalize(
        self, features: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Mean-center and L2-normalize features.

        Args:
            features: Feature matrix of shape (n_samples, n_features).

        Returns:
            Mean-centered and L2-normalized features.

        Raises:
            RuntimeError: If called before fit().
        """
        if self.x_mean is None:
            raise RuntimeError("Cannot normalize before fitting. Call fit() first.")

        return l2_normalize(features - self.x_mean)

    def fit(
        self,
        support_features: npt.NDArray[np.float32],
        support_labels: npt.NDArray[np.int64],
    ) -> None:
        """Fit the SimpleShot classifier on support features.

        Args:
            support_features: Support set features of shape (n_support, n_features).
            support_labels: Support set labels of shape (n_support,).

        Raises:
            ValueError: If support_features and support_labels have mismatched shapes.
        """
        if len(support_features) != len(support_labels):
            raise ValueError(
                f"Features ({len(support_features)}) and labels ({len(support_labels)}) "
                "must have the same length"
            )

        unique_labels = np.unique(support_labels)
        self.label_encoder = {int(label): idx for idx, label in enumerate(unique_labels)}
        self.label_decoder = {idx: int(label) for label, idx in self.label_encoder.items()}

        encoded_labels = np.array([self.label_encoder[int(label)] for label in support_labels])

        self.x_mean = support_features.mean(axis=0, keepdims=True)

        x_norm = self.mean_normalize(support_features)

        clf = sklearn.neighbors.NearestCentroid()
        clf.fit(x_norm, encoded_labels)

        self.centroids = (
            torch.from_numpy(clf.centroids_).type(torch.float32).to(self.device)
        )

        # Retain normalized support features for k-NN search
        self._support_features_normed = x_norm
        self._support_labels_encoded = encoded_labels

        self.is_fitted = True

    def predict(
        self,
        query_features: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.int64]:
        """Predict labels for query features.

        Args:
            query_features: Query features of shape (n_queries, n_features).

        Returns:
            Predicted labels of shape (n_queries,).

        Raises:
            RuntimeError: If the classifier hasn't been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction. Call fit() first.")

        x_norm = self.mean_normalize(query_features)

        x_test = torch.from_numpy(x_norm).type(torch.float32).to(self.device)

        distances = torch.linalg.vector_norm(
            x_test[:, None, :] - self.centroids[None, :, :],
            axis=2,
        )

        pred_indices = torch.argmin(distances, dim=1).cpu().numpy()

        predictions = np.array([self.label_decoder[int(idx)] for idx in pred_indices])

        return predictions

    def predict_proba(
        self,
        query_features: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Predict class probabilities using softmax over negative distances.

        Args:
            query_features: Query features of shape (n_queries, n_features).

        Returns:
            Class probabilities of shape (n_queries, n_classes).

        Raises:
            RuntimeError: If the classifier hasn't been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction. Call fit() first.")

        x_norm = self.mean_normalize(query_features)

        x_test = torch.from_numpy(x_norm).type(torch.float32).to(self.device)

        distances = torch.linalg.vector_norm(
            x_test[:, None, :] - self.centroids[None, :, :],
            axis=2,
        )

        probabilities = torch.softmax(-distances, dim=1)

        return probabilities.cpu().numpy()

    def predict_detail(
        self,
        query_features: npt.NDArray[np.float32],
    ) -> list[PredictionDetail]:
        """Predict with full diagnostic output per query.

        Returns probabilities (same as ``predict_proba``), raw L2 distances,
        and cosine similarities to each class centroid.

        Args:
            query_features: Query features of shape (n_queries, n_features).

        Returns:
            One ``PredictionDetail`` per query row.

        Raises:
            RuntimeError: If the classifier hasn't been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Classifier must be fitted before prediction. Call fit() first."
            )

        x_norm = self.mean_normalize(query_features)
        x_test = torch.from_numpy(x_norm).type(torch.float32).to(self.device)

        # L2 distances to centroids — same as predict / predict_proba
        distances = torch.linalg.vector_norm(
            x_test[:, None, :] - self.centroids[None, :, :],
            axis=2,
        )

        probabilities = torch.softmax(-distances, dim=1)

        # Cosine similarity — handles unnormalized centroids correctly
        cosine_sims = F.cosine_similarity(
            x_test[:, None, :], self.centroids[None, :, :], dim=2,
        )

        results: list[PredictionDetail] = []
        for i in range(x_test.shape[0]):
            results.append(
                PredictionDetail(
                    probabilities=probabilities[i].cpu().numpy(),
                    l2_distances=distances[i].cpu().numpy(),
                    cosine_similarities=cosine_sims[i].cpu().numpy(),
                )
            )
        return results

    def find_nearest_support(
        self,
        query_features: npt.NDArray[np.float32],
        k: int = 5,
    ) -> list[list[NearestSupportMatch]]:
        """Find k nearest support-set samples for each query.

        Args:
            query_features: Query features of shape (n_queries, n_features).
            k: Number of nearest neighbors to return per query.

        Returns:
            Nested list: one list of ``NearestSupportMatch`` per query,
            sorted by ascending L2 distance.

        Raises:
            RuntimeError: If the classifier hasn't been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Classifier must be fitted before prediction. Call fit() first."
            )

        x_norm = self.mean_normalize(query_features)
        x_query = torch.from_numpy(x_norm).type(torch.float32).to(self.device)
        x_support = torch.from_numpy(
            self._support_features_normed
        ).type(torch.float32).to(self.device)

        # L2 distances: (n_queries, n_support)
        dists = torch.cdist(x_query, x_support)

        # Cosine similarities: both are L2-normalized, so dot product = cosine
        cosines = x_query @ x_support.T

        # Clamp k to actual support size
        actual_k = min(k, x_support.shape[0])
        _, topk_indices = torch.topk(dists, actual_k, dim=1, largest=False)

        results: list[list[NearestSupportMatch]] = []
        for i in range(x_query.shape[0]):
            matches: list[NearestSupportMatch] = []
            for j in topk_indices[i]:
                j_int = int(j)
                encoded_label = int(self._support_labels_encoded[j_int])
                matches.append(
                    NearestSupportMatch(
                        support_index=j_int,
                        label_id=self.label_decoder[encoded_label],
                        l2_distance=float(dists[i, j_int]),
                        cosine_similarity=float(cosines[i, j_int]),
                    )
                )
            results.append(matches)
        return results
