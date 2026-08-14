"""Stage 2: Visual embedding classification using BioCLIP 2 + SimpleShot."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from seedlearn.pipeline.config import ClassifierConfig
from seedlearn.pipeline.protocol import StageResult

logger = logging.getLogger(__name__)


class ClassificationStage:
    """Stage 2: Classify seedling images using BioCLIP 2 embeddings + SimpleShot.

    Supports both single-rank classification (via ``_fit_classifier()`` /
    ``load_from_cache()``) and multi-rank classification (via
    ``_fit_multirank()`` / ``load_from_multirank_cache()``).

    Args:
        config: Classifier configuration.
    """

    def __init__(self, config: ClassifierConfig) -> None:
        self._config = config
        self._extractor = None  # Lazy-loaded FeatureExtractor
        self._classifier = None  # SimpleShot instance (single-rank)
        self._label_names: dict[int, str] = {}  # label_id -> label string
        self._support_image_paths: npt.NDArray | None = None
        self._is_fitted = False

        # Multi-rank state
        self._is_multirank = False
        self._classifiers: dict[str, Any] = {}  # rank -> SimpleShot
        self._rank_label_names: dict[str, dict[int, str]] = {}  # rank -> {id: name}
        self._distance_thresholds: dict[str, float] = {}
        self._taxonomy: dict[str, dict[str, str]] = {}

    @property
    def name(self) -> str:
        """Return stage name identifier."""
        return "classification"

    def _get_extractor(self):
        """Lazy-load the FeatureExtractor.

        Returns:
            Initialized FeatureExtractor instance.
        """
        if self._extractor is None:
            from seedlearn.clip.encoder import FeatureExtractor

            self._extractor = FeatureExtractor(
                device=self._config.device,
                model_str=self._config.model_str,
            )
        return self._extractor

    # ------------------------------------------------------------------ #
    #  Single-rank fitting
    # ------------------------------------------------------------------ #

    def _fit_classifier(
        self,
        support_features: npt.NDArray[np.float32],
        support_labels: npt.NDArray[np.int64],
        label_names: dict[int, str],
        support_image_paths: npt.NDArray | None = None,
    ) -> None:
        """Fit the SimpleShot classifier on pre-computed support features.

        Args:
            support_features: Support set features, shape (n_support, 768).
            support_labels: Integer label IDs, shape (n_support,).
            label_names: Mapping from label_id -> human-readable label string.
            support_image_paths: Optional paths to support images, shape (n_support,).
        """
        from seedlearn.clip.simpleshot import SimpleShot

        self._classifier = SimpleShot(device=self._config.device)
        self._classifier.fit(support_features, support_labels)
        self._label_names = label_names
        self._support_image_paths = support_image_paths
        self._is_fitted = True
        logger.info(
            "Classifier fitted: %d support samples, %d classes",
            len(support_features),
            len(label_names),
        )

    def load_from_cache(
        self,
        cache_dir: Path | str,
        split_path: Path | str,
    ) -> None:
        """Load cached features and split, then fit classifier on training set.

        Args:
            cache_dir: Directory containing cached feature .npz files.
            split_path: Path to the split files (without extension).
        """
        from seedlearn.clip.cache import CachedFeatureExtractor
        from seedlearn.data.splits import load_split

        split = load_split(Path(split_path))
        cache_extractor = CachedFeatureExtractor(
            cache_dir=Path(cache_dir), device=self._config.device,
        )
        cache_name = f"{self._config.rank}_features"
        features, labels, image_paths = cache_extractor.load_cached_features(
            cache_name
        )

        train_features = features[split.train_indices]
        train_labels = labels[split.train_indices]
        train_image_paths = image_paths[split.train_indices]

        self._fit_classifier(
            train_features, train_labels, split.id_to_label, train_image_paths
        )

    # ------------------------------------------------------------------ #
    #  Multi-rank fitting
    # ------------------------------------------------------------------ #

    def _fit_multirank(
        self,
        support_features: npt.NDArray[np.float32],
        rank_labels: dict[str, npt.NDArray[np.int64]],
        rank_label_names: dict[str, dict[int, str]],
        support_image_paths: npt.NDArray | None = None,
    ) -> None:
        """Fit one SimpleShot classifier per taxonomy rank.

        Args:
            support_features: Shared support features, shape (n_support, 768).
            rank_labels: Per-rank integer labels, e.g. {"family": array[N], ...}.
            rank_label_names: Per-rank label decoders, e.g. {"family": {0: "Fabaceae", ...}}.
            support_image_paths: Optional paths to support images.
        """
        from seedlearn.clip.simpleshot import SimpleShot

        self._classifiers = {}
        self._rank_label_names = rank_label_names
        self._support_image_paths = support_image_paths

        for rank_name, labels in rank_labels.items():
            classifier = SimpleShot(device=self._config.device)
            classifier.fit(support_features, labels)
            self._classifiers[rank_name] = classifier
            logger.info(
                "Fitted %s classifier: %d classes",
                rank_name,
                len(rank_label_names[rank_name]),
            )

        self._is_multirank = True
        self._is_fitted = True

    def load_from_multirank_cache(
        self,
        cache_dir: Path | str,
        split_paths: dict[str, Path | str],
    ) -> None:
        """Load multi-rank cache and fit classifiers for each rank with available splits.

        Args:
            cache_dir: Directory containing ``features.npz`` and ``features_meta.json``.
            split_paths: Mapping of rank name to split path (without extension),
                e.g. {"family": Path("splits/family/split_seed42"), ...}.
        """
        from seedlearn.clip.cache import load_multirank_cache
        from seedlearn.data.splits import load_split

        features, rank_labels, meta, image_paths = load_multirank_cache(cache_dir)

        rank_label_names: dict[str, dict[int, str]] = {}
        train_mask = np.zeros(len(features), dtype=bool)
        reference_train_set: set[int] | None = None

        for rank_name, split_path in split_paths.items():
            if rank_name not in rank_labels:
                logger.warning("No labels for rank '%s' in cache, skipping", rank_name)
                continue
            split = load_split(Path(split_path))
            current_train_set = set(split.train_indices.tolist())
            if reference_train_set is None:
                reference_train_set = current_train_set
            elif current_train_set != reference_train_set:
                logger.warning(
                    "Train indices for rank '%s' differ from first rank; "
                    "using union may include val/test samples as training data",
                    rank_name,
                )
            train_mask[split.train_indices] = True
            rank_label_names[rank_name] = split.id_to_label

        train_indices = np.where(train_mask)[0]
        train_features = features[train_indices]
        train_rank_labels = {
            rank: labels[train_indices]
            for rank, labels in rank_labels.items()
            if rank in split_paths
        }
        train_image_paths = image_paths[train_indices]

        self._fit_multirank(
            train_features, train_rank_labels, rank_label_names, train_image_paths
        )

        if "taxonomy" in meta:
            self._taxonomy = meta["taxonomy"]

    # ------------------------------------------------------------------ #
    #  Calibration
    # ------------------------------------------------------------------ #

    def calibrate_distance_thresholds(
        self,
        features: npt.NDArray[np.float32],
        val_indices: npt.NDArray[np.int64],
        rank_labels: dict[str, npt.NDArray[np.int64]],
        percentile: float | None = None,
    ) -> dict[str, float]:
        """Compute per-rank distance thresholds from validation set.

        For each rank, predict on validation samples. For correctly classified
        samples, take the Nth percentile of top-1 L2 distances as the threshold.

        Args:
            features: Full feature array.
            val_indices: Indices of validation samples.
            rank_labels: Per-rank label arrays (full, not sliced).
            percentile: Percentile for threshold (default from config).

        Returns:
            Dict of rank -> distance threshold.
        """
        if percentile is None:
            percentile = self._config.ood_percentile

        thresholds: dict[str, float] = {}
        val_features = features[val_indices]

        for rank_name, classifier in self._classifiers.items():
            if rank_name not in rank_labels:
                continue

            val_labels = rank_labels[rank_name][val_indices]
            details = classifier.predict_detail(val_features)

            correct_distances: list[float] = []
            for i, detail in enumerate(details):
                pred_idx = int(np.argmax(detail.probabilities))
                pred_label = classifier.label_decoder[pred_idx]
                if pred_label == val_labels[i]:
                    correct_distances.append(float(detail.l2_distances[pred_idx]))

            if correct_distances:
                thresholds[rank_name] = float(np.percentile(correct_distances, percentile))
            else:
                thresholds[rank_name] = float("inf")

        self._distance_thresholds = thresholds
        return thresholds

    # ------------------------------------------------------------------ #
    #  Hierarchical consistency
    # ------------------------------------------------------------------ #

    def _check_hierarchical_consistency(
        self, predictions_by_rank: dict[str, list[dict]]
    ) -> dict[str, Any]:
        """Check if top-1 predictions form a valid taxonomic hierarchy.

        Args:
            predictions_by_rank: Per-rank prediction lists (top-1 used).

        Returns:
            Dict with consistency flags and notes.
        """
        result: dict[str, Any] = {"consistent": True, "notes": []}

        top1 = {}
        for rank_name, preds in predictions_by_rank.items():
            if preds:
                top1[rank_name] = preds[0]["rank_value"]

        # Check genus -> family
        if "genus" in top1 and "family" in top1:
            expected_family = self._taxonomy.get("genus_to_family", {}).get(top1["genus"])
            agrees = expected_family == top1["family"] if expected_family else True
            result["family_genus_agree"] = agrees
            if not agrees:
                result["consistent"] = False
                result["notes"].append(
                    f"Genus '{top1['genus']}' belongs to family '{expected_family}', "
                    f"but family classifier predicted '{top1['family']}'"
                )

        # Check species -> genus
        if "species" in top1 and "genus" in top1:
            expected_genus = self._taxonomy.get("species_to_genus", {}).get(top1["species"])
            agrees = expected_genus == top1["genus"] if expected_genus else True
            result["species_genus_agree"] = agrees
            if not agrees:
                result["consistent"] = False
                result["notes"].append(
                    f"Species '{top1['species']}' belongs to genus '{expected_genus}', "
                    f"but genus classifier predicted '{top1['genus']}'"
                )

        # Check species -> family
        if "species" in top1 and "family" in top1:
            expected_family = self._taxonomy.get("species_to_family", {}).get(top1["species"])
            agrees = expected_family == top1["family"] if expected_family else True
            result["species_family_agree"] = agrees
            if not agrees:
                result["consistent"] = False
                result["notes"].append(
                    f"Species '{top1['species']}' belongs to family '{expected_family}', "
                    f"but family classifier predicted '{top1['family']}'"
                )

        return result

    # ------------------------------------------------------------------ #
    #  Input validation / skip
    # ------------------------------------------------------------------ #

    def validate_input(self, context: dict[str, Any]) -> list[str]:
        """Check that required context keys are present.

        Args:
            context: Pipeline context dictionary.

        Returns:
            List of error messages (empty if valid).
        """
        errors: list[str] = []
        if "image_paths" not in context or not context["image_paths"]:
            errors.append("Missing required 'image_paths' in context")
        return errors

    def skip(self, context: dict[str, Any]) -> StageResult:
        """Return a skipped result without performing inference.

        Args:
            context: Pipeline context dictionary (unused).

        Returns:
            StageResult marked as skipped.
        """
        return StageResult(stage_name=self.name, data={}, skipped=True)

    # ------------------------------------------------------------------ #
    #  Run (dispatcher)
    # ------------------------------------------------------------------ #

    def run(self, context: dict[str, Any]) -> StageResult:
        """Extract features, classify, return predictions.

        Dispatches to single-rank or multi-rank prediction based on how
        the classifier was fitted.

        Args:
            context: Must contain 'image_paths' (list of file paths).

        Returns:
            StageResult with predictions.
        """
        start = time.perf_counter()
        try:
            if not self._is_fitted:
                raise RuntimeError(
                    "Classifier not fitted. Call _fit_classifier() or "
                    "load_from_cache() before run()."
                )

            image_paths = context["image_paths"]
            paths = [Path(p) for p in image_paths]

            extractor = self._get_extractor()
            features = extractor.extract_from_paths(
                paths, normalize=True, show_progress=False
            )

            mean_features = features.mean(axis=0, keepdims=True)
            norm = np.linalg.norm(mean_features, axis=1, keepdims=True)
            if norm > 0:
                mean_features = mean_features / norm

            if self._is_multirank:
                data = self._run_multirank(mean_features, features, image_paths)
            else:
                data = self._run_single_rank(mean_features, features, image_paths)

            elapsed = (time.perf_counter() - start) * 1000
            return StageResult(
                stage_name=self.name,
                data=data,
                elapsed_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Classification failed: %s", exc)
            return StageResult(
                stage_name=self.name,
                data={},
                error=str(exc),
                elapsed_ms=elapsed,
            )

    # ------------------------------------------------------------------ #
    #  Single-rank prediction (legacy path)
    # ------------------------------------------------------------------ #

    def _run_single_rank(
        self,
        mean_features: npt.NDArray[np.float32],
        features: npt.NDArray[np.float32],
        image_paths: list,
    ) -> dict[str, Any]:
        """Run single-rank classification (legacy path)."""
        detail = self._classifier.predict_detail(mean_features)[0]

        top_indices = np.argsort(detail.probabilities)[::-1][: self._config.top_k]
        predictions = []
        for rank_pos, idx in enumerate(top_indices, start=1):
            label_id = self._classifier.label_decoder[int(idx)]
            label_name = self._label_names.get(label_id, f"class_{label_id}")
            predictions.append({
                "rank_value": label_name,
                "softmax_score": float(detail.probabilities[idx]),
                "rank_position": rank_pos,
                "l2_distance": float(detail.l2_distances[idx]),
                "cosine_similarity": float(detail.cosine_similarities[idx]),
            })

        sorted_probs = np.sort(detail.probabilities)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) >= 2 else 1.0

        per_image_predictions = []
        if len(features) > 1:
            per_image_details = self._classifier.predict_detail(features)
            for img_path, img_detail in zip(image_paths, per_image_details):
                top_idx = int(np.argmax(img_detail.probabilities))
                top_label_id = self._classifier.label_decoder[top_idx]
                per_image_predictions.append({
                    "image_path": str(img_path),
                    "top1_label": self._label_names.get(top_label_id, f"class_{top_label_id}"),
                    "top1_softmax_score": float(img_detail.probabilities[top_idx]),
                })

        knn_matches = self._classifier.find_nearest_support(mean_features, k=5)
        nearest_support = []
        for match in knn_matches[0]:
            entry: dict[str, Any] = {
                "label": self._label_names.get(match.label_id, f"class_{match.label_id}"),
                "l2_distance": match.l2_distance,
                "cosine_similarity": match.cosine_similarity,
            }
            if self._support_image_paths is not None:
                entry["image_path"] = str(self._support_image_paths[match.support_index])
            nearest_support.append(entry)

        return {
            "predictions": predictions,
            "margin": margin,
            "per_image_predictions": per_image_predictions,
            "nearest_support": nearest_support,
            "embedding_dim": int(features.shape[1]),
            "num_images_pooled": len(image_paths),
        }

    # ------------------------------------------------------------------ #
    #  Multi-rank prediction
    # ------------------------------------------------------------------ #

    def _run_multirank(
        self,
        mean_features: npt.NDArray[np.float32],
        features: npt.NDArray[np.float32],
        image_paths: list,
    ) -> dict[str, Any]:
        """Run multi-rank classification with consistency and OOD checks."""
        predictions_by_rank: dict[str, list[dict]] = {}
        margin_by_rank: dict[str, float] = {}

        for rank_name, classifier in self._classifiers.items():
            label_names = self._rank_label_names.get(rank_name, {})
            details = classifier.predict_detail(mean_features)
            detail = details[0]

            top_indices = np.argsort(detail.probabilities)[::-1][: self._config.top_k]
            predictions = []
            for rank_pos, idx in enumerate(top_indices, start=1):
                label_id = classifier.label_decoder[int(idx)]
                label_name = label_names.get(label_id, f"class_{label_id}")
                predictions.append({
                    "rank_value": label_name,
                    "softmax_score": float(detail.probabilities[idx]),
                    "rank_position": rank_pos,
                    "l2_distance": float(detail.l2_distances[idx]),
                    "cosine_similarity": float(detail.cosine_similarities[idx]),
                })

            predictions_by_rank[rank_name] = predictions

            sorted_probs = np.sort(detail.probabilities)[::-1]
            margin_by_rank[rank_name] = (
                float(sorted_probs[0] - sorted_probs[1])
                if len(sorted_probs) >= 2
                else 1.0
            )

        # Per-image predictions (use primary rank — first in classifiers)
        per_image_predictions = []
        if len(features) > 1:
            primary_rank = next(iter(self._classifiers))
            primary_clf = self._classifiers[primary_rank]
            primary_names = self._rank_label_names.get(primary_rank, {})
            per_image_details = primary_clf.predict_detail(features)
            for img_path, img_detail in zip(image_paths, per_image_details):
                top_idx = int(np.argmax(img_detail.probabilities))
                top_label_id = primary_clf.label_decoder[top_idx]
                per_image_predictions.append({
                    "image_path": str(img_path),
                    "top1_label": primary_names.get(top_label_id, f"class_{top_label_id}"),
                    "top1_softmax_score": float(img_detail.probabilities[top_idx]),
                })

        # k-NN nearest support (use primary rank)
        primary_rank = next(iter(self._classifiers))
        primary_clf = self._classifiers[primary_rank]
        primary_names = self._rank_label_names.get(primary_rank, {})
        knn_matches = primary_clf.find_nearest_support(mean_features, k=5)
        nearest_support = []
        for match in knn_matches[0]:
            entry: dict[str, Any] = {
                "label": primary_names.get(match.label_id, f"class_{match.label_id}"),
                "l2_distance": match.l2_distance,
                "cosine_similarity": match.cosine_similarity,
            }
            if self._support_image_paths is not None:
                entry["image_path"] = str(self._support_image_paths[match.support_index])
            nearest_support.append(entry)

        data: dict[str, Any] = {
            "predictions_by_rank": predictions_by_rank,
            "margin_by_rank": margin_by_rank,
            "per_image_predictions": per_image_predictions,
            "nearest_support": nearest_support,
            "embedding_dim": int(features.shape[1]),
            "num_images_pooled": len(image_paths),
        }

        # Hierarchical consistency check
        if self._taxonomy:
            data["hierarchical_consistency"] = self._check_hierarchical_consistency(
                predictions_by_rank
            )

        # Distance-based OOD confidence gate
        if self._distance_thresholds:
            flags = []
            gate: dict[str, Any] = {"flags": flags}
            for rank_name, preds in predictions_by_rank.items():
                if rank_name in self._distance_thresholds and preds:
                    top1_dist = preds[0]["l2_distance"]
                    threshold = self._distance_thresholds[rank_name]
                    in_dist = top1_dist <= threshold
                    gate[f"{rank_name}_in_distribution"] = in_dist
                    if not in_dist:
                        flags.append(f"{rank_name}_distance_exceeds_threshold")
            data["confidence_gate"] = gate

        return data
