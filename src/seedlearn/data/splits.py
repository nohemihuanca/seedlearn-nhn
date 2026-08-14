"""Dataset splitting, episode sampling, and k-shot validation.

This module merges the functionality of the original ``data/dataset.py`` and
``data/sampler.py`` into a single cohesive unit focused on creating and
managing stratified splits and few-shot episodes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from seedlearn.data.catalog import ImageRecord


# ---------------------------------------------------------------------------
# DatasetSplit
# ---------------------------------------------------------------------------


@dataclass
class DatasetSplit:
    """Represents train/val/test splits of a dataset.

    Attributes:
        train_indices: Indices of training samples.
        val_indices: Indices of validation samples.
        test_indices: Indices of test samples.
        label_to_id: Mapping from label strings to integer IDs.
        id_to_label: Mapping from integer IDs to label strings.
        num_classes: Number of unique classes.
        split_info: Metadata about the split (ratios, seed, etc.).
    """

    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]
    num_classes: int
    split_info: dict[str, any]


# ---------------------------------------------------------------------------
# Split creation helpers
# ---------------------------------------------------------------------------


def _compute_class_statistics(
    indices: np.ndarray,
    labels: np.ndarray,
    id_to_label: dict[int, str],
) -> dict:
    """Compute statistical summary of class distribution for a split.

    Args:
        indices: Indices of samples in this split.
        labels: Full array of all labels.
        id_to_label: Mapping from integer IDs to label strings.

    Returns:
        Dictionary with min/max/mean/median/std/class_counts.
    """
    split_labels = labels[indices]
    unique_labels, counts = np.unique(split_labels, return_counts=True)

    class_counts = {
        id_to_label[int(label)]: int(count)
        for label, count in zip(unique_labels, counts)
    }

    counts_array = counts.astype(float)

    return {
        "min_samples": int(counts_array.min()),
        "max_samples": int(counts_array.max()),
        "mean_samples": float(counts_array.mean()),
        "median_samples": float(np.median(counts_array)),
        "std_samples": float(counts_array.std()),
        "num_classes": len(unique_labels),
        "class_counts": class_counts,
    }


def _compute_feasible_k_shots(train_stats: dict) -> dict:
    """Determine feasible k-shot values for this split.

    Args:
        train_stats: Training set statistics from ``_compute_class_statistics``.

    Returns:
        Dictionary with guaranteed/recommended k-shot values and warning threshold.
    """
    min_train = train_stats["min_samples"]
    guaranteed = min_train

    recommended = []
    for k in [1, 5, 10, 20, 50]:
        if k <= guaranteed:
            recommended.append(k)

    warning_threshold = max(5, guaranteed)

    return {
        "guaranteed": guaranteed,
        "recommended": recommended if recommended else [guaranteed],
        "warning_threshold": warning_threshold,
    }


def _identify_problematic_classes(
    train_stats: dict,
    val_stats: dict,
    test_stats: dict,
    warning_threshold: int,
) -> list[dict]:
    """Identify classes with insufficient samples.

    Args:
        train_stats: Training set statistics.
        val_stats: Validation set statistics.
        test_stats: Test set statistics.
        warning_threshold: Threshold below which to warn.

    Returns:
        List of warnings for problematic classes.
    """
    warnings: list[dict] = []

    all_classes = set(train_stats["class_counts"].keys())
    all_classes.update(val_stats["class_counts"].keys())
    all_classes.update(test_stats["class_counts"].keys())

    for class_name in sorted(all_classes):
        train_count = train_stats["class_counts"].get(class_name, 0)
        val_count = val_stats["class_counts"].get(class_name, 0)
        test_count = test_stats["class_counts"].get(class_name, 0)

        if train_count < warning_threshold or train_count < 5:
            warnings.append(
                {
                    "class_name": class_name,
                    "train_samples": train_count,
                    "val_samples": val_count,
                    "test_samples": test_count,
                    "max_feasible_k": train_count,
                    "warning": f"Only {train_count} training samples - cannot support k>{train_count}",
                }
            )

    return warnings


def create_stratified_split(
    records: Sequence[ImageRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    min_samples_per_class: int = 3,
) -> DatasetSplit:
    """Create stratified train/validation/test splits.

    Args:
        records: List of ImageRecord objects.
        train_ratio: Proportion of data for training (default 0.70).
        val_ratio: Proportion of data for validation (default 0.15).
        test_ratio: Proportion of data for testing (default 0.15).
        random_seed: Random seed for reproducibility.
        min_samples_per_class: Minimum samples required per class (default 3).

    Returns:
        DatasetSplit object containing train/val/test indices and metadata.

    Raises:
        ValueError: If ratios don't sum to 1.0 or if any class has too few samples.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    labels = np.array([r.label_id for r in records])
    label_to_id = {r.label: r.label_id for r in records}
    id_to_label = {v: k for k, v in label_to_id.items()}
    num_classes = len(label_to_id)

    unique_labels, counts = np.unique(labels, return_counts=True)
    min_count = counts.min()
    required_samples = min_samples_per_class * 3

    if min_count < required_samples:
        logging.warning(
            "Some classes have fewer than %d total samples (min: %d). "
            "These classes may have <min_samples_per_class in some splits.",
            required_samples,
            min_count,
        )

    # First split: separate test set
    splitter_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_ratio,
        random_state=random_seed,
    )
    trainval_idx, test_idx = next(splitter_test.split(np.arange(len(labels)), labels))

    # Second split: separate train and val from the remaining data
    trainval_labels = labels[trainval_idx]
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)

    splitter_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio_adjusted,
        random_state=random_seed,
    )
    train_idx_rel, val_idx_rel = next(
        splitter_val.split(np.arange(len(trainval_labels)), trainval_labels)
    )

    train_idx = trainval_idx[train_idx_rel]
    val_idx = trainval_idx[val_idx_rel]

    # Compute split statistics
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    train_stats = _compute_class_statistics(train_idx, labels, id_to_label)
    val_stats = _compute_class_statistics(val_idx, labels, id_to_label)
    test_stats = _compute_class_statistics(test_idx, labels, id_to_label)

    feasible_k = _compute_feasible_k_shots(train_stats)
    class_warnings = _identify_problematic_classes(
        train_stats, val_stats, test_stats, feasible_k["warning_threshold"]
    )

    split_info = {
        "random_seed": random_seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "total_samples": len(records),
        "num_classes": num_classes,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "train_class_counts": {
            id_to_label[int(label)]: int(count)
            for label, count in zip(*np.unique(train_labels, return_counts=True))
        },
        "val_class_counts": {
            id_to_label[int(label)]: int(count)
            for label, count in zip(*np.unique(val_labels, return_counts=True))
        },
        "test_class_counts": {
            id_to_label[int(label)]: int(count)
            for label, count in zip(*np.unique(test_labels, return_counts=True))
        },
        "class_statistics": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
        "feasible_k_shots": feasible_k,
        "class_warnings": class_warnings,
    }

    logging.info(
        "Created stratified split: train=%d, val=%d, test=%d (%d classes)",
        len(train_idx),
        len(val_idx),
        len(test_idx),
        num_classes,
    )

    return DatasetSplit(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        num_classes=num_classes,
        split_info=split_info,
    )


def create_individual_split(
    records: Sequence[ImageRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> DatasetSplit:
    """Create train/val/test splits grouped by individual to prevent data leakage.

    All images belonging to the same individual (same ``individual_id``) are
    placed in the same split, preventing information leakage across train/test
    boundaries.

    Args:
        records: List of ImageRecord objects with individual_id populated.
        train_ratio: Proportion of data for training (default 0.70).
        val_ratio: Proportion of data for validation (default 0.15).
        test_ratio: Proportion of data for testing (default 0.15).
        random_seed: Random seed for reproducibility.

    Returns:
        DatasetSplit with indices respecting individual boundaries.

    Raises:
        ValueError: If ratios don't sum to 1.0 or records lack individual_id.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    groups = np.array([r.individual_id for r in records])
    if any(g == "" for g in groups):
        raise ValueError(
            "Records must have individual_id populated for individual-level splits"
        )

    labels = np.array([r.label_id for r in records])
    label_to_id = {r.label: r.label_id for r in records}
    id_to_label = {v: k for k, v in label_to_id.items()}
    num_classes = len(label_to_id)

    # Pass 1: split off test set (test_ratio of individuals)
    splitter_test = GroupShuffleSplit(
        n_splits=1,
        test_size=test_ratio,
        random_state=random_seed,
    )
    trainval_idx, test_idx = next(
        splitter_test.split(np.arange(len(labels)), labels, groups)
    )

    # Pass 2: split remaining into train/val (adjusted val_ratio)
    trainval_labels = labels[trainval_idx]
    trainval_groups = groups[trainval_idx]
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)

    splitter_val = GroupShuffleSplit(
        n_splits=1,
        test_size=val_ratio_adjusted,
        random_state=random_seed,
    )
    train_idx_rel, val_idx_rel = next(
        splitter_val.split(
            np.arange(len(trainval_labels)), trainval_labels, trainval_groups
        )
    )

    train_idx = trainval_idx[train_idx_rel]
    val_idx = trainval_idx[val_idx_rel]

    # Compute individual counts per split
    train_individuals = set(groups[train_idx])
    val_individuals = set(groups[val_idx])
    test_individuals = set(groups[test_idx])

    # Compute split statistics
    train_stats = _compute_class_statistics(train_idx, labels, id_to_label)
    val_stats = _compute_class_statistics(val_idx, labels, id_to_label)
    test_stats = _compute_class_statistics(test_idx, labels, id_to_label)

    feasible_k = _compute_feasible_k_shots(train_stats)
    class_warnings = _identify_problematic_classes(
        train_stats, val_stats, test_stats, feasible_k["warning_threshold"]
    )

    split_info = {
        "split_type": "individual",
        "random_seed": random_seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "total_samples": len(records),
        "num_classes": num_classes,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "total_individuals": len(set(groups)),
        "train_individuals": len(train_individuals),
        "val_individuals": len(val_individuals),
        "test_individuals": len(test_individuals),
        "class_statistics": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
        "feasible_k_shots": feasible_k,
        "class_warnings": class_warnings,
    }

    logging.info(
        "Created individual-level split: train=%d (%d indiv), val=%d (%d indiv), "
        "test=%d (%d indiv), %d classes",
        len(train_idx),
        len(train_individuals),
        len(val_idx),
        len(val_individuals),
        len(test_idx),
        len(test_individuals),
        num_classes,
    )

    return DatasetSplit(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        num_classes=num_classes,
        split_info=split_info,
    )


def save_split(split: DatasetSplit, output_path: Path) -> None:
    """Save dataset split to disk.

    Args:
        split: DatasetSplit object to save.
        output_path: Path to save the split (will create .npz and .json files).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    indices_path = output_path.with_suffix(".npz")
    np.savez(
        indices_path,
        train_indices=split.train_indices,
        val_indices=split.val_indices,
        test_indices=split.test_indices,
    )

    metadata_path = output_path.with_suffix(".json")
    metadata = {
        "label_to_id": split.label_to_id,
        "id_to_label": {str(k): v for k, v in split.id_to_label.items()},
        "num_classes": split.num_classes,
        "split_info": split.split_info,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Saved split to %s", output_path.parent)


def load_split(split_path: Path) -> DatasetSplit:
    """Load dataset split from disk.

    Args:
        split_path: Path to the split files (without extension).

    Returns:
        DatasetSplit object.

    Raises:
        FileNotFoundError: If split files don't exist.
    """
    indices_path = split_path.with_suffix(".npz")
    metadata_path = split_path.with_suffix(".json")

    if not indices_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Split files not found at {split_path}")

    data = np.load(indices_path)
    train_indices = data["train_indices"]
    val_indices = data["val_indices"]
    test_indices = data["test_indices"]

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    id_to_label = {int(k): v for k, v in metadata["id_to_label"].items()}

    logging.info("Loaded split from %s", split_path.parent)

    return DatasetSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        label_to_id=metadata["label_to_id"],
        id_to_label=id_to_label,
        num_classes=metadata["num_classes"],
        split_info=metadata["split_info"],
    )


def sample_individual_from_split(
    records: Sequence[ImageRecord],
    split: DatasetSplit,
    partition: str = "test",
    seed: int | None = None,
) -> tuple[str, int]:
    """Sample a random individual ID from a split partition.

    Selects one individual uniformly at random from the test or validation
    partition. Never returns a training individual.

    Args:
        records: Full list of ImageRecord objects (same order used to create split).
        split: Loaded DatasetSplit object.
        partition: Which partition to sample from ("test" or "val").
        seed: Random seed for reproducibility. If None, a random seed is generated
            and returned so the selection can be reproduced.

    Returns:
        Tuple of (individual_id, seed_used).

    Raises:
        ValueError: If partition is not "test" or "val".
    """
    if partition not in ("test", "val"):
        raise ValueError(f"Invalid partition '{partition}'. Must be 'test' or 'val'.")

    if partition == "test":
        indices = split.test_indices
    else:
        indices = split.val_indices

    individuals = sorted(set(records[i].individual_id for i in indices))

    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))

    rng = np.random.default_rng(seed)
    chosen = rng.choice(individuals)

    return str(chosen), seed


# ---------------------------------------------------------------------------
# Few-shot episode sampling
# ---------------------------------------------------------------------------


@dataclass
class FewShotEpisode:
    """Represents a single few-shot learning episode.

    Attributes:
        support_indices: Indices of support set samples.
        query_indices: Indices of query set samples.
        support_labels: Labels for support samples.
        query_labels: Labels for query samples.
        classes: List of class IDs included in this episode.
        n_way: Number of classes in the episode.
        k_shot: Number of support examples per class.
        n_query: Number of query examples per class.
    """

    support_indices: np.ndarray
    query_indices: np.ndarray
    support_labels: np.ndarray
    query_labels: np.ndarray
    classes: np.ndarray
    n_way: int
    k_shot: int
    n_query: int


class NShotSampler:
    """Sampler for creating n-way k-shot episodes.

    This class handles the creation of few-shot learning episodes by
    randomly sampling support and query sets for each class.
    """

    def __init__(
        self,
        records: Sequence[ImageRecord],
        n_way: int | None = None,
        k_shot: int = 5,
        n_query: int = 15,
        random_seed: int | None = None,
    ) -> None:
        """Initialize the n-shot sampler.

        Args:
            records: List of ImageRecord objects to sample from.
            n_way: Number of classes per episode (None = use all classes).
            k_shot: Number of support examples per class.
            n_query: Number of query examples per class.
            random_seed: Random seed for reproducibility.

        Raises:
            ValueError: If k_shot + n_query exceeds available samples for any class.
        """
        self.records = records
        self.k_shot = k_shot
        self.n_query = n_query
        self.rng = np.random.default_rng(random_seed)

        # Group indices by class
        self.class_to_indices: dict[int, np.ndarray] = {}
        for idx, record in enumerate(records):
            if record.label_id not in self.class_to_indices:
                self.class_to_indices[record.label_id] = []
            self.class_to_indices[record.label_id].append(idx)

        for label_id in self.class_to_indices:
            self.class_to_indices[label_id] = np.array(self.class_to_indices[label_id])

        self.all_classes = np.array(sorted(self.class_to_indices.keys()))
        self.num_classes = len(self.all_classes)

        if n_way is None:
            self.n_way = self.num_classes
        else:
            self.n_way = min(n_way, self.num_classes)

        min_required = k_shot + n_query
        for label_id, indices in self.class_to_indices.items():
            if len(indices) < min_required:
                raise ValueError(
                    f"Class {label_id} has only {len(indices)} samples, "
                    f"but {min_required} required (k_shot={k_shot} + n_query={n_query})"
                )

        logging.info(
            "Initialized NShotSampler: %d classes, %d-way, %d-shot, %d-query",
            self.num_classes,
            self.n_way,
            self.k_shot,
            self.n_query,
        )

    def sample_episode(self) -> FewShotEpisode:
        """Sample a single few-shot learning episode.

        Returns:
            FewShotEpisode object containing support and query sets.
        """
        if self.n_way == self.num_classes:
            sampled_classes = self.all_classes
        else:
            sampled_classes = self.rng.choice(
                self.all_classes, size=self.n_way, replace=False
            )

        support_indices = []
        query_indices = []
        support_labels = []
        query_labels = []

        for class_id in sampled_classes:
            class_indices = self.class_to_indices[class_id]
            shuffled_indices = self.rng.permutation(class_indices)
            support_idx = shuffled_indices[: self.k_shot]
            query_idx = shuffled_indices[self.k_shot : self.k_shot + self.n_query]

            support_indices.extend(support_idx)
            query_indices.extend(query_idx)
            support_labels.extend([class_id] * self.k_shot)
            query_labels.extend([class_id] * self.n_query)

        return FewShotEpisode(
            support_indices=np.array(support_indices),
            query_indices=np.array(query_indices),
            support_labels=np.array(support_labels),
            query_labels=np.array(query_labels),
            classes=sampled_classes,
            n_way=self.n_way,
            k_shot=self.k_shot,
            n_query=self.n_query,
        )

    def sample_episodes(self, num_episodes: int) -> list[FewShotEpisode]:
        """Sample multiple few-shot learning episodes.

        Args:
            num_episodes: Number of episodes to sample.

        Returns:
            List of FewShotEpisode objects.
        """
        return [self.sample_episode() for _ in range(num_episodes)]


def create_fixed_support_set(
    records: Sequence[ImageRecord],
    k_shot: int,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a fixed support set with k examples per class.

    Args:
        records: List of ImageRecord objects.
        k_shot: Number of support examples per class.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (support_indices, support_labels).

    Raises:
        ValueError: If any class has fewer than k_shot samples.
    """
    rng = np.random.default_rng(random_seed)

    class_to_indices: dict[int, list[int]] = {}
    for idx, record in enumerate(records):
        if record.label_id not in class_to_indices:
            class_to_indices[record.label_id] = []
        class_to_indices[record.label_id].append(idx)

    support_indices = []
    support_labels = []

    for class_id in sorted(class_to_indices.keys()):
        indices = np.array(class_to_indices[class_id])

        if len(indices) < k_shot:
            raise ValueError(
                f"Class {class_id} has only {len(indices)} samples, "
                f"but {k_shot} requested for support set"
            )

        sampled_indices = rng.choice(indices, size=k_shot, replace=False)

        support_indices.extend(sampled_indices)
        support_labels.extend([class_id] * k_shot)

    return np.array(support_indices), np.array(support_labels)


def validate_k_shot_feasibility(
    split_info: dict,
    k_shot: int,
    strict: bool = True,
) -> tuple[bool, list[str]]:
    """Validate if k-shot is feasible for this split.

    Args:
        split_info: Split metadata dictionary (from split JSON file).
        k_shot: Desired k-shot value.
        strict: If True, fail if any class can't support k-shot.

    Returns:
        Tuple of (is_valid, messages).
    """
    messages: list[str] = []

    if "feasible_k_shots" not in split_info:
        msg = (
            "WARNING: Split metadata doesn't contain feasibility info. "
            "This split was created before enhanced validation was added. "
            "Consider recreating splits for full validation support."
        )
        messages.append(msg)
        return (not strict, messages)

    feasible = split_info["feasible_k_shots"]
    guaranteed = feasible["guaranteed"]
    recommended = feasible.get("recommended", [])

    if k_shot > guaranteed:
        messages.append(
            f"ERROR: k={k_shot} exceeds guaranteed threshold ({guaranteed}). "
            f"Some classes have fewer than {k_shot} training samples."
        )

        if "class_warnings" in split_info and split_info["class_warnings"]:
            problematic = [
                w for w in split_info["class_warnings"] if w["train_samples"] < k_shot
            ]

            if problematic:
                messages.append(f"  Problematic classes ({len(problematic)}):")
                for warning in problematic[:5]:
                    messages.append(
                        f"    - {warning['class_name']}: "
                        f"only {warning['train_samples']} train samples"
                    )
                if len(problematic) > 5:
                    messages.append(f"    ... and {len(problematic) - 5} more")

        if recommended:
            messages.append(f"  Recommended k-shot values: {recommended}")
        else:
            messages.append(f"  Try k-shot <= {guaranteed}")

        messages.append("")
        messages.append("Options:")
        messages.append("  1. Use a smaller k-shot value (see recommendations above)")
        messages.append("  2. Recreate splits with different train/val/test ratios")
        messages.append("  3. Filter out rare classes from the dataset")
        messages.append(
            "  4. Use --skip-validation to proceed anyway (not recommended)"
        )

        return (False, messages)

    if k_shot in recommended:
        messages.append(
            f"k={k_shot} is feasible and recommended (guaranteed: {guaranteed})"
        )
    else:
        messages.append(
            f"k={k_shot} is feasible (guaranteed: {guaranteed}, recommended: {recommended})"
        )

    if "class_warnings" in split_info and split_info["class_warnings"]:
        num_warnings = len(split_info["class_warnings"])
        if num_warnings > 0:
            messages.append(
                f"  Note: {num_warnings} classes have <5 training samples, "
                f"but all have >= {k_shot} samples."
            )

    return (True, messages)
