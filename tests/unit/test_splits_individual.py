"""Tests for individual-level data splits."""

import numpy as np
import pytest

from seedlearn.data.catalog import ImageRecord
from seedlearn.data.splits import create_individual_split, sample_individual_from_split


@pytest.fixture
def records_with_individuals():
    """Create records where each individual has multiple images."""
    records = []
    label_id = 0
    for family_idx in range(3):  # 3 families
        family = f"Family_{family_idx}"
        for indiv_idx in range(10):  # 10 individuals per family
            individual_id = f"PP{family_idx * 10 + indiv_idx:03d}"
            for img_idx in range(5):  # 5 images per individual
                records.append(
                    ImageRecord(
                        image_path=f"/fake/{family}/{individual_id}_{img_idx:03d}.jpg",
                        label=family,
                        family=family,
                        genus=f"Genus_{family_idx}",
                        species=f"species_{family_idx}",
                        label_id=label_id,
                        individual_id=individual_id,
                    )
                )
        label_id += 1
    return records  # 3 * 10 * 5 = 150 records


class TestIndividualSplit:
    def test_no_individual_leakage(self, records_with_individuals):
        """All images of an individual must be in the same split."""
        split = create_individual_split(records_with_individuals, random_seed=42)

        train_ids = {
            records_with_individuals[i].individual_id for i in split.train_indices
        }
        val_ids = {records_with_individuals[i].individual_id for i in split.val_indices}
        test_ids = {
            records_with_individuals[i].individual_id for i in split.test_indices
        }

        assert train_ids.isdisjoint(val_ids), "Train/val individual overlap!"
        assert train_ids.isdisjoint(test_ids), "Train/test individual overlap!"
        assert val_ids.isdisjoint(test_ids), "Val/test individual overlap!"

    def test_all_images_covered(self, records_with_individuals):
        """All image indices must be present across the three splits."""
        split = create_individual_split(records_with_individuals, random_seed=42)
        all_indices = (
            set(split.train_indices) | set(split.val_indices) | set(split.test_indices)
        )
        assert all_indices == set(range(len(records_with_individuals)))

    def test_approximate_ratios(self, records_with_individuals):
        """Splits should approximately match 70/15/15 at individual level."""
        split = create_individual_split(records_with_individuals, random_seed=42)
        n = len(records_with_individuals)
        train_frac = len(split.train_indices) / n
        assert 0.55 < train_frac < 0.85

    def test_reproducible(self, records_with_individuals):
        """Same seed must produce identical splits."""
        s1 = create_individual_split(records_with_individuals, random_seed=42)
        s2 = create_individual_split(records_with_individuals, random_seed=42)
        np.testing.assert_array_equal(
            sorted(s1.train_indices), sorted(s2.train_indices)
        )
        np.testing.assert_array_equal(sorted(s1.test_indices), sorted(s2.test_indices))

    def test_different_seeds_differ(self, records_with_individuals):
        """Different seeds should produce different splits."""
        s1 = create_individual_split(records_with_individuals, random_seed=42)
        s2 = create_individual_split(records_with_individuals, random_seed=99)
        assert set(s1.test_indices) != set(s2.test_indices)

    def test_missing_individual_id_raises(self):
        """Records without individual_id should raise ValueError."""
        records = [
            ImageRecord(
                image_path="/fake.jpg",
                label="test",
                family="F",
                genus="G",
                species="S",
                label_id=0,
            )
        ]
        with pytest.raises(ValueError, match="individual_id"):
            create_individual_split(records)

    def test_split_info_contains_individual_counts(self, records_with_individuals):
        """split_info must report individual counts per split."""
        split = create_individual_split(records_with_individuals, random_seed=42)
        info = split.split_info
        assert info["split_type"] == "individual"
        assert info["total_individuals"] == 30
        assert (
            info["train_individuals"]
            + info["val_individuals"]
            + info["test_individuals"]
        ) == 30

    def test_no_duplicate_indices(self, records_with_individuals):
        """No index should appear in more than one split."""
        split = create_individual_split(records_with_individuals, random_seed=42)
        train_set = set(split.train_indices)
        val_set = set(split.val_indices)
        test_set = set(split.test_indices)
        assert len(train_set) + len(val_set) + len(test_set) == len(
            train_set | val_set | test_set
        )

    def test_invalid_ratios_raises(self, records_with_individuals):
        """Ratios not summing to 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            create_individual_split(
                records_with_individuals,
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3,
            )


class TestSampleIndividualFromSplit:
    def test_returns_individual_in_partition(self, records_with_individuals):
        """Sampled individual must belong to the requested partition."""
        split = create_individual_split(records_with_individuals, random_seed=42)
        test_individuals = {
            records_with_individuals[i].individual_id for i in split.test_indices
        }
        individual_id, _ = sample_individual_from_split(
            records_with_individuals,
            split,
            partition="test",
            seed=1,
        )
        assert individual_id in test_individuals

    def test_never_returns_training_individual(self, records_with_individuals):
        """Sampled individual must never be from the training set."""
        split = create_individual_split(records_with_individuals, random_seed=42)
        train_individuals = {
            records_with_individuals[i].individual_id for i in split.train_indices
        }
        for seed in range(50):
            individual_id, _ = sample_individual_from_split(
                records_with_individuals,
                split,
                partition="test",
                seed=seed,
            )
            assert individual_id not in train_individuals

    def test_val_partition(self, records_with_individuals):
        """Sampling from val partition returns a val individual."""
        split = create_individual_split(records_with_individuals, random_seed=42)
        val_individuals = {
            records_with_individuals[i].individual_id for i in split.val_indices
        }
        individual_id, _ = sample_individual_from_split(
            records_with_individuals,
            split,
            partition="val",
            seed=1,
        )
        assert individual_id in val_individuals

    def test_reproducible_with_seed(self, records_with_individuals):
        """Same seed must return same individual."""
        split = create_individual_split(records_with_individuals, random_seed=42)
        id1, seed1 = sample_individual_from_split(
            records_with_individuals,
            split,
            partition="test",
            seed=123,
        )
        id2, seed2 = sample_individual_from_split(
            records_with_individuals,
            split,
            partition="test",
            seed=123,
        )
        assert id1 == id2
        assert seed1 == seed2

    def test_no_seed_returns_used_seed(self, records_with_individuals):
        """When no seed given, the returned seed should reproduce the result."""
        split = create_individual_split(records_with_individuals, random_seed=42)
        id1, used_seed = sample_individual_from_split(
            records_with_individuals,
            split,
            partition="test",
        )
        id2, _ = sample_individual_from_split(
            records_with_individuals,
            split,
            partition="test",
            seed=used_seed,
        )
        assert id1 == id2

    def test_invalid_partition_raises(self, records_with_individuals):
        """Invalid partition name must raise ValueError."""
        split = create_individual_split(records_with_individuals, random_seed=42)
        with pytest.raises(ValueError, match="partition"):
            sample_individual_from_split(
                records_with_individuals,
                split,
                partition="train",
            )
