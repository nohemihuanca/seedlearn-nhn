"""Feature extraction utilities for few-shot learning.

This module provides utilities for extracting image embeddings using BioClip2
and caching them to disk for efficient reuse across experiments.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from seedlearn.data.catalog import ImageRecord
from seedlearn.data.loader import ImageRecordDataset, create_dataloader


def _batched(items: Sequence, batch_size: int):
    """Batch items into chunks of batch_size.

    Args:
        items: Sequence to batch.
        batch_size: Size of each batch.

    Yields:
        Batches of items.
    """
    it = iter(items)
    while (batch := list(itertools.islice(it, batch_size))):
        yield batch


class FeatureExtractor:
    """Extract image embeddings using BioClip2.

    This class handles batch processing and provides an interface for
    extracting features from images.
    """

    def __init__(
        self,
        device: torch.device | str = "cuda",
        batch_size: int = 256,
        model_str: str = "hf-hub:imageomics/bioclip-2",
    ) -> None:
        """Initialize the feature extractor.

        Args:
            device: Torch device to use for computation.
            batch_size: Batch size for feature extraction.
            model_str: BioClip model identifier.
        """
        from bioclip.predict import BaseClassifier

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.classifier = BaseClassifier(device=self.device, model_str=model_str)
        logging.info("Initialized FeatureExtractor on device: %s", self.device)

    def extract_from_images(
        self,
        images: Sequence[Image.Image],
        normalize: bool = True,
        show_progress: bool = True,
    ) -> npt.NDArray[np.float32]:
        """Extract features from PIL images.

        Args:
            images: List of PIL Image objects.
            normalize: Whether to L2-normalize features.
            show_progress: Whether to show progress bar.

        Returns:
            Feature array of shape (num_images, feature_dim).
        """
        all_features = []

        iterator = _batched(images, self.batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=(len(images) + self.batch_size - 1) // self.batch_size,
                desc="Extracting features",
                unit="batch",
            )

        for batch in iterator:
            features = self.classifier.create_image_features(batch, normalize=normalize)
            all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    def extract_from_paths(
        self,
        image_paths: Sequence[Path],
        normalize: bool = True,
        show_progress: bool = True,
    ) -> npt.NDArray[np.float32]:
        """Extract features from image file paths.

        Args:
            image_paths: List of paths to image files.
            normalize: Whether to L2-normalize features.
            show_progress: Whether to show progress bar.

        Returns:
            Feature array of shape (num_images, feature_dim).
        """
        all_features = []

        iterator = _batched(image_paths, self.batch_size)
        if show_progress:
            total_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size
            iterator = tqdm(
                iterator,
                total=total_batches,
                desc="Extracting features",
                unit="batch",
            )

        for batch_paths in iterator:
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            features = self.classifier.create_image_features(images, normalize=normalize)
            all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    def extract_from_records(
        self,
        records: Sequence[ImageRecord],
        normalize: bool = True,
        show_progress: bool = True,
    ) -> npt.NDArray[np.float32]:
        """Extract features from ImageRecord objects.

        Args:
            records: List of ImageRecord objects.
            normalize: Whether to L2-normalize features.
            show_progress: Whether to show progress bar.

        Returns:
            Feature array of shape (num_records, feature_dim).
        """
        image_paths = [r.image_path for r in records]
        return self.extract_from_paths(image_paths, normalize, show_progress)

    def extract_from_records_optimized(
        self,
        records: Sequence[ImageRecord],
        normalize: bool = True,
        show_progress: bool = True,
        num_workers: int = 8,
        prefetch_factor: int = 2,
    ) -> npt.NDArray[np.float32]:
        """Extract features using optimized PyTorch DataLoader.

        Args:
            records: List of ImageRecord objects.
            normalize: Whether to L2-normalize features.
            show_progress: Whether to show progress bar.
            num_workers: Number of parallel data loading workers.
            prefetch_factor: Number of batches to prefetch per worker.

        Returns:
            Feature array of shape (num_records, feature_dim).
        """
        dataset = ImageRecordDataset(
            records=records,
            transform=self.classifier.preprocess,
        )

        dataloader = create_dataloader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
            shuffle=False,
            drop_last=False,
        )

        all_features = []
        all_labels = []

        iterator = dataloader
        if show_progress:
            iterator = tqdm(
                dataloader,
                desc="Extracting features (optimized)",
                unit="batch",
            )

        with torch.no_grad():
            for batch_images, batch_labels in iterator:
                batch_images = batch_images.to(self.device, non_blocking=True)
                features = self.classifier.model.encode_image(batch_images)

                if normalize:
                    features = F.normalize(features, dim=-1)

                all_features.append(features.cpu())
                all_labels.append(batch_labels)

                if show_progress and hasattr(iterator, "n") and iterator.n % 10 == 0:
                    if self.device.type == "cuda":
                        allocated = torch.cuda.memory_allocated(self.device) / 1e9
                        reserved = torch.cuda.memory_reserved(self.device) / 1e9
                        logging.debug(
                            f"GPU Memory: {allocated:.2f}GB allocated, "
                            f"{reserved:.2f}GB reserved"
                        )

        features = torch.cat(all_features, dim=0).numpy()

        labels = torch.cat(all_labels, dim=0).numpy()
        expected_labels = np.array([r.label_id for r in records])
        if not np.array_equal(labels, expected_labels):
            logging.warning("Label order mismatch - DataLoader may have shuffled data")

        return features
