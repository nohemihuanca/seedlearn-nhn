"""PyTorch Dataset for efficient image loading with parallel workers.

This module provides a Dataset implementation that enables parallel loading
and preprocessing of images, optimized for GPU feature extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ImagePathDataset(Dataset):
    """PyTorch Dataset for loading images from file paths.

    Attributes:
        image_paths: List of paths to images.
        transform: Preprocessing transform to apply to each image.
        labels: Optional list of labels for each image.
    """

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        transform: Callable | None = None,
        labels: List[int] | None = None,
    ):
        """Initialize the dataset.

        Args:
            image_paths: List of paths to image files.
            transform: Callable that takes a PIL image and returns a tensor.
            labels: Optional list of labels corresponding to each image.
        """
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform
        self.labels = labels

        if labels is not None and len(labels) != len(image_paths):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match "
                f"number of images ({len(image_paths)})"
            )

        logger.debug(f"Created dataset with {len(self)} images")

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, tuple]:
        """Load and preprocess an image.

        Args:
            idx: Index of the image to load.

        Returns:
            If labels are provided: (image_tensor, label)
            Otherwise: image_tensor
        """
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform is not None:
            image = self.transform(image)
        else:
            import torchvision.transforms as T
            image = T.ToTensor()(image)

        if self.labels is not None:
            return image, self.labels[idx]
        return image


class ImageRecordDataset(Dataset):
    """PyTorch Dataset for loading images from ImageRecord objects.

    Attributes:
        records: List of ImageRecord objects.
        transform: Preprocessing transform to apply to each image.
    """

    def __init__(
        self,
        records: List,
        transform: Callable | None = None,
    ):
        """Initialize the dataset.

        Args:
            records: List of ImageRecord objects (dataclass instances).
            transform: Callable that takes a PIL image and returns a tensor.
        """
        self.records = records
        self.transform = transform

        self.image_paths = [Path(r.image_path) for r in records]
        self.labels = [r.label_id for r in records]

        logger.debug(
            f"Created dataset with {len(self)} images "
            f"across {len(set(self.labels))} classes"
        )

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple:
        """Load and preprocess an image with its label.

        Args:
            idx: Index of the image to load.

        Returns:
            Tuple of (image_tensor, label).
        """
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform is not None:
            image = self.transform(image)
        else:
            import torchvision.transforms as T
            image = T.ToTensor()(image)

        return image, self.labels[idx]


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    shuffle: bool = False,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    """Create an optimized DataLoader for GPU feature extraction.

    Args:
        dataset: PyTorch Dataset to load from.
        batch_size: Number of samples per batch.
        num_workers: Number of parallel data loading workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        prefetch_factor: Number of batches to prefetch per worker.
        persistent_workers: Whether to keep workers alive between epochs.
        shuffle: Whether to shuffle data.
        drop_last: Whether to drop incomplete last batch.

    Returns:
        Configured DataLoader.
    """
    if not torch.cuda.is_available():
        pin_memory = False
        num_workers = min(num_workers, 4)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=drop_last,
    )

    logger.info(
        f"Created DataLoader: batch_size={batch_size}, "
        f"num_workers={num_workers}, pin_memory={pin_memory}"
    )

    return loader
