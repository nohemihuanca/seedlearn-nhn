"""Data loading, splitting, and dataset utilities for seedlearn."""

from seedlearn.data.catalog import (
    ImageRecord,
    format_label,
    iter_image_paths,
    load_catalog,
    load_dataset,
    load_image,
)
from seedlearn.data.constants import (
    DEFAULT_CATALOG,
    IMAGE_EXTENSIONS,
    RANK_COLUMN_MAP,
    SHARED_DATA,
    SHARED_EMBEDDINGS,
    SHARED_EXPERIMENTS,
    SHARED_SPLITS,
    get_catalog_version,
    get_optimal_batch_size,
)
from seedlearn.data.loader import (
    ImagePathDataset,
    ImageRecordDataset,
    create_dataloader,
)
from seedlearn.data.splits import (
    DatasetSplit,
    FewShotEpisode,
    NShotSampler,
    create_fixed_support_set,
    create_individual_split,
    create_stratified_split,
    load_split,
    save_split,
    validate_k_shot_feasibility,
)

__all__ = [
    # catalog
    "ImageRecord",
    "format_label",
    "iter_image_paths",
    "load_catalog",
    "load_dataset",
    "load_image",
    # constants
    "DEFAULT_CATALOG",
    "IMAGE_EXTENSIONS",
    "RANK_COLUMN_MAP",
    "SHARED_DATA",
    "SHARED_EMBEDDINGS",
    "SHARED_EXPERIMENTS",
    "SHARED_SPLITS",
    "get_catalog_version",
    "get_optimal_batch_size",
    # loader
    "ImagePathDataset",
    "ImageRecordDataset",
    "create_dataloader",
    # splits
    "DatasetSplit",
    "FewShotEpisode",
    "NShotSampler",
    "create_fixed_support_set",
    "create_individual_split",
    "create_stratified_split",
    "load_split",
    "save_split",
    "validate_k_shot_feasibility",
]
