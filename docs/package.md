# Package Reference — `seedlearn`

**Navigation**: [SeedLearn](../README.md) > **Package Reference**

---

## Overview

`seedlearn` is an installable Python package (`pip install -e .`) providing five modules:

| Module | Purpose |
|--------|---------|
| `seedlearn.data` | Image loading, catalog I/O, data partitioning, few-shot sampling |
| `seedlearn.clip` | BioCLIP 2 feature extraction, caching, SimpleShot classifier, metrics |
| `seedlearn.pipeline` | 5-stage classification pipeline (config, stages, RAG, vision-LLM client) |
| `seedlearn.components.analyzers` | Vision-LLM prompt templates for morphological trait extraction |
| `seedlearn.reporting` | Plotly-based HTML report generation |

> **Note**: The `seedlearn.pipeline` module has its own dedicated reference at [Pipeline Reference](pipeline.md) with full architecture, per-stage technical details, config options, and design rationale. This page covers only the data, clip, analyzers, and reporting modules.

```
src/seedlearn/
├── __init__.py              # __version__ = "0.1.0"
├── data/
│   ├── constants.py         # Paths, rank maps, batch sizing
│   ├── catalog.py           # ImageRecord, load_dataset, load_catalog
│   ├── loader.py            # PyTorch Dataset + DataLoader
│   └── splits.py            # Stratified splits, individual splits, few-shot episodes
├── clip/
│   ├── encoder.py           # FeatureExtractor (BioCLIP 2)
│   ├── cache.py             # CachedFeatureExtractor (.npz caching)
│   ├── simpleshot.py        # SimpleShot classifier + l2_normalize
│   └── metrics.py           # EvaluationResult, compute_metrics
├── pipeline/                # → See docs/pipeline.md for full reference
│   ├── config.py            # PipelineConfig + 6 sub-configs + YAML loader
│   ├── protocol.py          # StageResult + PipelineStage Protocol
│   ├── result.py            # PipelineResult container
│   ├── vlm_client.py        # InferenceClient + JSON recovery + message building
│   ├── rag.py               # RAGIndex (FAISS + sentence-transformers)
│   └── stages/              # morphology, classification, trait_retrieval,
│                            #   evidence, reasoning
├── components/analyzers/
│   └── prompts.py           # PromptStyle enum, 24-trait assessment form
└── reporting/
    ├── html.py              # Charts, HTML templates, interpretation text (SimpleShot)
    └── pipeline_html.py     # Pipeline per-specimen visual report (5-stage)
```

---

## `seedlearn.data`

### Constants (`data.constants`)

```python
RANK_COLUMN_MAP: dict[str, str]
# {"family": "FAMILY", "genus": "GENUS", "species": "SPECIES"}

IMAGE_EXTENSIONS: set[str]
# {".jpg", ".jpeg", ".png", ".webp"}

SHARED_EXPERIMENTS: Path
# /nfs/roberts/project/pi_lsc4/shared/seedlearn/data/experiments/simpleshot

DEFAULT_CATALOG: str
# /nfs/.../species_catalog_v2026-01-29_12K_20260129_123334.csv
```

```python
SHARED_EMBEDDINGS: Path
# /nfs/roberts/project/pi_lsc4/shared/seedlearn/data/embeddings

SHARED_SPLITS: Path
# /nfs/roberts/project/pi_lsc4/shared/seedlearn/data/splits

def get_catalog_version(catalog_path: Path) -> str:
    """Extract version string from catalog filename via regex.
    e.g. 'species_catalog_v2026-01-29_12K_...' → '2026-01-29_v2026-01-29_12K'"""

def get_optimal_batch_size(device: torch.device) -> int:
    """Auto-select batch size by GPU tier (H200: 2048, A6000: 1024, etc.)."""
```

### Catalog & Loading (`data.catalog`)

```python
@dataclass
class ImageRecord:
    image_path: Path      # Absolute path to image
    label: str            # Label for the specified rank
    family: str
    genus: str
    species: str          # Species binomial
    label_id: int = -1    # Integer ID assigned during dataset creation
    individual_id: str = ""  # Plant individual ID (ID_YPS) for group-aware splitting
```

```python
def load_catalog(catalog_path: Path) -> pd.DataFrame:
    """Load and validate species catalog CSV."""

def load_dataset(
    catalog_path: Path, rank: str = "species"
) -> tuple[list[ImageRecord], dict[str, int]]:
    """Load dataset from catalog. Returns (records, label_to_id mapping).
    Filters out individuals with missing directories."""

def load_image(image_path: Path) -> Image.Image:
    """Load image as PIL RGB."""

def format_label(row: pd.Series, rank: str) -> str:
    """Format taxonomic label for rank (handles case and underscores)."""

def iter_image_paths(directory: Path) -> Iterable[Path]:
    """Yield image file paths with valid extensions from directory."""
```

### PyTorch Datasets (`data.loader`)

```python
class ImagePathDataset(Dataset):
    """Dataset from file paths. Returns (image_tensor,) or (image_tensor, label)."""
    def __init__(self, image_paths, transform=None, labels=None): ...

class ImageRecordDataset(Dataset):
    """Dataset from ImageRecord objects. Returns (image_tensor, label_id)."""
    def __init__(self, records, transform=None): ...

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Create optimized DataLoader. Auto-adjusts for CPU (caps workers at 4)."""
```

### Partitioning & Sampling (`data.splits`)

The `data.splits` module provides two partitioning strategies and a few-shot sampler. Choosing the right partition strategy is critical for honest evaluation.

**Why individual-level partitions matter**: Seedling photos of the same plant typically share the same pot, soil, label stake, and growth chamber. If two photos of the same plant appear in both train and test (as happens with image-level `StratifiedShuffleSplit`), the classifier can "cheat" by memorizing environmental context instead of learning morphological features. `create_individual_split()` groups by `ID_YPS` so all photos of the same physical plant stay in the same partition.

**Two-level partitioning**: The partition determines which images train the classifier (support set) vs. which test it (query set). This happens in two stages:

```
All ~12K images (all embedded to .npz cache regardless of partition)
  → Level 1: GroupShuffleSplit by ID_YPS → 70% train / 15% val / 15% test
  → Level 2: From train, sample k per class → support set (SimpleShot's training data)
  → Test partition → query set (what gets classified)
```

Every image gets a 768-dim BioCLIP 2 embedding regardless of partition. The `.npz` cache stores features for the entire dataset. Split files are just index arrays that select subsets — no data is duplicated.

```python
@dataclass
class DatasetSplit:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]
    num_classes: int
    split_info: dict[str, Any]   # Ratios, class counts, feasible k-shots

@dataclass
class FewShotEpisode:
    support_indices: np.ndarray
    query_indices: np.ndarray
    support_labels: np.ndarray
    query_labels: np.ndarray
    classes: np.ndarray
    n_way: int
    k_shot: int
    n_query: int

class NShotSampler:
    """Sample n-way k-shot episodes from a dataset."""
    def __init__(self, records, n_way=None, k_shot=5, n_query=15, random_seed=None): ...
    def sample_episode(self) -> FewShotEpisode: ...
    def sample_episodes(self, num_episodes: int) -> list[FewShotEpisode]: ...
```

```python
def create_stratified_split(
    records: Sequence[ImageRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    min_samples_per_class: int = 3,
) -> DatasetSplit:
    """Stratified train/val/test split with feasibility metadata.
    WARNING: Splits at image level — different images of same individual can
    leak across splits. Use create_individual_split() for evaluation."""

def create_individual_split(
    records: Sequence[ImageRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> DatasetSplit:
    """Group-aware split: all images of the same individual stay together.
    Uses GroupShuffleSplit on individual_id (ID_YPS). Prevents data leakage
    for evaluation. Raises ValueError if any record has empty individual_id."""

def save_split(split: DatasetSplit, output_path: Path) -> None:
    """Save split to .npz + .json metadata."""

def load_split(split_path: Path) -> DatasetSplit:
    """Load split from disk."""

def create_fixed_support_set(
    records: Sequence[ImageRecord], k_shot: int, random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed support set with k examples per class. Returns (indices, labels)."""

def validate_k_shot_feasibility(
    split_info: dict, k_shot: int, strict: bool = True
) -> tuple[bool, list[str]]:
    """Check if k-shot is feasible for a split. Returns (is_valid, messages)."""
```

---

## `seedlearn.clip`

### How BioCLIP 2 Embeddings Enable Taxonomic Classification

> **Deep dive**: See [Embedding Architecture Reference](clip/embedding-architecture.md) for the complete technical reference — 428M-parameter model internals, software stack (pybioclip vs open_clip), preprocessing pipeline, latent space correctness guarantees, and GPU extraction guide.

The `clip` module uses [BioCLIP 2](https://imageomics.github.io/bioclip/) — a ViT-L/14 trained with hierarchical contrastive learning on TreeOfLife-200M (~214M organism images, 952K taxa) — to extract 768-dimensional image embeddings. Understanding **why** these embeddings work for taxonomy requires distinguishing pre-training from inference.

**During pre-training**, BioCLIP 2 had two encoders (image + text) trained together. The contrastive loss reshaped the ViT's weights so that images with matching taxonomic text produce similar vectors, and mismatched pairs produce dissimilar vectors. This taught the ViT which visual features are predictive of taxonomy — leaf arrangement, venation, cotyledon shape — while suppressing features that aren't (pot color, camera angle, lighting).

**At inference in our pipeline**, the text encoder is discarded. We only use the image encoder with its frozen pre-trained weights. The 768-dim output vectors don't "contain" text — they represent visual features that the ViT learned to extract *because* those features predicted taxonomy during pre-training. The text shaped the encoder's perception; the encoder now works without text.

This is why `SimpleShot` (nearest-centroid classification) works with as few as 1-5 examples per class: the embedding space already clusters images by taxonomy, so computing class centroids and finding the nearest one is sufficient. A generic image model (ImageNet ViT, vanilla CLIP, DINOv2) would cluster by visual similarity (background, lighting) instead of biology, and nearest-centroid would fail.

**What's stored per embedding**: Each image produces one 768-dim float vector. The `.npz` cache pairs each vector with a single integer `label_id` mapping to one taxonomic name at the rank specified during extraction:

| `--rank` | Label string | Example |
|----------|-------------|---------|
| `family` | Family name | `Fabaceae` |
| `genus` | Genus name | `Acacia` |
| `species` | Genus + epithet | `Acacia dealbata` |

No trait descriptions, characteristics, or multi-rank labels are attached. The embedding itself is a purely visual 768-float vector — zero text. The label is used only by SimpleShot to know which embeddings to average together for a class centroid.

**Evidence independence**: The visual embeddings from this module are deliberately kept separate from text-based evidence (vision-LLM trait extraction, RAG literature matching) until the pipeline's Stage 4 (evidence synthesis). This provides ensemble diversity — visual features and published botanical descriptions can agree (convergent evidence) or disagree (a signal worth investigating). The disagreement surfaces as a quality flag rather than being hidden by premature fusion. See [Pipeline Reference](pipeline.md) for the full multi-stage architecture.

### Feature Extraction (`clip.encoder`)

```python
class FeatureExtractor:
    """Extract image embeddings using BioCLIP 2."""

    def __init__(
        self,
        device: torch.device | str = "cuda",
        batch_size: int = 256,
        model_str: str = "hf-hub:imageomics/bioclip-2",
    ): ...

    def extract_from_paths(
        self, image_paths, normalize=True, show_progress=True
    ) -> npt.NDArray[np.float32]: ...
    # Shape: (num_images, 768)

    def extract_from_records(
        self, records, normalize=True, show_progress=True
    ) -> npt.NDArray[np.float32]: ...

    def extract_from_records_optimized(
        self, records, normalize=True, show_progress=True,
        num_workers=8, prefetch_factor=2,
    ) -> npt.NDArray[np.float32]: ...
    # Faster via PyTorch DataLoader parallel loading
```

**Requires**: `pybioclip` (lazy-imported to avoid hard dependency).

### Feature Caching (`clip.cache`)

```python
class CachedFeatureExtractor:
    """Feature extractor with .npz disk caching."""

    def __init__(
        self, cache_dir: Path, device="cuda", batch_size=256,
        model_str="hf-hub:imageomics/bioclip-2",
    ): ...

    # --- Single-rank (legacy) ---

    def extract_and_cache(
        self, records, cache_name: str, normalize=True,
        force_recompute=False, use_optimized=True,
        num_workers=8, prefetch_factor=2,
    ) -> npt.NDArray[np.float32]:
        """Extract features; load from cache if available.
        Produces {cache_name}.npz with features, labels, image_paths."""

    def load_cached_features(
        self, cache_name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (features, labels, image_paths) from cache."""

    # --- Multi-rank (v2 — recommended for pipeline) ---

    def extract_and_cache_multirank(
        self, records, normalize=True,
        force_recompute=False, use_optimized=True,
        num_workers=8, prefetch_factor=2,
    ) -> npt.NDArray[np.float32]:
        """Extract features once, store labels for all three ranks.
        Produces features.npz + features_meta.json with taxonomy map."""
```

**Multi-rank cache format** (v2):

| File | Contents |
|------|----------|
| `features.npz` | `features` (N, 768), `{rank}_labels` × 3, `image_paths`, `individual_ids` |
| `features_meta.json` | `label_maps` (per-rank name→id), `taxonomy` (genus→family, species→genus, species→family) |

```python
# Standalone loader (no model required)
def load_multirank_cache(
    cache_dir: Path | str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict, np.ndarray]:
    """Load v2 multi-rank cache. Returns (features, rank_labels, meta, image_paths).
    Raises FileNotFoundError if features.npz missing."""

# Helper (used internally by extract_and_cache_multirank)
def _build_rank_labels(
    records: Sequence[ImageRecord],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    """Build per-rank integer label arrays from ImageRecords.
    Returns (rank_labels, rank_label_to_id)."""
```

### SimpleShot Classifier (`clip.simpleshot`)

```python
class FewShotClassifier(ABC):
    """Abstract base for few-shot classifiers."""
    def fit(self, support_features, support_labels) -> None: ...
    def predict(self, query_features) -> np.ndarray: ...
    def predict_proba(self, query_features) -> np.ndarray: ...

class SimpleShot(FewShotClassifier):
    """SimpleShot: mean-centering + L2-normalize + nearest-centroid.

    Algorithm:
      1. Compute global mean of support features
      2. Mean-center all features (subtract mean)
      3. L2-normalize centered features
      4. Predict via nearest-centroid (sklearn)
    """
    def __init__(self, device="cpu"): ...
    def fit(self, support_features, support_labels) -> None: ...
    def predict(self, query_features) -> np.ndarray: ...
    def predict_proba(self, query_features) -> np.ndarray: ...
    def mean_normalize(self, features) -> np.ndarray: ...

def l2_normalize(features: np.ndarray) -> np.ndarray:
    """L2-normalize feature vectors row-wise (handles near-zero norms)."""
```

### Metrics (`clip.metrics`)

```python
@dataclass
class EvaluationResult:
    accuracy: float
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    top5_accuracy: float | None
    per_class_metrics: dict[str, dict[str, float]]
    confusion_matrix: np.ndarray
    num_samples: int
    num_classes: int

def compute_metrics(
    y_true, y_pred, y_proba=None, label_names=None
) -> EvaluationResult:
    """Comprehensive metrics including top-5 accuracy (if y_proba provided)."""

def save_evaluation_results(
    results: EvaluationResult, output_dir: Path,
    label_names=None, experiment_info=None,
) -> None:
    """Save metrics.json, per_class_metrics.csv, confusion_matrix.csv."""

def print_results_summary(results: EvaluationResult, experiment_name="Experiment") -> None:
def compare_results(results_dict: dict[str, EvaluationResult], output_path=None) -> pd.DataFrame:
```

**Lazy imports**: `FeatureExtractor` and `CachedFeatureExtractor` are lazy-loaded in `clip/__init__.py` via `__getattr__` to avoid requiring `pybioclip` at import time.

---

## `seedlearn.components.analyzers`

### Prompt System (`components.analyzers.prompts`)

```python
class PromptStyle(str, Enum):
    SYS1 = "sys1"   # Form + justifications + conservative rules (single image)
    SYS2 = "sys2"   # Form only + notes (cleanest output)
    SYS3 = "sys3"   # Form + notes + detailed expert report
    SYS4 = "sys4"   # Multi-image analysis + conservative rules
    JSON = "json"    # JSON schema output

def get_prompt(style: PromptStyle | str) -> str:
    """Get prompt template by style. Raises ValueError if unknown."""

def is_multi_image_style(style: PromptStyle | str) -> bool:
    """True for styles supporting multiple images (currently only SYS4)."""

def is_json_style(style: PromptStyle | str) -> bool:
    """True for styles expecting JSON output."""

def list_prompts() -> dict[str, str]:
    """List all prompts with descriptions."""
```

All prompts extract **24 morphological traits** organized into sections:
- **A. Leaf Arrangement & Architecture** (2 traits)
- **B. Leaf Complexity** (4 traits)
- **C. Leaf Morphology** (8 traits)
- **D. Stem & Shoot Traits** (4 traits)
- **E. Other Visible Seedling Traits** (4 traits)
- **F/G. Notes & Report** (2 fields)

---

## `seedlearn.reporting`

### Pipeline Reports (`reporting.pipeline_html`)

```python
def generate_pipeline_report(result_dict: dict) -> str:
    """Generate complete HTML report from pipeline result dict.
    Handles both single-rank and multi-rank classification output.
    Includes Plotly charts, per-stage sections, timing waterfall."""
```

### SimpleShot Reports (`reporting.html`)

**Chart builders** (return `plotly.graph_objects.Figure`):

| Function | Purpose |
|----------|---------|
| `build_summary_table(metrics, k_shot)` | Summary metrics table |
| `build_support_set_distribution(support_set)` | Bar chart of class distribution in support set |
| `build_label_support_chart(predictions, top_k)` | Test set label distribution |
| `build_per_label_metrics(report_df, top_k)` | Grouped bar: precision/recall/F1 per class |
| `build_confusion_matrix(cm_df, top_k)` | Heatmap confusion matrix |
| `build_top_errors(predictions, top_n=15)` | Top N error pairs |
| `build_support_vs_performance(per_class_metrics, support_classes)` | Scatter: support size vs F1 |

**HTML template functions**:

| Function | Purpose |
|----------|---------|
| `get_html_header(rank, k_shot)` | HTML header with embedded CSS |
| `get_html_footer()` | Closing tags |
| `wrap_plotly_div(html, section_title, chart_type, k_shot, rank)` | Wrap chart in section with interpretation |
| `get_evaluation_context(rank, k_shot, metrics, num_classes, total_images)` | Contextual interpretation |
| `get_support_set_section(k_shot, num_classes, support_seed)` | Support set analysis |
| `get_conclusions_section(rank, k_shot, accuracy, top5_accuracy, ...)` | Conclusions & recommendations |

**Interpretation functions**:

| Function | Purpose |
|----------|---------|
| `interpret_accuracy(accuracy, k_shot, rank, num_classes)` | Natural language accuracy interpretation |
| `interpret_top5_accuracy(top1, top5)` | Top-5 vs Top-1 analysis |
| `interpret_confusion_patterns(top_errors, rank)` | Error pattern interpretation |
| `interpret_per_class_performance(best, worst, rank)` | Performance variation |
| `generate_key_findings(accuracy, top5, k_shot, rank, num_classes, baselines)` | Executive summary |

---

## Dependency Graph

```
seedlearn/
├── data/
│   ├── constants.py        ← torch
│   ├── catalog.py          ← pandas, PIL, constants
│   ├── loader.py           ← torch, PIL, torchvision
│   └── splits.py           ← numpy, sklearn (GroupShuffleSplit for individual splits)
├── clip/
│   ├── encoder.py          ← bioclip (lazy), torch, PIL, data.loader
│   ├── cache.py            ← numpy, torch, encoder
│   ├── simpleshot.py       ← sklearn, torch, numpy
│   └── metrics.py          ← sklearn, pandas, numpy
├── pipeline/               ← See docs/pipeline.md
│   ├── config.py           ← pyyaml, typing
│   ├── protocol.py         ← (stdlib only)
│   ├── result.py           ← protocol
│   ├── vlm_client.py       ← openai
│   ├── rag.py              ← faiss, sentence_transformers, numpy
│   └── stages/             ← pipeline.config, pipeline.protocol, pipeline.vlm_client,
│                              pipeline.rag, clip (lazy), data (lazy), analyzers.prompts
├── components/analyzers/
│   └── prompts.py          ← enum (stdlib only)
└── reporting/
    ├── html.py             ← pandas, plotly
    └── pipeline_html.py    ← plotly, json, re
```

---

## Key Defaults

| Setting | Value | Location |
|---------|-------|----------|
| Default rank | `"species"` | `load_dataset()` |
| Train/val/test partition | 70/15/15 | `create_stratified_split()` |
| BioCLIP model | `"hf-hub:imageomics/bioclip-2"` | `FeatureExtractor` |
| Feature dimensions | 768 (ViT-L/14) | BioCLIP 2 output |
| Extraction batch size | 256 | `FeatureExtractor` |
| DataLoader workers | 8 | `create_dataloader()` |
| Random seed | 42 | Splits, samplers |
| H200 batch size | 2048 | `get_optimal_batch_size()` |
