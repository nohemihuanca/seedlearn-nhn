# Pipeline Reference — `seedlearn.pipeline`

**Navigation**: [SeedLearn](../README.md) > **Pipeline Reference**

---

## Overview

The SeedLearn pipeline is a 5-stage classification system that combines vision-language models, few-shot visual embeddings, and literature-based trait retrieval to identify tropical tree seedlings. Each stage produces independent evidence channels; a deterministic synthesis step assembles them into a structured document that a text-only LLM uses for final classification with reasoning.

**Key property**: Stages 1-3 are independent evidence sources. They can run in any order, be skipped individually, or be swapped for different implementations. Stage 4 is deterministic (no ML). Stage 5 is the only stage that interprets evidence.

### Why 5 Stages?

The demo repository had a 3-step monolithic pipeline: vision-LLM extraction → string-match traits → vision-LLM synthesis (same model, with images). This was redesigned for three reasons:

1. **Evidence channel independence** — The original pipeline used the same vision-LLM for extraction AND synthesis, passing images twice. The new design extracts visual information once (Stages 1-2) and reasons over text only (Stage 5), so the reasoning model never sees images and can't hallucinate visual details.

2. **Model efficiency** — Stage 5 uses a text-only LLM, not a vision-LLM. A text-only model dedicates all parameters to reasoning rather than splitting them between vision and language encoders. This means better reasoning per GPU dollar.

3. **Reproducibility** — The Stage 4 evidence document is deterministic. Given the same Stage 1-3 outputs, it always produces the same text. You can archive evidence documents and re-run Stage 5 with different models, temperatures, or prompts without re-running any vision stages.

---

## Architecture

```
Input: 1-N images of a seedling specimen (from held-out test individuals)
  │
  ├─→ Stage 1: Vision-LLM Morphological Extraction    [vision-LLM]
  │   (images → 24-trait morphology dict)
  │
  ├─→ Stage 2: Visual Embedding Classification        [BioCLIP 2 — GPU]
  │   (images → feature vectors → top-k predictions)
  │
  ├─→ Stage 3: Literature-Based Trait Retrieval        [RAG — CPU]
  │   (morphology traits → semantic search → literature matches + convergence)
  │
  ├─→ Stage 4: Evidence Synthesis                      [Deterministic — no LLM]
  │   (stages 1-3 outputs → structured Markdown evidence document)
  │
  └─→ Stage 5: LLM Reasoning & Classification         [Text-only LLM]
      (evidence document → final classification + reasoning + alternatives)

Output: PipelineResult (JSON-serializable with per-stage data, timing, errors)
```

### Data Flow Between Stages

Stages communicate through a shared `context: dict[str, Any]` dictionary. Each stage reads from upstream keys and writes its output under its own key:

| Stage | Reads from context | Writes to context |
|-------|--------------------|-------------------|
| Stage 1 | `image_paths` | `morphology` → `{traits, raw_response, thinking}` |
| Stage 2 | `image_paths` | `classification` → `{predictions, ...}` (single-rank) or `{predictions_by_rank, margin_by_rank, hierarchical_consistency, ...}` (multi-rank) |
| Stage 3 | `morphology`, optionally `classification` | `trait_retrieval` → `{query, rag_matches, convergence}` |
| Stage 4 | `morphology`, `classification`, `trait_retrieval` | `evidence_synthesis` → `{evidence_document, quality_flags}` |
| Stage 5 | `evidence_synthesis` | `reasoning` → `{classification, raw_response, thinking}` |

**Stage 3 depends on Stage 1** (it needs trait data to compose the RAG query). Stages 1 and 2 are independent and could theoretically run in parallel. Stage 4 needs at least one of Stages 1-3. Stage 5 needs Stage 4.

---

## Stage Protocol

Every stage implements the `PipelineStage` protocol (`pipeline/protocol.py`):

```python
@runtime_checkable
class PipelineStage(Protocol):
    @property
    def name(self) -> str: ...
    def validate_input(self, context: dict[str, Any]) -> list[str]: ...
    def run(self, context: dict[str, Any]) -> StageResult: ...
    def skip(self, context: dict[str, Any]) -> StageResult: ...
```

**Design decision**: This is a `Protocol` (structural typing), not an ABC (nominal typing). Any class with these four members satisfies the protocol — no inheritance required. This was chosen over an ABC because stages have very different initialization signatures (vision-LLM stages need endpoint configs, classification needs cached features, RAG needs an index), and forcing a common `__init__` via ABC would mean either a bloated base class or repeated `**kwargs` patterns.

### StageResult

Every stage returns a `StageResult` dataclass:

```python
@dataclass
class StageResult:
    stage_name: str              # "morphology", "classification", etc.
    data: dict[str, Any]         # Stage-specific output (see per-stage docs)
    skipped: bool = False        # True if stage was skipped
    error: str | None = None     # Error message if stage failed
    elapsed_ms: float = 0.0      # Wall-clock execution time
```

**Error handling pattern**: Stages catch all exceptions internally, log them, and return a `StageResult` with `error` set. The pipeline runner can then decide whether to continue (graceful degradation) or abort. This prevents one stage failure from crashing the entire pipeline.

---

## Configuration

### Hierarchy

```
code defaults (dataclass fields) → config.yaml → CLI overrides (dot-notation)
```

Every config field has a sensible default in the dataclass definition. A YAML file can override any subset. CLI flags override both.

### Config Classes (`pipeline/config.py`)

```python
@dataclass
class PipelineConfig:
    skip_stages: list[str]                                    # e.g. ["trait_retrieval"]
    vlm: VLMConfig                                            # Stage 1
    classifier: ClassifierConfig                              # Stage 2
    trait_retrieval: TraitRetrievalConfig                      # Stage 3
    evidence_synthesis: EvidenceSynthesisConfig                # Stage 4
    reasoning: ReasoningConfig                                # Stage 5
    output: OutputConfig                                      # Output format/dir
```

#### VLMConfig (Stage 1)

| Field | Default | Purpose |
|-------|---------|---------|
| `model` | `"Qwen/Qwen3-VL-32B-Instruct-FP8"` | HuggingFace model ID |
| `endpoint` | `"http://localhost:8000/v1"` | OpenAI-compatible API endpoint |
| `prompt_style` | `"sys4"` | Prompt variant (sys1-sys4, json). `sys4` is multi-image. |
| `image_mode` | `"file"` | `"file"` for `file://` URIs (local vLLM), `"base64"` for remote |
| `max_images` | `10` | Max images per specimen |
| `max_tokens` | `8192` | Max response tokens |
| `temperature` | `0.6` | Sampling temperature |
| `top_p` / `top_k` / `min_p` | `0.95` / `20` / `-1.0` | Generation params |

#### ClassifierConfig (Stage 2)

| Field | Default | Purpose |
|-------|---------|---------|
| `rank` | `"family"` | Taxonomic rank to classify (single-rank mode) |
| `ranks` | `["family", "genus", "species"]` | Ranks to classify (multi-rank mode) |
| `k_shot` | `10` | Support examples per class |
| `top_k` | `5` | Number of top predictions to return |
| `split_seed` | `42` | Random seed for data splits |
| `device` | `"cuda"` | Compute device |
| `catalog` | `None` | Species catalog CSV (None = DEFAULT_CATALOG) |
| `model_str` | `"hf-hub:imageomics/bioclip-2"` | Vision backbone (768-dim) |
| `feature_aggregation` | `"mean"` | Multi-image feature pooling strategy |
| `ood_percentile` | `95.0` | Percentile for distance-based OOD confidence gating |

#### TraitRetrievalConfig (Stage 3)

| Field | Default | Purpose |
|-------|---------|---------|
| `enabled` | `True` | Whether RAG retrieval is active |
| `index_path` | `None` | Pre-built FAISS index directory |
| `descriptions_csv` | `None` | NLP descriptions CSV path |
| `embedding_model` | `"all-MiniLM-L6-v2"` | Sentence-transformer model |
| `top_k` | `20` | Max RAG search results |
| `min_similarity` | `0.3` | Cosine similarity floor |
| `cross_reference` | `True` | Cross-reference with Stage 2 |

#### EvidenceSynthesisConfig (Stage 4)

| Field | Default | Purpose |
|-------|---------|---------|
| `include_raw_traits` | `False` | Include raw JSON in evidence doc |
| `include_rag_passages` | `True` | Include literature excerpts |
| `max_rag_excerpts` | `5` | Max excerpts per candidate taxon |
| `convergence_threshold` | `0.3` | Min agreement score for convergence; also min acceptable top-prediction softmax_score |

#### ReasoningConfig (Stage 5)

| Field | Default | Purpose |
|-------|---------|---------|
| `model` | `"Qwen/Qwen3-VL-32B-Instruct-FP8"` | Defaults to same model as Stage 1 (single-server setup) |
| `endpoint` | `"http://localhost:8000/v1"` | API endpoint (can differ from Stage 1) |
| `max_tokens` | `4096` | Max reasoning response tokens |
| `temperature` / `top_p` / `top_k` | `0.6` / `0.95` / `20` | Generation params |

#### PromptsConfig (Cross-Stage)

Centralizes paths to user-customizable prompt text files. When a path is set, the stage loads the prompt from that file instead of its hardcoded default. Missing files fall back gracefully with a warning log.

| Field | Default | Purpose |
|-------|---------|---------|
| `morphology` | `None` | Path to Stage 1 system prompt file |
| `rag_query` | `None` | Path to Stage 3 RAG query template file (uses `{traits}` placeholder) |
| `reasoning` | `None` | Path to Stage 5 system prompt file |

Default prompt files ship with the repo in `configs/prompts/`:

| File | Stage | Description |
|------|-------|-------------|
| `stage1_morphology.txt` | 1 | Multi-image morphological assessment (SYS4 content) |
| `stage3_rag_query.txt` | 3 | RAG query template with `{traits}` variable |
| `stage5_reasoning.txt` | 5 | Classification reasoning system prompt |

**To customize prompts:**

1. Copy a default prompt file: `cp configs/prompts/stage5_reasoning.txt my_prompts/reasoning.txt`
2. Edit the copy with your changes
3. Point `pipeline.yaml` to your file:

```yaml
prompts:
  reasoning: "my_prompts/reasoning.txt"
```

Or via CLI override: `--override prompts.reasoning=my_prompts/reasoning.txt`

**Priority chain:** `prompt file path (from YAML/CLI)` > `prompt registry shorthand (prompt_style)` > `hardcoded default in source code`.

Stage 1 retains the `prompt_style` registry (`sys1`-`sys4`, `json`) for benchmarking. When a `prompts.morphology` file path is set, it takes priority over `prompt_style`.

### Loading Config

```python
from seedlearn.pipeline.config import load_config

# Code defaults only
config = load_config(path=None)

# YAML file
config = load_config("configs/pipeline.yaml")

# YAML + CLI overrides (dot-notation)
config = load_config("configs/pipeline.yaml", overrides={
    "vlm.model": "Qwen/Qwen3-VL-7B-Instruct",
    "classifier.rank": "species",
    "classifier.top_k": "10",
})
```

The `_apply_overrides` function handles type coercion automatically — string CLI values are cast to the field's declared type (`bool`, `int`, `float`, `list`, `str`).

---

## Stage 1: Vision-LLM Morphological Extraction

**File**: `pipeline/stages/morphology.py`
**Class**: `MorphologyStage`
**Config**: `VLMConfig`

### What It Does

Sends 1-N seedling images to a vision-language model with a structured prompt requesting 24 morphological traits. The vision-LLM returns either a filled-in assessment form or structured JSON, which is parsed into a `traits` dictionary.

### The 24 Traits

Organized into sections (from `seedlearn.components.analyzers.prompts`):

- **A. Leaf Arrangement & Architecture** (2): leaf position, phyllotaxis
- **B. Leaf Complexity** (4): simple/compound, number of leaflets, leaflet arrangement, rachis
- **C. Leaf Morphology** (8): shape, margin, apex, base, venation, texture, color, size
- **D. Stem & Shoot Traits** (4): woodiness, bark, color, lenticels
- **E. Other Visible Seedling Traits** (4): stipules, pulvinus, latex, thorns
- **F/G. Notes & Report** (2): additional observations, confidence notes

### Prompt Styles

| Style | Description | Output Format |
|-------|-------------|---------------|
| `sys1` | Form + justifications + conservative rules (single image) | Form |
| `sys2` | Form only + notes (cleanest output) | Form |
| `sys3` | Form + notes + detailed expert report | Form |
| `sys4` | Multi-image analysis + conservative rules | Form |
| `json` | JSON schema output | JSON |

**Default is `sys4`** — designed for multi-image input, which is the standard case (specimens have 4-5 photos from different angles).

### How It Works

1. `get_prompt(config.prompt_style)` retrieves the prompt template
2. `build_messages(system_prompt, image_paths, image_mode)` constructs OpenAI-format messages with images as `file://` URIs (local vLLM) or base64 data URIs (remote)
3. `InferenceClient.chat(messages)` sends the request and returns `InferenceResponse` with content, thinking blocks, usage stats, and timing
4. Response is parsed: `JSONParser.parse()` for JSON styles, `FormParser.parse()` for form styles (both from `components.analyzers.parsers`)

### Form Parser (`FormParser`)

The form parser (in `components/analyzers/parsers.py`) handles the messy reality of vision-LLM outputs:
- Matches numbered trait lines (`1.` through `24.`) to a fixed `_TRAIT_MAP` of `(section, field_name)` tuples
- Strips parenthetical justifications (e.g. `"whorled (leaves arranged...)"` → `"whorled"`)
- Groups traits into 5 sections: `leaf_arrangement`, `leaf_complexity`, `leaf_morphology`, `stem_traits`, `special_features`
- Preserves free-text notes from section F without stripping
- Returns a nested `dict[str, dict[str, str]]` grouped by section

### JSON Parser (`parse_json_response`)

Multi-pass recovery for malformed vision-LLM JSON:
1. Strip thinking blocks, remove markdown fences, try `json.loads`
2. Regex-extract outermost `{...}` object
3. Truncation recovery — close unmatched braces/brackets
4. Aggressive fallback — strip back character by character to last valid comma
5. Return `None` if all strategies fail

**Why this complexity**: VLMs frequently produce truncated or wrapped JSON. The benchmarks showed ~15% of responses needed some form of JSON recovery. Without this, those specimens would be lost data.

### Output

```python
StageResult.data = {
    "traits": {
        "leaf_arrangement": {"relative_position": "opposite", "spacing": "clustered"},
        "leaf_complexity": {"type": "simple", ...},
        "leaf_morphology": {"margin": "entire", "shape": "elliptic", ...},
        "stem_traits": {...},
        "special_features": {...},
        "notes": "..."
    },
    "raw_response": "<full model output including thinking blocks>",
    "thinking": "<extracted thinking text, if present>",
}
```

---

## Stage 2: Visual Embedding Classification

**File**: `pipeline/stages/classification.py`
**Class**: `ClassificationStage`
**Config**: `ClassifierConfig`

### What It Does

Extracts BioCLIP 2 embeddings from specimen images, mean-pools them into a single embedding, and classifies using a pre-fitted SimpleShot nearest-centroid classifier. Returns top-k predictions with probabilities.

### Why BioCLIP 2 (Pre-Training vs. Inference)

> **Deep dive**: See [Embedding Architecture Reference](clip/embedding-architecture.md) for the complete technical reference — model internals (parameter counts, layer structure, projection math), software stack (pybioclip vs open_clip), preprocessing details, correctness checklist, and GPU extraction guide.

BioCLIP 2 is a ViT-L/14 trained with **hierarchical contrastive learning** on TreeOfLife-200M (~214M organism images spanning 952K taxa). It has two encoders:

- **Image encoder** (ViT-L/14) — pixels in, 768-dim vector out
- **Text encoder** — taxonomic text in, 768-dim vector out

During pre-training, the contrastive loss pushes image-text pairs with matching taxonomy close together in the shared 768-dim space, and mismatched pairs apart:

```
(photo of Fabaceae seedling) ←→ "Fabaceae, legume family"    → push CLOSE
(photo of Fabaceae seedling) ←→ "Poaceae, grass family"      → push APART
```

This does **not** inject text into image vectors. It **reshapes the ViT's weights** so that images whose taxonomic text matches produce similar vectors. After training, the ViT has learned which visual features are predictive of taxonomy — leaf arrangement, venation patterns, cotyledon shape, stem pubescence — because those features predicted which text description the image was paired with during training. Pixel-level features that don't correlate with taxonomy (pot color, camera angle, lighting) get suppressed because they didn't help predict the text.

**At inference in our pipeline, we throw away the text encoder entirely.** We only use the image encoder with its pre-trained weights frozen:

```
seedling photo → ViT-L/14 (frozen weights from pre-training) → 768-dim vector
```

The 768 dimensions don't "contain" taxonomic text. They represent visual features that the ViT learned to extract *because* those features were predictive of taxonomy during pre-training. The text shaped the encoder's perception; the encoder now works without any text.

When we embed all ~12K images, same-family images cluster in 768-dim space — not because we told the system anything about families, but because the ViT was pre-trained to produce similar vectors for images that share taxonomic descriptions. A generic model (ImageNet ViT, vanilla CLIP, DINOv2) would cluster by visual similarity (pot color, background, lighting) instead of taxonomy.

**Analogy**: Train a student with flash cards — photo on front, family name on back. After months of study, take away the cards and show new photos. They can still sort by family because they internalized which visual features correlate with each family. The text is gone; the perceptual skill persists. That's what BioCLIP 2's frozen ViT weights are — internalized perceptual skill shaped by taxonomic text, applied to pixels alone.

### How SimpleShot Works (Step by Step)

1. **Pre-compute** BioCLIP 2 embeddings for all training images → `.npz` cache
2. **Fit**: Load cached features, compute per-class centroid (mean of support vectors), L2-normalize
3. **At inference**: Extract features for query images → mean-pool across N images → L2-normalize → compute negative L2 distance to each centroid → softmax with temperature scaling → class probabilities

### Multi-Image Handling

**Decision**: Mean pooling of features across N images per specimen.

```python
# Extract features for all images: shape (N, 768)
features = extractor.extract_from_paths(paths, normalize=True)

# Mean-pool: shape (1, 768)
mean_features = features.mean(axis=0, keepdims=True)

# Re-normalize after pooling
norm = np.linalg.norm(mean_features, axis=1, keepdims=True)
mean_features = mean_features / norm
```

**Why mean pooling over alternatives**:
- **Per-image voting** loses cross-image information (top view alone might be ambiguous; combined with stem close-up, it's diagnostic)
- **Feature concatenation** requires fixed N and matching support set structure
- Mean pooling is what the centroid approach already does for support sets, so it's mathematically consistent

### Pre-Fitting

The classifier must be fitted before `run()`. Three paths:

```python
# Path 1: From raw features (for testing)
stage._fit_classifier(support_features, support_labels, label_names)

# Path 2: Single-rank from cached features + split file
stage.load_from_cache(cache_dir="/path/to/cache", split_path="/path/to/split")

# Path 3: Multi-rank from v2 cache + per-rank splits (recommended for pipeline)
stage.load_from_multirank_cache(
    cache_dir="/path/to/cache",
    split_paths={
        "family": Path("splits/family/split_seed42"),
        "genus": Path("splits/genus/split_seed42"),
        "species": Path("splits/species/split_seed42"),
    },
)
```

`load_from_cache` (single-rank) loads `{rank}_features.npz`, loads the split file, extracts training indices, and fits one SimpleShot classifier.

`load_from_multirank_cache` (multi-rank) loads `features.npz` + `features_meta.json`, loads all three rank splits, fits independent SimpleShot classifiers per rank on shared features, and loads taxonomy cross-reference for hierarchical consistency checking. Warns if per-rank train indices differ (potential leakage).

### Known Limitations (Documented by Design)

1. **Closed-set only**: Can only predict species/families present in the training set. Novel taxa are impossible.
2. **No fine-tuning**: Pure feature extraction + nearest centroid. BioCLIP 2's internal knowledge is limited to pretraining data.
3. **Static support set**: The k-shot support set is sampled once per fit. Different random samples → different centroids → different results.
4. **Feature aggregation**: Mean pooling loses per-image discriminative signal. Future options: attention-weighted pooling, max pooling.

### End-to-End Data Flow: From Images to Classification

Understanding how data flows through Stage 2 requires seeing the full pipeline from raw images to predictions. There are two distinct phases: **offline preparation** (done once) and **runtime inference** (per specimen).

**Offline preparation** (one-time, before any pipeline runs):

```
All ~12K images in catalog
  → BioCLIP 2 encodes EVERY image → (N, 768) float32 feature cache (.npz)
  → GroupShuffleSplit by ID_YPS → 70% train / 15% val / 15% test (index arrays)
```

Every image gets embedded regardless of which partition it ends up in. The `.npz` cache stores features for the entire dataset. The split files are just index arrays that select subsets — no data is duplicated or moved.

**Runtime inference** (per specimen, during pipeline execution):

```
From train partition: sample k images per class → support set
  → SimpleShot computes class centroids from support features
  → New specimen images → BioCLIP 2 → mean-pool → nearest centroid → prediction
```

The classifier only sees `k × num_classes` support images from the training partition. The rest of the training images are unused at inference time. The split indices tell SimpleShot which cached features to use as support and which to evaluate on.

### Partitioning and Data Leakage Prevention

Partitioning happens at **two levels**. Neither determines which images get embedded — all images are embedded. The partition determines which embeddings train the classifier (support set) vs. which test it (query set).

**Level 1 — Individual-level partitioning** (`data/splits.py`, `create_individual_split()`):

Uses sklearn's `GroupShuffleSplit` with `individual_id` (the `ID_YPS` column from the catalog CSV) as the group key. All photos of the same physical plant land in the same partition:

```
Plant PP001 (5 photos) → ALL go to train
Plant PP002 (3 photos) → ALL go to test
Plant PP003 (4 photos) → ALL go to val
```

This prevents data leakage. Seedling photos typically share the same pot, soil, label stake, and growth chamber across multiple photos of the same plant. Without grouping by `ID_YPS`, the model memorizes environmental context (recognizing a specific pot or bench position) instead of learning morphological features that distinguish taxa.

The older `create_stratified_split()` shuffles at the image level, so two photos of the same plant can appear in both train and test — inflating accuracy by letting the model "cheat" with background cues.

**Level 2 — k-shot support sampling** (`data/splits.py`, `create_fixed_support_set()`):

Within the training partition, exactly `k` images per class are sampled to form the support set. The classifier only ever sees these `k × num_classes` images. The remaining training images only influence partition statistics and feasibility validation.

### Evidence Independence

Stage 2's visual classification is deliberately kept **independent** from the other evidence sources until Stage 4 (evidence synthesis). The visual embeddings know nothing about the Stage 1 vision-LLM trait extraction or the Stage 3 RAG literature matches. This separation provides ensemble diversity: when BioCLIP's visual features and published botanical descriptions independently point to the same family, that's convergent evidence from unrelated sources. When they disagree, it's a signal worth investigating — and the disagreement surfaces as a quality flag in Stage 4, rather than being hidden by premature fusion.

Think of BioCLIP 2 as a domain-specific perceptual hash: similar biology gets similar vectors, even when pixel-level appearance varies (different photos, angles, lighting). The trait text from Stage 3 provides a completely different axis of evidence — what published literature says about the morphological features the vision-LLM observed. These two evidence streams only merge at Stage 4.

### Output

**Single-rank mode** (backward compatible):

```python
StageResult.data = {
    "predictions": [
        {"rank_value": "Fabaceae", "softmax_score": 0.45, "rank_position": 1, "l2_distance": 0.62},
        {"rank_value": "Meliaceae", "softmax_score": 0.22, "rank_position": 2, "l2_distance": 0.81},
        ...
    ],
    "margin": 0.23,
    "per_image_predictions": [{"image_path": "...", "top1_label": "Fabaceae", "top1_softmax_score": 0.43}, ...],
    "nearest_support": [{"label": "Fabaceae", "l2_distance": 0.58, "cosine_similarity": 0.83}, ...],
    "embedding_dim": 768,
    "num_images_pooled": 4,
}
```

**Multi-rank mode** (recommended, auto-detected when `features.npz` present):

```python
StageResult.data = {
    "predictions_by_rank": {
        "family": [{"rank_value": "Fabaceae", "softmax_score": 0.45, "rank_position": 1, "l2_distance": 0.62}, ...],
        "genus": [{"rank_value": "Inga", "softmax_score": 0.31, "rank_position": 1, "l2_distance": 0.69}, ...],
        "species": [{"rank_value": "Inga vera", "softmax_score": 0.18, "rank_position": 1, "l2_distance": 0.74}, ...],
    },
    "margin_by_rank": {"family": 0.23, "genus": 0.12, "species": 0.05},
    "hierarchical_consistency": {
        "consistent": True,
        "family_genus_agree": True, "species_genus_agree": True, "species_family_agree": True,
        "notes": [],
    },
    "confidence_gate": {"family_in_distribution": True, ...},  # Only if calibrated
    "per_image_predictions": [...],
    "nearest_support": [...],
    "embedding_dim": 768,
    "num_images_pooled": 4,
}
```

The `run()` method dispatches to `_run_single_rank()` or `_run_multirank()` based on how the classifier was fitted. Multi-rank mode classifies at all three taxonomic levels simultaneously using independent SimpleShot classifiers on shared feature vectors, then cross-validates via taxonomy lookup.

---

## Stage 3: Literature-Based Trait Retrieval

**File**: `pipeline/stages/trait_retrieval.py`
**Class**: `TraitRetrievalStage`
**Config**: `TraitRetrievalConfig`
**Supporting**: `pipeline/rag.py` (`RAGIndex`)

### What It Does

Converts the vision-LLM's morphological assessment into a natural language query, searches a FAISS vector index of 3,498 botanical descriptions from published literature, and cross-references the top matches with Stage 2's visual classification to identify convergence and divergence signals.

### Why RAG Over String Matching

The demo repository used exact/partial string matching against a structured trait CSV (24 categorical columns). This was almost useless because:
- Most cells were empty or `[]`
- Matching "compound leaves" to a column value `["compound"]` captured no semantic similarity
- Rich botanical descriptions were ignored entirely

The NLP descriptions CSV has 3,498 passages averaging ~627 characters each, covering 362 taxa (85.8% of the dataset). These are the actual published botanical descriptions from Flora of BCI, Manual de Plantas de Costa Rica, and other references. RAG leverages the full semantic content.

### RAGIndex (`pipeline/rag.py`)

**FAISS IndexFlatIP** — inner product on L2-normalized vectors = cosine similarity.

```python
class RAGIndex:
    # Build from descriptions
    @classmethod
    def build(descriptions: list[dict], model_name="all-MiniLM-L6-v2") -> RAGIndex

    # Search by semantic similarity
    def search(query: str, top_k=10, min_similarity=0.0) -> list[dict]

    # Persist to disk
    def save(directory: Path) -> None
    def load(directory: Path) -> RAGIndex
```

**Building the index** (one-time offline step):

```bash
python scripts/build_rag_index.py \
    --descriptions data/traits/latest/concatenated_output_nlp.csv \
    --output data/traits/latest/rag_index/
```

This produces `index.faiss` (FAISS binary index) and `metadata.json` (taxon, rank, description for each entry) in the output directory.

### Query Composition (`_compose_query`)

Converts the Stage 1 trait dictionary into a natural language search query:

```python
# Input traits: {"leaf position": "opposite", "margin": "entire", "latex": "unclear"}
# Output: "Tropical tree seedling with leaf position: opposite, margin: entire"
# (skips "unclear", "n/a", "not observed", "not visible")
```

Filters out uninformative values before composing. Falls back to `"tropical tree seedling"` if all traits are unclear.

### Cross-Referencing (`_cross_reference`)

Matches RAG taxa against Stage 2 predictions by name (case-insensitive):

| Signal | Definition |
|--------|-----------|
| **Strong convergence** | Taxon in both RAG (score ≥ 0.5) and visual (softmax_score ≥ 0.3) |
| **Moderate convergence** | Taxon in both but below strong thresholds |
| **RAG only** | Taxon in top RAG matches but not in visual top-k |
| **Visual only** | Taxon in visual top-k but not in RAG matches |

**Scientific significance**: When BioCLIP's visual features AND published botanical descriptions independently point to the same family, that's convergent evidence from two completely independent sources. When they disagree, it's a signal worth investigating.

### Output

```python
StageResult.data = {
    "query": "Tropical tree seedling with leaf position: opposite, margin: entire, ...",
    "rag_matches": [
        {"taxon": "Fabaceae", "rank": "family", "description": "...", "score": 0.87},
        ...
    ],
    "convergence": [
        {"taxon": "Fabaceae", "signal": "strong", "rag_score": 0.87,
         "visual_softmax_score": 0.45, "source": "both"},
        ...
    ],
}
```

---

## Stage 4: Evidence Synthesis

**File**: `pipeline/stages/evidence.py`
**Class**: `EvidenceSynthesisStage`
**Config**: `EvidenceSynthesisConfig`

### What It Does

Assembles outputs from Stages 1-3 into a structured Markdown evidence document and computes quality flags. **This stage is entirely deterministic** — no machine learning, no LLM calls, no randomness. Same inputs always produce the same output.

### Why Deterministic?

The original demo combined evidence formatting and LLM reasoning in a single vision-LLM call with images. This meant:
- You couldn't inspect what evidence the LLM actually saw
- You couldn't re-run reasoning without re-running the vision-LLM
- The evidence document was different every time (LLM stochasticity)

By making Stage 4 deterministic, the evidence document becomes an auditable, reproducible artifact. You can diff evidence documents across specimens, archive them, and use them for quality control.

### Evidence Document Structure

```markdown
# Evidence Summary

## Morphological Profile
- **leaf position**: opposite
- **phyllotaxis**: decussate
- **leaf complexity**: compound
...

## Visual Classification
- #1: **Fabaceae** (similarity share: 45.2%)
- #2: **Meliaceae** (similarity share: 22.1%)
...

## Literature Evidence
### Matching Taxa (by trait similarity)
- **Fabaceae** (family, similarity: 0.87)
  - compound leaves, stipules, pulvinus at base of petiole...
...

## Convergence Analysis
- **Fabaceae** [STRONG CONVERGENCE] (RAG: 0.87, Visual: 45.2%)
- **Sapindaceae** [Literature only] (RAG: 0.71, Visual: 0.0%)
...
```

### Quality Flags

Rule-based quality assessment:

| Flag | Trigger | Purpose |
|------|---------|---------|
| Low morphological quality | ≥5 of 24 traits are "unclear"/"n/a"/"not observed" | Image quality or model limitations |
| Low classification confidence | Top prediction softmax_score < `convergence_threshold` (0.3) | Ambiguous specimen or out-of-distribution |
| No evidence available | Both `traits` and `predictions` are empty | Upstream failures |

### Output

```python
StageResult.data = {
    "evidence_document": "# Evidence Summary\n\n## Morphological Profile\n...",
    "quality_flags": ["Low morphological quality: 8 of 24 traits are unclear/not observed"],
}
```

---

## Stage 5: LLM Reasoning & Classification

**File**: `pipeline/stages/reasoning.py`
**Class**: `ReasoningStage`
**Config**: `ReasoningConfig`

### What It Does

Sends the Stage 4 evidence document to a **text-only LLM** (not a vision-LLM) with a system prompt instructing it to classify the specimen, explain its reasoning, and list alternatives. Returns structured JSON with the classification.

### Why Text-Only (Not a Vision-LLM)?

This is the core architectural decision behind the Stage 4/5 split:

1. **Parameter efficiency**: A 32B text-only model has all 32B parameters for reasoning. A 32B vision-LLM splits parameters between vision encoder and language model — you get less reasoning capacity for the same GPU footprint.
2. **Different model choices**: You might want a small fast vision-LLM for Stage 1 extraction (where vision quality matters most) and a large capable text model for Stage 5 reasoning (where analytical depth matters most).
3. **Multi-pass iteration**: Re-running Stage 5 with different prompts, temperatures, or models is cheap (text-only, fast). Re-running Stage 1 is expensive (vision-LLM + images). This enables rapid experimentation with reasoning strategies.

### System Prompt

The built-in system prompt instructs the LLM to:
1. Analyze ALL evidence from the evidence document
2. Identify the most likely family (and genus/species if sufficient evidence)
3. Assess confidence (high/medium/low)
4. Explain reasoning citing specific traits and evidence
5. List plausible alternatives

Expected JSON response format:
```json
{
  "predicted_family": "Fabaceae",
  "predicted_genus": "Inga",
  "predicted_species": "Inga marginata",
  "confidence": "high",
  "reasoning": "Strong convergence between visual (45%) and literature (0.87) for Fabaceae. Compound opposite leaves with pulvinus are diagnostic for Fabaceae...",
  "supporting_features": ["compound leaves", "pulvinus present", "stipules"],
  "alternatives": [
    {"taxon": "Meliaceae", "reason": "Similar compound leaf arrangement but lacks pulvinus"}
  ]
}
```

### Fallback Logic

If the LLM response cannot be parsed as JSON (after multi-pass recovery), the stage falls back to Stage 2's top visual prediction with `confidence: "low"` and a warning. This prevents the pipeline from producing no output even when the LLM fails.

### Output

```python
StageResult.data = {
    "classification": {
        "predicted_family": "Fabaceae",
        "predicted_genus": "Inga",
        "predicted_species": "Inga marginata",
        "confidence": "high",
        "reasoning": "...",
        "supporting_features": [...],
        "alternatives": [...],
    },
    "raw_response": "<full model output>",
    "thinking": "<extracted thinking text>",
}
```

---

## Pipeline Result (`pipeline/result.py`)

```python
@dataclass
class PipelineResult:
    specimen_id: str
    image_paths: list[str]
    stage_results: dict[str, StageResult]    # stage_name -> StageResult

    @property
    def total_elapsed_ms(self) -> float       # Sum of all stage times

    def add_stage_result(sr: StageResult)     # Add a stage result
    def get_stage_data(stage_name) -> dict    # Get data from a specific stage
    def to_dict() -> dict[str, Any]           # JSON-serializable nested dict
```

### Serialized Format

```json
{
  "specimen_id": "PP123",
  "image_paths": ["/path/to/img1.jpg", "/path/to/img2.jpg"],
  "stages": {
    "morphology": {"data": {...}, "skipped": false, "error": null, "elapsed_ms": 8432.1},
    "classification": {"data": {...}, "skipped": false, "error": null, "elapsed_ms": 145.3},
    "trait_retrieval": {"data": {...}, "skipped": false, "error": null, "elapsed_ms": 52.7},
    "evidence_synthesis": {"data": {...}, "skipped": false, "error": null, "elapsed_ms": 1.2},
    "reasoning": {"data": {...}, "skipped": false, "error": null, "elapsed_ms": 3200.5}
  },
  "total_elapsed_ms": 11831.8
}
```

---

## Inference Client (`pipeline/vlm_client.py`)

Shared inference client used by both Stage 1 (with images) and Stage 5 (text-only).

### Classes

```python
@dataclass
class InferenceConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8"
    api_key: str = "EMPTY"
    timeout: float = 172800          # 48 hours (vLLM can be slow on first load)
    max_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = -1.0
    image_mode: str = "file"         # "file" or "base64"

@dataclass
class InferenceResponse:
    content: str                     # Cleaned response (thinking blocks stripped)
    raw_content: str                 # Unmodified model output
    thinking: str | None = None      # Extracted thinking block text
    model: str = ""
    usage: dict[str, int] = {}       # {prompt_tokens, completion_tokens, total_tokens}
    processing_time_ms: float = 0.0

class InferenceClient:
    def __init__(config: InferenceConfig | None)
    def chat(messages, strip_thinking=True) -> InferenceResponse
    def health_check() -> bool
```

### Utility Functions

| Function | Purpose |
|----------|---------|
| `strip_thinking(text)` | Remove `<think>...</think>` blocks from model output |
| `parse_json_response(text)` | Multi-pass JSON extraction with truncation recovery |
| `build_messages(system_prompt, image_paths, image_mode, user_text)` | Build OpenAI-compatible message array with images |

### VLMConfig → InferenceConfig Mapping

Stage configs (`VLMConfig`, `ReasoningConfig`) use `endpoint` as the field name. The `InferenceConfig` (shared client) uses `base_url`. Stages map between them:

```python
# In MorphologyStage / ReasoningStage:
InferenceConfig(
    base_url=self._config.endpoint,   # endpoint → base_url
    model=self._config.model,
    ...
)
```

This allows Stage 1 and Stage 5 to use different endpoints (e.g., different vLLM servers or different models).

---

## RAG Index (`pipeline/rag.py`)

### Building the Index

One-time offline step via CLI:

```bash
python scripts/build_rag_index.py \
    --descriptions data/traits/latest/concatenated_output_nlp.csv \
    --output data/traits/latest/rag_index/ \
    --model all-MiniLM-L6-v2 \
    --taxon-col accepted_name \
    --rank-col taxonomic_rank \
    --description-col trait_description_nlp
```

Produces:
- `index.faiss` — FAISS binary index (inner-product on L2-normalized vectors = cosine similarity)
- `metadata.json` — `{model_name, entries: [{taxon, rank, description}, ...]}`

### Index Size

- 3,498 botanical descriptions
- 362 taxa (85.8% coverage): 61 families, 128 genera, 172 species
- Average description length: ~627 characters
- Embedding dimension: 384 (`all-MiniLM-L6-v2`)

### API

```python
# Build from descriptions
index = RAGIndex.build(descriptions, model_name="all-MiniLM-L6-v2")

# Search by semantic similarity
results = index.search("compound opposite leaves with pulvinus", top_k=20, min_similarity=0.3)
# → [{"taxon": "Fabaceae", "rank": "family", "description": "...", "score": 0.87}, ...]

# Persist and restore
index.save("data/traits/latest/rag_index/")
index = RAGIndex.load("data/traits/latest/rag_index/")
```

---

## Dropped Stage: Botanical Catalog Lookup

Originally between Stages 2 and 3. **Dropped** because it injected dataset composition bias:

- `image_count` → species with more photos appear "more likely"
- `forest_types` → collection site bias (where we sampled, not where the plant is)
- `species_count` → families we sampled more appear bigger
- `genera` list → which genera we happened to collect

The only legitimate signal (taxonomy hierarchy) is already known to the vision-LLM from pretraining. Verified against a real demo output: the vision-LLM ignored catalog statistics entirely in its reasoning. Stage 3 (RAG) provides the same taxonomic knowledge through published literature instead, without dataset-specific bias.

---

## Dependencies

### Core (always required)

```
pyyaml          # Config YAML parsing
numpy           # Array operations
```

### Pipeline optional group (`pip install -e ".[pipeline]"`)

```
faiss-cpu           # FAISS vector search (Stage 3 RAG)
openai              # Vision-LLM/LLM API client (Stages 1, 5)
sentence-transformers  # Text embedding for RAG (Stage 3)
```

### Lazy imports

Heavy dependencies are imported inside methods, not at module level:

| Import | Where | Why |
|--------|-------|-----|
| `seedlearn.clip.encoder.FeatureExtractor` | `ClassificationStage._get_extractor()` | Requires `pybioclip` + `torch` |
| `seedlearn.clip.simpleshot.SimpleShot` | `ClassificationStage._fit_classifier()` | Requires `sklearn` |
| `seedlearn.clip.cache.CachedFeatureExtractor` | `ClassificationStage.load_from_cache()` | Requires `torch` |
| `seedlearn.clip.cache.load_multirank_cache` | `ClassificationStage.load_from_multirank_cache()` | Requires `numpy` |
| `seedlearn.data.splits.load_split` | `ClassificationStage.load_from_cache()` / `load_from_multirank_cache()` | Requires `numpy` + `sklearn` |

This means you can `import seedlearn.pipeline` without having `torch`, `pybioclip`, or GPU drivers installed — useful for testing, CI, and non-GPU stages.

---

## Module Map

```
src/seedlearn/pipeline/
├── __init__.py          # Re-exports: PipelineConfig, load_config, PipelineStage,
│                        #   StageResult, PipelineResult, InferenceClient,
│                        #   InferenceConfig, InferenceResponse, build_messages,
│                        #   parse_json_response, strip_thinking
├── config.py            # PipelineConfig + 6 sub-configs + YAML loader + CLI overrides
├── protocol.py          # StageResult dataclass + PipelineStage Protocol
├── result.py            # PipelineResult container (accumulates StageResults)
├── vlm_client.py        # InferenceClient, InferenceConfig, InferenceResponse,
│                        #   build_messages, parse_json_response, strip_thinking
├── rag.py               # RAGIndex (FAISS + sentence-transformers)
└── stages/
    ├── __init__.py
    ├── morphology.py     # Stage 1: MorphologyStage (uses FormParser from analyzers)
    ├── classification.py # Stage 2: ClassificationStage + single/multi-rank dispatch + OOD gating
    ├── trait_retrieval.py# Stage 3: TraitRetrievalStage + _compose_query + _cross_reference
    ├── evidence.py       # Stage 4: EvidenceSynthesisStage + formatting + quality flags
    └── reasoning.py      # Stage 5: ReasoningStage + _SYSTEM_PROMPT + _build_fallback
```

---

## Future Work (Documented, Not Implemented)

| Feature | Stage | Description |
|---------|-------|-------------|
| Knockout/elimination rules | Stage 4 | High-confidence traits as hard filters (e.g., "simple leaves" eliminates most Fabaceae) |
| BioCLIP 2 text encoder for RAG | Stage 3 | Domain-specific embeddings instead of `all-MiniLM-L6-v2` |
| Multi-pass reasoning | Stage 5 | Re-run Stage 5 with different temperatures/prompts, aggregate results |
| LoRA fine-tuning | Stage 2 | Fine-tune BioCLIP 2 on our dataset for improved visual classification |
| Open-vocabulary classification | Stage 2 | Predict ANY species via text-image alignment, not just closed-set |
