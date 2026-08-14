# CLI Reference

**Navigation**: [SeedLearn](../README.md) > **CLI Reference**

---

## Workflow

### Experiment pipeline (few-shot evaluation)

```
extract_embeddings.py → create_splits.py → run_simpleshot.py → generate_report.py
                                         └─ run_experiments.py ──┘
```

1. **Extract** BioCLIP 2 features (GPU, ~5 min)
2. **Partition** dataset into train/val/test (CPU, seconds)
3. **Run** experiments — single or batch (CPU with cached features)
4. **Report** — single experiment or learning curve (CPU)

### Classification pipeline (per-specimen)

```
run_pipeline.py  (requires cached features, splits, RAG index, and optionally vLLM)
```

5. **Classify** a specimen through all 5 stages — see [`run_pipeline.py`](#run_pipelinepy) below

---

## `extract_embeddings.py`

Extract and cache BioCLIP 2 image embeddings to `.npz` files for reuse across experiments.

```bash
# Multi-rank extraction (default — recommended)
python scripts/extract_embeddings.py --catalog $CATALOG --device cuda

# Single-rank extraction (legacy)
python scripts/extract_embeddings.py --catalog $CATALOG --rank species --device cuda
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--catalog` | Path | `DEFAULT_CATALOG` | Species catalog CSV |
| `--rank` | `{family,genus,species}` | `None` | Taxonomic rank (omit for multi-rank v2 cache) |
| `--cache-dir` | Path | auto | Cache directory (auto-derived from catalog) |
| `--cache-name` | str | `{rank}_features` | Cache filename prefix (single-rank only) |
| `--device` | str | `cuda` | Torch device |
| `--batch-size` | int | auto | Batch size (auto-selected by GPU tier) |
| `--model-str` | str | `hf-hub:imageomics/bioclip-2` | Model identifier |
| `--num-workers` | int | `8` | DataLoader parallel workers |
| `--prefetch-factor` | int | `2` | Batches to prefetch per worker |
| `--no-optimize` | flag | | Disable optimized DataLoader extraction |
| `--force-recompute` | flag | | Recompute even if cache exists |
| `--verbose` | flag | | Debug logging |

### Behavior

- **Multi-rank mode** (default, `--rank` omitted): Extracts features once, stores family/genus/species labels in a single `features.npz` + `features_meta.json` sidecar with taxonomy cross-reference map. This is the recommended mode for the pipeline.
- **Single-rank mode** (`--rank` specified): Legacy behavior, produces `{rank}_features.npz` for backward compatibility with experiment scripts.
- Auto-falls back to CPU if CUDA unavailable
- Auto-selects batch size per GPU tier (H200: 2048, A6000: 1024, etc.)
- Skips extraction if cache already exists (use `--force-recompute` to override)

#### Output files

| Mode | Files | Contents |
|------|-------|----------|
| Multi-rank | `features.npz` + `features_meta.json` | Features, per-rank labels, image paths, individual IDs, taxonomy map |
| Single-rank | `{rank}_features.npz` | Features, labels, image paths |

---

## `create_splits.py`

Create multiple train/val/test partitions with different random seeds.

```bash
# Image-level stratified partitions (default, allows data leakage)
python scripts/create_splits.py \
    --catalog $CATALOG \
    --rank species \
    --num-seeds 5

# Individual-level partitions (prevents leakage, required for honest evaluation)
python scripts/create_splits.py \
    --catalog $CATALOG \
    --rank family \
    --split-type individual \
    --num-seeds 1
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--catalog` | Path | `DEFAULT_CATALOG` | Species catalog CSV |
| `--rank` | `{family,genus,species}` | `species` | Taxonomic rank |
| `--output-dir` | Path | auto | Output directory (auto-derived) |
| `--train-ratio` | float | `0.70` | Training proportion |
| `--val-ratio` | float | `0.15` | Validation proportion |
| `--test-ratio` | float | `0.15` | Test proportion |
| `--num-seeds` | int | `5` | Number of random partitions |
| `--start-seed` | int | `42` | Starting random seed |
| `--split-type` | `{stratified,individual}` | `stratified` | `stratified`: image-level (fast, allows leakage). `individual`: groups by ID_YPS so all images of the same plant stay together (prevents leakage). |
| `--verbose` | flag | | Debug logging |

### Behavior

- Validates ratios sum to 1.0
- Generates N partitions with seeds `start_seed` to `start_seed + num_seeds - 1`
- Output: `{output_dir}/{rank}/split_seed{N}.json` per seed
- Reports feasibility info: guaranteed k-shot, class distribution, warnings

---

## `run_simpleshot.py`

Run a single SimpleShot few-shot learning experiment.

```bash
python scripts/run_simpleshot.py \
    --rank species \
    --split-path splits/species/split_seed42 \
    --k-shot 10 \
    --device cpu
```

### Arguments

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--catalog` | Path | `DEFAULT_CATALOG` | | Species catalog CSV |
| `--rank` | `{family,genus,species}` | | **yes** | Taxonomic rank |
| `--split-path` | Path | | **yes** | Split file path (without extension) |
| `--k-shot` | int | | **yes** | Support examples per class |
| `--cache-dir` | Path | auto | | Cached features directory |
| `--cache-name` | str | auto | | Cache filename |
| `--output-dir` | Path | auto | | Results directory |
| `--device` | str | `cpu` | | Torch device |
| `--support-seed` | int | `42` | | Support set sampling seed |
| `--verbose` | flag | | | Debug logging |

### Output Files

```
{output_dir}/{rank}/{k}_shot/{split_name}/
├── metrics.json              # Accuracy, F1 scores, top-5
├── predictions.csv           # Per-sample predictions
├── per_class_metrics.csv     # Per-class precision/recall/F1
├── confusion_matrix.csv      # Full confusion matrix
├── support_set.json          # Support set metadata
└── experiment_info.json      # Configuration + git hash
```

---

## `run_experiments.py`

Batch orchestration across all combinations of split seeds and k-shot values.

```bash
python scripts/run_experiments.py \
    --rank species \
    --k-shots 1 5 10 20 50 \
    --device cuda
```

### Arguments

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--catalog` | Path | `DEFAULT_CATALOG` | | Species catalog CSV |
| `--rank` | `{family,genus,species}` | | **yes** | Taxonomic rank |
| `--k-shots` | int+ | | **yes** | K-shot values (e.g. `1 5 10 20 50`) |
| `--split-seeds` | int+ | auto | | Split seeds (auto-discovered if omitted) |
| `--splits-dir` | Path | auto | | Splits directory |
| `--device` | str | `cuda` | | Torch device |
| `--support-seed` | int | `42` | | Support set sampling seed |
| `--continue-on-error` | flag | | | Don't stop batch on failure |
| `--skip-reports` | flag | | | Skip auto report generation |
| `--baseline-blind` | float | | | Blind baseline for reports |
| `--baseline-closed` | float | | | Closed-set baseline for reports |
| `--verbose` | flag | | | Debug logging |

### Behavior

- Auto-discovers split seeds from `splits/{rank}/split_seed*.json`
- Pre-validates all split + k-shot combinations before running
- Launches `run_simpleshot.py` as subprocess per experiment (600s timeout)
- Output: `batch_summary_{rank}_{timestamp}.json` with all results

### Batch Summary Format

```json
{
  "config": { "catalog": "...", "rank": "species", "k_shots": [1,5,10,20,50], ... },
  "summary": { "total": 25, "successful": 25, "failed": 0, "runtime_seconds": 120 },
  "results": [
    { "experiment_name": "...", "k_shot": 10, "split_seed": 42,
      "success": true, "accuracy": 0.723, "output_dir": "..." }
  ]
}
```

---

## `generate_report.py`

Generate interactive HTML reports with Plotly visualizations.

### Subcommand: `single`

Detailed report for a single experiment.

```bash
python scripts/generate_report.py single \
    --experiment-dir results/species/10_shot/split_seed42/
```

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--experiment-dir` | Path | | **yes** | Experiment output directory |
| `--rank` | `{family,genus,species}` | auto | | Auto-detected from `experiment_info.json` |
| `--output-dir` | Path | `{exp}/visual` | | Output directory |
| `--top-k` | int | `0` | | Show only top-k labels (0 = all) |
| `--baseline-blind` | float | | | Blind baseline reference line |
| `--baseline-closed` | float | | | Closed-set baseline reference line |

**Output**: `simpleshot_report.html` — interactive report with summary table, support distribution, per-class metrics, confusion matrix, top errors, and support vs. performance correlation.

### Subcommand: `learning-curve`

Learning curve across k-shot values with confidence intervals.

```bash
python scripts/generate_report.py learning-curve \
    --rank species \
    --baseline-blind 0.0167 \
    --baseline-closed 0.1133
```

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--rank` | `{family,genus,species}` | | **yes** | Taxonomic rank |
| `--batch-summary` | Path | auto | | Batch summary JSON (auto-discovers most recent) |
| `--output-dir` | Path | auto | | Output directory |
| `--baseline-blind` | float | | | Zero-shot baseline reference line |
| `--baseline-closed` | float | | | Closed-set baseline reference line |

**Output**: `learning_curve_report.html` — Top-1 and Top-5 accuracy curves with 95% CI error bars, baseline reference lines, and marginal improvement chart.

### Subcommand: `pipeline`

Generate HTML report from a pipeline result JSON file.

```bash
python scripts/generate_report.py pipeline \
    --result-json results/pipeline/SRAPHEDE2.json
```

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--result-json` | Path | | **yes** | Pipeline result JSON file |
| `--output` | Path | `{json}.html` | | Output HTML path |

**Output**: Interactive HTML report with per-stage evidence sections, Plotly charts, convergence analysis, timing waterfall, and quality flags.

---

## `run_pipeline.py`

Run the 5-stage classification pipeline on one or more specimen images.

```bash
# Stages 2-4 only (no vLLM needed)
python scripts/run_pipeline.py \
    --images /path/to/img1.jpg /path/to/img2.jpg \
    --cache-dir data/embeddings/2026-01-29_v2026-01-29_12K \
    --split-path data/splits/2026-01-29_v2026-01-29_12K/family/split_seed42 \
    --rag-index data/traits/latest/rag_index/ \
    --skip morphology reasoning \
    --device cpu

# Full pipeline with specimen lookup
python scripts/run_pipeline.py \
    --specimen PP123 \
    --catalog $CATALOG \
    --cache-dir ... --split-path ... --rag-index ...
```

### Arguments

**Input:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--images` | Path+ | | Image file paths (at least one of `--images` or `--specimen` required) |
| `--specimen` | str | | Specimen ID for output naming; resolves images from catalog if `--images` not given |

**Data artifacts:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--catalog` | Path | | Species catalog CSV (required for `--specimen` lookup) |
| `--cache-dir` | Path | | BioCLIP 2 cached feature `.npz` directory (Stage 2) |
| `--split-path` | Path | | Split file path without extension (Stage 2) |
| `--rag-index` | Path | | Pre-built FAISS RAG index directory (Stage 3) |

**Configuration:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | Path | | Pipeline YAML config file |
| `--skip` | str+ | | Stage names to skip (e.g. `morphology reasoning`) |

**Vision-LLM overrides (Stage 1):**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--vlm-model` | str | | Vision-LLM model name |
| `--vlm-endpoint` | str | | OpenAI-compatible endpoint URL |
| `--prompt-style` | str | | Prompt style (`sys1`–`sys4`, `json`) |
| `--image-mode` | str | | Image encoding mode |

**Classifier overrides (Stage 2):**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--rank` | str | | Taxonomic rank |
| `--k-shot` | int | | Support examples per class |
| `--top-k` | int | | Number of top predictions |
| `--device` | str | | Torch device |

**Reasoning overrides (Stage 5):**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--reasoning-model` | str | | Text LLM model name |
| `--reasoning-endpoint` | str | | Text LLM endpoint URL |

**Output:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-dir` | str | | Results output directory |
| `--report` | flag | | Generate HTML report alongside JSON output |
| `--verbose` | flag | | Debug logging |

### Behavior

- At least one of `--images` or `--specimen` is required
- `--specimen` without `--images` requires `--catalog` to resolve image paths from the catalog CSV
- **Multi-rank auto-detection**: If `features.npz` exists in `--cache-dir`, the pipeline uses multi-rank classification (family + genus + species simultaneously) with hierarchical consistency checking. It auto-discovers sibling split directories for genus and species. Falls back to single-rank mode if only `{rank}_features.npz` is found.
- Stage 2 (classification) requires `--cache-dir` and `--split-path`; warns if missing and stage not skipped
- Stage 3 (trait retrieval) requires `--rag-index`; warns if missing and stage not skipped
- Stages 1 and 5 require a running vLLM server; skip them with `--skip morphology reasoning` for CPU-only testing
- Output: `{output_dir}/{specimen_id}.json` (+ `.html` if `--report`) with all stage results

### Pipeline Stages

```
Stage 1: Morphology (Vision-LLM)  →  Stage 2: Classification (SimpleShot)
    →  Stage 3: Trait Retrieval (RAG)  →  Stage 4: Evidence Synthesis
    →  Stage 5: Reasoning (LLM)
```

See [Pipeline Reference](pipeline.md) for full stage documentation.

---

## iNaturalist Data Pipeline

Three scripts for data acquisition from iNaturalist. See [iNaturalist Pipeline](inaturalist.md) for full documentation.

| Script | Purpose |
|--------|---------|
| `download_inaturalist.py` | Download observations from iNaturalist API |
| `convert_inaturalist.py` | Convert photo-level metadata CSV to individual-level species CSV |
| `sort_inaturalist.py` | Sort downloaded images into ML-ready hierarchical directory structure |

---

## STRI Trait Scraper Pipeline

### `scrape_stri_traits.py`

Scrape morphological traits from STRI Panama Biota identification keys into per-key
trait matrix CSVs.

```bash
# Scrape all 11 identification keys
python scripts/scrape_stri_traits.py --keys all

# Scrape specific keys (by checklist ID)
python scripts/scrape_stri_traits.py --keys 59 178 185

# Smallest key first for testing
python scripts/scrape_stri_traits.py --keys 70 --verbose

# Force re-fetch cached HTML
python scripts/scrape_stri_traits.py --keys 59 --force-refresh
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--keys` | str+ | `all` | Checklist IDs to scrape (e.g., `59 178`) or `all` |
| `--output-dir` | Path | `data/traits/stri_web_keys/` | Output directory |
| `--delay` | float | `1.0` | Seconds between HTTP requests |
| `--force-refresh` | flag | | Re-fetch HTML even if cached locally |
| `--verbose` | flag | | Debug logging |

#### Output

```
{output_dir}/
├── raw_html/cl{id}_{slug}/                Cached HTML responses
│   ├── unfiltered_all_species.html
│   ├── filtered_attr_1-1_habit_tree.html
│   └── ...
└── per_key_trait_matrices/
    ├── cl{id}_{slug}_trait_matrix.csv      Species x trait presence/absence
    └── cl{id}_{slug}_scrape_metadata.json  Timestamps, schema, counts
```

See [Web Scraper](webscraper.md) for data format details and trait schema.

---

### `merge_stri_trait_sources.py`

Merge per-key trait matrices into a unified multi-source database with provenance
tracking and consensus columns.

```bash
# Merge with default paths
python scripts/merge_stri_trait_sources.py --verbose

# Custom input/output directories
python scripts/merge_stri_trait_sources.py \
    --input-dir data/traits/stri_web_keys/per_key_trait_matrices \
    --output-dir data/traits/stri_web_keys/merged
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input-dir` | Path | `data/traits/stri_web_keys/per_key_trait_matrices` | Directory with per-key CSVs |
| `--output-dir` | Path | `data/traits/stri_web_keys/merged` | Output directory |
| `--verbose` | flag | | Debug logging |

#### Output

```
{output_dir}/
├── stri_all_sources_merged_trait_matrix.csv     Source-tagged columns + consensus
├── stri_all_sources_consensus_trait_matrix.csv   Consensus only (any-true across sources)
└── merge_report.json                            Source stats, species counts, coverage
```

---

## VLM Benchmark Pipeline

### `run_vlm_stage1.py` — Run VLM morphological extraction benchmark

Benchmark orchestrator that calls vLLM's OpenAI-compatible API to extract 24
morphological traits from seedling images. See [benchmarks.md](benchmarks.md)
for full documentation.

**Usage:**

```bash
python tests/benchmarks/run_vlm_stage1.py \
    --samples tests/benchmarks/configs/stage1_samples.json \
    --model Qwen/Qwen3-VL-32B-Instruct-FP8 \
    --prompt sys4
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--samples` | — | Samples JSON file (required) |
| `--model` | — | Model name/path (required) |
| `--prompt` | `sys1` | Prompt style (`sys1`–`sys4`, `json`) |
| `--prompt-file` | None | Path to custom prompt text file (overrides --prompt) |
| `--mode` | `multi` | Inference mode: `multi` (all images), `single` (per-image), `both` |
| `--workers` | `8` | Concurrent workers for single-image mode |
| `--all-prompts` | — | Run all styles sequentially |
| `--port` | `8000` | vLLM server port |
| `--output-dir` | `results/vlm_benchmark/` | Output directory |
| `--list-prompts` | — | List available prompt styles and exit |
| `--report` | — | Generate HTML comparison report from results directory |

---

### `score_vlm_stage1.py` — Score VLM benchmark results against ground truth

Scores VLM morphological trait extraction results against a ground truth CSV.
Produces per-trait accuracy, confusion matrices, per-specimen scorecards,
multi-vs-single comparison, and an interactive HTML report.

**Usage:**

```bash
# Score a single run
python tests/benchmarks/score_vlm_stage1.py \
    --results results/vlm_benchmark/run1/ \
    --ground-truth tests/benchmarks/configs/stage1_ground_truth_active.csv \
    --output results/vlm_benchmark/run1/scores/

# Score a legacy result file
python tests/benchmarks/score_vlm_stage1.py \
    --results-file results/vlm_benchmark/model_sys4_results.json \
    --ground-truth tests/benchmarks/configs/stage1_ground_truth_active.csv \
    --output /tmp/scores/
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--results` | — | One or more result directory paths |
| `--results-file` | — | Legacy result JSON file (alternative to --results) |
| `--ground-truth` | — | Path to ground truth CSV (required) |
| `--output` | — | Output directory for scores (required) |
| `--mode` | `multi` | Which results to score: `multi` or `single` |

**Outputs:** summary.json, per_trait_accuracy.csv, per_specimen_scorecard.csv, confusion_matrices/\*.json, multi_vs_single.csv (if both modes present), report.html

---

### `sweep_vlm_models.sh` — Automated multi-model benchmark sweep

Cycles through VLM models, starting/stopping vLLM for each, running benchmarks,
and auto-scoring results against ground truth.

**Usage:**

```bash
# Full sweep (all registered models)
./tests/benchmarks/sweep_vlm_models.sh

# Single model
./tests/benchmarks/sweep_vlm_models.sh --model "Qwen/Qwen3-VL-32B-Thinking-FP8"

# Custom mode and workers
./tests/benchmarks/sweep_vlm_models.sh --mode both --workers 8
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | all | Run only this model |
| `--specimen` | all | Run only this specimen |
| `--mode` | `both` | Benchmark mode: multi, single, both |
| `--workers` | `8` | Concurrent workers for single mode |
| `--no-score` | — | Skip automatic scoring after each run |

---

## Typical Workflows

### Full experiment sweep

```bash
CATALOG="data/raw/2026-01-29/sorted_12K/metadata/species_catalog_v2026-01-29_12K_20260129_123334.csv"

# GPU node
srun --partition=gpu_h200 --gpus=1 --mem=32G --time=04:00:00 --cpus-per-task=8 --pty bash
source .venv/bin/activate

# 1. Extract (once per catalog)
python scripts/extract_embeddings.py --catalog $CATALOG --rank species --device cuda

# 2. Partition (once per catalog)
python scripts/create_splits.py --catalog $CATALOG --rank species

# 3. Batch experiments
python scripts/run_experiments.py --catalog $CATALOG --rank species \
    --k-shots 1 5 10 20 50 --device cuda

# 4. Learning curve
python scripts/generate_report.py learning-curve --rank species \
    --baseline-blind 0.0167 --baseline-closed 0.1133
```

### Quick single experiment

```bash
python scripts/run_simpleshot.py \
    --rank family \
    --split-path splits/family/split_seed42 \
    --k-shot 10

python scripts/generate_report.py single \
    --experiment-dir results/family/10_shot/split_seed42/
```

### STRI trait scraping

```bash
source .venv/bin/activate

# 1. Scrape smallest key first to verify
python scripts/scrape_stri_traits.py --keys 70 --verbose

# 2. Scrape all 10 identification keys (~5 min)
python scripts/scrape_stri_traits.py --keys all

# 3. Merge per-key matrices into unified database
python scripts/merge_stri_trait_sources.py --verbose
```
