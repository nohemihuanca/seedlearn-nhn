# Vision-LLM Benchmark Pipeline

**Navigation**: [SeedLearn](../README.md) > **Vision-LLM Benchmarks**

> **Grading traits against human annotators?** See
> [`docs/human-grading.md`](human-grading.md) for the human-annotation grading
> system (model-vs-human and inter-annotator agreement, plus Roni's species-ID
> accuracy) and its reproducible runbook.

---

## Overview

The benchmark pipeline evaluates vision-language models (vision-LLMs) on morphological trait extraction from seedling images. A vision-LLM examines multi-angle photos of a seedling and fills out a standardized 24-trait morphological assessment form.

**Two-stage concept**:
- **Stage 1** (implemented): Extract morphological traits from images using vision-LLMs
- **Stage 2** (planned): Integrate extracted traits with literature knowledge for species ID

```
tests/benchmarks/
├── sweep_vlm_models.sh        # Multi-model sweep orchestrator
├── common.sh                  # vLLM lifecycle helpers
├── run_vlm_stage1.py          # Core benchmark script
├── score_vlm_stage1.py        # Scoring CLI
├── scoring/                   # Scoring package
│   ├── matcher.py             # Prediction ↔ ground truth matching
│   ├── loader.py              # CSV/JSON loading
│   ├── metrics.py             # Accuracy, confusion, multi-vs-single
│   └── report_html.py         # HTML report generator
└── configs/
    ├── stage1_samples.json    # 21 specimens, ~103 images
    ├── stage1_ground_truth.csv        # Full 24-trait ground truth
    ├── stage1_ground_truth_active.csv # Active 5-trait subset
    └── stage1_trait_valid_values.csv  # Controlled vocabulary
```

---

## `sweep_vlm_models.sh` — Model Sweep

Automated sweep through multiple vision-LLM models, handling vLLM startup/teardown per model.

### Usage

```bash
# Full sweep (all models, all specimens)
./tests/benchmarks/sweep_vlm_models.sh

# Single specimen
./tests/benchmarks/sweep_vlm_models.sh --specimen "Fabaceae_Inga_punctata"

# Single model
./tests/benchmarks/sweep_vlm_models.sh --model "Qwen/Qwen3-VL-32B-Instruct-FP8"

# Both (fastest iteration)
./tests/benchmarks/sweep_vlm_models.sh \
    --specimen "Fabaceae_Inga_punctata" \
    --model "Qwen/Qwen3-VL-32B-Instruct-FP8"
```

### Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--specimen` | `-s` | all 21 | Run only specified specimen |
| `--model` | `-m` | all models | Run only specified model |
| `--samples` | | `tests/benchmarks/configs/stage1_samples.json` | Custom samples file |
| `--help` | `-h` | | Show specimen and model list |

### Configured Models

| Model | Context Limit |
|-------|--------------|
| `Qwen/Qwen3-VL-32B-Instruct-FP8` | 32,768 |
| `Qwen/Qwen3-VL-32B-Thinking-FP8` | 32,768 |
| `RedHatAI/gemma-3-27b-it-FP8-dynamic` | 8,192 |

All models are FP8-quantized to fit H200 (141 GB HBM3).

### Behavior

1. Starts vLLM for each model via `common.sh` helpers
2. Waits for health check, runs benchmark, stops vLLM
3. 5-second GPU cleanup between models
4. Trap-based cleanup on exit
5. Generates comparison HTML report after all models complete

### Per-Model vLLM Parameters

The sweep script configures model-specific vLLM flags and generation parameters.
See `docs/research/morph-improve-03192026/model_vllm_parameters.md` for full
details from each model's HuggingFace card.

Key differences:
- **Gemma 3**: Requires `--enforce-eager` for stability (hybrid KV cache)
- **Gemma 3 FP8**: Also needs `--enable-chunked-prefill`
- **All models**: `temperature=0.1` for structured extraction (vs default 0.6)
- **VRAM-constrained models** (122B, 119B): Auto-downgraded to `--mode single`

---

## `common.sh` — vLLM Lifecycle Helpers

Reusable bash functions sourced by `sweep_vlm_models.sh`.

### Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `log` | `msg` | Timestamped logging `[YYYY-MM-DD HH:MM:SS] msg` |
| `log_section` | `msg` | Section header with decorative border |
| `find_free_port` | | Find available port via Python socket |
| `wait_for_health` | `url timeout_seconds` | Poll `/models` endpoint until healthy or timeout |
| `start_vllm` | `model port [log_file] [max_model_len]` | Start vLLM in background |
| `stop_vllm` | `port` | Gracefully stop vLLM (PID file + port-based kill) |
| `cleanup_vllm` | | Emergency cleanup (registered via trap) |

### vLLM Launch Flags

```bash
vllm serve $model \
    --dtype auto \
    --trust-remote-code \
    --port $port \
    --allowed-local-media-path / \
    --max-model-len $max_model_len \
    --limit-mm-per-prompt '{"image": 10}'
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `VLLM_STARTUP_TIMEOUT` | 300 | Seconds to wait for health |
| `VLLM_HEALTH_INTERVAL` | 10 | Seconds between health polls |
| `HF_HOME` | `$HOME/.cache/huggingface` | Model cache directory |

---

## `run_vlm_stage1.py` — Core Benchmark

Python benchmark orchestrator that calls vLLM's OpenAI-compatible API.

### Usage

```bash
# Run benchmark
python tests/benchmarks/run_vlm_stage1.py \
    --samples tests/benchmarks/configs/stage1_samples.json \
    --model Qwen/Qwen3-VL-32B-Instruct-FP8 \
    --prompt sys4

# List available prompts
python tests/benchmarks/run_vlm_stage1.py --list-prompts

# Generate comparison report from results
python tests/benchmarks/run_vlm_stage1.py --report tests/benchmarks/results/sys4_model_comp/

# Run all prompt styles
python tests/benchmarks/run_vlm_stage1.py \
    --samples tests/benchmarks/configs/stage1_samples.json \
    --model Qwen/Qwen3-VL-32B-Instruct-FP8 \
    --all-prompts
```

### Arguments

**Action arguments** (mutually exclusive with benchmark mode):

| Argument | Description |
|----------|-------------|
| `--list-prompts` | List available prompt styles and exit |
| `--report DIR` | Generate HTML comparison report from results directory |

**Benchmark arguments**:

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--samples` | Path | | **yes** | Samples JSON file |
| `--model` | str | | **yes** | Model name/path |
| `--prompt` | `{sys1..sys4,json}` | `sys1` | | Prompt style |
| `--all-prompts` | flag | | | Run all styles sequentially |
| `--prompt-file` | Path | | | Path to custom prompt text file (overrides --prompt). Uses `load_prompt()` from the configurable prompt system. |
| `--mode` | `{multi,single,both}` | `multi` | | Inference mode: `multi` (all images), `single` (per-image), `both` |
| `--workers`, `-w` | int | `8` | | Concurrent workers for single-image mode. vLLM handles batching via continuous batching. |
| `--examples` | Path | | | Few-shot examples JSON |

**Server arguments**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--port` | int | `8000` | vLLM server port |
| `--vlm-url` | str | | Full vLLM URL (overrides --port) |

**Generation arguments**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max-tokens` | int | `8192` | Max generation tokens |
| `--temperature` | float | `0.6` | Sampling temperature |
| `--top-p` | float | `0.95` | Nucleus sampling |
| `--top-k` | int | `20` | Top-k sampling |
| `--min-p` | float | `-1.0` | Minimum probability |

**Output arguments**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-dir` | Path | `results/vlm_benchmark/` | Output directory |
| `--save-name` | str | | Custom filename prefix |
| `--no-timestamp` | flag | | Omit timestamp from filenames |
| `--quiet` | flag | | Reduce verbosity |

---

## Prompt System

All prompts assign the role: *"You are a botanical expert specializing in identifying seedling plant species."*

| Style | Single/Multi | Output Format | Use Case |
|-------|-------------|---------------|----------|
| `sys1` | single | Key:value + parenthetical justifications | Detailed single-image analysis |
| `sys2` | single | Key:value + notes (cleanest) | Minimal output |
| `sys3` | single | Key:value + detailed expert report | Most verbose |
| `sys4` | **multi** | Key:value + justifications | **Recommended** for multi-angle photos |
| `json` | both | Structured JSON | Machine-readable output |

### Conservative Annotation Rules (sys1, sys4)

1. Do NOT guess or mention family/genus/species
2. Write `unclear` if trait not visible
3. Describe damage neutrally (no inferences about cause)
4. Don't infer habitat unless clearly visible
5. Strict criteria for leaf arrangement classification

---

## 24 Morphological Traits

The assessment form extracts these traits per specimen:

| Section | Traits |
|---------|--------|
| **A. Leaf Arrangement** | leaf relative position, leaf spacing |
| **B. Leaf Complexity** | leaf complexity, compound leaf type, number of leaflets, leaflet arrangement |
| **C. Leaf Morphology** | leaf margin, leaf shape, leaf apex, leaf base, venation type, secondary veins, leaf surface features, leaf surface trichomes |
| **D. Petiole** | petiole length, petiole features |
| **E. Stem & Shoot** | stem type, stem trichomes, stem color, stem texture |
| **F. Other Traits** | stipules, latex, pulvinus, tendrils |
| **G. Notes** | free-text observations |

---

## Sample Dataset

`tests/benchmarks/configs/stage1_samples.json` contains 21 specimens from 17 families, each with 4-5 images.

Image paths reference: `/nfs/roberts/project/pi_lsc4/shared/seedlearn/data/raw/2025-10-23/sorted_12K/training/...`

---

## Output Format

### Per-model output directory

```
results/sys4_model_comp/
└── {timestamp}_{model_safe_name}/
    ├── results.json          # Full responses + thinking chains + metadata
    ├── results.csv           # 24 traits × 21 specimens
    ├── vllm.log              # vLLM server log
    └── summary.txt           # SUCCESS or FAILED status
```

### CSV format (24 trait columns + metadata)

```csv
specimen,image_paths,leaf relative position,leaf spacing,...,notes
Fabaceae_Inga_punctata,"['/path/img1',...]",alternate,clustered,...,""
```

### JSON format

```json
{
  "model": "Qwen/Qwen3-VL-32B-Instruct-FP8",
  "prompt_style": "sys4",
  "generation_kwargs": { "max_tokens": 8192, "temperature": 0.6, ... },
  "num_samples": 21,
  "total_runtime": "12.34 minutes",
  "answers": { "specimen_id": "leaf relative position: alternate\n..." },
  "cots": { "specimen_id": "<thinking block content>" },
  "raw_results": { "specimen_id": "<full response>" }
}
```

---

## Scoring VLM Trait Extraction

The scoring engine (`tests/benchmarks/score_vlm_stage1.py`) evaluates VLM morphological predictions against STRI-derived ground truth. It produces per-trait accuracy, per-specimen scorecards, confusion matrices, and an HTML report.

```
tests/benchmarks/
├── score_vlm_stage1.py               # Scorer CLI
├── scoring/                           # Scoring package
│   ├── __init__.py
│   ├── matcher.py                     # Prediction ↔ ground truth matching
│   ├── loader.py                      # CSV/JSON loading + normalization
│   ├── metrics.py                     # Accuracy, confusion, multi-vs-single
│   └── report_html.py                 # Standalone HTML report generator
└── configs/
    ├── stage1_ground_truth.csv        # Full 24-trait ground truth (21 specimens)
    ├── stage1_ground_truth_active.csv # Active 5-trait subset (STRI traits only)
    ├── stage1_trait_valid_values.csv   # Controlled vocabulary per trait
    └── stage1_test_results.json       # Synthetic data for integration tests
```

### CLI Usage

```bash
# Score a single result directory
python tests/benchmarks/score_vlm_stage1.py \
    --results data/benchmarks/qwen3-vl-32b/ \
    --ground-truth tests/benchmarks/configs/stage1_ground_truth_active.csv \
    --output data/benchmarks/scores/qwen3-vl-32b/

# Score a legacy JSON results file
python tests/benchmarks/score_vlm_stage1.py \
    --results-file data/benchmarks/legacy_results.json \
    --ground-truth tests/benchmarks/configs/stage1_ground_truth_active.csv \
    --output data/benchmarks/scores/legacy/

# Compare multiple runs
python tests/benchmarks/score_vlm_stage1.py \
    --results data/benchmarks/run1/ data/benchmarks/run2/ \
    --ground-truth tests/benchmarks/configs/stage1_ground_truth_active.csv \
    --output data/benchmarks/scores/combined/

# Score single-image mode predictions
python tests/benchmarks/score_vlm_stage1.py \
    --results data/benchmarks/qwen3-vl-32b/ \
    --ground-truth tests/benchmarks/configs/stage1_ground_truth_active.csv \
    --output data/benchmarks/scores/single/ \
    --mode single
```

### Scorer Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--results` | Path(s) | one of | Result directories to score |
| `--results-file` | Path | one of | Legacy JSON results file |
| `--ground-truth` | Path | **yes** | Ground truth CSV |
| `--output` | Path | **yes** | Output directory |
| `--mode` | `multi\|single` | | Inference mode to score (default: `multi`) |

### Ground Truth Format

The ground truth CSV uses the following structure per trait:

| Column Pattern | Description |
|----------------|-------------|
| `{trait}` | Accepted value(s), pipe-delimited for multi-label (e.g., `alternate \| opposite`) |
| `{trait}_source` | Provenance tag (e.g., `stri_consensus`) |
| `{trait}_stri_keys` | STRI identification key IDs used as source |

Empty trait cells mean no ground truth is available for that trait-specimen pair. To add annotations, fill trait columns with the correct value and set the source/key columns for traceability.

### Scoring Semantics

| Outcome | Meaning |
|---------|---------|
| **Exact match** | Normalized prediction matches any pipe-delimited GT value |
| **Parent match** | Prediction is a recognized subtype of a GT value (e.g., `serrate` → `toothed`) |
| **Mismatch** | Prediction does not match any GT value |
| **Abstention** | Model responded with `unclear`, `not visible`, `n/a`, or empty |
| **No ground truth** | GT cell is empty for this trait-specimen pair (excluded from scoring) |

Accuracy is computed over answered cells only (abstentions excluded). Strict accuracy includes abstentions as incorrect.

### Multi-Image vs Single-Image Comparison

When `run_vlm_stage1.py` is called with `--mode both`, it produces `multi/` and `single/` subdirectories. The scorer auto-detects these and computes:

- Per-trait accuracy for multi-image (all photos at once) vs mean single-image accuracy
- Majority vote accuracy across individual images
- Inter-image consistency (agreement rate across single-image predictions)
- Resolution effect (multi accuracy minus single mean accuracy)

### Output Files

| File | Description |
|------|-------------|
| `summary.json` | All metrics in machine-readable format |
| `per_trait_accuracy.csv` | One row per trait: accuracy, strict accuracy, abstention rate |
| `per_specimen_scorecard.csv` | One row per specimen: per-trait predicted vs GT vs result |
| `confusion_matrices/{trait}.json` | Per-trait confusion matrix |
| `multi_vs_single.csv` | Comparison metrics (if both modes available) |
| `report.html` | Standalone HTML report with heatmaps and per-trait breakdowns |

---

## Resource Requirements

| Resource | Recommendation |
|----------|---------------|
| GPU | H200 (141 GB) for full model list |
| CPU | 8 cores |
| Memory | 64 GB system RAM |
| Time | ~15 min per model (21 specimens), ~2 hours full sweep |

### SLURM example

```bash
srun --partition=gpu_h200 --gpus=1 --mem=64G --time=04:00:00 --cpus-per-task=8 --pty bash
source .venv/bin/activate
./tests/benchmarks/sweep_vlm_models.sh
```
