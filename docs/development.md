# Development Guide

**Navigation**: [SeedLearn](../README.md) > **Development Guide**

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Platform | RHEL 8+ | Tested on Yale HPC (Roberts cluster) |
| Python | 3.13+ | Managed via `uv` |
| CUDA | 12.x | Required for feature extraction and vision-LLM benchmarks |
| GPU | H200 / H100 / A100 | H200 recommended for vision-LLM models |

---

## Installation

```bash
# 1. Create virtual environment
module load uv
uv venv -p 3.13
source .venv/bin/activate

# 2. Install PyTorch with CUDA support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# 3. Install seedlearn (editable) with all dependency groups
uv pip install -e ".[clip,pipeline,dev]"

# 4. Install pybioclip + vLLM (custom wheels for RHEL 8)
uv pip install /nfs/roberts/project/pi_lsc4/shared/seedlearn/software/py_wheels/pybioclip-*.whl
uv pip install /nfs/roberts/project/pi_lsc4/shared/seedlearn/software/py_wheels/vllm-*.whl
```

### Verify

```bash
python -c "from seedlearn.clip import SimpleShot; print('ok')"
python -c "from bioclip.predict import BaseClassifier; print('ok')"
```

### Dependency Groups

| Group | Packages | Purpose |
|-------|----------|---------|
| core | torch, numpy, pandas, scikit-learn, scipy, pillow, tqdm, plotly, pyyaml | Base functionality |
| `.[clip]` | open-clip-torch, pybioclip, timm | BioCLIP 2 model loading + feature extraction |
| `.[pipeline]` | faiss-cpu, openai, sentence-transformers | 5-stage pipeline (RAG, LLM inference) |
| `.[dev]` | pytest, pytest-cov | Testing |

### Key CLIP Package Versions (Verified 2026-02-12)

| Package | Required Version | Purpose |
|---------|-----------------|---------|
| `pybioclip` | 2.1.1 ([custom wheel](https://github.com/mitchellxh/pybioclip), commit `ddeb58f`) | High-level API for BioCLIP models, TreeOfLife preprocessing, GPU-accelerated aggregation |
| `open-clip-torch` | >= 3.2.0 | Model loading framework (loads BioCLIP 2 weights from HF Hub via `create_model_from_pretrained`) |
| `torch` | >= 2.0 (CUDA 12.x) | GPU inference, `torch.compile()` optimization |

pybioclip wraps open_clip with BioCLIP-specific preprocessing and `torch.compile()`. The `open-clip-torch` package is the **loading framework** — it provides the ViT architecture and HuggingFace Hub download machinery. The actual model weights come from `hf-hub:imageomics/bioclip-2` (1.7 GB, ViT-L/14 trained on 214M organism images), NOT generic OpenCLIP/LAION weights. See [Embedding Architecture Reference](clip/embedding-architecture.md) for the full software stack diagram and technical details.

### Custom Wheels

Both `pybioclip` and `vLLM` are distributed as pre-built wheels on shared NFS:

```
/nfs/roberts/project/pi_lsc4/shared/seedlearn/software/
├── py_wheels/
│   ├── pybioclip-2.1.1-py3-none-any.whl     # Fork with GPU-accelerated aggregation (ddeb58f)
│   └── vllm-0.14.0rc2...-cp313-linux.whl    # Source-built for RHEL 8 glibc 2.28
```

| Wheel | Source | Why custom |
|-------|--------|------------|
| `pybioclip` | [mitchellxh/pybioclip](https://github.com/mitchellxh/pybioclip) commit `ddeb58f` | Adds GPU-accelerated taxonomy aggregation via precomputed lookup tables |
| `vLLM` | See [vllm-install.md](vllm-install.md) | RHEL 8 glibc 2.28 < required 2.31 for PyPI wheels |

> **TODO — Reproducibility**: Clone the pybioclip fork to the shared NFS directory
> (`/nfs/.../software/pybioclip/`) so the source is co-located with the wheel.
> This provides an audit trail from wheel back to exact source. Rebuild the wheel
> with a version suffix (e.g. `2.1.1+gpu1`) to distinguish it from upstream.

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

### Coverage

```bash
pytest tests/ --cov=seedlearn --cov-report=term-missing
```

Target: **80% coverage** minimum.

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── test_cache.py        # CachedFeatureExtractor tests
│   ├── test_catalog.py      # Catalog loading, ImageRecord, label formatting
│   ├── test_parsers.py      # Vision-LLM response parsing
│   └── test_simpleshot.py   # SimpleShot classifier tests
└── benchmarks/              # Vision-LLM model evaluation
    ├── run_vlm_stage1.py    # Core benchmark script
    ├── sweep_vlm_models.sh  # Multi-model sweep
    └── common.sh            # vLLM lifecycle helpers
```

---

## Project Conventions

### Code Style

- **PEP 8** — enforced
- **Type hints** — required on all function signatures
- **Docstrings** — Google style with `Args`, `Returns`, `Raises` sections
- **Imports** — standard lib first, then third-party, then local

### Architecture

Follow [SOLID principles](https://en.wikipedia.org/wiki/SOLID):
- **Single Responsibility**: One module, one purpose (e.g. `encoder.py` only extracts features)
- **Open/Closed**: `FewShotClassifier` ABC allows new classifiers without modifying existing code
- **Dependency Inversion**: Scripts depend on `seedlearn` package abstractions, not concrete implementations

### Git Conventions

- Branch: `dev` for active development, `main` for stable releases
- Commit messages: imperative mood, describe *why* not *what*
- No `Co-Authored-By: Claude` in commits

### File Hygiene

- No dead code — delete, don't comment out
- No unused imports

---

## How Data Directories Auto-Derive

Output paths are derived from the catalog filename by `get_catalog_version()`:

```python
from seedlearn.data.constants import SHARED_EMBEDDINGS, SHARED_SPLITS, get_catalog_version

# Catalog: species_catalog_v2026-01-29_12K_20260129_123334.csv
#                           ^^^^^^^^^^^^^^^^^^
# Extracts: "2026-01-29_v2026-01-29_12K"

version = get_catalog_version(catalog_path)
embeddings = SHARED_EMBEDDINGS / version   # data/embeddings/2026-01-29_v2026-01-29_12K/
splits     = SHARED_SPLITS / version       # data/splits/2026-01-29_v2026-01-29_12K/
```

This means the same catalog always maps to the same data directories, so embeddings and splits are automatically co-located by dataset version.

---

## Adding New Vision-LLM Models to Benchmark Sweep

1. Add the model to the `MODELS` array in `tests/benchmarks/sweep_vlm_models.sh`
2. Add context limit to `MODEL_CONTEXT` associative array
3. Verify the model fits in GPU memory (check with `nvidia-smi` during load)
4. Test with a single specimen first:
   ```bash
   ./tests/benchmarks/sweep_vlm_models.sh \
       --model "org/new-model-name" \
       --specimen "Fabaceae_Inga_punctata"
   ```

---

## Adding New Experiments

### New taxonomic rank or dataset

1. Prepare a new species catalog CSV following the [21-column schema](data.md#species-catalog-csv-schema-21-columns)
2. Update `DEFAULT_CATALOG` in `src/seedlearn/data/constants.py`
3. Run the full pipeline: extract → split → experiment → report

### New classifier

1. Create a new class inheriting from `FewShotClassifier` in `seedlearn/clip/`
2. Implement `fit()` and `predict()` methods
3. Add a new script in `scripts/` or extend `run_simpleshot.py`
4. Add tests in `tests/unit/`

### New prompt style for vision-LLM benchmarks

1. Add a new `SYSTEM_PROMPT_N` constant in `src/seedlearn/components/analyzers/prompts.py`
2. Add the new style to `PromptStyle` enum
3. Register it in `PROMPTS` dict and (if multi-image) `MULTI_IMAGE_PROMPTS` set
4. Update `list_prompts()` description

