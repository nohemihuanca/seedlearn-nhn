# BioCLIP 2 Embedding Architecture — Definitive Reference

**Navigation**: [SeedLearn](../../README.md) > [Pipeline Reference](../pipeline.md) > **Embedding Architecture**

**Date**: 2026-02-12
**Status**: Verified against installed packages and model weights
**Supersedes**: Portions of `bioclip2_findings.md` (Jan 2026); that document retains the fine-tuning roadmap

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Software Stack](#software-stack)
3. [Model Architecture](#model-architecture)
4. [Pre-Training: How the Latent Space Was Shaped](#pre-training-how-the-latent-space-was-shaped)
5. [Inference Pipeline: Image to 768-dim Vector](#inference-pipeline-image-to-768-dim-vector)
6. [The Shared Contrastive Space](#the-shared-contrastive-space)
7. [Preprocessing Pipeline](#preprocessing-pipeline)
8. [How SimpleShot Leverages the Embedding Space](#how-simpleshot-leverages-the-embedding-space)
9. [BioCLIP 1 vs BioCLIP 2](#bioclip-1-vs-bioclip-2)
10. [Correctness Checklist: What Could Go Wrong](#correctness-checklist-what-could-go-wrong)
11. [GPU Requirements for Extraction](#gpu-requirements-for-extraction)
12. [Current Classification Output: What Gets Returned](#current-classification-output-what-gets-returned)
13. [Limitations of Current Implementation](#limitations-of-current-implementation)
14. [Improvement Paths (Ordered by Impact vs Effort)](#improvement-paths-ordered-by-impact-vs-effort)
15. [Future: LoRA Fine-Tuning Path](#future-lora-fine-tuning-path)
16. [Sources](#sources)

---

## Executive Summary

BioCLIP 2 is a 428M-parameter Vision Transformer (ViT-L/14) whose weights were trained on 214 million organism images paired with taxonomic text. We use **only the image encoder** (304M params) to convert seedling photos into 768-dimensional vectors. These vectors land in a space where taxonomically similar organisms cluster together — not because we inject any text at inference time, but because the pre-training contrastive loss reshaped the ViT's weights to extract visual features that correlate with taxonomy.

**This is not a static embedder.** It is a deep neural network with 24 transformer layers, 302M trainable parameters in self-attention and MLP blocks, plus a learned linear projection. Every pixel passes through learned attention patterns that detect biologically relevant features — leaf venation, cotyledon shape, stem pubescence — because those features predicted taxonomic identity during pre-training.

SimpleShot works on these embeddings because same-family images produce similar vectors. Computing a class centroid (average vector) and assigning queries to the nearest centroid is sufficient when the embedding space already encodes taxonomic structure.

---

## Software Stack

Three software components work together. Understanding their roles prevents confusion about what does what.

### Component Roles

```
┌─────────────────────────────────────────────────────────┐
│  pybioclip 2.1.1                                        │
│  High-level Python API for BioCLIP models               │
│  • BaseClassifier: wraps model loading + preprocessing  │
│  • create_image_features(): batch embedding extraction   │
│  • TreeOfLife-specific preprocessing (stretch-to-square) │
│  • torch.compile() optimization                         │
│  • CLI tools: bioclip predict, bioclip embed            │
│  • Pre-computed text embeddings for zero-shot taxonomy   │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  open-clip-torch 3.2.0                            │  │
│  │  OpenCLIP: open-source CLIP implementation        │  │
│  │  • Model architecture definitions (CLIP, ViT)     │  │
│  │  • Weight loading from HuggingFace Hub            │  │
│  │  • create_model_and_transforms() API              │  │
│  │  • encode_image(), encode_text() methods          │  │
│  │  • Standard CLIP preprocessing pipeline           │  │
│  │                                                   │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  Model Weights (HuggingFace Hub)            │  │  │
│  │  │  imageomics/bioclip-2                       │  │  │
│  │  │  • open_clip_model.safetensors (1.7 GB)     │  │  │
│  │  │  • open_clip_config.json (architecture)     │  │  │
│  │  │  • Auto-downloaded on first use             │  │  │
│  │  │  • Cached at ~/.cache/huggingface/hub/      │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Is pybioclip Required?

**For our current pipeline: yes**, because our `FeatureExtractor` class (`src/seedlearn/clip/encoder.py`) imports `bioclip.predict.BaseClassifier`. pybioclip handles:

1. Model loading with `torch.compile()` optimization
2. TreeOfLife-specific preprocessing (see [Preprocessing Pipeline](#preprocessing-pipeline))
3. Correct handling of the `encode_image()` return value

You **could** use `open_clip` directly — `open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')` loads the same model with the same weights. But you'd get different preprocessing (see below) and miss the `torch.compile()` optimization.

### Installed Versions (Verified 2026-02-12)

| Package | Installed | Latest on PyPI | Status |
|---------|-----------|---------------|--------|
| `pybioclip` | 2.1.1 | 2.1.1 (Sept 2025) | Current |
| `open-clip-torch` | 3.2.0 | 3.2.0 (Sept 2025) | Current |
| `torch` | 2.7.1+cu129 | — | CUDA 12.9 |

pybioclip is installed from a custom wheel at `/nfs/roberts/project/pi_lsc4/shared/seedlearn/software/py_wheels/` (fork of Imageomics/pybioclip v2.1.1, commit `ddeb58f` — adds GPU-accelerated taxonomy aggregation).

---

## Model Architecture

BioCLIP 2 is a **dual-encoder CLIP model** with a vision encoder and a text encoder. At inference, we use only the vision encoder.

### Full Model

```
BioCLIP 2 (CLIP)                    Total: 427.6M parameters
├── Visual Encoder (ViT-L/14)       304.0M parameters
│   ├── Patch Embedding (Conv2d)    602K params
│   │   14×14 patches → 1024-dim tokens
│   │   Input: (B, 3, 224, 224) → Output: (B, 256, 1024)
│   │   (224/14 = 16 patches per side → 256 patch tokens + 1 CLS token)
│   │
│   ├── Transformer (24 layers)     302.3M params
│   │   Each layer:
│   │   ├── LayerNorm               2,048 params
│   │   ├── Multi-Head Attention    4.2M params (16 heads, 1024-dim)
│   │   ├── LayerNorm               2,048 params
│   │   └── MLP (1024→4096→1024)    8.4M params
│   │
│   ├── LayerNorm (post)            2,048 params
│   │
│   └── Projection (linear)         786K params
│       1024-dim → 768-dim (learned linear map to shared space)
│
└── Text Encoder                    123.7M parameters
    (12 layers, 768-dim, 12 heads)
    DISCARDED at inference — only used during pre-training
```

### What Makes This a Transformer (Not a Static Embedder)

A **static embedder** would be a fixed mathematical transformation: PCA, random projection, histogram of gradients, bag-of-visual-words. No learning. No ability to adapt what features it extracts.

BioCLIP 2's visual encoder is the opposite:

1. **302M learned parameters** in 24 self-attention layers. Each parameter was optimized during 30 epochs over 214M images.
2. **Self-attention** allows every 14×14 patch to attend to every other patch. The model learns which spatial relationships matter — a leaf tip attending to the stem base to assess phyllotaxis, cotyledons attending to each other to assess symmetry.
3. **Hierarchical feature extraction**: Early layers learn edges and textures. Middle layers learn organ-level features (leaf shape, venation patterns). Late layers learn whole-plant taxonomy-discriminative features.
4. **The projection layer is learned** — the 1024×768 matrix was optimized during pre-training to map visual features into the shared contrastive space. A static embedder would use a random or identity projection.

**Empirical proof**: If you swap BioCLIP 2's weights for random initialization (same architecture), the embeddings are random noise. If you swap for ImageNet-trained ViT-L/14 weights, embeddings cluster by visual similarity (background, lighting) not taxonomy. The weights ARE the embedding quality.

---

## Pre-Training: How the Latent Space Was Shaped

### Dataset: TreeOfLife-200M

| Property | Value |
|----------|-------|
| Total images | ~214 million |
| Taxa covered | 952,257 |
| Primary source | GBIF citizen science (~151M images) |
| Additional sources | GBIF museum/herbarium specimens (~52M), BIOSCAN-5M, EOL, FathomNet |
| Experience replay | 26M LAION-2B samples (prevents catastrophic forgetting) |

### Training Objective: Hierarchical Contrastive Learning

The training used **standard CLIP contrastive loss** (symmetric InfoNCE), but the "hierarchy" comes from how taxonomic text is formatted. Each image was paired with one of 5 text variations, randomly selected per step:

| Format | Example |
|--------|---------|
| `sci` | "Acacia dealbata" |
| `com` | "Silver Wattle" |
| `taxon` | "Plantae Tracheophyta Magnoliopsida Fabales Fabaceae Acacia dealbata" |
| `sci_com` | "Acacia dealbata Silver Wattle" |
| `taxon_com` | "Plantae Tracheophyta Magnoliopsida Fabales Fabaceae Acacia dealbata Silver Wattle" |

**Why this creates hierarchy**: The text encoder processes tokens left-to-right. When it encodes "Plantae Tracheophyta Magnoliopsida **Fabales Fabaceae** Acacia dealbata", the attention mechanism means the token for "Fabaceae" can attend to all preceding tokens (Kingdom, Phylum, Class, Order). Two species in the same family share the first 5 tokens of their `taxon` string, so their text embeddings are naturally more similar than species in different families. The contrastive loss then pulls image embeddings of same-family species closer together, creating a hierarchical embedding geometry.

### What the Contrastive Loss Actually Does

```
Batch of (image, text) pairs:
  (photo of Fabaceae seedling)  ←→  "Fabaceae legume family"    → push CLOSE
  (photo of Fabaceae seedling)  ←→  "Poaceae grass family"      → push APART

This does NOT inject text into image vectors.
This RESHAPES the ViT's weights so that:
  - Images whose text matches → similar 768-dim vectors
  - Images whose text differs → dissimilar 768-dim vectors
```

After training: the ViT has learned which visual features correlate with taxonomy (leaf arrangement, venation, cotyledon morphology) and which don't (pot color, camera angle, lighting). Features that didn't help predict the text got suppressed by the contrastive loss.

### Experience Replay (Catastrophic Forgetting Prevention)

BioCLIP 2 was initialized from a CLIP ViT-L/14 pre-trained on LAION-2B (2 billion general internet images). Fine-tuning on biological images alone would cause the model to "forget" general visual understanding. To prevent this, 26M LAION-2B samples were interleaved during training (2,816 bio samples + 320 LAION samples per GPU per step).

The training code uses a **dual visual projector** design during training:
- Primary projector (`proj`): Maps to shared space for taxonomic contrastive loss
- Continual projector (`continual_proj`): Maps to shared space for LAION caption contrastive loss

**At inference, only the primary projector is present in the released weights.** The continual projector was used only during training to prevent forgetting and is not included in the HuggingFace-hosted model. We verified: `model.visual.continual_proj` does not exist in the loaded model.

### Training Infrastructure

| Parameter | Value |
|-----------|-------|
| Hardware | 32× NVIDIA H100-80GB (4 nodes) |
| Duration | 30 epochs, ~10 days |
| Optimizer | AdamW, max LR 1e-4 |
| Precision | bfloat16 |
| Batch (per GPU) | 2,816 bio + 320 replay |
| Global batch | ~100K effective |
| Initialization | CLIP ViT-L/14 (LAION-2B) |

**Source**: [BioCLIP 2 Paper (ArXiv 2505.23883)](https://arxiv.org/abs/2505.23883)

---

## Inference Pipeline: Image to 768-dim Vector

This is what actually happens when our code calls `encode_image()`:

```
Input: RGB image (any size)
  │
  ▼
Preprocessing (pybioclip's TreeOfLife transform):
  ToTensor() → Resize(224, 224) → Normalize(mean, std)
  │
  ▼
Patch Embedding (Conv2d, 14×14 kernel, stride 14):
  (B, 3, 224, 224) → (B, 256, 1024)
  16×16 grid of patches, each projected to 1024-dim
  │
  ▼
Prepend CLS token + add positional embeddings:
  (B, 257, 1024)
  │
  ▼
24× Transformer Layers:
  Each: LayerNorm → MultiHead Self-Attention (16 heads) → LayerNorm → MLP
  (B, 257, 1024) → (B, 257, 1024)
  │
  ▼
Pool (CLS token extraction):
  (B, 257, 1024) → (B, 1024)
  │
  ▼
LayerNorm:
  (B, 1024) → (B, 1024)
  │
  ▼
Learned Linear Projection (visual.proj):
  (B, 1024) @ proj[1024, 768] → (B, 768)
  │
  ▼
L2 Normalization (optional, applied by our code):
  (B, 768) → (B, 768) with unit norm
  │
  ▼
Output: 768-dim embedding vector on the unit hypersphere
```

**The text encoder follows a similar path** (token embedding → 12 transformer layers → projection → 768-dim), but we never use it at inference. It was only needed during pre-training to provide the contrastive signal that shaped the visual encoder's weights.

**Analogy**: Train a student with flash cards — photo on front, family name on back. After months of study, take away the cards and show new photos. They sort by family because they internalized which visual features correlate with each family. The flash cards (text) are gone; the perceptual skill persists in how they look at photos. That's what BioCLIP 2's frozen ViT weights are — internalized perceptual skill shaped by taxonomic text, applied to pixels alone.

---

## The Shared Contrastive Space

### What "Shared Space" Means

The 768-dim output from the image encoder and the 768-dim output from the text encoder live in the **same vector space**. Cosine similarity between an image embedding and a text embedding is meaningful — it measures how well the image matches the text's semantic content.

**Verified empirically** (2026-02-12):

```python
img_features = model.encode_image(img, normalize=True)    # shape: (1, 768)
text_features = model.encode_text(text, normalize=True)    # shape: (4, 768)
similarities = img_features @ text_features.T              # cosine similarity
# Both are L2-normalized, so dot product = cosine similarity
```

Result with a random image:
```
"Fabaceae"                          → 0.384 cosine sim
"Poaceae"                           → 0.417 cosine sim
"Rosaceae"                          → 0.391 cosine sim
"a photo of a tree seedling"        → 0.647 cosine sim   ← highest, as expected
```

### Why SimpleShot Doesn't Use the Text Encoder

SimpleShot needs labeled support images, not text descriptions. The workflow is:

1. Embed all training images with the image encoder → 768-dim vectors
2. Group by label → compute centroid (average vector) per class
3. For a new query image → embed it → find nearest centroid

This works because the image encoder was trained to produce similar vectors for taxonomically similar organisms. Two Fabaceae seedlings produce more similar 768-dim vectors than a Fabaceae and a Poaceae seedling — the ViT learned this during pre-training.

**The text encoder would be needed for zero-shot classification** (classify using text descriptions with no labeled images). We don't use zero-shot because we have labeled training data and SimpleShot with support images outperforms zero-shot.

### How the Projection Guarantees Space Correctness

The 768-dim embedding is not the ViT's raw internal representation. It's the output of a **learned linear projection** (`visual.proj`, shape 1024×768) that was optimized during pre-training to map visual features into the shared contrastive space. This projection is:

- **Learned, not random**: Its 786K parameters were optimized by backpropagation through the contrastive loss
- **Included in the released weights**: It's part of `open_clip_model.safetensors`
- **Applied automatically**: `model.encode_image()` includes the projection
- **Not discarded**: Unlike some CLIP variants that have a separate "projection head" removed at inference, BioCLIP 2's projection is integral to the model

Loading `hf-hub:imageomics/bioclip-2` through `open_clip` loads both the ViT weights AND the projection weights. You get the correct shared space automatically.

---

## Preprocessing Pipeline

### pybioclip's TreeOfLife Transform (What Our Code Uses)

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), interpolation=bilinear, antialias=True),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])
```

Key properties:
- **Direct resize to 224×224**: Stretches the image to square, potentially distorting aspect ratio
- **Bilinear interpolation**
- **CLIP-standard normalization** values (same as OpenAI CLIP, LAION CLIP, etc.)

### open_clip's Default Transform

```python
transforms.Compose([
    transforms.Resize(224, interpolation=bicubic, antialias=True),    # shortest edge → 224
    transforms.CenterCrop((224, 224)),                                 # crop to square
    _convert_to_rgb,
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])
```

Key properties:
- **Shortest-edge resize + center crop**: Preserves aspect ratio, crops to square
- **Bicubic interpolation**
- **Explicit RGB conversion**
- **Same normalization values**

### Which One Is Correct?

| Property | pybioclip (ours) | open_clip default | HF config specifies |
|----------|-----------------|-------------------|---------------------|
| Resize strategy | Stretch to square | Shortest edge + center crop | `"resize_mode": "shortest"` |
| Interpolation | Bilinear | Bicubic | `"interpolation": "bicubic"` |
| Normalization | CLIP standard | CLIP standard | CLIP standard |

The HuggingFace-hosted `open_clip_config.json` specifies bicubic + shortest-edge, matching open_clip's default. However, **pybioclip deliberately overrides this for TreeOfLife models**. The pybioclip package is maintained by the Imageomics group (the same team that built BioCLIP 2), so this is an intentional choice — likely because:

1. Organism photos are often non-square. Center-cropping can cut off diagnostic features (leaf tips, roots, labels).
2. Stretching ensures the entire organism is visible to the model.
3. The authors tested this preprocessing during model development and benchmarking.

**Our code uses pybioclip's preprocessing in both code paths** (verified 2026-02-12: both paths produce identical 768-dim vectors with max diff = 0.0). This is consistent and intentional.

### Consistency Verification

We verified that both extraction code paths in `encoder.py` produce identical results:

```python
# Path 1: pybioclip's create_image_features (non-optimized)
features_1 = classifier.create_image_features([img], normalize=True)

# Path 2: manual preprocessing + encode_image (optimized DataLoader path)
preprocessed = classifier.preprocess(img).unsqueeze(0)
features_2 = model.encode_image(preprocessed)
features_2 = F.normalize(features_2, dim=-1)

# Result: max absolute difference = 0.0 (identical)
```

---

## How SimpleShot Leverages the Embedding Space

### The Algorithm

SimpleShot (`src/seedlearn/clip/simpleshot.py`) is nearest-centroid classification in the embedding space:

1. **Compute global mean** of all support features
2. **Mean-center** all features (subtract global mean) — removes bias toward "average organism"
3. **L2-normalize** all features — projects onto unit hypersphere, makes cosine similarity = dot product
4. **Compute class centroids** — average of each class's support features
5. **Classify queries** — assign to nearest centroid by cosine distance

### Why This Works

In BioCLIP 2's embedding space:
- Same-family images cluster together (Fabaceae seedlings → similar vectors)
- Different-family images are separated
- The clustering is by **taxonomy**, not visual similarity (pot color, background)

This means computing centroids and measuring distances is sufficient — no neural network training needed. SimpleShot needs as few as 1-5 images per class because the pre-trained space already has the taxonomic structure.

### Two-Level Data Division: Partitioning vs Support Sampling

There are two distinct operations — using precise terms avoids confusion:

**Level 1 — Partitioning** (`create_individual_split()`):
Divides the full dataset into train/val/test **partitions** using `GroupShuffleSplit` with `individual_id` (the `ID_YPS` column) as the group key. All photos of the same physical plant land in the same partition. This prevents data leakage — seedling photos share pot, soil, label stake, and growth chamber across multiple photos of the same plant.

**Level 2 — Support set sampling** (`create_fixed_support_set()`):
Within the training partition, exactly `k` images per class are sampled to form the **support set**. The classifier sees only `k × num_classes` images. Images from the test partition form the **query set** (what gets classified).

```
All ~12K images
  → BioCLIP 2 encodes every image → (N, 768) feature cache
  → GroupShuffleSplit by ID_YPS → 70% train / 15% val / 15% test  (PARTITIONING)
  → From train: sample k per class → support set                   (SUPPORT SAMPLING)
  → Test partition → query set (what gets classified)
```

| Term | What it means | Code reference |
|------|--------------|----------------|
| **Partition** | Train/val/test division of the full dataset | `create_individual_split()`, `DatasetSplit` |
| **Support set** | k labeled examples per class the classifier fits on | `create_fixed_support_set()` |
| **Query set** | Images being classified (from test partition) | Test indices from `DatasetSplit` |
| **k-shot** | Number of support examples per class | `k_shot` parameter |
| **n-way** | Number of classes in the classification task | `n_way` parameter |

---

## BioCLIP 1 vs BioCLIP 2

| Aspect | BioCLIP 1 | BioCLIP 2 |
|--------|-----------|-----------|
| **HuggingFace** | `imageomics/bioclip` | `imageomics/bioclip-2` |
| **Vision encoder** | ViT-B/16 (86M params) | ViT-L/14 (304M params) |
| **Total parameters** | ~150M | ~428M |
| **Embedding dim** | 512 | 768 |
| **Hidden dim** | 768 | 1024 |
| **Transformer layers** | 12 | 24 |
| **Patch size** | 16×16 | 14×14 (finer spatial resolution) |
| **Training data** | TreeOfLife-10M (10M images, 454K taxa) | TreeOfLife-200M (214M images, 952K taxa) |
| **Data sources** | EOL, iNaturalist, BIOSCAN-1M | GBIF, BIOSCAN-5M, EOL, FathomNet |
| **Initialization** | OpenAI CLIP ViT-B/16 | LAION-2B CLIP ViT-L/14 |
| **Experience replay** | None | 26M LAION-2B samples |
| **Training hardware** | 8× A100-80GB, 4 days | 32× H100-80GB, 10 days |
| **Zero-shot species acc** | 37.6% | **55.6% (+18.1%)** |
| **Rare species acc** | 34.9% | **55.3% (+20.4%)** |
| **Publication** | CVPR 2024 | NeurIPS 2025 Spotlight |

Key improvements:
1. **3.5× larger vision encoder** (86M → 304M params, 12 → 24 layers)
2. **21× more training data** (10M → 214M images)
3. **Experience replay** prevents catastrophic forgetting of general visual concepts
4. **Finer patches** (16×16 → 14×14) capture more spatial detail
5. **Emergent properties** at scale: intra-species variation separation, ecological trait correlation

**Impact on our pipeline**: All existing 512-dim `.npz` caches from BioCLIP 1 are incompatible with BioCLIP 2's 768-dim output and must be regenerated. The SimpleShot classifier code is dimension-agnostic (numpy broadcasting) and needs no changes.

---

## Correctness Checklist: What Could Go Wrong

### Critical Failure Modes

| Issue | Impact | How to Detect | Current Status |
|-------|--------|---------------|----------------|
| **Wrong model loaded** (BioCLIP 1 instead of 2) | 512-dim features, wrong space | Check `features.shape[1]` — must be 768 | Fixed: default changed to `bioclip-2` |
| **Wrong preprocessing** | Degraded accuracy (features in wrong region of space) | Compare transforms to pybioclip's `preprocess_img` | Correct: using pybioclip's TreeOfLife transform |
| **Features not normalized** | Distance computation broken | Check `np.linalg.norm(features, axis=1)` ≈ 1.0 | Correct: L2-normalized in both code paths |
| **Mixed model versions in cache** | Incoherent feature space | Cache labels checked against records | Correct: cache includes label verification |
| **Individual leakage in splits** | Inflated accuracy (memorize pot/background) | Verify all images of same ID_YPS in same partition | Script supports `--split-type individual` |
| **Old 512-dim cache loaded** | Dimension mismatch, crash or silent degradation | Verify `features.shape[1] == 768` | Old caches from Oct 2025 dataset |

### Validation Commands

After generating new embeddings, verify:

```python
import numpy as np

data = np.load("path/to/family_features.npz")
features = data["features"]

# Check 1: Correct dimensions
assert features.shape[1] == 768, f"Expected 768-dim, got {features.shape[1]}"

# Check 2: L2-normalized
norms = np.linalg.norm(features, axis=1)
assert np.allclose(norms, 1.0, atol=1e-5), f"Features not normalized: mean norm = {norms.mean():.4f}"

# Check 3: Not degenerate (all same vector)
assert features.std() > 0.01, "Features appear degenerate (near-zero variance)"

# Check 4: Reasonable cosine similarity range
sim_matrix = features @ features.T
off_diag = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]
print(f"Inter-image cosine similarity: mean={off_diag.mean():.3f}, std={off_diag.std():.3f}")
# Expect: mean around 0.3-0.6, std around 0.1-0.2
```

---

## GPU Requirements for Extraction

### VRAM Estimates

| Component | VRAM (float32) | VRAM (float16) |
|-----------|---------------|----------------|
| Model weights (304M visual params) | ~1.2 GB | ~0.6 GB |
| Text encoder (loaded but unused) | ~0.5 GB | ~0.25 GB |
| Activations (batch 256, 224×224) | ~4-8 GB | ~2-4 GB |
| PyTorch overhead | ~1-2 GB | ~1-2 GB |
| **Total estimate** | **~7-12 GB** | **~4-7 GB** |

BioCLIP 2 fits easily on any modern GPU. An H200 with 140GB VRAM is vastly more than needed — you could run batch sizes of 1024+ without issue.

### Recommended Extraction Command

```bash
srun --partition=gpu_h200 --gpus=1 --mem=64G --time=01:00:00 --cpus-per-task=8 --pty bash
source .venv/bin/activate

CATALOG="data/raw/2026-01-29/sorted_12K/metadata/species_catalog_v2026-01-29_12K_20260129_123334.csv"

python scripts/extract_embeddings.py \
    --catalog "$CATALOG" \
    --rank family \
    --device cuda \
    --model-str "hf-hub:imageomics/bioclip-2" \
    --batch-size 512 \
    --verbose
```

**Expected runtime**: ~3-5 minutes for ~12K images on H200 (dominated by data loading, not GPU compute).

**Expected output**: `.npz` file in auto-derived cache directory with:
- `features`: `(N, 768)` float32 — L2-normalized BioCLIP 2 embeddings
- `labels`: `(N,)` int — family label IDs
- `image_paths`: `(N,)` str — file paths for debugging

---

## Current Classification Output: What Gets Returned

### End-to-End Data Flow at Classification Time

```
OFFLINE (one-time, before any classification):
  All ~12K images → BioCLIP 2 → (12K, 768) float32 cache (.npz)
  Each cached entry stores: [768-dim features, integer label_id, image_path string]

  Partition file selects: train indices / val indices / test indices

  SimpleShot.fit(train_features, train_labels):
    1. Compute global mean of all train features
    2. Mean-center all train features
    3. L2-normalize
    4. Compute one centroid per class (AVERAGE of all support images for that class)
    → Result: 52 centroids (one per family), each a 768-dim vector

AT CLASSIFICATION TIME (per specimen):
  Specimen images → BioCLIP 2 → (N, 768) per-image features
  Mean-pool across N images → single (1, 768) vector
  Mean-center using saved global mean
  L2-normalize

  Compute distance to ALL 52 centroids
  Softmax over negative distances → probabilities
  Sort by probability → top-k
```

### What the Stage Returns

From `classification.py:170-183`:

```python
predictions = [
    {"rank_value": "Fabaceae",  "confidence": 0.42, "rank_position": 1},
    {"rank_value": "Poaceae",   "confidence": 0.18, "rank_position": 2},
    {"rank_value": "Rosaceae",  "confidence": 0.09, "rank_position": 3},
    # ... top_k entries
]
# Plus metadata: embedding_dim: 768, num_images_pooled: N
```

That's it — a class name, a softmax confidence, and a rank position.

### What's Available but Discarded

| Data | Available in `ImageRecord` | Used by SimpleShot | Present in output |
|------|---------------------------|-------------------|-------------------|
| Image path | Yes | No | No |
| Family name | Yes | Only if `--rank family` | As the label |
| Genus name | Yes | Only if `--rank genus` | No (if rank=family) |
| Species name | Yes | Only if `--rank species` | No (if rank=family) |
| Individual ID (ID_YPS) | Yes | No | No |
| Per-image embedding | Yes (in cache) | Collapsed into centroid | No |
| Raw distances to centroids | Computed internally | Used for softmax | No (only probabilities) |
| Per-image predictions | Could be computed | Not done (mean-pool first) | No |

---

## Limitations of Current Implementation

### 1. Centroid collapse — no individual image matches

SimpleShot computes `centroid_Fabaceae = mean(all Fabaceae support embeddings)`. This single 768-dim point represents the entire family. Within-class variation is lost. A family with diverse morphology (some species look like grasses, others like trees) gets one averaged point that may not resemble any actual specimen. You cannot answer "which training image was most similar to my query?" because the match is against an average, not a specific image.

### 2. Single-rank classification

We extract at one rank (`--rank family`). The cache labels are family integers. We cannot simultaneously predict "this is Fabaceae, genus Acacia, species A. dealbata" from one run. Each rank needs a separate classifier fitted on the same cached embeddings with different label assignments.

### 3. Mean-pooling across specimen images hides disagreement

If a specimen has 5 photos and 4 say Fabaceae while 1 says Poaceae, the mean-pool blends them and the disagreement is never surfaced. That 1 dissenting image could be a quality signal (bad angle, occluded, mislabeled photo).

### 4. No distance or similarity information

The softmax probabilities convey relative ranking but not absolute similarity. A confidence of 0.42 for Fabaceae could mean "clearly Fabaceae, nothing else close" or "everything looks equally bad and Fabaceae won by a tiny margin." Raw distances would distinguish these cases.

### 5. No confusion context

The output doesn't convey that "Fabaceae and Caesalpiniaceae centroids are very close in this space" — information that would help Stage 5 (reasoning) understand if a classification is uncertain because of genuine morphological similarity between families.

---

## Improvement Paths (Ordered by Impact vs Effort)

### Level 0: Enrich the existing output (low effort, high value)

Add richer data to the classification output **without changing SimpleShot**:

```python
{
    "rank_value": "Fabaceae",
    "confidence": 0.42,
    "rank_position": 1,
    "raw_distance": 0.31,              # Euclidean distance to centroid
    "cosine_similarity": 0.85,          # Direct cosine sim to centroid
    "margin": 0.14,                     # Gap between #1 and #2 confidence
    "per_image_predictions": [          # Before mean-pooling
        {"image": "img001.jpg", "top1": "Fabaceae", "confidence": 0.51},
        {"image": "img002.jpg", "top1": "Fabaceae", "confidence": 0.38},
        {"image": "img003.jpg", "top1": "Poaceae",  "confidence": 0.29},  # ← disagreement
    ],
}
```

This gives Stage 5 (reasoning LLM) much richer evidence to work with — margin tells it whether to trust the classification, per-image breakdown surfaces problematic photos.

### Level 1: k-NN alongside SimpleShot (moderate effort, high value)

Run a k-nearest-neighbor search on the **individual support embeddings** (not centroids). This answers "which actual training images look most like my query?":

```python
{
    "knn_matches": [
        {"image_path": "Fabaceae/Acacia/.../img.jpg", "family": "Fabaceae",
         "genus": "Acacia", "species": "Acacia dealbata", "distance": 0.12},
        {"image_path": "Fabaceae/Mimosa/.../img.jpg", "family": "Fabaceae",
         "genus": "Mimosa", "species": "Mimosa pudica", "distance": 0.15},
    ]
}
```

This is powerful because:
- You see the actual species composition of nearest neighbors (not just family)
- Multiple families in top-k = uncertainty signal
- The genus/species breakdown within a family prediction adds granularity
- Stage 5 can reason about "4 of 5 nearest neighbors are Fabaceae, all genus Acacia" vs "5 nearest are Fabaceae but spread across 4 different genera"

### Level 2: Multi-rank classification (moderate effort, high value)

Run family, genus, and species classifiers simultaneously from the same 768-dim embeddings (same cache, different label assignments):

```python
{
    "family_prediction":  {"rank_value": "Fabaceae",         "confidence": 0.42},
    "genus_prediction":   {"rank_value": "Acacia",           "confidence": 0.28},
    "species_prediction": {"rank_value": "Acacia dealbata",  "confidence": 0.15},
    "hierarchical_consistent": True,  # Acacia dealbata IS in Fabaceae
}
```

Hierarchical consistency becomes a quality signal — if the family classifier says Fabaceae but the genus classifier says Quercus (Fagaceae), that inconsistency flags a problematic classification.

### Level 3: LoRA fine-tuning (high effort, highest value)

Adapt BioCLIP 2's ViT weights specifically for our seedling taxa. This changes the embedding space itself — the 768-dim vectors would be better tuned for distinguishing our 52 families / 164 species. SimpleShot accuracy improves because centroids become more separable.

### Key Insight

The `.npz` cache of (N, 768) embeddings + labels + paths is the foundation for all improvement levels. **The extraction job doesn't change regardless of which improvements we add.** Levels 0-2 are purely about how we query and present results from that cache. Only Level 3 (LoRA) requires re-extraction.

---

## Future: LoRA Fine-Tuning Path

See `docs/clip/bioclip2_findings.md` Section 3 for the full fine-tuning feasibility analysis and approach hierarchy.

**Summary**: SimpleShot on frozen BioCLIP 2 embeddings (Level 0) is the baseline that must be established first. LoRA fine-tuning (Level 2) adapts the last few ViT layers specifically for our seedling data. With ~49 images per species, LoRA is the ceiling of what's responsible — full fine-tuning would overfit.

The embedding architecture described in this document is the foundation for both SimpleShot and future LoRA work. LoRA would modify the ViT weights (adding low-rank adapters to attention matrices), which would change the 768-dim output vectors. The SimpleShot infrastructure (cache, splits, classifier) would be reused with new embeddings.

---

## Sources

### Papers

- **BioCLIP 2**: Stevens et al., "BioCLIP 2." NeurIPS 2025 Spotlight. [ArXiv 2505.23883](https://arxiv.org/abs/2505.23883)
- **BioCLIP 1**: Stevens et al., "BioCLIP: A Vision Foundation Model for the Tree of Life." CVPR 2024. [ArXiv 2311.18803](https://arxiv.org/abs/2311.18803)
- **CLIP-LoRA**: Zanella & Ben Ayed, CVPR Workshop 2024. Reference for LoRA on CLIP models.

### Code and Models

- **BioCLIP 2 Model Card**: [huggingface.co/imageomics/bioclip-2](https://huggingface.co/imageomics/bioclip-2)
- **BioCLIP 2 GitHub**: [github.com/Imageomics/bioclip-2](https://github.com/Imageomics/bioclip-2)
- **BioCLIP 2 Project Page**: [imageomics.github.io/bioclip-2](https://imageomics.github.io/bioclip-2/)
- **pybioclip**: [github.com/Imageomics/pybioclip](https://github.com/Imageomics/pybioclip) — [PyPI](https://pypi.org/project/pybioclip/)
- **OpenCLIP**: [github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)

### Verification

All architecture details, parameter counts, preprocessing pipelines, and embedding space properties were verified by introspecting the installed packages and loaded model on 2026-02-12. The verification code is in the session transcript and can be re-run at any time.
