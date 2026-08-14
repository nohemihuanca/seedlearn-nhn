#!/usr/bin/env bash
#SBATCH --job-name=seedlearn-step2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/step2_%j.out
#SBATCH --error=logs/step2_%j.err

# Step 2: BioCLIP 2 embedding extraction (GPU) + individual splits (CPU)
# Submitted to 'gpu' partition (RTX 5000 Ada, 32GB VRAM — plenty for BioCLIP 2's ~10GB requirement)

set -euo pipefail

REPO_DIR="/nfs/roberts/project/pi_mjh225/mjh225/repos/mitchellxh/seedlearn-dev"
CATALOG="$REPO_DIR/data/raw/2026-01-29/sorted_12K/metadata/species_catalog_v2026-01-29_12K_20260129_123334.csv"

cd "$REPO_DIR"
mkdir -p logs
source .venv/bin/activate

echo "========================================"
echo "SeedLearn Step 2: Data Preparation"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================"
echo ""

# --- Step 2a: Extract BioCLIP 2 embeddings (family rank) ---
echo "=== Step 2a: BioCLIP 2 Embedding Extraction ==="
echo "Model: hf-hub:imageomics/bioclip-2 (ViT-L/14, 768-dim)"
echo "Catalog: $CATALOG"
echo ""

python scripts/extract_embeddings.py \
    --catalog "$CATALOG" \
    --rank family \
    --device cuda \
    --model-str "hf-hub:imageomics/bioclip-2" \
    --batch-size 256 \
    --num-workers 8 \
    --verbose

echo ""
echo "--- Verifying extraction output ---"
python -c "
import numpy as np
from seedlearn.data.constants import SHARED_EMBEDDINGS, get_catalog_version
from pathlib import Path

cache_dir = SHARED_EMBEDDINGS / get_catalog_version(Path('$CATALOG'))
cache_path = cache_dir / 'family_features.npz'
print(f'Cache path: {cache_path}')
assert cache_path.exists(), f'Cache not found at {cache_path}'

data = np.load(cache_path)
features = data['features']
labels = data['labels']
norms = np.linalg.norm(features, axis=1)

print(f'Features shape: {features.shape}')
print(f'Labels shape: {labels.shape}')
print(f'Unique labels: {len(np.unique(labels))}')
print(f'L2 norms: mean={norms.mean():.4f}, std={norms.std():.6f}')

assert features.shape[1] == 768, f'FAIL: Expected 768-dim, got {features.shape[1]}'
assert np.allclose(norms, 1.0, atol=1e-5), f'FAIL: Not L2-normalized (mean norm={norms.mean():.4f})'
assert features.std() > 0.01, f'FAIL: Degenerate features (std={features.std():.6f})'

print('All checks PASSED')
print(f'Cache directory: {cache_dir}')
"

echo ""

# --- Step 2b: Create individual-level splits (family rank) ---
echo "=== Step 2b: Individual-Level Splits ==="
echo "Split type: individual (grouped by ID_YPS)"
echo ""

python scripts/create_splits.py \
    --catalog "$CATALOG" \
    --rank family \
    --split-type individual \
    --num-seeds 5 \
    --start-seed 42 \
    --verbose

echo ""
echo "--- Verifying split output ---"
python -c "
import json
from seedlearn.data.constants import SHARED_SPLITS, get_catalog_version
from pathlib import Path

splits_dir = SHARED_SPLITS / get_catalog_version(Path('$CATALOG')) / 'family'
print(f'Splits directory: {splits_dir}')

for seed in [42, 43, 44, 45, 46]:
    info_path = splits_dir / f'split_seed{seed}.json'
    npz_path = splits_dir / f'split_seed{seed}.npz'
    assert info_path.exists(), f'Missing: {info_path}'
    assert npz_path.exists(), f'Missing: {npz_path}'

    with open(info_path) as f:
        info = json.load(f)

    print(f'seed={seed}: split_type={info.get(\"split_type\", \"unknown\")}, '
          f'train={info[\"train_size\"]}, val={info[\"val_size\"]}, test={info[\"test_size\"]}')

print('All checks PASSED')
print(f'Splits directory: {splits_dir}')
"

echo ""
echo "========================================"
echo "Step 2 complete at $(date)"
echo "========================================"
