#!/bin/bash
# =============================================================================
# Baseline job: condition D (Stage 2 top-1 only, no vLLM)
# GPU needed for BioCLIP feature extraction, but no vLLM server
# =============================================================================
#SBATCH --job-name=ablation-baseline
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/roberts/project/pi_mjh225/mjh225/repos/mitchellxh/seedlearn-dev/experiments/ablation/outputs/logs/baseline_%j.out
#SBATCH --error=/nfs/roberts/project/pi_mjh225/mjh225/repos/mitchellxh/seedlearn-dev/experiments/ablation/outputs/logs/baseline_%j.err

set -euo pipefail

PROJECT_ROOT="/nfs/roberts/project/pi_mjh225/mjh225/repos/mitchellxh/seedlearn-dev"
cd "$PROJECT_ROOT"
source .venv/bin/activate

echo "=== Baseline (condition D): $(hostname), GPU: $CUDA_VISIBLE_DEVICES ==="

python experiments/ablation/runners/baseline_runner.py \
    --output-dir experiments/ablation/outputs/condition_D \
    --device cuda \
    --max-specimens 50

echo "=== Baseline done ($(date)) ==="
