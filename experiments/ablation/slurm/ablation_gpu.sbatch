#!/bin/bash
# =============================================================================
# Ablation GPU array job: conditions A, B, C
# Each array task starts vLLM and processes a shard of test specimens
#
# Array index encodes: CONDITION_IDX = TASK_ID / NUM_SHARDS
#                      SHARD_IDX     = TASK_ID % NUM_SHARDS
#
# Usage: sbatch ablation_gpu.sbatch
#   or:  sbatch --array=0-2 ablation_gpu.sbatch   (3 tasks, 1 shard each)
# =============================================================================
#SBATCH --job-name=ablation
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=/nfs/roberts/project/pi_mjh225/mjh225/repos/mitchellxh/seedlearn-dev/experiments/ablation/outputs/logs/ablation_%A_%a.out
#SBATCH --error=/nfs/roberts/project/pi_mjh225/mjh225/repos/mitchellxh/seedlearn-dev/experiments/ablation/outputs/logs/ablation_%A_%a.err

set -euo pipefail

# --- Configuration ---
CONDITIONS=(A B C)
NUM_SHARDS=4
PROJECT_ROOT="/nfs/roberts/project/pi_mjh225/mjh225/repos/mitchellxh/seedlearn-dev"

CONDITION_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SHARDS))
SHARD_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SHARDS))
CONDITION="${CONDITIONS[$CONDITION_IDX]}"

echo "=== Ablation job: condition=$CONDITION shard=$SHARD_IDX/$NUM_SHARDS ==="
echo "Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES, Time: $(date)"

# --- Environment ---
cd "$PROJECT_ROOT"
source .venv/bin/activate
source tests/benchmarks/common.sh

# --- Start vLLM ---
PORT=$(find_free_port)
LOG_FILE="experiments/ablation/outputs/logs/vllm_${CONDITION}_shard${SHARD_IDX}.log"
start_vllm "Qwen/Qwen3-VL-32B-Instruct-FP8" "$PORT" "$LOG_FILE" 32768

# Ensure cleanup on exit
trap "stop_vllm $PORT; sleep 2" EXIT

# --- Run batch ---
python experiments/ablation/runners/batch_runner.py \
    --condition "$CONDITION" \
    --shard-index "$SHARD_IDX" \
    --num-shards "$NUM_SHARDS" \
    --vlm-endpoint "http://localhost:${PORT}/v1" \
    --output-dir "experiments/ablation/outputs/condition_${CONDITION}" \
    --max-specimens 50

echo "=== Done: condition=$CONDITION shard=$SHARD_IDX ($(date)) ==="
