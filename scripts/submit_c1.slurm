#!/usr/bin/env bash
#SBATCH --job-name=trait-c1
#SBATCH --partition=gpu_devel
#SBATCH --gpus=h200:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/trait_c1_%j.out
#
# Submit the C1 (upgraded-model) condition as a batch job:
#   sbatch scripts/submit_c1.slurm
#
# Serves Qwen/Qwen3.6-35B-A3B-FP8 from the dedicated .venv-vllm017 (vLLM >= 0.17,
# which registers the Qwen3_5Moe architecture), runs Stage-1 inference for all 114
# specimens via the main .venv client, grades against the human annotators, and
# rebuilds the comparison report across C1 + the existing C0/C2/C3/C4 + K1-K3.
#
# Overrides (export before sbatch):
#   MODEL_C1=<hf-id>     upgraded model id
#   PRIOR_TS=<ts>        base-model run timestamp to reuse (default 20260713_161841)
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-/nfs/roberts/project/pi_lsc4/lmb242/seedlearn}"
[ -f "$REPO_DIR/scripts/c1_upgraded_ladder.sh" ] || REPO_DIR="/nfs/roberts/project/pi_lsc4/lmb242/seedlearn"
cd "$REPO_DIR"

echo "Repo:  $(pwd)"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Start: $(date)"

# Main venv for the pipeline client + grading; serving uses .venv-vllm017 via VLLM_BIN.
source .venv/bin/activate
echo "client vLLM: $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo n/a)"
echo "serve vLLM:  $(.venv-vllm017/bin/python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo n/a)"

bash scripts/c1_upgraded_ladder.sh

echo "End:   $(date)"
