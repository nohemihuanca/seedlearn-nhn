#!/usr/bin/env bash
#SBATCH --job-name=trait-upmargin
#SBATCH --partition=gpu_devel
#SBATCH --gpus=h200:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --output=logs/trait_upmargin_%j.out
#
# Submit the upgraded-model leaf-margin ladder v2 as a batch job:
#   sbatch scripts/submit_upgraded_margin.slurm
#
# Serves Qwen/Qwen3.6-35B-A3B-FP8 once from .venv-vllm017 and runs the three
# leaf-margin prompt variants (margin_only, margin_rich, margin_rich + few-shot),
# then rebuilds the leaf-margin comparison against the existing C0/C1 (+ K1-K3).
# Wall-time is 5h — three sequential Stage-1 passes over 114 specimens plus a
# cold 35GB model load can exceed the 3h C1 budget.
#
# Overrides (export before sbatch): MODEL_UP, C0_DIR, C1_DIR.
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-/nfs/roberts/project/pi_lsc4/lmb242/seedlearn}"
[ -f "$REPO_DIR/scripts/upgraded_margin_ladder.sh" ] || REPO_DIR="/nfs/roberts/project/pi_lsc4/lmb242/seedlearn"
cd "$REPO_DIR"

echo "Repo:  $(pwd)"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Start: $(date)"

source .venv/bin/activate
echo "client vLLM: $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo n/a)"
echo "serve vLLM:  $(.venv-vllm017/bin/python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo n/a)"

bash scripts/upgraded_margin_ladder.sh

echo "End:   $(date)"
