#!/usr/bin/env bash
#SBATCH --job-name=trait-experiments
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/trait_experiments_%j.out
#
# Submit the full leaf-margin experiment ladder as a batch job:
#   sbatch scripts/submit_trait_experiments.slurm
#
# Resources mirror the interactive session:
#   srun --partition=gpu_devel --gpus=h200 --mem=80G --time=04:00:00 --cpus-per-task=8 --pty bash
# If your cluster's GRES type is not literally "h200:1", adjust the --gpus line
# above to match the working srun form.
#
# It serves each model via vLLM on the allocated node, runs Stage-1 inference for
# every local condition (C0/C2/C3/C4 on the base model; C1 best-effort on the
# upgraded model), grades against the human annotators, and writes the comparison
# report — no manual steps.
#
# Useful overrides (export before sbatch, or pass with --export):
#   MODEL_UPGRADED=<hf-id>   exact servable id for C1 (default may not load)
#   SKIP_C1=1                skip the upgraded-model condition entirely
#   VLLM_EXTRA_FLAGS_C1="--enforce-eager"   extra vLLM flags just for C1
#
set -euo pipefail

# sbatch sets SLURM_SUBMIT_DIR to where the job was submitted from; submit from
# the repo root. Fall back to this checkout's known path.
REPO_DIR="${SLURM_SUBMIT_DIR:-/nfs/roberts/project/pi_lsc4/lmb242/seedlearn}"
[ -f "$REPO_DIR/scripts/trait_experiment_ladder.sh" ] || REPO_DIR="/nfs/roberts/project/pi_lsc4/lmb242/seedlearn"
cd "$REPO_DIR"

echo "Repo:  $(pwd)"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Start: $(date)"

source .venv/bin/activate
bash scripts/trait_experiment_ladder.sh

echo "End:   $(date)"
