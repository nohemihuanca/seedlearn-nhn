#!/bin/bash
# =============================================================================
# Master launch script: submits all ablation experiment jobs with dependencies
#
# Usage:
#   cd seedlearn-dev
#   bash experiments/ablation/slurm/launch_all.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

# Create log directory
mkdir -p experiments/ablation/outputs/logs

echo "=== Ablation Experiment Launcher ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# --- Submit GPU array job (conditions A, B, C) ---
# 3 conditions × 4 shards = 12 array tasks, each on 1 H200
ABLATION_JOB=$(sbatch --parsable experiments/ablation/slurm/ablation_gpu.sbatch)
echo "Submitted ablation GPU array job: $ABLATION_JOB (12 tasks × 1 H200)"

# --- Submit baseline job (condition D) ---
BASELINE_JOB=$(sbatch --parsable experiments/ablation/slurm/baseline_gpu.sbatch)
echo "Submitted baseline GPU job: $BASELINE_JOB (1 H200, no vLLM)"

echo ""
echo "=== All jobs submitted ==="
echo "Ablation (A/B/C): $ABLATION_JOB"
echo "Baseline (D):     $BASELINE_JOB"
echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f experiments/ablation/outputs/logs/ablation_${ABLATION_JOB}_*.out"
echo ""
echo "After all complete, run analysis:"
echo "  python experiments/ablation/analysis/compute_metrics.py"
