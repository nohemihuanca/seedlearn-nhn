#!/usr/bin/env bash
#SBATCH --job-name=gpu-probe
#SBATCH --partition=gpu_devel
#SBATCH --gpus=h200:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=logs/gpu_probe_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/nfs/roberts/project/pi_lsc4/lmb242/seedlearn}"
echo "Node: $(hostname)"
echo "=== nvidia-smi driver/CUDA ==="
nvidia-smi --query-gpu=name,driver_version --format=csv 2>&1 | head -4
nvidia-smi 2>&1 | grep -iE "CUDA Version" | head -1
echo "=== new venv torch.cuda test (cu130) ==="
.venv-vllm017/bin/python - <<'PY' 2>&1 | grep -vE "NVML|Triton" | tail -8
import torch
print("torch:", torch.__version__)
try:
    ok = torch.cuda.is_available()
    print("cuda.is_available:", ok)
    if ok:
        x = torch.randn(64, 64, device="cuda"); y = (x@x).sum().item()
        print("matmul on GPU OK, sum=", round(y,2))
        print("device:", torch.cuda.get_device_name(0))
except Exception as e:
    print("TORCH CUDA ERROR:", type(e).__name__, e)
PY
echo "Done: $(date)"
