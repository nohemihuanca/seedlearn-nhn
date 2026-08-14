# vLLM Installation

**Navigation**: [SeedLearn](../README.md) > **vLLM Installation**

---

## Why a Custom Wheel?

vLLM requires a **custom-built wheel** on the Bouchet YCRC cluster because RHEL 8's glibc (2.28) is too old for pre-built PyPI wheels (which require glibc 2.31+). A pre-built wheel for Python 3.13 + CUDA 12.x + H200 (SM 9.0) is available on shared NFS.

---

## Quick Install

```bash
source .venv/bin/activate
uv pip install /nfs/roberts/project/pi_lsc4/shared/seedlearn/software/py_wheels/vllm-*.whl
```

### Verify

```bash
python -c "import vllm; print(f'vLLM {vllm.__version__}')"
```

---

## Wheel Details

| Property | Value |
|----------|-------|
| Location | `/nfs/roberts/project/pi_lsc4/shared/seedlearn/software/py_wheels/` |
| Version | 0.14.0rc2 (dev build, 2026-01-22) |
| Python | cp313 (Python 3.13) |
| CUDA | 12.x, SM 9.0 (Hopper — H200/H100) |
| Platform | linux_x86_64 |
| Size | ~804 MB |

---

## Runtime Requirements

vLLM stages (1 and 5) need a GPU. The server and client must run on the **same host** (the pipeline connects to `localhost:8000`). Use an [OOD Interactive Desktop](https://ood-bouchet.ycrc.yale.edu) with a GPU — see the [Quick Start](../README.md#1-request-an-interactive-desktop) for desktop settings.

```bash
# Terminal 1 (OOD Desktop): Start vLLM server
source .venv/bin/activate
bash scripts/start_vllm.sh
# Wait for "Uvicorn running on http://0.0.0.0:8000"

# Terminal 2 (OOD Desktop): Run pipeline
source .venv/bin/activate
python scripts/run_pipeline.py ...
```

See [Pipeline Reference](pipeline.md) for endpoint configuration and model selection.

---

## Rebuilding the Wheel

If you need a new wheel (different Python version, newer vLLM, different CUDA), the source build takes ~30-90 minutes on a 16-32 core SLURM node.

### Prerequisites

```bash
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=160G --time=1:30:00 --pty bash
module load GCC/13.3.0 CUDA/12.9.1 Ninja/1.12.1-GCCcore-13.3.0
```

### Build Steps

```bash
# 1. Clone and checkout desired version
cd /nfs/roberts/project/pi_mjh225/mjh225/software/vllm
git clone https://github.com/vllm-project/vllm.git vllm-src && cd vllm-src
git checkout v0.14.0  # or desired tag

# 2. Force H200 architecture (env vars are NOT reliable)
sed -i 's/set(CUDA_SUPPORTED_ARCHS.*/set(CUDA_SUPPORTED_ARCHS "9.0")/' CMakeLists.txt

# 3. Set up build environment
uv venv --python 3.13 && source .venv/bin/activate
uv pip install pip setuptools wheel build ninja packaging numpy
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# 4. Build
export TORCH_CUDA_ARCH_LIST="9.0" VLLM_TARGET_DEVICE=cuda MAX_JOBS=24
rm -rf build/ dist/ *.egg-info/
pip wheel . --no-deps --wheel-dir=dist/

# 5. Verify SM 9.0 during build (in another terminal)
ps aux | grep nvcc | grep -o "arch=compute_[0-9]*"
# Expected: arch=compute_90

# 6. Deploy wheel
cp dist/vllm-*.whl /nfs/roberts/project/pi_lsc4/shared/seedlearn/software/py_wheels/
```

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| Wrong SM version (sm_80) | Re-run `sed` command, clean build dir, rebuild |
| OOM during build | Reduce `MAX_JOBS` to 8 or 4, request more SLURM RAM |
| `ninja: not found` | `module load Ninja/1.12.1-GCCcore-13.3.0` |
| Wheel incompatible | Ensure build and target venv have matching Python versions |

---

## GPU Architecture Reference

| GPU | Compute Capability | CUDA Arch |
|-----|--------------------|-----------|
| H200 / H100 | SM 9.0 | Hopper |
| A100 | SM 8.0 | Ampere |
| V100 | SM 7.0 | Volta |

---

## See Also

- [Pipeline Reference](pipeline.md) — vLLM endpoint configuration, model selection
- [Vision-LLM Benchmarks](benchmarks.md) — Model sweep, server lifecycle helpers
- [start_vllm.sh](../scripts/start_vllm.sh) — vLLM server wrapper script
