#!/usr/bin/env bash
# C1-only runner: serve the upgraded Qwen3.6 MoE VLM and grade it into the ladder.
#
# The base-model conditions (C0/C2/C3/C4) and Kaili's external runs (K1-K3) are
# already computed under trait_grading/model_run/. This script fills in the one
# condition that couldn't run before — C1, the upgraded model — because its
# architecture (Qwen3_5MoeForConditionalGeneration) needs vLLM >= 0.17, which the
# main .venv (0.14) does not have. It serves that model from a dedicated newer
# vLLM venv (VLLM_BIN) while the Stage-1 client + grading keep running from the
# active main .venv, then rebuilds the comparison report across all conditions.
#
# Usage (from repo root, main .venv active):
#   bash scripts/run_c1_upgraded.sh
#
# Overrides:
#   MODEL_C1=<hf-id>          upgraded model (default: Qwen/Qwen3.6-35B-A3B-FP8)
#   VLLM_BIN=<path/to/vllm>   vLLM binary that supports the arch (default below)
#   PRIOR_TS=<YYYYmmdd_HHMMSS>  timestamp of the completed base-model run to reuse
#   MAX_MODEL_LEN=<int>       served context length (default 32768)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"
# Must be set BEFORE sourcing common.sh, which pins it to 600 at source time.
# The cold model has to load 35GB + torch.compile + Triton JIT + CUDA graphs.
export VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-1800}"
# shellcheck source=/dev/null
source tests/benchmarks/common.sh

# ─── Config ──────────────────────────────────────────────────────────────────
PORT="${PORT:-8000}"
MODEL_C1="${MODEL_C1:-Qwen/Qwen3.6-35B-A3B-FP8}"
# Newer vLLM install that registers Qwen3_5MoeForConditionalGeneration.
export VLLM_BIN="${VLLM_BIN:-$REPO_DIR/.venv-vllm017/bin/vllm}"
# Serve flags (appended verbatim to `vllm serve`; common.sh word-splits them, so
# the JSON value below must contain NO spaces):
#  --reasoning-parser qwen3   recommended by the vLLM recipe for this model.
#  --gdn-prefill-backend triton  This Qwen3.5/3.6 MoE uses gated-delta-network
#    linear attention whose FlashInfer prefill kernel is JIT-compiled with nvcc.
#    The serving venv has torch's CUDA runtime but not the full toolkit (no nvcc),
#    so FlashInfer JIT fails. Triton compiles with its own bundled toolchain — no
#    nvcc needed. (attention already resolves to FLASH_ATTN, MoE to Triton.)
#  --default-chat-template-kwargs {"enable_thinking":false}  Qwen3.6 is a hybrid
#    reasoning model. With thinking ON it spent the whole token budget "thinking"
#    and emitted an EMPTY final form for 63/114 specimens (first C1 run), biasing
#    the comparison. Disabling thinking (a) fixes that truncation and (b) makes it
#    a fair apples-to-apples test vs the non-reasoning Instruct baseline (C0).
_C1_DEFAULT_FLAGS='--reasoning-parser qwen3 --gdn-prefill-backend triton --default-chat-template-kwargs {"enable_thinking":false}'
export VLLM_EXTRA_FLAGS="${VLLM_EXTRA_FLAGS:-$_C1_DEFAULT_FLAGS}"
# The FlashInfer top-k/top-p sampler is likewise JIT-compiled (needs nvcc). Fall
# back to vLLM's native sampler to avoid a second nvcc trap during warmup.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

CATALOG="data/raw/2026-01-29/sorted_12K/metadata/species_catalog_v2026-01-29_12K_20260129_123334.csv"
SPECIMEN_SRC="trait_grading/keys/curator_taxonomic_key.csv"
SKIP_STAGES="classification trait_retrieval evidence_synthesis reasoning"  # Stage 1 only

TS="$(date +%Y%m%d_%H%M%S)"
RUNROOT="trait_grading/model_run"
REPORT_DIR="trait_grading/reports/experiments/c1_${TS}"
# Timestamp of the completed base-model ladder run whose C0/C2/C3/C4 dirs we reuse.
PRIOR_TS="${PRIOR_TS:-20260713_161841}"

cleanup() { stop_vllm "$PORT" 2>/dev/null || true; }
trap cleanup EXIT

# vLLM 0.25 pulls in torchcodec, which dlopens FFmpeg's shared libs at import
# time — even though we only feed images, not video. Without them the server
# crashes on startup (libavutil.so.* not found). Put FFmpeg (+ its LAME/x264/x265
# deps) on LD_LIBRARY_PATH via the module system, with hardcoded EB paths as a
# fallback if `module` is unavailable in this shell.
FFMPEG_MODULE="${FFMPEG_MODULE:-FFmpeg/7.0.2-GCCcore-13.3.0}"
load_ffmpeg_libs() {
    for init in /etc/profile.d/z00_lmod.sh /etc/profile.d/modules.sh \
                /etc/profile.d/lmod.sh "${MODULESHOME:-}/init/bash"; do
        [ -f "$init" ] && . "$init" 2>/dev/null && break
    done
    if command -v module >/dev/null 2>&1 && module load "$FFMPEG_MODULE" 2>/dev/null; then
        log "Loaded FFmpeg libs via module: $FFMPEG_MODULE"
        return 0
    fi
    local base=/apps/software/2024a/software
    export LD_LIBRARY_PATH="$base/FFmpeg/7.0.2-GCCcore-13.3.0/lib:$base/LAME/3.100-GCCcore-13.3.0/lib:$base/x264/20240513-GCCcore-13.3.0/lib:$base/x265/3.6-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
    log "Loaded FFmpeg libs via hardcoded EB paths (module unavailable)"
}
load_ffmpeg_libs

# ─── Serve upgraded model + run C1 (Stage-1 only) ────────────────────────────
log_section "C1 upgraded model: $MODEL_C1"
log "Serving via: $VLLM_BIN"
log "Extra flags:  $VLLM_EXTRA_FLAGS"
if [ ! -x "$VLLM_BIN" ]; then
    log "ERROR: VLLM_BIN not executable: $VLLM_BIN"
    exit 1
fi

# Preflight: confirm the serving venv can actually import torchcodec's libs now,
# so we fail fast with a clear message instead of a 600s health-check timeout.
VLLM_VENV_PY="$(dirname "$VLLM_BIN")/python"
if ! "$VLLM_VENV_PY" -c "import torch, torchcodec; from torchcodec._internally_replaced_utils import load_torchcodec_shared_libraries as L; L()" 2>/dev/null; then
    log "ERROR: torchcodec shared libs still not loadable — FFmpeg libs missing. Set FFMPEG_MODULE or fix LD_LIBRARY_PATH."
    exit 1
fi
log "Preflight OK: torchcodec shared libs load."

C1_OUT="$RUNROOT/C1_upgraded_model_${TS}"
if start_vllm "$MODEL_C1" "$PORT" "logs/vllm_c1_${TS}.log" "$MAX_MODEL_LEN"; then
    if python scripts/run_benchmark_pipeline.py \
        --catalog "$CATALOG" --specimen-source "$SPECIMEN_SRC" \
        --skip $SKIP_STAGES \
        --vlm-model "$MODEL_C1" --vlm-endpoint "http://localhost:${PORT}/v1" \
        --prompt-style sys4 --output-dir "$C1_OUT"; then
        log "  ✓ C1_upgraded_model -> $C1_OUT"
    else
        log "  ✗ C1 inference FAILED"
    fi
    stop_vllm "$PORT"
else
    log "ERROR: $MODEL_C1 failed to serve on vLLM ($VLLM_BIN). See logs/vllm_c1_${TS}.log"
    exit 1
fi

# ─── Grade + compare: new C1 + reused base-model + external ──────────────────
log_section "Grading + comparison"
RUNS=()
[ -d "$C1_OUT" ] && RUNS+=(--run "C1_upgraded_model=${C1_OUT}")
for label in C0_baseline C2_single_trait C3_enriched_desc C4_image_fewshot; do
    d="$RUNROOT/${label}_${PRIOR_TS}"
    [ -d "$d" ] && RUNS+=(--run "${label}=${d}") || log "note: prior dir missing: $d"
done
for k in K1_gpt-5.4_all-traits K2_gpt-5.1_per-trait K3_gpt-5.1_per-section; do
    [ -d "$RUNROOT/$k" ] && RUNS+=(--run "${k}=${RUNROOT}/${k}")
done

if [ "${#RUNS[@]}" -eq 0 ]; then
    log "ERROR: no conditions to compare"
    exit 1
fi

[ -d "$C1_OUT" ] && python scripts/leaf_margin_headroom.py --results-dir "$C1_OUT" || true

python scripts/compare_trait_experiments.py "${RUNS[@]}" \
    --baseline C0_baseline --out-dir "$REPORT_DIR"
log "Done. Report: $REPORT_DIR/leaf_margin_comparison.html"
