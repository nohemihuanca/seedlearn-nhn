#!/usr/bin/env bash
# Upgraded-model leaf-margin ladder v2: serve Qwen3.6 once and re-run the three
# leaf-margin prompt variants (C2 margin_only, C3 margin_rich, C4 margin_rich +
# few-shot images) on it, then rebuild the leaf-margin comparison against the
# existing C0 baseline + C1 upgraded runs (+ K1-K3).
#
# The prompt-variant conditions were only ever run on the OLD base model; this
# isolates whether any prompt tweak helps once the model is already the strong one.
# Serving mechanics (vLLM >= 0.17 venv, FFmpeg libs, triton prefill, thinking OFF)
# mirror scripts/c1_upgraded_ladder.sh — see there for the full rationale.
#
# Usage (from repo root, main .venv active; normally via scripts/submit_upgraded_margin.slurm):
#   bash scripts/upgraded_margin_ladder.sh [--dry-run]
#
# Overrides:
#   MODEL_UP=<hf-id>        upgraded model (default: Qwen/Qwen3.6-35B-A3B-FP8)
#   VLLM_BIN=<path>         vLLM binary that registers the arch (default .venv-vllm017)
#   C0_DIR=<dir>           existing C0 baseline run dir to reuse in the report
#   C1_DIR=<dir>           existing C1 upgraded run dir to reuse in the report
#   MAX_MODEL_LEN=<int>    served context length (default 32768)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"
export VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-1800}"
# shellcheck source=/dev/null
source tests/benchmarks/common.sh

# ─── Config ──────────────────────────────────────────────────────────────────
PORT="${PORT:-8000}"
MODEL_UP="${MODEL_UP:-Qwen/Qwen3.6-35B-A3B-FP8}"
export VLLM_BIN="${VLLM_BIN:-$REPO_DIR/.venv-vllm017/bin/vllm}"
# Serve flags identical to the C1 run (no spaces — common.sh word-splits them).
_UP_DEFAULT_FLAGS='--reasoning-parser qwen3 --gdn-prefill-backend triton --default-chat-template-kwargs {"enable_thinking":false}'
export VLLM_EXTRA_FLAGS="${VLLM_EXTRA_FLAGS:-$_UP_DEFAULT_FLAGS}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

CATALOG="data/raw/2026-01-29/sorted_12K/metadata/species_catalog_v2026-01-29_12K_20260129_123334.csv"
SPECIMEN_SRC="trait_grading/keys/curator_taxonomic_key.csv"
SKIP_STAGES="classification trait_retrieval evidence_synthesis reasoning"  # Stage 1 only
EXAMPLES="configs/experiments/leaf_margin_examples.json"
EXEMPLAR_DIR="trait_grading/exemplars/leaf_margin"

TS="$(date +%Y%m%d_%H%M%S)"
RUNROOT="trait_grading/model_run"
REPORT_DIR="trait_grading/reports/experiments/upgraded_margin_${TS}"
# Existing runs to reuse in the comparison (no re-inference for C0/C1).
C0_DIR="${C0_DIR:-$RUNROOT/C0_baseline_20260713_161841}"
C1_DIR="${C1_DIR:-$RUNROOT/C1_upgraded_model_20260713_215241}"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

cleanup() { [ "$DRY" = 1 ] || stop_vllm "$PORT" 2>/dev/null || true; }
trap cleanup EXIT

# FFmpeg libs for torchcodec (vLLM 0.25 dlopens them at import) — see c1 script.
FFMPEG_MODULE="${FFMPEG_MODULE:-FFmpeg/7.0.2-GCCcore-13.3.0}"
load_ffmpeg_libs() {
    for init in /etc/profile.d/z00_lmod.sh /etc/profile.d/modules.sh \
                /etc/profile.d/lmod.sh "${MODULESHOME:-}/init/bash"; do
        [ -f "$init" ] && . "$init" 2>/dev/null && break
    done
    if command -v module >/dev/null 2>&1 && module load "$FFMPEG_MODULE" 2>/dev/null; then
        log "Loaded FFmpeg libs via module: $FFMPEG_MODULE"; return 0
    fi
    local base=/apps/software/2024a/software
    export LD_LIBRARY_PATH="$base/FFmpeg/7.0.2-GCCcore-13.3.0/lib:$base/LAME/3.100-GCCcore-13.3.0/lib:$base/x264/20240513-GCCcore-13.3.0/lib:$base/x265/3.6-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}"
    log "Loaded FFmpeg libs via hardcoded EB paths (module unavailable)"
}
[ "$DRY" = 1 ] || load_ffmpeg_libs

# ─── Run one Stage-1 condition against the served endpoint ───────────────────
run_condition() {
    local label="$1" prompt="$2" examples="${3:-}"
    local out="$RUNROOT/${label}_${TS}"
    local args=(python scripts/run_benchmark_pipeline.py
        --catalog "$CATALOG" --specimen-source "$SPECIMEN_SRC"
        --skip $SKIP_STAGES
        --vlm-model "$MODEL_UP" --vlm-endpoint "http://localhost:${PORT}/v1"
        --prompt-style "$prompt" --output-dir "$out")
    [ -n "$examples" ] && args+=(--examples "$examples")
    if [ "$DRY" = 1 ]; then printf '  %s\n' "${args[*]}"; return 0; fi
    log "Running condition: $label ($MODEL_UP, $prompt${examples:+, few-shot})"
    if "${args[@]}"; then log "  ✓ $label -> $out"; else log "  ✗ $label FAILED (continuing)"; fi
}

# ─── Serve upgraded model once, run the three margin variants ────────────────
log_section "Upgraded model (margin variants): $MODEL_UP"
if [ "$DRY" != 1 ] && [ ! -x "$VLLM_BIN" ]; then
    log "ERROR: VLLM_BIN not executable: $VLLM_BIN"; exit 1
fi
if [ "$DRY" != 1 ]; then
    VLLM_VENV_PY="$(dirname "$VLLM_BIN")/python"
    if ! "$VLLM_VENV_PY" -c "import torch, torchcodec; from torchcodec._internally_replaced_utils import load_torchcodec_shared_libraries as L; L()" 2>/dev/null; then
        log "ERROR: torchcodec shared libs not loadable — FFmpeg libs missing."; exit 1
    fi
    log "Preflight OK: torchcodec shared libs load."
fi

# Output dirs are deterministic from label + TS (checked with -d before use).
C2_OUT="$RUNROOT/C2u_margin_only_${TS}"
C3_OUT="$RUNROOT/C3u_margin_rich_${TS}"
C4_OUT="$RUNROOT/C4u_image_fewshot_${TS}"
if [ "$DRY" = 1 ] || start_vllm "$MODEL_UP" "$PORT" "logs/vllm_upmargin_${TS}.log" "$MAX_MODEL_LEN"; then
    run_condition C2u_margin_only  margin_only ""
    run_condition C3u_margin_rich  margin_rich ""
    if ls "$EXEMPLAR_DIR"/*.png >/dev/null 2>&1; then
        run_condition C4u_image_fewshot margin_rich "$EXAMPLES"
    else
        log "C4 skipped: no exemplar images in $EXEMPLAR_DIR/"; C4_OUT=""
    fi
    [ "$DRY" = 1 ] || stop_vllm "$PORT"
else
    log "ERROR: $MODEL_UP failed to serve on vLLM ($VLLM_BIN). See logs/vllm_upmargin_${TS}.log"; exit 1
fi

# ─── Grade + compare: new margin variants + reused C0/C1 + external K ────────
log_section "Grading + comparison"
if [ "$DRY" = 1 ]; then
    log "Would compare C0/C1 (reused) + C2u/C3u/C4u + K1-K3 -> $REPORT_DIR"; exit 0
fi

RUNS=()
[ -d "$C0_DIR" ] && RUNS+=(--run "C0_baseline=$C0_DIR") || log "note: C0 dir missing: $C0_DIR"
[ -d "$C1_DIR" ] && RUNS+=(--run "C1_upgraded_model=$C1_DIR") || log "note: C1 dir missing: $C1_DIR"
[ -d "$C2_OUT" ] && RUNS+=(--run "C2u_margin_only=$C2_OUT")
[ -d "$C3_OUT" ] && RUNS+=(--run "C3u_margin_rich=$C3_OUT")
[ -d "$C4_OUT" ] && RUNS+=(--run "C4u_image_fewshot=$C4_OUT")
for k in K1_gpt-5.4_all-traits K2_gpt-5.1_per-trait K3_gpt-5.1_per-section; do
    [ -d "$RUNROOT/$k" ] && RUNS+=(--run "${k}=${RUNROOT}/${k}")
done
if [ "${#RUNS[@]}" -eq 0 ]; then log "ERROR: no conditions to compare"; exit 1; fi

python scripts/compare_trait_experiments.py "${RUNS[@]}" \
    --baseline C0_baseline --out-dir "$REPORT_DIR"
log "Done. Report: $REPORT_DIR/leaf_margin_comparison.html"
