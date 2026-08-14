#!/usr/bin/env bash
# Leaf-margin experiment ladder orchestrator (plan unit U3).
#
# Serves each model once via the vLLM lifecycle helpers in
# tests/benchmarks/common.sh (which allow --allowed-local-media-path / so both
# specimen images under data/ and exemplar drawings under trait_grading/ are
# readable), runs each local condition as a Stage-1-only benchmark into a labeled
# model_run dir, then grades everything (local C0-C4 + external K1-K3) into one
# comparison report.
#
# Conditions mirror configs/experiments/leaf_margin_ladder.yaml. Each condition is
# best-effort: a failure is logged and the ladder continues, so one bad run (or an
# unservable upgraded model) never sinks the whole job.
#
# Usage:
#   bash scripts/trait_experiment_ladder.sh                 # full ladder + report
#   bash scripts/trait_experiment_ladder.sh --dry-run       # print planned commands
#   MODEL_UPGRADED=<hf-id> bash scripts/trait_experiment_ladder.sh   # enable C1
#   SKIP_C1=1 bash scripts/trait_experiment_ladder.sh       # skip the upgraded model
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"
# shellcheck source=/dev/null
source tests/benchmarks/common.sh

# ─── Config ──────────────────────────────────────────────────────────────────
PORT="${PORT:-8000}"
MODEL_BASE="${MODEL_BASE:-Qwen/Qwen3-VL-32B-Instruct-FP8}"
# C1 upgraded model. Uncertain to serve via vLLM (MoE/GGUF); best-effort. Override
# with the exact servable HF id, or set SKIP_C1=1 to skip.
MODEL_UPGRADED="${MODEL_UPGRADED:-Qwen/Qwen3.6-35B-A3B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

CATALOG="data/raw/2026-01-29/sorted_12K/metadata/species_catalog_v2026-01-29_12K_20260129_123334.csv"
SPECIMEN_SRC="trait_grading/keys/curator_taxonomic_key.csv"
SKIP_STAGES="classification trait_retrieval evidence_synthesis reasoning"  # Stage 1 only
EXAMPLES="configs/experiments/leaf_margin_examples.json"
EXEMPLAR_DIR="trait_grading/exemplars/leaf_margin"

TS="$(date +%Y%m%d_%H%M%S)"
RUNROOT="trait_grading/model_run"
REPORT_DIR="trait_grading/reports/experiments/ladder_${TS}"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

cleanup() { [ "$DRY" = 1 ] || stop_vllm "$PORT" 2>/dev/null || true; }
trap cleanup EXIT

# ─── Run one condition (Stage-1 only) ────────────────────────────────────────
run_condition() {
    local label="$1" model="$2" prompt="$3" examples="${4:-}"
    local out="$RUNROOT/${label}_${TS}"
    local args=(python scripts/run_benchmark_pipeline.py
        --catalog "$CATALOG" --specimen-source "$SPECIMEN_SRC"
        --skip $SKIP_STAGES
        --vlm-model "$model" --vlm-endpoint "http://localhost:${PORT}/v1"
        --prompt-style "$prompt" --output-dir "$out")
    [ -n "$examples" ] && args+=(--examples "$examples")
    if [ "$DRY" = 1 ]; then printf '  %s\n' "${args[*]}"; return 0; fi
    log "Running condition: $label ($model, $prompt${examples:+, few-shot})"
    if "${args[@]}"; then log "  ✓ $label -> $out"; else log "  ✗ $label FAILED (continuing)"; fi
}

# ─── Base model: C0, C2, C3, C4 (served once) ────────────────────────────────
log_section "Base model: $MODEL_BASE"
if [ "$DRY" = 1 ] || start_vllm "$MODEL_BASE" "$PORT" "logs/vllm_base_${TS}.log" "$MAX_MODEL_LEN"; then
    run_condition C0_baseline      "$MODEL_BASE" sys4        ""
    run_condition C2_single_trait  "$MODEL_BASE" margin_only ""
    run_condition C3_enriched_desc "$MODEL_BASE" margin_rich ""
    if ls "$EXEMPLAR_DIR"/*.png >/dev/null 2>&1; then
        run_condition C4_image_fewshot "$MODEL_BASE" margin_rich "$EXAMPLES"
    else
        log "C4 skipped: no exemplar images in $EXEMPLAR_DIR/ (copy entire.png/toothed.png/lobed.png there)"
    fi
    [ "$DRY" = 1 ] || stop_vllm "$PORT"
else
    log "ERROR: base model failed to serve — skipping C0/C2/C3/C4"
fi

# ─── Upgraded model: C1 (best-effort) ────────────────────────────────────────
if [ "${SKIP_C1:-0}" = 1 ]; then
    log "C1 skipped (SKIP_C1=1)"
elif [ "$DRY" = 1 ]; then
    log_section "Upgraded model: $MODEL_UPGRADED"
    run_condition C1_upgraded_model "$MODEL_UPGRADED" sys4 ""
else
    log_section "Upgraded model: $MODEL_UPGRADED (best-effort)"
    if VLLM_EXTRA_FLAGS="${VLLM_EXTRA_FLAGS_C1:-}" start_vllm "$MODEL_UPGRADED" "$PORT" "logs/vllm_upgraded_${TS}.log" "$MAX_MODEL_LEN"; then
        run_condition C1_upgraded_model "$MODEL_UPGRADED" sys4 ""
        stop_vllm "$PORT"
    else
        log "C1 skipped: $MODEL_UPGRADED failed to serve on vLLM (set MODEL_UPGRADED to a servable id, or SKIP_C1=1)"
    fi
fi

# ─── Grade + compare everything that ran ─────────────────────────────────────
log_section "Grading + comparison"
if [ "$DRY" = 1 ]; then
    log "Would run: leaf_margin_headroom.py + compare_trait_experiments.py -> $REPORT_DIR"
    exit 0
fi

RUNS=()
for label in C0_baseline C1_upgraded_model C2_single_trait C3_enriched_desc C4_image_fewshot; do
    [ -d "$RUNROOT/${label}_${TS}" ] && RUNS+=(--run "${label}=${RUNROOT}/${label}_${TS}")
done
for k in K1_gpt-5.4_all-traits K2_gpt-5.1_per-trait K3_gpt-5.1_per-section; do
    [ -d "$RUNROOT/$k" ] && RUNS+=(--run "${k}=${RUNROOT}/${k}")
done

if [ "${#RUNS[@]}" -eq 0 ]; then
    log "ERROR: no conditions produced output — nothing to compare"
    exit 1
fi

# Headroom readout on the fresh baseline (if it ran).
[ -d "$RUNROOT/C0_baseline_${TS}" ] && \
    python scripts/leaf_margin_headroom.py --results-dir "$RUNROOT/C0_baseline_${TS}" || true

python scripts/compare_trait_experiments.py "${RUNS[@]}" \
    --baseline C0_baseline --out-dir "$REPORT_DIR"
log "Done. Report: $REPORT_DIR/leaf_margin_comparison.html"
