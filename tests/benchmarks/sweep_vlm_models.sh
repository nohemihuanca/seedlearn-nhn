#!/bin/bash
# =============================================================================
# VLM Model Sweep for Morphological Extraction
# =============================================================================
# Cycles through VLM models, running each on the benchmark specimen set.
# Each model's results are saved to a timestamped directory.
#
# Usage:
#   # Full sweep (all 6 models, all 21 specimens)
#   ./tests/benchmarks/sweep_vlm_models.sh
#
#   # Single specimen (quick iteration)
#   ./tests/benchmarks/sweep_vlm_models.sh --specimen "Fabaceae_Inga_punctata"
#
#   # Single model
#   ./tests/benchmarks/sweep_vlm_models.sh --model "Qwen/Qwen3-VL-32B-Instruct-FP8"
#
#   # Both (fastest iteration)
#   ./tests/benchmarks/sweep_vlm_models.sh --specimen "Fabaceae_Inga_punctata" \
#       --model "Qwen/Qwen3-VL-32B-Instruct-FP8"
#
# Prerequisites:
#   - Run on a GPU node: srun --partition=gpu_h200 --gpus=1 --mem=64G --time=04:00:00 --cpus-per-task=8 --pty bash
#   - Activate venv: source .venv/bin/activate
# =============================================================================

set -euo pipefail

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# Source common helpers
source "$SCRIPT_DIR/common.sh"

# =============================================================================
# Configuration
# =============================================================================

# ─── Model Registry ──────────────────────────────────────────────────────
# Models verified against vLLM v0.19.1 architecture registry.
# Models requiring newer vLLM are commented out with upgrade notes.

DEFAULT_MODELS=(
    # --- Supported on vLLM v0.14.0rc2+ ---
    "Qwen/Qwen3-VL-32B-Thinking-FP8"       # Drop-in thinking variant
    "google/gemma-3-27b-it"                 # BF16, ~54GB
    "RedHatAI/gemma-3-27b-it-FP8-dynamic"  # FP8, ~27GB
    # --- Qwen3.5 early-fusion models (requires vLLM v0.18+) ---
    "Qwen/Qwen3.5-27B-FP8"
    # "Qwen/Qwen3.5-122B-A10B-FP8"  # OOM on single H200 — needs TP=2
    # --- Gemma 4 (requires vLLM v0.19+) ---
    "google/gemma-4-31b-it"                 # BF16, ~62GB, hybrid SWA+global attention
    # --- Requires custom vLLM fork (PR #37081 pending) ---
    # "mistralai/Mistral-Small-4-119B-2603" # Needs: mistral_common>=1.10.0 + fork
)

# Per-model context limits (max_model_len for vLLM)
declare -A MODEL_CONTEXT=(
    ["Qwen/Qwen3-VL-32B-Instruct-FP8"]=32768
    ["Qwen/Qwen3-VL-32B-Thinking-FP8"]=32768
    ["google/gemma-3-27b-it"]=32768             # 128K native, plenty of VRAM at BF16
    ["RedHatAI/gemma-3-27b-it-FP8-dynamic"]=32768
    ["Qwen/Qwen3.5-27B-FP8"]=32768
    ["Qwen/Qwen3.5-122B-A10B-FP8"]=8192       # Tight — needs --enforce-eager + 0.99 util
    ["google/gemma-4-31b-it"]=32768              # 256K native, 32K for benchmark VRAM budget
    # ["mistralai/Mistral-Small-4-119B-2603"]=4096
)

# Per-model vLLM extra flags (appended to vllm serve command)
# Source: HuggingFace model cards, verified 2026-03-20
declare -A MODEL_EXTRA_FLAGS=(
    ["Qwen/Qwen3-VL-32B-Instruct-FP8"]=""
    ["Qwen/Qwen3-VL-32B-Thinking-FP8"]=""
    ["google/gemma-3-27b-it"]="--enforce-eager"
    ["RedHatAI/gemma-3-27b-it-FP8-dynamic"]="--enforce-eager --enable-chunked-prefill"
    ["Qwen/Qwen3.5-122B-A10B-FP8"]="--enforce-eager --gpu-memory-utilization 0.99"
    ["google/gemma-4-31b-it"]="--kv-cache-dtype fp8"
)

# Per-model generation parameter overrides (passed to run_vlm_stage1.py)
# Default: temperature=0.6 top_p=0.95 top_k=20 (Qwen3-VL defaults)
# For structured extraction, we use low temperature across all models.
declare -A MODEL_TEMPERATURE=(
    ["Qwen/Qwen3-VL-32B-Instruct-FP8"]="0.1"
    ["Qwen/Qwen3-VL-32B-Thinking-FP8"]="0.1"
    ["google/gemma-3-27b-it"]="0.1"
    ["RedHatAI/gemma-3-27b-it-FP8-dynamic"]="0.1"
    ["google/gemma-4-31b-it"]="0.1"
)

# Models that can only run single-image mode (VRAM-constrained)
declare -A MODEL_SINGLE_ONLY=(
    ["Qwen/Qwen3.5-122B-A10B-FP8"]=1
    # ["mistralai/Mistral-Small-4-119B-2603"]=1
)

# Default context for unknown models
DEFAULT_CONTEXT=32768

# Fixed prompt style
PROMPT_STYLE="sys4"

# Benchmark mode (multi, single, both)
BENCHMARK_MODE="both"

# Default samples file
DEFAULT_SAMPLES="$SCRIPT_DIR/configs/stage1_samples.json"

# Results directory
RESULTS_BASE="$SCRIPT_DIR/results"

# =============================================================================
# Argument Parsing
# =============================================================================

SPECIMEN=""
SINGLE_MODEL=""
SAMPLES_FILE="$DEFAULT_SAMPLES"
WORKERS=8
GROUND_TRUTH="$SCRIPT_DIR/configs/stage1_ground_truth_active.csv"

while [[ $# -gt 0 ]]; do
    case $1 in
        --specimen|-s)
            SPECIMEN="$2"
            shift 2
            ;;
        --model|-m)
            SINGLE_MODEL="$2"
            shift 2
            ;;
        --mode)
            BENCHMARK_MODE="$2"
            shift 2
            ;;
        --workers|-w)
            WORKERS="$2"
            shift 2
            ;;
        --samples)
            SAMPLES_FILE="$2"
            shift 2
            ;;
        --no-score)
            GROUND_TRUTH=""
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --specimen, -s NAME   Run only this specimen (default: all)"
            echo "  --model, -m MODEL     Run only this model (default: all)"
            echo "  --mode MODE           Benchmark mode: multi, single, both (default: both)"
            echo "  --workers, -w N       Concurrent workers for single mode (default: 8)"
            echo "  --samples FILE        Custom samples JSON"
            echo "  --no-score            Skip automatic scoring after each run"
            echo "  --help, -h            Show this help"
            echo ""
            echo "Available models:"
            printf '  %s\n' "${DEFAULT_MODELS[@]}"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================

# Determine which models to run
if [[ -n "$SINGLE_MODEL" ]]; then
    MODELS=("$SINGLE_MODEL")
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi

# Handle single specimen
TEMP_SAMPLES=""
if [[ -n "$SPECIMEN" ]]; then
    log "Filtering to single specimen: $SPECIMEN"
    TEMP_SAMPLES=$(mktemp --suffix=.json)
    python3 -c "
import json
import sys
with open('$SAMPLES_FILE') as f:
    all_samples = json.load(f)
specimen = '$SPECIMEN'
if specimen not in all_samples:
    print(f'ERROR: Specimen \"{specimen}\" not found in samples file', file=sys.stderr)
    print(f'Available: {list(all_samples.keys())}', file=sys.stderr)
    sys.exit(1)
with open('$TEMP_SAMPLES', 'w') as f:
    json.dump({specimen: all_samples[specimen]}, f, indent=2)
print(f'Created temp samples with 1 specimen: $TEMP_SAMPLES')
"
    SAMPLES_FILE="$TEMP_SAMPLES"
fi

# Cleanup temp file on exit
cleanup() {
    if [[ -n "$TEMP_SAMPLES" && -f "$TEMP_SAMPLES" ]]; then
        rm -f "$TEMP_SAMPLES"
    fi
    cleanup_vllm
}
trap cleanup EXIT

# Find free port
PORT=$(find_free_port)
log "Using port: $PORT"

# Create results base directory
mkdir -p "$RESULTS_BASE"

# Track all output directories for final comparison
declare -a OUTPUT_DIRS=()

# =============================================================================
# Main Loop
# =============================================================================

log_section "VLM Model Sweep"
log "Models: ${#MODELS[@]}"
log "Samples: $SAMPLES_FILE"
log "Prompt: $PROMPT_STYLE"
log "Mode: $BENCHMARK_MODE"
log "Workers: $WORKERS"
log "Port: $PORT"
if [[ -n "$GROUND_TRUTH" ]]; then
    log "Auto-scoring: $GROUND_TRUTH"
fi

SWEEP_START=$(date +%s)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for MODEL in "${MODELS[@]}"; do
    log_section "MODEL: $MODEL"

    # Create output directory with timestamp and model name
    MODEL_SAFE=$(echo "$MODEL" | tr '/' '_')
    OUTPUT_DIR="$RESULTS_BASE/${TIMESTAMP}_${MODEL_SAFE}"
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_DIRS+=("$OUTPUT_DIR")

    VLLM_LOG="$OUTPUT_DIR/vllm.log"

    # Get model-specific context limit
    MAX_MODEL_LEN="${MODEL_CONTEXT[$MODEL]:-$DEFAULT_CONTEXT}"
    log "Context limit: $MAX_MODEL_LEN"

    # Determine mode — some models can only do single-image due to VRAM
    RUN_MODE="$BENCHMARK_MODE"
    if [[ -n "${MODEL_SINGLE_ONLY[$MODEL]:-}" ]]; then
        if [[ "$RUN_MODE" == "both" || "$RUN_MODE" == "multi" ]]; then
            log "WARNING: $MODEL is single-only (VRAM-constrained). Forcing --mode single"
            RUN_MODE="single"
        fi
    fi

    # Get model-specific extra flags
    EXTRA_FLAGS="${MODEL_EXTRA_FLAGS[$MODEL]:-}"
    if [[ -n "$EXTRA_FLAGS" ]]; then
        log "Extra vLLM flags: $EXTRA_FLAGS"
    fi

    # Start vLLM (with extra flags appended via environment)
    export VLLM_EXTRA_FLAGS="$EXTRA_FLAGS"
    if ! start_vllm "$MODEL" "$PORT" "$VLLM_LOG" "$MAX_MODEL_LEN"; then
        log "ERROR: Failed to start vLLM for $MODEL"
        echo "FAILED: vLLM startup" > "$OUTPUT_DIR/summary.txt"
        continue
    fi

    # Run benchmark
    MODEL_START=$(date +%s)

    # Get per-model temperature (default 0.1 for structured extraction)
    TEMP="${MODEL_TEMPERATURE[$MODEL]:-0.1}"
    log "Running benchmark (mode=$RUN_MODE, temp=$TEMP)..."
    if python "$SCRIPT_DIR/run_vlm_stage1.py" \
        --samples "$SAMPLES_FILE" \
        --model "$MODEL" \
        --prompt "$PROMPT_STYLE" \
        --mode "$RUN_MODE" \
        --workers "$WORKERS" \
        --temperature "$TEMP" \
        --port "$PORT" \
        --output-dir "$OUTPUT_DIR" \
        --no-timestamp \
        --save-name "results"; then

        MODEL_END=$(date +%s)
        MODEL_DURATION=$((MODEL_END - MODEL_START))

        log "Benchmark complete: ${MODEL_DURATION}s"
        echo "SUCCESS: ${MODEL_DURATION}s" >> "$OUTPUT_DIR/summary.txt"

        # Auto-score if ground truth available
        if [[ -n "$GROUND_TRUTH" && -f "$GROUND_TRUTH" ]]; then
            log "Scoring results..."
            python "$SCRIPT_DIR/score_vlm_stage1.py" \
                --results "$OUTPUT_DIR" \
                --ground-truth "$GROUND_TRUTH" \
                --output "$OUTPUT_DIR/scores" \
                2>&1 | tail -15
        fi
    else
        log "ERROR: Benchmark failed for $MODEL"
        echo "FAILED: benchmark error" > "$OUTPUT_DIR/summary.txt"
    fi

    # Stop vLLM
    stop_vllm "$PORT"

    # Brief pause for GPU memory cleanup
    log "Waiting for GPU memory cleanup..."
    sleep 5
done

# =============================================================================
# Generate Comparison Report
# =============================================================================

SWEEP_END=$(date +%s)
SWEEP_DURATION=$((SWEEP_END - SWEEP_START))

log_section "SWEEP COMPLETE"
log "Total time: ${SWEEP_DURATION}s ($(( SWEEP_DURATION / 60 ))m)"
log "Results in:"
printf '  %s\n' "${OUTPUT_DIRS[@]}"

# Generate comparison HTML if multiple models ran
if [[ ${#OUTPUT_DIRS[@]} -gt 1 ]]; then
    log "Generating comparison report..."
    COMPARISON_FILE="$RESULTS_BASE/comparison_${TIMESTAMP}.html"

    # Collect all JSON result files
    JSON_FILES=()
    for dir in "${OUTPUT_DIRS[@]}"; do
        if [[ -f "$dir/results.json" ]]; then
            JSON_FILES+=("$dir/results.json")
        fi
    done

    if [[ ${#JSON_FILES[@]} -gt 0 ]]; then
        # Use run_vlm_stage1.py's report generator
        python "$SCRIPT_DIR/run_vlm_stage1.py" --report "$RESULTS_BASE" 2>/dev/null || true
        log "Comparison report: $RESULTS_BASE/comparison_*.html"
    fi
fi

log_section "DONE"
echo ""
echo "To view results:"
echo "  CSV:     ls ${RESULTS_BASE}/${TIMESTAMP}_*/results.csv"
echo "  JSON:    ls ${RESULTS_BASE}/${TIMESTAMP}_*/results.json"
echo "  Scores:  ls ${RESULTS_BASE}/${TIMESTAMP}_*/scores/report.html"
if [[ ${#OUTPUT_DIRS[@]} -gt 1 ]]; then
    echo "  Compare: ${RESULTS_BASE}/comparison_${TIMESTAMP}*.html"
    echo ""
    echo "To compare all scored runs:"
    echo "  python $SCRIPT_DIR/score_vlm_stage1.py \\"
    echo "    --results ${RESULTS_BASE}/${TIMESTAMP}_*/ \\"
    echo "    --ground-truth $SCRIPT_DIR/configs/stage1_ground_truth_active.csv \\"
    echo "    --output ${RESULTS_BASE}/comparison_${TIMESTAMP}/"
fi
