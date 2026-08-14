#!/bin/bash
# =============================================================================
# Common helper functions for VLM benchmarking
# =============================================================================
# Source this file from benchmark scripts:
#   source "$(dirname "$0")/common.sh"
#
# Provides:
#   - start_vllm MODEL PORT [LOG_FILE] [MAX_MODEL_LEN]
#   - stop_vllm PORT
#   - wait_for_health URL TIMEOUT
#   - find_free_port
#   - log, log_section

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"  # 10 minutes (Qwen3.5 flashinfer JIT needs ~5min)
VLLM_HEALTH_INTERVAL="${VLLM_HEALTH_INTERVAL:-10}"   # seconds between health checks
VLLM_PID_FILE="/tmp/vllm_benchmark_$$.pid"

# Ensure HF cache is set
if [[ -z "${HF_HOME:-}" ]]; then
    export HF_HOME="${HOME}/.cache/huggingface"
fi

# =============================================================================
# Logging
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_section() {
    echo ""
    echo "============================================================================="
    log "$*"
    echo "============================================================================="
}

# =============================================================================
# Port Management
# =============================================================================

find_free_port() {
    python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

# =============================================================================
# vLLM Management
# =============================================================================

# Wait for vLLM health endpoint to respond
# Usage: wait_for_health URL TIMEOUT_SECONDS
wait_for_health() {
    local url="$1"
    local timeout="$2"
    local elapsed=0

    log "Waiting for vLLM at $url (timeout: ${timeout}s)..."

    while [[ $elapsed -lt $timeout ]]; do
        if curl -s "${url}/models" >/dev/null 2>&1; then
            log "vLLM is ready (took ${elapsed}s)"
            return 0
        fi
        sleep "$VLLM_HEALTH_INTERVAL"
        elapsed=$((elapsed + VLLM_HEALTH_INTERVAL))
        echo -n "."
    done

    echo ""
    log "ERROR: vLLM failed to start within ${timeout}s"
    return 1
}

# Start vLLM server in background
# Usage: start_vllm MODEL PORT [LOG_FILE] [MAX_MODEL_LEN]
start_vllm() {
    local model="$1"
    local port="$2"
    local log_file="${3:-/tmp/vllm_${port}.log}"
    local max_model_len="${4:-32768}"

    log "Starting vLLM: $model on port $port"
    log "Context length: $max_model_len"
    log "Log file: $log_file"

    # Kill any existing vLLM on this port
    stop_vllm "$port" 2>/dev/null || true

    # Build vLLM command with optional extra flags from VLLM_EXTRA_FLAGS env
    local extra_flags="${VLLM_EXTRA_FLAGS:-}"

    # Start vLLM in background.
    # VLLM_BIN lets a caller serve with a different vLLM install (e.g. a newer
    # venv that supports a model architecture the main venv doesn't) while the
    # pipeline client keeps running from the active environment. Defaults to the
    # vllm on PATH, so existing callers are unaffected.
    # shellcheck disable=SC2086
    "${VLLM_BIN:-vllm}" serve "$model" \
        --dtype auto \
        --trust-remote-code \
        --port "$port" \
        --allowed-local-media-path / \
        --max-model-len "$max_model_len" \
        --limit-mm-per-prompt '{"image": 10}' \
        $extra_flags \
        > "$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$VLLM_PID_FILE"
    log "vLLM started with PID $pid"

    # Wait for health
    if ! wait_for_health "http://localhost:${port}/v1" "$VLLM_STARTUP_TIMEOUT"; then
        log "ERROR: vLLM failed to become healthy"
        stop_vllm "$port" || true
        return 1
    fi

    return 0
}

# Stop vLLM server
# Usage: stop_vllm PORT
stop_vllm() {
    local port="$1"

    log "Stopping vLLM on port $port..."

    # Try PID file first
    if [[ -f "$VLLM_PID_FILE" ]]; then
        local pid
        pid=$(cat "$VLLM_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            sleep 2
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$VLLM_PID_FILE"
    fi

    # Also kill by port (backup)
    local pids
    pids=$(lsof -ti ":$port" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo "$pids" | xargs kill 2>/dev/null || true
        sleep 2
        echo "$pids" | xargs kill -9 2>/dev/null || true
    fi

    log "vLLM stopped"
}

# Cleanup on exit
cleanup_vllm() {
    if [[ -f "$VLLM_PID_FILE" ]]; then
        local pid
        pid=$(cat "$VLLM_PID_FILE")
        kill "$pid" 2>/dev/null || true
        rm -f "$VLLM_PID_FILE"
    fi
}

# Register cleanup trap
trap cleanup_vllm EXIT
