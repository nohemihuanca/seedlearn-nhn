#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for the seedlearn pipeline.
#
# The pipeline's Stage 1 (VLM morphology) and Stage 5 (LLM reasoning) both
# call an OpenAI-compatible endpoint. This script starts vLLM with the correct
# flags for local image access and sensible defaults.
#
# Usage:
#   # Default: Qwen3-VL-32B on port 8000 (handles both Stage 1 and Stage 5)
#   bash scripts/start_vllm.sh
#
#   # Custom model or port
#   bash scripts/start_vllm.sh --model Qwen/Qwen3-VL-32B-Instruct-FP8 --port 8000
#
#   # OOD Interactive Desktop (recommended — see README for GPU desktop settings)
#   # Open a terminal in your OOD Desktop session, then:
#   bash scripts/start_vllm.sh
#
# Environment variables (alternative to flags):
#   VLLM_MODEL   Model name (default: Qwen/Qwen3-VL-32B-Instruct-FP8)
#   VLLM_PORT    Port number (default: 8000)
#
# Health check (from another terminal):
#   curl -s http://localhost:8000/v1/models | python3 -m json.tool
#
# Pipeline usage (from another terminal):
#   python scripts/run_pipeline.py --specimen SRAPHEDE2 \
#       --catalog data/raw/2026-01-29/sorted_12K/metadata/species_catalog_v2026-01-29_12K_20260129_123334.csv \
#       --cache-dir data/embeddings/2026-01-29_v2026-01-29_12K \
#       --split-path data/splits/2026-01-29_v2026-01-29_12K/family/split_seed42 \
#       --rag-index /nfs/roberts/project/pi_lsc4/shared/seedlearn/data/traits/latest/rag_index/ \
#       --vlm-endpoint http://localhost:8000/v1 \
#       --reasoning-endpoint http://localhost:8000/v1

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-32B-Instruct-FP8}"
PORT="${VLLM_PORT:-8000}"

# ─── Parse flags ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)  MODEL="$2"; shift 2 ;;
        --port)   PORT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown flag: $1 (use --help)"; exit 1 ;;
    esac
done

# ─── Resolve repo root ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ─── Local image path ───────────────────────────────────────────────────────
# Stage 1 sends file:// URLs for specimen images. vLLM requires explicit
# permission to read local files via --allowed-local-media-path.
# NOTE: vLLM only supports a SINGLE --allowed-local-media-path flag (the
# argparse type is str, not list — a second flag silently overwrites the
# first). We resolve the repo's data/ symlink to its real filesystem path
# so vLLM's resolved-path check matches.
MEDIA_PATH="$(realpath "$REPO_DIR/data" 2>/dev/null)"
if [[ -z "$MEDIA_PATH" ]]; then
    MEDIA_PATH="/nfs/roberts/project/pi_lsc4/shared/seedlearn/data"
fi

# ─── Environment ─────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════════"
echo "  SeedLearn — vLLM Server"
echo "════════════════════════════════════════════════════════════════════════"
echo "  Model : $MODEL"
echo "  Port  : $PORT"
echo "  GPU   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Host  : $(hostname)"
echo "  Date  : $(date)"
echo ""
echo "  Allowed local media path:"
echo "    - $MEDIA_PATH"
echo ""
echo "  Health check:"
echo "    curl -s http://localhost:${PORT}/v1/models | python3 -m json.tool"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# ─── Start server (foreground) ───────────────────────────────────────────────
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --trust-remote-code \
    --allowed-local-media-path "$MEDIA_PATH" \
    --limit-mm-per-prompt '{"image": 10}' \
    --dtype auto
