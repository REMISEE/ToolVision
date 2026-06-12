#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OCR_PORT="${OCR_PORT:-28080}"
export GROUNDEDSAM2_PORT="${GROUNDEDSAM2_PORT:-28081}"
export DEPTH_PORT="${DEPTH_PORT:-28082}"
export COUNTGD_PORT="${COUNTGD_PORT:-28083}"

export OCR_CUDA_VISIBLE_DEVICES="${OCR_CUDA_VISIBLE_DEVICES:-0}"
export GROUNDEDSAM2_CUDA_VISIBLE_DEVICES="${GROUNDEDSAM2_CUDA_VISIBLE_DEVICES:-0}"
export DEPTH_CUDA_VISIBLE_DEVICES="${DEPTH_CUDA_VISIBLE_DEVICES:-0}"
export COUNTGD_CUDA_VISIBLE_DEVICES="${COUNTGD_CUDA_VISIBLE_DEVICES:-0}"

export DEPTH_GROUNDEDSAM2_BASE_URL="${DEPTH_GROUNDEDSAM2_BASE_URL:-http://127.0.0.1:${GROUNDEDSAM2_PORT}}"

export SERVICE_LOG_DIR="${SERVICE_LOG_DIR:-${SCRIPT_DIR}/../outputs/service_logs_280xx}"
export SERVICE_PID_DIR="${SERVICE_PID_DIR:-${SCRIPT_DIR}/../outputs/service_pids_280xx}"

action="${1:-status}"
target="${2:-all}"

if [[ "$action" == "warmup" ]]; then
  cd "${SCRIPT_DIR}/.."
  python3 scripts/warmup_external_services.py "$target" \
    --ocr-host 127.0.0.1 \
    --ocr-port "$OCR_PORT" \
    --groundedsam2-host 127.0.0.1 \
    --groundedsam2-port "$GROUNDEDSAM2_PORT" \
    --depth-host 127.0.0.1 \
    --depth-port "$DEPTH_PORT" \
    --countgd-host 127.0.0.1 \
    --countgd-port "$COUNTGD_PORT" \
    --timeout-s "${SERVICE_WARMUP_TIMEOUT_S:-300}"
  exit 0
fi

export SKIP_WARMUP="${SKIP_WARMUP:-1}"
exec bash "${SCRIPT_DIR}/launch_external_services.sh" "$action" "$target"
