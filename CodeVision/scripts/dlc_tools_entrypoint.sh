#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
TOOLS_CUDA_VISIBLE_DEVICES="${TOOLS_CUDA_VISIBLE_DEVICES:-0}"

export WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/cpfs/delinmao}"
export TOOLVISION_ROOT="${TOOLVISION_ROOT:-${WORKSPACE_ROOT}/ToolVision}"
export ROOT_DIR

export OCR_HOST="${OCR_HOST:-0.0.0.0}"
export GROUNDEDSAM2_HOST="${GROUNDEDSAM2_HOST:-0.0.0.0}"
export DEPTH_HOST="${DEPTH_HOST:-0.0.0.0}"
export COUNTGD_HOST="${COUNTGD_HOST:-0.0.0.0}"

export OCR_PORT="${OCR_PORT:-8080}"
export GROUNDEDSAM2_PORT="${GROUNDEDSAM2_PORT:-8081}"
export DEPTH_PORT="${DEPTH_PORT:-8082}"
export COUNTGD_PORT="${COUNTGD_PORT:-8083}"

export OCR_CUDA_VISIBLE_DEVICES="${OCR_CUDA_VISIBLE_DEVICES:-${TOOLS_CUDA_VISIBLE_DEVICES}}"
export GROUNDEDSAM2_CUDA_VISIBLE_DEVICES="${GROUNDEDSAM2_CUDA_VISIBLE_DEVICES:-${TOOLS_CUDA_VISIBLE_DEVICES}}"
export DEPTH_CUDA_VISIBLE_DEVICES="${DEPTH_CUDA_VISIBLE_DEVICES:-${TOOLS_CUDA_VISIBLE_DEVICES}}"
export COUNTGD_CUDA_VISIBLE_DEVICES="${COUNTGD_CUDA_VISIBLE_DEVICES:-${TOOLS_CUDA_VISIBLE_DEVICES}}"
export DEPTH_GROUNDEDSAM2_BASE_URL="${DEPTH_GROUNDEDSAM2_BASE_URL:-http://127.0.0.1:${GROUNDEDSAM2_PORT}}"

export SERVICE_LOG_DIR="${SERVICE_LOG_DIR:-${ROOT_DIR}/outputs/dlc_tool_services/logs}"
export SERVICE_PID_DIR="${SERVICE_PID_DIR:-${ROOT_DIR}/outputs/dlc_tool_services/pids}"
export SKIP_WARMUP="${SKIP_WARMUP:-0}"
export SERVICE_WARMUP_TIMEOUT_S="${SERVICE_WARMUP_TIMEOUT_S:-600}"

cd "${ROOT_DIR}"

cleanup() {
  bash scripts/launch_external_services.sh stop all >/dev/null 2>&1 || true
}
trap cleanup EXIT TERM INT

echo "Starting CodeVision tool services"
echo "TOOLS_CUDA_VISIBLE_DEVICES=${TOOLS_CUDA_VISIBLE_DEVICES}"
echo "OCR_BASE_URL=http://<tool-host>:${OCR_PORT}"
echo "GROUNDEDSAM2_BASE_URL=http://<tool-host>:${GROUNDEDSAM2_PORT}"
echo "DEPTH_BASE_URL=http://<tool-host>:${DEPTH_PORT}"
echo "COUNTGD_BASE_URL=http://<tool-host>:${COUNTGD_PORT}"

bash scripts/launch_external_services.sh start all
bash scripts/launch_external_services.sh status all

while true; do
  sleep "${TOOLS_KEEPALIVE_SECONDS:-300}"
  bash scripts/launch_external_services.sh status all
done
