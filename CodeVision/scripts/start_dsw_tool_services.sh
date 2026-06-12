#!/usr/bin/env bash
set -euo pipefail

# Start CodeVision external tool services on the current DSW instance.
# Defaults use ports 18080-18083 because this is the DLC-reachable range verified
# from the current PAI VPC setup.

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
TOOLS_CUDA_VISIBLE_DEVICES="${TOOLS_CUDA_VISIBLE_DEVICES:-0}"

export OCR_HOST="${OCR_HOST:-0.0.0.0}"
export GROUNDEDSAM2_HOST="${GROUNDEDSAM2_HOST:-0.0.0.0}"
export DEPTH_HOST="${DEPTH_HOST:-0.0.0.0}"
export COUNTGD_HOST="${COUNTGD_HOST:-0.0.0.0}"

export OCR_PORT="${OCR_PORT:-18080}"
export GROUNDEDSAM2_PORT="${GROUNDEDSAM2_PORT:-18081}"
export DEPTH_PORT="${DEPTH_PORT:-18082}"
export COUNTGD_PORT="${COUNTGD_PORT:-18083}"

export OCR_CUDA_VISIBLE_DEVICES="${OCR_CUDA_VISIBLE_DEVICES:-${TOOLS_CUDA_VISIBLE_DEVICES}}"
export GROUNDEDSAM2_CUDA_VISIBLE_DEVICES="${GROUNDEDSAM2_CUDA_VISIBLE_DEVICES:-${TOOLS_CUDA_VISIBLE_DEVICES}}"
export DEPTH_CUDA_VISIBLE_DEVICES="${DEPTH_CUDA_VISIBLE_DEVICES:-${TOOLS_CUDA_VISIBLE_DEVICES}}"
export COUNTGD_CUDA_VISIBLE_DEVICES="${COUNTGD_CUDA_VISIBLE_DEVICES:-${TOOLS_CUDA_VISIBLE_DEVICES}}"
export DEPTH_GROUNDEDSAM2_BASE_URL="${DEPTH_GROUNDEDSAM2_BASE_URL:-http://127.0.0.1:${GROUNDEDSAM2_PORT}}"

export SERVICE_LOG_DIR="${SERVICE_LOG_DIR:-${ROOT_DIR}/outputs/dsw_tool_services/logs}"
export SERVICE_PID_DIR="${SERVICE_PID_DIR:-${ROOT_DIR}/outputs/dsw_tool_services/pids}"
export SERVICE_WARMUP_TIMEOUT_S="${SERVICE_WARMUP_TIMEOUT_S:-600}"

cd "${ROOT_DIR}"

check_port_free() {
  local port="$1"
  if command -v ss >/dev/null 2>&1 && ss -ltn "( sport = :${port} )" | awk 'NR > 1 { found=1 } END { exit found ? 0 : 1 }'; then
    echo "Port ${port} is already in use. Stop the existing process or override tool ports before starting services." >&2
    ss -ltnp "( sport = :${port} )" >&2 || true
    exit 1
  fi
}

if [[ "${ALLOW_OCCUPIED_TOOL_PORTS:-0}" != "1" ]]; then
  check_port_free "${OCR_PORT}"
  check_port_free "${GROUNDEDSAM2_PORT}"
  check_port_free "${DEPTH_PORT}"
  check_port_free "${COUNTGD_PORT}"
fi

bash scripts/launch_external_services.sh start all
bash scripts/launch_external_services.sh status all
bash scripts/dsw_tool_urls.sh
