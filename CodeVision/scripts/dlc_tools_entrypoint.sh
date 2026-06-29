#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
TOOL_REPLICA_COUNT="${TOOL_REPLICA_COUNT:-1}"
TOOL_REPLICA_GPUS="${TOOL_REPLICA_GPUS:-0}"
TOOL_PORT_BASE="${TOOL_PORT_BASE:-18080}"
TOOL_PORT_STRIDE="${TOOL_PORT_STRIDE:-10}"
TOOL_RUN_ID="${TOOL_RUN_ID:-${DLC_JOB_ID:-${PAI_JOB_ID:-${HOSTNAME:-tool}_$$}}}"
TOOL_OUTPUT_ROOT="${TOOL_OUTPUT_ROOT:-${ROOT_DIR}/outputs/dlc_tool_services/${TOOL_RUN_ID}}"

export WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/cpfs/delinmao}"
export TOOLVISION_ROOT="${TOOLVISION_ROOT:-${WORKSPACE_ROOT}/ToolVision}"
export ROOT_DIR

# DLC starts replicas serially. Keep startup non-blocking so a slow warmup on
# replica 0 does not prevent replica 1 from binding its ports.
export SKIP_WARMUP="${SKIP_WARMUP:-1}"
export SERVICE_WARMUP_TIMEOUT_S="${SERVICE_WARMUP_TIMEOUT_S:-600}"

cd "${ROOT_DIR}"

IFS=',' read -r -a REPLICA_GPUS <<<"${TOOL_REPLICA_GPUS}"

setup_replica_env() {
  local replica_id="$1"
  local port_base=$((TOOL_PORT_BASE + replica_id * TOOL_PORT_STRIDE))
  local gpu="${REPLICA_GPUS[$replica_id]:-${REPLICA_GPUS[0]:-0}}"

  export OCR_HOST="${OCR_HOST:-0.0.0.0}"
  export GROUNDEDSAM2_HOST="${GROUNDEDSAM2_HOST:-0.0.0.0}"
  export DEPTH_HOST="${DEPTH_HOST:-0.0.0.0}"
  export COUNTGD_HOST="${COUNTGD_HOST:-0.0.0.0}"

  export OCR_PORT="$((port_base + 0))"
  export GROUNDEDSAM2_PORT="$((port_base + 1))"
  export DEPTH_PORT="$((port_base + 2))"
  export COUNTGD_PORT="$((port_base + 3))"

  export OCR_CUDA_VISIBLE_DEVICES="${gpu}"
  export GROUNDEDSAM2_CUDA_VISIBLE_DEVICES="${gpu}"
  export DEPTH_CUDA_VISIBLE_DEVICES="${gpu}"
  export COUNTGD_CUDA_VISIBLE_DEVICES="${gpu}"
  export DEPTH_GROUNDEDSAM2_BASE_URL="http://127.0.0.1:${GROUNDEDSAM2_PORT}"

  export SERVICE_LOG_DIR="${TOOL_OUTPUT_ROOT}/replica_${replica_id}/logs"
  export SERVICE_PID_DIR="${TOOL_OUTPUT_ROOT}/replica_${replica_id}/pids"
}

cleanup() {
  local replica_id
  for ((replica_id = 0; replica_id < TOOL_REPLICA_COUNT; replica_id++)); do
    setup_replica_env "${replica_id}"
    bash scripts/launch_external_services.sh stop all >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT TERM INT

echo "Starting CodeVision tool services"
echo "TOOL_REPLICA_COUNT=${TOOL_REPLICA_COUNT}"
echo "TOOL_REPLICA_GPUS=${TOOL_REPLICA_GPUS}"
echo "TOOL_PORT_BASE=${TOOL_PORT_BASE}"
echo "TOOL_PORT_STRIDE=${TOOL_PORT_STRIDE}"
echo "TOOL_RUN_ID=${TOOL_RUN_ID}"
echo "TOOL_OUTPUT_ROOT=${TOOL_OUTPUT_ROOT}"
echo "SKIP_WARMUP=${SKIP_WARMUP}"

for ((replica_id = 0; replica_id < TOOL_REPLICA_COUNT; replica_id++)); do
  setup_replica_env "${replica_id}"
  echo "Starting tool replica ${replica_id} on gpu=${OCR_CUDA_VISIBLE_DEVICES}"
  echo "OCR_BASE_URL=http://<tool-host>:${OCR_PORT}"
  echo "GROUNDEDSAM2_BASE_URL=http://<tool-host>:${GROUNDEDSAM2_PORT}"
  echo "DEPTH_BASE_URL=http://<tool-host>:${DEPTH_PORT}"
  echo "COUNTGD_BASE_URL=http://<tool-host>:${COUNTGD_PORT}"
  bash scripts/launch_external_services.sh start all
  bash scripts/launch_external_services.sh status all
done

while true; do
  sleep "${TOOLS_KEEPALIVE_SECONDS:-300}"
  for ((replica_id = 0; replica_id < TOOL_REPLICA_COUNT; replica_id++)); do
    setup_replica_env "${replica_id}"
    echo "Replica ${replica_id} status:"
    bash scripts/launch_external_services.sh status all
  done
done
