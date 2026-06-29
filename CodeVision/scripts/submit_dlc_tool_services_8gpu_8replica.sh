#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOB_NAME="${JOB_NAME:-cv-tool-svc-8gpu-8rep-$(date +%m%d%H%M)}"
export TOOL_RUN_ID="${TOOL_RUN_ID:-${JOB_NAME}}"
export TOOL_REPLICA_COUNT="${TOOL_REPLICA_COUNT:-8}"
export TOOL_REPLICA_GPUS="${TOOL_REPLICA_GPUS:-0,1,2,3,4,5,6,7}"
export TOOL_PORT_BASE="${TOOL_PORT_BASE:-18080}"
export TOOL_PORT_STRIDE="${TOOL_PORT_STRIDE:-10}"
export WORKER_GPU="${WORKER_GPU:-8}"
export WORKER_CPU="${WORKER_CPU:-110}"
export WORKER_MEMORY="${WORKER_MEMORY:-1500Gi}"
export WORKER_SHARED_MEMORY="${WORKER_SHARED_MEMORY:-1500Gi}"
export PRIORITY="${PRIORITY:-6}"
export SKIP_WARMUP="${SKIP_WARMUP:-1}"
export SERVICE_WARMUP_TIMEOUT_S="${SERVICE_WARMUP_TIMEOUT_S:-600}"
export TOOLS_KEEPALIVE_SECONDS="${TOOLS_KEEPALIVE_SECONDS:-300}"

exec bash "${SCRIPT_DIR}/submit_dlc_tool_services.sh"
