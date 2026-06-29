#!/usr/bin/env bash
set -euo pipefail

# Submit a long-running DLC job that hosts CodeVision external tool services.
# After the job starts, get the pod IP with:
#   dlc get job <job_id> -w 240810 --show_detail -r cn-wulanchabu -e pai-dlc.cn-wulanchabu.aliyuncs.com
# Then point rollout jobs at http://<pod-ip>:18080-18083 or 18090-18093.

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
DLC_BIN="${DLC_BIN:-$(command -v dlc_pai 2>/dev/null || command -v dlc 2>/dev/null || echo /etc/dsw/runtime/export_bin/dlc)}"

JOB_NAME="${JOB_NAME:-cv-tool-services-2gpu}"
WORKER_IMAGE="${WORKER_IMAGE:-dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/torcheasyrec:1.1.0-pytorch2.10.0-gpu-py311-cu129-ubuntu22.04}"

TOOL_REPLICA_COUNT="${TOOL_REPLICA_COUNT:-2}"
TOOL_REPLICA_GPUS="${TOOL_REPLICA_GPUS:-0,1}"
TOOL_PORT_BASE="${TOOL_PORT_BASE:-18080}"
TOOL_PORT_STRIDE="${TOOL_PORT_STRIDE:-10}"
TOOL_RUN_ID="${TOOL_RUN_ID:-${JOB_NAME}}"
TOOLS_KEEPALIVE_SECONDS="${TOOLS_KEEPALIVE_SECONDS:-300}"
SERVICE_WARMUP_TIMEOUT_S="${SERVICE_WARMUP_TIMEOUT_S:-600}"
SKIP_WARMUP="${SKIP_WARMUP:-1}"
DRY_RUN="${DRY_RUN:-0}"

DATA_SOURCE_URIS="${DATA_SOURCE_URIS:-cpfs://cpfs-298fffb575a502fe.cn-wulanchabu/ptc-29f47d9393ad2b16/exp-29f2869e7d984aa6/::/mnt/cpfs,oss://pai-wlcb-ai-oss.oss-cn-wulanchabu-internal.aliyuncs.com/::/mnt/oss}"
RESOURCE_ID="${RESOURCE_ID:-quotaev2tl4w6aw0}"
WORKSPACE_ID="${WORKSPACE_ID:-240810}"
VPC_ID="${VPC_ID:-vpc-0jl5rpw5qokp6p2ettip6}"
SWITCH_ID="${SWITCH_ID:-vsw-0jlmr9rjzed093yr9c0kz}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-sg-0jl0pd5qaerdj75wmred}"
PRIORITY="${PRIORITY:-9}"
EXTENDED_CIDRS="${EXTENDED_CIDRS:-10.1.255.0/29,10.1.255.8/29,10.1.16.0/20}"
CUSTOM_PORT_LAST=$((TOOL_PORT_BASE + (TOOL_REPLICA_COUNT - 1) * TOOL_PORT_STRIDE + 3))
ADVANCED_SETTINGS="${ADVANCED_SETTINGS:-createSvcForAllWorkers=true,customPortList=${TOOL_PORT_BASE}-${CUSTOM_PORT_LAST}}"
DLC_WORKERS="${DLC_WORKERS:-1}"
WORKER_CPU="${WORKER_CPU:-32}"
WORKER_MEMORY="${WORKER_MEMORY:-300Gi}"
WORKER_SHARED_MEMORY="${WORKER_SHARED_MEMORY:-300Gi}"
WORKER_GPU="${WORKER_GPU:-${TOOL_REPLICA_COUNT}}"

shell_quote() {
  printf '%q' "$1"
}

append_env() {
  local name="$1"
  local value="$2"
  COMMAND+=" ${name}=$(shell_quote "${value}")"
}

COMMAND="cd $(shell_quote "${ROOT_DIR}") && bash -n scripts/dlc_tools_entrypoint.sh && cp scripts/dlc_tools_entrypoint.sh /tmp/dlc_tools_entrypoint.sh && chmod +x /tmp/dlc_tools_entrypoint.sh &&"
append_env TOOL_REPLICA_COUNT "${TOOL_REPLICA_COUNT}"
append_env TOOL_REPLICA_GPUS "${TOOL_REPLICA_GPUS}"
append_env TOOL_PORT_BASE "${TOOL_PORT_BASE}"
append_env TOOL_PORT_STRIDE "${TOOL_PORT_STRIDE}"
append_env TOOL_RUN_ID "${TOOL_RUN_ID}"
append_env TOOLS_KEEPALIVE_SECONDS "${TOOLS_KEEPALIVE_SECONDS}"
append_env SERVICE_WARMUP_TIMEOUT_S "${SERVICE_WARMUP_TIMEOUT_S}"
append_env SKIP_WARMUP "${SKIP_WARMUP}"
COMMAND+=" bash /tmp/dlc_tools_entrypoint.sh"

echo "Submitting ${JOB_NAME}"
echo "DLC_BIN=${DLC_BIN}"
echo "TOOL_REPLICA_COUNT=${TOOL_REPLICA_COUNT}"
echo "TOOL_REPLICA_GPUS=${TOOL_REPLICA_GPUS}"
echo "TOOL_PORT_BASE=${TOOL_PORT_BASE}"
echo "TOOL_PORT_STRIDE=${TOOL_PORT_STRIDE}"
echo "TOOL_RUN_ID=${TOOL_RUN_ID}"
echo "SKIP_WARMUP=${SKIP_WARMUP}"
echo "RESOURCE_ID=${RESOURCE_ID}"
echo "WORKSPACE_ID=${WORKSPACE_ID}"
echo "VPC_ID=${VPC_ID}"
echo "SWITCH_ID=${SWITCH_ID}"
echo "SECURITY_GROUP_ID=${SECURITY_GROUP_ID}"
echo "EXTENDED_CIDRS=${EXTENDED_CIDRS}"
echo "ADVANCED_SETTINGS=${ADVANCED_SETTINGS}"
echo "WORKER_GPU=${WORKER_GPU}"
echo "WORKER_CPU=${WORKER_CPU}"
echo "WORKER_MEMORY=${WORKER_MEMORY}"
echo "WORKER_SHARED_MEMORY=${WORKER_SHARED_MEMORY}"
echo "Port groups:"
for ((i = 0; i < TOOL_REPLICA_COUNT; i++)); do
  base=$((TOOL_PORT_BASE + i * TOOL_PORT_STRIDE))
  echo "  replica ${i}: ${base}-$((base + 3))"
done

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  echo "DRY_RUN=1, not submitting."
  echo "${DLC_BIN} submit pytorchjob --name=${JOB_NAME} --command=$(shell_quote "${COMMAND}") ..."
  exit 0
fi

"${DLC_BIN}" submit pytorchjob \
  --name="${JOB_NAME}" \
  --command="${COMMAND}" \
  --data_source_uris="${DATA_SOURCE_URIS}" \
  --resource_id="${RESOURCE_ID}" \
  --workspace_id="${WORKSPACE_ID}" \
  --vpc_id="${VPC_ID}" \
  --switch_id="${SWITCH_ID}" \
  --security_group_id="${SECURITY_GROUP_ID}" \
  --priority="${PRIORITY}" \
  --extended_cidrs="${EXTENDED_CIDRS}" \
  --advanced_settings="${ADVANCED_SETTINGS}" \
  --workers="${DLC_WORKERS}" \
  --worker_image="${WORKER_IMAGE}" \
  --worker_cpu="${WORKER_CPU}" \
  --worker_memory="${WORKER_MEMORY}" \
  --worker_shared_memory="${WORKER_SHARED_MEMORY}" \
  --worker_gpu="${WORKER_GPU}"
