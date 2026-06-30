#!/usr/bin/env bash
set -euo pipefail

# Submit a long-running OpenAI-compatible step-judge service to DLC.
# Default server backend is `vllm serve`; set JUDGE_SERVER_CMD to override.

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
DLC_BIN="${DLC_BIN:-$(command -v dlc_pai 2>/dev/null || command -v dlc 2>/dev/null || echo /etc/dsw/runtime/export_bin/dlc)}"

JOB_NAME="${JOB_NAME:-cv-step-judge-1gpu}"
WORKER_IMAGE="${WORKER_IMAGE:-dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/torcheasyrec:1.1.0-pytorch2.10.0-gpu-py311-cu129-ubuntu22.04}"

JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-step-judge}"
JUDGE_REPLICA_COUNT="${JUDGE_REPLICA_COUNT:-1}"
JUDGE_REPLICA_GPUS="${JUDGE_REPLICA_GPUS:-0}"
JUDGE_REPLICA_GPU_GROUPS="${JUDGE_REPLICA_GPU_GROUPS:-}"
JUDGE_PORT_BASE="${JUDGE_PORT_BASE:-19080}"
JUDGE_PORT_STRIDE="${JUDGE_PORT_STRIDE:-10}"
JUDGE_TP_SIZE="${JUDGE_TP_SIZE:-1}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-8192}"
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.85}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
JUDGE_SERVER_CMD="${JUDGE_SERVER_CMD:-}"
JUDGE_VLLM_EXTRA_ARGS="${JUDGE_VLLM_EXTRA_ARGS:-}"
JUDGE_RUN_ID="${JUDGE_RUN_ID:-${JOB_NAME}}"
JUDGE_WARMUP_TIMEOUT_S="${JUDGE_WARMUP_TIMEOUT_S:-900}"
JUDGE_KEEPALIVE_SECONDS="${JUDGE_KEEPALIVE_SECONDS:-300}"
SKIP_JUDGE_WARMUP="${SKIP_JUDGE_WARMUP:-0}"
DRY_RUN="${DRY_RUN:-0}"

DATA_SOURCE_URIS="${DATA_SOURCE_URIS:-cpfs://cpfs-298fffb575a502fe.cn-wulanchabu/ptc-29f47d9393ad2b16/exp-29f2869e7d984aa6/::/mnt/cpfs,oss://pai-wlcb-ai-oss.oss-cn-wulanchabu-internal.aliyuncs.com/::/mnt/oss}"
RESOURCE_ID="${RESOURCE_ID:-quotaev2tl4w6aw0}"
WORKSPACE_ID="${WORKSPACE_ID:-240810}"
VPC_ID="${VPC_ID:-vpc-0jl5rpw5qokp6p2ettip6}"
SWITCH_ID="${SWITCH_ID:-vsw-0jlmr9rjzed093yr9c0kz}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-sg-0jl0pd5qaerdj75wmred}"
PRIORITY="${PRIORITY:-8}"
EXTENDED_CIDRS="${EXTENDED_CIDRS:-10.1.255.0/29,10.1.255.8/29,10.1.16.0/20}"
CUSTOM_PORT_LAST=$((JUDGE_PORT_BASE + (JUDGE_REPLICA_COUNT - 1) * JUDGE_PORT_STRIDE))
ADVANCED_SETTINGS="${ADVANCED_SETTINGS:-createSvcForAllWorkers=true,customPortList=${JUDGE_PORT_BASE}-${CUSTOM_PORT_LAST}}"
DLC_WORKERS="${DLC_WORKERS:-1}"
WORKER_CPU="${WORKER_CPU:-32}"
WORKER_MEMORY="${WORKER_MEMORY:-300Gi}"
WORKER_SHARED_MEMORY="${WORKER_SHARED_MEMORY:-300Gi}"
WORKER_GPU="${WORKER_GPU:-$((JUDGE_REPLICA_COUNT * JUDGE_TP_SIZE))}"

if [[ -z "${JUDGE_SERVER_CMD}" && -z "${JUDGE_MODEL_PATH}" ]]; then
  echo "JUDGE_MODEL_PATH is required unless JUDGE_SERVER_CMD is set." >&2
  exit 2
fi

shell_quote() {
  printf '%q' "$1"
}

append_env() {
  local name="$1"
  local value="$2"
  COMMAND+=" ${name}=$(shell_quote "${value}")"
}

COMMAND="cd $(shell_quote "${ROOT_DIR}") && bash -n scripts/dlc_step_judge_entrypoint.sh && cp scripts/dlc_step_judge_entrypoint.sh /tmp/dlc_step_judge_entrypoint.sh && chmod +x /tmp/dlc_step_judge_entrypoint.sh &&"
append_env JUDGE_MODEL_PATH "${JUDGE_MODEL_PATH}"
append_env JUDGE_MODEL_NAME "${JUDGE_MODEL_NAME}"
append_env JUDGE_REPLICA_COUNT "${JUDGE_REPLICA_COUNT}"
append_env JUDGE_REPLICA_GPUS "${JUDGE_REPLICA_GPUS}"
append_env JUDGE_REPLICA_GPU_GROUPS "${JUDGE_REPLICA_GPU_GROUPS}"
append_env JUDGE_PORT_BASE "${JUDGE_PORT_BASE}"
append_env JUDGE_PORT_STRIDE "${JUDGE_PORT_STRIDE}"
append_env JUDGE_TP_SIZE "${JUDGE_TP_SIZE}"
append_env JUDGE_MAX_MODEL_LEN "${JUDGE_MAX_MODEL_LEN}"
append_env JUDGE_GPU_MEMORY_UTILIZATION "${JUDGE_GPU_MEMORY_UTILIZATION}"
if [[ -n "${JUDGE_API_KEY}" ]]; then
  append_env JUDGE_API_KEY "${JUDGE_API_KEY}"
fi
append_env JUDGE_SERVER_CMD "${JUDGE_SERVER_CMD}"
append_env JUDGE_VLLM_EXTRA_ARGS "${JUDGE_VLLM_EXTRA_ARGS}"
append_env JUDGE_RUN_ID "${JUDGE_RUN_ID}"
append_env JUDGE_WARMUP_TIMEOUT_S "${JUDGE_WARMUP_TIMEOUT_S}"
append_env JUDGE_KEEPALIVE_SECONDS "${JUDGE_KEEPALIVE_SECONDS}"
append_env SKIP_JUDGE_WARMUP "${SKIP_JUDGE_WARMUP}"
COMMAND+=" bash /tmp/dlc_step_judge_entrypoint.sh"

echo "Submitting ${JOB_NAME}"
echo "DLC_BIN=${DLC_BIN}"
echo "JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH:-<custom-command>}"
echo "JUDGE_MODEL_NAME=${JUDGE_MODEL_NAME}"
echo "JUDGE_REPLICA_COUNT=${JUDGE_REPLICA_COUNT}"
echo "JUDGE_REPLICA_GPUS=${JUDGE_REPLICA_GPUS}"
echo "JUDGE_REPLICA_GPU_GROUPS=${JUDGE_REPLICA_GPU_GROUPS:-<single-gpu-list>}"
echo "JUDGE_PORT_BASE=${JUDGE_PORT_BASE}"
echo "JUDGE_PORT_STRIDE=${JUDGE_PORT_STRIDE}"
echo "JUDGE_TP_SIZE=${JUDGE_TP_SIZE}"
echo "JUDGE_MAX_MODEL_LEN=${JUDGE_MAX_MODEL_LEN}"
echo "JUDGE_GPU_MEMORY_UTILIZATION=${JUDGE_GPU_MEMORY_UTILIZATION}"
echo "JUDGE_API_KEY_SET=$([[ -n "${JUDGE_API_KEY}" ]] && echo yes || echo no)"
echo "SKIP_JUDGE_WARMUP=${SKIP_JUDGE_WARMUP}"
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
echo "Replica URLs:"
for ((i = 0; i < JUDGE_REPLICA_COUNT; i++)); do
  port=$((JUDGE_PORT_BASE + i * JUDGE_PORT_STRIDE))
  echo "  replica ${i}: STEP_JUDGE_BASE_URL=http://<judge-host>:${port}"
done

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  echo "DRY_RUN=1, not submitting."
  printf '%s\n' "${DLC_BIN} submit pytorchjob --name=${JOB_NAME} --command=$(shell_quote "${COMMAND}") ..." | sed \
    -e 's/\(JUDGE_API_KEY=\)[^\\ ]*/\1<redacted>/g'
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
