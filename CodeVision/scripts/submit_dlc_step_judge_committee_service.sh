#!/usr/bin/env bash
set -euo pipefail

# Submit the 8-GPU ToolVision step-judge committee service to DLC.
# Layout:
#   GPU0: Qwen3-VL-2B
#   GPU1: Qwen3-VL-4B
#   GPU2: Qwen3-VL-8B
#   GPU3-6: Qwen3-VL-32B TP=4
#   GPU7: Qwen3-VL-8B test endpoint
#   CPU: OpenAI-compatible committee gateway

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
DLC_BIN="${DLC_BIN:-$(command -v dlc_pai 2>/dev/null || command -v dlc 2>/dev/null || echo /etc/dsw/runtime/export_bin/dlc)}"
DLC_REGION="${DLC_REGION:-cn-wulanchabu}"
DLC_ENDPOINT="${DLC_ENDPOINT:-pai-dlc.cn-wulanchabu.aliyuncs.com}"
COMMITTEE_ENV_FILE="${COMMITTEE_ENV_FILE:-/mnt/cpfs/delinmao/log1}"
if [[ -f "${COMMITTEE_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "${COMMITTEE_ENV_FILE}"
  set +a
fi

JOB_NAME="${JOB_NAME:-cv-step-judge-committee-8gpu}"
WORKER_IMAGE="${WORKER_IMAGE:-dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/torcheasyrec:1.1.0-pytorch2.10.0-gpu-py311-cu129-ubuntu22.04}"

JUDGE_RUN_ID="${JUDGE_RUN_ID:-${JOB_NAME}}"
JUDGE_VLLM_ENV="${JUDGE_VLLM_ENV:-/mnt/cpfs/delinmao/envs/codevision_new}"
JUDGE_MODEL_2B_PATH="${JUDGE_MODEL_2B_PATH:-/mnt/cpfs/delinmao/models/Qwen3-VL-2B-Instruct}"
JUDGE_MODEL_4B_PATH="${JUDGE_MODEL_4B_PATH:-/mnt/cpfs/delinmao/models/Qwen3-VL-4B-Instruct}"
JUDGE_MODEL_8B_PATH="${JUDGE_MODEL_8B_PATH:-/mnt/cpfs/delinmao/models/Qwen3-VL-8B-Instruct}"
JUDGE_MODEL_32B_PATH="${JUDGE_MODEL_32B_PATH:-/mnt/cpfs/public_data/public_model/Qwen3-vl/Qwen3-VL-32B-Instruct}"

JUDGE_PORT_2B="${JUDGE_PORT_2B:-19080}"
JUDGE_PORT_4B="${JUDGE_PORT_4B:-19090}"
JUDGE_PORT_8B="${JUDGE_PORT_8B:-19100}"
JUDGE_PORT_32B="${JUDGE_PORT_32B:-19110}"
JUDGE_PORT_8B_TEST="${JUDGE_PORT_8B_TEST:-19120}"
COMMITTEE_PORT="${COMMITTEE_PORT:-19200}"

COMMITTEE_API_KEY="${COMMITTEE_API_KEY:-committee-step-judge-key}"
JUDGE_LOCAL_API_KEY="${JUDGE_LOCAL_API_KEY:-local-step-judge-key}"
COMMITTEE_TEMPERATURE_A="${COMMITTEE_TEMPERATURE_A:-0.2}"
COMMITTEE_TEMPERATURE_B="${COMMITTEE_TEMPERATURE_B:-0.3}"
COMMITTEE_MAX_WORKERS="${COMMITTEE_MAX_WORKERS:-12}"
COMMITTEE_TIMEOUT_S="${COMMITTEE_TIMEOUT_S:-120}"
COMMITTEE_MAX_RETRIES="${COMMITTEE_MAX_RETRIES:-0}"
COMMITTEE_INCLUDE_8B_TEST="${COMMITTEE_INCLUDE_8B_TEST:-0}"

COMMITTEE_API1_NAME="${COMMITTEE_API1_NAME:-qwen36plus_api}"
COMMITTEE_API1_BASE_URL="${COMMITTEE_API1_BASE_URL:-${OFFLINE_SFT_QWEN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}}"
COMMITTEE_API1_MODEL="${COMMITTEE_API1_MODEL:-qwen3.6-plus}"
COMMITTEE_API1_API_KEY="${COMMITTEE_API1_API_KEY:-${OFFLINE_SFT_QWEN_API_KEY:-${DASHSCOPE_API_KEY:-}}}"
COMMITTEE_API1_REQUEST_BODY="${COMMITTEE_API1_REQUEST_BODY:-{\"enable_thinking\":false}}"

COMMITTEE_API2_NAME="${COMMITTEE_API2_NAME:-qwen35_122b_a10b_api}"
COMMITTEE_API2_BASE_URL="${COMMITTEE_API2_BASE_URL:-${OFFLINE_SFT_QWEN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}}"
COMMITTEE_API2_MODEL="${COMMITTEE_API2_MODEL:-qwen3.5-122b-a10b}"
COMMITTEE_API2_API_KEY="${COMMITTEE_API2_API_KEY:-${OFFLINE_SFT_QWEN_API_KEY:-${DASHSCOPE_API_KEY:-}}}"
COMMITTEE_API2_REQUEST_BODY="${COMMITTEE_API2_REQUEST_BODY:-{\"enable_thinking\":false}}"

JUDGE_MAX_MODEL_LEN_SMALL="${JUDGE_MAX_MODEL_LEN_SMALL:-8192}"
JUDGE_MAX_MODEL_LEN_32B="${JUDGE_MAX_MODEL_LEN_32B:-8192}"
JUDGE_GPU_MEMORY_UTILIZATION_SMALL="${JUDGE_GPU_MEMORY_UTILIZATION_SMALL:-0.82}"
JUDGE_GPU_MEMORY_UTILIZATION_32B="${JUDGE_GPU_MEMORY_UTILIZATION_32B:-0.90}"
JUDGE_WARMUP_TIMEOUT_S="${JUDGE_WARMUP_TIMEOUT_S:-1800}"
JUDGE_KEEPALIVE_SECONDS="${JUDGE_KEEPALIVE_SECONDS:-300}"
SKIP_JUDGE_WARMUP="${SKIP_JUDGE_WARMUP:-0}"
JUDGE_VLLM_EXTRA_ARGS_COMMON="${JUDGE_VLLM_EXTRA_ARGS_COMMON:-}"
JUDGE_VLLM_EXTRA_ARGS_32B="${JUDGE_VLLM_EXTRA_ARGS_32B:-}"
DRY_RUN="${DRY_RUN:-0}"

DATA_SOURCE_URIS="${DATA_SOURCE_URIS:-cpfs://cpfs-298fffb575a502fe.cn-wulanchabu/ptc-29f47d9393ad2b16/exp-29f2869e7d984aa6/::/mnt/cpfs,oss://pai-wlcb-ai-oss.oss-cn-wulanchabu-internal.aliyuncs.com/::/mnt/oss}"
RESOURCE_ID="${RESOURCE_ID:-quotaev2tl4w6aw0}"
WORKSPACE_ID="${WORKSPACE_ID:-240810}"
VPC_ID="${VPC_ID:-vpc-0jl5rpw5qokp6p2ettip6}"
SWITCH_ID="${SWITCH_ID:-vsw-0jlmr9rjzed093yr9c0kz}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-sg-0jl0pd5qaerdj75wmred}"
PRIORITY="${PRIORITY:-8}"
EXTENDED_CIDRS="${EXTENDED_CIDRS:-10.1.255.0/29,10.1.255.8/29,10.1.16.0/20}"
ADVANCED_SETTINGS="${ADVANCED_SETTINGS:-createSvcForAllWorkers=true,customPortList=${JUDGE_PORT_2B}-${COMMITTEE_PORT}}"
DLC_WORKERS="${DLC_WORKERS:-1}"
WORKER_CPU="${WORKER_CPU:-96}"
WORKER_MEMORY="${WORKER_MEMORY:-900Gi}"
WORKER_SHARED_MEMORY="${WORKER_SHARED_MEMORY:-500Gi}"
WORKER_GPU="${WORKER_GPU:-8}"

shell_quote() {
  printf '%q' "$1"
}

append_env() {
  local name="$1"
  local value="$2"
  COMMAND+=" ${name}=$(shell_quote "${value}")"
}

COMMAND="cd $(shell_quote "${ROOT_DIR}") && bash -n scripts/dlc_step_judge_committee_entrypoint.sh && python3 -m py_compile scripts/step_judge_committee_gateway.py &&"
append_env JUDGE_RUN_ID "${JUDGE_RUN_ID}"
append_env JUDGE_VLLM_ENV "${JUDGE_VLLM_ENV}"
append_env JUDGE_MODEL_2B_PATH "${JUDGE_MODEL_2B_PATH}"
append_env JUDGE_MODEL_4B_PATH "${JUDGE_MODEL_4B_PATH}"
append_env JUDGE_MODEL_8B_PATH "${JUDGE_MODEL_8B_PATH}"
append_env JUDGE_MODEL_32B_PATH "${JUDGE_MODEL_32B_PATH}"
append_env JUDGE_PORT_2B "${JUDGE_PORT_2B}"
append_env JUDGE_PORT_4B "${JUDGE_PORT_4B}"
append_env JUDGE_PORT_8B "${JUDGE_PORT_8B}"
append_env JUDGE_PORT_32B "${JUDGE_PORT_32B}"
append_env JUDGE_PORT_8B_TEST "${JUDGE_PORT_8B_TEST}"
append_env COMMITTEE_PORT "${COMMITTEE_PORT}"
append_env COMMITTEE_ENV_FILE "${COMMITTEE_ENV_FILE}"
append_env COMMITTEE_API_KEY "${COMMITTEE_API_KEY}"
append_env JUDGE_LOCAL_API_KEY "${JUDGE_LOCAL_API_KEY}"
append_env COMMITTEE_TEMPERATURE_A "${COMMITTEE_TEMPERATURE_A}"
append_env COMMITTEE_TEMPERATURE_B "${COMMITTEE_TEMPERATURE_B}"
append_env COMMITTEE_MAX_WORKERS "${COMMITTEE_MAX_WORKERS}"
append_env COMMITTEE_TIMEOUT_S "${COMMITTEE_TIMEOUT_S}"
append_env COMMITTEE_MAX_RETRIES "${COMMITTEE_MAX_RETRIES}"
append_env COMMITTEE_INCLUDE_8B_TEST "${COMMITTEE_INCLUDE_8B_TEST}"
append_env COMMITTEE_API1_NAME "${COMMITTEE_API1_NAME}"
append_env COMMITTEE_API1_BASE_URL "${COMMITTEE_API1_BASE_URL}"
append_env COMMITTEE_API1_MODEL "${COMMITTEE_API1_MODEL}"
append_env COMMITTEE_API1_REQUEST_BODY "${COMMITTEE_API1_REQUEST_BODY}"
append_env COMMITTEE_API2_NAME "${COMMITTEE_API2_NAME}"
append_env COMMITTEE_API2_BASE_URL "${COMMITTEE_API2_BASE_URL}"
append_env COMMITTEE_API2_MODEL "${COMMITTEE_API2_MODEL}"
append_env COMMITTEE_API2_REQUEST_BODY "${COMMITTEE_API2_REQUEST_BODY}"
if [[ ! -f "${COMMITTEE_ENV_FILE}" ]]; then
  append_env COMMITTEE_API1_API_KEY "${COMMITTEE_API1_API_KEY}"
  append_env COMMITTEE_API2_API_KEY "${COMMITTEE_API2_API_KEY}"
fi
append_env JUDGE_MAX_MODEL_LEN_SMALL "${JUDGE_MAX_MODEL_LEN_SMALL}"
append_env JUDGE_MAX_MODEL_LEN_32B "${JUDGE_MAX_MODEL_LEN_32B}"
append_env JUDGE_GPU_MEMORY_UTILIZATION_SMALL "${JUDGE_GPU_MEMORY_UTILIZATION_SMALL}"
append_env JUDGE_GPU_MEMORY_UTILIZATION_32B "${JUDGE_GPU_MEMORY_UTILIZATION_32B}"
append_env JUDGE_WARMUP_TIMEOUT_S "${JUDGE_WARMUP_TIMEOUT_S}"
append_env JUDGE_KEEPALIVE_SECONDS "${JUDGE_KEEPALIVE_SECONDS}"
append_env SKIP_JUDGE_WARMUP "${SKIP_JUDGE_WARMUP}"
append_env JUDGE_VLLM_EXTRA_ARGS_COMMON "${JUDGE_VLLM_EXTRA_ARGS_COMMON}"
append_env JUDGE_VLLM_EXTRA_ARGS_32B "${JUDGE_VLLM_EXTRA_ARGS_32B}"
COMMAND+=" bash scripts/dlc_step_judge_committee_entrypoint.sh"

echo "Submitting ${JOB_NAME}"
echo "DLC_BIN=${DLC_BIN}"
echo "DLC_REGION=${DLC_REGION}"
echo "DLC_ENDPOINT=${DLC_ENDPOINT}"
echo "ROOT_DIR=${ROOT_DIR}"
echo "WORKER_GPU=${WORKER_GPU}"
echo "ADVANCED_SETTINGS=${ADVANCED_SETTINGS}"
echo "Gateway:"
echo "  STEP_JUDGE_BASE_URL=http://<judge-host>:${COMMITTEE_PORT}"
echo "  STEP_JUDGE_MODEL=step-judge-committee"
echo "  STEP_JUDGE_API_KEY=<COMMITTEE_API_KEY>"
echo "Individual endpoints:"
echo "  2B:      http://<judge-host>:${JUDGE_PORT_2B}/v1"
echo "  4B:      http://<judge-host>:${JUDGE_PORT_4B}/v1"
echo "  8B:      http://<judge-host>:${JUDGE_PORT_8B}/v1"
echo "  32B:     http://<judge-host>:${JUDGE_PORT_32B}/v1"
echo "  8B-test: http://<judge-host>:${JUDGE_PORT_8B_TEST}/v1"
echo "API members: api1=$([[ -n "${COMMITTEE_API1_API_KEY}" ]] && echo enabled || echo disabled), api2=$([[ -n "${COMMITTEE_API2_API_KEY}" ]] && echo enabled || echo disabled)"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  echo "DRY_RUN=1, not submitting."
  printf '%s\n' "${DLC_BIN} -r ${DLC_REGION} -e ${DLC_ENDPOINT} submit pytorchjob --name=${JOB_NAME} --command=$(shell_quote "${COMMAND}") ..." | sed \
    -e 's/\(COMMITTEE_API_KEY=\)[^\\ ]*/\1<redacted>/g' \
    -e 's/\(JUDGE_LOCAL_API_KEY=\)[^\\ ]*/\1<redacted>/g' \
    -e 's/\(COMMITTEE_API1_API_KEY=\)[^\\ ]*/\1<redacted>/g' \
    -e 's/\(COMMITTEE_API2_API_KEY=\)[^\\ ]*/\1<redacted>/g'
  exit 0
fi

"${DLC_BIN}" -r "${DLC_REGION}" -e "${DLC_ENDPOINT}" submit pytorchjob \
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
