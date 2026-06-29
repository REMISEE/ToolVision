#!/usr/bin/env bash
set -euo pipefail

# Submit one DLC job that sequentially runs the current-prompt eval matrix.

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
DLC_BIN="${DLC_BIN:-dlc_pai}"

eval "$("${ROOT_DIR}/scripts/dsw_tool_urls.sh")"

check_tool_port() {
  local name="$1"
  local url="$2"
  local host_port="${url#http://}"
  local host="${host_port%%:*}"
  local port="${host_port##*:}"
  if ! timeout 2 bash -lc "</dev/tcp/${host}/${port}" >/dev/null 2>&1; then
    echo "Tool service ${name} is not reachable at ${url}." >&2
    echo "Run this submit script from the DSW that hosts the tool services, or set DSW_TOOL_HOST to that DSW IP." >&2
    exit 1
  fi
}

if [[ "${SKIP_TOOL_PORT_CHECK:-0}" != "1" ]]; then
  check_tool_port "OCR" "${OCR_BASE_URL}"
  check_tool_port "GroundedSAM2" "${GROUNDEDSAM2_BASE_URL}"
  check_tool_port "Depth" "${DEPTH_BASE_URL}"
  check_tool_port "CountGD" "${COUNTGD_BASE_URL}"
fi

JOB_NAME="${JOB_NAME:-cv-curprompt-eval-matrix}"
WORKER_IMAGE="${WORKER_IMAGE:-dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/torcheasyrec:1.1.0-pytorch2.10.0-gpu-py311-cu129-ubuntu22.04}"

MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mnt/cpfs/delinmao/Benchmarks}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"
SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-recipe/codevision/config/sp3.txt}"
TOOL_CFG_TEMPLATE_PATH="${TOOL_CFG_TEMPLATE_PATH:-recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml}"
PROJECT_NAME="${PROJECT_NAME:-CodeVision}"
EXP_PREFIX="${EXP_PREFIX:-current_prompt_tool_eval_matrix}"
DATASETS="${DATASETS:-fsc147 chartqa}"
TEMPERATURES="${TEMPERATURES:-0 0.7}"

NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
INFER_TP_SIZE="${INFER_TP_SIZE:-4}"
VAL_BSZ="${VAL_BSZ:-32}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-1}"
VAL_N_RESP_PER_PROMPT="${VAL_N_RESP_PER_PROMPT:-1}"
MAX_TURNS="${MAX_TURNS:-12}"
ROLLOUT_MAX_TOKENS_PER_TURN="${ROLLOUT_MAX_TOKENS_PER_TURN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
DRY_RUN="${DRY_RUN:-0}"

shell_quote() {
  printf '%q' "$1"
}

append_env() {
  local name="$1"
  local value="$2"
  TRAIN_COMMAND+=" ${name}=$(shell_quote "${value}")"
}

TRAIN_COMMAND="cd $(shell_quote "${ROOT_DIR}") &&"
append_env JOB_NAME "${JOB_NAME}"
append_env TRAIN_SCRIPT "recipe/codevision/eval_current_prompt_tool_matrix.sh"
append_env MODEL_PATH "${MODEL_PATH}"
append_env BENCHMARK_ROOT "${BENCHMARK_ROOT}"
append_env RESUME_MODE "${RESUME_MODE}"
append_env RESUME_FROM_PATH "${RESUME_FROM_PATH}"
append_env TOOL_CFG_TEMPLATE_PATH "${TOOL_CFG_TEMPLATE_PATH}"
append_env SYSTEM_PROMPT_PATH "${SYSTEM_PROMPT_PATH}"
append_env PROJECT_NAME "${PROJECT_NAME}"
append_env EXP_PREFIX "${EXP_PREFIX}"
append_env DATASETS "${DATASETS}"
append_env TEMPERATURES "${TEMPERATURES}"
append_env OCR_BASE_URL "${OCR_BASE_URL}"
append_env GROUNDEDSAM2_BASE_URL "${GROUNDEDSAM2_BASE_URL}"
append_env DEPTH_BASE_URL "${DEPTH_BASE_URL}"
append_env COUNTGD_BASE_URL "${COUNTGD_BASE_URL}"
append_env CODEVISION_ENV "${CODEVISION_ENV:-/mnt/cpfs/delinmao/envs/codevision_new}"
append_env DLC_ENTRYPOINT_DEBUG "${DLC_ENTRYPOINT_DEBUG:-1}"
append_env RAY_NODE_CHECK_TIMEOUT_SECONDS "${RAY_NODE_CHECK_TIMEOUT_SECONDS:-20}"
append_env TOOL_PREFLIGHT_CHECK "${TOOL_PREFLIGHT_CHECK:-1}"
append_env NGPUS_PER_NODE "${NGPUS_PER_NODE}"
append_env INFER_TP_SIZE "${INFER_TP_SIZE}"
append_env VAL_BSZ "${VAL_BSZ}"
append_env N_RESP_PER_PROMPT "${N_RESP_PER_PROMPT}"
append_env VAL_N_RESP_PER_PROMPT "${VAL_N_RESP_PER_PROMPT}"
append_env MAX_TURNS "${MAX_TURNS}"
append_env ROLLOUT_MAX_TOKENS_PER_TURN "${ROLLOUT_MAX_TOKENS_PER_TURN}"
append_env GPU_MEMORY_UTILIZATION "${GPU_MEMORY_UTILIZATION}"
append_env MAX_NUM_SEQS "${MAX_NUM_SEQS}"
append_env ROLLOUT_AGENT_NUM_WORKERS "${ROLLOUT_AGENT_NUM_WORKERS}"
TRAIN_COMMAND+=" bash scripts/dlc_ray_direct_entrypoint.sh"

echo "Submitting ${JOB_NAME}"
echo "DATASETS=${DATASETS}"
echo "TEMPERATURES=${TEMPERATURES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "RESUME_MODE=${RESUME_MODE}"
echo "RESUME_FROM_PATH=${RESUME_FROM_PATH}"
echo "SYSTEM_PROMPT_PATH=${SYSTEM_PROMPT_PATH}"
echo "TOOL_CFG_TEMPLATE_PATH=${TOOL_CFG_TEMPLATE_PATH}"
echo "OCR_BASE_URL=${OCR_BASE_URL}"
echo "GROUNDEDSAM2_BASE_URL=${GROUNDEDSAM2_BASE_URL}"
echo "DEPTH_BASE_URL=${DEPTH_BASE_URL}"
echo "COUNTGD_BASE_URL=${COUNTGD_BASE_URL}"
echo "NGPUS_PER_NODE=${NGPUS_PER_NODE}"
echo "N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT}"
echo "VAL_N_RESP_PER_PROMPT=${VAL_N_RESP_PER_PROMPT}"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  echo "DRY_RUN=1, not submitting."
  echo "${DLC_BIN} submit pytorchjob --name=${JOB_NAME} --command=$(shell_quote "${TRAIN_COMMAND}") ..."
  exit 0
fi

"${DLC_BIN}" submit pytorchjob \
  --name="${JOB_NAME}" \
  --command="${TRAIN_COMMAND}" \
  --data_source_uris="${DATA_SOURCE_URIS:-cpfs://cpfs-298fffb575a502fe.cn-wulanchabu/ptc-29f47d9393ad2b16/exp-29f2869e7d984aa6/::/mnt/cpfs,oss://pai-wlcb-ai-oss.oss-cn-wulanchabu-internal.aliyuncs.com/::/mnt/oss}" \
  --resource_id="${RESOURCE_ID:-quotaev2tl4w6aw0}" \
  --workspace_id="${WORKSPACE_ID:-240810}" \
  --vpc_id="${VPC_ID:-vpc-0jl5rpw5qokp6p2ettip6}" \
  --switch_id="${SWITCH_ID:-vsw-0jlmr9rjzed093yr9c0kz}" \
  --security_group_id="${SECURITY_GROUP_ID:-sg-0jl0pd5qaerdj75wmred}" \
  --priority="${PRIORITY:-8}" \
  --extended_cidrs="${EXTENDED_CIDRS:-10.1.255.0/29,10.1.255.8/29,10.1.16.0/20}" \
  --advanced_settings="${ADVANCED_SETTINGS:-createSvcForAllWorkers=true,customPortList=6379;6380-6383;8265;20000-25000}" \
  --workers="${DLC_WORKERS:-1}" \
  --worker_image="${WORKER_IMAGE}" \
  --worker_cpu="${WORKER_CPU:-110}" \
  --worker_memory="${WORKER_MEMORY:-1500Gi}" \
  --worker_shared_memory="${WORKER_SHARED_MEMORY:-1500Gi}" \
  --worker_gpu="${WORKER_GPU:-${NGPUS_PER_NODE}}"
