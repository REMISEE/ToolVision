#!/usr/bin/env bash
set -euo pipefail

# Submit one ToolVision MUT rollout8 eval job to DLC.
# Run from the DSW that hosts the external tool services.

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
DLC_BIN="${DLC_BIN:-$(command -v dlc_pai 2>/dev/null || command -v dlc 2>/dev/null || echo /etc/dsw/runtime/export_bin/dlc)}"

eval "$("${ROOT_DIR}/scripts/dsw_tool_urls.sh")"

check_tool_port() {
  local name="$1"
  local url="$2"
  local host_port="${url#http://}"
  local host="${host_port%%:*}"
  local port="${host_port##*:}"
  if ! timeout 2 bash -lc "</dev/tcp/${host}/${port}" >/dev/null 2>&1; then
    echo "Tool service ${name} is not reachable at ${url}." >&2
    echo "Start services first: bash scripts/start_dsw_tool_services.sh" >&2
    echo "Or set DSW_TOOL_HOST/OCR_PORT/GROUNDEDSAM2_PORT/DEPTH_PORT/COUNTGD_PORT explicitly." >&2
    exit 1
  fi
}

if [[ "${SKIP_TOOL_PORT_CHECK:-0}" != "1" ]]; then
  check_tool_port "OCR" "${OCR_BASE_URL}"
  check_tool_port "GroundedSAM2" "${GROUNDEDSAM2_BASE_URL}"
  check_tool_port "Depth" "${DEPTH_BASE_URL}"
  check_tool_port "CountGD" "${COUNTGD_BASE_URL}"
fi

WORKER_IMAGE="${WORKER_IMAGE:-dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/torcheasyrec:1.1.0-pytorch2.10.0-gpu-py311-cu129-ubuntu22.04}"
MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03}"
SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-recipe/codevision/config/sp3.txt}"
TOOL_CFG_TEMPLATE_PATH="${TOOL_CFG_TEMPLATE_PATH:-recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml}"

EVAL_PARQUET="${EVAL_PARQUET:-/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/toolvision_eval/smoke_mut_0_8_128_toolvision_eval.parquet}"
EXP_NAME="${EXP_NAME:-mut_rollout8_smoke_128_t0p7}"
PROJECT_NAME="${PROJECT_NAME:-CodeVision}"
SAVE_DIR="${SAVE_DIR:-./saves/${PROJECT_NAME}/${EXP_NAME}}"
JOB_NAME="${JOB_NAME:-cv-mut-rollout8-smoke}"

NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
INFER_TP_SIZE="${INFER_TP_SIZE:-4}"
VAL_BSZ="${VAL_BSZ:-32}"
VAL_N_RESP_PER_PROMPT="${VAL_N_RESP_PER_PROMPT:-8}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.7}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-True}"
MAX_TURNS="${MAX_TURNS:-12}"
ROLLOUT_MAX_TOKENS_PER_TURN="${ROLLOUT_MAX_TOKENS_PER_TURN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
SAVE_EVAL_METADATA="${SAVE_EVAL_METADATA:-1}"
SAVE_VAL_GENERATIONS="${SAVE_VAL_GENERATIONS:-1}"
SAVE_FULL_TRAJECTORY_ALL="${SAVE_FULL_TRAJECTORY_ALL:-1}"
DIAGNOSTIC_MAX_PER_BUCKET="${DIAGNOSTIC_MAX_PER_BUCKET:-1000000}"
DIAGNOSTIC_SAMPLE_SEED="${DIAGNOSTIC_SAMPLE_SEED:-42}"
STREAM_VALIDATION_DUMP="${STREAM_VALIDATION_DUMP:-True}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${EVAL_PARQUET}" ]]; then
  echo "Missing EVAL_PARQUET: ${EVAL_PARQUET}" >&2
  exit 1
fi
if [[ -z "${WORKER_IMAGE}" || "${WORKER_IMAGE}" == *"你的"* || "${WORKER_IMAGE}" == *"TODO"* ]]; then
  echo "WORKER_IMAGE must be a real DLC image URI, got: ${WORKER_IMAGE}" >&2
  exit 1
fi

shell_quote() {
  printf '%q' "$1"
}

append_env() {
  local name="$1"
  local value="$2"
  TRAIN_COMMAND+=" ${name}=$(shell_quote "${value}")"
}

TRAIN_COMMAND="cd $(shell_quote "${ROOT_DIR}") &&"
append_env TRAIN_SCRIPT "recipe/codevision/eval_vstar_tools_a100_4gpu.sh"
append_env MODEL_PATH "${MODEL_PATH}"
append_env EVAL_PARQUET "${EVAL_PARQUET}"
append_env TOOL_CFG_TEMPLATE_PATH "${TOOL_CFG_TEMPLATE_PATH}"
append_env SYSTEM_PROMPT_PATH "${SYSTEM_PROMPT_PATH}"
append_env PROJECT_NAME "${PROJECT_NAME}"
append_env EXP_NAME "${EXP_NAME}"
append_env SAVE_DIR "${SAVE_DIR}"
append_env OCR_BASE_URL "${OCR_BASE_URL}"
append_env GROUNDEDSAM2_BASE_URL "${GROUNDEDSAM2_BASE_URL}"
append_env DEPTH_BASE_URL "${DEPTH_BASE_URL}"
append_env COUNTGD_BASE_URL "${COUNTGD_BASE_URL}"
append_env DLC_ENTRYPOINT_DEBUG "${DLC_ENTRYPOINT_DEBUG:-1}"
append_env RAY_NODE_CHECK_TIMEOUT_SECONDS "${RAY_NODE_CHECK_TIMEOUT_SECONDS:-20}"
append_env TOOL_PREFLIGHT_CHECK "${TOOL_PREFLIGHT_CHECK:-1}"
append_env NGPUS_PER_NODE "${NGPUS_PER_NODE}"
append_env INFER_TP_SIZE "${INFER_TP_SIZE}"
append_env VAL_BSZ "${VAL_BSZ}"
append_env N_RESP_PER_PROMPT "1"
append_env VAL_N_RESP_PER_PROMPT "${VAL_N_RESP_PER_PROMPT}"
append_env MAX_TURNS "${MAX_TURNS}"
append_env VAL_TEMPERATURE "${VAL_TEMPERATURE}"
append_env VAL_TOP_P "${VAL_TOP_P}"
append_env VAL_DO_SAMPLE "${VAL_DO_SAMPLE}"
append_env ROLLOUT_MAX_TOKENS_PER_TURN "${ROLLOUT_MAX_TOKENS_PER_TURN}"
append_env GPU_MEMORY_UTILIZATION "${GPU_MEMORY_UTILIZATION}"
append_env MAX_NUM_SEQS "${MAX_NUM_SEQS}"
append_env ROLLOUT_AGENT_NUM_WORKERS "${ROLLOUT_AGENT_NUM_WORKERS}"
append_env SAVE_EVAL_METADATA "${SAVE_EVAL_METADATA}"
append_env SAVE_VAL_GENERATIONS "${SAVE_VAL_GENERATIONS}"
append_env SAVE_FULL_TRAJECTORY_ALL "${SAVE_FULL_TRAJECTORY_ALL}"
append_env DIAGNOSTIC_MAX_PER_BUCKET "${DIAGNOSTIC_MAX_PER_BUCKET}"
append_env DIAGNOSTIC_SAMPLE_SEED "${DIAGNOSTIC_SAMPLE_SEED}"
append_env STREAM_VALIDATION_DUMP "${STREAM_VALIDATION_DUMP}"
TRAIN_COMMAND+=" bash scripts/dlc_ray_direct_entrypoint.sh"

echo "Submitting ${JOB_NAME}"
echo "DLC_BIN=${DLC_BIN}"
echo "EVAL_PARQUET=${EVAL_PARQUET}"
echo "EXP_NAME=${EXP_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "SYSTEM_PROMPT_PATH=${SYSTEM_PROMPT_PATH}"
echo "TOOL_CFG_TEMPLATE_PATH=${TOOL_CFG_TEMPLATE_PATH}"
echo "OCR_BASE_URL=${OCR_BASE_URL}"
echo "GROUNDEDSAM2_BASE_URL=${GROUNDEDSAM2_BASE_URL}"
echo "DEPTH_BASE_URL=${DEPTH_BASE_URL}"
echo "COUNTGD_BASE_URL=${COUNTGD_BASE_URL}"
echo "VAL_N_RESP_PER_PROMPT=${VAL_N_RESP_PER_PROMPT}"
echo "VAL_TEMPERATURE=${VAL_TEMPERATURE}"
echo "VAL_TOP_P=${VAL_TOP_P}"
echo "VAL_DO_SAMPLE=${VAL_DO_SAMPLE}"
echo "NGPUS_PER_NODE=${NGPUS_PER_NODE}"
echo "INFER_TP_SIZE=${INFER_TP_SIZE}"
echo "VAL_BSZ=${VAL_BSZ}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS}"
echo "SAVE_FULL_TRAJECTORY_ALL=${SAVE_FULL_TRAJECTORY_ALL}"
echo "STREAM_VALIDATION_DUMP=${STREAM_VALIDATION_DUMP}"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  echo "DRY_RUN=1, not submitting."
  echo "${DLC_BIN} submit pytorchjob --name=${JOB_NAME} --command=$(shell_quote "${TRAIN_COMMAND}") ..."
  exit 0
fi

"${DLC_BIN}" submit pytorchjob \
  --name="${JOB_NAME}" \
  --command="${TRAIN_COMMAND}" \
  --data_source_uris="${DATA_SOURCE_URIS:-cpfs://cpfs-298fffb575a502fe.cn-wulanchabu/ptc-29f47d9393ad2b16/exp-29f2869e7d984aa6/::/mnt/cpfs,oss://pai-wlcb-ai-oss.oss-cn-wulanchabu-internal.aliyuncs.com/::/mnt/oss}" \
  --resource_id="${RESOURCE_ID:-quota1hdkwah70tk}" \
  --workspace_id="${WORKSPACE_ID:-245264}" \
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
