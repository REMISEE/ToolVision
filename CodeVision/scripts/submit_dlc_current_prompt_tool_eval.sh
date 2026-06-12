#!/usr/bin/env bash
set -euo pipefail

# Submit current-prompt, tool-enabled eval jobs to DLC.
#
# Default matrix:
#   - chartqa, fsc147
#   - validation temperature 0 and 0.7
#
# This is meant to isolate prompt/tool-schema/temperature effects from RL
# training. It runs ONLY_TEST through recipe/codevision/eval_vstar_tools_a100_4gpu.sh.

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
    echo "Set SKIP_TOOL_PORT_CHECK=1 only if you intentionally want to bypass this guard." >&2
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

PROJECT_NAME="${PROJECT_NAME:-CodeVision}"
EXP_PREFIX="${EXP_PREFIX:-current_prompt_tool_eval}"
DATASETS="${DATASETS:-chartqa fsc147}"
TEMPERATURES="${TEMPERATURES:-0 0.7}"

NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
INFER_TP_SIZE="${INFER_TP_SIZE:-4}"
VAL_BSZ="${VAL_BSZ:-32}"
VAL_N_RESP_PER_PROMPT="${VAL_N_RESP_PER_PROMPT:-1}"
MAX_TURNS="${MAX_TURNS:-12}"
ROLLOUT_MAX_TOKENS_PER_TURN="${ROLLOUT_MAX_TOKENS_PER_TURN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
SAVE_EVAL_METADATA="${SAVE_EVAL_METADATA:-1}"
SAVE_VAL_GENERATIONS="${SAVE_VAL_GENERATIONS:-0}"
SAVE_FULL_TRAJECTORY_ALL="${SAVE_FULL_TRAJECTORY_ALL:-0}"
DIAGNOSTIC_MAX_PER_BUCKET="${DIAGNOSTIC_MAX_PER_BUCKET:-200}"
DIAGNOSTIC_SAMPLE_SEED="${DIAGNOSTIC_SAMPLE_SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"

shell_quote() {
  printf '%q' "$1"
}

append_env() {
  local name="$1"
  local value="$2"
  TRAIN_COMMAND+=" ${name}=$(shell_quote "${value}")"
}

dataset_parquet() {
  case "$1" in
    chartqa)
      echo "/mnt/cpfs/delinmao/Benchmarks/ChartQA/chartqa_codevision_eval.parquet"
      ;;
    fsc147)
      echo "/mnt/cpfs/delinmao/Benchmarks/FSC147/fsc147_val_codevision_eval.parquet"
      ;;
    ocrbench)
      echo "/mnt/cpfs/delinmao/Benchmarks/OCRBench/ocrbench_codevision_eval.parquet"
      ;;
    *)
      echo "Unknown dataset '$1'. Supported: chartqa, fsc147, ocrbench." >&2
      exit 1
      ;;
  esac
}

temperature_settings() {
  case "$1" in
    0|0.0)
      echo "False 1.0"
      ;;
    *)
      echo "True ${VAL_TOP_P:-0.95}"
      ;;
  esac
}

if [[ -z "${WORKER_IMAGE}" || "${WORKER_IMAGE}" == *"你的"* || "${WORKER_IMAGE}" == *"TODO"* ]]; then
  echo "WORKER_IMAGE must be a real DLC image URI, got: ${WORKER_IMAGE}" >&2
  exit 1
fi

for dataset in ${DATASETS}; do
  eval_parquet="$(dataset_parquet "${dataset}")"
  if [[ ! -f "${eval_parquet}" ]]; then
    echo "Missing eval parquet for ${dataset}: ${eval_parquet}" >&2
    exit 1
  fi

  for temp in ${TEMPERATURES}; do
    read -r val_do_sample val_top_p <<<"$(temperature_settings "${temp}")"
    temp_tag="t${temp//./p}"
    exp_name="${EXP_PREFIX}_${dataset}_${temp_tag}"
    job_name="${JOB_NAME_PREFIX:-cv-curprompt-eval}-${dataset}-${temp_tag}"
    save_dir="./saves/${PROJECT_NAME}/${exp_name}"

    TRAIN_COMMAND="cd $(shell_quote "${ROOT_DIR}") &&"
    append_env TRAIN_SCRIPT "recipe/codevision/eval_vstar_tools_a100_4gpu.sh"
    append_env MODEL_PATH "${MODEL_PATH}"
    append_env EVAL_PARQUET "${eval_parquet}"
    append_env TOOL_CFG_TEMPLATE_PATH "${TOOL_CFG_TEMPLATE_PATH}"
    append_env SYSTEM_PROMPT_PATH "${SYSTEM_PROMPT_PATH}"
    append_env PROJECT_NAME "${PROJECT_NAME}"
    append_env EXP_NAME "${exp_name}"
    append_env SAVE_DIR "${save_dir}"
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
    append_env VAL_TEMPERATURE "${temp}"
    append_env VAL_TOP_P "${val_top_p}"
    append_env VAL_DO_SAMPLE "${val_do_sample}"
    append_env ROLLOUT_MAX_TOKENS_PER_TURN "${ROLLOUT_MAX_TOKENS_PER_TURN}"
    append_env GPU_MEMORY_UTILIZATION "${GPU_MEMORY_UTILIZATION}"
    append_env MAX_NUM_SEQS "${MAX_NUM_SEQS}"
    append_env ROLLOUT_AGENT_NUM_WORKERS "${ROLLOUT_AGENT_NUM_WORKERS}"
    append_env SAVE_EVAL_METADATA "${SAVE_EVAL_METADATA}"
    append_env SAVE_VAL_GENERATIONS "${SAVE_VAL_GENERATIONS}"
    append_env SAVE_FULL_TRAJECTORY_ALL "${SAVE_FULL_TRAJECTORY_ALL}"
    append_env DIAGNOSTIC_MAX_PER_BUCKET "${DIAGNOSTIC_MAX_PER_BUCKET}"
    append_env DIAGNOSTIC_SAMPLE_SEED "${DIAGNOSTIC_SAMPLE_SEED}"
    TRAIN_COMMAND+=" bash scripts/dlc_ray_direct_entrypoint.sh"

    echo "========== ${job_name} =========="
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "SYSTEM_PROMPT_PATH=${SYSTEM_PROMPT_PATH}"
    echo "TOOL_CFG_TEMPLATE_PATH=${TOOL_CFG_TEMPLATE_PATH}"
    echo "EVAL_PARQUET=${eval_parquet}"
    echo "EXP_NAME=${exp_name}"
    echo "SAVE_DIR=${save_dir}"
    echo "VAL_TEMPERATURE=${temp}"
    echo "VAL_DO_SAMPLE=${val_do_sample}"
    echo "VAL_TOP_P=${val_top_p}"
    echo "ROLLOUT_MAX_TOKENS_PER_TURN=${ROLLOUT_MAX_TOKENS_PER_TURN}"
    echo "NGPUS_PER_NODE=${NGPUS_PER_NODE}"

    if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
      echo "DRY_RUN=1, not submitting."
      echo "${DLC_BIN} submit pytorchjob --name=${job_name} --command=$(shell_quote "${TRAIN_COMMAND}") ..."
      continue
    fi

    "${DLC_BIN}" submit pytorchjob \
      --name="${job_name}" \
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
  done
done
