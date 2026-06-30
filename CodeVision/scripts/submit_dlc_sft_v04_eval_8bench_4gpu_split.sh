#!/usr/bin/env bash
set -euo pipefail

# Submit v04 SFT eval as two parallel 4-GPU DLC jobs.
# Both jobs use the already deployed DLC tool-service replica 3 by default.

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/cpfs/delinmao}"
DLC_BIN="${DLC_BIN:-$([[ -x /mnt/cpfs/delinmao/bin/dlc_pai ]] && echo /mnt/cpfs/delinmao/bin/dlc_pai || command -v dlc_pai 2>/dev/null || command -v dlc 2>/dev/null || echo /etc/dsw/runtime/export_bin/dlc)}"
DLC_REGION="${DLC_REGION:-cn-wulanchabu}"
DLC_ENDPOINT="${DLC_ENDPOINT:-pai-dlc.cn-wulanchabu.aliyuncs.com}"

JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-codevision_sft_v04_eval_t0}"
WORKER_IMAGE="${WORKER_IMAGE:-dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/torcheasyrec:1.1.0-pytorch2.10.0-gpu-py311-cu129-ubuntu22.04}"
CODEVISION_ENV="${CODEVISION_ENV:-${WORKSPACE_ROOT}/envs/codevision_new}"
MODEL_PATH="${MODEL_PATH:-${WORKSPACE_ROOT}/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v04}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${WORKSPACE_ROOT}/Benchmarks}"

PROJECT_NAME="${PROJECT_NAME:-CodeVision}"
EXP_PREFIX="${EXP_PREFIX:-mix200_sft_sp3_v04_eval}"
SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-recipe/codevision/config/sp3.txt}"
TOOL_CFG_TEMPLATE_PATH="${TOOL_CFG_TEMPLATE_PATH:-recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml}"

GROUP1_DATASETS="${GROUP1_DATASETS-vstar chartqa ocrbench countbench}"
GROUP2_DATASETS="${GROUP2_DATASETS-hrbench4k hrbench8k fsc147_val fsc147_test}"
TEMPERATURES="${TEMPERATURES:-0}"

TOOL_DLC_HOST="${TOOL_DLC_HOST:-172.17.0.142}"
TOOL_DLC_REPLICA="${TOOL_DLC_REPLICA:-3}"
TOOL_DLC_BASE_PORT="${TOOL_DLC_BASE_PORT:-18110}"
OCR_BASE_URL="${OCR_BASE_URL:-http://${TOOL_DLC_HOST}:${TOOL_DLC_BASE_PORT}}"
GROUNDEDSAM2_BASE_URL="${GROUNDEDSAM2_BASE_URL:-http://${TOOL_DLC_HOST}:$((TOOL_DLC_BASE_PORT + 1))}"
DEPTH_BASE_URL="${DEPTH_BASE_URL:-http://${TOOL_DLC_HOST}:$((TOOL_DLC_BASE_PORT + 2))}"
COUNTGD_BASE_URL="${COUNTGD_BASE_URL:-http://${TOOL_DLC_HOST}:$((TOOL_DLC_BASE_PORT + 3))}"

NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
INFER_TP_SIZE="${INFER_TP_SIZE:-4}"
VAL_BSZ="${VAL_BSZ:-32}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-1}"
VAL_N_RESP_PER_PROMPT="${VAL_N_RESP_PER_PROMPT:-1}"
MAX_TURNS="${MAX_TURNS:-12}"
ROLLOUT_MAX_TOKENS_PER_TURN="${ROLLOUT_MAX_TOKENS_PER_TURN:-2048}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-}"
MAX_RESP_LEN="${MAX_RESP_LEN:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
SAVE_EVAL_METADATA="${SAVE_EVAL_METADATA:-1}"
SAVE_VAL_GENERATIONS="${SAVE_VAL_GENERATIONS:-0}"
SAVE_FULL_TRAJECTORY_ALL="${SAVE_FULL_TRAJECTORY_ALL:-0}"
DIAGNOSTIC_MAX_PER_BUCKET="${DIAGNOSTIC_MAX_PER_BUCKET:-200}"
DIAGNOSTIC_SAMPLE_SEED="${DIAGNOSTIC_SAMPLE_SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"

ENABLE_LLM_JUDGE="${ENABLE_LLM_JUDGE:-0}"
LLM_JUDGE_BASE_URL="${LLM_JUDGE_BASE_URL:-${OFFLINE_SFT_QWEN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}}"
LLM_JUDGE_MODEL_NAME="${LLM_JUDGE_MODEL_NAME:-${OFFLINE_SFT_QWEN_MODEL:-qwen3.6-plus}}"
LLM_JUDGE_API_KEY="${LLM_JUDGE_API_KEY:-${OFFLINE_SFT_QWEN_API_KEY:-${DASHSCOPE_API_KEY:-${OPENAI_API_KEY:-}}}}"
LLM_JUDGE_TIMEOUT="${LLM_JUDGE_TIMEOUT:-100}"
LLM_JUDGE_MAX_RETRIES="${LLM_JUDGE_MAX_RETRIES:-3}"
LLM_JUDGE_ENABLE_THINKING="${LLM_JUDGE_ENABLE_THINKING:-0}"

shell_quote() {
  printf '%q' "$1"
}

append_env() {
  local name="$1"
  local value="$2"
  TRAIN_COMMAND+=" ${name}=$(shell_quote "${value}")"
}

benchmark_parquet() {
  case "$1" in
    vstar) echo "${VSTAR_PARQUET:-${BENCHMARK_ROOT}/vstar-bench/vstar_codevision_eval.parquet}" ;;
    chartqa) echo "${CHARTQA_PARQUET:-${BENCHMARK_ROOT}/ChartQA/chartqa_codevision_eval.parquet}" ;;
    ocrbench) echo "${OCRBENCH_PARQUET:-${BENCHMARK_ROOT}/OCRBench/ocrbench_codevision_eval.parquet}" ;;
    countbench) echo "${COUNTBENCH_PARQUET:-${BENCHMARK_ROOT}/countbench/countbench_codevision_eval.parquet}" ;;
    hrbench4k) echo "${HRBENCH4K_PARQUET:-${BENCHMARK_ROOT}/HR-Bench/hr_bench_4k_codevision_eval.parquet}" ;;
    hrbench8k) echo "${HRBENCH8K_PARQUET:-${BENCHMARK_ROOT}/HR-Bench/hr_bench_8k_codevision_eval.parquet}" ;;
    fsc147|fsc147_val) echo "${FSC147_VAL_PARQUET:-${BENCHMARK_ROOT}/FSC147/fsc147_val_codevision_eval.parquet}" ;;
    fsc147_test) echo "${FSC147_TEST_PARQUET:-${BENCHMARK_ROOT}/FSC147/fsc147_test_codevision_eval.parquet}" ;;
    mvtoolbench) echo "${MVTOOLBENCH_PARQUET:-${BENCHMARK_ROOT}/MVToolBench/mvtoolbench_codevision_eval.parquet}" ;;
    cvbench) echo "${CVBENCH_PARQUET:-${BENCHMARK_ROOT}/CV-Bench/cvbench_codevision_eval.parquet}" ;;
    pixmo_count) echo "${PIXMO_COUNT_PARQUET:-${BENCHMARK_ROOT}/Pixmo-Count/pixmo_count_codevision_eval.parquet}" ;;
    pixmo_count_lmms) echo "${PIXMO_COUNT_LMMS_PARQUET:-${BENCHMARK_ROOT}/Pixmo-Count-LMMS/pixmo_count_lmms_codevision_eval.parquet}" ;;
    ocrbench_v2) echo "${OCRBENCH_V2_PARQUET:-${BENCHMARK_ROOT}/OCRBench_v2/ocrbench_v2_codevision_eval.parquet}" ;;
    spatialmqa) echo "${SPATIALMQA_PARQUET:-${BENCHMARK_ROOT}/SpatialMQA/spatialmqa_codevision_eval.parquet}" ;;
    countqa) echo "${COUNTQA_PARQUET:-${BENCHMARK_ROOT}/CountQA/countqa_codevision_eval.parquet}" ;;
    arxivqa) echo "${ARXIVQA_PARQUET:-${BENCHMARK_ROOT}/ArxivQA/arxivqa_codevision_eval.parquet}" ;;
    docvqa|docvqa_val) echo "${DOCVQA_PARQUET:-${BENCHMARK_ROOT}/DocVQA/docvqa_val_codevision_eval.parquet}" ;;
    docvqa_test) echo "${DOCVQA_TEST_PARQUET:-${BENCHMARK_ROOT}/DocVQA/docvqa_test_codevision_eval.parquet}" ;;
    infovqa|infovqa_val) echo "${INFOVQA_PARQUET:-${BENCHMARK_ROOT}/InfoVQA/infovqa_val_codevision_eval.parquet}" ;;
    infovqa_test) echo "${INFOVQA_TEST_PARQUET:-${BENCHMARK_ROOT}/InfoVQA/infovqa_test_codevision_eval.parquet}" ;;
    mme_realworld|mmerealworld) echo "${MME_REALWORLD_PARQUET:-${BENCHMARK_ROOT}/MME-RealWorld/mme_realworld_codevision_eval.parquet}" ;;
    mme_realworld_lite|mmerealworld_lite) echo "${MME_REALWORLD_LITE_PARQUET:-${BENCHMARK_ROOT}/MME-RealWorld-Lite/mme_realworld_lite_codevision_eval.parquet}" ;;
    mme_realworld_cn|mmerealworld_cn) echo "${MME_REALWORLD_CN_PARQUET:-${BENCHMARK_ROOT}/MME-RealWorld-CN/mme_realworld_cn_codevision_eval.parquet}" ;;
    realworldqa) echo "${REALWORLDQA_PARQUET:-${BENCHMARK_ROOT}/RealWorldQA/realworldqa_codevision_eval.parquet}" ;;
    mmstar) echo "${MMSTAR_PARQUET:-${BENCHMARK_ROOT}/MMStar/mmstar_codevision_eval.parquet}" ;;
    mmvet) echo "${MMVET_PARQUET:-${BENCHMARK_ROOT}/MMVet/mmvet_codevision_eval.parquet}" ;;
    mmvet_hard) echo "${MMVET_HARD_PARQUET:-${BENCHMARK_ROOT}/MMVet-Hard/mmvet_hard_codevision_eval.parquet}" ;;
    *)
      echo "Unknown dataset '$1'." >&2
      exit 1
      ;;
  esac
}

check_tool_port() {
  local name="$1"
  local url="$2"
  local host_port="${url#http://}"
  local host="${host_port%%:*}"
  local port="${host_port##*:}"
  if ! timeout 2 bash -lc "</dev/tcp/${host}/${port}" >/dev/null 2>&1; then
    echo "Tool service ${name} is not reachable at ${url}." >&2
    echo "Set SKIP_TOOL_PORT_CHECK=1 only if this DSW cannot reach the DLC service but workers can." >&2
    exit 1
  fi
}

preflight_group() {
  local group_name="$1"
  local datasets="$2"
  local dataset
  for dataset in ${datasets//,/ }; do
    [[ -n "${dataset}" ]] || continue
    eval_parquet="$(benchmark_parquet "${dataset}")"
    if [[ ! -f "${eval_parquet}" ]]; then
      echo "Missing eval parquet for ${group_name}/${dataset}: ${eval_parquet}" >&2
      exit 1
    fi
  done
}

if [[ ! -d "${ROOT_DIR}" ]]; then
  echo "Missing ROOT_DIR: ${ROOT_DIR}" >&2
  exit 1
fi

if [[ ! -x "${CODEVISION_ENV}/bin/python" ]]; then
  echo "Missing codevision_new python: ${CODEVISION_ENV}/bin/python" >&2
  exit 1
fi

for required_file in config.json model.safetensors.index.json tokenizer_config.json; do
  if [[ ! -f "${MODEL_PATH}/${required_file}" ]]; then
    echo "Model output is incomplete; missing ${MODEL_PATH}/${required_file}" >&2
    exit 1
  fi
done

if [[ ! -f "${ROOT_DIR}/${SYSTEM_PROMPT_PATH}" ]]; then
  echo "Missing system prompt: ${ROOT_DIR}/${SYSTEM_PROMPT_PATH}" >&2
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/${TOOL_CFG_TEMPLATE_PATH}" ]]; then
  echo "Missing tool config template: ${ROOT_DIR}/${TOOL_CFG_TEMPLATE_PATH}" >&2
  exit 1
fi

preflight_group "group1" "${GROUP1_DATASETS}"
preflight_group "group2" "${GROUP2_DATASETS}"

if [[ "${SKIP_TOOL_PORT_CHECK:-0}" != "1" ]]; then
  check_tool_port "OCR" "${OCR_BASE_URL}"
  check_tool_port "GroundedSAM2" "${GROUNDEDSAM2_BASE_URL}"
  check_tool_port "Depth" "${DEPTH_BASE_URL}"
  check_tool_port "CountGD" "${COUNTGD_BASE_URL}"
fi

if [[ "${ENABLE_LLM_JUDGE}" == "1" || "${ENABLE_LLM_JUDGE,,}" == "true" ]]; then
  if [[ -z "${LLM_JUDGE_BASE_URL}" || -z "${LLM_JUDGE_MODEL_NAME}" || -z "${LLM_JUDGE_API_KEY}" ]]; then
    echo "ENABLE_LLM_JUDGE=1 requires LLM_JUDGE_BASE_URL, LLM_JUDGE_MODEL_NAME, and LLM_JUDGE_API_KEY." >&2
    exit 1
  fi
else
  LLM_JUDGE_BASE_URL=""
  LLM_JUDGE_MODEL_NAME=""
  LLM_JUDGE_API_KEY=""
fi

DLC_GLOBAL_ARGS=()
if [[ "$(basename "${DLC_BIN}")" != "dlc_pai" ]]; then
  DLC_GLOBAL_ARGS=(--region "${DLC_REGION}" --endpoint "${DLC_ENDPOINT}")
  if [[ -n "${DLC_ACCESS_ID:-${ALIBABA_CLOUD_ACCESS_KEY_ID:-}}" ]]; then
    DLC_GLOBAL_ARGS+=(--access_id "${DLC_ACCESS_ID:-${ALIBABA_CLOUD_ACCESS_KEY_ID}}")
  fi
  if [[ -n "${DLC_ACCESS_KEY:-${ALIBABA_CLOUD_ACCESS_KEY_SECRET:-}}" ]]; then
    DLC_GLOBAL_ARGS+=(--access_key "${DLC_ACCESS_KEY:-${ALIBABA_CLOUD_ACCESS_KEY_SECRET}}")
  fi
  if [[ -n "${DLC_SECURITY_TOKEN:-${ALIBABA_CLOUD_SECURITY_TOKEN:-}}" ]]; then
    DLC_GLOBAL_ARGS+=(--security_token "${DLC_SECURITY_TOKEN:-${ALIBABA_CLOUD_SECURITY_TOKEN}}")
  fi
fi

submit_group() {
  local group_name="$1"
  local datasets="$2"
  local job_name="${JOB_NAME_PREFIX}_${group_name}"
  local exp_prefix="${EXP_PREFIX}_${group_name}"

  TRAIN_COMMAND="cd $(shell_quote "${ROOT_DIR}") &&"
  append_env JOB_NAME "${job_name}"
  append_env TRAIN_SCRIPT "recipe/codevision/eval_current_prompt_tool_matrix.sh"
  append_env MODEL_PATH "${MODEL_PATH}"
  append_env RESUME_MODE "${RESUME_MODE:-auto}"
  append_env RESUME_FROM_PATH "${RESUME_FROM_PATH:-null}"
  append_env TOOL_CFG_TEMPLATE_PATH "${TOOL_CFG_TEMPLATE_PATH}"
  append_env SYSTEM_PROMPT_PATH "${SYSTEM_PROMPT_PATH}"
  append_env PROJECT_NAME "${PROJECT_NAME}"
  append_env EXP_PREFIX "${exp_prefix}"
  append_env DATASETS "${datasets}"
  append_env TEMPERATURES "${TEMPERATURES}"
  append_env BENCHMARK_ROOT "${BENCHMARK_ROOT}"
  append_env OCR_BASE_URL "${OCR_BASE_URL}"
  append_env GROUNDEDSAM2_BASE_URL "${GROUNDEDSAM2_BASE_URL}"
  append_env DEPTH_BASE_URL "${DEPTH_BASE_URL}"
  append_env COUNTGD_BASE_URL "${COUNTGD_BASE_URL}"
  append_env CODEVISION_ENV "${CODEVISION_ENV}"
  append_env DLC_ENTRYPOINT_DEBUG "${DLC_ENTRYPOINT_DEBUG:-1}"
  append_env RAY_NODE_CHECK_TIMEOUT_SECONDS "${RAY_NODE_CHECK_TIMEOUT_SECONDS:-20}"
  append_env TOOL_PREFLIGHT_CHECK "${TOOL_PREFLIGHT_CHECK:-1}"
  append_env NGPUS_PER_NODE "${NGPUS_PER_NODE}"
  append_env INFER_TP_SIZE "${INFER_TP_SIZE}"
  append_env TRAIN_BSZ "${TRAIN_BSZ:-64}"
  append_env TRAIN_MINI_BSZ "${TRAIN_MINI_BSZ:-32}"
  append_env VAL_BSZ "${VAL_BSZ}"
  append_env N_RESP_PER_PROMPT "${N_RESP_PER_PROMPT}"
  append_env VAL_N_RESP_PER_PROMPT "${VAL_N_RESP_PER_PROMPT}"
  append_env MAX_TURNS "${MAX_TURNS}"
  append_env ROLLOUT_MAX_TOKENS_PER_TURN "${ROLLOUT_MAX_TOKENS_PER_TURN}"
  append_env MAX_PROMPT_LEN "${MAX_PROMPT_LEN}"
  append_env MAX_RESP_LEN "${MAX_RESP_LEN}"
  append_env GPU_MEMORY_UTILIZATION "${GPU_MEMORY_UTILIZATION}"
  append_env MAX_NUM_SEQS "${MAX_NUM_SEQS}"
  append_env ROLLOUT_AGENT_NUM_WORKERS "${ROLLOUT_AGENT_NUM_WORKERS}"
  append_env DISABLE_TOOLS "${DISABLE_TOOLS:-0}"
  append_env SAVE_EVAL_METADATA "${SAVE_EVAL_METADATA}"
  append_env SAVE_VAL_GENERATIONS "${SAVE_VAL_GENERATIONS}"
  append_env SAVE_FULL_TRAJECTORY_ALL "${SAVE_FULL_TRAJECTORY_ALL}"
  append_env DIAGNOSTIC_MAX_PER_BUCKET "${DIAGNOSTIC_MAX_PER_BUCKET}"
  append_env DIAGNOSTIC_SAMPLE_SEED "${DIAGNOSTIC_SAMPLE_SEED}"
  append_env LLM_JUDGE_BASE_URL "${LLM_JUDGE_BASE_URL}"
  append_env LLM_JUDGE_MODEL_NAME "${LLM_JUDGE_MODEL_NAME}"
  append_env LLM_JUDGE_TIMEOUT "${LLM_JUDGE_TIMEOUT}"
  append_env LLM_JUDGE_MAX_RETRIES "${LLM_JUDGE_MAX_RETRIES}"
  append_env LLM_JUDGE_ENABLE_THINKING "${LLM_JUDGE_ENABLE_THINKING}"
  TRAIN_COMMAND+=" bash scripts/dlc_ray_direct_entrypoint.sh"

  DLC_ENV_ARGS=()
  if [[ -n "${LLM_JUDGE_API_KEY}" ]]; then
    DLC_ENV_ARGS+=(--envs "LLM_JUDGE_API_KEY=${LLM_JUDGE_API_KEY}")
  fi

  echo
  echo "Submitting ${job_name}"
  echo "DATASETS=${datasets}"
  echo "TEMPERATURES=${TEMPERATURES}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "CODEVISION_ENV=${CODEVISION_ENV}"
  echo "TOOL_DLC_REPLICA=${TOOL_DLC_REPLICA}"
  echo "OCR_BASE_URL=${OCR_BASE_URL}"
  echo "GROUNDEDSAM2_BASE_URL=${GROUNDEDSAM2_BASE_URL}"
  echo "DEPTH_BASE_URL=${DEPTH_BASE_URL}"
  echo "COUNTGD_BASE_URL=${COUNTGD_BASE_URL}"
  echo "NGPUS_PER_NODE=${NGPUS_PER_NODE}"
  echo "TRAIN_BSZ=${TRAIN_BSZ:-64}"
  echo "TRAIN_MINI_BSZ=${TRAIN_MINI_BSZ:-32}"
  echo "N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT}"
  echo "VAL_N_RESP_PER_PROMPT=${VAL_N_RESP_PER_PROMPT}"
  echo "MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-<default>}"
  echo "MAX_RESP_LEN=${MAX_RESP_LEN:-<default>}"
  echo "ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS}"
  echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
  echo "DISABLE_TOOLS=${DISABLE_TOOLS:-0}"
  echo "ENABLE_LLM_JUDGE=${ENABLE_LLM_JUDGE}"
  echo "LLM_JUDGE_API_KEY_SET=$([[ -n "${LLM_JUDGE_API_KEY}" ]] && echo yes || echo no)"

  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
    echo "DRY_RUN=1, not submitting."
    dry_run_command="${DLC_BIN} submit pytorchjob ${DLC_GLOBAL_ARGS[*]} --name=${job_name} --command=$(shell_quote "${TRAIN_COMMAND}") ..."
    printf '%s\n' "${dry_run_command}" | sed \
      -e 's/\(LLM_JUDGE_API_KEY=\)[^\\ ]*/\1<redacted>/g' \
      -e 's/\(access_key[=:]\)[^,} ]*/\1<redacted>/Ig' \
      -e 's/\(AccessKeySecret[=:]\)[^,} ]*/\1<redacted>/g'
    return 0
  fi

  "${DLC_BIN}" submit pytorchjob \
    "${DLC_GLOBAL_ARGS[@]}" \
    --name="${job_name}" \
    --command="${TRAIN_COMMAND}" \
    "${DLC_ENV_ARGS[@]}" \
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
    --worker_gpu="${WORKER_GPU:-${NGPUS_PER_NODE}}" \
    2>&1 | sed \
      -e 's/\(LLM_JUDGE_API_KEY[=:]\)[^,} ]*/\1<redacted>/g' \
      -e 's/"LLM_JUDGE_API_KEY"[[:space:]]*:[[:space:]]*"[^"]*"/"LLM_JUDGE_API_KEY":"<redacted>"/g' \
      -e 's/\(access_key[=:]\)[^,} ]*/\1<redacted>/Ig' \
      -e 's/\(AccessKeySecret[=:]\)[^,} ]*/\1<redacted>/g'
}

echo "Submitting split v04 eval with external DLC tool replica"
echo "GROUP1_DATASETS=${GROUP1_DATASETS}"
echo "GROUP2_DATASETS=${GROUP2_DATASETS}"
echo "TEMPERATURES=${TEMPERATURES}"
echo "DLC_BIN=${DLC_BIN}"

if [[ -n "${GROUP1_DATASETS//[[:space:],]/}" ]]; then
  submit_group "g1" "${GROUP1_DATASETS}"
else
  echo "GROUP1_DATASETS is empty; skip g1."
fi

if [[ -n "${GROUP2_DATASETS//[[:space:],]/}" ]]; then
  submit_group "g2" "${GROUP2_DATASETS}"
else
  echo "GROUP2_DATASETS is empty; skip g2."
fi
