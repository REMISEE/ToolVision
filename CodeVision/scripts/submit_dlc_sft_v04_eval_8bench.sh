#!/usr/bin/env bash
set -euo pipefail

# Submit a one-worker, 5-GPU DLC job for full v04 SFT tool eval.
#
# The worker starts the external tool services itself:
#   - 4 GPUs: Qwen3-VL / vLLM tensor parallel eval
#   - 1 GPU : OCR, GroundedSAM2, Depth, CountGD services
#
# Defaults match the v04 SFT prompt/tool setup: sp3 + sftclean.

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/cpfs/delinmao}"
DLC_BIN="${DLC_BIN:-$([[ -x /mnt/cpfs/delinmao/bin/dlc_pai ]] && echo /mnt/cpfs/delinmao/bin/dlc_pai || command -v dlc_pai 2>/dev/null || command -v dlc 2>/dev/null || echo /etc/dsw/runtime/export_bin/dlc)}"
DLC_REGION="${DLC_REGION:-cn-wulanchabu}"
DLC_ENDPOINT="${DLC_ENDPOINT:-pai-dlc.cn-wulanchabu.aliyuncs.com}"

JOB_NAME="${JOB_NAME:-codevision_sft_v04_eval_8bench}"
WORKER_IMAGE="${WORKER_IMAGE:-dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/torcheasyrec:1.1.0-pytorch2.10.0-gpu-py311-cu129-ubuntu22.04}"
CODEVISION_ENV="${CODEVISION_ENV:-${WORKSPACE_ROOT}/envs/codevision_new}"
MODEL_PATH="${MODEL_PATH:-${WORKSPACE_ROOT}/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v04}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${WORKSPACE_ROOT}/Benchmarks}"

BENCHMARKS="${BENCHMARKS:-vstar,chartqa,ocrbench,countbench,hrbench4k,hrbench8k,fsc147_val,fsc147_test}"
EXP_PREFIX="${EXP_PREFIX:-mix200_sft_sp3_v04}"
SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-recipe/codevision/config/sp3.txt}"
TOOL_CFG_TEMPLATE_PATH="${TOOL_CFG_TEMPLATE_PATH:-recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml}"

NUM_MODEL_GPUS="${NUM_MODEL_GPUS:-4}"
NUM_TOOL_GPUS="${NUM_TOOL_GPUS:-1}"
WORKER_GPU="${WORKER_GPU:-$((NUM_MODEL_GPUS + NUM_TOOL_GPUS))}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0,1,2,3,4}"

VAL_TEMPERATURE="${VAL_TEMPERATURE:-0}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-False}"
VAL_TOP_P="${VAL_TOP_P:-1.0}"
VAL_BSZ="${VAL_BSZ:-32}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
FSC147_N_RESP_PER_PROMPT="${FSC147_N_RESP_PER_PROMPT:-1}"
MAX_TURNS="${MAX_TURNS:-12}"
ROLLOUT_MAX_TOKENS_PER_TURN="${ROLLOUT_MAX_TOKENS_PER_TURN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
RAY_INIT_NUM_CPUS="${RAY_INIT_NUM_CPUS:-40}"

SAVE_EVAL_METADATA="${SAVE_EVAL_METADATA:-1}"
SAVE_VAL_GENERATIONS="${SAVE_VAL_GENERATIONS:-0}"
SAVE_FULL_TRAJECTORY_ALL="${SAVE_FULL_TRAJECTORY_ALL:-0}"
DIAGNOSTIC_MAX_PER_BUCKET="${DIAGNOSTIC_MAX_PER_BUCKET:-200}"
DIAGNOSTIC_SAMPLE_SEED="${DIAGNOSTIC_SAMPLE_SEED:-42}"

ENABLE_LLM_JUDGE="${ENABLE_LLM_JUDGE:-1}"
LLM_JUDGE_BASE_URL="${LLM_JUDGE_BASE_URL:-${OFFLINE_SFT_QWEN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}}"
LLM_JUDGE_MODEL_NAME="${LLM_JUDGE_MODEL_NAME:-${OFFLINE_SFT_QWEN_MODEL:-qwen3.6-plus}}"
LLM_JUDGE_API_KEY="${LLM_JUDGE_API_KEY:-${OFFLINE_SFT_QWEN_API_KEY:-${DASHSCOPE_API_KEY:-${OPENAI_API_KEY:-}}}}"
LLM_JUDGE_TIMEOUT="${LLM_JUDGE_TIMEOUT:-100}"
LLM_JUDGE_MAX_RETRIES="${LLM_JUDGE_MAX_RETRIES:-3}"
LLM_JUDGE_ENABLE_THINKING="${LLM_JUDGE_ENABLE_THINKING:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/outputs/dlc_sft_v04_eval_8bench/${JOB_NAME}_${TS}}"
LOCK_DIR="${LOCK_DIR:-${RUN_DIR}/launcher.lock}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/${JOB_NAME}_${TS}_ray}"
DRY_RUN_VALUE="${DRY_RUN:-0}"

shell_quote() {
  printf '%q' "$1"
}

append_export() {
  local name="$1"
  local value="$2"
  TRAIN_COMMAND+=" export ${name}=$(shell_quote "${value}");"
}

benchmark_parquet() {
  case "$1" in
    vstar)
      echo "${VSTAR_PARQUET:-${BENCHMARK_ROOT}/vstar-bench/vstar_codevision_eval.parquet}"
      ;;
    chartqa)
      echo "${CHARTQA_PARQUET:-${BENCHMARK_ROOT}/ChartQA/chartqa_codevision_eval.parquet}"
      ;;
    ocrbench)
      echo "${OCRBENCH_PARQUET:-${BENCHMARK_ROOT}/OCRBench/ocrbench_codevision_eval.parquet}"
      ;;
    countbench)
      echo "${COUNTBENCH_PARQUET:-${BENCHMARK_ROOT}/countbench/countbench_codevision_eval.parquet}"
      ;;
    hrbench4k)
      echo "${HRBENCH4K_PARQUET:-${BENCHMARK_ROOT}/HR-Bench/hr_bench_4k_codevision_eval.parquet}"
      ;;
    hrbench8k)
      echo "${HRBENCH8K_PARQUET:-${BENCHMARK_ROOT}/HR-Bench/hr_bench_8k_codevision_eval.parquet}"
      ;;
    fsc147_val)
      echo "${FSC147_VAL_PARQUET:-${BENCHMARK_ROOT}/FSC147/fsc147_val_codevision_eval.parquet}"
      ;;
    fsc147_test)
      echo "${FSC147_TEST_PARQUET:-${BENCHMARK_ROOT}/FSC147/fsc147_test_codevision_eval.parquet}"
      ;;
    mvtoolbench)
      echo "${MVTOOLBENCH_PARQUET:-${BENCHMARK_ROOT}/MVToolBench/mvtoolbench_codevision_eval.parquet}"
      ;;
    cvbench)
      echo "${CVBENCH_PARQUET:-${BENCHMARK_ROOT}/CV-Bench/cvbench_codevision_eval.parquet}"
      ;;
    pixmo_count)
      echo "${PIXMO_COUNT_PARQUET:-${BENCHMARK_ROOT}/Pixmo-Count/pixmo_count_codevision_eval.parquet}"
      ;;
    pixmo_count_lmms)
      echo "${PIXMO_COUNT_LMMS_PARQUET:-${BENCHMARK_ROOT}/Pixmo-Count-LMMS/pixmo_count_lmms_codevision_eval.parquet}"
      ;;
    ocrbench_v2)
      echo "${OCRBENCH_V2_PARQUET:-${BENCHMARK_ROOT}/OCRBench_v2/ocrbench_v2_codevision_eval.parquet}"
      ;;
    spatialmqa)
      echo "${SPATIALMQA_PARQUET:-${BENCHMARK_ROOT}/SpatialMQA/spatialmqa_codevision_eval.parquet}"
      ;;
    countqa)
      echo "${COUNTQA_PARQUET:-${BENCHMARK_ROOT}/CountQA/countqa_codevision_eval.parquet}"
      ;;
    *)
      case "$1" in
        docvqa|docvqa_val)
          echo "${DOCVQA_PARQUET:-${BENCHMARK_ROOT}/DocVQA/docvqa_val_codevision_eval.parquet}"
          ;;
        docvqa_test)
          echo "${DOCVQA_TEST_PARQUET:-${BENCHMARK_ROOT}/DocVQA/docvqa_test_codevision_eval.parquet}"
          ;;
        infovqa|infovqa_val)
          echo "${INFOVQA_PARQUET:-${BENCHMARK_ROOT}/InfoVQA/infovqa_val_codevision_eval.parquet}"
          ;;
        infovqa_test)
          echo "${INFOVQA_TEST_PARQUET:-${BENCHMARK_ROOT}/InfoVQA/infovqa_test_codevision_eval.parquet}"
          ;;
        mme_realworld|mmerealworld)
          echo "${MME_REALWORLD_PARQUET:-${BENCHMARK_ROOT}/MME-RealWorld/mme_realworld_codevision_eval.parquet}"
          ;;
        mme_realworld_lite|mmerealworld_lite)
          echo "${MME_REALWORLD_LITE_PARQUET:-${BENCHMARK_ROOT}/MME-RealWorld-Lite/mme_realworld_lite_codevision_eval.parquet}"
          ;;
        mme_realworld_cn|mmerealworld_cn)
          echo "${MME_REALWORLD_CN_PARQUET:-${BENCHMARK_ROOT}/MME-RealWorld-CN/mme_realworld_cn_codevision_eval.parquet}"
          ;;
        realworldqa)
          echo "${REALWORLDQA_PARQUET:-${BENCHMARK_ROOT}/RealWorldQA/realworldqa_codevision_eval.parquet}"
          ;;
        mmstar)
          echo "${MMSTAR_PARQUET:-${BENCHMARK_ROOT}/MMStar/mmstar_codevision_eval.parquet}"
          ;;
        mmvet)
          echo "${MMVET_PARQUET:-${BENCHMARK_ROOT}/MMVet/mmvet_codevision_eval.parquet}"
          ;;
        mmvet_hard)
          echo "${MMVET_HARD_PARQUET:-${BENCHMARK_ROOT}/MMVet-Hard/mmvet_hard_codevision_eval.parquet}"
          ;;
        *)
      echo "Unknown benchmark '$1'." >&2
      exit 1
      ;;
      esac
      ;;
  esac
}

if [[ -z "${WORKER_IMAGE}" || "${WORKER_IMAGE}" == *"你的"* || "${WORKER_IMAGE}" == *"TODO"* ]]; then
  echo "WORKER_IMAGE must be a real DLC image URI, got: ${WORKER_IMAGE}" >&2
  exit 1
fi

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

IFS=',' read -r -a BENCH_ARRAY <<< "${BENCHMARKS}"
NORMALIZED_BENCHMARKS=()
for bench in "${BENCH_ARRAY[@]}"; do
  bench="${bench//[[:space:]]/}"
  [[ -n "${bench}" ]] || continue
  eval_parquet="$(benchmark_parquet "${bench}")"
  if [[ ! -f "${eval_parquet}" ]]; then
    echo "Missing eval parquet for ${bench}: ${eval_parquet}" >&2
    exit 1
  fi
  NORMALIZED_BENCHMARKS+=("${bench}")
done
BENCHMARKS="$(IFS=,; echo "${NORMALIZED_BENCHMARKS[*]}")"

if [[ "${ENABLE_LLM_JUDGE}" == "1" || "${ENABLE_LLM_JUDGE,,}" == "true" ]]; then
  if [[ -z "${LLM_JUDGE_BASE_URL}" || -z "${LLM_JUDGE_MODEL_NAME}" || -z "${LLM_JUDGE_API_KEY}" ]]; then
    echo "ENABLE_LLM_JUDGE=1 requires LLM_JUDGE_BASE_URL, LLM_JUDGE_MODEL_NAME, and LLM_JUDGE_API_KEY." >&2
    echo "Export LLM_JUDGE_API_KEY or OPENAI_API_KEY before submitting, or set ENABLE_LLM_JUDGE=0." >&2
    exit 1
  fi
else
  LLM_JUDGE_BASE_URL=""
  LLM_JUDGE_MODEL_NAME=""
  LLM_JUDGE_API_KEY=""
fi

if [[ ! -x "${DLC_BIN}" ]] && ! command -v "${DLC_BIN}" >/dev/null 2>&1; then
  echo "DLC binary not found: ${DLC_BIN}" >&2
  exit 1
fi

TRAIN_COMMAND="cd $(shell_quote "${ROOT_DIR}") &&"
append_export WORKSPACE_ROOT "${WORKSPACE_ROOT}"
append_export ROOT_DIR "${ROOT_DIR}"
append_export CODEVISION_ENV "${CODEVISION_ENV}"
append_export MODEL_PATH "${MODEL_PATH}"
append_export BENCHMARK_ROOT "${BENCHMARK_ROOT}"
append_export BENCHMARKS "${BENCHMARKS}"
append_export EXP_PREFIX "${EXP_PREFIX}"
append_export SYSTEM_PROMPT_PATH "${SYSTEM_PROMPT_PATH}"
append_export TOOL_CFG_TEMPLATE_PATH "${TOOL_CFG_TEMPLATE_PATH}"
append_export NUM_MODEL_GPUS "${NUM_MODEL_GPUS}"
append_export NUM_TOOL_GPUS "${NUM_TOOL_GPUS}"
append_export GPU_CANDIDATES "${GPU_CANDIDATES}"
append_export WAIT_FOR_MODEL "${WAIT_FOR_MODEL:-1}"
append_export MAX_MODEL_WAIT_MINUTES "${MAX_MODEL_WAIT_MINUTES:-10}"
append_export MAX_GPU_WAIT_MINUTES "${MAX_GPU_WAIT_MINUTES:-20}"
append_export RAY_INIT_NUM_CPUS "${RAY_INIT_NUM_CPUS}"
append_export RUN_DIR "${RUN_DIR}"
append_export LOCK_DIR "${LOCK_DIR}"
append_export RAY_TMPDIR "${RAY_TMPDIR}"
append_export VAL_TEMPERATURE "${VAL_TEMPERATURE}"
append_export VAL_DO_SAMPLE "${VAL_DO_SAMPLE}"
append_export VAL_TOP_P "${VAL_TOP_P}"
append_export VAL_BSZ "${VAL_BSZ}"
append_export N_RESP_PER_PROMPT "${N_RESP_PER_PROMPT}"
append_export FSC147_N_RESP_PER_PROMPT "${FSC147_N_RESP_PER_PROMPT}"
append_export MAX_TURNS "${MAX_TURNS}"
append_export ROLLOUT_MAX_TOKENS_PER_TURN "${ROLLOUT_MAX_TOKENS_PER_TURN}"
append_export GPU_MEMORY_UTILIZATION "${GPU_MEMORY_UTILIZATION}"
append_export MAX_NUM_SEQS "${MAX_NUM_SEQS}"
append_export ROLLOUT_AGENT_NUM_WORKERS "${ROLLOUT_AGENT_NUM_WORKERS}"
append_export SAVE_EVAL_METADATA "${SAVE_EVAL_METADATA}"
append_export SAVE_VAL_GENERATIONS "${SAVE_VAL_GENERATIONS}"
append_export SAVE_FULL_TRAJECTORY_ALL "${SAVE_FULL_TRAJECTORY_ALL}"
append_export DIAGNOSTIC_MAX_PER_BUCKET "${DIAGNOSTIC_MAX_PER_BUCKET}"
append_export DIAGNOSTIC_SAMPLE_SEED "${DIAGNOSTIC_SAMPLE_SEED}"
append_export LLM_JUDGE_BASE_URL "${LLM_JUDGE_BASE_URL}"
append_export LLM_JUDGE_MODEL_NAME "${LLM_JUDGE_MODEL_NAME}"
append_export LLM_JUDGE_TIMEOUT "${LLM_JUDGE_TIMEOUT}"
append_export LLM_JUDGE_MAX_RETRIES "${LLM_JUDGE_MAX_RETRIES}"
append_export LLM_JUDGE_ENABLE_THINKING "${LLM_JUDGE_ENABLE_THINKING}"
TRAIN_COMMAND+=" bash scripts/run_tools_eval_all_wait_5gpu_nohup.sh"

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

DLC_ENV_ARGS=()
if [[ -n "${LLM_JUDGE_API_KEY}" ]]; then
  DLC_ENV_ARGS+=(--envs "LLM_JUDGE_API_KEY=${LLM_JUDGE_API_KEY}")
fi

echo "Submitting ${JOB_NAME}"
echo "ROOT_DIR=${ROOT_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "CODEVISION_ENV=${CODEVISION_ENV}"
echo "BENCHMARKS=${BENCHMARKS}"
echo "EXP_PREFIX=${EXP_PREFIX}"
echo "SYSTEM_PROMPT_PATH=${SYSTEM_PROMPT_PATH}"
echo "TOOL_CFG_TEMPLATE_PATH=${TOOL_CFG_TEMPLATE_PATH}"
echo "RUN_DIR=${RUN_DIR}"
echo "WORKER_IMAGE=${WORKER_IMAGE}"
echo "WORKER_GPU=${WORKER_GPU} (${NUM_MODEL_GPUS} model + ${NUM_TOOL_GPUS} tools)"
echo "VAL_TEMPERATURE=${VAL_TEMPERATURE}"
echo "VAL_DO_SAMPLE=${VAL_DO_SAMPLE}"
echo "VAL_TOP_P=${VAL_TOP_P}"
echo "N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT}"
echo "FSC147_N_RESP_PER_PROMPT=${FSC147_N_RESP_PER_PROMPT}"
echo "SAVE_EVAL_METADATA=${SAVE_EVAL_METADATA}"
echo "SAVE_VAL_GENERATIONS=${SAVE_VAL_GENERATIONS}"
echo "SAVE_FULL_TRAJECTORY_ALL=${SAVE_FULL_TRAJECTORY_ALL}"
echo "ENABLE_LLM_JUDGE=${ENABLE_LLM_JUDGE}"
echo "LLM_JUDGE_BASE_URL=$([[ -n "${LLM_JUDGE_BASE_URL}" ]] && echo "${LLM_JUDGE_BASE_URL}" || echo '<disabled>')"
echo "LLM_JUDGE_MODEL_NAME=$([[ -n "${LLM_JUDGE_MODEL_NAME}" ]] && echo "${LLM_JUDGE_MODEL_NAME}" || echo '<disabled>')"
echo "LLM_JUDGE_API_KEY_SET=$([[ -n "${LLM_JUDGE_API_KEY}" ]] && echo yes || echo no)"
echo "DLC_BIN=${DLC_BIN}"

if [[ "${DRY_RUN_VALUE}" == "1" || "${DRY_RUN_VALUE,,}" == "true" ]]; then
  echo "DRY_RUN=1, not submitting."
  dry_run_command="${DLC_BIN} submit pytorchjob ${DLC_GLOBAL_ARGS[*]} --name=${JOB_NAME} --command=$(shell_quote "${TRAIN_COMMAND}") ..."
  printf '%s\n' "${dry_run_command}" | sed \
    -e 's/\(LLM_JUDGE_API_KEY=\)[^\\ ]*/\1<redacted>/g' \
    -e 's/"LLM_JUDGE_API_KEY"[[:space:]]*:[[:space:]]*"[^"]*"/"LLM_JUDGE_API_KEY":"<redacted>"/g' \
    -e 's/\(OPENAI_API_KEY=\)[^\\ ]*/\1<redacted>/g' \
    -e 's/\(DASHSCOPE_API_KEY=\)[^\\ ]*/\1<redacted>/g' \
    -e 's/\(OFFLINE_SFT_QWEN_API_KEY=\)[^\\ ]*/\1<redacted>/g' \
    -e 's/\(access_key[=:]\)[^,} ]*/\1<redacted>/Ig' \
    -e 's/\(AccessKeySecret[=:]\)[^,} ]*/\1<redacted>/g'
  exit 0
fi

"${DLC_BIN}" submit pytorchjob \
  "${DLC_GLOBAL_ARGS[@]}" \
  --name="${JOB_NAME}" \
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
  --worker_gpu="${WORKER_GPU}" \
  2>&1 | sed \
    -e 's/\(LLM_JUDGE_API_KEY[=:]\)[^,} ]*/\1<redacted>/g' \
    -e 's/"LLM_JUDGE_API_KEY"[[:space:]]*:[[:space:]]*"[^"]*"/"LLM_JUDGE_API_KEY":"<redacted>"/g' \
    -e 's/\(OPENAI_API_KEY[=:]\)[^,} ]*/\1<redacted>/g' \
    -e 's/\(DASHSCOPE_API_KEY[=:]\)[^,} ]*/\1<redacted>/g' \
    -e 's/\(OFFLINE_SFT_QWEN_API_KEY[=:]\)[^,} ]*/\1<redacted>/g' \
    -e 's/\(access_key[=:]\)[^,} ]*/\1<redacted>/Ig' \
    -e 's/\(AccessKeySecret[=:]\)[^,} ]*/\1<redacted>/g'
