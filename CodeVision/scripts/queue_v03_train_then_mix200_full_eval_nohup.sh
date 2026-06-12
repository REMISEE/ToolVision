#!/usr/bin/env bash
# Queue v03 mix200 SFT after current GPU work finishes, then run full tool eval.
#
# Run from anywhere:
#   nohup bash /mnt/cpfs/delinmao/ToolVision/CodeVision/scripts/queue_v03_train_then_mix200_full_eval_nohup.sh \
#     > /mnt/cpfs/delinmao/logs/queue_v03_train_then_mix200_full_eval.log 2>&1 &

set -euo pipefail
trap '' HUP

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/cpfs/delinmao}"
LF_DIR="${LF_DIR:-${WORKSPACE_ROOT}/CodeVision/LLaMA-Factory}"
TV_DIR="${TV_DIR:-${WORKSPACE_ROOT}/ToolVision/CodeVision}"
LOG_DIR="${LOG_DIR:-${WORKSPACE_ROOT}/logs}"

TRAIN_CONFIG="${TRAIN_CONFIG:-examples/train_full/qwen3vl_sft_mix200_simple_notool_sp3_v03_finalonly.yaml}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-sft_mix200_sp3_v03_finalonly}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${LF_DIR}/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03}"
TRAIN_GPUS="${TRAIN_GPUS:-8}"
TRAIN_GPU_CANDIDATES="${TRAIN_GPU_CANDIDATES:-0,1,2,3,4,5,6,7}"

EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-${TRAIN_OUTPUT_DIR}}"
EVAL_GPUS="${EVAL_GPUS:-2,3,4,5,6}"
EVAL_EXP_PREFIX="${EVAL_EXP_PREFIX:-mix200_sft_sp3_v03_full}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-vstar,chartqa,ocrbench,countbench,hrbench4k,hrbench8k,fsc147_val,fsc147_test,mvtoolbench,cvbench,pixmo_count_lmms,countqa,spatialmqa,ocrbench_v2}"
EVAL_LOG="${EVAL_LOG:-${LOG_DIR}/eval_${EVAL_EXP_PREFIX}.log}"

mkdir -p "${LOG_DIR}"

echo "=== Queue v03 train -> full eval ==="
echo "time=$(date '+%F %T')"
echo "LF_DIR=${LF_DIR}"
echo "TV_DIR=${TV_DIR}"
echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "TRAIN_RUN_NAME=${TRAIN_RUN_NAME}"
echo "TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}"
echo "TRAIN_GPUS=${TRAIN_GPUS}"
echo "TRAIN_GPU_CANDIDATES=${TRAIN_GPU_CANDIDATES}"
echo "EVAL_MODEL_PATH=${EVAL_MODEL_PATH}"
echo "EVAL_GPUS=${EVAL_GPUS}"
echo "EVAL_EXP_PREFIX=${EVAL_EXP_PREFIX}"
echo "EVAL_BENCHMARKS=${EVAL_BENCHMARKS}"
echo "EVAL_LOG=${EVAL_LOG}"

if [[ ! -d "${LF_DIR}" ]]; then
  echo "Missing LLaMA-Factory dir: ${LF_DIR}" >&2
  exit 1
fi
if [[ ! -d "${TV_DIR}" ]]; then
  echo "Missing ToolVision CodeVision dir: ${TV_DIR}" >&2
  exit 1
fi
if [[ ! -f "${LF_DIR}/${TRAIN_CONFIG}" ]]; then
  echo "Missing training config: ${LF_DIR}/${TRAIN_CONFIG}" >&2
  exit 1
fi

if [[ -z "${LLM_JUDGE_API_KEY:-${OPENAI_API_KEY:-}}" ]]; then
  echo "Missing LLM_JUDGE_API_KEY or OPENAI_API_KEY; full eval uses LLM judge and should not start without it." >&2
  exit 1
fi

echo
echo "=== Start queued 8-GPU training ==="
cd "${LF_DIR}"
CONFIG_PATH="${TRAIN_CONFIG}" \
RUN_NAME="${TRAIN_RUN_NAME}" \
OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
NUM_GPUS="${TRAIN_GPUS}" \
GPU_CANDIDATES="${TRAIN_GPU_CANDIDATES}" \
MAX_USED_MEM_MB="${TRAIN_MAX_USED_MEM_MB:-1000}" \
MAX_GPU_UTIL="${TRAIN_MAX_GPU_UTIL:-5}" \
WAIT_INTERVAL_S="${TRAIN_WAIT_INTERVAL_S:-60}" \
bash examples/train_full/run_sft_mix200_simple_notool_finalonly_wait_nohup.sh

echo
echo "=== Training completed and model verified ==="
echo "model=${TRAIN_OUTPUT_DIR}"

for required_file in config.json model.safetensors.index.json tokenizer_config.json; do
  if [[ ! -f "${TRAIN_OUTPUT_DIR}/${required_file}" ]]; then
    echo "Training output is incomplete; missing ${TRAIN_OUTPUT_DIR}/${required_file}" >&2
    exit 1
  fi
done

echo
echo "=== Start full eval for v03 mix200 ==="
cd "${TV_DIR}"
export LLM_JUDGE_BASE_URL="${LLM_JUDGE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export LLM_JUDGE_MODEL_NAME="${LLM_JUDGE_MODEL_NAME:-qwen3.6-plus}"
export LLM_JUDGE_TIMEOUT="${LLM_JUDGE_TIMEOUT:-100}"
export LLM_JUDGE_MAX_RETRIES="${LLM_JUDGE_MAX_RETRIES:-3}"
export LLM_JUDGE_ENABLE_THINKING="${LLM_JUDGE_ENABLE_THINKING:-0}"
export LLM_JUDGE_API_KEY="${LLM_JUDGE_API_KEY:-${OPENAI_API_KEY:-}}"

CODEVISION_ENV="${CODEVISION_ENV:-${WORKSPACE_ROOT}/envs/codevision}" \
GPU_CANDIDATES="${EVAL_GPUS}" \
RAY_INIT_NUM_CPUS="${RAY_INIT_NUM_CPUS:-40}" \
MODEL_PATH="${EVAL_MODEL_PATH}" \
EXP_PREFIX="${EVAL_EXP_PREFIX}" \
BENCHMARKS="${EVAL_BENCHMARKS}" \
SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-recipe/codevision/config/sp3.txt}" \
TOOL_CFG_TEMPLATE_PATH="${TOOL_CFG_TEMPLATE_PATH:-recipe/codevision/config/code_image_tool_config_v03.yaml}" \
bash scripts/run_tools_eval_all_wait_5gpu_nohup.sh > "${EVAL_LOG}" 2>&1

echo "=== Full eval finished ==="
echo "eval_log=${EVAL_LOG}"
