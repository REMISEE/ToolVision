#!/usr/bin/env bash
set -euo pipefail

# Tool-enabled rollout8 for pass@16-derived MUT candidates.
# Required:
#   EVAL_PARQUET=/path/to/toolvision_eval.parquet
#   EXP_NAME=my_experiment_name

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

if [[ -z "${EVAL_PARQUET:-}" ]]; then
  echo "EVAL_PARQUET is required." >&2
  exit 1
fi
if [[ -z "${EXP_NAME:-}" ]]; then
  echo "EXP_NAME is required." >&2
  exit 1
fi

if [[ ! -f "${EVAL_PARQUET}" ]]; then
  echo "Missing EVAL_PARQUET: ${EVAL_PARQUET}" >&2
  exit 1
fi

if [[ -z "${OCR_BASE_URL:-}" || -z "${GROUNDEDSAM2_BASE_URL:-}" || -z "${DEPTH_BASE_URL:-}" || -z "${COUNTGD_BASE_URL:-}" ]]; then
  eval "$("${PROJECT_DIR}/scripts/dsw_tool_urls.sh")"
fi

export MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03}"
export SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-recipe/codevision/config/sp3.txt}"
export TOOL_CFG_TEMPLATE_PATH="${TOOL_CFG_TEMPLATE_PATH:-recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml}"

export NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
export INFER_TP_SIZE="${INFER_TP_SIZE:-4}"
export VAL_BSZ="${VAL_BSZ:-32}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-1}"
export VAL_N_RESP_PER_PROMPT="${VAL_N_RESP_PER_PROMPT:-8}"
export VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.7}"
export VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-True}"
export VAL_TOP_P="${VAL_TOP_P:-0.95}"

export MAX_TURNS="${MAX_TURNS:-12}"
export ROLLOUT_MAX_TOKENS_PER_TURN="${ROLLOUT_MAX_TOKENS_PER_TURN:-2048}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"

export SAVE_EVAL_METADATA="${SAVE_EVAL_METADATA:-1}"
export SAVE_VAL_GENERATIONS="${SAVE_VAL_GENERATIONS:-1}"
export SAVE_FULL_TRAJECTORY_ALL="${SAVE_FULL_TRAJECTORY_ALL:-1}"
export DIAGNOSTIC_MAX_PER_BUCKET="${DIAGNOSTIC_MAX_PER_BUCKET:-1000000}"
export STREAM_VALIDATION_DUMP="${STREAM_VALIDATION_DUMP:-True}"

echo "EVAL_PARQUET=${EVAL_PARQUET}"
echo "EXP_NAME=${EXP_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "VAL_N_RESP_PER_PROMPT=${VAL_N_RESP_PER_PROMPT}"
echo "VAL_TEMPERATURE=${VAL_TEMPERATURE}"
echo "VAL_TOP_P=${VAL_TOP_P}"
echo "NGPUS_PER_NODE=${NGPUS_PER_NODE}"
echo "INFER_TP_SIZE=${INFER_TP_SIZE}"
echo "SAVE_FULL_TRAJECTORY_ALL=${SAVE_FULL_TRAJECTORY_ALL}"
echo "STREAM_VALIDATION_DUMP=${STREAM_VALIDATION_DUMP}"

bash recipe/codevision/eval_vstar_tools_a100_4gpu.sh
