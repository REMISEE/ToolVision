#!/usr/bin/env bash
set -euo pipefail

# Run a small current-prompt, tool-enabled eval matrix inside one DLC job.
# Intended to isolate prompt/tool-schema/temperature effects before launching RL.

PROJECT_NAME="${PROJECT_NAME:-CodeVision}"
EXP_PREFIX="${EXP_PREFIX:-current_prompt_tool_eval_matrix}"
DATASETS="${DATASETS:-fsc147 chartqa}"
TEMPERATURES="${TEMPERATURES:-0 0.7}"

MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03}"
SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-recipe/codevision/config/sp3.txt}"
TOOL_CFG_TEMPLATE_PATH="${TOOL_CFG_TEMPLATE_PATH:-recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml}"

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

echo "MODEL_PATH=${MODEL_PATH}"
echo "SYSTEM_PROMPT_PATH=${SYSTEM_PROMPT_PATH}"
echo "TOOL_CFG_TEMPLATE_PATH=${TOOL_CFG_TEMPLATE_PATH}"
echo "DATASETS=${DATASETS}"
echo "TEMPERATURES=${TEMPERATURES}"

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

    echo "========== eval ${dataset} temp=${temp} =========="
    EVAL_PARQUET="${eval_parquet}" \
    MODEL_PATH="${MODEL_PATH}" \
    TOOL_CFG_TEMPLATE_PATH="${TOOL_CFG_TEMPLATE_PATH}" \
    SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH}" \
    PROJECT_NAME="${PROJECT_NAME}" \
    EXP_NAME="${exp_name}" \
    SAVE_DIR="./saves/${PROJECT_NAME}/${exp_name}" \
    VAL_BSZ="${VAL_BSZ}" \
    N_RESP_PER_PROMPT=1 \
    VAL_N_RESP_PER_PROMPT="${VAL_N_RESP_PER_PROMPT}" \
    MAX_TURNS="${MAX_TURNS}" \
    VAL_TEMPERATURE="${temp}" \
    VAL_TOP_P="${val_top_p}" \
    VAL_DO_SAMPLE="${val_do_sample}" \
    ROLLOUT_MAX_TOKENS_PER_TURN="${ROLLOUT_MAX_TOKENS_PER_TURN}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS}" \
    SAVE_EVAL_METADATA="${SAVE_EVAL_METADATA}" \
    SAVE_VAL_GENERATIONS="${SAVE_VAL_GENERATIONS}" \
    SAVE_FULL_TRAJECTORY_ALL="${SAVE_FULL_TRAJECTORY_ALL}" \
    DIAGNOSTIC_MAX_PER_BUCKET="${DIAGNOSTIC_MAX_PER_BUCKET}" \
    DIAGNOSTIC_SAMPLE_SEED="${DIAGNOSTIC_SAMPLE_SEED}" \
    bash recipe/codevision/eval_vstar_tools_a100_4gpu.sh
  done
done
