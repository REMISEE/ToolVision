#!/usr/bin/env bash
set -euo pipefail

# Run a small current-prompt, tool-enabled eval matrix inside one DLC job.
# Intended to isolate prompt/tool-schema/temperature effects before launching RL.

PROJECT_NAME="${PROJECT_NAME:-CodeVision}"
EXP_PREFIX="${EXP_PREFIX:-current_prompt_tool_eval_matrix}"
DATASETS="${DATASETS:-fsc147 chartqa}"
TEMPERATURES="${TEMPERATURES:-0 0.7}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mnt/cpfs/delinmao/Benchmarks}"

MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"
SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-recipe/codevision/config/sp3.txt}"
TOOL_CFG_TEMPLATE_PATH="${TOOL_CFG_TEMPLATE_PATH:-recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml}"

VAL_BSZ="${VAL_BSZ:-32}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-1}"
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
    vstar)
      echo "${VSTAR_PARQUET:-${BENCHMARK_ROOT}/vstar-bench/vstar_codevision_eval.parquet}"
      ;;
    chartqa)
      echo "${CHARTQA_PARQUET:-${BENCHMARK_ROOT}/ChartQA/chartqa_codevision_eval.parquet}"
      ;;
    fsc147|fsc147_val)
      echo "${FSC147_VAL_PARQUET:-${BENCHMARK_ROOT}/FSC147/fsc147_val_codevision_eval.parquet}"
      ;;
    fsc147_test)
      echo "${FSC147_TEST_PARQUET:-${BENCHMARK_ROOT}/FSC147/fsc147_test_codevision_eval.parquet}"
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
    arxivqa)
      echo "${ARXIVQA_PARQUET:-${BENCHMARK_ROOT}/ArxivQA/arxivqa_codevision_eval.parquet}"
      ;;
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
      echo "Unknown dataset '$1'." >&2
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
echo "RESUME_MODE=${RESUME_MODE}"
echo "RESUME_FROM_PATH=${RESUME_FROM_PATH}"
echo "SYSTEM_PROMPT_PATH=${SYSTEM_PROMPT_PATH}"
echo "TOOL_CFG_TEMPLATE_PATH=${TOOL_CFG_TEMPLATE_PATH}"
echo "DATASETS=${DATASETS}"
echo "TEMPERATURES=${TEMPERATURES}"
echo "N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT}"
echo "VAL_N_RESP_PER_PROMPT=${VAL_N_RESP_PER_PROMPT}"

for dataset in ${DATASETS//,/ }; do
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
    RESUME_MODE="${RESUME_MODE}" \
    RESUME_FROM_PATH="${RESUME_FROM_PATH}" \
    TOOL_CFG_TEMPLATE_PATH="${TOOL_CFG_TEMPLATE_PATH}" \
    SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH}" \
    PROJECT_NAME="${PROJECT_NAME}" \
    EXP_NAME="${exp_name}" \
    SAVE_DIR="./saves/${PROJECT_NAME}/${exp_name}" \
    VAL_BSZ="${VAL_BSZ}" \
    N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT}" \
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
