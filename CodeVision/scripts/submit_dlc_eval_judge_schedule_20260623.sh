#!/usr/bin/env bash
set -euo pipefail

# Judge-on eval schedule for MUT v1 RL checkpoints and ArxivQA baselines.
# This script intentionally does not store API keys. Export one of:
#   OFFLINE_SFT_QWEN_API_KEY / DASHSCOPE_API_KEY / OPENAI_API_KEY / LLM_JUDGE_API_KEY
#
# Usage:
#   bash scripts/submit_dlc_eval_judge_schedule_20260623.sh plan
#   bash scripts/submit_dlc_eval_judge_schedule_20260623.sh merge
#   bash scripts/submit_dlc_eval_judge_schedule_20260623.sh submit-wave1
#   bash scripts/submit_dlc_eval_judge_schedule_20260623.sh submit-wave2

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
cd "${ROOT_DIR}"

TOOL_HOST="${TOOL_HOST:-172.17.1.140}"
PRIORITY="${PRIORITY:-8}"
DATASETS_ALL="${DATASETS_ALL:-vstar chartqa ocrbench countbench hrbench4k hrbench8k fsc147_val fsc147_test arxivqa}"

V03_MODEL="${V03_MODEL:-/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03}"
BASE_THINKING_MODEL="${BASE_THINKING_MODEL:-/mnt/cpfs/delinmao/models/Qwen3-VL-8B-Thinking}"
MERGED_ROOT="${MERGED_ROOT:-${ROOT_DIR}/saves/ToolVisionRL/merged_hf}"

export LLM_JUDGE_BASE_URL="${LLM_JUDGE_BASE_URL:-${OFFLINE_SFT_QWEN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}}"
export LLM_JUDGE_MODEL_NAME="${LLM_JUDGE_MODEL_NAME:-${OFFLINE_SFT_QWEN_MODEL:-qwen3.6-plus}}"
export LLM_JUDGE_API_KEY="${LLM_JUDGE_API_KEY:-${OFFLINE_SFT_QWEN_API_KEY:-${DASHSCOPE_API_KEY:-${OPENAI_API_KEY:-}}}}"

replica_base_port() {
  local replica="$1"
  echo $((18080 + replica * 10))
}

check_judge_env() {
  if [[ -z "${LLM_JUDGE_API_KEY}" ]]; then
    echo "Missing judge key. Export OFFLINE_SFT_QWEN_API_KEY or LLM_JUDGE_API_KEY first." >&2
    exit 1
  fi
}

check_model_ready() {
  local model_path="$1"
  if [[ ! -f "${model_path}/config.json" ]]; then
    return 1
  fi
  if [[ -f "${model_path}/model.safetensors.index.json" ]]; then
    return 0
  fi
  compgen -G "${model_path}/*.safetensors" >/dev/null
}

submit_eval() {
  local wave="$1"
  local replica="$2"
  local model_path="$3"
  local job_name="$4"
  local exp_prefix="$5"
  local datasets="$6"
  local base_port
  base_port="$(replica_base_port "${replica}")"

  if ! check_model_ready "${model_path}"; then
    echo "[skip:${wave}] model not ready: ${model_path}" >&2
    return 0
  fi

  echo "[submit:${wave}] replica=${replica} port=${base_port} job=${job_name}"
  TOOL_DLC_HOST="${TOOL_HOST}" \
  TOOL_DLC_BASE_PORT="${base_port}" \
  TOOL_DLC_REPLICA="${replica}" \
  MODEL_PATH="${model_path}" \
  JOB_NAME_PREFIX="${job_name}" \
  EXP_PREFIX="${exp_prefix}" \
  GROUP1_DATASETS="${datasets}" \
  GROUP2_DATASETS= \
  NGPUS_PER_NODE=8 \
  WORKER_GPU=8 \
  INFER_TP_SIZE=4 \
  VAL_BSZ=64 \
  MAX_NUM_SEQS=32 \
  ROLLOUT_AGENT_NUM_WORKERS=16 \
  SAVE_VAL_GENERATIONS=1 \
  ENABLE_LLM_JUDGE=1 \
  LLM_JUDGE_BASE_URL="${LLM_JUDGE_BASE_URL}" \
  LLM_JUDGE_MODEL_NAME="${LLM_JUDGE_MODEL_NAME}" \
  LLM_JUDGE_API_KEY="${LLM_JUDGE_API_KEY}" \
  LLM_JUDGE_ENABLE_THINKING=0 \
  PRIORITY="${PRIORITY}" \
  bash scripts/submit_dlc_sft_v04_eval_8bench_4gpu_split.sh
}

merge_missing() {
  echo "[merge] mutv1_128bs step 180"
  STEPS="180" \
  TARGET_PREFIX=mutv1_128bs \
  bash scripts/merge_mutv1_128bs_eval_checkpoints.sh

  echo "[merge] mutv1_resume70 steps 280 360 389"
  SOURCE_ROOT="${ROOT_DIR}/saves/ToolVisionRL/mutv1_resume70_0618" \
  TARGET_PREFIX=mutv1_resume70 \
  STEPS="280 360 389" \
  bash scripts/merge_mutv1_128bs_eval_checkpoints.sh
}

print_plan() {
  cat <<EOF
Judge-on eval schedule.

Tool host: ${TOOL_HOST}
Avoided replicas: 0, 1, 5
Used replicas:
  replica 2 -> ${TOOL_HOST}:18100-18103
  replica 3 -> ${TOOL_HOST}:18110-18113
  replica 4 -> ${TOOL_HOST}:18120-18123
  replica 6 -> ${TOOL_HOST}:18140-18143
  replica 7 -> ${TOOL_HOST}:18150-18153

Wave 1:
  replica 2  base Thinking ArxivQA
  replica 3  v03 ArxivQA
  replica 4  mutv1_128bs step60 allbench, rerun with LLM judge
  replica 6  mutv1_128bs step140 allbench, rerun with LLM judge
  replica 7  mutv1_128bs step180 allbench, requires merge

Wave 2:
  replica 2  mutv1_resume70 step280 allbench, requires merge
  replica 3  mutv1_resume70 step360 allbench, requires merge
  replica 4  mutv1_resume70 step389 allbench, requires merge

Required before submit:
  export OFFLINE_SFT_QWEN_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
  export OFFLINE_SFT_QWEN_MODEL='qwen3.6-plus'
  export OFFLINE_SFT_QWEN_API_KEY='...'

Commands:
  bash scripts/submit_dlc_eval_judge_schedule_20260623.sh merge
  bash scripts/submit_dlc_eval_judge_schedule_20260623.sh submit-wave1
  # after wave1 finishes:
  bash scripts/submit_dlc_eval_judge_schedule_20260623.sh submit-wave2
EOF
}

submit_wave1() {
  check_judge_env
  submit_eval wave1 2 "${BASE_THINKING_MODEL}" \
    cv-base-thinking-arxivqa-judge-8gpu \
    base_thinking_arxivqa_judge_8gpu \
    "arxivqa"
  submit_eval wave1 3 "${V03_MODEL}" \
    cv-v03-arxivqa-judge-8gpu \
    v03_arxivqa_judge_8gpu \
    "arxivqa"
  submit_eval wave1 4 "${MERGED_ROOT}/mutv1_128bs_global_step_60" \
    cv-mutv1-128bs-s60-judge-8gpu \
    mutv1_128bs_s60_judge_allbench_8gpu \
    "${DATASETS_ALL}"
  submit_eval wave1 6 "${MERGED_ROOT}/mutv1_128bs_global_step_140" \
    cv-mutv1-128bs-s140-judge-8gpu \
    mutv1_128bs_s140_judge_allbench_8gpu \
    "${DATASETS_ALL}"
  submit_eval wave1 7 "${MERGED_ROOT}/mutv1_128bs_global_step_180" \
    cv-mutv1-128bs-s180-judge-8gpu \
    mutv1_128bs_s180_judge_allbench_8gpu \
    "${DATASETS_ALL}"
}

submit_wave2() {
  check_judge_env
  submit_eval wave2 2 "${MERGED_ROOT}/mutv1_resume70_global_step_280" \
    cv-mutv1-r70-s280-judge-8gpu \
    mutv1_resume70_s280_judge_allbench_8gpu \
    "${DATASETS_ALL}"
  submit_eval wave2 3 "${MERGED_ROOT}/mutv1_resume70_global_step_360" \
    cv-mutv1-r70-s360-judge-8gpu \
    mutv1_resume70_s360_judge_allbench_8gpu \
    "${DATASETS_ALL}"
  submit_eval wave2 4 "${MERGED_ROOT}/mutv1_resume70_global_step_389" \
    cv-mutv1-r70-s389-judge-8gpu \
    mutv1_resume70_s389_judge_allbench_8gpu \
    "${DATASETS_ALL}"
}

case "${1:-plan}" in
  plan) print_plan ;;
  merge) merge_missing ;;
  submit-wave1) submit_wave1 ;;
  submit-wave2) submit_wave2 ;;
  *)
    echo "Unknown command: ${1}" >&2
    echo "Use: plan | merge | submit-wave1 | submit-wave2" >&2
    exit 1
    ;;
esac
