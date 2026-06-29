#!/usr/bin/env bash
set -euo pipefail

# Submit the expanded benchmark panel as several independent DLC jobs.
# Each group uses one external tool-service replica to avoid overloading a
# single replica. Prepare parquet files first with:
#   bash scripts/prepare_big_eval_benchmarks_20260624.sh

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/cpfs/delinmao}"
cd "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:-${WORKSPACE_ROOT}/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH}")}"

TOOL_DLC_HOST="${TOOL_DLC_HOST:-172.17.1.140}"
PRIORITY="${PRIORITY:-8}"
ENABLE_LLM_JUDGE="${ENABLE_LLM_JUDGE:-1}"

JOB_NAME_PREFIX_BASE="${JOB_NAME_PREFIX_BASE:-cv-bigbench-${MODEL_TAG}}"
EXP_PREFIX_BASE="${EXP_PREFIX_BASE:-bigbench_${MODEL_TAG}}"

GROUP_A="${GROUP_A:-realworldqa mmstar cvbench spatialmqa}"
GROUP_B="${GROUP_B:-docvqa_val infovqa_val ocrbench_v2}"
GROUP_C="${GROUP_C:-mme_realworld_lite mme_realworld mme_realworld_cn}"
GROUP_D="${GROUP_D:-pixmo_count pixmo_count_lmms countqa mvtoolbench mmvet}"

REPLICA_A="${REPLICA_A:-4}"
REPLICA_B="${REPLICA_B:-5}"
REPLICA_C="${REPLICA_C:-6}"
REPLICA_D="${REPLICA_D:-7}"

replica_base_port() {
  local replica="$1"
  echo $((18080 + replica * 10))
}

submit_group() {
  local suffix="$1"
  local replica="$2"
  local datasets="$3"
  local base_port
  [[ -n "${datasets// /}" ]] || return 0
  base_port="$(replica_base_port "${replica}")"

  echo "Submitting ${suffix}: replica=${replica} base_port=${base_port} datasets=${datasets}"
  TOOL_DLC_HOST="${TOOL_DLC_HOST}" \
  TOOL_DLC_REPLICA="${replica}" \
  TOOL_DLC_BASE_PORT="${base_port}" \
  MODEL_PATH="${MODEL_PATH}" \
  JOB_NAME_PREFIX="${JOB_NAME_PREFIX_BASE}-${suffix}" \
  EXP_PREFIX="${EXP_PREFIX_BASE}_${suffix}" \
  GROUP1_DATASETS="${datasets}" \
  GROUP2_DATASETS= \
  TEMPERATURES=0 \
  NGPUS_PER_NODE=8 \
  WORKER_GPU=8 \
  INFER_TP_SIZE=4 \
  VAL_BSZ="${VAL_BSZ:-64}" \
  N_RESP_PER_PROMPT=1 \
  VAL_N_RESP_PER_PROMPT=1 \
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}" \
  ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-16}" \
  SAVE_EVAL_METADATA=1 \
  SAVE_VAL_GENERATIONS="${SAVE_VAL_GENERATIONS:-1}" \
  SAVE_FULL_TRAJECTORY_ALL=0 \
  DIAGNOSTIC_MAX_PER_BUCKET="${DIAGNOSTIC_MAX_PER_BUCKET:-200}" \
  ENABLE_LLM_JUDGE="${ENABLE_LLM_JUDGE}" \
  PRIORITY="${PRIORITY}" \
  bash scripts/submit_dlc_sft_v04_eval_8bench_4gpu_split.sh
}

echo "Expanded bigbench eval"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOOL_DLC_HOST=${TOOL_DLC_HOST}"
echo "ENABLE_LLM_JUDGE=${ENABLE_LLM_JUDGE}"
echo "PRIORITY=${PRIORITY}"

submit_group "gA" "${REPLICA_A}" "${GROUP_A}"
submit_group "gB" "${REPLICA_B}" "${GROUP_B}"
submit_group "gC" "${REPLICA_C}" "${GROUP_C}"
submit_group "gD" "${REPLICA_D}" "${GROUP_D}"
