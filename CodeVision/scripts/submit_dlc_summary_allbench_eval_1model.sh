#!/usr/bin/env bash
set -euo pipefail

# Submit one ToolVision-agent eval job for every benchmark row we keep in the
# current summary, using local validation rows for DocVQA/InfoVQA.
# Base Thinking official/direct baselines should not use this script.

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
cd "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a merged HF model directory}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH}")}"
TOOL_DLC_HOST="${TOOL_DLC_HOST:?Set TOOL_DLC_HOST to the running tool DLC pod IP}"
TOOL_DLC_REPLICA="${TOOL_DLC_REPLICA:?Set TOOL_DLC_REPLICA, e.g. 1}"
TOOL_DLC_BASE_PORT="${TOOL_DLC_BASE_PORT:-$((18080 + TOOL_DLC_REPLICA * 10))}"

SUMMARY_DATASETS="${SUMMARY_DATASETS:-vstar chartqa ocrbench countbench hrbench4k hrbench8k fsc147_val fsc147_test arxivqa mme_realworld_lite mme_realworld_cn realworldqa mmstar docvqa_val infovqa_val}"

echo "Summary all-bench eval, 1 model / 1 tool replica"
echo "MODEL_TAG=${MODEL_TAG}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOOL_DLC_HOST=${TOOL_DLC_HOST}"
echo "TOOL_DLC_REPLICA=${TOOL_DLC_REPLICA}"
echo "TOOL_DLC_BASE_PORT=${TOOL_DLC_BASE_PORT}"
echo "SUMMARY_DATASETS=${SUMMARY_DATASETS}"

TOOL_DLC_HOST="${TOOL_DLC_HOST}" \
TOOL_DLC_REPLICA="${TOOL_DLC_REPLICA}" \
TOOL_DLC_BASE_PORT="${TOOL_DLC_BASE_PORT}" \
MODEL_PATH="${MODEL_PATH}" \
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-cv-summary-${MODEL_TAG}-r${TOOL_DLC_REPLICA}}" \
EXP_PREFIX="${EXP_PREFIX:-summary_${MODEL_TAG}_r${TOOL_DLC_REPLICA}}" \
GROUP1_DATASETS="${SUMMARY_DATASETS}" \
GROUP2_DATASETS= \
TEMPERATURES=0 \
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}" \
WORKER_GPU="${WORKER_GPU:-8}" \
INFER_TP_SIZE="${INFER_TP_SIZE:-4}" \
VAL_BSZ="${VAL_BSZ:-16}" \
N_RESP_PER_PROMPT=1 \
VAL_N_RESP_PER_PROMPT=1 \
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}" \
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}" \
SAVE_EVAL_METADATA=1 \
SAVE_VAL_GENERATIONS="${SAVE_VAL_GENERATIONS:-1}" \
SAVE_FULL_TRAJECTORY_ALL=0 \
DIAGNOSTIC_MAX_PER_BUCKET="${DIAGNOSTIC_MAX_PER_BUCKET:-200}" \
ENABLE_LLM_JUDGE="${ENABLE_LLM_JUDGE:-1}" \
PRIORITY="${PRIORITY:-6}" \
bash scripts/submit_dlc_sft_v04_eval_8bench_4gpu_split.sh
