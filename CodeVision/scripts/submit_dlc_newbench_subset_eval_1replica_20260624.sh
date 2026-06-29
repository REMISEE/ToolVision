#!/usr/bin/env bash
set -euo pipefail

# Submit the 7-benchmark newbench subset as ONE DLC job on ONE tool replica.
#
# Use this for ToolVision-agent eval models. Do not use it for base
# Qwen3-VL Thinking official/direct baselines.

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
cd "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a HF model directory}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH}")}"
TOOL_DLC_HOST="${TOOL_DLC_HOST:?Set TOOL_DLC_HOST to the current running tool DLC pod IP. Use scripts/dlc_tool_urls_from_job.sh <job_id> after submitting tool services.}"
TOOL_DLC_REPLICA="${TOOL_DLC_REPLICA:?Set TOOL_DLC_REPLICA, e.g. 4 or 7}"
TOOL_DLC_BASE_PORT="${TOOL_DLC_BASE_PORT:-$((18080 + TOOL_DLC_REPLICA * 10))}"

NEWBENCH_DATASETS="${NEWBENCH_DATASETS:-realworldqa mmstar docvqa_val infovqa_val mme_realworld_lite mme_realworld_cn mmvet}"

echo "Newbench 7 eval, 1-replica mode"
echo "MODEL_TAG=${MODEL_TAG}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOOL_DLC_HOST=${TOOL_DLC_HOST}"
echo "TOOL_DLC_REPLICA=${TOOL_DLC_REPLICA}"
echo "TOOL_DLC_BASE_PORT=${TOOL_DLC_BASE_PORT}"
echo "NEWBENCH_DATASETS=${NEWBENCH_DATASETS}"

TOOL_DLC_HOST="${TOOL_DLC_HOST}" \
TOOL_DLC_REPLICA="${TOOL_DLC_REPLICA}" \
TOOL_DLC_BASE_PORT="${TOOL_DLC_BASE_PORT}" \
MODEL_PATH="${MODEL_PATH}" \
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-cv-newbench-${MODEL_TAG}-r${TOOL_DLC_REPLICA}}" \
EXP_PREFIX="${EXP_PREFIX:-newbench_${MODEL_TAG}_r${TOOL_DLC_REPLICA}}" \
GROUP1_DATASETS="${NEWBENCH_DATASETS}" \
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
PRIORITY="${PRIORITY:-8}" \
bash scripts/submit_dlc_sft_v04_eval_8bench_4gpu_split.sh
