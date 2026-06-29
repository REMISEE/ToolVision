#!/usr/bin/env bash
set -euo pipefail

# 8-GPU ArxivQA holdout eval for merged HF export of mutv1_128bs_0618/global_step_60.
# This is a thin wrapper around the shared eval submitter.

cd "${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"

export MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/ToolVision/CodeVision/saves/ToolVisionRL/merged_hf/mutv1_128bs_global_step_60}"
export RESUME_MODE="${RESUME_MODE:-disable}"
export RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"

export JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-cv-mutv1-128bs-s60-merged-arxivqa-8gpu}"
export EXP_PREFIX="${EXP_PREFIX:-mutv1_128bs_s60_merged_arxivqa_8gpu}"
export GROUP1_DATASETS="${GROUP1_DATASETS:-arxivqa}"
export GROUP2_DATASETS="${GROUP2_DATASETS:-}"
export TEMPERATURES="${TEMPERATURES:-0}"

# Use one TP=4 rollout engine per 4 GPUs; on 8 GPUs this gives two rollout
# replicas rather than one TP=8 engine, which is usually better for 8B eval.
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
export WORKER_GPU="${WORKER_GPU:-8}"
export INFER_TP_SIZE="${INFER_TP_SIZE:-4}"
export VAL_BSZ="${VAL_BSZ:-64}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-16}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"

export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-1}"
export VAL_N_RESP_PER_PROMPT="${VAL_N_RESP_PER_PROMPT:-1}"
export SAVE_EVAL_METADATA="${SAVE_EVAL_METADATA:-1}"
export SAVE_VAL_GENERATIONS="${SAVE_VAL_GENERATIONS:-1}"
export SAVE_FULL_TRAJECTORY_ALL="${SAVE_FULL_TRAJECTORY_ALL:-0}"
export DIAGNOSTIC_MAX_PER_BUCKET="${DIAGNOSTIC_MAX_PER_BUCKET:-200}"
export ENABLE_LLM_JUDGE="${ENABLE_LLM_JUDGE:-0}"
export PRIORITY="${PRIORITY:-8}"

# Default to replica 0 from the 172.17.1.140 tool service set.
export TOOL_DLC_HOST="${TOOL_DLC_HOST:-172.17.1.140}"
export TOOL_DLC_BASE_PORT="${TOOL_DLC_BASE_PORT:-18080}"

bash scripts/submit_dlc_sft_v04_eval_8bench_4gpu_split.sh
