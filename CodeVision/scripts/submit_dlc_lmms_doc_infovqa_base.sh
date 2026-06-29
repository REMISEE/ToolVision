#!/usr/bin/env bash
set -euo pipefail

# No-tool lmms-eval baseline for Qwen3-VL-8B-Thinking on local DocVQA/InfoVQA
# validation sets. This is intentionally separate from ToolVision agent eval.

export JOB_NAME="${JOB_NAME:-cv-lmms-base-doc-infovqa-val-8gpu}"
export RUN_NAME="${RUN_NAME:-base_qwen3vl8bthinking_doc_infovqa_val_lmms_t0_len32k_gen4096}"
export TASK_NAME="${TASK_NAME:-tv_docvqa_val_local,tv_infovqa_val_local}"
export MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/models/Qwen3-VL-8B-Thinking}"
export CONDA_ENV="${CONDA_ENV:-/mnt/cpfs/delinmao/envs/codevision}"
export INCLUDE_PATH="${INCLUDE_PATH:-/mnt/cpfs/delinmao/ToolVision/CodeVision/lmms_tasks}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export PRIORITY="${PRIORITY:-6}"

bash /mnt/cpfs/delinmao/ToolVision/CodeVision/scripts/submit_dlc_lmms_countbench_base.sh
