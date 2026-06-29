#!/usr/bin/env bash
set -euo pipefail

# Submit GSPO on the MUT v2 balanced-order mixture.
#
# Data:
#   outputs/analysis/mut_v2_20260617/mut_v2_train_balanced.parquet
#
# Each sequential 64-prompt batch is ordered as:
#   regular_9_15  28  mut_weight=0.0, regular_tool_penalty=0.05
#   hard_regular  10  mut_weight=0.0, regular_tool_penalty=0.0
#   mut           20  mut_weight=0.5, regular_tool_penalty=0.0
#   weak_clean     6  mut_weight=0.2, regular_tool_penalty=0.0
#
# Keep DATA_SHUFFLE=False for this balanced-order parquet.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_MUT_V2="/mnt/cpfs/delinmao/ToolVision/CodeVision/outputs/analysis/mut_v2_20260617/mut_v2_train_balanced.parquet"
DEFAULT_TRAIN_FILES="['${TRAIN_MUT_V2}']"

if [[ ! -f "${TRAIN_MUT_V2}" ]]; then
  echo "Expected MUT v2 train parquet is missing: ${TRAIN_MUT_V2}" >&2
  echo "Build it with: python3 recipe/codevision/tools/build_mut_v2_train.py" >&2
  exit 1
fi

if [[ -n "${TRAIN_FILES:-}" && "${TRAIN_FILES}" != "${DEFAULT_TRAIN_FILES}" && "${ALLOW_TRAIN_FILES_OVERRIDE:-0}" != "1" ]]; then
  echo "TRAIN_FILES is already set to a non-default value:" >&2
  echo "  ${TRAIN_FILES}" >&2
  echo "This launcher is meant for MUT v2. Set ALLOW_TRAIN_FILES_OVERRIDE=1 if this is intentional." >&2
  exit 1
fi

export JOB_NAME="${JOB_NAME:-cv-mutv2}"
export EXP_NAME="${EXP_NAME:-mutv2}"

export MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03}"
export SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-recipe/codevision/config/sp3.txt}"
export TOOL_CFG_TEMPLATE_PATH="${TOOL_CFG_TEMPLATE_PATH:-recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml}"
export TRAIN_FILES="${TRAIN_FILES:-${DEFAULT_TRAIN_FILES}}"

export TOOL_REWARD_MODE="${TOOL_REWARD_MODE:-mut_clean}"
export TOOL_REWARD_MUT_PROTOCOL_WEIGHT="${TOOL_REWARD_MUT_PROTOCOL_WEIGHT:-0.2}"
export TOOL_REWARD_MUT_TURN_PENALTY_WEIGHT="${TOOL_REWARD_MUT_TURN_PENALTY_WEIGHT:-0.05}"
export TOOL_REWARD_MUT_TURN_PENALTY_THRESHOLD="${TOOL_REWARD_MUT_TURN_PENALTY_THRESHOLD:-6}"

export FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.2}"
export TOOL_REWARD_TOOL_ERROR_WEIGHT="${TOOL_REWARD_TOOL_ERROR_WEIGHT:-0.02}"
export TOOL_REWARD_INVALID_CALL_WEIGHT="${TOOL_REWARD_INVALID_CALL_WEIGHT:-0.02}"

export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-True}"
export ROLLOUT_MAX_TOKENS_PER_TURN="${ROLLOUT_MAX_TOKENS_PER_TURN:-2048}"

export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export TRAIN_BSZ="${TRAIN_BSZ:-64}"
export TRAIN_MINI_BSZ="${TRAIN_MINI_BSZ:-32}"
export DATA_SHUFFLE="${DATA_SHUFFLE:-False}"

export SAVE_FREQ="${SAVE_FREQ:-10}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-12}"
export MAX_CRITIC_CKPT_TO_KEEP="${MAX_CRITIC_CKPT_TO_KEEP:-12}"
export LOG_TRAIN_FREQ="${LOG_TRAIN_FREQ:-5}"
export LOG_TRAIN_GENERATIONS="${LOG_TRAIN_GENERATIONS:-64}"
export LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-8}"

export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export TEST_FREQ="${TEST_FREQ:--1}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"

export ENABLE_LLM_JUDGE="${ENABLE_LLM_JUDGE:-1}"
export TOOLVISION_RL_USE_LLM_JUDGE="${TOOLVISION_RL_USE_LLM_JUDGE:-0}"
export REWARD_LAUNCH_ASYNC="${REWARD_LAUNCH_ASYNC:-False}"
export LLM_JUDGE_TIMEOUT="${LLM_JUDGE_TIMEOUT:-30}"
export LLM_JUDGE_MAX_RETRIES="${LLM_JUDGE_MAX_RETRIES:-2}"

export ENABLE_WANDB="${ENABLE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"

echo "Submitting MUT v2 GSPO"
echo "ROOT_DIR=${ROOT_DIR}"
echo "JOB_NAME=${JOB_NAME}"
echo "EXP_NAME=${EXP_NAME}"
echo "TRAIN_FILES=${TRAIN_FILES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOOL_REWARD_MODE=${TOOL_REWARD_MODE}"
echo "DATA_SHUFFLE=${DATA_SHUFFLE}"
echo "TRAIN_BSZ=${TRAIN_BSZ}"
echo "N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE}"
echo "ROLLOUT_TOP_P=${ROLLOUT_TOP_P}"
echo "ROLLOUT_MAX_TOKENS_PER_TURN=${ROLLOUT_MAX_TOKENS_PER_TURN}"
echo "SAVE_FREQ=${SAVE_FREQ}"

exec bash "${SCRIPT_DIR}/submit_dlc_gspo_direct_full.sh"
