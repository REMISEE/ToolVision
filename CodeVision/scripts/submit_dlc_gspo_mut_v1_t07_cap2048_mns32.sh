#!/usr/bin/env bash
set -euo pipefail

# Submit GSPO on the MUT v1 mixture:
#   mut     8,464 rows, mut_weight=0.5
#   weak    6,486 rows, mut_weight=0.2
#   regular 10,000 rows, mut_weight=0.0
#
# Reward mode:
#   R = R_acc + 0.2 * R_protocol + mut_weight * R_mut
#       - 0.05 * max(0, NumTurns - 6)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_MUT_V1="/mnt/cpfs/delinmao/ToolVision/CodeVision/outputs/analysis/mut_v1_20260616/mut_v1_train.parquet"
DEFAULT_TRAIN_FILES="['${TRAIN_MUT_V1}']"

if [[ ! -f "${TRAIN_MUT_V1}" ]]; then
  echo "Expected MUT v1 train parquet is missing: ${TRAIN_MUT_V1}" >&2
  exit 1
fi

if [[ -n "${TRAIN_FILES:-}" && "${TRAIN_FILES}" != "${DEFAULT_TRAIN_FILES}" && "${ALLOW_TRAIN_FILES_OVERRIDE:-0}" != "1" ]]; then
  echo "TRAIN_FILES is already set to a non-default value:" >&2
  echo "  ${TRAIN_FILES}" >&2
  echo "This launcher is meant for MUT v1. Set ALLOW_TRAIN_FILES_OVERRIDE=1 if this is intentional." >&2
  exit 1
fi

export JOB_NAME="${JOB_NAME:-codevision_gspo_mut_v1_t07_cap2048_mns32_0616}"
export EXP_NAME="${EXP_NAME:-qwen3vl8b_gspo_mut_v1_t07_cap2048_mns32_0616}"

export MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03}"
export SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-recipe/codevision/config/sp3.txt}"
export TOOL_CFG_TEMPLATE_PATH="${TOOL_CFG_TEMPLATE_PATH:-recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml}"
export TRAIN_FILES="${TRAIN_FILES:-${DEFAULT_TRAIN_FILES}}"

export TOOL_REWARD_MODE="${TOOL_REWARD_MODE:-mut_clean}"
export TOOL_REWARD_MUT_PROTOCOL_WEIGHT="${TOOL_REWARD_MUT_PROTOCOL_WEIGHT:-0.2}"
export TOOL_REWARD_MUT_TURN_PENALTY_WEIGHT="${TOOL_REWARD_MUT_TURN_PENALTY_WEIGHT:-0.05}"
export TOOL_REWARD_MUT_TURN_PENALTY_THRESHOLD="${TOOL_REWARD_MUT_TURN_PENALTY_THRESHOLD:-6}"

# Kept explicit for old modes/metrics; mut_clean reads its own weights above.
export FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.2}"
export TOOL_REWARD_TOOL_ERROR_WEIGHT="${TOOL_REWARD_TOOL_ERROR_WEIGHT:-0.02}"
export TOOL_REWARD_INVALID_CALL_WEIGHT="${TOOL_REWARD_INVALID_CALL_WEIGHT:-0.02}"

export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-True}"
export ROLLOUT_MAX_TOKENS_PER_TURN="${ROLLOUT_MAX_TOKENS_PER_TURN:-2048}"

# Safe scheduler setting from the GSPO format-collapse probes.
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export TRAIN_BSZ="${TRAIN_BSZ:-64}"
export TRAIN_MINI_BSZ="${TRAIN_MINI_BSZ:-32}"

export SAVE_FREQ="${SAVE_FREQ:-10}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-12}"
export MAX_CRITIC_CKPT_TO_KEEP="${MAX_CRITIC_CKPT_TO_KEEP:-12}"
export LOG_TRAIN_FREQ="${LOG_TRAIN_FREQ:-5}"
export LOG_TRAIN_GENERATIONS="${LOG_TRAIN_GENERATIONS:-64}"
export LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-8}"

export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export TEST_FREQ="${TEST_FREQ:--1}"

export ENABLE_LLM_JUDGE="${ENABLE_LLM_JUDGE:-1}"
export TOOLVISION_RL_USE_LLM_JUDGE="${TOOLVISION_RL_USE_LLM_JUDGE:-0}"
export REWARD_LAUNCH_ASYNC="${REWARD_LAUNCH_ASYNC:-False}"
export LLM_JUDGE_TIMEOUT="${LLM_JUDGE_TIMEOUT:-30}"
export LLM_JUDGE_MAX_RETRIES="${LLM_JUDGE_MAX_RETRIES:-2}"

export ENABLE_WANDB="${ENABLE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"

echo "Submitting MUT v1 GSPO"
echo "ROOT_DIR=${ROOT_DIR}"
echo "JOB_NAME=${JOB_NAME}"
echo "EXP_NAME=${EXP_NAME}"
echo "TRAIN_FILES=${TRAIN_FILES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOOL_REWARD_MODE=${TOOL_REWARD_MODE}"
echo "TOOL_REWARD_MUT_PROTOCOL_WEIGHT=${TOOL_REWARD_MUT_PROTOCOL_WEIGHT}"
echo "TOOL_REWARD_MUT_TURN_PENALTY_WEIGHT=${TOOL_REWARD_MUT_TURN_PENALTY_WEIGHT}"
echo "TOOL_REWARD_MUT_TURN_PENALTY_THRESHOLD=${TOOL_REWARD_MUT_TURN_PENALTY_THRESHOLD}"
echo "ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE}"
echo "ROLLOUT_TOP_P=${ROLLOUT_TOP_P}"
echo "ROLLOUT_MAX_TOKENS_PER_TURN=${ROLLOUT_MAX_TOKENS_PER_TURN}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "SAVE_FREQ=${SAVE_FREQ}"

exec bash "${SCRIPT_DIR}/submit_dlc_gspo_direct_full.sh"
