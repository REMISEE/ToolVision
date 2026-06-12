#!/usr/bin/env bash
set -euo pipefail

# 2026-06-10 RL restart launcher.
#
# This is the conservative path for the next full run:
#   - reuse the last-good "before_newdata" reward preset;
#   - replace the old 34k train_no_ood data with the screened 26k mixture;
#   - lower rollout sampling to t=0.7/top_p=0.95;
#   - cap each assistant turn at 2048 tokens to stop long-tail runaways;
#   - keep checkpoints every 50 steps for early rollback/eval.
#
# The actual DLC submit still goes through submit_dlc_gspo_direct_final_before_newdata.sh,
# which then execs submit_dlc_gspo_direct_full.sh. This wrapper only pins the
# run-specific defaults so the submit command is not a page of env vars.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_26K="/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_clean_no_extra_image/parquet/train_medium_clean_21k_plus_benchmark_pass16_partial_26591_rewardfix_fsc09.parquet"
DEFAULT_TRAIN_FILES="['${TRAIN_26K}']"

if [[ ! -f "${TRAIN_26K}" ]]; then
  echo "Expected 26k train parquet is missing: ${TRAIN_26K}" >&2
  exit 1
fi

if [[ -n "${TRAIN_FILES:-}" && "${TRAIN_FILES}" != "${DEFAULT_TRAIN_FILES}" && "${ALLOW_TRAIN_FILES_OVERRIDE:-0}" != "1" ]]; then
  echo "TRAIN_FILES is already set to a non-default value:" >&2
  echo "  ${TRAIN_FILES}" >&2
  echo "This launcher is meant for the screened 26k run. Set ALLOW_TRAIN_FILES_OVERRIDE=1 if this is intentional." >&2
  exit 1
fi

export JOB_NAME="${JOB_NAME:-codevision_gspo_before_newdata_26k_t07_cap2048_fmtguard_0610}"
export EXP_NAME="${EXP_NAME:-qwen3vl8b_gspo_before_newdata_26k_t07_cap2048_fmtguard_0610}"

# Same SFT checkpoint and SFT-clean prompt/tool schema as the last-good run.
export MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03}"
export SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-recipe/codevision/config/sp3.txt}"
export TOOL_CFG_TEMPLATE_PATH="${TOOL_CFG_TEMPLATE_PATH:-recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml}"

# Screened 26k data. Do not use the old 34k train_no_ood default for this run.
export TRAIN_FILES="${TRAIN_FILES:-${DEFAULT_TRAIN_FILES}}"

# Keep the before_newdata reward shape explicit so the downstream full launcher
# cannot silently fall back to its generic simple_penalty defaults.
export TOOL_REWARD_MODE="${TOOL_REWARD_MODE:-rnec_with_clean}"
export FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.2}"
export TOOL_REWARD_BETA="${TOOL_REWARD_BETA:-0.3}"
export TOOL_REWARD_CLEAN_TOOL_BASELINE_WEIGHT="${TOOL_REWARD_CLEAN_TOOL_BASELINE_WEIGHT:-0.05}"
export TOOL_REWARD_CLEAN_TOOL_WEIGHT="${TOOL_REWARD_CLEAN_TOOL_WEIGHT:-0.15}"
export TOOL_REWARD_TOOL_ERROR_WEIGHT="${TOOL_REWARD_TOOL_ERROR_WEIGHT:-0.02}"
export TOOL_REWARD_INVALID_CALL_WEIGHT="${TOOL_REWARD_INVALID_CALL_WEIGHT:-0.02}"
export TOOL_REWARD_OVERUSE_WEIGHT="${TOOL_REWARD_OVERUSE_WEIGHT:-0.05}"
export TOOL_REWARD_OVERUSE_THRESHOLD="${TOOL_REWARD_OVERUSE_THRESHOLD:-4}"

# Rollout guardrails for the low-format step-0 issue.
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-True}"
export ROLLOUT_MAX_TOKENS_PER_TURN="${ROLLOUT_MAX_TOKENS_PER_TURN:-2048}"

# Preserve the previous full-run training shape unless explicitly overridden.
export TRAIN_BSZ="${TRAIN_BSZ:-64}"
export TRAIN_MINI_BSZ="${TRAIN_MINI_BSZ:-32}"
export TRAIN_MICRO_BSZ_PER_GPU="${TRAIN_MICRO_BSZ_PER_GPU:-1}"
export INFER_MICRO_BSZ_PER_GPU="${INFER_MICRO_BSZ_PER_GPU:-1}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export MAX_TURNS="${MAX_TURNS:-12}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"

# More checkpoints and early generations so we can inspect/cut quickly.
export SAVE_FREQ="${SAVE_FREQ:-50}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-12}"
export MAX_CRITIC_CKPT_TO_KEEP="${MAX_CRITIC_CKPT_TO_KEEP:-12}"
export LOG_TRAIN_FREQ="${LOG_TRAIN_FREQ:-5}"
export LOG_TRAIN_GENERATIONS="${LOG_TRAIN_GENERATIONS:-64}"
export LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-8}"

# No validation inside the full RL job by default. Evaluate checkpoints with the
# eval launcher after we see stable early rollout behavior.
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export TEST_FREQ="${TEST_FREQ:--1}"

# Judge-family fallback only. Rule-family samples should not call the judge on
# every mismatch.
export ENABLE_LLM_JUDGE="${ENABLE_LLM_JUDGE:-1}"
export TOOLVISION_RL_USE_LLM_JUDGE="${TOOLVISION_RL_USE_LLM_JUDGE:-0}"
export REWARD_LAUNCH_ASYNC="${REWARD_LAUNCH_ASYNC:-False}"
export LLM_JUDGE_TIMEOUT="${LLM_JUDGE_TIMEOUT:-30}"
export LLM_JUDGE_MAX_RETRIES="${LLM_JUDGE_MAX_RETRIES:-2}"

# Keep W&B on by default; submit_dlc_gspo_direct_full.sh will fail fast if the
# key is missing, which is better than launching an untracked run.
export ENABLE_WANDB="${ENABLE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"

echo "Submitting screened-26k before_newdata restart"
echo "ROOT_DIR=${ROOT_DIR}"
echo "JOB_NAME=${JOB_NAME}"
echo "EXP_NAME=${EXP_NAME}"
echo "TRAIN_FILES=${TRAIN_FILES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "SYSTEM_PROMPT_PATH=${SYSTEM_PROMPT_PATH}"
echo "TOOL_CFG_TEMPLATE_PATH=${TOOL_CFG_TEMPLATE_PATH}"
echo "ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE}"
echo "ROLLOUT_TOP_P=${ROLLOUT_TOP_P}"
echo "ROLLOUT_MAX_TOKENS_PER_TURN=${ROLLOUT_MAX_TOKENS_PER_TURN}"
echo "SAVE_FREQ=${SAVE_FREQ}"
echo "LOG_TRAIN_FREQ=${LOG_TRAIN_FREQ}"
echo "TOOL_REWARD_MODE=${TOOL_REWARD_MODE}"
echo "TOOL_REWARD_INVALID_CALL_WEIGHT=${TOOL_REWARD_INVALID_CALL_WEIGHT}"

exec bash "${SCRIPT_DIR}/submit_dlc_gspo_direct_final_before_newdata.sh"
