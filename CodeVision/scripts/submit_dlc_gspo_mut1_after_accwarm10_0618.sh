#!/usr/bin/env bash
set -euo pipefail

# Continue from the acc-only warmup checkpoint with the normal MUT v1 mut_clean reward.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TOOL_IP="${TOOL_IP:-}"
TOOL_REPLICA_INDEX="${TOOL_REPLICA_INDEX:-3}"
TOOL_PORT_BASE="${TOOL_PORT_BASE:-18080}"
TOOL_PORT_STRIDE="${TOOL_PORT_STRIDE:-10}"

if [[ -n "${TOOL_IP}" ]]; then
  base=$((TOOL_PORT_BASE + TOOL_REPLICA_INDEX * TOOL_PORT_STRIDE))
  export OCR_BASE_URL="${OCR_BASE_URL:-http://${TOOL_IP}:$((base + 0))}"
  export GROUNDEDSAM2_BASE_URL="${GROUNDEDSAM2_BASE_URL:-http://${TOOL_IP}:$((base + 1))}"
  export DEPTH_BASE_URL="${DEPTH_BASE_URL:-http://${TOOL_IP}:$((base + 2))}"
  export COUNTGD_BASE_URL="${COUNTGD_BASE_URL:-http://${TOOL_IP}:$((base + 3))}"
fi

export JOB_NAME="${JOB_NAME:-cv-mut1-after-accwarm10}"
export EXP_NAME="${EXP_NAME:-mutv1_after_accwarm10_0618}"
export RESUME_MODE="${RESUME_MODE:-resume_path}"
export RESUME_FROM_PATH="${RESUME_FROM_PATH:-./saves/ToolVisionRL/mutv1_accwarm10_0618/global_step_10}"

export TOOL_REWARD_MODE="${TOOL_REWARD_MODE:-mut_clean}"
export TOOL_REWARD_MUT_PROTOCOL_WEIGHT="${TOOL_REWARD_MUT_PROTOCOL_WEIGHT:-0.2}"
export TOOL_REWARD_MUT_TURN_PENALTY_WEIGHT="${TOOL_REWARD_MUT_TURN_PENALTY_WEIGHT:-0.05}"
export TOOL_REWARD_MUT_TURN_PENALTY_THRESHOLD="${TOOL_REWARD_MUT_TURN_PENALTY_THRESHOLD:-6}"

export TRAIN_BSZ="${TRAIN_BSZ:-64}"
export TRAIN_MINI_BSZ="${TRAIN_MINI_BSZ:-32}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export LOG_TRAIN_FREQ="${LOG_TRAIN_FREQ:-5}"
export LOG_TRAIN_GENERATIONS="${LOG_TRAIN_GENERATIONS:-64}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-12}"
export MAX_CRITIC_CKPT_TO_KEEP="${MAX_CRITIC_CKPT_TO_KEEP:-12}"

cd "${ROOT_DIR}"
exec bash scripts/submit_dlc_gspo_mut_v1_t07_cap2048_mns32.sh
