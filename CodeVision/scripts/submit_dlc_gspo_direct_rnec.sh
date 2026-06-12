#!/usr/bin/env bash
set -euo pipefail

# Strategy A: keep the full RL training shape unchanged and only ablate the
# reward. This enables R_nec, merges tool-error/invalid-call into one light
# penalty, and does not add the tool-correct fallback bonus from Strategy B.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOB_NAME="${JOB_NAME:-codevision_gspo_direct_rnec}"
export EXP_NAME="${EXP_NAME:-qwen3vl8b_gspo_full40k_rnec}"

export TOOL_REWARD_MODE="${TOOL_REWARD_MODE:-rnec_only}"
export FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.2}"
export TOOL_REWARD_BETA="${TOOL_REWARD_BETA:-0.3}"
export TOOL_REWARD_OVERUSE_WEIGHT="${TOOL_REWARD_OVERUSE_WEIGHT:-0.05}"
export TOOL_REWARD_TOOL_ERROR_WEIGHT="${TOOL_REWARD_TOOL_ERROR_WEIGHT:-0.1}"
export TOOL_REWARD_INVALID_CALL_WEIGHT="${TOOL_REWARD_INVALID_CALL_WEIGHT:-0.1}"
export TOOL_REWARD_OVERUSE_THRESHOLD="${TOOL_REWARD_OVERUSE_THRESHOLD:-4}"

exec "${SCRIPT_DIR}/submit_dlc_gspo_direct_full.sh"
