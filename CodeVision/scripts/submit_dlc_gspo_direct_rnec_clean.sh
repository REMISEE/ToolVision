#!/usr/bin/env bash
set -euo pipefail

# Design B (rnec_with_clean):
# Same training shape as submit_dlc_gspo_direct_rnec.sh, but the reward swaps
# part of the bad-tool penalty for a symmetric clean-tool reward. The intent
# is to remove the GRPO "afraid to use a tool" gradient: under advantage
# normalization, penalizing bad tool use is NOT equivalent to rewarding clean
# tool use — the former makes attempting a tool strictly worse than not
# attempting when outcomes are equal, the latter does not.
#
# Final formula (per sample):
#   R_total = R_acc
#           + 0.2 * R_fmt
#           + 0.3 * R_nec
#           + 0.05 * I(used_tool && !bad_tool_episode)        ← clean-tool bonus (new)
#           - 0.02 * I(bad_tool_episode)                       ← bad-tool penalty (reduced from 0.10)
#           - 0.05 * max(0, code_count - 4)                    ← overuse penalty (unchanged)
#
# bad_tool_episode = (tool_exec_error_count > 0) OR (invalid_tool_call > 0).
# overuse threshold of 4 means tool calls 1-4 are free; the 5th call onward
# is penalized (code_count counts <tool_call> blocks ≈ tool-use turns).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOB_NAME="${JOB_NAME:-codevision_gspo_direct_rnec_clean}"
export EXP_NAME="${EXP_NAME:-qwen3vl8b_gspo_full40k_rnec_clean}"

export TOOL_REWARD_MODE="${TOOL_REWARD_MODE:-rnec_with_clean}"
export FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.2}"
export TOOL_REWARD_BETA="${TOOL_REWARD_BETA:-0.3}"
export TOOL_REWARD_CLEAN_TOOL_WEIGHT="${TOOL_REWARD_CLEAN_TOOL_WEIGHT:-0.05}"
export TOOL_REWARD_TOOL_ERROR_WEIGHT="${TOOL_REWARD_TOOL_ERROR_WEIGHT:-0.02}"
export TOOL_REWARD_INVALID_CALL_WEIGHT="${TOOL_REWARD_INVALID_CALL_WEIGHT:-0.02}"
export TOOL_REWARD_OVERUSE_WEIGHT="${TOOL_REWARD_OVERUSE_WEIGHT:-0.05}"
export TOOL_REWARD_OVERUSE_THRESHOLD="${TOOL_REWARD_OVERUSE_THRESHOLD:-4}"

exec "${SCRIPT_DIR}/submit_dlc_gspo_direct_full.sh"
