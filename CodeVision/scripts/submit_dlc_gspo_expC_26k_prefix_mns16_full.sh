#!/usr/bin/env bash
set -euo pipefail

# Experiment C: scheduler-pressure control.
#
# Keep the screened 26k data and vLLM prefix caching enabled, but reduce
# max_num_seqs to the eval-style setting. This isolates high-concurrency
# scheduling pressure from prefix-cache behavior.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOB_NAME="${JOB_NAME:-codevision_gspo_expC_26k_prefix_mns16_full_0611}"
export EXP_NAME="${EXP_NAME:-qwen3vl8b_gspo_expC_26k_prefix_mns16_full_0611}"
export ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-True}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

echo "Submitting Experiment C: 26k data with prefix caching enabled and max_num_seqs=16"
echo "ROLLOUT_ENABLE_PREFIX_CACHING=${ROLLOUT_ENABLE_PREFIX_CACHING}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"

exec bash "${SCRIPT_DIR}/submit_dlc_gspo_before_newdata_26k_t07_cap2048_fmtguard.sh"
