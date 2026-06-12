#!/usr/bin/env bash
set -euo pipefail

# Experiment B: infra control.
#
# Keep the screened 26k data and RL reward shape unchanged, but disable vLLM
# prefix caching. This tests whether the low-format rollout is caused by the
# high-concurrency shared-prefix serving path rather than data OOD.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOB_NAME="${JOB_NAME:-codevision_gspo_expB_26k_nopc_full_0611}"
export EXP_NAME="${EXP_NAME:-qwen3vl8b_gspo_expB_26k_nopc_full_0611}"
export ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-False}"

# Leave max_num_seqs at the current RL default for this first infra ablation.
# If B still fails, run C with MAX_NUM_SEQS=16 to isolate scheduler pressure.
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"

echo "Submitting Experiment B: 26k data with vLLM prefix caching disabled"
echo "ROLLOUT_ENABLE_PREFIX_CACHING=${ROLLOUT_ENABLE_PREFIX_CACHING}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"

exec bash "${SCRIPT_DIR}/submit_dlc_gspo_before_newdata_26k_t07_cap2048_fmtguard.sh"
