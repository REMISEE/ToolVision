#!/usr/bin/env bash
set -euo pipefail

# Probe: find a faster safe max_num_seqs setting after expC showed
# max_num_seqs=16 fixes RL rollout format. Keep prefix caching enabled and
# use the screened 26k data; only raise max_num_seqs to 64.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOB_NAME="${JOB_NAME:-codevision_gspo_probe_mns64_20step_0612}"
export EXP_NAME="${EXP_NAME:-qwen3vl8b_gspo_probe_mns64_20step_0612}"
export ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-True}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"

# Dense logging for a short probe. Do not save checkpoints; this is only for
# measuring format stability and throughput at the new max_num_seqs value.
export SAVE_FREQ="${SAVE_FREQ:--1}"
export LOG_TRAIN_FREQ="${LOG_TRAIN_FREQ:-2}"
export LOG_TRAIN_GENERATIONS="${LOG_TRAIN_GENERATIONS:-64}"

echo "Submitting max_num_seqs=64 probe, 20 training steps"
echo "ROLLOUT_ENABLE_PREFIX_CACHING=${ROLLOUT_ENABLE_PREFIX_CACHING}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}"
echo "SAVE_FREQ=${SAVE_FREQ}"

exec bash "${SCRIPT_DIR}/submit_dlc_gspo_before_newdata_26k_t07_cap2048_fmtguard.sh"
