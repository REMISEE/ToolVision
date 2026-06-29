#!/usr/bin/env bash
set -euo pipefail

# Probe: run the original 40k RL training data under the same max_num_seqs=32
# rollout setting. This tests whether the old "OOD / harder source" mixture is
# still usable once the vLLM scheduler-pressure issue is controlled.
#
# Use DLC tool replica 1 by default so this can run alongside the 26k mns32
# probe without both jobs hitting one tool replica.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_40K="/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_clean_no_extra_image/parquet/train.parquet"
if [[ ! -f "${TRAIN_40K}" ]]; then
  echo "Missing original 40k train parquet: ${TRAIN_40K}" >&2
  exit 1
fi

TOOL_DLC_HOST="${TOOL_DLC_HOST:-172.17.2.38}"
TOOL_DLC_REPLICA="${TOOL_DLC_REPLICA:-1}"
case "${TOOL_DLC_REPLICA}" in
  0) TOOL_DLC_BASE_PORT=18080 ;;
  1) TOOL_DLC_BASE_PORT=18090 ;;
  *)
    echo "TOOL_DLC_REPLICA must be 0 or 1, got: ${TOOL_DLC_REPLICA}" >&2
    exit 1
    ;;
esac

export OCR_BASE_URL="${OCR_BASE_URL:-http://${TOOL_DLC_HOST}:${TOOL_DLC_BASE_PORT}}"
export GROUNDEDSAM2_BASE_URL="${GROUNDEDSAM2_BASE_URL:-http://${TOOL_DLC_HOST}:$((TOOL_DLC_BASE_PORT + 1))}"
export DEPTH_BASE_URL="${DEPTH_BASE_URL:-http://${TOOL_DLC_HOST}:$((TOOL_DLC_BASE_PORT + 2))}"
export COUNTGD_BASE_URL="${COUNTGD_BASE_URL:-http://${TOOL_DLC_HOST}:$((TOOL_DLC_BASE_PORT + 3))}"

export ALLOW_TRAIN_FILES_OVERRIDE=1
export TRAIN_FILES="${TRAIN_FILES:-['${TRAIN_40K}']}"
export JOB_NAME="${JOB_NAME:-codevision_gspo_probe_mns32_40k_20step_0615}"
export EXP_NAME="${EXP_NAME:-qwen3vl8b_gspo_probe_mns32_40k_20step_0615}"
export ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-True}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"
export PRIORITY="${PRIORITY:-9}"

# Dense logging for a short probe. Do not save checkpoints; this is only for
# measuring format stability and data-distribution effects.
export SAVE_FREQ="${SAVE_FREQ:--1}"
export LOG_TRAIN_FREQ="${LOG_TRAIN_FREQ:-2}"
export LOG_TRAIN_GENERATIONS="${LOG_TRAIN_GENERATIONS:-64}"

echo "Submitting max_num_seqs=32 original-40k probe, 20 training steps"
echo "TRAIN_FILES=${TRAIN_FILES}"
echo "TOOL_DLC_HOST=${TOOL_DLC_HOST}"
echo "TOOL_DLC_REPLICA=${TOOL_DLC_REPLICA}"
echo "OCR_BASE_URL=${OCR_BASE_URL}"
echo "GROUNDEDSAM2_BASE_URL=${GROUNDEDSAM2_BASE_URL}"
echo "DEPTH_BASE_URL=${DEPTH_BASE_URL}"
echo "COUNTGD_BASE_URL=${COUNTGD_BASE_URL}"
echo "ROLLOUT_ENABLE_PREFIX_CACHING=${ROLLOUT_ENABLE_PREFIX_CACHING}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}"
echo "SAVE_FREQ=${SAVE_FREQ}"
echo "PRIORITY=${PRIORITY}"

exec bash "${SCRIPT_DIR}/submit_dlc_gspo_before_newdata_26k_t07_cap2048_fmtguard.sh"
