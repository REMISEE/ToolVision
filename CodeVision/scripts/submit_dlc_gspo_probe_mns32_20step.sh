#!/usr/bin/env bash
set -euo pipefail

# Probe: find whether max_num_seqs=32 is still rollout-format safe.
# This is the midpoint after max_num_seqs=16 was clean and 64/256 showed
# heavy malformed tool-call / format failures.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOOL_DLC_HOST="${TOOL_DLC_HOST:-172.17.2.38}"
TOOL_DLC_REPLICA="${TOOL_DLC_REPLICA:-0}"
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

export JOB_NAME="${JOB_NAME:-codevision_gspo_probe_mns32_26k_20step_0615}"
export EXP_NAME="${EXP_NAME:-qwen3vl8b_gspo_probe_mns32_26k_20step_0615}"
export ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-True}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"
export PRIORITY="${PRIORITY:-9}"

# Dense logging for a short probe. Do not save checkpoints; this is only for
# measuring format stability and throughput at the new max_num_seqs value.
export SAVE_FREQ="${SAVE_FREQ:--1}"
export LOG_TRAIN_FREQ="${LOG_TRAIN_FREQ:-2}"
export LOG_TRAIN_GENERATIONS="${LOG_TRAIN_GENERATIONS:-64}"

echo "Submitting max_num_seqs=32 probe, 20 training steps"
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
