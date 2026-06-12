#!/usr/bin/env bash
set -euo pipefail

# Experiment A: data control.
#
# Keep the current RL/vLLM serving configuration unchanged, but replace the
# 26k train data with eval-style benchmark parquets that previously showed
# stable tool-format behavior. This isolates data/OOD effects from infra.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCH_FILES=(
  "/mnt/cpfs/delinmao/Benchmarks/CountQA/countqa_codevision_eval.parquet"
  "/mnt/cpfs/delinmao/Benchmarks/FSC147/fsc147_val_codevision_eval.parquet"
  "/mnt/cpfs/delinmao/Benchmarks/FSC147/fsc147_test_codevision_eval.parquet"
  "/mnt/cpfs/delinmao/Benchmarks/HR-Bench/hr_bench_4k_codevision_eval.parquet"
  "/mnt/cpfs/delinmao/Benchmarks/HR-Bench/hr_bench_8k_codevision_eval.parquet"
  "/mnt/cpfs/delinmao/Benchmarks/vstar-bench/vstar_codevision_eval.parquet"
)

for path in "${BENCH_FILES[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing benchmark parquet: ${path}" >&2
    exit 1
  fi
done

train_files="["
for path in "${BENCH_FILES[@]}"; do
  if [[ "${train_files}" != "[" ]]; then
    train_files+=","
  fi
  train_files+="'${path}'"
done
train_files+="]"

export ALLOW_TRAIN_FILES_OVERRIDE=1
export TRAIN_FILES="${TRAIN_FILES:-${train_files}}"
export JOB_NAME="${JOB_NAME:-codevision_gspo_expA_evalbench_full_0611}"
export EXP_NAME="${EXP_NAME:-qwen3vl8b_gspo_expA_evalbench_full_0611}"

# Keep infra identical to the current 26k run unless explicitly overridden.
export ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-True}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"

echo "Submitting Experiment A: eval-benchmark data control"
echo "TRAIN_FILES=${TRAIN_FILES}"
echo "ROLLOUT_ENABLE_PREFIX_CACHING=${ROLLOUT_ENABLE_PREFIX_CACHING}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"

exec bash "${SCRIPT_DIR}/submit_dlc_gspo_before_newdata_26k_t07_cap2048_fmtguard.sh"
