#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-smoke}"

CODEVISION_ROOT="${CODEVISION_ROOT:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
LMMS_ROOT="${LMMS_ROOT:-/mnt/cpfs/delinmao/lmms-eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun}"
MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/models/Qwen3-VL-8B-Instruct}"
TASK_PATH="${TASK_PATH:-${CODEVISION_ROOT}/lmms_tasks}"
PORT="${PORT:-12361}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-4}"

export CUDA_VISIBLE_DEVICES

activate_env() {
  source /opt/conda/etc/profile.d/conda.sh
  conda activate lmms-eval
  cd "${LMMS_ROOT}"
}

prepare() {
  local mode="$1"
  local limit_arg=()
  if [[ "${2:-}" != "" ]]; then
    limit_arg=(--limit "$2")
  fi
  cd "${CODEVISION_ROOT}"
  python3 recipe/codevision/tools/prepare_lmms_pass16_rerun.py \
    --sources gqa,textvqa,fsc147 \
    --mode "${mode}" \
    --output-root "${OUTPUT_ROOT}" \
    "${limit_arg[@]}"
}

run_lmms() {
  local tasks="$1"
  local repeats="$2"
  local out_dir="$3"
  local limit_arg=()
  if [[ "${4:-}" != "" ]]; then
    limit_arg=(--limit "$4")
  fi

  activate_env
  accelerate launch --num_processes="${NUM_PROCESSES}" --main_process_port="${PORT}" -m lmms_eval \
    --model qwen3_vl \
    --model_args "pretrained=${MODEL_PATH},max_pixels=6422528,attn_implementation=sdpa,interleave_visuals=False" \
    --include_path "${TASK_PATH}" \
    --tasks "${tasks}" \
    --batch_size "${BATCH_SIZE}" \
    "${limit_arg[@]}" \
    --repeats "${repeats}" \
    --log_samples \
    --output_path "${out_dir}" \
    --verbosity INFO
}

case "${PHASE}" in
  preflight)
    activate_env
    accelerate launch --num_processes="${NUM_PROCESSES}" --main_process_port="${PORT}" -m lmms_eval \
      --model qwen3_vl \
      --model_args "pretrained=${MODEL_PATH},max_pixels=6422528,attn_implementation=sdpa,interleave_visuals=False" \
      --tasks gqa_lite \
      --batch_size "${BATCH_SIZE}" \
      --limit 200 \
      --log_samples \
      --output_path "${OUTPUT_ROOT}/preflight_gqa_lite"
    ;;
  prepare_smoke)
    prepare real 20
    ;;
  smoke)
    run_lmms "tv_pass16_gqa,tv_pass16_textvqa,tv_pass16_fsc147" 2 "${OUTPUT_ROOT}/smoke_qwen3vl8b_instruct" 20
    ;;
  prepare_control)
    prepare real 300
    prepare blank 300
    prepare shuffled 300
    ;;
  control)
    run_lmms "tv_pass16_gqa,tv_pass16_gqa_blank,tv_pass16_gqa_shuffled,tv_pass16_textvqa,tv_pass16_textvqa_blank,tv_pass16_textvqa_shuffled,tv_pass16_fsc147,tv_pass16_fsc147_blank,tv_pass16_fsc147_shuffled" 4 "${OUTPUT_ROOT}/control_qwen3vl8b_instruct" ""
    ;;
  prepare_full)
    prepare real ""
    ;;
  full)
    run_lmms "tv_pass16_gqa,tv_pass16_textvqa,tv_pass16_fsc147" 16 "${OUTPUT_ROOT}/full_qwen3vl8b_instruct" ""
    ;;
  convert)
    cd "${CODEVISION_ROOT}"
    python3 recipe/codevision/tools/convert_lmms_samples_to_pass16.py \
      --input "${OUTPUT_ROOT}/full_qwen3vl8b_instruct" \
      --output "${OUTPUT_ROOT}/full_qwen3vl8b_instruct/pass16_like.parquet" \
      --expected-repeats 16
    PYTHONPATH="${CODEVISION_ROOT}" python3 recipe/codevision/tools/rescore_pass16_v2.py \
      --input "${OUTPUT_ROOT}/full_qwen3vl8b_instruct/pass16_like.parquet" \
      --output "${OUTPUT_ROOT}/full_qwen3vl8b_instruct/pass16_like_v2.parquet" \
      --summary "${OUTPUT_ROOT}/full_qwen3vl8b_instruct/rerun_0602_by_source.csv"
    ;;
  *)
    echo "Unknown phase: ${PHASE}" >&2
    echo "Valid phases: preflight prepare_smoke smoke prepare_control control prepare_full full convert" >&2
    exit 2
    ;;
esac
