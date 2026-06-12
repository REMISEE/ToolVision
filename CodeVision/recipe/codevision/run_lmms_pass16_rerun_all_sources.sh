#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-prepare_smoke}"

CODEVISION_ROOT="${CODEVISION_ROOT:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
LMMS_ROOT="${LMMS_ROOT:-/mnt/cpfs/delinmao/lmms-eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all}"
MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/models/Qwen3-VL-8B-Instruct}"
TASK_PATH="${TASK_PATH:-${CODEVISION_ROOT}/lmms_tasks}"
PORT="${PORT:-12362}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
VLLM_MAX_PIXELS="${VLLM_MAX_PIXELS:-6422528}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_SMOKE_OUTPUT_DIR="${VLLM_SMOKE_OUTPUT_DIR:-${OUTPUT_ROOT}/vllm_smoke_qwen3vl8b_instruct}"
VLLM_OUTPUT_DIR="${VLLM_OUTPUT_DIR:-${OUTPUT_ROOT}/vllm_full_qwen3vl8b_instruct}"
TV_DROP_IMAGE_IF_RAW_PIXELS_GT="${TV_DROP_IMAGE_IF_RAW_PIXELS_GT:-0}"
TV_BLANK_IMAGE_SIZE="${TV_BLANK_IMAGE_SIZE:-64}"
SOURCES="${SOURCES:-gqa,textvqa,fsc147,ai2d,arxivqa,chartqa,countqa,docvqa,infographicvqa,mmstar,ocrbench,pixmo_count,refl4,sat2,virgorlsa}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/cpfs/delinmao/envs/lmms-eval/bin/python3}"
LMMS_CONDA_ENV="${LMMS_CONDA_ENV:-lmms-eval}"
VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-codevision}"

export CUDA_VISIBLE_DEVICES
export SOURCES
export TV_DROP_IMAGE_IF_RAW_PIXELS_GT
export TV_BLANK_IMAGE_SIZE

tasks_from_sources() {
  python3 - <<'PY'
import os
sources = [s.strip() for s in os.environ["SOURCES"].split(",") if s.strip()]
print(",".join(f"tv_pass16_{source}" for source in sources))
PY
}

activate_env() {
  local env_name="${1:-${LMMS_CONDA_ENV}}"
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${env_name}"
  cd "${LMMS_ROOT}"
}

prepare() {
  local mode="$1"
  local limit_arg=()
  if [[ "${2:-}" != "" ]]; then
    limit_arg=(--limit "$2")
  fi
  cd "${CODEVISION_ROOT}"
  "${PYTHON_BIN}" recipe/codevision/tools/prepare_lmms_pass16_all_sources.py \
    --sources "${SOURCES}" \
    --mode "${mode}" \
    --output-root "${OUTPUT_ROOT}" \
    "${limit_arg[@]}"
}

run_lmms() {
  local repeats="$1"
  local out_dir="$2"
  local limit_arg=()
  if [[ "${3:-}" != "" ]]; then
    limit_arg=(--limit "$3")
  fi
  local tasks
  tasks="$(tasks_from_sources)"

  activate_env "${LMMS_CONDA_ENV}"
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

run_lmms_vllm() {
  local repeats="$1"
  local out_dir="$2"
  local limit_arg=()
  if [[ "${3:-}" != "" ]]; then
    limit_arg=(--limit "$3")
  fi
  local tasks
  tasks="$(tasks_from_sources)"

  activate_env "${VLLM_CONDA_ENV}"
  export PYTHONPATH="${LMMS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  WORKERS="${WORKERS:-32}" accelerate launch --num_processes="${NUM_PROCESSES}" --main_process_port="${PORT}" -m lmms_eval \
    --model vllm_generate \
    --model_args "model=${MODEL_PATH},tensor_parallel_size=1,data_parallel_size=${NUM_PROCESSES},gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION},disable_log_stats=True,max_pixels=${VLLM_MAX_PIXELS},max_model_len=${VLLM_MAX_MODEL_LEN},max_num_seqs=${VLLM_MAX_NUM_SEQS},max_new_tokens=1" \
    --include_path "${TASK_PATH}" \
    --tasks "${tasks}" \
    --batch_size "${VLLM_BATCH_SIZE}" \
    "${limit_arg[@]}" \
    --repeats "${repeats}" \
    --log_samples \
    --output_path "${out_dir}" \
    --verbosity INFO
}

case "${PHASE}" in
  prepare_smoke)
    prepare real 20
    ;;
  smoke)
    run_lmms 2 "${OUTPUT_ROOT}/smoke_qwen3vl8b_instruct" 20
    ;;
  prepare_control)
    prepare real 300
    prepare blank 300
    prepare shuffled 300
    ;;
  prepare_full)
    prepare real ""
    ;;
  full)
    run_lmms 16 "${OUTPUT_ROOT}/full_qwen3vl8b_instruct" ""
    ;;
  vllm_smoke)
    run_lmms_vllm 16 "${VLLM_SMOKE_OUTPUT_DIR}" 100
    ;;
  vllm_full)
    run_lmms_vllm 16 "${VLLM_OUTPUT_DIR}" ""
    ;;
  convert)
    cd "${CODEVISION_ROOT}"
    python3 recipe/codevision/tools/convert_lmms_samples_to_pass16.py \
      --input "${OUTPUT_ROOT}/full_qwen3vl8b_instruct" \
      --output "${OUTPUT_ROOT}/full_qwen3vl8b_instruct/pass16_like.parquet" \
      --reference-input-dir "${OUTPUT_ROOT}/inputs" \
      --expected-repeats 16
    PYTHONPATH="${CODEVISION_ROOT}" python3 recipe/codevision/tools/rescore_pass16_v2.py \
      --input "${OUTPUT_ROOT}/full_qwen3vl8b_instruct/pass16_like.parquet" \
      --output "${OUTPUT_ROOT}/full_qwen3vl8b_instruct/pass16_like_v2.parquet" \
      --summary "${OUTPUT_ROOT}/full_qwen3vl8b_instruct/all_sources_by_source.csv"
    ;;
  vllm_convert)
    cd "${CODEVISION_ROOT}"
    python3 recipe/codevision/tools/convert_lmms_samples_to_pass16.py \
      --input "${VLLM_OUTPUT_DIR}" \
      --output "${VLLM_OUTPUT_DIR}/pass16_like.parquet" \
      --reference-input-dir "${OUTPUT_ROOT}/inputs" \
      --expected-repeats 16
    PYTHONPATH="${CODEVISION_ROOT}" python3 recipe/codevision/tools/rescore_pass16_v2.py \
      --input "${VLLM_OUTPUT_DIR}/pass16_like.parquet" \
      --output "${VLLM_OUTPUT_DIR}/pass16_like_v2.parquet" \
      --summary "${VLLM_OUTPUT_DIR}/by_source.csv"
    ;;
  *)
    echo "Unknown phase: ${PHASE}" >&2
    echo "Valid phases: prepare_smoke smoke prepare_control prepare_full full vllm_smoke vllm_full convert vllm_convert" >&2
    exit 2
    ;;
esac
