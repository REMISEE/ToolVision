#!/usr/bin/env bash
# Wait for an SFT model and free GPUs, then run all tool-enabled CodeVision evals.
#
# Default topology:
#   - 4 GPUs for the evaluation model / vLLM tensor parallel
#   - 1 GPU for external tool services
#
# Example:
#   cd /mnt/cpfs/delinmao/ToolVision/CodeVision
#   nohup bash scripts/run_tools_eval_all_wait_5gpu_nohup.sh > /mnt/cpfs/delinmao/eval_wait.log 2>&1 &

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/cpfs/delinmao}"
ROOT_DIR="${ROOT_DIR:-${WORKSPACE_ROOT}/ToolVision/CodeVision}"
CODEVISION_ENV="${CODEVISION_ENV:-${WORKSPACE_ROOT}/envs/codevision_new}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${WORKSPACE_ROOT}/Benchmarks}"

MODEL_PATH="${MODEL_PATH:-${WORKSPACE_ROOT}/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-drop-simple-notool}"
WAIT_FOR_MODEL="${WAIT_FOR_MODEL:-1}"
MODEL_WAIT_INTERVAL_S="${MODEL_WAIT_INTERVAL_S:-60}"
MAX_MODEL_WAIT_MINUTES="${MAX_MODEL_WAIT_MINUTES:-0}"

NUM_MODEL_GPUS="${NUM_MODEL_GPUS:-4}"
NUM_TOOL_GPUS="${NUM_TOOL_GPUS:-1}"
NUM_GPUS=$((NUM_MODEL_GPUS + NUM_TOOL_GPUS))
MAX_USED_MEM_MB="${MAX_USED_MEM_MB:-1000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-5}"
GPU_WAIT_INTERVAL_S="${GPU_WAIT_INTERVAL_S:-60}"
MAX_GPU_WAIT_MINUTES="${MAX_GPU_WAIT_MINUTES:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/outputs/nohup_tools_eval_all_${NUM_MODEL_GPUS}model_${NUM_TOOL_GPUS}tool/${TS}}"
SERVICE_LOG_DIR="${RUN_DIR}/service_logs"
SERVICE_PID_DIR="${RUN_DIR}/service_pids"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/tv_ray_${NUM_MODEL_GPUS}m_${NUM_TOOL_GPUS}t}"
LOCK_DIR="${LOCK_DIR:-${ROOT_DIR}/outputs/.tools_eval_all_${NUM_MODEL_GPUS}model_${NUM_TOOL_GPUS}tool.lock}"

VSTAR_PARQUET_DEFAULT="${BENCHMARK_ROOT}/vstar-bench/vstar_codevision_eval.parquet"
CHARTQA_PARQUET_DEFAULT="${BENCHMARK_ROOT}/ChartQA/chartqa_codevision_eval.parquet"
OCRBENCH_PARQUET_DEFAULT="${BENCHMARK_ROOT}/OCRBench/ocrbench_codevision_eval.parquet"
COUNTBENCH_PARQUET_DEFAULT="${BENCHMARK_ROOT}/countbench/countbench_codevision_eval.parquet"
HRBENCH4K_PARQUET_DEFAULT="${BENCHMARK_ROOT}/HR-Bench/hr_bench_4k_codevision_eval.parquet"
HRBENCH8K_PARQUET_DEFAULT="${BENCHMARK_ROOT}/HR-Bench/hr_bench_8k_codevision_eval.parquet"
MVTOOLBENCH_PARQUET_DEFAULT="${BENCHMARK_ROOT}/MVToolBench/mvtoolbench_codevision_eval.parquet"
FSC147_VAL_PARQUET_DEFAULT="${BENCHMARK_ROOT}/FSC147/fsc147_val_codevision_eval.parquet"
FSC147_TEST_PARQUET_DEFAULT="${BENCHMARK_ROOT}/FSC147/fsc147_test_codevision_eval.parquet"
CVBENCH_PARQUET_DEFAULT="${BENCHMARK_ROOT}/CV-Bench/cvbench_codevision_eval.parquet"
PIXMO_COUNT_PARQUET_DEFAULT="${BENCHMARK_ROOT}/Pixmo-Count/pixmo_count_codevision_eval.parquet"
PIXMO_COUNT_LMMS_PARQUET_DEFAULT="${BENCHMARK_ROOT}/Pixmo-Count-LMMS/pixmo_count_lmms_codevision_eval.parquet"
OCRBENCH_V2_PARQUET_DEFAULT="${BENCHMARK_ROOT}/OCRBench_v2/ocrbench_v2_codevision_eval.parquet"
SPATIALMQA_PARQUET_DEFAULT="${BENCHMARK_ROOT}/SpatialMQA/spatialmqa_codevision_eval.parquet"
COUNTQA_PARQUET_DEFAULT="${BENCHMARK_ROOT}/CountQA/countqa_codevision_eval.parquet"
DOCVQA_PARQUET_DEFAULT="${BENCHMARK_ROOT}/DocVQA/docvqa_val_codevision_eval.parquet"
DOCVQA_TEST_PARQUET_DEFAULT="${BENCHMARK_ROOT}/DocVQA/docvqa_test_codevision_eval.parquet"
INFOVQA_PARQUET_DEFAULT="${BENCHMARK_ROOT}/InfoVQA/infovqa_val_codevision_eval.parquet"
INFOVQA_TEST_PARQUET_DEFAULT="${BENCHMARK_ROOT}/InfoVQA/infovqa_test_codevision_eval.parquet"
MME_REALWORLD_PARQUET_DEFAULT="${BENCHMARK_ROOT}/MME-RealWorld/mme_realworld_codevision_eval.parquet"
MME_REALWORLD_LITE_PARQUET_DEFAULT="${BENCHMARK_ROOT}/MME-RealWorld-Lite/mme_realworld_lite_codevision_eval.parquet"
MME_REALWORLD_CN_PARQUET_DEFAULT="${BENCHMARK_ROOT}/MME-RealWorld-CN/mme_realworld_cn_codevision_eval.parquet"
REALWORLDQA_PARQUET_DEFAULT="${BENCHMARK_ROOT}/RealWorldQA/realworldqa_codevision_eval.parquet"
MMSTAR_PARQUET_DEFAULT="${BENCHMARK_ROOT}/MMStar/mmstar_codevision_eval.parquet"
MMVET_PARQUET_DEFAULT="${BENCHMARK_ROOT}/MMVet/mmvet_codevision_eval.parquet"
MMVET_HARD_PARQUET_DEFAULT="${BENCHMARK_ROOT}/MMVet-Hard/mmvet_hard_codevision_eval.parquet"

mkdir -p "${RUN_DIR}" "${SERVICE_LOG_DIR}" "${SERVICE_PID_DIR}" "${RAY_TMPDIR}"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  lock_pid=""
  if [[ -f "${LOCK_DIR}/launcher.pid" ]]; then
    lock_pid="$(cat "${LOCK_DIR}/launcher.pid" 2>/dev/null || true)"
  fi
  if [[ -n "${lock_pid}" ]] && ! kill -0 "${lock_pid}" 2>/dev/null; then
    echo "Removing stale eval launcher lock: ${LOCK_DIR} (pid=${lock_pid})" >&2
    rm -rf "${LOCK_DIR}"
    mkdir "${LOCK_DIR}"
  else
  echo "Another eval launcher appears to be active: ${LOCK_DIR}" >&2
  echo "Remove this lock directory only if that launcher is gone." >&2
  exit 1
  fi
fi
trap 'set +e; cd "${ROOT_DIR}" 2>/dev/null && bash scripts/launch_external_services.sh stop all; rm -rf "${LOCK_DIR}"' EXIT
echo "$$" > "${LOCK_DIR}/launcher.pid"

source_conda() {
  CONDA_SH_PATH="${CONDA_SH_PATH:-}"
  if [[ -n "${CONDA_SH_PATH}" && -f "${CONDA_SH_PATH}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH_PATH}"
  elif [[ -f "/mnt/public/apps/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "/mnt/public/apps/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "/opt/conda/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  else
    echo "conda.sh not found; set CONDA_SH_PATH explicitly" >&2
    exit 1
  fi
}

model_ready() {
  [[ -f "${MODEL_PATH}/config.json" ]] &&
    [[ -f "${MODEL_PATH}/model.safetensors.index.json" ]] &&
    [[ -f "${MODEL_PATH}/tokenizer_config.json" ]]
}

wait_for_model() {
  [[ "${WAIT_FOR_MODEL}" == "1" ]] || return 0
  local start_ts now elapsed_min
  start_ts="$(date +%s)"
  while ! model_ready; do
    now="$(date +%s)"
    elapsed_min="$(((now - start_ts) / 60))"
    echo "[$(date '+%F %T')] waiting for SFT model: ${MODEL_PATH}" >&2
    echo "  need config.json, model.safetensors.index.json, tokenizer_config.json" >&2
    if [[ "${MAX_MODEL_WAIT_MINUTES}" != "0" && "${elapsed_min}" -ge "${MAX_MODEL_WAIT_MINUTES}" ]]; then
      echo "Timed out after ${MAX_MODEL_WAIT_MINUTES} minutes waiting for model." >&2
      exit 1
    fi
    sleep "${MODEL_WAIT_INTERVAL_S}"
  done
  echo "[$(date '+%F %T')] model is ready: ${MODEL_PATH}" >&2
}

candidate_gpus() {
  if [[ -n "${GPU_CANDIDATES:-}" ]]; then
    tr ',' '\n' <<< "${GPU_CANDIDATES}" | sed '/^$/d'
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | sed '/^$/d'
  else
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits
  fi
}

free_gpus() {
  local candidates
  candidates="$(candidate_gpus | tr '\n' ' ')"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F ', *' -v candidates="${candidates}" -v max_mem="${MAX_USED_MEM_MB}" -v max_util="${MAX_GPU_UTIL}" '
      BEGIN {
        split(candidates, arr, " ")
        for (i in arr) {
          if (arr[i] != "") allowed[arr[i]] = 1
        }
      }
      ($1 in allowed) && ($2 <= max_mem) && ($3 <= max_util) { print $1 }
    '
}

select_gpus_or_wait() {
  local start_ts now elapsed_min selected count
  start_ts="$(date +%s)"
  while true; do
    selected="$(free_gpus | head -n "${NUM_GPUS}" | paste -sd, -)"
    if [[ -n "${selected}" ]]; then
      count="$(tr ',' '\n' <<< "${selected}" | sed '/^$/d' | wc -l)"
    else
      count=0
    fi
    if [[ "${count}" -ge "${NUM_GPUS}" ]]; then
      echo "${selected}"
      return 0
    fi

    now="$(date +%s)"
    elapsed_min="$(((now - start_ts) / 60))"
    echo "[$(date '+%F %T')] waiting for ${NUM_GPUS} free GPUs; found ${count}. candidates=$(candidate_gpus | paste -sd, -) threshold=mem<=${MAX_USED_MEM_MB}MB util<=${MAX_GPU_UTIL}%" >&2
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >&2 || true
    if [[ "${MAX_GPU_WAIT_MINUTES}" != "0" && "${elapsed_min}" -ge "${MAX_GPU_WAIT_MINUTES}" ]]; then
      echo "Timed out after ${MAX_GPU_WAIT_MINUTES} minutes waiting for GPUs." >&2
      exit 1
    fi
    sleep "${GPU_WAIT_INTERVAL_S}"
  done
}

run_one() {
  local bench="$1"
  local parquet=""
  local exp_name=""
  local n_resp="${DEFAULT_N_RESP_PER_PROMPT}"
  case "${bench}" in
    vstar)
      parquet="${VSTAR_PARQUET:-${VSTAR_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_vstar"
      ;;
    chartqa)
      parquet="${CHARTQA_PARQUET:-${CHARTQA_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_chartqa"
      ;;
    ocrbench)
      parquet="${OCRBENCH_PARQUET:-${OCRBENCH_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_ocrbench"
      ;;
    countbench)
      parquet="${COUNTBENCH_PARQUET:-${COUNTBENCH_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_countbench"
      ;;
    hrbench4k)
      parquet="${HRBENCH4K_PARQUET:-${HRBENCH4K_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_hrbench4k"
      ;;
    hrbench8k)
      parquet="${HRBENCH8K_PARQUET:-${HRBENCH8K_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_hrbench8k"
      ;;
    mvtoolbench)
      parquet="${MVTOOLBENCH_PARQUET:-${MVTOOLBENCH_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_mvtoolbench"
      ;;
    fsc147_val)
      parquet="${FSC147_VAL_PARQUET:-${FSC147_VAL_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_fsc147_val"
      n_resp="${FSC147_N_RESP_PER_PROMPT}"
      ;;
    fsc147_test)
      parquet="${FSC147_TEST_PARQUET:-${FSC147_TEST_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_fsc147_test"
      n_resp="${FSC147_N_RESP_PER_PROMPT}"
      ;;
    cvbench)
      parquet="${CVBENCH_PARQUET:-${CVBENCH_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_cvbench"
      ;;
    pixmo_count)
      parquet="${PIXMO_COUNT_PARQUET:-${PIXMO_COUNT_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_pixmo_count"
      ;;
    pixmo_count_lmms)
      parquet="${PIXMO_COUNT_LMMS_PARQUET:-${PIXMO_COUNT_LMMS_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_pixmo_count_lmms"
      ;;
    ocrbench_v2)
      parquet="${OCRBENCH_V2_PARQUET:-${OCRBENCH_V2_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_ocrbench_v2"
      ;;
    spatialmqa)
      parquet="${SPATIALMQA_PARQUET:-${SPATIALMQA_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_spatialmqa"
      ;;
    countqa)
      parquet="${COUNTQA_PARQUET:-${COUNTQA_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_countqa"
      ;;
    docvqa|docvqa_val)
      parquet="${DOCVQA_PARQUET:-${DOCVQA_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_docvqa_val"
      ;;
    docvqa_test)
      parquet="${DOCVQA_TEST_PARQUET:-${DOCVQA_TEST_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_docvqa_test"
      ;;
    infovqa|infovqa_val)
      parquet="${INFOVQA_PARQUET:-${INFOVQA_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_infovqa_val"
      ;;
    infovqa_test)
      parquet="${INFOVQA_TEST_PARQUET:-${INFOVQA_TEST_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_infovqa_test"
      ;;
    mme_realworld|mmerealworld)
      parquet="${MME_REALWORLD_PARQUET:-${MME_REALWORLD_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_mme_realworld"
      ;;
    mme_realworld_lite|mmerealworld_lite)
      parquet="${MME_REALWORLD_LITE_PARQUET:-${MME_REALWORLD_LITE_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_mme_realworld_lite"
      ;;
    mme_realworld_cn|mmerealworld_cn)
      parquet="${MME_REALWORLD_CN_PARQUET:-${MME_REALWORLD_CN_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_mme_realworld_cn"
      ;;
    realworldqa)
      parquet="${REALWORLDQA_PARQUET:-${REALWORLDQA_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_realworldqa"
      ;;
    mmstar)
      parquet="${MMSTAR_PARQUET:-${MMSTAR_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_mmstar"
      ;;
    mmvet)
      parquet="${MMVET_PARQUET:-${MMVET_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_mmvet"
      ;;
    mmvet_hard)
      parquet="${MMVET_HARD_PARQUET:-${MMVET_HARD_PARQUET_DEFAULT}}"
      exp_name="${EXP_PREFIX}_mmvet_hard"
      ;;
    *)
      echo "Unknown benchmark: ${bench}" >&2
      return 1
      ;;
  esac

  if [[ ! -f "${parquet}" ]]; then
    echo "Missing parquet for ${bench}: ${parquet}" >&2
    return 1
  fi

  echo
  echo "=== Running ${bench} ==="
  echo "EVAL_PARQUET=${parquet}"
  echo "EXP_NAME=${exp_name}"
  echo "N_RESP_PER_PROMPT=${n_resp}"

  export OCR_BASE_URL="http://127.0.0.1:${OCR_PORT}"
  export GROUNDEDSAM2_BASE_URL="http://127.0.0.1:${GROUNDEDSAM2_PORT}"
  export DEPTH_BASE_URL="http://127.0.0.1:${DEPTH_PORT}"
  export COUNTGD_BASE_URL="http://127.0.0.1:${COUNTGD_PORT}"
  export EVAL_PARQUET="${parquet}"
  export EXP_NAME="${exp_name}"
  export N_RESP_PER_PROMPT="${n_resp}"

  bash recipe/codevision/eval_vstar_tools_a100_4gpu.sh
}

source_conda
conda activate "${CODEVISION_ENV}"

wait_for_model
ALLOCATED_GPU_IDS_RAW="$(select_gpus_or_wait)"
IFS=',' read -r -a ALLOCATED_GPU_IDS <<< "${ALLOCATED_GPU_IDS_RAW}"
MODEL_GPU_IDS=("${ALLOCATED_GPU_IDS[@]:0:${NUM_MODEL_GPUS}}")
TOOL_GPU_INDEX="${NUM_MODEL_GPUS}"
TOOLS_GPU_PHYSICAL_ID="${TOOLS_GPU_PHYSICAL_ID:-${ALLOCATED_GPU_IDS[${TOOL_GPU_INDEX}]}}"
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-$(IFS=,; echo "${MODEL_GPU_IDS[*]}")}"

PORT_BASE="${PORT_BASE:-$((26000 + ($$ % 1000) * 10))}"
OCR_PORT="${OCR_PORT:-${PORT_BASE}}"
GROUNDEDSAM2_PORT="${GROUNDEDSAM2_PORT:-$((PORT_BASE + 1))}"
DEPTH_PORT="${DEPTH_PORT:-$((PORT_BASE + 2))}"
COUNTGD_PORT="${COUNTGD_PORT:-$((PORT_BASE + 3))}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}"
export SERVICE_LOG_DIR
export SERVICE_PID_DIR
export OCR_PORT GROUNDEDSAM2_PORT DEPTH_PORT COUNTGD_PORT
export OCR_CUDA_VISIBLE_DEVICES="${TOOLS_GPU_PHYSICAL_ID}"
export GROUNDEDSAM2_CUDA_VISIBLE_DEVICES="${TOOLS_GPU_PHYSICAL_ID}"
export DEPTH_CUDA_VISIBLE_DEVICES="${TOOLS_GPU_PHYSICAL_ID}"
export COUNTGD_CUDA_VISIBLE_DEVICES="${TOOLS_GPU_PHYSICAL_ID}"
export DEPTH_GROUNDEDSAM2_BASE_URL="http://127.0.0.1:${GROUNDEDSAM2_PORT}"

export MODEL_PATH
export NGPUS_PER_NODE="${NUM_MODEL_GPUS}"
export INFER_TP_SIZE="${NUM_MODEL_GPUS}"
export VAL_BSZ="${VAL_BSZ:-32}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
DEFAULT_N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
FSC147_N_RESP_PER_PROMPT="${FSC147_N_RESP_PER_PROMPT:-1}"
export N_RESP_PER_PROMPT="${DEFAULT_N_RESP_PER_PROMPT}"
export MAX_TURNS="${MAX_TURNS:-12}"
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
unset RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES
unset RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export LLM_JUDGE_TRUST_ENV="${LLM_JUDGE_TRUST_ENV:-0}"
export RAY_TMPDIR
export CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES}"

BENCHMARKS="${BENCHMARKS:-vstar,chartqa,ocrbench,countbench,hrbench4k,hrbench8k,fsc147_val,fsc147_test}"
EXP_PREFIX="${EXP_PREFIX:-sft_drop_simple_notool_${NUM_MODEL_GPUS}gpu}"

echo "=== Nohup tool eval all ==="
echo "ROOT_DIR=${ROOT_DIR}"
echo "RUN_DIR=${RUN_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "ALLOCATED_GPU_IDS=${ALLOCATED_GPU_IDS_RAW}"
echo "MODEL_CUDA_VISIBLE_DEVICES=${MODEL_CUDA_VISIBLE_DEVICES}"
echo "TOOLS_GPU_PHYSICAL_ID=${TOOLS_GPU_PHYSICAL_ID}"
echo "OCR_PORT=${OCR_PORT} GROUNDEDSAM2_PORT=${GROUNDEDSAM2_PORT} DEPTH_PORT=${DEPTH_PORT} COUNTGD_PORT=${COUNTGD_PORT}"
echo "BENCHMARKS=${BENCHMARKS}"
echo "VAL_BSZ=${VAL_BSZ} MAX_NUM_SEQS=${MAX_NUM_SEQS} GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
if [[ -n "${LLM_JUDGE_BASE_URL:-}" ]]; then
  echo "LLM_JUDGE_BASE_URL=${LLM_JUDGE_BASE_URL}"
  echo "LLM_JUDGE_MODEL_NAME=${LLM_JUDGE_MODEL_NAME:-<auto>}"
  echo "LLM_JUDGE_API_KEY_SET=$([[ -n "${LLM_JUDGE_API_KEY:-${OPENAI_API_KEY:-}}" ]] && echo yes || echo no)"
fi

echo "=== Starting external services on dedicated tool GPU ==="
bash scripts/launch_external_services.sh start all

IFS=',' read -r -a BENCH_ARRAY <<< "${BENCHMARKS}"
for bench in "${BENCH_ARRAY[@]}"; do
  run_one "${bench}"
done

echo "All requested benchmarks finished."
