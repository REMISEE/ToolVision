#!/usr/bin/env bash
#SBATCH -J tools-eval-all-3gpu
#SBATCH -p a100_global
#SBATCH --gres=gpu:3
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH -t 12:00:00
#SBATCH -o /mnt/cpfs/delinmao/logs/%x-%j.out
#SBATCH -e /mnt/cpfs/delinmao/logs/%x-%j.err

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/cpfs/delinmao}"
ROOT_DIR="${ROOT_DIR:-${WORKSPACE_ROOT}/ToolVision/CodeVision}"
RUN_DIR="${ROOT_DIR}/outputs/slurm_tools_eval_all_3gpu/${SLURM_JOB_ID:-manual}"
SERVICE_LOG_DIR="${RUN_DIR}/service_logs"
SERVICE_PID_DIR="${RUN_DIR}/service_pids"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/tv_ray_3gpu}"

CODEVISION_ENV="${CODEVISION_ENV:-codevision_new}"

BENCHMARK_ROOT="${BENCHMARK_ROOT:-${WORKSPACE_ROOT}/Benchmarks}"
VSTAR_PARQUET_DEFAULT="${BENCHMARK_ROOT}/vstar-bench/vstar_codevision_eval.parquet"
CHARTQA_PARQUET_DEFAULT="${BENCHMARK_ROOT}/ChartQA/chartqa_codevision_eval.parquet"
OCRBENCH_PARQUET_DEFAULT="${BENCHMARK_ROOT}/OCRBench/ocrbench_codevision_eval.parquet"
COUNTBENCH_PARQUET_DEFAULT="${BENCHMARK_ROOT}/countbench/countbench_codevision_eval.parquet"
HRBENCH4K_PARQUET_DEFAULT="${BENCHMARK_ROOT}/HR-Bench/hr_bench_4k_codevision_eval.parquet"
HRBENCH8K_PARQUET_DEFAULT="${BENCHMARK_ROOT}/HR-Bench/hr_bench_8k_codevision_eval.parquet"
FSC147_VAL_PARQUET_DEFAULT="${BENCHMARK_ROOT}/FSC147/fsc147_val_codevision_eval.parquet"
FSC147_TEST_PARQUET_DEFAULT="${BENCHMARK_ROOT}/FSC147/fsc147_test_codevision_eval.parquet"
CVBENCH_PARQUET_DEFAULT="${BENCHMARK_ROOT}/CV-Bench/cvbench_codevision_eval.parquet"
PIXMO_COUNT_PARQUET_DEFAULT="${BENCHMARK_ROOT}/Pixmo-Count/pixmo_count_codevision_eval.parquet"
PIXMO_COUNT_LMMS_PARQUET_DEFAULT="${BENCHMARK_ROOT}/Pixmo-Count-LMMS/pixmo_count_lmms_codevision_eval.parquet"
OCRBENCH_V2_PARQUET_DEFAULT="${BENCHMARK_ROOT}/OCRBench_v2/ocrbench_v2_codevision_eval.parquet"
SPATIALMQA_PARQUET_DEFAULT="${BENCHMARK_ROOT}/SpatialMQA/spatialmqa_codevision_eval.parquet"
COUNTQA_PARQUET_DEFAULT="${BENCHMARK_ROOT}/CountQA/countqa_codevision_eval.parquet"

mkdir -p "${RUN_DIR}" "${SERVICE_LOG_DIR}" "${SERVICE_PID_DIR}" "${RAY_TMPDIR}"

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

conda activate "${CODEVISION_ENV}"

ALLOCATED_GPU_IDS_RAW="${CUDA_VISIBLE_DEVICES:-}"
IFS=',' read -r -a ALLOCATED_GPU_IDS <<< "${ALLOCATED_GPU_IDS_RAW}"
if [[ "${#ALLOCATED_GPU_IDS[@]}" -lt 3 ]]; then
  echo "Expected 3 allocated GPUs, got: ${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
  exit 1
fi

MODEL_GPU_IDS=("${ALLOCATED_GPU_IDS[@]:0:2}")
TOOLS_GPU_PHYSICAL_ID="${TOOLS_GPU_PHYSICAL_ID:-${ALLOCATED_GPU_IDS[2]}}"
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-$(IFS=,; echo "${MODEL_GPU_IDS[*]}")}"

PORT_BASE="${PORT_BASE:-$((26000 + (${SLURM_JOB_ID:-0} % 1000) * 10))}"
OCR_PORT="${OCR_PORT:-${PORT_BASE}}"
GROUNDEDSAM2_PORT="${GROUNDEDSAM2_PORT:-$((PORT_BASE + 1))}"
DEPTH_PORT="${DEPTH_PORT:-$((PORT_BASE + 2))}"
COUNTGD_PORT="${COUNTGD_PORT:-$((PORT_BASE + 3))}"

cleanup() {
  set +e
  cd "${ROOT_DIR}" || exit 0
  bash scripts/launch_external_services.sh stop all
}
trap cleanup EXIT

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

export MODEL_PATH="${MODEL_PATH:-${WORKSPACE_ROOT}/outputs/qwen3vl_sft/full}"
export NGPUS_PER_NODE=2
export INFER_TP_SIZE=2
export VAL_BSZ="${VAL_BSZ:-32}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
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
EXP_PREFIX="${EXP_PREFIX:-tools_a100_2gpu}"

echo "=== Allocated GPUs ==="
nvidia-smi -L
echo "SLURM CUDA_VISIBLE_DEVICES=${ALLOCATED_GPU_IDS_RAW:-<unset>}"
echo "MODEL_CUDA_VISIBLE_DEVICES=${MODEL_CUDA_VISIBLE_DEVICES}"
echo "TOOLS_GPU_PHYSICAL_ID=${TOOLS_GPU_PHYSICAL_ID}"
echo "OCR_PORT=${OCR_PORT} GROUNDEDSAM2_PORT=${GROUNDEDSAM2_PORT} DEPTH_PORT=${DEPTH_PORT} COUNTGD_PORT=${COUNTGD_PORT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "BENCHMARKS=${BENCHMARKS}"
if [[ -n "${LLM_JUDGE_BASE_URL:-}" ]]; then
  echo "LLM_JUDGE_BASE_URL=${LLM_JUDGE_BASE_URL}"
  echo "LLM_JUDGE_MODEL_NAME=${LLM_JUDGE_MODEL_NAME:-<auto>}"
  echo "LLM_JUDGE_API_KEY_SET=$([[ -n "${LLM_JUDGE_API_KEY:-${OPENAI_API_KEY:-}}" ]] && echo yes || echo no)"
fi

echo "=== Starting external services on dedicated tool GPU ==="
bash scripts/launch_external_services.sh start all

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

  bash recipe/codevision/eval_vstar_tools_a100_2gpu.sh
}

IFS=',' read -r -a BENCH_ARRAY <<< "${BENCHMARKS}"
for bench in "${BENCH_ARRAY[@]}"; do
  run_one "${bench}"
done

echo "All requested benchmarks finished."
