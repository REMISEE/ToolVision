#!/usr/bin/env bash
#SBATCH -J vstar-tools-eval
#SBATCH -p a100_global
#SBATCH --gres=gpu:5
#SBATCH -c 40
#SBATCH --mem=256G
#SBATCH -t 04:00:00
#SBATCH -o /mnt/cpfs/delinmao/logs/%x-%j.out
#SBATCH -e /mnt/cpfs/delinmao/logs/%x-%j.err

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/cpfs/delinmao}"
ROOT_DIR="${ROOT_DIR:-${WORKSPACE_ROOT}/ToolVision/CodeVision}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/Benchmarks/vstar-bench}"
RUN_DIR="${ROOT_DIR}/outputs/slurm_vstar_tools_eval_5gpu/${SLURM_JOB_ID:-manual}"
SERVICE_LOG_DIR="${RUN_DIR}/service_logs"
SERVICE_PID_DIR="${RUN_DIR}/service_pids"
RAY_TMPDIR="${RAY_TMPDIR:-${WORKSPACE_ROOT}/ray_tmp/vstar_tools_eval_5gpu}"

CODEVISION_ENV="${CODEVISION_ENV:-codevision_new}"

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
if [[ "${#ALLOCATED_GPU_IDS[@]}" -lt 5 ]]; then
  echo "Expected 5 allocated GPUs, got: ${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
  exit 1
fi

MODEL_GPU_IDS=("${ALLOCATED_GPU_IDS[@]:0:4}")
TOOLS_GPU_PHYSICAL_ID="${TOOLS_GPU_PHYSICAL_ID:-${ALLOCATED_GPU_IDS[4]}}"
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-$(IFS=,; echo "${MODEL_GPU_IDS[*]}")}"

PORT_BASE="${PORT_BASE:-$((24000 + (${SLURM_JOB_ID:-0} % 1000) * 10))}"
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

EVAL_PARQUET="${EVAL_PARQUET:-${VSTAR_PARQUET:-${DATA_ROOT}/vstar_codevision_eval.parquet}}"
MODEL_PATH="${MODEL_PATH:-${WORKSPACE_ROOT}/outputs/qwen3vl_sft/full}"

echo "=== Allocated GPUs ==="
nvidia-smi -L
echo "SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "MODEL_CUDA_VISIBLE_DEVICES=${MODEL_CUDA_VISIBLE_DEVICES}"
echo "TOOLS_GPU_PHYSICAL_ID=${TOOLS_GPU_PHYSICAL_ID}"
echo "OCR_PORT=${OCR_PORT} GROUNDEDSAM2_PORT=${GROUNDEDSAM2_PORT} DEPTH_PORT=${DEPTH_PORT} COUNTGD_PORT=${COUNTGD_PORT}"
echo "EVAL_PARQUET=${EVAL_PARQUET}"
echo "MODEL_PATH=${MODEL_PATH}"
if [[ -n "${LLM_JUDGE_BASE_URL:-}" ]]; then
  echo "LLM_JUDGE_BASE_URL=${LLM_JUDGE_BASE_URL}"
  echo "LLM_JUDGE_MODEL_NAME=${LLM_JUDGE_MODEL_NAME:-<auto>}"
  echo "LLM_JUDGE_API_KEY_SET=$([[ -n "${LLM_JUDGE_API_KEY:-${OPENAI_API_KEY:-}}" ]] && echo yes || echo no)"
fi

echo "=== Starting external services on dedicated tool GPU ==="
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
bash scripts/launch_external_services.sh start all

echo "=== Running tool-enabled V*Bench eval on 4 model GPUs ==="
export OCR_BASE_URL="http://127.0.0.1:${OCR_PORT}"
export GROUNDEDSAM2_BASE_URL="http://127.0.0.1:${GROUNDEDSAM2_PORT}"
export DEPTH_BASE_URL="http://127.0.0.1:${DEPTH_PORT}"
export COUNTGD_BASE_URL="http://127.0.0.1:${COUNTGD_PORT}"
export EVAL_PARQUET
export MODEL_PATH
export NGPUS_PER_NODE=4
export INFER_TP_SIZE=4
export VAL_BSZ="${VAL_BSZ:-32}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
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

bash recipe/codevision/eval_vstar_tools_a100_4gpu.sh
