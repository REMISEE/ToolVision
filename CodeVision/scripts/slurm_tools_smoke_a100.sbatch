#!/usr/bin/env bash
#SBATCH -J cvtools-smoke
#SBATCH -p a100_global
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /mnt/cpfs/delinmao/logs/%x-%j.out
#SBATCH -e /mnt/cpfs/delinmao/logs/%x-%j.err

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/cpfs/delinmao}"
REPO_ROOT="${TOOLVISION_ROOT:-${WORKSPACE_ROOT}/ToolVision}"
ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/CodeVision}"
RUN_DIR="${ROOT_DIR}/outputs/slurm_smoke/${SLURM_JOB_ID:-manual}"
SERVICE_LOG_DIR="${RUN_DIR}/service_logs"
SERVICE_PID_DIR="${RUN_DIR}/service_pids"
TEXT_IMAGE_PATH="${RUN_DIR}/smoke_text.png"
OCR_DEMO_OUT_DIR="${RUN_DIR}/ocr_demo"
TOOL_DEMO_OUT_DIR="${RUN_DIR}/tool_demo"
TOOL_IMAGE_PATH="${REPO_ROOT}/offline_sft_pipeline/example/image_0.png"

OCR_ENV="${OCR_ENV:-paddleocr}"
GROUNDEDSAM2_ENV="${GROUNDEDSAM2_ENV:-groundedsam2}"
DEPTH_ENV="${DEPTH_ENV:-depth-pro}"
COUNTGD_ENV="${COUNTGD_ENV:-countgd}"
CODEVISION_ENV="${CODEVISION_ENV:-codevision}"

mkdir -p "${RUN_DIR}" "${SERVICE_LOG_DIR}" "${SERVICE_PID_DIR}" "${OCR_DEMO_OUT_DIR}" "${TOOL_DEMO_OUT_DIR}"

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

CONDA_BASE="$(conda info --base)"

resolve_env_python() {
  local env_spec="$1"
  if [[ -d "${env_spec}" && -x "${env_spec}/bin/python" ]]; then
    echo "${env_spec}/bin/python"
    return
  fi
  echo "${CONDA_BASE}/envs/${env_spec}/bin/python"
}

CODEVISION_PY="$(resolve_env_python "${CODEVISION_ENV}")"
PADDLEOCR_PY="$(resolve_env_python "${OCR_ENV}")"
ALLOCATED_GPU_IDS_RAW="${CUDA_VISIBLE_DEVICES:-}"
IFS=',' read -r -a ALLOCATED_GPU_IDS <<< "${ALLOCATED_GPU_IDS_RAW}"
TOOLS_GPU_PHYSICAL_ID="${TOOLS_GPU_PHYSICAL_ID:-${ALLOCATED_GPU_IDS[0]:-0}}"
PORT_BASE="${PORT_BASE:-$((20000 + (${SLURM_JOB_ID:-0} % 1000) * 10))}"
OCR_PORT="${OCR_PORT:-${PORT_BASE}}"
GROUNDEDSAM2_PORT="${GROUNDEDSAM2_PORT:-$((PORT_BASE + 1))}"
DEPTH_PORT="${DEPTH_PORT:-$((PORT_BASE + 2))}"
COUNTGD_PORT="${COUNTGD_PORT:-$((PORT_BASE + 3))}"

if [[ ! -x "${CODEVISION_PY}" ]]; then
  echo "Missing codevision python: ${CODEVISION_PY}" >&2
  exit 1
fi
if [[ ! -x "${PADDLEOCR_PY}" ]]; then
  echo "Missing paddleocr python: ${PADDLEOCR_PY}" >&2
  exit 1
fi
if [[ ! -f "${TOOL_IMAGE_PATH}" ]]; then
  echo "Missing tool demo image: ${TOOL_IMAGE_PATH}" >&2
  exit 1
fi

cleanup() {
  set +e
  cd "${ROOT_DIR}" || exit 0
  bash scripts/launch_external_services.sh stop all
}
trap cleanup EXIT

cd "${ROOT_DIR}"

export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}"
export SERVICE_LOG_DIR
export SERVICE_PID_DIR
export OCR_PORT GROUNDEDSAM2_PORT DEPTH_PORT COUNTGD_PORT
export OCR_CUDA_VISIBLE_DEVICES="${TOOLS_GPU_PHYSICAL_ID}"
export GROUNDEDSAM2_CUDA_VISIBLE_DEVICES="${TOOLS_GPU_PHYSICAL_ID}"
export DEPTH_CUDA_VISIBLE_DEVICES="${TOOLS_GPU_PHYSICAL_ID}"
export COUNTGD_CUDA_VISIBLE_DEVICES="${TOOLS_GPU_PHYSICAL_ID}"
export DEPTH_GROUNDEDSAM2_BASE_URL="http://127.0.0.1:${GROUNDEDSAM2_PORT}"

echo "=== Allocated GPUs ==="
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "TOOLS_GPU_PHYSICAL_ID=${TOOLS_GPU_PHYSICAL_ID}"
echo "OCR_PORT=${OCR_PORT} GROUNDEDSAM2_PORT=${GROUNDEDSAM2_PORT} DEPTH_PORT=${DEPTH_PORT} COUNTGD_PORT=${COUNTGD_PORT}"
echo "=== Starting external services on one GPU ==="
bash scripts/launch_external_services.sh start all

echo "=== Generating OCR smoke image ==="
"${PADDLEOCR_PY}" - <<'PY' "${TEXT_IMAGE_PATH}"
from pathlib import Path
from PIL import Image, ImageDraw
import sys

out_path = Path(sys.argv[1])
out_path.parent.mkdir(parents=True, exist_ok=True)
image = Image.new("RGB", (960, 320), "white")
draw = ImageDraw.Draw(image)
draw.text((40, 60), "PaddleOCR PP-OCRv5 smoke", fill="black")
draw.text((40, 140), "GroundedSAM2 DepthPro CountGD", fill="black")
draw.text((40, 220), "Job {}".format(out_path.parent.name), fill="black")
image.save(out_path)
print(out_path)
PY

echo "=== Probing PaddleOCR service ==="
"${CODEVISION_PY}" scripts/probe_paddleocr_service.py \
  --image "${TEXT_IMAGE_PATH}" \
  --base-url "http://127.0.0.1:${OCR_PORT}" \
  --output-json "${RUN_DIR}/paddleocr_probe.json"

echo "=== Running OCR helper smoke ==="
"${CODEVISION_PY}" recipe/codevision/demo_code_image_tool_ocr_only.py \
  --image "${TEXT_IMAGE_PATH}" \
  --out-dir "${OCR_DEMO_OUT_DIR}" \
  --external-call-mode service \
  --ocr-serving-base-url "http://127.0.0.1:${OCR_PORT}" \
  --ocr-model-name paddleocr \
  --dump-ocr-json

echo "=== Running grounding/depth/count helper smoke ==="
"${CODEVISION_PY}" recipe/codevision/demo_code_image_tool_external.py \
  --image "${TOOL_IMAGE_PATH}" \
  --out-dir "${TOOL_DEMO_OUT_DIR}" \
  --cases ground_box,ground_depth,count_assist \
  --external-call-mode service \
  --ocr-serving-base-url "http://127.0.0.1:${OCR_PORT}" \
  --grounded-sam2-base-url "http://127.0.0.1:${GROUNDEDSAM2_PORT}" \
  --depth-base-url "http://127.0.0.1:${DEPTH_PORT}" \
  --countgd-base-url "http://127.0.0.1:${COUNTGD_PORT}" \
  --ocr-model-name paddleocr

echo "Smoke run finished. Outputs under ${RUN_DIR}"
