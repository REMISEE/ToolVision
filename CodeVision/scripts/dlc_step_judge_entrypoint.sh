#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
JUDGE_RUN_ID="${JUDGE_RUN_ID:-${DLC_JOB_ID:-${PAI_JOB_ID:-${HOSTNAME:-judge}_$$}}}"
JUDGE_OUTPUT_ROOT="${JUDGE_OUTPUT_ROOT:-${ROOT_DIR}/outputs/dlc_step_judge/${JUDGE_RUN_ID}}"

JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-step-judge}"
JUDGE_HOST="${JUDGE_HOST:-0.0.0.0}"
JUDGE_PORT_BASE="${JUDGE_PORT_BASE:-19080}"
JUDGE_PORT_STRIDE="${JUDGE_PORT_STRIDE:-10}"
JUDGE_REPLICA_COUNT="${JUDGE_REPLICA_COUNT:-1}"
JUDGE_REPLICA_GPUS="${JUDGE_REPLICA_GPUS:-0}"
JUDGE_REPLICA_GPU_GROUPS="${JUDGE_REPLICA_GPU_GROUPS:-}"
JUDGE_TP_SIZE="${JUDGE_TP_SIZE:-1}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-8192}"
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.85}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
JUDGE_SERVER_CMD="${JUDGE_SERVER_CMD:-}"
JUDGE_VLLM_EXTRA_ARGS="${JUDGE_VLLM_EXTRA_ARGS:-}"
JUDGE_WARMUP_TIMEOUT_S="${JUDGE_WARMUP_TIMEOUT_S:-900}"
JUDGE_KEEPALIVE_SECONDS="${JUDGE_KEEPALIVE_SECONDS:-300}"
SKIP_JUDGE_WARMUP="${SKIP_JUDGE_WARMUP:-0}"

cd "${ROOT_DIR}"
mkdir -p "${JUDGE_OUTPUT_ROOT}/logs" "${JUDGE_OUTPUT_ROOT}/pids"

if [[ -z "${JUDGE_SERVER_CMD}" && -z "${JUDGE_MODEL_PATH}" ]]; then
  echo "JUDGE_MODEL_PATH is required unless JUDGE_SERVER_CMD is set." >&2
  exit 2
fi

IFS=';' read -r -a GPU_GROUPS <<<"${JUDGE_REPLICA_GPU_GROUPS}"
if [[ -z "${JUDGE_REPLICA_GPU_GROUPS}" ]]; then
  IFS=',' read -r -a SINGLE_GPUS <<<"${JUDGE_REPLICA_GPUS}"
  GPU_GROUPS=()
  for gpu in "${SINGLE_GPUS[@]}"; do
    [[ -n "${gpu}" ]] && GPU_GROUPS+=("${gpu}")
  done
fi
if [[ "${#GPU_GROUPS[@]}" -eq 0 ]]; then
  GPU_GROUPS=("0")
fi

PIDS=()

wait_for_http() {
  local url="$1"
  local timeout_s="$2"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    if python3 - "$url" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
    raise SystemExit(0 if 200 <= response.status < 500 else 1)
PY
    then
      return 0
    fi
    sleep 5
  done
  return 1
}

start_replica() {
  local replica_id="$1"
  local port=$((JUDGE_PORT_BASE + replica_id * JUDGE_PORT_STRIDE))
  local gpu_group="${GPU_GROUPS[$replica_id]:-${GPU_GROUPS[0]}}"
  local log_file="${JUDGE_OUTPUT_ROOT}/logs/replica_${replica_id}.log"
  local pid_file="${JUDGE_OUTPUT_ROOT}/pids/replica_${replica_id}.pid"

  echo "Starting step judge replica ${replica_id}: gpu_group=${gpu_group} port=${port}"
  echo "STEP_JUDGE_BASE_URL=http://<judge-host>:${port}"
  echo "STEP_JUDGE_MODEL=${JUDGE_MODEL_NAME}"

  (
    export CUDA_VISIBLE_DEVICES="${gpu_group}"
    export JUDGE_HOST JUDGE_MODEL_PATH JUDGE_MODEL_NAME JUDGE_API_KEY
    export JUDGE_PORT="${port}"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

    if [[ -n "${JUDGE_SERVER_CMD}" ]]; then
      exec bash -lc "${JUDGE_SERVER_CMD}"
    fi

    if ! command -v vllm >/dev/null 2>&1; then
      echo "vllm is not available in PATH; set JUDGE_SERVER_CMD to a compatible server command." >&2
      exit 127
    fi

    args=(
      serve "${JUDGE_MODEL_PATH}"
      --host "${JUDGE_HOST}"
      --port "${port}"
      --served-model-name "${JUDGE_MODEL_NAME}"
      --tensor-parallel-size "${JUDGE_TP_SIZE}"
      --trust-remote-code
      --max-model-len "${JUDGE_MAX_MODEL_LEN}"
      --gpu-memory-utilization "${JUDGE_GPU_MEMORY_UTILIZATION}"
    )
    if [[ -n "${JUDGE_API_KEY}" ]]; then
      args+=(--api-key "${JUDGE_API_KEY}")
    fi
    if [[ -n "${JUDGE_VLLM_EXTRA_ARGS}" ]]; then
      # shellcheck disable=SC2206
      extra_args=(${JUDGE_VLLM_EXTRA_ARGS})
      args+=("${extra_args[@]}")
    fi
    exec vllm "${args[@]}"
  ) >"${log_file}" 2>&1 &

  local pid="$!"
  echo "${pid}" >"${pid_file}"
  PIDS+=("${pid}")

  if [[ "${SKIP_JUDGE_WARMUP}" != "1" && "${SKIP_JUDGE_WARMUP,,}" != "true" ]]; then
    if ! wait_for_http "http://127.0.0.1:${port}/health" "${JUDGE_WARMUP_TIMEOUT_S}"; then
      echo "Step judge replica ${replica_id} did not become healthy. Last log lines:" >&2
      tail -n 80 "${log_file}" >&2 || true
      exit 1
    fi
  fi
}

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT TERM INT

echo "Starting step judge service"
echo "JUDGE_RUN_ID=${JUDGE_RUN_ID}"
echo "JUDGE_OUTPUT_ROOT=${JUDGE_OUTPUT_ROOT}"
echo "JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH:-<custom-command>}"
echo "JUDGE_MODEL_NAME=${JUDGE_MODEL_NAME}"
echo "JUDGE_REPLICA_COUNT=${JUDGE_REPLICA_COUNT}"
echo "JUDGE_REPLICA_GPUS=${JUDGE_REPLICA_GPUS}"
echo "JUDGE_REPLICA_GPU_GROUPS=${JUDGE_REPLICA_GPU_GROUPS:-<single-gpu-list>}"
echo "JUDGE_PORT_BASE=${JUDGE_PORT_BASE}"
echo "JUDGE_PORT_STRIDE=${JUDGE_PORT_STRIDE}"
echo "JUDGE_TP_SIZE=${JUDGE_TP_SIZE}"
echo "SKIP_JUDGE_WARMUP=${SKIP_JUDGE_WARMUP}"

for ((replica_id = 0; replica_id < JUDGE_REPLICA_COUNT; replica_id++)); do
  start_replica "${replica_id}"
done

while true; do
  sleep "${JUDGE_KEEPALIVE_SECONDS}"
  for ((replica_id = 0; replica_id < JUDGE_REPLICA_COUNT; replica_id++)); do
    pid="${PIDS[$replica_id]}"
    port=$((JUDGE_PORT_BASE + replica_id * JUDGE_PORT_STRIDE))
    if kill -0 "${pid}" >/dev/null 2>&1; then
      echo "step judge replica ${replica_id} alive pid=${pid} url=http://<judge-host>:${port}"
    else
      echo "step judge replica ${replica_id} exited pid=${pid}; failing service job" >&2
      tail -n 80 "${JUDGE_OUTPUT_ROOT}/logs/replica_${replica_id}.log" >&2 || true
      exit 1
    fi
  done
done
