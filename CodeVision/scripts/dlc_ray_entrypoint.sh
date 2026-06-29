#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-recipe/codevision/qwen3_vl_gspo.sh}"
# Do not fall back to the old shared codevision env; it was polluted by package
# upgrades and no longer imports Qwen3-VL/verl reliably.
CODEVISION_ENV="${CODEVISION_ENV:-/mnt/cpfs/delinmao/envs/codevision_new}"

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export TORCH_DIST_INIT_BARRIER_TIMEOUT="${TORCH_DIST_INIT_BARRIER_TIMEOUT:-1800}"

if [[ -x "${CODEVISION_ENV}/bin/ray" ]]; then
  export PATH="${CODEVISION_ENV}/bin:${PATH}"
fi

RANK="${RANK:-${NODE_RANK:-${DLC_RANK:-0}}}"
WORLD_SIZE="${WORLD_SIZE:-${DLC_WORKER_NUM:-1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${WORKER_GPU:-${GPUS_PER_NODE:-8}}}"
MASTER_ADDR="${MASTER_ADDR:-${HEAD_ADDR:-127.0.0.1}}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_NODE_MANAGER_PORT="${RAY_NODE_MANAGER_PORT:-6380}"
RAY_OBJECT_MANAGER_PORT="${RAY_OBJECT_MANAGER_PORT:-6381}"
RAY_RUNTIME_ENV_AGENT_PORT="${RAY_RUNTIME_ENV_AGENT_PORT:-6382}"
RAY_DASHBOARD_AGENT_PORT="${RAY_DASHBOARD_AGENT_PORT:-6383}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-20000}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-25000}"

DLC_RUN_ID="${DLC_RUN_ID:-${JOB_NAME:-codevision_gspo}_${MASTER_ADDR}_${WORLD_SIZE}}"
DONE_DIR="${DLC_RAY_DONE_DIR:-${ROOT_DIR}/outputs/dlc_ray_done}"
DONE_FILE="${DONE_DIR}/${DLC_RUN_ID}.done"

mkdir -p "${DONE_DIR}"
cd "${ROOT_DIR}"

ray_common_args=(
  --node-manager-port="${RAY_NODE_MANAGER_PORT}"
  --object-manager-port="${RAY_OBJECT_MANAGER_PORT}"
  --runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT}"
  --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_PORT}"
  --min-worker-port="${RAY_MIN_WORKER_PORT}"
  --max-worker-port="${RAY_MAX_WORKER_PORT}"
  --num-gpus="${NPROC_PER_NODE}"
)

if [[ "${RANK}" == "0" ]]; then
  rm -f "${DONE_FILE}"
  ray stop --force >/dev/null 2>&1 || true
  ray start --head \
    --port="${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    "${ray_common_args[@]}"

  expected_nodes="${EXPECTED_RAY_NODES:-${WORLD_SIZE}}"
  wait_timeout="${RAY_WAIT_TIMEOUT_SECONDS:-300}"
  wait_interval="${RAY_WAIT_INTERVAL_SECONDS:-5}"
  check_timeout="${RAY_NODE_CHECK_TIMEOUT_SECONDS:-20}"
  elapsed=0
  echo "[INFO] Waiting for ${expected_nodes} Ray nodes at ${MASTER_ADDR}:${RAY_PORT}"
  while true; do
    current_nodes="$(timeout "${check_timeout}" python3 - "${MASTER_ADDR}:${RAY_PORT}" <<'PY' 2>/dev/null || echo 0
import sys
import ray

ray.init(address=sys.argv[1], ignore_reinit_error=True)
print(sum(1 for node in ray.nodes() if node.get("Alive")))
ray.shutdown()
PY
)"
    echo "[INFO] Ray nodes ready: ${current_nodes}/${expected_nodes} (${elapsed}s)"
    if [[ "${current_nodes}" -ge "${expected_nodes}" ]]; then
      break
    fi
    if [[ "${elapsed}" -ge "${wait_timeout}" ]]; then
      echo "[WARN] Timed out waiting for Ray nodes; proceeding with ${current_nodes}/${expected_nodes}." >&2
      break
    fi
    sleep "${wait_interval}"
    elapsed=$((elapsed + wait_interval))
  done

  export RAY_ADDRESS="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}"
  export NNODES="${WORLD_SIZE}"
  export NGPUS_PER_NODE="${NPROC_PER_NODE}"
  export RAY_JOB_NO_WAIT="${RAY_JOB_NO_WAIT:-0}"

  status=0
  bash "${TRAIN_SCRIPT}" || status=$?
  echo "${status}" >"${DONE_FILE}"
  ray stop --force >/dev/null 2>&1 || true
  exit "${status}"
fi

sleep "${RAY_WORKER_STARTUP_SLEEP:-15}"
ray stop --force >/dev/null 2>&1 || true
ray start \
  --address="${MASTER_ADDR}:${RAY_PORT}" \
  "${ray_common_args[@]}"

while [[ ! -f "${DONE_FILE}" ]]; do
  sleep "${DLC_DONE_POLL_SECONDS:-30}"
done

status="$(cat "${DONE_FILE}" 2>/dev/null || echo 1)"
ray stop --force >/dev/null 2>&1 || true
exit "${status}"
