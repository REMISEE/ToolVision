#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
COMMITTEE_ENV_FILE="${COMMITTEE_ENV_FILE:-/mnt/cpfs/delinmao/log1}"
if [[ -f "${COMMITTEE_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "${COMMITTEE_ENV_FILE}"
  set +a
fi

JUDGE_RUN_ID="${JUDGE_RUN_ID:-${DLC_JOB_ID:-${PAI_JOB_ID:-${HOSTNAME:-committee}_$$}}}"
JUDGE_OUTPUT_ROOT="${JUDGE_OUTPUT_ROOT:-${ROOT_DIR}/outputs/dlc_step_judge_committee/${JUDGE_RUN_ID}}"
JUDGE_VLLM_ENV="${JUDGE_VLLM_ENV:-/mnt/cpfs/delinmao/envs/codevision_new}"
if [[ -d "${JUDGE_VLLM_ENV}" ]]; then
  export PATH="${JUDGE_VLLM_ENV}/bin:${PATH}"
fi

JUDGE_MODEL_2B_PATH="${JUDGE_MODEL_2B_PATH:-/mnt/cpfs/delinmao/models/Qwen3-VL-2B-Instruct}"
JUDGE_MODEL_4B_PATH="${JUDGE_MODEL_4B_PATH:-/mnt/cpfs/delinmao/models/Qwen3-VL-4B-Instruct}"
JUDGE_MODEL_8B_PATH="${JUDGE_MODEL_8B_PATH:-/mnt/cpfs/delinmao/models/Qwen3-VL-8B-Instruct}"
JUDGE_MODEL_32B_PATH="${JUDGE_MODEL_32B_PATH:-/mnt/cpfs/public_data/public_model/Qwen3-vl/Qwen3-VL-32B-Instruct}"

JUDGE_MODEL_2B_NAME="${JUDGE_MODEL_2B_NAME:-qwen3-vl-2b-step-judge}"
JUDGE_MODEL_4B_NAME="${JUDGE_MODEL_4B_NAME:-qwen3-vl-4b-step-judge}"
JUDGE_MODEL_8B_NAME="${JUDGE_MODEL_8B_NAME:-qwen3-vl-8b-step-judge}"
JUDGE_MODEL_32B_NAME="${JUDGE_MODEL_32B_NAME:-qwen3-vl-32b-step-judge}"
JUDGE_MODEL_8B_TEST_NAME="${JUDGE_MODEL_8B_TEST_NAME:-qwen3-vl-8b-step-judge-test}"

JUDGE_PORT_2B="${JUDGE_PORT_2B:-19080}"
JUDGE_PORT_4B="${JUDGE_PORT_4B:-19090}"
JUDGE_PORT_8B="${JUDGE_PORT_8B:-19100}"
JUDGE_PORT_32B="${JUDGE_PORT_32B:-19110}"
JUDGE_PORT_8B_TEST="${JUDGE_PORT_8B_TEST:-19120}"
COMMITTEE_PORT="${COMMITTEE_PORT:-19200}"

JUDGE_GPU_2B="${JUDGE_GPU_2B:-0}"
JUDGE_GPU_4B="${JUDGE_GPU_4B:-1}"
JUDGE_GPU_8B="${JUDGE_GPU_8B:-2}"
JUDGE_GPU_32B="${JUDGE_GPU_32B:-3,4,5,6}"
JUDGE_GPU_8B_TEST="${JUDGE_GPU_8B_TEST:-7}"
JUDGE_TP_32B="${JUDGE_TP_32B:-4}"

JUDGE_MAX_MODEL_LEN_SMALL="${JUDGE_MAX_MODEL_LEN_SMALL:-8192}"
JUDGE_MAX_MODEL_LEN_32B="${JUDGE_MAX_MODEL_LEN_32B:-8192}"
JUDGE_GPU_MEMORY_UTILIZATION_SMALL="${JUDGE_GPU_MEMORY_UTILIZATION_SMALL:-0.82}"
JUDGE_GPU_MEMORY_UTILIZATION_32B="${JUDGE_GPU_MEMORY_UTILIZATION_32B:-0.90}"
JUDGE_LOCAL_API_KEY="${JUDGE_LOCAL_API_KEY:-local-step-judge-key}"
COMMITTEE_API_KEY="${COMMITTEE_API_KEY:-committee-step-judge-key}"
COMMITTEE_MODEL_NAME="${COMMITTEE_MODEL_NAME:-step-judge-committee}"
COMMITTEE_TEMPERATURE_A="${COMMITTEE_TEMPERATURE_A:-0.2}"
COMMITTEE_TEMPERATURE_B="${COMMITTEE_TEMPERATURE_B:-0.3}"
COMMITTEE_MAX_WORKERS="${COMMITTEE_MAX_WORKERS:-12}"
COMMITTEE_TIMEOUT_S="${COMMITTEE_TIMEOUT_S:-120}"
COMMITTEE_MAX_RETRIES="${COMMITTEE_MAX_RETRIES:-0}"
COMMITTEE_INCLUDE_8B_TEST="${COMMITTEE_INCLUDE_8B_TEST:-0}"

COMMITTEE_API1_NAME="${COMMITTEE_API1_NAME:-qwen36plus_api}"
COMMITTEE_API1_BASE_URL="${COMMITTEE_API1_BASE_URL:-${OFFLINE_SFT_QWEN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}}"
COMMITTEE_API1_MODEL="${COMMITTEE_API1_MODEL:-qwen3.6-plus}"
COMMITTEE_API1_API_KEY="${COMMITTEE_API1_API_KEY:-${OFFLINE_SFT_QWEN_API_KEY:-${DASHSCOPE_API_KEY:-}}}"
COMMITTEE_API1_REQUEST_BODY="${COMMITTEE_API1_REQUEST_BODY:-{\"enable_thinking\":false}}"

COMMITTEE_API2_NAME="${COMMITTEE_API2_NAME:-qwen35_122b_a10b_api}"
COMMITTEE_API2_BASE_URL="${COMMITTEE_API2_BASE_URL:-${OFFLINE_SFT_QWEN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}}"
COMMITTEE_API2_MODEL="${COMMITTEE_API2_MODEL:-qwen3.5-122b-a10b}"
COMMITTEE_API2_API_KEY="${COMMITTEE_API2_API_KEY:-${OFFLINE_SFT_QWEN_API_KEY:-${DASHSCOPE_API_KEY:-}}}"
COMMITTEE_API2_REQUEST_BODY="${COMMITTEE_API2_REQUEST_BODY:-{\"enable_thinking\":false}}"

JUDGE_WARMUP_TIMEOUT_S="${JUDGE_WARMUP_TIMEOUT_S:-1800}"
JUDGE_KEEPALIVE_SECONDS="${JUDGE_KEEPALIVE_SECONDS:-300}"
SKIP_JUDGE_WARMUP="${SKIP_JUDGE_WARMUP:-0}"
JUDGE_VLLM_EXTRA_ARGS_COMMON="${JUDGE_VLLM_EXTRA_ARGS_COMMON:-}"
JUDGE_VLLM_EXTRA_ARGS_32B="${JUDGE_VLLM_EXTRA_ARGS_32B:-}"

cd "${ROOT_DIR}"
mkdir -p "${JUDGE_OUTPUT_ROOT}/logs" "${JUDGE_OUTPUT_ROOT}/pids"

SERVICE_PIDS=()

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

start_vllm() {
  local label="$1"
  local model_path="$2"
  local model_name="$3"
  local port="$4"
  local gpu_group="$5"
  local tp_size="$6"
  local max_model_len="$7"
  local gpu_memory_utilization="$8"
  local extra_args="${9:-}"
  local log_file="${JUDGE_OUTPUT_ROOT}/logs/${label}.log"
  local pid_file="${JUDGE_OUTPUT_ROOT}/pids/${label}.pid"

  if [[ ! -d "${model_path}" ]]; then
    echo "Missing model path for ${label}: ${model_path}" >&2
    exit 2
  fi

  echo "Starting ${label}: gpu=${gpu_group} tp=${tp_size} port=${port} model=${model_name}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_group}"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    if ! command -v vllm >/dev/null 2>&1; then
      echo "vllm is not available in PATH; use an image/env that includes vLLM." >&2
      exit 127
    fi
    args=(
      serve "${model_path}"
      --host 0.0.0.0
      --port "${port}"
      --served-model-name "${model_name}"
      --api-key "${JUDGE_LOCAL_API_KEY}"
      --tensor-parallel-size "${tp_size}"
      --trust-remote-code
      --max-model-len "${max_model_len}"
      --gpu-memory-utilization "${gpu_memory_utilization}"
    )
    if [[ -n "${JUDGE_VLLM_EXTRA_ARGS_COMMON}" ]]; then
      # shellcheck disable=SC2206
      common_args=(${JUDGE_VLLM_EXTRA_ARGS_COMMON})
      args+=("${common_args[@]}")
    fi
    if [[ -n "${extra_args}" ]]; then
      # shellcheck disable=SC2206
      model_args=(${extra_args})
      args+=("${model_args[@]}")
    fi
    exec vllm "${args[@]}"
  ) >"${log_file}" 2>&1 &

  local pid="$!"
  echo "${pid}" >"${pid_file}"
  SERVICE_PIDS+=("${pid}")

  if [[ "${SKIP_JUDGE_WARMUP}" != "1" && "${SKIP_JUDGE_WARMUP,,}" != "true" ]]; then
    if ! wait_for_http "http://127.0.0.1:${port}/health" "${JUDGE_WARMUP_TIMEOUT_S}"; then
      echo "${label} did not become healthy. Last log lines:" >&2
      tail -n 80 "${log_file}" >&2 || true
      exit 1
    fi
  fi
}

build_committee_json() {
  python3 - <<'PY'
import json
import os

temp_a = float(os.environ.get("COMMITTEE_TEMPERATURE_A", "0.2"))
temp_b = float(os.environ.get("COMMITTEE_TEMPERATURE_B", "0.3"))
timeout_s = float(os.environ.get("COMMITTEE_TIMEOUT_S", "120"))
max_retries = int(os.environ.get("COMMITTEE_MAX_RETRIES", "0"))
members = []


def add_pair(name, base_url, model, api_key_env, request_body=None):
    for suffix, temperature in (("a", temp_a), ("b", temp_b)):
        members.append(
            {
                "name": f"{name}_{suffix}",
                "base_url": base_url,
                "model": model,
                "api_key_env": api_key_env,
                "temperature": temperature,
                "timeout_s": timeout_s,
                "max_retries": max_retries,
                "request_body": request_body or {"max_tokens": 256},
            }
        )


add_pair("qwen3_vl_2b", f"http://127.0.0.1:{os.environ['JUDGE_PORT_2B']}/v1", os.environ["JUDGE_MODEL_2B_NAME"], "JUDGE_LOCAL_API_KEY")
add_pair("qwen3_vl_4b", f"http://127.0.0.1:{os.environ['JUDGE_PORT_4B']}/v1", os.environ["JUDGE_MODEL_4B_NAME"], "JUDGE_LOCAL_API_KEY")
add_pair("qwen3_vl_8b", f"http://127.0.0.1:{os.environ['JUDGE_PORT_8B']}/v1", os.environ["JUDGE_MODEL_8B_NAME"], "JUDGE_LOCAL_API_KEY")
add_pair("qwen3_vl_32b", f"http://127.0.0.1:{os.environ['JUDGE_PORT_32B']}/v1", os.environ["JUDGE_MODEL_32B_NAME"], "JUDGE_LOCAL_API_KEY")

if os.environ.get("COMMITTEE_INCLUDE_8B_TEST", "0").lower() in {"1", "true", "yes", "on"}:
    add_pair("qwen3_vl_8b_test", f"http://127.0.0.1:{os.environ['JUDGE_PORT_8B_TEST']}/v1", os.environ["JUDGE_MODEL_8B_TEST_NAME"], "JUDGE_LOCAL_API_KEY")

api1_key = os.environ.get("COMMITTEE_API1_API_KEY", "")
if api1_key and os.environ.get("COMMITTEE_API1_BASE_URL") and os.environ.get("COMMITTEE_API1_MODEL"):
    os.environ["COMMITTEE_API1_API_KEY_RUNTIME"] = api1_key
    add_pair(
        os.environ.get("COMMITTEE_API1_NAME", "api1"),
        os.environ["COMMITTEE_API1_BASE_URL"],
        os.environ["COMMITTEE_API1_MODEL"],
        "COMMITTEE_API1_API_KEY_RUNTIME",
        json.loads(os.environ.get("COMMITTEE_API1_REQUEST_BODY", "{}") or "{}"),
    )

api2_key = os.environ.get("COMMITTEE_API2_API_KEY", "")
if api2_key and os.environ.get("COMMITTEE_API2_BASE_URL") and os.environ.get("COMMITTEE_API2_MODEL"):
    os.environ["COMMITTEE_API2_API_KEY_RUNTIME"] = api2_key
    add_pair(
        os.environ.get("COMMITTEE_API2_NAME", "api2"),
        os.environ["COMMITTEE_API2_BASE_URL"],
        os.environ["COMMITTEE_API2_MODEL"],
        "COMMITTEE_API2_API_KEY_RUNTIME",
        json.loads(os.environ.get("COMMITTEE_API2_REQUEST_BODY", "{}") or "{}"),
    )

print(json.dumps(members, ensure_ascii=False))
PY
}

start_gateway() {
  local log_file="${JUDGE_OUTPUT_ROOT}/logs/committee_gateway.log"
  local pid_file="${JUDGE_OUTPUT_ROOT}/pids/committee_gateway.pid"
  export COMMITTEE_HOST="${COMMITTEE_HOST:-0.0.0.0}"
  export COMMITTEE_PORT
  export COMMITTEE_MODEL_NAME
  export COMMITTEE_API_KEY
  export COMMITTEE_MAX_WORKERS
  export COMMITTEE_TIMEOUT_S
  export COMMITTEE_MAX_RETRIES
  export COMMITTEE_LOG_JSONL="${COMMITTEE_LOG_JSONL:-${JUDGE_OUTPUT_ROOT}/logs/committee_requests.jsonl}"
  export COMMITTEE_JUDGES_JSON

  echo "Starting committee gateway on port ${COMMITTEE_PORT}"
  python3 scripts/step_judge_committee_gateway.py >"${log_file}" 2>&1 &
  local pid="$!"
  echo "${pid}" >"${pid_file}"
  SERVICE_PIDS+=("${pid}")

  if [[ "${SKIP_JUDGE_WARMUP}" != "1" && "${SKIP_JUDGE_WARMUP,,}" != "true" ]]; then
    if ! wait_for_http "http://127.0.0.1:${COMMITTEE_PORT}/health" "${JUDGE_WARMUP_TIMEOUT_S}"; then
      echo "committee gateway did not become healthy. Last log lines:" >&2
      tail -n 80 "${log_file}" >&2 || true
      exit 1
    fi
  fi
}

cleanup() {
  for pid in "${SERVICE_PIDS[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT TERM INT

echo "Starting ToolVision step judge committee"
echo "JUDGE_RUN_ID=${JUDGE_RUN_ID}"
echo "JUDGE_OUTPUT_ROOT=${JUDGE_OUTPUT_ROOT}"
echo "JUDGE_VLLM_ENV=${JUDGE_VLLM_ENV}"
echo "GPU layout: 2B=${JUDGE_GPU_2B}, 4B=${JUDGE_GPU_4B}, 8B=${JUDGE_GPU_8B}, 32B=${JUDGE_GPU_32B}, 8B-test=${JUDGE_GPU_8B_TEST}"
echo "Ports: 2B=${JUDGE_PORT_2B}, 4B=${JUDGE_PORT_4B}, 8B=${JUDGE_PORT_8B}, 32B=${JUDGE_PORT_32B}, 8B-test=${JUDGE_PORT_8B_TEST}, gateway=${COMMITTEE_PORT}"
echo "API members: api1=$([[ -n "${COMMITTEE_API1_API_KEY}" ]] && echo enabled || echo disabled), api2=$([[ -n "${COMMITTEE_API2_API_KEY}" ]] && echo enabled || echo disabled)"

start_vllm "qwen3_vl_2b" "${JUDGE_MODEL_2B_PATH}" "${JUDGE_MODEL_2B_NAME}" "${JUDGE_PORT_2B}" "${JUDGE_GPU_2B}" 1 "${JUDGE_MAX_MODEL_LEN_SMALL}" "${JUDGE_GPU_MEMORY_UTILIZATION_SMALL}" ""
start_vllm "qwen3_vl_4b" "${JUDGE_MODEL_4B_PATH}" "${JUDGE_MODEL_4B_NAME}" "${JUDGE_PORT_4B}" "${JUDGE_GPU_4B}" 1 "${JUDGE_MAX_MODEL_LEN_SMALL}" "${JUDGE_GPU_MEMORY_UTILIZATION_SMALL}" ""
start_vllm "qwen3_vl_8b" "${JUDGE_MODEL_8B_PATH}" "${JUDGE_MODEL_8B_NAME}" "${JUDGE_PORT_8B}" "${JUDGE_GPU_8B}" 1 "${JUDGE_MAX_MODEL_LEN_SMALL}" "${JUDGE_GPU_MEMORY_UTILIZATION_SMALL}" ""
start_vllm "qwen3_vl_32b" "${JUDGE_MODEL_32B_PATH}" "${JUDGE_MODEL_32B_NAME}" "${JUDGE_PORT_32B}" "${JUDGE_GPU_32B}" "${JUDGE_TP_32B}" "${JUDGE_MAX_MODEL_LEN_32B}" "${JUDGE_GPU_MEMORY_UTILIZATION_32B}" "${JUDGE_VLLM_EXTRA_ARGS_32B}"
start_vllm "qwen3_vl_8b_test" "${JUDGE_MODEL_8B_PATH}" "${JUDGE_MODEL_8B_TEST_NAME}" "${JUDGE_PORT_8B_TEST}" "${JUDGE_GPU_8B_TEST}" 1 "${JUDGE_MAX_MODEL_LEN_SMALL}" "${JUDGE_GPU_MEMORY_UTILIZATION_SMALL}" ""

export JUDGE_LOCAL_API_KEY
export COMMITTEE_API1_API_KEY_RUNTIME="${COMMITTEE_API1_API_KEY}"
export COMMITTEE_API2_API_KEY_RUNTIME="${COMMITTEE_API2_API_KEY}"
COMMITTEE_JUDGES_JSON="$(build_committee_json)"
export COMMITTEE_JUDGES_JSON
start_gateway

echo "Committee ready."
echo "Use in RL:"
echo "  export TOOL_REWARD_MODE=mut_clean_step_v1"
echo "  export STEP_REWARD_ENABLE=True"
echo "  export STEP_JUDGE_BASE_URL=http://<judge-host>:${COMMITTEE_PORT}"
echo "  export STEP_JUDGE_MODEL=${COMMITTEE_MODEL_NAME}"
echo "  export STEP_JUDGE_API_KEY=${COMMITTEE_API_KEY}"
echo "  export STEP_JUDGE_NUM_JUDGMENTS=1"
echo "Individual test endpoints:"
echo "  2B:       http://<judge-host>:${JUDGE_PORT_2B}/v1  model=${JUDGE_MODEL_2B_NAME}"
echo "  4B:       http://<judge-host>:${JUDGE_PORT_4B}/v1  model=${JUDGE_MODEL_4B_NAME}"
echo "  8B:       http://<judge-host>:${JUDGE_PORT_8B}/v1  model=${JUDGE_MODEL_8B_NAME}"
echo "  32B:      http://<judge-host>:${JUDGE_PORT_32B}/v1  model=${JUDGE_MODEL_32B_NAME}"
echo "  8B-test:  http://<judge-host>:${JUDGE_PORT_8B_TEST}/v1  model=${JUDGE_MODEL_8B_TEST_NAME}"

while true; do
  sleep "${JUDGE_KEEPALIVE_SECONDS}"
  for pid in "${SERVICE_PIDS[@]}"; do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "A committee service process exited: pid=${pid}" >&2
      exit 1
    fi
  done
  echo "committee alive gateway=http://<judge-host>:${COMMITTEE_PORT} members=$(python3 - <<'PY'
import json
import os
print(len(json.loads(os.environ.get("COMMITTEE_JUDGES_JSON", "[]"))))
PY
)"
done
