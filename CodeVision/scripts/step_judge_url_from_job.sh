#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dlc_judge_job_id_or_name> [pod_ip_override]" >&2
  exit 2
fi

JOB_ID="$1"
POD_IP_OVERRIDE="${2:-}"
DLC_BIN="${DLC_BIN:-$(command -v dlc_pai 2>/dev/null || command -v dlc 2>/dev/null || echo /etc/dsw/runtime/export_bin/dlc)}"
WORKSPACE_ID="${WORKSPACE_ID:-240810}"
DLC_REGION="${DLC_REGION:-cn-wulanchabu}"
DLC_ENDPOINT="${DLC_ENDPOINT:-pai-dlc.cn-wulanchabu.aliyuncs.com}"
JUDGE_REPLICA_COUNT="${JUDGE_REPLICA_COUNT:-1}"
JUDGE_PORT_BASE="${JUDGE_PORT_BASE:-19080}"
JUDGE_PORT_STRIDE="${JUDGE_PORT_STRIDE:-10}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-step-judge}"

if [[ -n "${POD_IP_OVERRIDE}" ]]; then
  JUDGE_HOST="${POD_IP_OVERRIDE}"
else
  RAW="$("${DLC_BIN}" get job "${JOB_ID}" -w "${WORKSPACE_ID}" --show_detail -r "${DLC_REGION}" -e "${DLC_ENDPOINT}")"
  JUDGE_HOST="$(python3 - "${RAW}" <<'PY'
import json
import sys

raw = sys.argv[1]
start = raw.find("{")
if start < 0:
    raise SystemExit("dlc output does not contain JSON")
data = json.loads(raw[start:])
for pod in data.get("Pods", []):
    ip = pod.get("Ip") or pod.get("IP")
    status = pod.get("Status", "")
    if ip and status.lower() not in {"failed", "succeeded", "finished"}:
        print(ip)
        break
else:
    pods = data.get("Pods", [])
    if pods and (pods[0].get("Ip") or pods[0].get("IP")):
        print(pods[0].get("Ip") or pods[0].get("IP"))
    else:
        raise SystemExit("no pod IP found")
PY
)"
fi

echo "STEP_JUDGE_DLC_HOST=${JUDGE_HOST}"
echo "export STEP_REWARD_ENABLE=True"
echo "export TOOL_REWARD_MODE=mut_clean_step_v1"
echo "export STEP_JUDGE_MODEL=${JUDGE_MODEL_NAME}"
echo
for ((i = 0; i < JUDGE_REPLICA_COUNT; i++)); do
  port=$((JUDGE_PORT_BASE + i * JUDGE_PORT_STRIDE))
  echo "# judge replica ${i}"
  echo "export STEP_JUDGE_BASE_URL=http://${JUDGE_HOST}:${port}"
  echo
done
