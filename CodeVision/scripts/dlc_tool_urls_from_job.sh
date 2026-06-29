#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dlc_tool_job_id_or_name> [pod_ip_override]" >&2
  exit 2
fi

JOB_ID="$1"
POD_IP_OVERRIDE="${2:-}"
DLC_BIN="${DLC_BIN:-$(command -v dlc_pai 2>/dev/null || command -v dlc 2>/dev/null || echo /etc/dsw/runtime/export_bin/dlc)}"
WORKSPACE_ID="${WORKSPACE_ID:-240810}"
DLC_REGION="${DLC_REGION:-cn-wulanchabu}"
DLC_ENDPOINT="${DLC_ENDPOINT:-pai-dlc.cn-wulanchabu.aliyuncs.com}"
TOOL_REPLICA_COUNT="${TOOL_REPLICA_COUNT:-2}"
TOOL_PORT_BASE="${TOOL_PORT_BASE:-18080}"
TOOL_PORT_STRIDE="${TOOL_PORT_STRIDE:-10}"

if [[ -n "${POD_IP_OVERRIDE}" ]]; then
  TOOL_HOST="${POD_IP_OVERRIDE}"
else
  RAW="$("${DLC_BIN}" get job "${JOB_ID}" -w "${WORKSPACE_ID}" --show_detail -r "${DLC_REGION}" -e "${DLC_ENDPOINT}")"
  TOOL_HOST="$(python3 - "${RAW}" <<'PY'
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

echo "TOOL_DLC_HOST=${TOOL_HOST}"
echo
for ((i = 0; i < TOOL_REPLICA_COUNT; i++)); do
  base=$((TOOL_PORT_BASE + i * TOOL_PORT_STRIDE))
  echo "# replica ${i}"
  echo "export OCR_BASE_URL=http://${TOOL_HOST}:$((base + 0))"
  echo "export GROUNDEDSAM2_BASE_URL=http://${TOOL_HOST}:$((base + 1))"
  echo "export DEPTH_BASE_URL=http://${TOOL_HOST}:$((base + 2))"
  echo "export COUNTGD_BASE_URL=http://${TOOL_HOST}:$((base + 3))"
  echo
done
