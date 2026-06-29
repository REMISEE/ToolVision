#!/usr/bin/env bash
set -euo pipefail

# Submit a no-tool lmms-eval CountBench baseline job.
# This is intentionally separate from the ToolVision agent eval scripts.

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/cpfs/delinmao}"
CODEVISION_ROOT="${CODEVISION_ROOT:-${WORKSPACE_ROOT}/ToolVision/CodeVision}"
LMMS_EVAL_DIR="${LMMS_EVAL_DIR:-${WORKSPACE_ROOT}/lmms-eval}"
DLC_BIN="${DLC_BIN:-$([[ -x ${WORKSPACE_ROOT}/bin/dlc_pai ]] && echo ${WORKSPACE_ROOT}/bin/dlc_pai || command -v dlc_pai 2>/dev/null || command -v dlc 2>/dev/null || echo /etc/dsw/runtime/export_bin/dlc)}"
DLC_REGION="${DLC_REGION:-cn-wulanchabu}"
DLC_ENDPOINT="${DLC_ENDPOINT:-pai-dlc.cn-wulanchabu.aliyuncs.com}"

JOB_NAME="${JOB_NAME:-cv-lmms-countbench-base-thinking-8gpu}"
RUN_NAME="${RUN_NAME:-base_qwen3vl8bthinking_countbench_lmms_t0}"
MODEL_PATH="${MODEL_PATH:-${WORKSPACE_ROOT}/models/Qwen3-VL-8B-Thinking}"
MODEL_BACKEND="${MODEL_BACKEND:-vllm_generate}"
TASK_NAME="${TASK_NAME:-tv_countbench_local}"
INCLUDE_PATH="${INCLUDE_PATH:-${CODEVISION_ROOT}/lmms_tasks}"
OUTPUT_PATH="${OUTPUT_PATH:-${LMMS_EVAL_DIR}/logs/${RUN_NAME}}"

CONDA_SETUP="${CONDA_SETUP:-/opt/conda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-${WORKSPACE_ROOT}/envs/codevision}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${WORKSPACE_ROOT}/cache/hf/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${WORKSPACE_ROOT}/cache/hf/datasets}"

NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
WORKER_GPU="${WORKER_GPU:-${NGPUS_PER_NODE}}"
NUM_PROCESSES="${NUM_PROCESSES:-${NGPUS_PER_NODE}}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-${NUM_PROCESSES}}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12368}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_PIXELS="${MAX_PIXELS:-6422528}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
WORKERS="${WORKERS:-32}"
LIMIT="${LIMIT:-}"
VERBOSITY="${VERBOSITY:-INFO}"
PRIORITY="${PRIORITY:-6}"
DRY_RUN="${DRY_RUN:-0}"

WORKER_IMAGE="${WORKER_IMAGE:-dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/torcheasyrec:1.1.0-pytorch2.10.0-gpu-py311-cu129-ubuntu22.04}"
DATA_SOURCE_URIS="${DATA_SOURCE_URIS:-cpfs://cpfs-298fffb575a502fe.cn-wulanchabu/ptc-29f47d9393ad2b16/exp-29f2869e7d984aa6/::/mnt/cpfs,oss://pai-wlcb-ai-oss.oss-cn-wulanchabu-internal.aliyuncs.com/::/mnt/oss}"
RESOURCE_ID="${RESOURCE_ID:-quotaev2tl4w6aw0}"
WORKSPACE_ID="${WORKSPACE_ID:-240810}"
VPC_ID="${VPC_ID:-vpc-0jl5rpw5qokp6p2ettip6}"
SWITCH_ID="${SWITCH_ID:-vsw-0jlmr9rjzed093yr9c0kz}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-sg-0jl0pd5qaerdj75wmred}"
EXTENDED_CIDRS="${EXTENDED_CIDRS:-10.1.255.0/29,10.1.255.8/29,10.1.16.0/20}"
ADVANCED_SETTINGS="${ADVANCED_SETTINGS:-createSvcForAllWorkers=true,customPortList=6379;6380-6383;8265;20000-25000}"

shell_quote() {
  printf '%q' "$1"
}

if [[ ! -d "${LMMS_EVAL_DIR}" ]]; then
  echo "Missing lmms-eval dir: ${LMMS_EVAL_DIR}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing MODEL_PATH: ${MODEL_PATH}" >&2
  exit 1
fi
IFS=',' read -ra REQUESTED_TASKS <<< "${TASK_NAME}"
for requested_task in "${REQUESTED_TASKS[@]}"; do
  requested_task="${requested_task//[[:space:]]/}"
  [[ -n "${requested_task}" ]] || continue
  if ! find "${INCLUDE_PATH}" -type f -name "${requested_task}.yaml" -print -quit | grep -q .; then
    echo "Missing local lmms task '${requested_task}' under INCLUDE_PATH=${INCLUDE_PATH}" >&2
    exit 1
  fi
done
if [[ "${CONDA_ENV}" == */* && ! -d "${CONDA_ENV}" ]]; then
  echo "Missing CONDA_ENV path: ${CONDA_ENV}" >&2
  exit 1
fi
if [[ "${MODEL_BACKEND}" == vllm* ]]; then
  if ! bash -lc "source $(printf '%q' "${CONDA_SETUP}") && conda activate $(printf '%q' "${CONDA_ENV}") && python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec('vllm') is not None else 1)
PY"; then
    echo "CONDA_ENV=${CONDA_ENV} cannot find the vllm package required by MODEL_BACKEND=${MODEL_BACKEND}." >&2
    echo "Use CONDA_ENV=${WORKSPACE_ROOT}/envs/codevision for the Qwen3-VL lmms/vLLM baseline jobs." >&2
    exit 1
  fi
fi
if [[ "${TP_SIZE}" -le 0 || "${DP_SIZE}" -le 0 || "${NUM_PROCESSES}" -le 0 ]]; then
  echo "TP_SIZE, DP_SIZE, and NUM_PROCESSES must be positive." >&2
  exit 1
fi
if [[ "${MODEL_BACKEND}" == "vllm_generate" && $((TP_SIZE * DP_SIZE)) -ne "${NUM_PROCESSES}" ]]; then
  echo "For vllm external launcher, NUM_PROCESSES must equal TP_SIZE * DP_SIZE." >&2
  echo "Got NUM_PROCESSES=${NUM_PROCESSES}, TP_SIZE=${TP_SIZE}, DP_SIZE=${DP_SIZE}" >&2
  exit 1
fi

if [[ "${MODEL_BACKEND}" == "vllm_generate" ]]; then
  MODEL_ARGS="model=${MODEL_PATH},tensor_parallel_size=${TP_SIZE},data_parallel_size=${DP_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},disable_log_stats=True,max_pixels=${MAX_PIXELS},max_model_len=${MAX_MODEL_LEN},max_num_seqs=${MAX_NUM_SEQS},max_new_tokens=${MAX_NEW_TOKENS}"
else
  MODEL_ARGS="pretrained=${MODEL_PATH},max_pixels=${MAX_PIXELS},attn_implementation=sdpa,interleave_visuals=False"
fi

EVAL_COMMAND="cd $(shell_quote "${LMMS_EVAL_DIR}") &&"
EVAL_COMMAND+=" source $(shell_quote "${CONDA_SETUP}") && conda activate $(shell_quote "${CONDA_ENV}") &&"
EVAL_COMMAND+=" mkdir -p $(shell_quote "${OUTPUT_PATH}") &&"
EVAL_COMMAND+=" HF_ENDPOINT=$(shell_quote "${HF_ENDPOINT}")"
EVAL_COMMAND+=" HF_HUB_CACHE=$(shell_quote "${HF_HUB_CACHE}")"
EVAL_COMMAND+=" HF_DATASETS_CACHE=$(shell_quote "${HF_DATASETS_CACHE}")"
EVAL_COMMAND+=" WORKERS=$(shell_quote "${WORKERS}")"
EVAL_COMMAND+=" accelerate launch --num_processes=$(shell_quote "${NUM_PROCESSES}") --main_process_port=$(shell_quote "${MAIN_PROCESS_PORT}") -m lmms_eval"
EVAL_COMMAND+=" --model $(shell_quote "${MODEL_BACKEND}")"
EVAL_COMMAND+=" --model_args $(shell_quote "${MODEL_ARGS}")"
EVAL_COMMAND+=" --include_path $(shell_quote "${INCLUDE_PATH}")"
EVAL_COMMAND+=" --tasks $(shell_quote "${TASK_NAME}")"
EVAL_COMMAND+=" --batch_size $(shell_quote "${BATCH_SIZE}")"
EVAL_COMMAND+=" --log_samples"
EVAL_COMMAND+=" --output_path $(shell_quote "${OUTPUT_PATH}")"
EVAL_COMMAND+=" --verbosity $(shell_quote "${VERBOSITY}")"
if [[ -n "${LIMIT}" ]]; then
  EVAL_COMMAND+=" --limit $(shell_quote "${LIMIT}")"
fi
EVAL_COMMAND+=" && if ! find $(shell_quote "${OUTPUT_PATH}") -type f -name $(shell_quote "*_results.json") -print -quit | grep -q .; then"
EVAL_COMMAND+=" echo $(shell_quote "No lmms-eval results JSON found under ${OUTPUT_PATH}") >&2; exit 1; fi"

DLC_GLOBAL_ARGS=()
if [[ "$(basename "${DLC_BIN}")" != "dlc_pai" ]]; then
  DLC_GLOBAL_ARGS=(--region "${DLC_REGION}" --endpoint "${DLC_ENDPOINT}")
fi

echo "Submitting lmms no-tool baseline"
echo "JOB_NAME=${JOB_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "TASK_NAME=${TASK_NAME}"
echo "MODEL_BACKEND=${MODEL_BACKEND}"
echo "NUM_PROCESSES=${NUM_PROCESSES} TP_SIZE=${TP_SIZE} DP_SIZE=${DP_SIZE}"
echo "BATCH_SIZE=${BATCH_SIZE} MAX_MODEL_LEN=${MAX_MODEL_LEN} MAX_NUM_SEQS=${MAX_NUM_SEQS} MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  echo "DRY_RUN=1; command:"
  printf '%s submit pytorchjob --name=%q --command=%q ...\n' "${DLC_BIN}" "${JOB_NAME}" "${EVAL_COMMAND}"
  exit 0
fi

"${DLC_BIN}" submit pytorchjob \
  "${DLC_GLOBAL_ARGS[@]}" \
  --name="${JOB_NAME}" \
  --command="${EVAL_COMMAND}" \
  --data_source_uris="${DATA_SOURCE_URIS}" \
  --resource_id="${RESOURCE_ID}" \
  --workspace_id="${WORKSPACE_ID}" \
  --vpc_id="${VPC_ID}" \
  --switch_id="${SWITCH_ID}" \
  --security_group_id="${SECURITY_GROUP_ID}" \
  --priority="${PRIORITY}" \
  --extended_cidrs="${EXTENDED_CIDRS}" \
  --advanced_settings="${ADVANCED_SETTINGS}" \
  --workers=1 \
  --worker_image="${WORKER_IMAGE}" \
  --worker_cpu="${WORKER_CPU:-110}" \
  --worker_memory="${WORKER_MEMORY:-1500Gi}" \
  --worker_shared_memory="${WORKER_SHARED_MEMORY:-1500Gi}" \
  --worker_gpu="${WORKER_GPU}"
