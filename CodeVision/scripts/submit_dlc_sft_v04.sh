#!/usr/bin/env bash
set -euo pipefail

# Submit one-node 8-GPU LLaMA-Factory SFT for the v04 mixture.
# SFT logging defaults to report_to=none in the YAML. If ENABLE_WANDB=1, export
# WANDB_API_KEY before running. The key is passed through DLC envs rather than
# embedded into the user command.

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory}"
DLC_BIN="${DLC_BIN:-$([[ -x /mnt/cpfs/delinmao/bin/dlc_pai ]] && echo /mnt/cpfs/delinmao/bin/dlc_pai || command -v dlc_pai 2>/dev/null || command -v dlc 2>/dev/null || echo /etc/dsw/runtime/export_bin/dlc)}"
DLC_REGION="${DLC_REGION:-cn-wulanchabu}"
DLC_ENDPOINT="${DLC_ENDPOINT:-pai-dlc.cn-wulanchabu.aliyuncs.com}"

JOB_NAME="${JOB_NAME:-codevision_sft_v04}"
WORKER_IMAGE="${WORKER_IMAGE:-dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/torcheasyrec:1.1.0-pytorch2.10.0-gpu-py311-cu129-ubuntu22.04}"
CONFIG_PATH="${CONFIG_PATH:-examples/train_full/qwen3vl_sft_mix200_simple_notool_sp3_v04_finalonly.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v04}"
LLAMAFACTORY_PREFIX="${LLAMAFACTORY_PREFIX:-/mnt/cpfs/delinmao/envs/llamafactory}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-${LLAMAFACTORY_PREFIX}/bin/llamafactory-cli}"
LLAMAFACTORY_PYTHON="${LLAMAFACTORY_PYTHON:-${LLAMAFACTORY_PREFIX}/bin/python}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-CodeVisionSFT}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-sft_mix200_sp3_v04_finalonly}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
DRY_RUN_VALUE="${DRY_RUN:-0}"

if [[ -z "${WORKER_IMAGE}" || "${WORKER_IMAGE}" == *"你的"* || "${WORKER_IMAGE}" == *"TODO"* ]]; then
  echo "WORKER_IMAGE must be a real DLC image URI, got: ${WORKER_IMAGE}" >&2
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/${CONFIG_PATH}" ]]; then
  echo "Missing training config: ${ROOT_DIR}/${CONFIG_PATH}" >&2
  exit 1
fi

DATASET_JSON="${ROOT_DIR}/data/codevision_sft_mix200_simple_notool_sp3_v04.json"
if [[ ! -f "${DATASET_JSON}" ]]; then
  echo "Missing v04 dataset JSON: ${DATASET_JSON}" >&2
  echo "Run scripts/build_sft_v04_dataset.py before submitting training." >&2
  if [[ "${DRY_RUN_VALUE}" != "1" && "${DRY_RUN_VALUE,,}" != "true" ]]; then
    exit 1
  fi
  echo "DRY_RUN is set, continuing to print the DLC command shape." >&2
fi

if [[ "${ENABLE_WANDB}" == "1" || "${ENABLE_WANDB,,}" == "true" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ENABLE_WANDB=1 requires WANDB_API_KEY in the submit environment." >&2
    exit 1
  fi
  if [[ "${EXTRA_TRAIN_ARGS}" != *"report_to="* ]]; then
    EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS} report_to=wandb"
  fi
  if [[ "${EXTRA_TRAIN_ARGS}" != *"run_name="* ]]; then
    EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS} run_name=${WANDB_RUN_NAME}"
  fi
fi

shell_quote() {
  printf '%q' "$1"
}

append_export() {
  local name="$1"
  local value="$2"
  TRAIN_COMMAND+=" export ${name}=$(shell_quote "${value}");"
}

TRAIN_COMMAND="cd $(shell_quote "${ROOT_DIR}") &&"
append_export PATH "${LLAMAFACTORY_PREFIX}/bin:${PATH}"
append_export PYTHONPATH "${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
append_export FORCE_TORCHRUN "${FORCE_TORCHRUN:-1}"
append_export DISABLE_VERSION_CHECK "${DISABLE_VERSION_CHECK:-1}"
append_export PYTORCH_CUDA_ALLOC_CONF "${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
append_export CUDA_HOME "${CUDA_HOME:-/usr/local/cuda-12.9}"
append_export WANDB_PROJECT "${WANDB_PROJECT}"
append_export WANDB_MODE "${WANDB_MODE:-online}"

TRAIN_COMMAND+=" if [[ -x $(shell_quote "${LLAMAFACTORY_CLI}") ]]; then"
TRAIN_COMMAND+=" $(shell_quote "${LLAMAFACTORY_CLI}") train $(shell_quote "${CONFIG_PATH}") ${EXTRA_TRAIN_ARGS};"
TRAIN_COMMAND+=" else $(shell_quote "${LLAMAFACTORY_PYTHON}") -m llamafactory.cli train $(shell_quote "${CONFIG_PATH}") ${EXTRA_TRAIN_ARGS}; fi"

DLC_GLOBAL_ARGS=()
if [[ "$(basename "${DLC_BIN}")" != "dlc_pai" ]]; then
  DLC_GLOBAL_ARGS=(--region "${DLC_REGION}" --endpoint "${DLC_ENDPOINT}")
  if [[ -n "${DLC_ACCESS_ID:-${ALIBABA_CLOUD_ACCESS_KEY_ID:-}}" ]]; then
    DLC_GLOBAL_ARGS+=(--access_id "${DLC_ACCESS_ID:-${ALIBABA_CLOUD_ACCESS_KEY_ID}}")
  fi
  if [[ -n "${DLC_ACCESS_KEY:-${ALIBABA_CLOUD_ACCESS_KEY_SECRET:-}}" ]]; then
    DLC_GLOBAL_ARGS+=(--access_key "${DLC_ACCESS_KEY:-${ALIBABA_CLOUD_ACCESS_KEY_SECRET}}")
  fi
  if [[ -n "${DLC_SECURITY_TOKEN:-${ALIBABA_CLOUD_SECURITY_TOKEN:-}}" ]]; then
    DLC_GLOBAL_ARGS+=(--security_token "${DLC_SECURITY_TOKEN:-${ALIBABA_CLOUD_SECURITY_TOKEN}}")
  fi
fi

DLC_ENV_ARGS=()
if [[ "${ENABLE_WANDB}" == "1" || "${ENABLE_WANDB,,}" == "true" ]]; then
  DLC_ENV_ARGS+=(--envs "WANDB_API_KEY=${WANDB_API_KEY}")
fi

echo "Submitting ${JOB_NAME}"
echo "ROOT_DIR=${ROOT_DIR}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "WORKER_IMAGE=${WORKER_IMAGE}"
echo "ENABLE_WANDB=${ENABLE_WANDB}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_RUN_NAME=${WANDB_RUN_NAME}"
echo "EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS}"
echo "DLC_BIN=${DLC_BIN}"
echo "DLC_REGION=${DLC_REGION}"
echo "DLC_ENDPOINT=${DLC_ENDPOINT}"

if [[ "${DRY_RUN_VALUE}" == "1" || "${DRY_RUN_VALUE,,}" == "true" ]]; then
  echo "DRY_RUN=1, not submitting."
  dry_run_command="${DLC_BIN} submit pytorchjob ${DLC_GLOBAL_ARGS[*]} --name=${JOB_NAME} --command=$(shell_quote "${TRAIN_COMMAND}") ..."
  printf '%s\n' "${dry_run_command}" | sed \
    -e 's/\(WANDB_API_KEY=\)[^\\ ]*/\1<redacted>/g' \
    -e 's/"WANDB_API_KEY"[[:space:]]*:[[:space:]]*"[^"]*"/"WANDB_API_KEY":"<redacted>"/g'
  exit 0
fi

"${DLC_BIN}" submit pytorchjob \
  "${DLC_GLOBAL_ARGS[@]}" \
  --name="${JOB_NAME}" \
  --command="${TRAIN_COMMAND}" \
  "${DLC_ENV_ARGS[@]}" \
  --data_source_uris="${DATA_SOURCE_URIS:-cpfs://cpfs-298fffb575a502fe.cn-wulanchabu/ptc-29f47d9393ad2b16/exp-29f2869e7d984aa6/::/mnt/cpfs,oss://pai-wlcb-ai-oss.oss-cn-wulanchabu-internal.aliyuncs.com/::/mnt/oss}" \
  --resource_id="${RESOURCE_ID:-quotaev2tl4w6aw0}" \
  --workspace_id="${WORKSPACE_ID:-240810}" \
  --vpc_id="${VPC_ID:-vpc-0jl5rpw5qokp6p2ettip6}" \
  --switch_id="${SWITCH_ID:-vsw-0jlmr9rjzed093yr9c0kz}" \
  --security_group_id="${SECURITY_GROUP_ID:-sg-0jl0pd5qaerdj75wmred}" \
  --priority="${PRIORITY:-8}" \
  --extended_cidrs="${EXTENDED_CIDRS:-10.1.255.0/29,10.1.255.8/29,10.1.16.0/20}" \
  --advanced_settings="${ADVANCED_SETTINGS:-createSvcForAllWorkers=true,customPortList=6379;6380-6383;8265}" \
  --workers="${DLC_WORKERS:-1}" \
  --worker_image="${WORKER_IMAGE}" \
  --worker_cpu="${WORKER_CPU:-110}" \
  --worker_memory="${WORKER_MEMORY:-1500Gi}" \
  --worker_shared_memory="${WORKER_SHARED_MEMORY:-1500Gi}" \
  --worker_gpu="${WORKER_GPU:-8}" \
  2>&1 | sed \
    -e 's/\(WANDB_API_KEY[=:]\)[^,} ]*/\1<redacted>/g' \
    -e 's/"WANDB_API_KEY"[[:space:]]*:[[:space:]]*"[^"]*"/"WANDB_API_KEY":"<redacted>"/g' \
    -e 's/\(access_key[=:]\)[^,} ]*/\1<redacted>/Ig' \
    -e 's/\(AccessKeySecret[=:]\)[^,} ]*/\1<redacted>/g'
