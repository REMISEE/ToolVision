#!/usr/bin/env bash
set -euo pipefail

# Submit a tiny DLC smoke job for the direct-driver GSPO launcher.
# Run this from DSW after starting tool services with start_dsw_tool_services.sh.

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
DLC_BIN="${DLC_BIN:-dlc_pai}"

eval "$("${ROOT_DIR}/scripts/dsw_tool_urls.sh")"

JOB_NAME="${JOB_NAME:-codevision_gspo_direct_smoke}"
WORKER_IMAGE="${WORKER_IMAGE:-dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/torcheasyrec:1.1.0-pytorch2.10.0-gpu-py311-cu129-ubuntu22.04}"
MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03}"
TRAIN_FILES_ARG="${TRAIN_FILES:-['/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k/parquet/train.parquet']}"
TEST_FILES_ARG="${TEST_FILES:-['/mnt/cpfs/delinmao/Benchmarks/MVToolBench/mvtoolbench_codevision_eval.parquet']}"

shell_quote() {
  printf '%q' "$1"
}

if [[ -z "${WORKER_IMAGE}" || "${WORKER_IMAGE}" == *"你的"* || "${WORKER_IMAGE}" == *"TODO"* ]]; then
  echo "WORKER_IMAGE must be a real DLC image URI, got: ${WORKER_IMAGE}" >&2
  exit 1
fi

TRAIN_COMMAND="cd ${ROOT_DIR} && \
MODEL_PATH=${MODEL_PATH} \
TRAIN_FILES=$(shell_quote "${TRAIN_FILES_ARG}") \
TEST_FILES=$(shell_quote "${TEST_FILES_ARG}") \
ENABLE_TOOLS=1 \
OCR_BASE_URL=${OCR_BASE_URL} \
GROUNDEDSAM2_BASE_URL=${GROUNDEDSAM2_BASE_URL} \
DEPTH_BASE_URL=${DEPTH_BASE_URL} \
COUNTGD_BASE_URL=${COUNTGD_BASE_URL} \
CODEVISION_ENV=${CODEVISION_ENV:-/mnt/cpfs/delinmao/envs/codevision_new} \
DLC_ENTRYPOINT_DEBUG=${DLC_ENTRYPOINT_DEBUG:-1} \
RAY_NODE_CHECK_TIMEOUT_SECONDS=${RAY_NODE_CHECK_TIMEOUT_SECONDS:-20} \
TOOL_PREFLIGHT_CHECK=${TOOL_PREFLIGHT_CHECK:-1} \
VAL_BEFORE_TRAIN=False \
TEST_FREQ=-1 \
TRAIN_BSZ=${TRAIN_BSZ:-64} \
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-2} \
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1} \
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2} \
TRAINER_LOGGER='[\"console\"]' \
bash scripts/dlc_ray_direct_entrypoint.sh"

echo "Submitting ${JOB_NAME}"
echo "OCR_BASE_URL=${OCR_BASE_URL}"
echo "GROUNDEDSAM2_BASE_URL=${GROUNDEDSAM2_BASE_URL}"
echo "DEPTH_BASE_URL=${DEPTH_BASE_URL}"
echo "COUNTGD_BASE_URL=${COUNTGD_BASE_URL}"
echo "TRAIN_FILES=${TRAIN_FILES_ARG}"
echo "TEST_FILES=${TEST_FILES_ARG}"

"${DLC_BIN}" submit pytorchjob \
  --name="${JOB_NAME}" \
  --command="${TRAIN_COMMAND}" \
  --data_source_uris="${DATA_SOURCE_URIS:-cpfs://cpfs-298fffb575a502fe.cn-wulanchabu/ptc-29f47d9393ad2b16/exp-29f2869e7d984aa6/::/mnt/cpfs,oss://pai-wlcb-ai-oss.oss-cn-wulanchabu-internal.aliyuncs.com/::/mnt/oss}" \
  --resource_id="${RESOURCE_ID:-quotaev2tl4w6aw0}" \
  --workspace_id="${WORKSPACE_ID:-240810}" \
  --vpc_id="${VPC_ID:-vpc-0jl5rpw5qokp6p2ettip6}" \
  --switch_id="${SWITCH_ID:-vsw-0jlmr9rjzed093yr9c0kz}" \
  --security_group_id="${SECURITY_GROUP_ID:-sg-0jl0pd5qaerdj75wmred}" \
  --priority="${PRIORITY:-9}" \
  --extended_cidrs="${EXTENDED_CIDRS:-10.1.255.0/29,10.1.255.8/29,10.1.16.0/20}" \
  --advanced_settings="${ADVANCED_SETTINGS:-createSvcForAllWorkers=true,customPortList=6379;6380-6383;8265;20000-25000}" \
  --workers="${DLC_WORKERS:-2}" \
  --worker_image="${WORKER_IMAGE}" \
  --worker_cpu="${WORKER_CPU:-110}" \
  --worker_memory="${WORKER_MEMORY:-1500Gi}" \
  --worker_shared_memory="${WORKER_SHARED_MEMORY:-1500Gi}" \
  --worker_gpu="${WORKER_GPU:-8}"
