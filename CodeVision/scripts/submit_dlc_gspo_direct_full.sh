#!/usr/bin/env bash
set -euo pipefail

# Submit the main CodeVision GSPO RL run to DLC.
# This launcher keeps qwen3_vl.sh's main training shape:
# 2 nodes x 8 GPUs, train batch 64, rollout.n 8, max turns 12,
# 16k prompt / 16k response, 1 epoch, no validation by default.

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
DLC_BIN="${DLC_BIN:-dlc_pai}"

if [[ -z "${OCR_BASE_URL:-}" || -z "${GROUNDEDSAM2_BASE_URL:-}" || -z "${DEPTH_BASE_URL:-}" || -z "${COUNTGD_BASE_URL:-}" ]]; then
  eval "$("${ROOT_DIR}/scripts/dsw_tool_urls.sh")"
fi

check_tool_port() {
  local name="$1"
  local url="$2"
  local host_port="${url#http://}"
  local host="${host_port%%:*}"
  local port="${host_port##*:}"
  if ! timeout 2 bash -lc "</dev/tcp/${host}/${port}" >/dev/null 2>&1; then
    echo "Tool service ${name} is not reachable at ${url}." >&2
    echo "Run this submit script from the DSW that hosts the tool services, or set DSW_TOOL_HOST to that DSW IP." >&2
    exit 1
  fi
}

if [[ "${SKIP_TOOL_PORT_CHECK:-0}" != "1" ]]; then
  check_tool_port "OCR" "${OCR_BASE_URL}"
  check_tool_port "GroundedSAM2" "${GROUNDEDSAM2_BASE_URL}"
  check_tool_port "Depth" "${DEPTH_BASE_URL}"
  check_tool_port "CountGD" "${COUNTGD_BASE_URL}"
fi

JOB_NAME="${JOB_NAME:-codevision_gspo_direct_full26k}"
WORKER_IMAGE="${WORKER_IMAGE:-dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/torcheasyrec:1.1.0-pytorch2.10.0-gpu-py311-cu129-ubuntu22.04}"
CODEVISION_ENV="${CODEVISION_ENV:-/mnt/cpfs/delinmao/envs/codevision_new}"
MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03}"
TOOL_CFG_TEMPLATE_PATH="${TOOL_CFG_TEMPLATE_PATH:-recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml}"
SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-recipe/codevision/config/sp3.txt}"

# Use the medium-clean 26k mixture by default. The older 40k files are left
# untouched but should be passed explicitly if needed for ablations.
TRAIN_FILES_ARG="${TRAIN_FILES:-['/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_clean_no_extra_image/parquet/train_medium_clean_21k_plus_benchmark_pass16_partial_26591_rewardfix_fsc09.parquet']}"
TEST_FILES_ARG="${TEST_FILES:-['/mnt/cpfs/delinmao/Benchmarks/MVToolBench/mvtoolbench_codevision_eval.parquet']}"

# These names become the W&B project/run name and the local checkpoint path.
PROJECT_NAME="${PROJECT_NAME:-ToolVisionRL}"
EXP_NAME="${EXP_NAME:-qwen3vl8b_gspo_full26k}"
SAVE_DIR="${SAVE_DIR:-./saves/${PROJECT_NAME}/${EXP_NAME}}"

ENABLE_WANDB="${ENABLE_WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_BASE_URL="${WANDB_BASE_URL:-}"

# Enable judge-family fallback by default. Keep TOOLVISION_RL_USE_LLM_JUDGE=0
# so non-judge rule families do not call the judge on every rule mismatch.
ENABLE_LLM_JUDGE="${ENABLE_LLM_JUDGE:-1}"
TOOLVISION_RL_USE_LLM_JUDGE="${TOOLVISION_RL_USE_LLM_JUDGE:-0}"

# If judge is enabled, accept either the LLM_JUDGE_* names used by reward.py or
# the OFFLINE_SFT_QWEN_* / DASHSCOPE_API_KEY names used for DashScope-compatible checks.
LLM_JUDGE_BASE_URL="${LLM_JUDGE_BASE_URL:-${OFFLINE_SFT_QWEN_BASE_URL:-}}"
LLM_JUDGE_MODEL_NAME="${LLM_JUDGE_MODEL_NAME:-${OFFLINE_SFT_QWEN_MODEL:-}}"
LLM_JUDGE_API_KEY="${LLM_JUDGE_API_KEY:-${OFFLINE_SFT_QWEN_API_KEY:-${DASHSCOPE_API_KEY:-${OPENAI_API_KEY:-}}}}"
LLM_JUDGE_TIMEOUT="${LLM_JUDGE_TIMEOUT:-100}"
LLM_JUDGE_MAX_RETRIES="${LLM_JUDGE_MAX_RETRIES:-3}"
LLM_JUDGE_ENABLE_THINKING="${LLM_JUDGE_ENABLE_THINKING:-0}"

STEP_REWARD_ENABLE="${STEP_REWARD_ENABLE:-False}"
STEP_REWARD_WEIGHT="${STEP_REWARD_WEIGHT:-0.2}"
STEP_REWARD_TAU="${STEP_REWARD_TAU:-0.1}"
STEP_REWARD_CAP="${STEP_REWARD_CAP:-0.5}"
STEP_JUDGE_BASE_URL="${STEP_JUDGE_BASE_URL:-}"
STEP_JUDGE_MODEL="${STEP_JUDGE_MODEL:-}"
STEP_JUDGE_API_KEY_ENV="${STEP_JUDGE_API_KEY_ENV:-STEP_JUDGE_API_KEY}"
STEP_JUDGE_TIMEOUT="${STEP_JUDGE_TIMEOUT:-60}"
STEP_JUDGE_MAX_RETRIES="${STEP_JUDGE_MAX_RETRIES:-1}"
STEP_JUDGE_MAX_IMAGES="${STEP_JUDGE_MAX_IMAGES:-4}"
STEP_JUDGE_MAX_OBSERVATION_CHARS="${STEP_JUDGE_MAX_OBSERVATION_CHARS:-4000}"
if [[ ! "${STEP_JUDGE_API_KEY_ENV}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "STEP_JUDGE_API_KEY_ENV must be a valid environment variable name, got: ${STEP_JUDGE_API_KEY_ENV}" >&2
  exit 1
fi
STEP_JUDGE_API_KEY_VALUE="${!STEP_JUDGE_API_KEY_ENV:-}"

shell_quote() {
  printf '%q' "$1"
}

append_env() {
  local name="$1"
  local value="$2"
  TRAIN_COMMAND+=" ${name}=$(shell_quote "${value}")"
}

if [[ -z "${WORKER_IMAGE}" || "${WORKER_IMAGE}" == *"你的"* || "${WORKER_IMAGE}" == *"TODO"* ]]; then
  echo "WORKER_IMAGE must be a real DLC image URI, got: ${WORKER_IMAGE}" >&2
  exit 1
fi

if [[ "${ENABLE_WANDB}" == "1" || "${ENABLE_WANDB,,}" == "true" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ENABLE_WANDB=1 requires WANDB_API_KEY in the submit environment." >&2
    echo "Export WANDB_API_KEY before running this script, or set ENABLE_WANDB=0." >&2
    exit 1
  fi
  TRAINER_LOGGER_VALUE='["console","wandb"]'
else
  TRAINER_LOGGER_VALUE='["console"]'
fi

if [[ "${ENABLE_LLM_JUDGE}" == "1" || "${ENABLE_LLM_JUDGE,,}" == "true" ]]; then
  if [[ -z "${LLM_JUDGE_BASE_URL}" || -z "${LLM_JUDGE_MODEL_NAME}" || -z "${LLM_JUDGE_API_KEY}" ]]; then
    echo "ENABLE_LLM_JUDGE=1 requires LLM_JUDGE_BASE_URL, LLM_JUDGE_MODEL_NAME, and LLM_JUDGE_API_KEY." >&2
    echo "The 40k RL set contains judge-family samples; set ENABLE_LLM_JUDGE=0 only if exact-only scoring is intentional." >&2
    exit 1
  fi
fi

if [[ "${STEP_REWARD_ENABLE}" == "1" || "${STEP_REWARD_ENABLE,,}" == "true" ]]; then
  if [[ -z "${STEP_JUDGE_BASE_URL}" || -z "${STEP_JUDGE_MODEL}" ]]; then
    echo "STEP_REWARD_ENABLE=True requires STEP_JUDGE_BASE_URL and STEP_JUDGE_MODEL." >&2
    echo "Keep STEP_REWARD_ENABLE=False until the step judge service is deployed." >&2
    exit 1
  fi
fi

TRAIN_COMMAND="cd $(shell_quote "${ROOT_DIR}") &&"
append_env MODEL_PATH "${MODEL_PATH}"
append_env TRAIN_FILES "${TRAIN_FILES_ARG}"
append_env TEST_FILES "${TEST_FILES_ARG}"
append_env TOOL_CFG_TEMPLATE_PATH "${TOOL_CFG_TEMPLATE_PATH}"
append_env SYSTEM_PROMPT_PATH "${SYSTEM_PROMPT_PATH}"
append_env PROJECT_NAME "${PROJECT_NAME}"
append_env EXP_NAME "${EXP_NAME}"
append_env SAVE_DIR "${SAVE_DIR}"
append_env ENABLE_TOOLS "1"
append_env OCR_BASE_URL "${OCR_BASE_URL}"
append_env GROUNDEDSAM2_BASE_URL "${GROUNDEDSAM2_BASE_URL}"
append_env DEPTH_BASE_URL "${DEPTH_BASE_URL}"
append_env COUNTGD_BASE_URL "${COUNTGD_BASE_URL}"
append_env CODEVISION_ENV "${CODEVISION_ENV}"
append_env DLC_ENTRYPOINT_DEBUG "${DLC_ENTRYPOINT_DEBUG:-1}"
append_env RAY_NODE_CHECK_TIMEOUT_SECONDS "${RAY_NODE_CHECK_TIMEOUT_SECONDS:-20}"
append_env TOOL_PREFLIGHT_CHECK "${TOOL_PREFLIGHT_CHECK:-1}"

# Full run does not run validation by default; TEST_FILES is kept only so the
# underlying recipe can still be overridden without editing this launcher.
append_env VAL_BEFORE_TRAIN "${VAL_BEFORE_TRAIN:-False}"
append_env TEST_FREQ "${TEST_FREQ:--1}"
append_env SAVE_FREQ "${SAVE_FREQ:-50}"
append_env MAX_ACTOR_CKPT_TO_KEEP "${MAX_ACTOR_CKPT_TO_KEEP:-5}"
append_env MAX_CRITIC_CKPT_TO_KEEP "${MAX_CRITIC_CKPT_TO_KEEP:-5}"
append_env RESUME_MODE "${RESUME_MODE:-auto}"
append_env RESUME_FROM_PATH "${RESUME_FROM_PATH:-null}"
append_env TRAIN_BSZ "${TRAIN_BSZ:-64}"
append_env DATA_SHUFFLE "${DATA_SHUFFLE:-True}"
append_env TRAIN_MINI_BSZ "${TRAIN_MINI_BSZ:-32}"
append_env TRAIN_MICRO_BSZ_PER_GPU "${TRAIN_MICRO_BSZ_PER_GPU:-1}"
append_env INFER_MICRO_BSZ_PER_GPU "${INFER_MICRO_BSZ_PER_GPU:-1}"
append_env N_RESP_PER_PROMPT "${N_RESP_PER_PROMPT:-8}"
append_env MAX_TURNS "${MAX_TURNS:-12}"
append_env TOTAL_EPOCHS "${TOTAL_EPOCHS:-1}"
append_env TOTAL_TRAINING_STEPS "${TOTAL_TRAINING_STEPS:-null}"
append_env INFER_TP_SIZE "${INFER_TP_SIZE:-4}"
append_env ROLLOUT_TEMPERATURE "${ROLLOUT_TEMPERATURE:-0.7}"
append_env ROLLOUT_TOP_P "${ROLLOUT_TOP_P:-0.95}"
append_env ROLLOUT_DO_SAMPLE "${ROLLOUT_DO_SAMPLE:-True}"
append_env ROLLOUT_MAX_TOKENS_PER_TURN "${ROLLOUT_MAX_TOKENS_PER_TURN:-2048}"
append_env ROLLOUT_ENABLE_PREFIX_CACHING "${ROLLOUT_ENABLE_PREFIX_CACHING:-True}"
append_env MAX_NUM_SEQS "${MAX_NUM_SEQS:-1024}"
append_env ROLLOUT_DATA_DIR "${ROLLOUT_DATA_DIR:-${SAVE_DIR}/rollout_generations}"
append_env TRAINER_LOGGER "${TRAINER_LOGGER_VALUE}"
append_env TOOLVISION_RL_USE_LLM_JUDGE "${TOOLVISION_RL_USE_LLM_JUDGE}"
append_env TOOL_REWARD_MODE "${TOOL_REWARD_MODE:-simple_penalty}"
append_env TOOL_REWARD_ALPHA "${TOOL_REWARD_ALPHA:-1.0}"
append_env TOOL_REWARD_BETA "${TOOL_REWARD_BETA:-0.0}"
append_env TOOL_REWARD_GAMMA "${TOOL_REWARD_GAMMA:-0.5}"
append_env TOOL_REWARD_DELTA "${TOOL_REWARD_DELTA:-0.5}"
append_env TOOL_REWARD_OVERUSE_WEIGHT "${TOOL_REWARD_OVERUSE_WEIGHT:-0.05}"
append_env TOOL_REWARD_TOOL_ERROR_WEIGHT "${TOOL_REWARD_TOOL_ERROR_WEIGHT:-0.2}"
append_env TOOL_REWARD_INVALID_CALL_WEIGHT "${TOOL_REWARD_INVALID_CALL_WEIGHT:-0.2}"
append_env TOOL_REWARD_OVERUSE_THRESHOLD "${TOOL_REWARD_OVERUSE_THRESHOLD:-4}"
append_env TOOL_REWARD_CLEAN_TOOL_WEIGHT "${TOOL_REWARD_CLEAN_TOOL_WEIGHT:-0.0}"
append_env TOOL_REWARD_CLEAN_TOOL_BASELINE_WEIGHT "${TOOL_REWARD_CLEAN_TOOL_BASELINE_WEIGHT:-0.05}"
append_env TOOL_REWARD_MUT_PROTOCOL_WEIGHT "${TOOL_REWARD_MUT_PROTOCOL_WEIGHT:-0.2}"
append_env TOOL_REWARD_MUT_TURN_PENALTY_WEIGHT "${TOOL_REWARD_MUT_TURN_PENALTY_WEIGHT:-0.05}"
append_env TOOL_REWARD_MUT_TURN_PENALTY_THRESHOLD "${TOOL_REWARD_MUT_TURN_PENALTY_THRESHOLD:-6}"
append_env STEP_REWARD_ENABLE "${STEP_REWARD_ENABLE}"
append_env STEP_REWARD_WEIGHT "${STEP_REWARD_WEIGHT}"
append_env STEP_REWARD_TAU "${STEP_REWARD_TAU}"
append_env STEP_REWARD_CAP "${STEP_REWARD_CAP}"
append_env STEP_JUDGE_BASE_URL "${STEP_JUDGE_BASE_URL}"
append_env STEP_JUDGE_MODEL "${STEP_JUDGE_MODEL}"
append_env STEP_JUDGE_API_KEY_ENV "${STEP_JUDGE_API_KEY_ENV}"
append_env STEP_JUDGE_TIMEOUT "${STEP_JUDGE_TIMEOUT}"
append_env STEP_JUDGE_MAX_RETRIES "${STEP_JUDGE_MAX_RETRIES}"
append_env STEP_JUDGE_MAX_IMAGES "${STEP_JUDGE_MAX_IMAGES}"
append_env STEP_JUDGE_MAX_OBSERVATION_CHARS "${STEP_JUDGE_MAX_OBSERVATION_CHARS}"
if [[ -n "${STEP_JUDGE_API_KEY_VALUE}" ]]; then
  append_env "${STEP_JUDGE_API_KEY_ENV}" "${STEP_JUDGE_API_KEY_VALUE}"
fi
append_env REWARD_LAUNCH_ASYNC "${REWARD_LAUNCH_ASYNC:-False}"
append_env LOG_TRAIN_GENERATIONS "${LOG_TRAIN_GENERATIONS:-8}"
append_env LOG_VAL_GENERATIONS "${LOG_VAL_GENERATIONS:-8}"
append_env LOG_TRAIN_FREQ "${LOG_TRAIN_FREQ:-20}"
append_env FORMAT_REWARD_WEIGHT "${FORMAT_REWARD_WEIGHT:-0.2}"
append_env EXEC_REWARD_WEIGHT "${EXEC_REWARD_WEIGHT:-0.0}"
append_env EMERGE_REWARD_WEIGHT "${EMERGE_REWARD_WEIGHT:-0.2}"

if [[ "${ENABLE_WANDB}" == "1" || "${ENABLE_WANDB,,}" == "true" ]]; then
  append_env WANDB_MODE "${WANDB_MODE}"
  append_env WANDB_API_KEY "${WANDB_API_KEY}"
  [[ -n "${WANDB_ENTITY}" ]] && append_env WANDB_ENTITY "${WANDB_ENTITY}"
  [[ -n "${WANDB_BASE_URL}" ]] && append_env WANDB_BASE_URL "${WANDB_BASE_URL}"
fi

if [[ "${ENABLE_LLM_JUDGE}" == "1" || "${ENABLE_LLM_JUDGE,,}" == "true" ]]; then
  append_env LLM_JUDGE_BASE_URL "${LLM_JUDGE_BASE_URL}"
  append_env LLM_JUDGE_MODEL_NAME "${LLM_JUDGE_MODEL_NAME}"
  append_env LLM_JUDGE_API_KEY "${LLM_JUDGE_API_KEY}"
  append_env LLM_JUDGE_TIMEOUT "${LLM_JUDGE_TIMEOUT}"
  append_env LLM_JUDGE_MAX_RETRIES "${LLM_JUDGE_MAX_RETRIES}"
  append_env LLM_JUDGE_ENABLE_THINKING "${LLM_JUDGE_ENABLE_THINKING}"
fi

TRAIN_COMMAND+=" bash scripts/dlc_ray_direct_entrypoint.sh"

echo "Submitting ${JOB_NAME}"
echo "PROJECT_NAME=${PROJECT_NAME}"
echo "EXP_NAME=${EXP_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_FILES=${TRAIN_FILES_ARG}"
echo "TEST_FILES=${TEST_FILES_ARG}"
echo "TOOL_CFG_TEMPLATE_PATH=${TOOL_CFG_TEMPLATE_PATH:-<default>}"
echo "SYSTEM_PROMPT_PATH=${SYSTEM_PROMPT_PATH:-<default>}"
echo "SAVE_DIR=${SAVE_DIR}"
echo "OCR_BASE_URL=${OCR_BASE_URL}"
echo "GROUNDEDSAM2_BASE_URL=${GROUNDEDSAM2_BASE_URL}"
echo "DEPTH_BASE_URL=${DEPTH_BASE_URL}"
echo "COUNTGD_BASE_URL=${COUNTGD_BASE_URL}"
echo "CODEVISION_ENV=${CODEVISION_ENV}"
echo "ENABLE_WANDB=${ENABLE_WANDB}"
echo "ENABLE_LLM_JUDGE=${ENABLE_LLM_JUDGE}"
echo "LLM_JUDGE_BASE_URL=$([[ -n "${LLM_JUDGE_BASE_URL}" ]] && echo "${LLM_JUDGE_BASE_URL}" || echo '<unset>')"
echo "LLM_JUDGE_MODEL_NAME=$([[ -n "${LLM_JUDGE_MODEL_NAME}" ]] && echo "${LLM_JUDGE_MODEL_NAME}" || echo '<unset>')"
echo "TRAIN_BSZ=${TRAIN_BSZ:-64}"
echo "DATA_SHUFFLE=${DATA_SHUFFLE:-True}"
echo "N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}"
echo "TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}"
echo "SAVE_FREQ=${SAVE_FREQ:-50}"
echo "RESUME_MODE=${RESUME_MODE:-auto}"
echo "RESUME_FROM_PATH=${RESUME_FROM_PATH:-null}"
echo "ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}"
echo "ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.95}"
echo "ROLLOUT_DO_SAMPLE=${ROLLOUT_DO_SAMPLE:-True}"
echo "ROLLOUT_MAX_TOKENS_PER_TURN=${ROLLOUT_MAX_TOKENS_PER_TURN:-2048}"
echo "ROLLOUT_ENABLE_PREFIX_CACHING=${ROLLOUT_ENABLE_PREFIX_CACHING:-True}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS:-1024}"
echo "ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${SAVE_DIR}/rollout_generations}"
echo "TOOL_REWARD_MODE=${TOOL_REWARD_MODE:-simple_penalty}"
echo "TOOL_REWARD_BETA=${TOOL_REWARD_BETA:-0.0}"
echo "TOOL_REWARD_MUT_PROTOCOL_WEIGHT=${TOOL_REWARD_MUT_PROTOCOL_WEIGHT:-0.2}"
echo "TOOL_REWARD_MUT_TURN_PENALTY_WEIGHT=${TOOL_REWARD_MUT_TURN_PENALTY_WEIGHT:-0.05}"
echo "TOOL_REWARD_MUT_TURN_PENALTY_THRESHOLD=${TOOL_REWARD_MUT_TURN_PENALTY_THRESHOLD:-6}"
echo "STEP_REWARD_ENABLE=${STEP_REWARD_ENABLE}"
echo "STEP_REWARD_WEIGHT=${STEP_REWARD_WEIGHT}"
echo "STEP_REWARD_TAU=${STEP_REWARD_TAU}"
echo "STEP_REWARD_CAP=${STEP_REWARD_CAP}"
echo "STEP_JUDGE_BASE_URL=$([[ -n "${STEP_JUDGE_BASE_URL}" ]] && echo "${STEP_JUDGE_BASE_URL}" || echo '<unset>')"
echo "STEP_JUDGE_MODEL=$([[ -n "${STEP_JUDGE_MODEL}" ]] && echo "${STEP_JUDGE_MODEL}" || echo '<unset>')"
echo "STEP_JUDGE_API_KEY_ENV=${STEP_JUDGE_API_KEY_ENV}"
echo "FORMAT_REWARD_WEIGHT=${FORMAT_REWARD_WEIGHT:-0.2}"

dry_run_flag="${DRY_RUN:-0}"
if [[ "${dry_run_flag}" == "1" || "${dry_run_flag,,}" == "true" ]]; then
  echo "DRY_RUN=1, not submitting."
  dry_run_command="${DLC_BIN} submit pytorchjob --name=${JOB_NAME} --command=$(shell_quote "${TRAIN_COMMAND}") ..."
  printf '%s\n' "${dry_run_command}" | sed \
    -e 's/\(WANDB_API_KEY=\)[^\\ ]*/\1<redacted>/g' \
    -e 's/\(LLM_JUDGE_API_KEY=\)[^\\ ]*/\1<redacted>/g' \
    -e 's/\(STEP_JUDGE_API_KEY=\)[^\\ ]*/\1<redacted>/g' \
    -e "s/\(${STEP_JUDGE_API_KEY_ENV}=\)[^\\ ]*/\1<redacted>/g" \
    -e 's/\(OPENAI_API_KEY=\)[^\\ ]*/\1<redacted>/g' \
    -e 's/\(DASHSCOPE_API_KEY=\)[^\\ ]*/\1<redacted>/g' \
    -e 's/\(OFFLINE_SFT_QWEN_API_KEY=\)[^\\ ]*/\1<redacted>/g'
  exit 0
fi

"${DLC_BIN}" submit pytorchjob \
  --name="${JOB_NAME}" \
  --command="${TRAIN_COMMAND}" \
  --data_source_uris="${DATA_SOURCE_URIS:-cpfs://cpfs-298fffb575a502fe.cn-wulanchabu/ptc-29f47d9393ad2b16/exp-29f2869e7d984aa6/::/mnt/cpfs,oss://pai-wlcb-ai-oss.oss-cn-wulanchabu-internal.aliyuncs.com/::/mnt/oss}" \
  --resource_id="${RESOURCE_ID:-quotaev2tl4w6aw0}" \
  --workspace_id="${WORKSPACE_ID:-240810}" \
  --vpc_id="${VPC_ID:-vpc-0jl5rpw5qokp6p2ettip6}" \
  --switch_id="${SWITCH_ID:-vsw-0jlmr9rjzed093yr9c0kz}" \
  --security_group_id="${SECURITY_GROUP_ID:-sg-0jl0pd5qaerdj75wmred}" \
  --priority="${PRIORITY:-8}" \
  --extended_cidrs="${EXTENDED_CIDRS:-10.1.255.0/29,10.1.255.8/29,10.1.16.0/20}" \
  --advanced_settings="${ADVANCED_SETTINGS:-createSvcForAllWorkers=true,customPortList=6379;6380-6383;8265;20000-25000}" \
  --workers="${DLC_WORKERS:-2}" \
  --worker_image="${WORKER_IMAGE}" \
  --worker_cpu="${WORKER_CPU:-110}" \
  --worker_memory="${WORKER_MEMORY:-1500Gi}" \
  --worker_shared_memory="${WORKER_SHARED_MEMORY:-1500Gi}" \
  --worker_gpu="${WORKER_GPU:-8}"
