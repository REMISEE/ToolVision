#!/usr/bin/env bash
set -euo pipefail

# Evaluate the 40k ToolVision/CodeVision parquet with 16 sampled completions per
# question and dump every generation for later per-question pass@16 aggregation.
#
# This is intentionally an evaluation-only launcher. By default it uses the raw
# prompt in the parquet, disables tools, and does not replace the system prompt,
# matching the earlier Innovator-style pass@16 difficulty probe more closely than
# the RL training launcher.

PROJECT_DIR="${PROJECT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/cpfs/delinmao/envs/codevision_new/bin/python3}"

MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/models/Qwen3-VL-8B-Instruct}"
DATA_PARQUET="${DATA_PARQUET:-/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_clean_no_extra_image/parquet/train.parquet}"
OUT_DIR="${OUT_DIR:-/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_pass16}"

EXP_NAME="${EXP_NAME:-qwen3vl8b_raw40k_pass16}"
SAVE_DIR="${SAVE_DIR:-${OUT_DIR}/saves}"
VAL_DATA_DIR="${VAL_DATA_DIR:-${OUT_DIR}/validation_generations}"
VAL_METRICS_OUTPUT="${VAL_METRICS_OUTPUT:-${OUT_DIR}/metrics.json}"

N_SAMPLES="${N_SAMPLES:-16}"
VAL_BSZ="${VAL_BSZ:-128}"
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
INFER_TP_SIZE="${INFER_TP_SIZE:-1}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-16384}"
MAX_RESP_LEN="${MAX_RESP_LEN:-4096}"
MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION:-6422528}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LEN + MAX_RESP_LEN))}"
RAY_INIT_NUM_CPUS="${RAY_INIT_NUM_CPUS:-32}"

# Sampling settings. Use sampled decoding for pass@16; override from env if
# another policy is desired.
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-True}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"
VAL_TOP_P="${VAL_TOP_P:-1.0}"
VAL_TOP_K="${VAL_TOP_K:--1}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LLM_JUDGE_BASE_URL="${LLM_JUDGE_BASE_URL:-}"

if [[ "${DISABLE_PROXY:-1}" == "1" ]]; then
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
fi

cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}" "${SAVE_DIR}" "${VAL_DATA_DIR}"

echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_PARQUET=${DATA_PARQUET}"
echo "OUT_DIR=${OUT_DIR}"
echo "N_SAMPLES=${N_SAMPLES}"
echo "VAL_DATA_DIR=${VAL_DATA_DIR}"
echo "VAL_METRICS_OUTPUT=${VAL_METRICS_OUTPUT}"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  --config-path="${PROJECT_DIR}/recipe/codevision/config" \
  --config-name=grpo_trainer \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.train_batch_size=1 \
  data.val_batch_size="${VAL_BSZ}" \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESP_LEN}" \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.return_raw_chat=True \
  data.train_files="['${DATA_PARQUET}']" \
  data.val_files="['${DATA_PARQUET}']" \
  data.return_multi_modal_inputs=False \
  data.replace_system_prompt=False \
  data.enable_image_resize=True \
  data.max_image_resolution="${MAX_IMAGE_RESOLUTION}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${INFER_TP_SIZE}" \
  actor_rollout_ref.rollout.multi_turn.enable=False \
  actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_NUM_WORKERS}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.val_kwargs.n="${N_SAMPLES}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P}" \
  actor_rollout_ref.rollout.val_kwargs.top_k="${VAL_TOP_K}" \
  +reward_model.tool_reward.enable=False \
  +reward_model.format_reward_weight=0.0 \
  +reward_model.exec_reward_weight=0.0 \
  +reward_model.emerge_reward_weight=0.0 \
  trainer.logger='["console"]' \
  trainer.project_name=CodeVision \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.default_local_dir="${SAVE_DIR}" \
  trainer.resume_mode=disable \
  trainer.val_before_train=True \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  trainer.total_epochs=1 \
  trainer.log_val_generations=0 \
  +trainer.only_test=True \
  +trainer.val_metrics_output="${VAL_METRICS_OUTPUT}" \
  +trainer.validation_data_dir="${VAL_DATA_DIR}" \
  ray_kwargs.ray_init.num_cpus="${RAY_INIT_NUM_CPUS}"
