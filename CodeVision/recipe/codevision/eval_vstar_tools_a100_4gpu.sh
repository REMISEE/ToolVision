#!/usr/bin/env bash
# =============================================================================
# V*Bench tool-enabled eval on a single 4xA100 node.
#
# This script is intentionally separate from eval_vstar_base.sh so the base
# path can remain stable while we iterate on the tool-enabled setup.
#
# Expected flow:
#   1) Prepare normalized eval inputs:
#       python recipe/codevision/prepare_vstar_bench.py \
#         --output-parquet /mnt/cpfs/delinmao/Benchmarks/vstar-bench/vstar_codevision_eval.parquet
#   2) Start external services (usually via the companion slurm wrapper).
#   3) Run:
#       bash recipe/codevision/eval_vstar_tools_a100_4gpu.sh
#
# Override MODEL_PATH to test the tool-SFT checkpoint.
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(pwd)"
PROJECT_NAME="CodeVision"
EXP_NAME="${EXP_NAME:-vstar_tools_a100_4gpu}"
SAVE_DIR="./saves/${PROJECT_NAME}/${EXP_NAME}"

export HYDRA_FULL_ERROR=1
export LLM_JUDGE_BASE_URL="${LLM_JUDGE_BASE_URL:-}"
export LLM_JUDGE_MODEL_NAME="${LLM_JUDGE_MODEL_NAME:-}"
export LLM_JUDGE_API_KEY="${LLM_JUDGE_API_KEY:-${OPENAI_API_KEY:-}}"
export LLM_JUDGE_TIMEOUT="${LLM_JUDGE_TIMEOUT:-100}"
export LLM_JUDGE_MAX_RETRIES="${LLM_JUDGE_MAX_RETRIES:-3}"

NNODES=1
NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/cpfs/delinmao}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${WORKSPACE_ROOT}/Benchmarks}"
MODEL_PATH="${MODEL_PATH:-${WORKSPACE_ROOT}/outputs/qwen3vl_sft/full}"
DEFAULT_EVAL_PARQUET="${BENCHMARK_ROOT}/vstar-bench/vstar_codevision_eval.parquet"
EVAL_PARQUET="${EVAL_PARQUET:-${VSTAR_PARQUET:-${DEFAULT_EVAL_PARQUET}}}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"

test_files="['${EVAL_PARQUET}']"
train_files="${test_files}"

adv_estimator="grpo"
loss_agg_mode="token-mean"
clip_ratio_low=0.2
clip_ratio_high=0.28
clip_ratio_c=10.0
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

cfg_path="${PROJECT_DIR}/recipe/codevision/config"
cfg_name="grpo_trainer"
tool_cfg_path="${TOOL_CFG_TEMPLATE_PATH:-recipe/codevision/config/code_image_tool_config_v02.yaml}"
new_sp_path="${SYSTEM_PROMPT_PATH:-recipe/codevision/config/sp2.txt}"

OCR_BASE_URL="${OCR_BASE_URL:-http://127.0.0.1:8080}"
GROUNDEDSAM2_BASE_URL="${GROUNDEDSAM2_BASE_URL:-http://127.0.0.1:8081}"
DEPTH_BASE_URL="${DEPTH_BASE_URL:-http://127.0.0.1:8082}"
COUNTGD_BASE_URL="${COUNTGD_BASE_URL:-http://127.0.0.1:8083}"

train_bsz=64
train_mini_bsz=32
train_micro_bsz_per_gpu=1
infer_micro_bsz_per_gpu=1

VAL_BSZ="${VAL_BSZ:-32}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
VAL_N_RESP_PER_PROMPT="${VAL_N_RESP_PER_PROMPT:-1}"
MAX_TURNS="${MAX_TURNS:-12}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0}"
VAL_TOP_P="${VAL_TOP_P:-1.0}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-False}"
ROLLOUT_MAX_TOKENS_PER_TURN="${ROLLOUT_MAX_TOKENS_PER_TURN:-2048}"

max_prompt_len=$((1024 * 16))
max_resp_len=$((1024 * 16))
max_tool_resp_len=$((1024 * 10))
max_image_resolution=$((1024 * 8 * 28 * 28))

offload=True
train_sp_size=1
INFER_TP_SIZE="${INFER_TP_SIZE:-4}"
use_dynamic_bsz=False
actor_ppo_max_token_len=$((max_prompt_len + max_resp_len))
infer_ppo_max_token_len=$((max_prompt_len + max_resp_len))
max_num_batched_tokens=$((max_prompt_len + max_resp_len))
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
RAY_INIT_NUM_CPUS="${RAY_INIT_NUM_CPUS:-}"
RAY_INIT_INCLUDE_DASHBOARD="${RAY_INIT_INCLUDE_DASHBOARD:-}"

tool_reward_enable=True
tool_reward_alpha=1.0
tool_reward_beta=0.0
tool_reward_gamma=0.5
tool_reward_delta=0.5
format_reward_weight=0.1
exec_reward_weight=0.0
emerge_reward_weight=0.2

exploration_reward_enable=False
exploration_reward_weight=0.0
exploration_decay_steps=0

val_before_train=True
test_freq=20
save_freq=400
total_epochs=1
log_val_generations=8
log_train_generations=8
log_train_freq=20

ONLY_TEST=True
mkdir -p "${SAVE_DIR}"
VAL_METRICS_OUTPUT="${SAVE_DIR}/metrics.json"
RUNTIME_TOOL_CFG_PATH="${SAVE_DIR}/tool_config.runtime.yaml"
SAVE_VAL_GENERATIONS="${SAVE_VAL_GENERATIONS:-0}"
if [[ "${SAVE_VAL_GENERATIONS}" == "1" || "${SAVE_VAL_GENERATIONS,,}" == "true" || "${SAVE_VAL_GENERATIONS,,}" == "yes" ]]; then
  VAL_DATA_DIR="${VAL_DATA_DIR:-${SAVE_DIR}/generations}"
else
  VAL_DATA_DIR="${VAL_DATA_DIR:-}"
fi
SAVE_EVAL_METADATA="${SAVE_EVAL_METADATA:-1}"
if [[ "${SAVE_EVAL_METADATA}" == "1" || "${SAVE_EVAL_METADATA,,}" == "true" || "${SAVE_EVAL_METADATA,,}" == "yes" ]]; then
  DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-${SAVE_DIR}/diagnostics}"
else
  DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-}"
fi
DIAGNOSTIC_MAX_PER_BUCKET="${DIAGNOSTIC_MAX_PER_BUCKET:-50}"
DIAGNOSTIC_SAMPLE_SEED="${DIAGNOSTIC_SAMPLE_SEED:-42}"
SAVE_FULL_TRAJECTORY_ALL="${SAVE_FULL_TRAJECTORY_ALL:-0}"
STREAM_VALIDATION_DUMP="${STREAM_VALIDATION_DUMP:-True}"

python3 - <<'PY' \
  "${tool_cfg_path}" \
  "${RUNTIME_TOOL_CFG_PATH}" \
  "${OCR_BASE_URL}" \
  "${GROUNDEDSAM2_BASE_URL}" \
  "${DEPTH_BASE_URL}" \
  "${COUNTGD_BASE_URL}"
from pathlib import Path
import sys
import yaml

src_path = Path(sys.argv[1])
dst_path = Path(sys.argv[2])
ocr_base_url = sys.argv[3]
groundedsam2_base_url = sys.argv[4]
depth_base_url = sys.argv[5]
countgd_base_url = sys.argv[6]

config = yaml.safe_load(src_path.read_text(encoding="utf-8"))
code_schema = config["tools"][0]["tool_schema"]["function"]["parameters"]["properties"]["code"]
code_schema["description"] = code_schema["description"].replace("\n", "\\n")
services = config["tools"][0]["config"]["external_services"]
services["paddleocr"]["base_url"] = ocr_base_url
if "paddleocr_vl" in services:
    services["paddleocr_vl"]["serving_base_url"] = ocr_base_url
services["grounded_sam2"]["base_url"] = groundedsam2_base_url
services["depth"]["base_url"] = depth_base_url
services["countgd"]["base_url"] = countgd_base_url
dst_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY
tool_cfg_path="${RUNTIME_TOOL_CFG_PATH}"

echo "MODEL_PATH=${MODEL_PATH}"
echo "EVAL_PARQUET=${EVAL_PARQUET}"
echo "NGPUS_PER_NODE=${NGPUS_PER_NODE}"
echo "INFER_TP_SIZE=${INFER_TP_SIZE}"
echo "VAL_BSZ=${VAL_BSZ}"
echo "N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT}"
echo "VAL_N_RESP_PER_PROMPT=${VAL_N_RESP_PER_PROMPT}"
echo "MAX_TURNS=${MAX_TURNS}"
echo "VAL_TEMPERATURE=${VAL_TEMPERATURE}"
echo "VAL_TOP_P=${VAL_TOP_P}"
echo "VAL_DO_SAMPLE=${VAL_DO_SAMPLE}"
echo "ROLLOUT_MAX_TOKENS_PER_TURN=${ROLLOUT_MAX_TOKENS_PER_TURN}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "RAY_INIT_NUM_CPUS=${RAY_INIT_NUM_CPUS:-<auto>}"
echo "RAY_INIT_INCLUDE_DASHBOARD=${RAY_INIT_INCLUDE_DASHBOARD:-<auto>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
echo "OCR_BASE_URL=${OCR_BASE_URL}"
echo "GROUNDEDSAM2_BASE_URL=${GROUNDEDSAM2_BASE_URL}"
echo "DEPTH_BASE_URL=${DEPTH_BASE_URL}"
echo "COUNTGD_BASE_URL=${COUNTGD_BASE_URL}"
echo "TOOL_CFG_PATH=${tool_cfg_path}"
echo "SYSTEM_PROMPT_PATH=${new_sp_path}"
echo "SAVE_VAL_GENERATIONS=${SAVE_VAL_GENERATIONS}"
echo "VAL_DATA_DIR=${VAL_DATA_DIR:-<disabled>}"
echo "SAVE_EVAL_METADATA=${SAVE_EVAL_METADATA}"
echo "DIAGNOSTICS_DIR=${DIAGNOSTICS_DIR:-<disabled>}"
echo "DIAGNOSTIC_MAX_PER_BUCKET=${DIAGNOSTIC_MAX_PER_BUCKET}"
echo "SAVE_FULL_TRAJECTORY_ALL=${SAVE_FULL_TRAJECTORY_ALL}"
echo "STREAM_VALIDATION_DUMP=${STREAM_VALIDATION_DUMP}"
if [[ -n "${LLM_JUDGE_BASE_URL}" ]]; then
  echo "LLM_JUDGE_BASE_URL=${LLM_JUDGE_BASE_URL}"
  echo "LLM_JUDGE_MODEL_NAME=${LLM_JUDGE_MODEL_NAME:-<auto>}"
  echo "LLM_JUDGE_API_KEY_SET=$([[ -n "${LLM_JUDGE_API_KEY}" ]] && echo yes || echo no)"
else
  echo "LLM_JUDGE_BASE_URL=<disabled>"
fi

ray_init_extra_args=()
if [[ -n "${RAY_INIT_INCLUDE_DASHBOARD}" ]]; then
  ray_init_extra_args+=("+ray_kwargs.ray_init.include_dashboard=${RAY_INIT_INCLUDE_DASHBOARD}")
fi

python3 -m verl.trainer.main_ppo \
    --config-path=${cfg_path} \
    --config-name=${cfg_name} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    data.train_batch_size=${train_bsz} \
    data.val_batch_size=${VAL_BSZ} \
    data.max_prompt_length=${max_prompt_len} \
    data.max_response_length=${max_resp_len} \
    data.filter_overlong_prompts=False \
    data.truncation="error" \
    data.return_raw_chat=True \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.return_multi_modal_inputs=False \
    +data.replace_system_prompt=True \
    +data.new_sp_path=${new_sp_path} \
    +data.enable_image_resize=True \
    +data.max_image_resolution=${max_image_resolution} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_micro_bsz_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${train_sp_size} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N_RESP_PER_PROMPT} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${INFER_TP_SIZE} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_response_tokens_per_turn=${ROLLOUT_MAX_TOKENS_PER_TURN} \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${max_tool_resp_len} \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${tool_cfg_path} \
    actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_AGENT_NUM_WORKERS} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    +reward_model.tool_reward.enable=${tool_reward_enable} \
    +reward_model.tool_reward.alpha=${tool_reward_alpha} \
    +reward_model.tool_reward.beta=${tool_reward_beta} \
    +reward_model.tool_reward.gamma=${tool_reward_gamma} \
    +reward_model.tool_reward.delta=${tool_reward_delta} \
    +reward_model.format_reward_weight=${format_reward_weight} \
    +reward_model.exec_reward_weight=${exec_reward_weight} \
    +reward_model.emerge_reward_weight=${emerge_reward_weight} \
    +reward_model.exploration_reward.enable=${exploration_reward_enable} \
    +reward_model.exploration_reward.weight=${exploration_reward_weight} \
    +reward_model.exploration_reward.decay_steps=${exploration_decay_steps} \
    trainer.critic_warmup=0 \
    trainer.val_before_train=${val_before_train} \
    trainer.logger='["console"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    ray_kwargs.ray_init.num_cpus=${RAY_INIT_NUM_CPUS:-null} \
    "${ray_init_extra_args[@]}" \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.log_val_generations=${log_val_generations} \
    +trainer.log_train_generations=${log_train_generations} \
    +trainer.log_train_freq=${log_train_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.resume_from_path=${RESUME_FROM_PATH} \
    +trainer.only_test=${ONLY_TEST} \
    +trainer.validation_data_dir=${VAL_DATA_DIR:-null} \
    +trainer.diagnostics_dir=${DIAGNOSTICS_DIR:-null} \
    +trainer.diagnostic_max_per_bucket=${DIAGNOSTIC_MAX_PER_BUCKET} \
    +trainer.diagnostic_sample_seed=${DIAGNOSTIC_SAMPLE_SEED} \
    +trainer.save_full_trajectory_all=${SAVE_FULL_TRAJECTORY_ALL} \
    +trainer.stream_validation_dump=${STREAM_VALIDATION_DUMP} \
    +trainer.val_metrics_output=${VAL_METRICS_OUTPUT}
