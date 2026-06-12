#!/usr/bin/env bash
# =============================================================================
# V*Bench eval on local Qwen3-VL (base by default), single-node 8x4090.
#
# This script is a trimmed-down copy of `eval.sh` tailored for:
#   1) running `verl.trainer.main_ppo` directly (no `ray job submit`), letting
#      `ray.init()` auto-start a local single-node cluster;
#   2) evaluating on V*Bench with per-category metrics
#      (data_source == category in the parquet -> val-<category>/accuracy/mean);
#   3) being safe on 8x24GB RTX 4090 GPUs.
#
# Prerequisites:
#   - cd /mnt/cpfs/delinmao/ToolVision/CodeVision
#   - export PYTHONPATH=$(pwd):$PYTHONPATH
#   - (first time) convert jsonl -> parquet:
#       python recipe/codevision/tools/convert_vstar_to_parquet.py \
#           --root /mnt/cpfs/delinmao/Benchmarks/vstar-bench \
#           --out  /mnt/cpfs/delinmao/Benchmarks/vstar-bench/vstar_eval.parquet
#
# Run:
#   bash recipe/codevision/eval_vstar_base.sh
#
# After it finishes:
#   python recipe/codevision/tools/aggregate_vstar_metrics.py \
#       ./saves/CodeVision/vstar_base/metrics.json
#
# To evaluate the SFT checkpoint instead, only change MODEL_PATH and EXP_NAME
# (and optionally start external services per
# docs/external_services_quickstart_20260327.md if you want real tool calls):
#   MODEL_PATH=/mnt/cpfs/delinmao/outputs/qwen3vl_sft/full
#   EXP_NAME=vstar_sft
# =============================================================================

set -euo pipefail

# ------------------------------------------------------------
#  Experiment Metadata
# ------------------------------------------------------------
PROJECT_DIR="$(pwd)"
PROJECT_NAME="CodeVision"
EXP_NAME="${EXP_NAME:-vstar_base}"
SAVE_DIR="./saves/${PROJECT_NAME}/${EXP_NAME}"

# ------------------------------------------------------------
#  Runtime env (replaces verl/trainer/runtime_env.yaml for local run)
# ------------------------------------------------------------
export HYDRA_FULL_ERROR=1
# export VLLM_USE_V1=1
# export VLLM_ALLREDUCE_USE_SYMM_MEM=0
# export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here: vLLM v1 uses
# CuMemAllocator and explicitly asserts it is incompatible with expandable segments
# (see vllm/device_allocator/cumem.py). If you need it for pure PyTorch, export it
# only in a shell that does not run this script.

# LLM judge fallback is only invoked when rule-based extract fails (non A/B/C/D).
# V*Bench is A/B/C/D so this can stay empty; set it if you want judge fallback.
export LLM_JUDGE_BASE_URL="${LLM_JUDGE_BASE_URL:-}"

# ------------------------------------------------------------
#  Ray & Cluster Settings (single-node 8 GPUs)
# ------------------------------------------------------------
NNODES=1
NGPUS_PER_NODE=8

# ------------------------------------------------------------
#  Model & Data Paths
# ------------------------------------------------------------
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/cpfs/delinmao}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${WORKSPACE_ROOT}/Benchmarks}"
MODEL_PATH="${MODEL_PATH:-${WORKSPACE_ROOT}/models/Qwen3-VL-8B-Thinking}"

VSTAR_PARQUET="${VSTAR_PARQUET:-${BENCHMARK_ROOT}/vstar-bench/vstar_eval.parquet}"

# main_ppo.py unconditionally constructs train_dataset, so we point train_files
# to the same parquet (only_test returns before the training loop actually uses it).
test_files="['${VSTAR_PARQUET}']"
train_files="${test_files}"

# ------------------------------------------------------------
#  Core Algorithm Hyper-parameters (kept for config compatibility)
# ------------------------------------------------------------
adv_estimator="grpo"
loss_agg_mode="token-mean"
clip_ratio_low=0.2
clip_ratio_high=0.28
clip_ratio_c=10.0
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

# Config File
cfg_path="${PROJECT_DIR}/recipe/codevision/config"
cfg_name="grpo_trainer"
tool_cfg_path="recipe/codevision/config/code_image_tool_config.yaml"
new_sp_path="recipe/codevision/config/sp.txt"

# ------------------------------------------------------------
#  Batch Size (eval-oriented)
# ------------------------------------------------------------
# train_* values are only consumed for config parsing; only_test doesn't iterate.
train_bsz=64
train_mini_bsz=32
train_micro_bsz_per_gpu=1
infer_micro_bsz_per_gpu=1

# Eval settings (4090: keep val_bsz small — multimodal merge OOMs easily at 64)
val_bsz=32
n_resp_per_prompt=1
max_turns=6

# ------------------------------------------------------------
#  Sequence Length  (4090 24GB: balance KV vs real prompt length)
# ------------------------------------------------------------
# Qwen3-VL + sp.txt + injected tool schema can easily exceed 8K *tokens* even
# when the raw user text is short (vision tokens + long system). 8K caused:
#   NotImplementedError: sequence_length=8463 > max_length=8192
# Keep response budget modest (V* answers are one letter).
# Avoid 16K+8K on 4090: vLLM Qwen3-VL multimodal forward + val_bsz>1 can OOM in
# merge_multimodal_embeddings (masked_scatter). 13K prompt > observed ~8463 tok.
max_prompt_len=$((1024 * 13))
max_resp_len=$((1024 * 4))
max_tool_resp_len=$((1024 * 6))
# Cap vision pixels to cut vision-token count (V* images are usually <2MP).
max_image_resolution=$((2 * 1024 * 1024))

# ------------------------------------------------------------
#  Performance (4090-safe)
# ------------------------------------------------------------
offload=True
train_sp_size=1
infer_tp_size=2
use_dynamic_bsz=False
actor_ppo_max_token_len=$((max_prompt_len + max_resp_len))
infer_ppo_max_token_len=$((max_prompt_len + max_resp_len))
# vLLM batching / admission caps (critical for 24GB KV budget)
max_num_batched_tokens=$((max_prompt_len + max_resp_len))
max_num_seqs=32
# hybrid_engine + param/optimizer offload + free_cache_engine means during
# rollout FSDP params live on CPU, so vLLM can take most of the GPU.
gpu_memory_utilization=0.85

# ------------------------------------------------------------
#  Reward Model (kept same as training config for parity)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
#  Trainer Schedule & Logging
# ------------------------------------------------------------
val_before_train=True
test_freq=20
save_freq=400
total_epochs=1
log_val_generations=8
log_train_generations=8
log_train_freq=20

# ------------------------------------------------------------
#  Only-test mode
# ------------------------------------------------------------
ONLY_TEST=True
mkdir -p "${SAVE_DIR}"
VAL_METRICS_OUTPUT="${SAVE_DIR}/metrics.json"

# ------------------------------------------------------------
#  Launch (local ray, no `ray job submit`)
# ------------------------------------------------------------
python3 -m verl.trainer.main_ppo \
    --config-path=${cfg_path} \
    --config-name=${cfg_name} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    data.train_batch_size=${train_bsz} \
    data.val_batch_size=${val_bsz} \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp_size} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${max_tool_resp_len} \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${tool_cfg_path} \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs} \
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
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.log_val_generations=${log_val_generations} \
    +trainer.log_train_generations=${log_train_generations} \
    +trainer.log_train_freq=${log_train_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.resume_mode=auto \
    +trainer.only_test=${ONLY_TEST} \
    +trainer.val_metrics_output=${VAL_METRICS_OUTPUT}
