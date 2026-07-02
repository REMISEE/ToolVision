#!/usr/bin/env bash
set -euo pipefail

# Direct-driver GSPO launcher. This is intentionally separate from
# qwen3_vl_gspo.sh, which uses Ray Jobs.

PROJECT_DIR="$(pwd)"
PROJECT_NAME="${PROJECT_NAME:-CodeVision}"
EXP_NAME="${EXP_NAME:-qwen3vl8b_gspo_direct}"
SAVE_DIR="${SAVE_DIR:-}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export LLM_JUDGE_BASE_URL="${LLM_JUDGE_BASE_URL:-}"

NNODES="${NNODES:-2}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
ENABLE_TOOLS="${ENABLE_TOOLS:-1}"
RAY_INIT_ADDRESS="${RAY_INIT_ADDRESS:-auto}"
SET_RAY_INIT_ADDRESS="${SET_RAY_INIT_ADDRESS:-0}"

MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03}"
train_files="${TRAIN_FILES:-['/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_clean_no_extra_image/parquet/train_medium_clean_21k_plus_benchmark_pass16_partial_26591_rewardfix_fsc09.parquet']}"
test_files="${TEST_FILES:-['/mnt/cpfs/delinmao/Benchmarks/MVToolBench/mvtoolbench_codevision_eval.parquet']}"

adv_estimator="grpo"
policy_loss_mode="gspo"
loss_agg_mode="seq-mean-token-mean"
clip_ratio_low="${CLIP_RATIO_LOW:-0.2}"
clip_ratio_high="${CLIP_RATIO_HIGH:-0.28}"
clip_ratio_c="${CLIP_RATIO_C:-10.0}"
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef="${KL_LOSS_COEF:-0.001}"

cfg_path="${PROJECT_DIR}/recipe/codevision/config"
cfg_name="${CFG_NAME:-grpo_trainer}"
tool_cfg_template_path="${TOOL_CFG_TEMPLATE_PATH:-recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml}"
tool_cfg_path="${TOOL_CFG_PATH:-}"
new_sp_path="${SYSTEM_PROMPT_PATH:-recipe/codevision/config/sp3.txt}"

ALLOW_LOCAL_TOOLS="${ALLOW_LOCAL_TOOLS:-0}"
OCR_BASE_URL="${OCR_BASE_URL:-}"
GROUNDEDSAM2_BASE_URL="${GROUNDEDSAM2_BASE_URL:-}"
DEPTH_BASE_URL="${DEPTH_BASE_URL:-}"
COUNTGD_BASE_URL="${COUNTGD_BASE_URL:-}"

train_bsz="${TRAIN_BSZ:-64}"
val_bsz="${VAL_BSZ:-256}"
data_shuffle="${DATA_SHUFFLE:-True}"
train_mini_bsz="${TRAIN_MINI_BSZ:-32}"
train_micro_bsz_per_gpu="${TRAIN_MICRO_BSZ_PER_GPU:-1}"
infer_micro_bsz_per_gpu="${INFER_MICRO_BSZ_PER_GPU:-1}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-8}"
max_turns="${MAX_TURNS:-12}"

max_prompt_len="${MAX_PROMPT_LEN:-$((1024 * 16))}"
max_resp_len="${MAX_RESP_LEN:-$((1024 * 16))}"
max_tool_resp_len="${MAX_TOOL_RESP_LEN:-$((1024 * 10))}"
max_image_resolution="${MAX_IMAGE_RESOLUTION:-$((1024 * 8 * 28 * 28))}"

offload="${OFFLOAD:-True}"
train_sp_size="${TRAIN_SP_SIZE:-1}"
infer_tp_size="${INFER_TP_SIZE:-4}"
use_dynamic_bsz="${USE_DYNAMIC_BSZ:-False}"
actor_ppo_max_token_len="${ACTOR_PPO_MAX_TOKEN_LEN:-$((max_prompt_len + max_resp_len))}"
infer_ppo_max_token_len="${INFER_PPO_MAX_TOKEN_LEN:-$((max_prompt_len + max_resp_len))}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-$((max_prompt_len + max_resp_len))}"
max_num_seqs="${MAX_NUM_SEQS:-1024}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.7}"
enable_prefix_caching="${ROLLOUT_ENABLE_PREFIX_CACHING:-True}"
rollout_agent_num_workers="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
rollout_temperature="${ROLLOUT_TEMPERATURE:-0.7}"
rollout_top_p="${ROLLOUT_TOP_P:-0.95}"
rollout_do_sample="${ROLLOUT_DO_SAMPLE:-True}"
rollout_max_tokens_per_turn="${ROLLOUT_MAX_TOKENS_PER_TURN:-2048}"

tool_reward_enable=True
multi_turn_enable=True
tool_reward_mode="${TOOL_REWARD_MODE:-simple_penalty}"
tool_reward_alpha="${TOOL_REWARD_ALPHA:-1.0}"
tool_reward_beta="${TOOL_REWARD_BETA:-0.0}"
tool_reward_gamma="${TOOL_REWARD_GAMMA:-0.5}"
tool_reward_delta="${TOOL_REWARD_DELTA:-0.5}"
tool_reward_overuse_weight="${TOOL_REWARD_OVERUSE_WEIGHT:-0.05}"
tool_reward_tool_error_weight="${TOOL_REWARD_TOOL_ERROR_WEIGHT:-0.2}"
tool_reward_invalid_call_weight="${TOOL_REWARD_INVALID_CALL_WEIGHT:-0.2}"
tool_reward_overuse_threshold="${TOOL_REWARD_OVERUSE_THRESHOLD:-4}"
# Design B (rnec_with_clean) — clean-tool exploration bonus weight.
# Default 0.0 keeps legacy/rnec_only/simple_penalty modes unchanged.
tool_reward_clean_tool_weight="${TOOL_REWARD_CLEAN_TOOL_WEIGHT:-0.0}"
tool_reward_clean_tool_baseline_weight="${TOOL_REWARD_CLEAN_TOOL_BASELINE_WEIGHT:-0.05}"
tool_reward_mut_protocol_weight="${TOOL_REWARD_MUT_PROTOCOL_WEIGHT:-0.2}"
tool_reward_mut_turn_penalty_weight="${TOOL_REWARD_MUT_TURN_PENALTY_WEIGHT:-0.05}"
tool_reward_mut_turn_penalty_threshold="${TOOL_REWARD_MUT_TURN_PENALTY_THRESHOLD:-6}"
step_reward_enable="${STEP_REWARD_ENABLE:-False}"
step_reward_weight="${STEP_REWARD_WEIGHT:-0.2}"
step_reward_tau="${STEP_REWARD_TAU:-0.1}"
step_reward_cap="${STEP_REWARD_CAP:-0.5}"
step_reward_use_mut_weight="${STEP_REWARD_USE_MUT_WEIGHT:-False}"
step_judge_base_url="${STEP_JUDGE_BASE_URL:-}"
step_judge_model="${STEP_JUDGE_MODEL:-}"
step_judge_api_key_env="${STEP_JUDGE_API_KEY_ENV:-STEP_JUDGE_API_KEY}"
step_judge_timeout="${STEP_JUDGE_TIMEOUT:-60}"
step_judge_max_retries="${STEP_JUDGE_MAX_RETRIES:-1}"
step_judge_num_judgments="${STEP_JUDGE_NUM_JUDGMENTS:-1}"
step_judge_aggregation="${STEP_JUDGE_AGGREGATION:-mean}"
step_judge_prompt_mode="${STEP_JUDGE_PROMPT_MODE:-context}"
step_judge_max_images="${STEP_JUDGE_MAX_IMAGES:-8}"
step_judge_max_observation_chars="${STEP_JUDGE_MAX_OBSERVATION_CHARS:-12000}"
step_judge_max_context_chars="${STEP_JUDGE_MAX_CONTEXT_CHARS:-60000}"
format_reward_weight="${FORMAT_REWARD_WEIGHT:-0.2}"
# Async reward computation: fire reward_fn as a Ray future at the start of the
# reward stage and only block on it right before advantage compute. The reward
# work (rule scoring + optional LLM judge round-trips) overlaps with
# old_log_prob and ref_log_prob compute, which are GPU-heavy and run for tens
# of seconds. Net effect: hides the reward latency under existing compute.
# Default False keeps legacy runs unchanged.
reward_launch_async="${REWARD_LAUNCH_ASYNC:-False}"
exec_reward_weight="${EXEC_REWARD_WEIGHT:-0.0}"
emerge_reward_weight="${EMERGE_REWARD_WEIGHT:-0.2}"

if [[ "${ENABLE_TOOLS}" == "0" || "${ENABLE_TOOLS,,}" == "false" ]]; then
    tool_reward_enable=False
    multi_turn_enable=False
    max_turns=1
    tool_cfg_path=null
    EXP_NAME="${EXP_NAME}_notool"
fi

exploration_reward_enable=False
exploration_reward_weight=0.0
exploration_decay_steps=0

val_before_train="${VAL_BEFORE_TRAIN:-True}"
test_freq="${TEST_FREQ:-20}"
save_freq="${SAVE_FREQ:-50}"
max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP:-5}"
max_critic_ckpt_to_keep="${MAX_CRITIC_CKPT_TO_KEEP:-5}"
resume_mode="${RESUME_MODE:-auto}"
resume_from_path="${RESUME_FROM_PATH:-null}"
total_epochs="${TOTAL_EPOCHS:-1}"
total_training_steps="${TOTAL_TRAINING_STEPS:-null}"
log_val_generations="${LOG_VAL_GENERATIONS:-8}"
log_train_generations="${LOG_TRAIN_GENERATIONS:-8}"
log_train_freq="${LOG_TRAIN_FREQ:-20}"
logger="${TRAINER_LOGGER:-'[\"console\",\"wandb\"]'}"

if [[ -z "${SAVE_DIR}" ]]; then
    SAVE_DIR="./saves/${PROJECT_NAME}/${EXP_NAME}"
fi

rollout_data_dir="${ROLLOUT_DATA_DIR:-${SAVE_DIR}/rollout_generations}"
validation_data_dir="${VALIDATION_DATA_DIR:-null}"

if [[ "${multi_turn_enable}" == "True" || "${multi_turn_enable}" == "true" ]]; then
    mkdir -p "${SAVE_DIR}"
    if [[ -z "${tool_cfg_path}" ]]; then
        if [[ "${ALLOW_LOCAL_TOOLS}" == "1" || "${ALLOW_LOCAL_TOOLS,,}" == "true" ]]; then
            OCR_BASE_URL="${OCR_BASE_URL:-http://127.0.0.1:8080}"
            GROUNDEDSAM2_BASE_URL="${GROUNDEDSAM2_BASE_URL:-http://127.0.0.1:8081}"
            DEPTH_BASE_URL="${DEPTH_BASE_URL:-http://127.0.0.1:8082}"
            COUNTGD_BASE_URL="${COUNTGD_BASE_URL:-http://127.0.0.1:8083}"
        fi
        if [[ -z "${OCR_BASE_URL}" || -z "${GROUNDEDSAM2_BASE_URL}" || -z "${DEPTH_BASE_URL}" || -z "${COUNTGD_BASE_URL}" ]]; then
            echo "ENABLE_TOOLS=1 requires OCR_BASE_URL, GROUNDEDSAM2_BASE_URL, DEPTH_BASE_URL, and COUNTGD_BASE_URL, or an explicit TOOL_CFG_PATH." >&2
            echo "For same-node local debugging only, set ALLOW_LOCAL_TOOLS=1 to use 127.0.0.1 defaults." >&2
            exit 1
        fi
        tool_cfg_path="${SAVE_DIR}/tool_config.runtime.yaml"
        python3 - <<'PY' \
          "${tool_cfg_template_path}" \
          "${tool_cfg_path}" \
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
    fi
fi

echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_FILES=${train_files}"
echo "TEST_FILES=${test_files}"
echo "ENABLE_TOOLS=${ENABLE_TOOLS}"
echo "TOOL_CFG_PATH=${tool_cfg_path}"
echo "SYSTEM_PROMPT_PATH=${new_sp_path}"
echo "OCR_BASE_URL=${OCR_BASE_URL:-}"
echo "GROUNDEDSAM2_BASE_URL=${GROUNDEDSAM2_BASE_URL:-}"
echo "DEPTH_BASE_URL=${DEPTH_BASE_URL:-}"
echo "COUNTGD_BASE_URL=${COUNTGD_BASE_URL:-}"
echo "NNODES=${NNODES}"
echo "NGPUS_PER_NODE=${NGPUS_PER_NODE}"
echo "RAY_INIT_ADDRESS=${RAY_INIT_ADDRESS}"
echo "SET_RAY_INIT_ADDRESS=${SET_RAY_INIT_ADDRESS}"
echo "SAVE_FREQ=${save_freq}"
echo "MAX_ACTOR_CKPT_TO_KEEP=${max_actor_ckpt_to_keep}"
echo "RESUME_MODE=${resume_mode}"
echo "RESUME_FROM_PATH=${resume_from_path}"
echo "ROLLOUT_DATA_DIR=${rollout_data_dir}"
echo "ROLLOUT_TEMPERATURE=${rollout_temperature}"
echo "ROLLOUT_TOP_P=${rollout_top_p}"
echo "ROLLOUT_DO_SAMPLE=${rollout_do_sample}"
echo "ROLLOUT_MAX_TOKENS_PER_TURN=${rollout_max_tokens_per_turn}"
echo "ROLLOUT_ENABLE_PREFIX_CACHING=${enable_prefix_caching}"
echo "MAX_NUM_SEQS=${max_num_seqs}"
echo "TOOL_REWARD_MODE=${tool_reward_mode}"
echo "TOOL_REWARD_MUT_PROTOCOL_WEIGHT=${tool_reward_mut_protocol_weight}"
echo "TOOL_REWARD_MUT_TURN_PENALTY_WEIGHT=${tool_reward_mut_turn_penalty_weight}"
echo "TOOL_REWARD_MUT_TURN_PENALTY_THRESHOLD=${tool_reward_mut_turn_penalty_threshold}"
echo "STEP_REWARD_ENABLE=${step_reward_enable}"
echo "STEP_REWARD_WEIGHT=${step_reward_weight}"
echo "STEP_REWARD_TAU=${step_reward_tau}"
echo "STEP_REWARD_CAP=${step_reward_cap}"
echo "STEP_REWARD_USE_MUT_WEIGHT=${step_reward_use_mut_weight}"
echo "STEP_JUDGE_BASE_URL=$([[ -n "${step_judge_base_url}" ]] && echo "${step_judge_base_url}" || echo '<unset>')"
echo "STEP_JUDGE_MODEL=$([[ -n "${step_judge_model}" ]] && echo "${step_judge_model}" || echo '<unset>')"
echo "STEP_JUDGE_NUM_JUDGMENTS=${step_judge_num_judgments}"
echo "STEP_JUDGE_AGGREGATION=${step_judge_aggregation}"
echo "STEP_JUDGE_PROMPT_MODE=${step_judge_prompt_mode}"
echo "FORMAT_REWARD_WEIGHT=${format_reward_weight}"

if [[ "${DISABLE_TOOL_PROXY:-1}" == "1" ]]; then
    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
    tool_no_proxy="localhost,127.0.0.1"
    for url in "${OCR_BASE_URL:-}" "${GROUNDEDSAM2_BASE_URL:-}" "${DEPTH_BASE_URL:-}" "${COUNTGD_BASE_URL:-}" "${step_judge_base_url:-}"; do
        host="$(python3 - <<'PY' "${url}"
import sys
from urllib.parse import urlparse

print(urlparse(sys.argv[1]).hostname or "")
PY
)"
        if [[ -n "${host}" && ",${tool_no_proxy}," != *",${host},"* ]]; then
            tool_no_proxy="${tool_no_proxy},${host}"
        fi
    done
    export no_proxy="${no_proxy:-${tool_no_proxy}}"
    export NO_PROXY="${NO_PROXY:-${no_proxy}}"
    echo "NO_PROXY=${NO_PROXY}"
fi

RAY_INIT_OVERRIDES=()
if [[ "${SET_RAY_INIT_ADDRESS}" == "1" || "${SET_RAY_INIT_ADDRESS,,}" == "true" ]]; then
    RAY_INIT_OVERRIDES+=("+ray_kwargs.ray_init.address=${RAY_INIT_ADDRESS}")
fi

tool_preflight_check="${TOOL_PREFLIGHT_CHECK:-0}"
if [[ "${tool_preflight_check}" == "1" || "${tool_preflight_check,,}" == "true" ]]; then
    python3 - <<'PY'
import os
import urllib.request

for name in ["OCR_BASE_URL", "GROUNDEDSAM2_BASE_URL", "DEPTH_BASE_URL", "COUNTGD_BASE_URL"]:
    base = os.environ.get(name)
    if not base:
        continue
    print(f"[TOOL_PREFLIGHT] {name}={base}", flush=True)
    for path in ["/health", "/"]:
        url = base.rstrip("/") + path
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                print(f"[TOOL_PREFLIGHT] OK {url} status={response.status}", flush=True)
                break
        except Exception as exc:
            print(f"[TOOL_PREFLIGHT] FAIL {url} {exc!r}", flush=True)
PY
fi

python3 -m verl.trainer.main_ppo \
    --config-path=${cfg_path} \
    --config-name=${cfg_name} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    data.train_batch_size=${train_bsz} \
    data.val_batch_size=${val_bsz} \
    data.shuffle=${data_shuffle} \
    data.max_prompt_length=${max_prompt_len} \
    data.max_response_length=${max_resp_len} \
    data.filter_overlong_prompts=False \
    data.truncation="error" \
    data.return_raw_chat=True \
    data.train_files=${train_files} \
    data.val_files=${test_files} \
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
    actor_rollout_ref.actor.policy_loss.loss_mode=${policy_loss_mode} \
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
    actor_rollout_ref.rollout.temperature=${rollout_temperature} \
    actor_rollout_ref.rollout.top_p=${rollout_top_p} \
    actor_rollout_ref.rollout.do_sample=${rollout_do_sample} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp_size} \
    actor_rollout_ref.rollout.multi_turn.enable=${multi_turn_enable} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.max_response_tokens_per_turn=${rollout_max_tokens_per_turn} \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${max_tool_resp_len} \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${tool_cfg_path} \
    actor_rollout_ref.rollout.agent.num_workers=${rollout_agent_num_workers} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs} \
    actor_rollout_ref.rollout.enable_prefix_caching=${enable_prefix_caching} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    +reward_model.tool_reward.enable=${tool_reward_enable} \
    +reward_model.tool_reward.mode=${tool_reward_mode} \
    +reward_model.tool_reward.alpha=${tool_reward_alpha} \
    +reward_model.tool_reward.beta=${tool_reward_beta} \
    +reward_model.tool_reward.gamma=${tool_reward_gamma} \
    +reward_model.tool_reward.delta=${tool_reward_delta} \
    +reward_model.tool_reward.overuse_weight=${tool_reward_overuse_weight} \
    +reward_model.tool_reward.tool_error_weight=${tool_reward_tool_error_weight} \
    +reward_model.tool_reward.invalid_call_weight=${tool_reward_invalid_call_weight} \
    +reward_model.tool_reward.overuse_threshold=${tool_reward_overuse_threshold} \
    +reward_model.tool_reward.clean_tool_weight=${tool_reward_clean_tool_weight} \
    +reward_model.tool_reward.clean_tool_baseline_weight=${tool_reward_clean_tool_baseline_weight} \
    +reward_model.tool_reward.mut_protocol_weight=${tool_reward_mut_protocol_weight} \
    +reward_model.tool_reward.mut_turn_penalty_weight=${tool_reward_mut_turn_penalty_weight} \
    +reward_model.tool_reward.mut_turn_penalty_threshold=${tool_reward_mut_turn_penalty_threshold} \
    +reward_model.step_reward.enable=${step_reward_enable} \
    +reward_model.step_reward.weight=${step_reward_weight} \
    +reward_model.step_reward.tau=${step_reward_tau} \
    +reward_model.step_reward.cap=${step_reward_cap} \
    +reward_model.step_reward.use_mut_weight=${step_reward_use_mut_weight} \
    +reward_model.step_reward.base_url=${step_judge_base_url} \
    +reward_model.step_reward.model=${step_judge_model} \
    +reward_model.step_reward.api_key_env=${step_judge_api_key_env} \
    +reward_model.step_reward.timeout_s=${step_judge_timeout} \
    +reward_model.step_reward.max_retries=${step_judge_max_retries} \
    +reward_model.step_reward.num_judgments=${step_judge_num_judgments} \
    +reward_model.step_reward.aggregation=${step_judge_aggregation} \
    +reward_model.step_reward.prompt_mode=${step_judge_prompt_mode} \
    +reward_model.step_reward.max_images=${step_judge_max_images} \
    +reward_model.step_reward.max_observation_chars=${step_judge_max_observation_chars} \
    +reward_model.step_reward.max_context_chars=${step_judge_max_context_chars} \
    reward_model.launch_reward_fn_async=${reward_launch_async} \
    +reward_model.format_reward_weight=${format_reward_weight} \
    +reward_model.exec_reward_weight=${exec_reward_weight} \
    +reward_model.emerge_reward_weight=${emerge_reward_weight} \
    +reward_model.exploration_reward.enable=${exploration_reward_enable} \
    +reward_model.exploration_reward.weight=${exploration_reward_weight} \
    +reward_model.exploration_reward.decay_steps=${exploration_decay_steps} \
    trainer.critic_warmup=0 \
    trainer.val_before_train=${val_before_train} \
    trainer.logger=${logger} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${save_freq} \
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep} \
    trainer.max_critic_ckpt_to_keep=${max_critic_ckpt_to_keep} \
    trainer.test_freq=${test_freq} \
    trainer.log_val_generations=${log_val_generations} \
    +trainer.log_train_generations=${log_train_generations} \
    +trainer.log_train_freq=${log_train_freq} \
    trainer.rollout_data_dir=${rollout_data_dir} \
    trainer.validation_data_dir=${validation_data_dir} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.resume_mode=${resume_mode} \
    trainer.resume_from_path=${resume_from_path} \
    ray_kwargs.ray_init.num_cpus=${RAY_INIT_NUM_CPUS:-null} \
    "${RAY_INIT_OVERRIDES[@]}"
