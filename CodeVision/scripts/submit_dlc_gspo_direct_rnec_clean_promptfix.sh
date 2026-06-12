#!/usr/bin/env bash
set -euo pipefail

# Design B (rnec_with_clean) — PROMPT FIX variant.
#
# Same reward shaping as submit_dlc_gspo_direct_rnec_clean.sh, but points to the
# new SFT checkpoint trained with the placeholder-aligned v04 prompt. The intent
# of this run is to verify whether SFT-RL prompt alignment alone fixes the
# NumTurns→0 collapse, before touching reward parameters again.
#
# Single-variable experiment vs the previous rnec_clean run:
#   - MODEL_PATH:  sft-mix200-simple-notool-sp3-v03  →  sft-...-promptfix
#   - everything else unchanged (reward shape, weights, data, hyperparams)
#
# If NumTurns recovers (≥ 0.5) and R_acc trends up → prompt drift was the root
# cause, reward shaping is fine as-is.
# If NumTurns still collapses → next step is to bump TOOL_REWARD_CLEAN_TOOL_WEIGHT
# (e.g. 0.05 → 0.10 or 0.15) to compensate for residual bad_tool risk.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOB_NAME="${JOB_NAME:-codevision_gspo_direct_rnec_clean_promptfix}"
export EXP_NAME="${EXP_NAME:-qwen3vl8b_gspo_full40k_rnec_clean_promptfix}"

# === The only meaningful change vs submit_dlc_gspo_direct_rnec_clean.sh ===
export MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03-promptfix}"

# NOTE: REWARD_LAUNCH_ASYNC=True was attempted but triggers a Ray serialization
# bug: `compute_reward_async` (Ray remote) fails to deserialize the dynamically
# loaded custom reward function (`recipe/codevision/reward.py`) with
# `ModuleNotFoundError: No module named 'custom_module'`. Synchronous path
# works fine because the custom module is already loaded in the main process.
# Leaving disabled until we fix the serialization (probably needs explicit
# PYTHONPATH propagation or a registered Ray runtime_env).
export REWARD_LAUNCH_ASYNC="${REWARD_LAUNCH_ASYNC:-False}"
# Tighten LLM judge timeout (default 100s is way too generous for our judge model).
# Cuts worst-case stuck-call wait without hurting normal-latency calls.
export LLM_JUDGE_TIMEOUT="${LLM_JUDGE_TIMEOUT:-30}"
export LLM_JUDGE_MAX_RETRIES="${LLM_JUDGE_MAX_RETRIES:-2}"

# Log more trajectories per logging interval so we can see prompt+output diversity
# in wandb. Bump generations 8→32 and shorten interval 20→10 (so we get richer
# train rollouts every 10 steps instead of every 20).
# Trade-off: each trajectory is a multi-turn rollout with images + tool calls,
# so 32 generations × every 10 steps ≈ ~3.2 generations/step of upload overhead.
# Still well within wandb's tolerance.
export LOG_TRAIN_GENERATIONS="${LOG_TRAIN_GENERATIONS:-32}"
export LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-32}"
export LOG_TRAIN_FREQ="${LOG_TRAIN_FREQ:-10}"

# Reward params (identical to rnec_clean v1)
export TOOL_REWARD_MODE="${TOOL_REWARD_MODE:-rnec_with_clean}"
export FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.2}"
export TOOL_REWARD_BETA="${TOOL_REWARD_BETA:-0.3}"
export TOOL_REWARD_CLEAN_TOOL_WEIGHT="${TOOL_REWARD_CLEAN_TOOL_WEIGHT:-0.05}"
export TOOL_REWARD_TOOL_ERROR_WEIGHT="${TOOL_REWARD_TOOL_ERROR_WEIGHT:-0.02}"
export TOOL_REWARD_INVALID_CALL_WEIGHT="${TOOL_REWARD_INVALID_CALL_WEIGHT:-0.02}"
export TOOL_REWARD_OVERUSE_WEIGHT="${TOOL_REWARD_OVERUSE_WEIGHT:-0.05}"
export TOOL_REWARD_OVERUSE_THRESHOLD="${TOOL_REWARD_OVERUSE_THRESHOLD:-4}"

exec bash "${SCRIPT_DIR}/submit_dlc_gspo_direct_full.sh"
