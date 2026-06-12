#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Final pre-newdata RL run (2026-05-28).
#
# Purpose:
#   Last shot at making reward shaping work BEFORE we commit to generating new
#   SFT trajectories + MUT data. Combines every lesson from the previous 4 runs:
#
#   1. ORIGINAL SFT model (sft-mix200-simple-notool-sp3-v03).
#      - promptfix SFT was a regression (R_acc 0.375 vs 0.525 baseline).
#      - The placeholder rewrite confused JSON generation more than it helped.
#
#   2. SFT-clean tool config (code_image_tool_config_v03_sftclean.yaml).
#      - Concrete tool-call examples preserved (matches what SFT saw).
#      - Description-parameter example abstracted ("Count the cars" removed).
#      - Net drift from original SFT prompt: ~5%, vs 40% for placeholder version.
#
#   3. AGGRESSIVE clean_tool reward (0.05 → 0.15, ×3).
#      - Previous 0.05 was too weak vs the 0.7 expected reward of "skip tool".
#      - Empirically, with current ~10% tool-call JSON success rate, no clean_tool
#        weight under ~8 can flip the EV math. So 0.15 is "as much as we dare
#        without confusing it with the R_acc=1 main signal".
#      - Risk: ritual tool use. Mitigated by overuse_threshold staying at 4 +
#        bad_tool penalty intact.
#
#   4. Filtered training data (train_no_ood.parquet, 34000 rows).
#      - Dropped 6000 rows: mmk12 + wemath_standard + thinklite_vl_hard +
#        puzzlevqa. These categories the SFT data never touched and the model
#        has no behavioral prior for.
#      - Forces RL on distributions where tools should plausibly help.
#
# Pass/fail criterion (look at WandB after ~30-50 steps):
#   PASS: NumTurns ≥ 0.5 stable, R_acc ≥ 0.55
#         → reward shaping + alignment works on cleaned data, can iterate further.
#   FAIL: NumTurns → 0 within 30 steps (like all previous runs)
#         → reward shaping is structurally incapable of fixing this; commit to
#           the MUT/new-SFT-data direction without further reward experiments.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOB_NAME="${JOB_NAME:-codevision_gspo_direct_final_v1}"
export EXP_NAME="${EXP_NAME:-qwen3vl8b_gspo_final_v1_sftclean_clean015_nood}"

# === Lever 1: go back to the ORIGINAL SFT model (not promptfix) ===
export MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03}"

# === Lever 2: SFT-aligned 10-tool tool schema with abstracted description example ===
export TOOL_CFG_TEMPLATE_PATH="${TOOL_CFG_TEMPLATE_PATH:-recipe/codevision/config/code_image_tool_config_v03_sftclean.yaml}"
# system prompt stays sp3 (same as SFT)
export SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-recipe/codevision/config/sp3.txt}"

# === Lever 3: filtered training data ===
# train_no_ood.parquet = original train.parquet minus
#   mmk12, wemath_standard, thinklite_vl_hard, puzzlevqa (6000 rows dropped).
# Build with: python _filter_rl_parquet.py --input <train.parquet> --output <train_no_ood.parquet>
export TRAIN_FILES="${TRAIN_FILES:-['/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k_clean_no_extra_image/parquet/train_no_ood.parquet']}"

# === Lever 4: aggressive clean_tool, otherwise same as rnec_with_clean ===
export TOOL_REWARD_MODE="${TOOL_REWARD_MODE:-rnec_with_clean}"
export FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.2}"
export TOOL_REWARD_BETA="${TOOL_REWARD_BETA:-0.3}"
export TOOL_REWARD_CLEAN_TOOL_BASELINE_WEIGHT="${TOOL_REWARD_CLEAN_TOOL_BASELINE_WEIGHT:-0.05}"
export TOOL_REWARD_CLEAN_TOOL_WEIGHT="${TOOL_REWARD_CLEAN_TOOL_WEIGHT:-0.15}"   # success-gated bonus
export TOOL_REWARD_TOOL_ERROR_WEIGHT="${TOOL_REWARD_TOOL_ERROR_WEIGHT:-0.02}"
export TOOL_REWARD_INVALID_CALL_WEIGHT="${TOOL_REWARD_INVALID_CALL_WEIGHT:-0.02}"
export TOOL_REWARD_OVERUSE_WEIGHT="${TOOL_REWARD_OVERUSE_WEIGHT:-0.05}"
export TOOL_REWARD_OVERUSE_THRESHOLD="${TOOL_REWARD_OVERUSE_THRESHOLD:-4}"

# Async reward is broken in this codebase (custom_module pickling issue).
# Leave OFF.
export REWARD_LAUNCH_ASYNC="${REWARD_LAUNCH_ASYNC:-False}"

# Tighter LLM judge timeouts (small win)
export LLM_JUDGE_TIMEOUT="${LLM_JUDGE_TIMEOUT:-30}"
export LLM_JUDGE_MAX_RETRIES="${LLM_JUDGE_MAX_RETRIES:-2}"

# Log generations more frequently so we can see early behavior
export LOG_TRAIN_GENERATIONS="${LOG_TRAIN_GENERATIONS:-32}"
export LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-32}"
export LOG_TRAIN_FREQ="${LOG_TRAIN_FREQ:-10}"

exec bash "${SCRIPT_DIR}/submit_dlc_gspo_direct_full.sh"
