#!/usr/bin/env bash
#SBATCH -J tools-eval-all-base-cur-3gpu
#SBATCH -p a100_global
#SBATCH --gres=gpu:3
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH -t 12:00:00
#SBATCH -o /mnt/cpfs/delinmao/logs/%x-%j.out
#SBATCH -e /mnt/cpfs/delinmao/logs/%x-%j.err

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/cpfs/delinmao}"
export MODEL_PATH="${MODEL_PATH:-${WORKSPACE_ROOT}/models/Qwen3-VL-8B-Thinking}"
export EXP_PREFIX="${EXP_PREFIX:-base_cur}"
export LLM_JUDGE_TRUST_ENV=0

exec bash "${WORKSPACE_ROOT}/ToolVision/CodeVision/scripts/slurm_tools_eval_all_a100_3gpu.sbatch"
