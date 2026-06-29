#!/usr/bin/env bash
# Run mix200 SFT with the neutral sp3/v03 prompt on MVToolBench, ChartQA, and OCRBench.

set -euo pipefail

cd /mnt/cpfs/delinmao/ToolVision/CodeVision
mkdir -p /mnt/cpfs/delinmao/logs

export LLM_JUDGE_BASE_URL="${LLM_JUDGE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export LLM_JUDGE_MODEL_NAME="${LLM_JUDGE_MODEL_NAME:-qwen3.6-plus}"
export LLM_JUDGE_TIMEOUT="${LLM_JUDGE_TIMEOUT:-100}"
export LLM_JUDGE_MAX_RETRIES="${LLM_JUDGE_MAX_RETRIES:-3}"

if [[ -z "${LLM_JUDGE_API_KEY:-}" ]]; then
  read -s -p "DashScope API key: " LLM_JUDGE_API_KEY
  echo
  export LLM_JUDGE_API_KEY
fi

CODEVISION_ENV="${CODEVISION_ENV:-/mnt/cpfs/delinmao/envs/codevision_new}" \
GPU_CANDIDATES="${GPU_CANDIDATES:-2,3,4,5,6}" \
RAY_INIT_NUM_CPUS="${RAY_INIT_NUM_CPUS:-40}" \
MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool}" \
EXP_PREFIX="${EXP_PREFIX:-mix200_sft_v03}" \
BENCHMARKS="${BENCHMARKS:-mvtoolbench,chartqa,ocrbench}" \
SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-recipe/codevision/config/sp3.txt}" \
TOOL_CFG_TEMPLATE_PATH="${TOOL_CFG_TEMPLATE_PATH:-recipe/codevision/config/code_image_tool_config_v03.yaml}" \
nohup bash scripts/run_tools_eval_all_wait_5gpu_nohup.sh \
  > /mnt/cpfs/delinmao/logs/eval_mix200_sft_v03_mvtoolbench_chartqa_ocrbench_llmjudge.log 2>&1 &

echo "Started eval. Log: /mnt/cpfs/delinmao/logs/eval_mix200_sft_v03_mvtoolbench_chartqa_ocrbench_llmjudge.log"
