#!/usr/bin/env bash
set -euo pipefail

# Submit only the first expanded benchmark subset:
#   RealWorldQA, MMStar, DocVQA val, InfoVQA val,
#   MME-RealWorld-Lite, MME-RealWorld-CN, MMVet.
#
# This is intentionally a thin wrapper over submit_dlc_bigbench_eval_20260624.sh
# so it uses the same DLC/eval path as the existing benchmark jobs.

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
cd "${ROOT_DIR}"

GROUP_A="${GROUP_A:-realworldqa mmstar}" \
GROUP_B="${GROUP_B:-docvqa_val infovqa_val}" \
GROUP_C="${GROUP_C:-mme_realworld_lite mme_realworld_cn}" \
GROUP_D="${GROUP_D:-mmvet}" \
ENABLE_LLM_JUDGE="${ENABLE_LLM_JUDGE:-1}" \
bash scripts/submit_dlc_bigbench_eval_20260624.sh
