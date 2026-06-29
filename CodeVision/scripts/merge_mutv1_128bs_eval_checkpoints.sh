#!/usr/bin/env bash
set -euo pipefail

# Merge 16-way FSDP actor checkpoints into ordinary HuggingFace model
# directories so eval can run with any GPU world size.

cd "${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"

PYTHON="${CODEVISION_PYTHON:-/mnt/cpfs/delinmao/envs/codevision_new/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/cpfs/delinmao/ToolVision/CodeVision/saves/ToolVisionRL/mutv1_128bs_0618}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/cpfs/delinmao/ToolVision/CodeVision/saves/ToolVisionRL/merged_hf}"
TARGET_PREFIX="${TARGET_PREFIX:-mutv1_128bs}"
STEPS="${STEPS:-60 140}"

for step in ${STEPS}; do
  local_dir="${SOURCE_ROOT}/global_step_${step}/actor"
  target_dir="${TARGET_ROOT}/${TARGET_PREFIX}_global_step_${step}"

  if [[ ! -f "${local_dir}/fsdp_config.json" ]]; then
    echo "Missing FSDP checkpoint actor dir: ${local_dir}" >&2
    exit 1
  fi

  if [[ -f "${target_dir}/config.json" && -f "${target_dir}/model.safetensors.index.json" ]]; then
    echo "[skip] merged HF model already exists: ${target_dir}"
    continue
  fi

  mkdir -p "${target_dir}"
  echo "[merge] step=${step}"
  echo "        local_dir=${local_dir}"
  echo "        target_dir=${target_dir}"

  "${PYTHON}" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${local_dir}" \
    --target_dir "${target_dir}"
done
