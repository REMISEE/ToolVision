#!/usr/bin/env bash
set -euo pipefail

# Dry-run by default. Set APPLY=1 to delete checkpoint dirs.
# This only touches global_step_* checkpoint directories and tracker files.
# Rollout jsonl logs are kept for analysis.

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
APPLY="${APPLY:-0}"

cleanup_after_step() {
  local run_dir="$1"
  local keep_step="$2"
  local full_dir="${ROOT_DIR}/${run_dir}"

  if [[ ! -d "${full_dir}" ]]; then
    echo "[skip] missing ${full_dir}" >&2
    return
  fi

  echo
  echo "== ${run_dir}: keep global_step_${keep_step}, delete later checkpoints =="
  find "${full_dir}" -maxdepth 1 -type d -name 'global_step_*' | sort -V | while read -r ckpt; do
    local name step
    name="$(basename "${ckpt}")"
    step="${name#global_step_}"
    if [[ "${step}" =~ ^[0-9]+$ && "${step}" -gt "${keep_step}" ]]; then
      if [[ "${APPLY}" == "1" ]]; then
        echo "[delete] ${ckpt}"
        rm -rf "${ckpt}"
      else
        echo "[dry-run delete] ${ckpt}"
      fi
    else
      echo "[keep] ${ckpt}"
    fi
  done

  if [[ "${APPLY}" == "1" ]]; then
    printf '%s' "${keep_step}" > "${full_dir}/latest_checkpointed_iteration.txt"
    echo "[write] ${full_dir}/latest_checkpointed_iteration.txt -> ${keep_step}"
  else
    echo "[dry-run write] ${full_dir}/latest_checkpointed_iteration.txt -> ${keep_step}"
  fi
}

cleanup_after_step "saves/ToolVisionRL/mutv1_a" 70
cleanup_after_step "saves/ToolVisionRL/mutv2" 20

echo
if [[ "${APPLY}" == "1" ]]; then
  echo "Done."
else
  echo "Dry-run only. Re-run with APPLY=1 to delete."
fi
