#!/usr/bin/env bash
set -euo pipefail

# Materialize CodeVision eval parquet files for the expanded benchmark panel.
# This script only prepares local data; it does not launch model eval.

ROOT_DIR="${ROOT_DIR:-/mnt/cpfs/delinmao/ToolVision/CodeVision}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/cpfs/delinmao}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${WORKSPACE_ROOT}/Benchmarks}"
CODEVISION_ENV="${CODEVISION_ENV:-${WORKSPACE_ROOT}/envs/codevision_new}"
PYTHON_BIN="${PYTHON_BIN:-${CODEVISION_ENV}/bin/python}"

DATASETS="${DATASETS:-realworldqa mmstar mme_realworld_lite mme_realworld mme_realworld_cn docvqa_val infovqa_val mmvet pixmo_count pixmo_count_lmms spatialmqa cvbench countqa ocrbench_v2}"
if [[ -n "${MMVET_HARD_DATA_PATH:-}" ]]; then
  DATASETS="${DATASETS} mmvet_hard"
fi

cd "${ROOT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing python: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "Preparing expanded eval benchmarks"
echo "BENCHMARK_ROOT=${BENCHMARK_ROOT}"
echo "DATASETS=${DATASETS}"
echo "PYTHON_BIN=${PYTHON_BIN}"
if [[ -n "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]]; then
  echo "HF token: set"
else
  echo "HF token: not set; public datasets will still load, gated datasets may fail"
fi

"${PYTHON_BIN}" recipe/codevision/prepare_benchmarks.py \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --datasets ${DATASETS} \
  "$@"

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

root = Path("/mnt/cpfs/delinmao/Benchmarks")
expected = {
    "realworldqa": root / "RealWorldQA" / "realworldqa_codevision_eval.parquet",
    "mmstar": root / "MMStar" / "mmstar_codevision_eval.parquet",
    "mme_realworld_lite": root / "MME-RealWorld-Lite" / "mme_realworld_lite_codevision_eval.parquet",
    "mme_realworld": root / "MME-RealWorld" / "mme_realworld_codevision_eval.parquet",
    "mme_realworld_cn": root / "MME-RealWorld-CN" / "mme_realworld_cn_codevision_eval.parquet",
    "docvqa_val": root / "DocVQA" / "docvqa_val_codevision_eval.parquet",
    "infovqa_val": root / "InfoVQA" / "infovqa_val_codevision_eval.parquet",
    "mmvet": root / "MMVet" / "mmvet_codevision_eval.parquet",
    "pixmo_count": root / "Pixmo-Count" / "pixmo_count_codevision_eval.parquet",
    "pixmo_count_lmms": root / "Pixmo-Count-LMMS" / "pixmo_count_lmms_codevision_eval.parquet",
    "spatialmqa": root / "SpatialMQA" / "spatialmqa_codevision_eval.parquet",
    "cvbench": root / "CV-Bench" / "cvbench_codevision_eval.parquet",
    "countqa": root / "CountQA" / "countqa_codevision_eval.parquet",
    "ocrbench_v2": root / "OCRBench_v2" / "ocrbench_v2_codevision_eval.parquet",
}
missing = []
for name, path in expected.items():
    if path.exists():
        print(f"[ok] {name}: {path}")
    else:
        missing.append((name, path))
        print(f"[missing] {name}: {path}")
if missing:
    raise SystemExit(f"Missing {len(missing)} expected parquet files")
PY
