#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_JSONL="${INPUT_JSONL:-$ROOT_DIR/export_images/output/gqa/samples.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/offline_sft_pipeline/outputs/dataset_pipeline_runs}"
RUN_ID="${RUN_ID:-gqa_full_$(date -u +%Y%m%dT%H%M%SZ)}"
JUDGE_BACKEND="${JUDGE_BACKEND:-committee}"
JUDGE_MODELS_FILE="${JUDGE_MODELS_FILE:-}"
JUDGE_MAX_CONCURRENCY="${JUDGE_MAX_CONCURRENCY:-}"
SAMPLE_IDS_FILE="${SAMPLE_IDS_FILE:-}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PLANNER_DEBUG="${PLANNER_DEBUG:-0}"
EXECUTOR_DEBUG="${EXECUTOR_DEBUG:-0}"
RESUME="${RESUME:-1}"

if [[ ! -f "$INPUT_JSONL" ]]; then
  echo "input JSONL not found: $INPUT_JSONL" >&2
  exit 1
fi

cmd=(
  python offline_sft_pipeline/scripts/run_dataset_pipeline.py
  --input-jsonl "$INPUT_JSONL"
  --output-dir "$OUTPUT_DIR"
  --run-id "$RUN_ID"
  --judge-backend "$JUDGE_BACKEND"
)

if [[ -n "$SAMPLE_IDS_FILE" ]]; then
  if [[ ! -f "$SAMPLE_IDS_FILE" ]]; then
    echo "sample ids file not found: $SAMPLE_IDS_FILE" >&2
    exit 1
  fi
  cmd+=(--sample-ids-file "$SAMPLE_IDS_FILE")
fi

if [[ "$MAX_SAMPLES" != "0" ]]; then
  cmd+=(--max-samples "$MAX_SAMPLES")
fi

if [[ -n "$JUDGE_MODELS_FILE" ]]; then
  if [[ ! -f "$JUDGE_MODELS_FILE" ]]; then
    echo "judge models file not found: $JUDGE_MODELS_FILE" >&2
    exit 1
  fi
  cmd+=(--judge-models-file "$JUDGE_MODELS_FILE")
fi

if [[ -n "$JUDGE_MAX_CONCURRENCY" ]]; then
  cmd+=(--judge-max-concurrency "$JUDGE_MAX_CONCURRENCY")
fi

if [[ "$PLANNER_DEBUG" == "1" ]]; then
  cmd+=(--planner-debug)
fi

if [[ "$EXECUTOR_DEBUG" == "1" ]]; then
  cmd+=(--executor-debug)
fi

if [[ "$RESUME" == "1" ]]; then
  cmd+=(--resume)
else
  cmd+=(--no-resume)
fi

echo "RUN_ID=$RUN_ID"
echo "INPUT_JSONL=$INPUT_JSONL"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "JUDGE_BACKEND=$JUDGE_BACKEND"
echo "JUDGE_MODELS_FILE=${JUDGE_MODELS_FILE:-<default>}"
echo "JUDGE_MAX_CONCURRENCY=${JUDGE_MAX_CONCURRENCY:-<default>}"
echo "SAMPLE_IDS_FILE=${SAMPLE_IDS_FILE:-<none>}"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "PLANNER_DEBUG=$PLANNER_DEBUG"
echo "EXECUTOR_DEBUG=$EXECUTOR_DEBUG"
echo "RESUME=$RESUME"

"${cmd[@]}"
