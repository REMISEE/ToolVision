#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${SERVICE_LOG_DIR:-$ROOT_DIR/outputs/service_logs}"
PID_DIR="${SERVICE_PID_DIR:-$ROOT_DIR/outputs/service_pids}"

OCR_ENV="${OCR_ENV:-paddleocr}"
OCR_HOST="${OCR_HOST:-0.0.0.0}"
OCR_PORT="${OCR_PORT:-8080}"
OCR_PIPELINE="${OCR_PIPELINE:-OCR}"
OCR_DEVICE="${OCR_DEVICE:-}"
OCR_USE_HPIP="${OCR_USE_HPIP:-0}"
OCR_HPI_CONFIG="${OCR_HPI_CONFIG:-}"

GROUNDEDSAM2_ENV="${GROUNDEDSAM2_ENV:-groundedsam2}"
GROUNDEDSAM2_HOST="${GROUNDEDSAM2_HOST:-0.0.0.0}"
GROUNDEDSAM2_PORT="${GROUNDEDSAM2_PORT:-8081}"
GROUNDEDSAM2_DEVICE="${GROUNDEDSAM2_DEVICE:-cuda}"
GROUNDEDSAM2_CUDA_VISIBLE_DEVICES="${GROUNDEDSAM2_CUDA_VISIBLE_DEVICES:-}"
GROUNDEDSAM2_DEFAULT_TEXT_PROMPT="${GROUNDEDSAM2_DEFAULT_TEXT_PROMPT:-object.}"
GROUNDEDSAM2_BOX_THRESHOLD="${GROUNDEDSAM2_BOX_THRESHOLD:-0.35}"
GROUNDEDSAM2_TEXT_THRESHOLD="${GROUNDEDSAM2_TEXT_THRESHOLD:-0.25}"
GROUNDEDSAM2_SAM2_CHECKPOINT="${GROUNDEDSAM2_SAM2_CHECKPOINT:-../Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt}"
GROUNDEDSAM2_SAM2_MODEL_CONFIG="${GROUNDEDSAM2_SAM2_MODEL_CONFIG:-../Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml}"
GROUNDEDSAM2_GDINO_CONFIG="${GROUNDEDSAM2_GDINO_CONFIG:-../Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
GROUNDEDSAM2_GDINO_CHECKPOINT="${GROUNDEDSAM2_GDINO_CHECKPOINT:-../Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth}"


usage() {
  cat <<'EOF'
Usage:
  scripts/launch_external_services.sh start [ocr|groundedsam2|all]
  scripts/launch_external_services.sh stop [ocr|groundedsam2|all]
  scripts/launch_external_services.sh restart [ocr|groundedsam2|all]
  scripts/launch_external_services.sh status [ocr|groundedsam2|all]

Environment overrides:
  OCR_ENV, OCR_HOST, OCR_PORT, OCR_PIPELINE, OCR_DEVICE, OCR_USE_HPIP, OCR_HPI_CONFIG
  GROUNDEDSAM2_ENV, GROUNDEDSAM2_HOST, GROUNDEDSAM2_PORT, GROUNDEDSAM2_DEVICE
  GROUNDEDSAM2_CUDA_VISIBLE_DEVICES, GROUNDEDSAM2_DEFAULT_TEXT_PROMPT
  GROUNDEDSAM2_BOX_THRESHOLD, GROUNDEDSAM2_TEXT_THRESHOLD
  GROUNDEDSAM2_SAM2_CHECKPOINT, GROUNDEDSAM2_SAM2_MODEL_CONFIG
  GROUNDEDSAM2_GDINO_CONFIG, GROUNDEDSAM2_GDINO_CHECKPOINT
  SERVICE_LOG_DIR, SERVICE_PID_DIR
EOF
}


ensure_dirs() {
  mkdir -p "$LOG_DIR" "$PID_DIR"
}


pid_file() {
  local name="$1"
  echo "$PID_DIR/${name}.pid"
}


log_file() {
  local name="$1"
  echo "$LOG_DIR/${name}.log"
}


is_running() {
  local name="$1"
  local file
  file="$(pid_file "$name")"
  [[ -f "$file" ]] || return 1
  local pid
  pid="$(cat "$file")"
  kill -0 "$pid" 2>/dev/null
}


start_process() {
  local name="$1"
  shift
  ensure_dirs
  if is_running "$name"; then
    echo "[skip] $name already running pid=$(cat "$(pid_file "$name")") log=$(log_file "$name")"
    return
  fi

  local log
  log="$(log_file "$name")"
  nohup "$@" >"$log" 2>&1 &
  local pid=$!
  echo "$pid" >"$(pid_file "$name")"
  echo "[start] $name pid=$pid log=$log"
}


stop_process() {
  local name="$1"
  local file
  file="$(pid_file "$name")"
  if ! [[ -f "$file" ]]; then
    echo "[skip] $name not running"
    return
  fi

  local pid
  pid="$(cat "$file")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "[stop] $name pid=$pid"
  else
    echo "[clean] $name stale pid=$pid"
  fi
  rm -f "$file"
}


status_process() {
  local name="$1"
  if is_running "$name"; then
    echo "[up] $name pid=$(cat "$(pid_file "$name")") log=$(log_file "$name")"
  else
    echo "[down] $name"
  fi
}


start_ocr() {
  local -a cmd=(conda run --no-capture-output -n "$OCR_ENV" python scripts/launch_paddleocr_service.py
    --host "$OCR_HOST"
    --port "$OCR_PORT"
    --pipeline "$OCR_PIPELINE")
  if [[ -n "$OCR_DEVICE" ]]; then
    cmd+=(--device "$OCR_DEVICE")
  fi
  if [[ "$OCR_USE_HPIP" == "1" ]]; then
    cmd+=(--use-hpip)
  fi
  if [[ -n "$OCR_HPI_CONFIG" ]]; then
    cmd+=(--hpi-config "$OCR_HPI_CONFIG")
  fi
  start_process "paddleocr" "${cmd[@]}"
}


start_groundedsam2() {
  local -a cmd=(conda run --no-capture-output -n "$GROUNDEDSAM2_ENV")
  if [[ -n "$GROUNDEDSAM2_CUDA_VISIBLE_DEVICES" ]]; then
    cmd+=(env "CUDA_VISIBLE_DEVICES=$GROUNDEDSAM2_CUDA_VISIBLE_DEVICES")
  fi
  cmd+=(python scripts/launch_groundedsam2_service.py
    --host "$GROUNDEDSAM2_HOST"
    --port "$GROUNDEDSAM2_PORT"
    --device "$GROUNDEDSAM2_DEVICE"
    --default-text-prompt "$GROUNDEDSAM2_DEFAULT_TEXT_PROMPT"
    --box-threshold "$GROUNDEDSAM2_BOX_THRESHOLD"
    --text-threshold "$GROUNDEDSAM2_TEXT_THRESHOLD"
    --sam2-checkpoint "$GROUNDEDSAM2_SAM2_CHECKPOINT"
    --sam2-model-config "$GROUNDEDSAM2_SAM2_MODEL_CONFIG"
    --grounding-dino-config "$GROUNDEDSAM2_GDINO_CONFIG"
    --grounding-dino-checkpoint "$GROUNDEDSAM2_GDINO_CHECKPOINT")
  start_process "groundedsam2" "${cmd[@]}"
}


run_action() {
  local action="$1"
  local target="$2"
  case "$target" in
    all)
      run_action "$action" ocr
      run_action "$action" groundedsam2
      ;;
    ocr)
      case "$action" in
        start) start_ocr ;;
        stop) stop_process "paddleocr" ;;
        status) status_process "paddleocr" ;;
        restart)
          stop_process "paddleocr"
          start_ocr
          ;;
        *) usage; exit 1 ;;
      esac
      ;;
    groundedsam2)
      case "$action" in
        start) start_groundedsam2 ;;
        stop) stop_process "groundedsam2" ;;
        status) status_process "groundedsam2" ;;
        restart)
          stop_process "groundedsam2"
          start_groundedsam2
          ;;
        *) usage; exit 1 ;;
      esac
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}


main() {
  cd "$ROOT_DIR"
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found" >&2
    exit 1
  fi

  local action="${1:-status}"
  local target="${2:-all}"
  run_action "$action" "$target"
}


main "$@"
