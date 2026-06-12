#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/cpfs/delinmao}"
TOOLVISION_ROOT="${TOOLVISION_ROOT:-${WORKSPACE_ROOT}/ToolVision}"
ROOT_DIR="${ROOT_DIR:-${TOOLVISION_ROOT}/CodeVision}"
LOG_DIR="${SERVICE_LOG_DIR:-${ROOT_DIR}/outputs/service_logs}"
PID_DIR="${SERVICE_PID_DIR:-${ROOT_DIR}/outputs/service_pids}"

OCR_ENV="${OCR_ENV:-paddleocr}"
OCR_HOST="${OCR_HOST:-0.0.0.0}"
OCR_PORT="${OCR_PORT:-8080}"
OCR_PIPELINE="${OCR_PIPELINE:-$ROOT_DIR/recipe/codevision/config/ocr_no_doc_preprocessor.yaml}"
OCR_DEVICE="${OCR_DEVICE:-gpu:0}"
OCR_CUDA_VISIBLE_DEVICES="${OCR_CUDA_VISIBLE_DEVICES:-1}"
OCR_USE_HPIP="${OCR_USE_HPIP:-0}"
OCR_HPI_CONFIG="${OCR_HPI_CONFIG:-}"

GROUNDEDSAM2_ENV="${GROUNDEDSAM2_ENV:-groundedsam2}"
GROUNDEDSAM2_HOST="${GROUNDEDSAM2_HOST:-0.0.0.0}"
GROUNDEDSAM2_PORT="${GROUNDEDSAM2_PORT:-8081}"
GROUNDEDSAM2_DEVICE="${GROUNDEDSAM2_DEVICE:-cuda}"
GROUNDEDSAM2_CUDA_VISIBLE_DEVICES="${GROUNDEDSAM2_CUDA_VISIBLE_DEVICES:-1}"
GROUNDEDSAM2_DEFAULT_TEXT_PROMPT="${GROUNDEDSAM2_DEFAULT_TEXT_PROMPT:-object.}"
GROUNDEDSAM2_BOX_THRESHOLD="${GROUNDEDSAM2_BOX_THRESHOLD:-0.35}"
GROUNDEDSAM2_TEXT_THRESHOLD="${GROUNDEDSAM2_TEXT_THRESHOLD:-0.25}"
GROUNDEDSAM2_ROOT="${GROUNDEDSAM2_ROOT:-${TOOLVISION_ROOT}/Grounded-SAM-2}"
GROUNDEDSAM2_SAM2_CHECKPOINT="${GROUNDEDSAM2_SAM2_CHECKPOINT:-${GROUNDEDSAM2_ROOT}/checkpoints/sam2.1_hiera_tiny.pt}"
GROUNDEDSAM2_SAM2_MODEL_CONFIG="${GROUNDEDSAM2_SAM2_MODEL_CONFIG:-${GROUNDEDSAM2_ROOT}/sam2/configs/sam2.1/sam2.1_hiera_t.yaml}"
GROUNDEDSAM2_GDINO_CONFIG="${GROUNDEDSAM2_GDINO_CONFIG:-${GROUNDEDSAM2_ROOT}/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
GROUNDEDSAM2_GDINO_CHECKPOINT="${GROUNDEDSAM2_GDINO_CHECKPOINT:-${GROUNDEDSAM2_ROOT}/gdino_checkpoints/groundingdino_swint_ogc.pth}"

DEPTH_ENV="${DEPTH_ENV:-depth-pro}"
DEPTH_HOST="${DEPTH_HOST:-0.0.0.0}"
DEPTH_PORT="${DEPTH_PORT:-8082}"
DEPTH_DEVICE="${DEPTH_DEVICE:-cuda}"
DEPTH_CUDA_VISIBLE_DEVICES="${DEPTH_CUDA_VISIBLE_DEVICES:-2}"
DEPTH_ROOT="${DEPTH_ROOT:-${TOOLVISION_ROOT}/ml-depth-pro-main}"
DEPTH_CHECKPOINT_PATH="${DEPTH_CHECKPOINT_PATH:-checkpoints/depth_pro.pt}"
DEPTH_CACHE_SIZE="${DEPTH_CACHE_SIZE:-8}"
DEPTH_REQUEST_TIMEOUT="${DEPTH_REQUEST_TIMEOUT:-180}"
DEPTH_DEFAULT_TEXT_PROMPT="${DEPTH_DEFAULT_TEXT_PROMPT:-object.}"
DEPTH_GROUNDEDSAM2_BASE_URL="${DEPTH_GROUNDEDSAM2_BASE_URL:-http://127.0.0.1:8081}"
DEPTH_BOX_THRESHOLD="${DEPTH_BOX_THRESHOLD:-0.35}"
DEPTH_TEXT_THRESHOLD="${DEPTH_TEXT_THRESHOLD:-0.25}"

COUNTGD_ENV="${COUNTGD_ENV:-countgd}"
COUNTGD_HOST="${COUNTGD_HOST:-0.0.0.0}"
COUNTGD_PORT="${COUNTGD_PORT:-8083}"
COUNTGD_DEVICE="${COUNTGD_DEVICE:-cuda}"
COUNTGD_CUDA_VISIBLE_DEVICES="${COUNTGD_CUDA_VISIBLE_DEVICES:-3}"
COUNTGD_ROOT="${COUNTGD_ROOT:-${TOOLVISION_ROOT}/CountGD}"
COUNTGD_CONFIG_PATH="${COUNTGD_CONFIG_PATH:-config/cfg_fsc147_vit_b.py}"
COUNTGD_PRETRAIN_MODEL_PATH="${COUNTGD_PRETRAIN_MODEL_PATH:-checkpoints/checkpoint_fsc147_best.pth}"
COUNTGD_TEXT_ENCODER_TYPE="${COUNTGD_TEXT_ENCODER_TYPE:-checkpoints/bert-base-uncased}"
COUNTGD_DEFAULT_CONFIDENCE_THRESH="${COUNTGD_DEFAULT_CONFIDENCE_THRESH:-0.23}"
COUNTGD_DEFAULT_VISUALIZE="${COUNTGD_DEFAULT_VISUALIZE:-heatmap}"
COUNTGD_HEATMAP_SIGMA="${COUNTGD_HEATMAP_SIGMA:-5.0}"
SERVICE_WARMUP_TIMEOUT_S="${SERVICE_WARMUP_TIMEOUT_S:-300}"


usage() {
  cat <<'EOF'
Usage:
  scripts/launch_external_services.sh start [ocr|groundedsam2|depth|countgd|all]
  scripts/launch_external_services.sh stop [ocr|groundedsam2|depth|countgd|all]
  scripts/launch_external_services.sh restart [ocr|groundedsam2|depth|countgd|all]
  scripts/launch_external_services.sh status [ocr|groundedsam2|depth|countgd|all]

Environment overrides:
  OCR_ENV, OCR_HOST, OCR_PORT, OCR_PIPELINE, OCR_DEVICE, OCR_CUDA_VISIBLE_DEVICES, OCR_USE_HPIP, OCR_HPI_CONFIG
  GROUNDEDSAM2_ENV, GROUNDEDSAM2_HOST, GROUNDEDSAM2_PORT, GROUNDEDSAM2_DEVICE
  GROUNDEDSAM2_CUDA_VISIBLE_DEVICES, GROUNDEDSAM2_DEFAULT_TEXT_PROMPT
  GROUNDEDSAM2_BOX_THRESHOLD, GROUNDEDSAM2_TEXT_THRESHOLD
  GROUNDEDSAM2_SAM2_CHECKPOINT, GROUNDEDSAM2_SAM2_MODEL_CONFIG
  GROUNDEDSAM2_GDINO_CONFIG, GROUNDEDSAM2_GDINO_CHECKPOINT
  DEPTH_ENV, DEPTH_HOST, DEPTH_PORT, DEPTH_DEVICE
  DEPTH_CUDA_VISIBLE_DEVICES, DEPTH_ROOT, DEPTH_CHECKPOINT_PATH
  DEPTH_CACHE_SIZE, DEPTH_REQUEST_TIMEOUT, DEPTH_DEFAULT_TEXT_PROMPT
  DEPTH_GROUNDEDSAM2_BASE_URL
  DEPTH_BOX_THRESHOLD, DEPTH_TEXT_THRESHOLD
  COUNTGD_ENV, COUNTGD_HOST, COUNTGD_PORT, COUNTGD_DEVICE
  COUNTGD_CUDA_VISIBLE_DEVICES, COUNTGD_ROOT, COUNTGD_CONFIG_PATH
  COUNTGD_PRETRAIN_MODEL_PATH, COUNTGD_TEXT_ENCODER_TYPE
  COUNTGD_DEFAULT_CONFIDENCE_THRESH, COUNTGD_DEFAULT_VISUALIZE, COUNTGD_HEATMAP_SIGMA
  SERVICE_LOG_DIR, SERVICE_PID_DIR, SERVICE_WARMUP_TIMEOUT_S
EOF
}


ensure_dirs() {
  mkdir -p "$LOG_DIR" "$PID_DIR"
}


resolve_env_python() {
  local env_name="$1"
  local conda_base
  conda_base="$(conda info --base)"

  if [[ -d "$env_name" && -x "$env_name/bin/python" ]]; then
    echo "$env_name/bin/python"
    return
  fi

  if [[ "$env_name" == "base" ]]; then
    echo "$conda_base/bin/python"
    return
  fi

  local env_path=""
  env_path="$(conda env list | awk -v env_name="$env_name" '$1 == env_name { print $NF; exit }')"

  local python_path=""
  if [[ -n "$env_path" ]]; then
    python_path="$env_path/bin/python"
  else
    local candidate_dir
    for candidate_dir in "${CONDA_ENVS_DIR:-}" "${WORKSPACE_ROOT}/envs" "$conda_base/envs"; do
      [[ -n "$candidate_dir" ]] || continue
      if [[ -x "$candidate_dir/$env_name/bin/python" ]]; then
        python_path="$candidate_dir/$env_name/bin/python"
        break
      fi
    done
  fi

  if [[ ! -x "$python_path" ]]; then
    echo "Python not found for conda env '$env_name'. Set ${env_name^^}_ENV to an env path or set CONDA_ENVS_DIR." >&2
    exit 1
  fi
  echo "$python_path"
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


port_listener_pids() {
  local port="$1"
  if ! command -v ss >/dev/null 2>&1; then
    return
  fi
  ss -ltnp "( sport = :${port} )" 2>/dev/null \
    | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' \
    | sort -u
}


stop_port_listeners() {
  local name="$1"
  local port="$2"
  [[ -n "$port" ]] || return 0

  local pids
  pids="$(port_listener_pids "$port" || true)"
  [[ -n "$pids" ]] || return 0

  echo "[stop] $name port=$port pids=$(echo "$pids" | tr '\n' ' ')"
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    kill "$pid" 2>/dev/null || true
  done <<<"$pids"
  sleep "${SERVICE_STOP_DELAY_S:-5}"
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      echo "[kill] $name pid=$pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done <<<"$pids"
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
  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" >"$log" 2>&1 < /dev/null &
  else
    nohup "$@" >"$log" 2>&1 < /dev/null &
  fi
  local pid=$!
  echo "$pid" >"$(pid_file "$name")"
  echo "[start] $name pid=$pid log=$log"
}


stop_process() {
  local name="$1"
  local port="${2:-}"
  local file
  file="$(pid_file "$name")"
  if ! [[ -f "$file" ]]; then
    echo "[skip] $name not running"
    stop_port_listeners "$name" "$port"
    return
  fi

  local pid
  pid="$(cat "$file")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "[stop] $name pid=$pid"
  else
    echo "[clean] $name stale pid=$pid"
    stop_port_listeners "$name" "$port"
  fi
  rm -f "$file"
}


status_process() {
  local name="$1"
  local port="${2:-}"
  if is_running "$name"; then
    echo "[up] $name pid=$(cat "$(pid_file "$name")") log=$(log_file "$name")"
  elif [[ -n "$port" && -n "$(port_listener_pids "$port" || true)" ]]; then
    echo "[up:port-only] $name port=$port pids=$(port_listener_pids "$port" | tr '\n' ' ') log=$(log_file "$name")"
  else
    echo "[down] $name"
  fi
}


warmup_services() {
  local target="$1"
  local python_bin
  if command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
  else
    python_bin="$(resolve_env_python base)"
  fi

  "$python_bin" scripts/warmup_external_services.py "$target" \
    --ocr-host "127.0.0.1" \
    --ocr-port "$OCR_PORT" \
    --groundedsam2-host "127.0.0.1" \
    --groundedsam2-port "$GROUNDEDSAM2_PORT" \
    --depth-host "127.0.0.1" \
    --depth-port "$DEPTH_PORT" \
    --countgd-host "127.0.0.1" \
    --countgd-port "$COUNTGD_PORT" \
    --timeout-s "$SERVICE_WARMUP_TIMEOUT_S"
}


start_ocr() {
  local python_bin
  python_bin="$(resolve_env_python "$OCR_ENV")"
  local -a cmd=()
  if [[ -n "$OCR_CUDA_VISIBLE_DEVICES" ]]; then
    cmd+=(env "CUDA_VISIBLE_DEVICES=$OCR_CUDA_VISIBLE_DEVICES")
  fi
  cmd+=("$python_bin" scripts/launch_paddleocr_service.py
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
  local python_bin
  python_bin="$(resolve_env_python "$GROUNDEDSAM2_ENV")"
  local -a cmd=()
  if [[ -n "$GROUNDEDSAM2_CUDA_VISIBLE_DEVICES" ]]; then
    cmd+=(env "CUDA_VISIBLE_DEVICES=$GROUNDEDSAM2_CUDA_VISIBLE_DEVICES")
  fi
  cmd+=("$python_bin" scripts/launch_groundedsam2_service.py
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


start_depth() {
  local python_bin
  python_bin="$(resolve_env_python "$DEPTH_ENV")"
  local -a cmd=()
  if [[ -n "$DEPTH_CUDA_VISIBLE_DEVICES" ]]; then
    cmd+=(env "CUDA_VISIBLE_DEVICES=$DEPTH_CUDA_VISIBLE_DEVICES")
  fi
  cmd+=("$python_bin" scripts/launch_depth_service.py
    --host "$DEPTH_HOST"
    --port "$DEPTH_PORT"
    --depth-pro-root "$DEPTH_ROOT"
    --checkpoint-path "$DEPTH_CHECKPOINT_PATH"
    --cache-size "$DEPTH_CACHE_SIZE"
    --request-timeout "$DEPTH_REQUEST_TIMEOUT"
    --default-text-prompt "$DEPTH_DEFAULT_TEXT_PROMPT"
    --groundedsam2-base-url "$DEPTH_GROUNDEDSAM2_BASE_URL"
    --box-threshold "$DEPTH_BOX_THRESHOLD"
    --text-threshold "$DEPTH_TEXT_THRESHOLD")
  if [[ -n "$DEPTH_DEVICE" ]]; then
    cmd+=(--device "$DEPTH_DEVICE")
  fi
  start_process "depth" "${cmd[@]}"
}


start_countgd() {
  local python_bin
  python_bin="$(resolve_env_python "$COUNTGD_ENV")"
  local -a cmd=()
  if [[ -n "$COUNTGD_CUDA_VISIBLE_DEVICES" ]]; then
    cmd+=(env "CUDA_VISIBLE_DEVICES=$COUNTGD_CUDA_VISIBLE_DEVICES")
  fi
  cmd+=("$python_bin" scripts/launch_countgd_service.py
    --host "$COUNTGD_HOST"
    --port "$COUNTGD_PORT"
    --countgd-root "$COUNTGD_ROOT"
    --device "$COUNTGD_DEVICE"
    --config-path "$COUNTGD_CONFIG_PATH"
    --pretrain-model-path "$COUNTGD_PRETRAIN_MODEL_PATH"
    --text-encoder-type "$COUNTGD_TEXT_ENCODER_TYPE"
    --default-confidence-thresh "$COUNTGD_DEFAULT_CONFIDENCE_THRESH"
    --default-visualize "$COUNTGD_DEFAULT_VISUALIZE"
    --heatmap-sigma "$COUNTGD_HEATMAP_SIGMA")
  start_process "countgd" "${cmd[@]}"
}


run_action() {
  local action="$1"
  local target="$2"
  case "$target" in
    all)
      run_action "$action" ocr
      run_action "$action" groundedsam2
      run_action "$action" depth
      run_action "$action" countgd
      ;;
    ocr)
      case "$action" in
        start) start_ocr ;;
        stop) stop_process "paddleocr" "$OCR_PORT" ;;
        status) status_process "paddleocr" "$OCR_PORT" ;;
        restart)
          stop_process "paddleocr" "$OCR_PORT"
          start_ocr
          ;;
        *) usage; exit 1 ;;
      esac
      ;;
    groundedsam2)
      case "$action" in
        start) start_groundedsam2 ;;
        stop) stop_process "groundedsam2" "$GROUNDEDSAM2_PORT" ;;
        status) status_process "groundedsam2" "$GROUNDEDSAM2_PORT" ;;
        restart)
          stop_process "groundedsam2" "$GROUNDEDSAM2_PORT"
          start_groundedsam2
          ;;
        *) usage; exit 1 ;;
      esac
      ;;
    depth)
      case "$action" in
        start) start_depth ;;
        stop) stop_process "depth" "$DEPTH_PORT" ;;
        status) status_process "depth" "$DEPTH_PORT" ;;
        restart)
          stop_process "depth" "$DEPTH_PORT"
          start_depth
          ;;
        *) usage; exit 1 ;;
      esac
      ;;
    countgd)
      case "$action" in
        start) start_countgd ;;
        stop) stop_process "countgd" "$COUNTGD_PORT" ;;
        status) status_process "countgd" "$COUNTGD_PORT" ;;
        restart)
          stop_process "countgd" "$COUNTGD_PORT"
          start_countgd
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
  if [[ "$action" == "start" || "$action" == "restart" ]]; then
    if [[ "${SKIP_WARMUP:-0}" != "1" ]]; then
      sleep "${SERVICE_STARTUP_DELAY_S:-100}"
      warmup_services "$target"
    fi
  fi
}


main "$@"
