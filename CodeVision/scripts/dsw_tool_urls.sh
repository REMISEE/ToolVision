#!/usr/bin/env bash
set -euo pipefail

# Print CodeVision tool URLs for the current DSW instance.
# Override DSW_TOOL_HOST if the auto-detected address is not the one DLC can reach.

detect_dsw_host() {
  if [[ -n "${DSW_TOOL_HOST:-}" ]]; then
    echo "${DSW_TOOL_HOST}"
    return
  fi

  if command -v ip >/dev/null 2>&1; then
    local candidate
    candidate="$(ip -o -4 addr show scope global | awk '{print $4}' | cut -d/ -f1 | awk '
      $1 ~ /^172\./ { print; exit }
      $1 ~ /^10\./ && !seen10 { seen10=$1 }
      END { if (seen10) print seen10 }
    ')"
    if [[ -n "${candidate}" ]]; then
      echo "${candidate}"
      return
    fi
  fi

  hostname -I | awk '{print $1}'
}

host="$(detect_dsw_host)"
ocr_port="${OCR_PORT:-18080}"
groundedsam2_port="${GROUNDEDSAM2_PORT:-18081}"
depth_port="${DEPTH_PORT:-18082}"
countgd_port="${COUNTGD_PORT:-18083}"

cat <<EOF
export DSW_TOOL_HOST=${host}
export OCR_BASE_URL=http://${host}:${ocr_port}
export GROUNDEDSAM2_BASE_URL=http://${host}:${groundedsam2_port}
export DEPTH_BASE_URL=http://${host}:${depth_port}
export COUNTGD_BASE_URL=http://${host}:${countgd_port}
EOF
