#!/usr/bin/env bash

# Simple GPU utilization logger for DS8 debugging.
# Usage: scripts/gpu_util_logger.sh [log_path] [interval_seconds]

set -euo pipefail

LOG_PATH="${1:-/tmp/gpu_util_ds8.log}"
INTERVAL="${2:-1}"

mkdir -p "$(dirname "$LOG_PATH")"

echo "# timestamp,gpu_index,utilization_gpu_percent" >> "$LOG_PATH"

while true; do
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu \
      --format=csv,noheader,nounits >> "$LOG_PATH" || true
  else
    echo "$(date +'%Y-%m-%d %H:%M:%S'),-1,0" >> "$LOG_PATH"
  fi
  sleep "$INTERVAL"
done

