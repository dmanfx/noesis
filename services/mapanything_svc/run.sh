#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
HOST=${MA_SERVICE_HOST:-$(python3 - <<'PY'
from config import load_ma_config
cfg = load_ma_config()
print(cfg.get('service', {}).get('host', '127.0.0.1'))
PY
)}
PORT=${MA_SERVICE_PORT:-$(python3 - <<'PY'
from config import load_ma_config
cfg = load_ma_config()
print(cfg.get('service', {}).get('port', '8001'))
PY
)}
PYTHON_BIN=${PYTHON_BIN:-python3}
exec "${PYTHON_BIN}" -m uvicorn services.mapanything_svc.server:app --host "${HOST}" --port "${PORT}"
