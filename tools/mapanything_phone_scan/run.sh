#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
runtime_root="${NOESIS_PHONE_SCAN_RUNTIME_ROOT:-${repo_root}/data/mapanything_phone_scan_runtime}"
runtime_python="${runtime_root}/venv/bin/python"
export PYTHONPATH="${repo_root}/external/Depth-Anything-3/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${repo_root}"

runtime_ok=0
if [[ -x "${runtime_python}" ]]; then
    if "${runtime_python}" - <<'PY' >/dev/null 2>&1
from importlib.metadata import version
from pathlib import Path
import mapanything
import depth_anything_3

assert version("mapanything") == "1.1.3"
assert version("uniception") == "0.1.7"
assert "mapanything_phone_scan_runtime" in Path(mapanything.__file__).resolve().as_posix()
assert Path(depth_anything_3.__path__[0]).resolve() == Path("external/Depth-Anything-3/src/depth_anything_3").resolve()
PY
    then
        runtime_ok=1
    fi
fi

if [[ "${runtime_ok}" != "1" ]]; then
    echo "Preparing the isolated MapAnything phone runtime (first start only)..."
    "${script_dir}/bootstrap_runtime.sh"
fi

exec "${runtime_python}" -m tools.mapanything_phone_scan
