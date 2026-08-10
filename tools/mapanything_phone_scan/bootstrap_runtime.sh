#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
da3_source="${repo_root}/external/Depth-Anything-3"
runtime_root="${NOESIS_PHONE_SCAN_RUNTIME_ROOT:-${repo_root}/data/mapanything_phone_scan_runtime}"
venv_dir="${runtime_root}/venv"
python_bin="${NOESIS_PHONE_SCAN_BOOTSTRAP_PYTHON:-python3}"

# Keep the phone tool aligned with one tested official MapAnything release while
# reusing this machine's existing CUDA-enabled PyTorch and Noesis dependencies.
mapanything_commit="9d1db2dd728bd8a10e74b15d2eb646e1bf933791"
da3_commit="3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"

observed_da3_commit="$(git -C "${da3_source}" rev-parse HEAD)"
if [[ "${observed_da3_commit}" != "${da3_commit}" ]]; then
    echo "Official DA3 source revision mismatch: ${observed_da3_commit}" >&2
    exit 1
fi

mkdir -p "${runtime_root}"
if [[ ! -x "${venv_dir}/bin/python" ]]; then
    "${python_bin}" -m venv --system-site-packages "${venv_dir}"
fi

"${venv_dir}/bin/python" -m pip install \
    --disable-pip-version-check \
    --no-deps \
    --upgrade \
    "uniception==0.1.7"

"${venv_dir}/bin/python" -m pip install \
    --disable-pip-version-check \
    --no-deps \
    --upgrade \
    --force-reinstall \
    "mapanything @ git+https://github.com/facebookresearch/map-anything.git@${mapanything_commit}"

cd "${repo_root}"
PYTHONPATH="${da3_source}/src${PYTHONPATH:+:${PYTHONPATH}}" "${venv_dir}/bin/python" - <<'PY'
from importlib.metadata import version
from pathlib import Path

import mapanything
import depth_anything_3

expected = {"mapanything": "1.1.3", "uniception": "0.1.7"}
actual = {name: version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"MapAnything phone runtime version mismatch: {actual}")
module_path = Path(mapanything.__file__).resolve()
if "mapanything_phone_scan_runtime" not in module_path.as_posix():
    raise SystemExit(f"MapAnything did not resolve from the isolated phone runtime: {module_path}")
print(f"MapAnything phone runtime ready: {actual} ({module_path})")
print(f"DA3 phone runtime source ready: {Path(depth_anything_3.__path__[0]).resolve()}")
PY
