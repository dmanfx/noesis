#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v c++ >/dev/null 2>&1; then
  echo "[FAIL] Missing compiler (c++)." >&2
  exit 1
fi

if ! python3 -c "import pybind11" >/dev/null 2>&1; then
  echo "[FAIL] Missing pybind11 Python package. Install with:" >&2
  echo "  python3 -m pip install 'pybind11==2.12.0'" >&2
  exit 1
fi

PYBIND_VERSION="$(python3 -c 'import pybind11; print(getattr(pybind11, "__version__", ""))')"
if [[ "${PYBIND_VERSION}" != "2.12.0" ]]; then
  echo "[FAIL] pybind11 ${PYBIND_VERSION} is installed, but DeepStream Service Maker was built with pybind11 2.12.0." >&2
  echo "Install the matching version with:" >&2
  echo "  python3 -m pip install --user --force-reinstall 'pybind11==2.12.0'" >&2
  exit 1
fi

PYBIND_INCLUDES="$(python3 -m pybind11 --includes)"
EXT_SUFFIX="$(python3-config --extension-suffix)"
PKG_CFLAGS="$(pkg-config --cflags gstreamer-1.0)"

OUT="$ROOT/noesis_pose_meta_ext${EXT_SUFFIX}"
SRC="$ROOT/native/noesis_pose_meta_ext.cpp"

DS_SM_INC="/opt/nvidia/deepstream/deepstream/service-maker/includes"
DS_INC="/opt/nvidia/deepstream/deepstream/sources/includes"
DS_LIB="/opt/nvidia/deepstream/deepstream/lib"

echo "[INFO] Building ${OUT}"

c++ -O3 -shared -std=c++17 -fPIC \
  ${PYBIND_INCLUDES} \
  ${PKG_CFLAGS} \
  -I"${DS_SM_INC}" \
  -I"${DS_INC}" \
  "${SRC}" \
  -L"${DS_LIB}" \
  -Wl,-rpath,"${DS_LIB}" \
  -lnvds_service_maker \
  -o "${OUT}"

echo "[OK] Built ${OUT}"
