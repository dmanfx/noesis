#!/usr/bin/env bash
set -euo pipefail

DS9_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd -- "${DS9_ROOT}/.." && pwd)"
DS_HOME="${NOESIS_DEEPSTREAM_HOME:-/opt/nvidia/deepstream/deepstream-9.0}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
OUT_DIR="${NOESIS_NATIVE_EXT_DIR:-${DS9_ROOT}/native_extensions}"

if [[ ! -d "${DS_HOME}" ]]; then
  echo "[FAIL] DeepStream 9 home missing: ${DS_HOME}" >&2
  exit 1
fi
if [[ "${DS_HOME}" == *"deepstream-8.0"* ]]; then
  echo "[FAIL] Refusing to build DS9 extensions against DS8: ${DS_HOME}" >&2
  exit 1
fi
if [[ ! -d "${DS_HOME}/service-maker/includes" || ! -d "${DS_HOME}/sources/includes" ]]; then
  echo "[FAIL] DeepStream headers missing under ${DS_HOME}" >&2
  exit 1
fi
if ! command -v c++ >/dev/null 2>&1; then
  echo "[FAIL] Missing compiler (c++)." >&2
  exit 1
fi
if ! python3 -c "import pybind11" >/dev/null 2>&1; then
  echo "[FAIL] Missing pybind11 Python package." >&2
  exit 1
fi
PYBIND_VERSION="$(python3 -c 'import pybind11; print(getattr(pybind11, "__version__", ""))')"
if [[ "${PYBIND_VERSION}" != "2.12.0" ]]; then
  echo "[FAIL] pybind11 ${PYBIND_VERSION} is installed, but DeepStream Service Maker was built with pybind11 2.12.0." >&2
  echo "Install the matching version with:" >&2
  echo "  python3 -m pip install --break-system-packages --force-reinstall 'pybind11==2.12.0'" >&2
  exit 1
fi

PYBIND_INCLUDES="$(python3 -m pybind11 --includes)"
if command -v python3-config >/dev/null 2>&1; then
  EXT_SUFFIX="$(python3-config --extension-suffix)"
else
  EXT_SUFFIX="$(python3 - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
PY
)"
fi
PKG_CFLAGS="$(pkg-config --cflags gstreamer-1.0)"
DS_SM_INC="${DS_HOME}/service-maker/includes"
DS_INC="${DS_HOME}/sources/includes"
DS_LIB="${DS_HOME}/lib"
CUDA_INC="${CUDA_HOME}/include"
CUDA_LIB="${CUDA_HOME}/targets/x86_64-linux/lib"

mkdir -p "${OUT_DIR}"

build_ext() {
  local module="$1"
  local src="${DS9_ROOT}/native/${module}.cpp"
  local out="${OUT_DIR}/${module}${EXT_SUFFIX}"
  local kernel_src="${src%.cpp}_kernels.cu"
  local kernel_obj=""
  local extra_objects=()
  shift
  if [[ ! -f "${src}" ]]; then
    echo "[FAIL] Missing source: ${src}" >&2
    exit 1
  fi
  if [[ ! -f "${kernel_src}" && "${src}" == *_ext.cpp ]]; then
    kernel_src="${src%_ext.cpp}_kernels.cu"
  fi
  if [[ -f "${kernel_src}" ]]; then
    if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
      echo "[FAIL] CUDA compiler not found: ${CUDA_HOME}/bin/nvcc" >&2
      exit 1
    fi
    mkdir -p "${DS9_ROOT}/build/native"
    kernel_obj="${DS9_ROOT}/build/native/$(basename "${kernel_src%.cu}").o"
    echo "[INFO] Compiling CUDA kernel: ${kernel_src}"
    "${CUDA_HOME}/bin/nvcc" -O3 -std=c++17 -Xcompiler -fPIC \
      -I"${CUDA_INC}" \
      -c "${kernel_src}" \
      -o "${kernel_obj}"
    extra_objects+=("${kernel_obj}")
  fi
  echo "[INFO] Building ${out}"
  c++ -O3 -shared -std=c++17 -fPIC \
    ${PYBIND_INCLUDES} \
    ${PKG_CFLAGS} \
    -I"${DS_SM_INC}" \
    -I"${DS_INC}" \
    -I"${CUDA_INC}" \
    "${src}" \
    "${extra_objects[@]}" \
    -L"${DS_LIB}" \
    -L"${CUDA_LIB}" \
    -Wl,-rpath,"${DS_LIB}" \
    -Wl,-rpath,"${CUDA_LIB}" \
    -lnvds_service_maker \
    "$@" \
    -o "${out}"
}

build_ext noesis_pose_meta_ext
build_ext noesis_v3dt_meta_ext
build_ext noesis_reid_meta_ext
build_ext noesis_latency_ext
build_ext noesis_depth_meta_ext
build_ext noesis_depth_tracking_tensor_ext -lcudart -lnppig -lnppidei -lnppc

echo "[OK] Built DS9 native extensions in ${OUT_DIR}"
