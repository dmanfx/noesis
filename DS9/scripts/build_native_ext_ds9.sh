#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=ds9_build_env.sh
source "${ROOT}/scripts/ds9_build_env.sh"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <module_name> <source.cpp> [plain|cuda_npp]" >&2
  exit 2
fi

MODULE="$1"
SRC="$2"
KIND="${3:-plain}"

[[ -f "${SRC}" ]] || ds9_fail "Native extension source not found: ${SRC}"

ds9_require_python_build_tools
DS_HOME="$(ds9_require_deepstream_home)"
CUDA_HOME=""
if [[ "${KIND}" == "cuda_npp" ]]; then
  CUDA_HOME="$(ds9_require_cuda_home)"
fi

PYBIND_INCLUDES="$(python3 -m pybind11 --includes)"
EXT_SUFFIX="$(python3-config --extension-suffix)"
PKG_CFLAGS="$(pkg-config --cflags gstreamer-1.0)"

OUT_DIR="${NOESIS_NATIVE_EXT_DIR:-${ROOT}/native_extensions}"
mkdir -p "${OUT_DIR}"
OUT="${OUT_DIR}/${MODULE}${EXT_SUFFIX}"

DS_SM_INC="${DS_HOME}/service-maker/includes"
DS_INC="${DS_HOME}/sources/includes"
DS_LIB="${DS_HOME}/lib"

CXXFLAGS=(-O3 -shared -std=c++17 -fPIC)
LDFLAGS=(-L"${DS_LIB}" -Wl,-rpath,"${DS_LIB}" -lnvds_service_maker -lnvds_meta -lnvdsgst_meta)
EXTRA_OBJECTS=()

if [[ "${KIND}" == "cuda_npp" ]]; then
  CXXFLAGS+=(-I"${CUDA_HOME}/include")
  LDFLAGS+=(-L"${CUDA_HOME}/targets/x86_64-linux/lib" -L"${CUDA_HOME}/lib64")
  LDFLAGS+=(-Wl,-rpath,"${CUDA_HOME}/targets/x86_64-linux/lib" -Wl,-rpath,"${CUDA_HOME}/lib64")
  LDFLAGS+=(-lcudart -lnppig -lnppidei -lnppc)

  KERNEL_SRC="${SRC%.cpp}_kernels.cu"
  if [[ ! -f "${KERNEL_SRC}" && "${SRC}" == *_ext.cpp ]]; then
    KERNEL_SRC="${SRC%_ext.cpp}_kernels.cu"
  fi
  if [[ -f "${KERNEL_SRC}" ]]; then
    NVCC="${CUDA_HOME}/bin/nvcc"
    [[ -x "${NVCC}" ]] || ds9_fail "CUDA compiler not found: ${NVCC}"
    BUILD_DIR="${ROOT}/build/native"
    mkdir -p "${BUILD_DIR}"
    KERNEL_OBJ="${BUILD_DIR}/$(basename "${KERNEL_SRC%.cu}").o"
    echo "[INFO] Compiling CUDA kernel: ${KERNEL_SRC}"
    "${NVCC}" -O3 -std=c++17 -Xcompiler -fPIC \
      -I"${CUDA_HOME}/include" \
      -c "${KERNEL_SRC}" \
      -o "${KERNEL_OBJ}"
    EXTRA_OBJECTS+=("${KERNEL_OBJ}")
  fi
fi

echo "[INFO] Building ${MODULE} for DS9"
echo "[INFO] DeepStream root: ${DS_HOME}"
echo "[INFO] Output: ${OUT}"

# Deliberately unquoted word expansion for pybind/pkg-config include flag lists.
# shellcheck disable=SC2086
c++ "${CXXFLAGS[@]}" \
  ${PYBIND_INCLUDES} \
  ${PKG_CFLAGS} \
  -I"${DS_SM_INC}" \
  -I"${DS_INC}" \
  "${SRC}" \
  "${EXTRA_OBJECTS[@]}" \
  "${LDFLAGS[@]}" \
  -o "${OUT}"

echo "[OK] Built ${OUT}"
