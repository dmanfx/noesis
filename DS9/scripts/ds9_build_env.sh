#!/usr/bin/env bash
# Shared DS9 build helpers. These helpers intentionally refuse to use the
# mutable /opt/nvidia/deepstream/deepstream symlink.

set -euo pipefail

ds9_repo_root() {
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd
}

ds9_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

ds9_fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

ds9_realpath() {
  readlink -f "$1" 2>/dev/null || realpath "$1"
}

ds9_require_deepstream_home() {
  local candidate="${DS9_DEEPSTREAM_HOME:-/opt/nvidia/deepstream/deepstream-9.1}"
  if [[ ! -d "${candidate}" ]]; then
    ds9_fail "DeepStream 9 root not found: ${candidate}. Set DS9_DEEPSTREAM_HOME to an installed DS9 root."
  fi

  local resolved
  resolved="$(ds9_realpath "${candidate}")"
  case "${resolved}" in
    *deepstream-8.0*|*deepstream-8*) ds9_fail "Refusing DS8 DeepStream root for DS9 build: ${resolved}" ;;
  esac

  local version_h="${resolved}/sources/includes/nvds_version.h"
  local nvdsmeta_h="${resolved}/sources/includes/nvdsmeta.h"
  local custom_impl_h="${resolved}/sources/includes/nvdsinfer_custom_impl.h"
  local metadata_hpp="${resolved}/service-maker/includes/metadata.hpp"

  [[ -f "${version_h}" ]] || ds9_fail "Missing DS9 version header: ${version_h}"
  [[ -f "${nvdsmeta_h}" ]] || ds9_fail "Missing DS metadata header: ${nvdsmeta_h}"
  [[ -f "${custom_impl_h}" ]] || ds9_fail "Missing nvinfer custom parser header: ${custom_impl_h}"
  [[ -f "${metadata_hpp}" ]] || ds9_fail "Missing Service Maker metadata header: ${metadata_hpp}"

  if ! grep -Eq 'NVDS_VERSION_MAJOR[[:space:]]+9' "${version_h}" \
      || ! grep -Eq 'NVDS_VERSION_MINOR[[:space:]]+1' "${version_h}"; then
    ds9_fail "DeepStream root does not identify as version 9.1: ${resolved}"
  fi

  printf '%s\n' "${resolved}"
}

ds9_require_cuda_home() {
  local candidate="${DS9_CUDA_HOME:-/usr/local/cuda-13.2}"
  if [[ ! -d "${candidate}" && -d /usr/local/cuda ]]; then
    candidate="/usr/local/cuda"
  fi
  [[ -d "${candidate}" ]] || ds9_fail "CUDA root not found. Set DS9_CUDA_HOME to CUDA 13.2."
  [[ -f "${candidate}/include/cuda_runtime_api.h" ]] || ds9_fail "Missing CUDA headers under ${candidate}"
  local resolved version_file
  resolved="$(ds9_realpath "${candidate}")"
  version_file="${resolved}/version.json"
  if [[ -f "${version_file}" ]] && ! grep -Eq '"version"[[:space:]]*:[[:space:]]*"13\.2' "${version_file}"; then
    if [[ -z "${DS9_ALLOW_CUDA_MISMATCH:-}" ]]; then
      ds9_fail "CUDA root does not identify as CUDA 13.2: ${resolved}. Set DS9_ALLOW_CUDA_MISMATCH=1 only for investigation."
    fi
  fi
  printf '%s\n' "${resolved}"
}

ds9_require_python_build_tools() {
  command -v c++ >/dev/null 2>&1 || ds9_fail "Missing compiler: c++"
  command -v python3 >/dev/null 2>&1 || ds9_fail "Missing python3"
  python3 -c "import pybind11" >/dev/null 2>&1 || ds9_fail "Missing pybind11 Python package"

  local pybind_version
  pybind_version="$(python3 -c 'import pybind11; print(getattr(pybind11, "__version__", ""))')"
  if [[ "${pybind_version}" != "2.12.0" && -z "${DS9_ALLOW_PYBIND_MISMATCH:-}" ]]; then
    ds9_fail "pybind11 ${pybind_version} is installed; expected 2.12.0 for Service Maker. Set DS9_ALLOW_PYBIND_MISMATCH=1 only for investigation."
  fi

  command -v pkg-config >/dev/null 2>&1 || ds9_fail "Missing pkg-config"
  pkg-config --exists gstreamer-1.0 || ds9_fail "Missing gstreamer-1.0 pkg-config metadata"
}
