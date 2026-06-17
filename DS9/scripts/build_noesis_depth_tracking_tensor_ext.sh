#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT}/scripts/build_native_ext_ds9.sh" \
  noesis_depth_tracking_tensor_ext \
  "${ROOT}/native/noesis_depth_tracking_tensor_ext.cpp" \
  cuda_npp
