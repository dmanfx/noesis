#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT}/scripts/build_native_ext_ds9.sh" noesis_pose_meta_ext "${ROOT}/native/noesis_pose_meta_ext.cpp" cuda_runtime
"${ROOT}/scripts/build_native_ext_ds9.sh" noesis_depth_meta_ext "${ROOT}/native/noesis_depth_meta_ext.cpp"
"${ROOT}/scripts/build_native_ext_ds9.sh" noesis_depth_tracking_tensor_ext "${ROOT}/native/noesis_depth_tracking_tensor_ext.cpp" cuda_npp
"${ROOT}/scripts/build_native_ext_ds9.sh" noesis_reid_meta_ext "${ROOT}/native/noesis_reid_meta_ext.cpp" cuda_runtime
"${ROOT}/scripts/build_native_ext_ds9.sh" noesis_v3dt_meta_ext "${ROOT}/native/noesis_v3dt_meta_ext.cpp"
"${ROOT}/scripts/build_native_ext_ds9.sh" noesis_latency_ext "${ROOT}/native/noesis_latency_ext.cpp"
