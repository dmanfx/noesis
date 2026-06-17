#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/.."

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || fail "Missing required file: $1"
}

require_dir() {
  [[ -d "$1" ]] || fail "Missing required directory: $1"
}

echo "[INFO] Running DS9 static prep checks"

require_file "DS9/DS9_PREP_DECISIONS.md"
require_file "DS9/scripts/ds9_build_env.sh"
require_file "DS9/scripts/check_ds9_prereqs.sh"
require_file "DS9/scripts/build_all_native_ds9.sh"
require_file "DS9/scripts/build_all_parsers_ds9.sh"
require_file "DS9/scripts/build_nvdsroiexclude_ds9.sh"
require_file "DS9/noesis/ds9_runtime.py"
require_file "DS9/noesis/ds9_runtime_core.py"
require_file "DS9/csrc/nvdsroiexclude/gstnvdsroiexclude.cpp"
require_file "DS9/csrc/nvdsroiexclude/CMakeLists.txt"
require_file "DS9/config/infer.yaml"
require_file "DS9/config/config_nvdsanalytics_post.ini"
require_file "DS9/config/nvtracker.yaml"
require_file "DS9/config/depth_registration.json"
require_file "DS9/config/dewarper_g3_instant_charuco_1080.txt"
require_file "DS9/pipelines/config_preproc.ini"
require_file "DS9/pipelines/config_infer_primary_yolo11_seg.ini"
require_file "DS9/pipelines/config_infer_primary_yolo26_seg.template.ini"
require_file "DS9/pipelines/config_infer_primary_rfdetr_seg.ini"
require_file "DS9/pipelines/config_infer_secondary_yolo26_pose.ini"
require_file "DS9/pipelines/config_infer_secondary_reid_osnet.ini"
require_file "DS9/pipelines/config_infer_secondary_mapanything.ini"
require_file "DS9/build/config_infer_depth_tracking_da2_vits_294x518_b3_i1.ini"

if [[ -e "DS9/noesis/ds8_runtime.py" ]]; then
  fail "Stale DS8 runtime copy exists under DS9/noesis"
fi

rg -q "from noesis.ds9_runtime_core import main" DS9/noesis/ds9_runtime.py \
  || fail "DS9 launcher does not delegate to noesis.ds9_runtime_core"

if rg --no-ignore -n "from noesis\\.ds8_runtime|import noesis\\.ds8_runtime|noesis/ds8_runtime\\.py" \
  DS9/noesis/ds9_runtime.py DS9/noesis/ds9_runtime_core.py DS9/scripts -g '*.py' \
  >/tmp/ds9_old_runtime_refs.txt; then
  cat /tmp/ds9_old_runtime_refs.txt >&2
  fail "DS9 executable code still references the DS8 runtime entry point"
fi

if [[ -d "DS9/pipelines/nvdsinfer_yolo26_pose" ]]; then
  fail "No-op YOLO26 pose parser is still in active DS9 parser path"
fi
require_dir "DS9/archive/nvdsinfer_yolo26_pose"

if rg -n "custom-lib-path|parse-bbox" DS9/pipelines/config_infer_secondary_yolo26_pose.ini >/tmp/ds9_pose_parser_refs.txt; then
  cat /tmp/ds9_pose_parser_refs.txt >&2
  fail "Pose SGIE config still references a parser/custom lib"
fi

static_prep_paths=(
  DS9/config
  DS9/pipelines
  DS9/build/config_infer_depth_tracking_da2_vits_294x518_b3_i1.ini
)

if rg -n "/home/mayor" "${static_prep_paths[@]}" -g '!**/archive/**' >/tmp/ds9_local_path_refs.txt; then
  cat /tmp/ds9_local_path_refs.txt >&2
  fail "Active DS9 prep configs contain machine-local /home/mayor paths"
fi

if rg -n "/opt/nvidia/deepstream/deepstream/lib" "${static_prep_paths[@]}" -g '!**/archive/**' >/tmp/ds9_active_symlink_refs.txt; then
  cat /tmp/ds9_active_symlink_refs.txt >&2
  fail "Active DS9 prep configs bind to the mutable DeepStream symlink instead of deepstream-9.0"
fi

rg -q "NOESIS_DS9_ALLOW_PYDS_COMPAT" DS9/noesis/pipelines/hooks.py \
  || fail "hooks.py missing raw PyDS compatibility gate"
rg -q "NOESIS_DS9_ALLOW_NATIVE_TENSOR_COMPAT" DS9/noesis/pipelines/hooks.py \
  || fail "hooks.py missing native tensor compatibility gate"
rg -q "NOESIS_DS9_ENABLE_INTRINSICS_USER_META" DS9/noesis/metadata/intrinsics.py DS9/noesis/pipelines/hooks.py \
  || fail "intrinsics user-meta gate missing"
rg -q "NOESIS_DS9_ENABLE_INPROCESS_LATENCY" DS9/noesis/telemetry/latency_metrics.py \
  || fail "in-process latency gate missing"
rg -q "nvds_remove_obj_meta_from_frame" DS9/csrc/nvdsroiexclude/gstnvdsroiexclude.cpp \
  || fail "nvdsroiexclude source does not call official object-meta removal API"

for script in DS9/scripts/*.sh; do
  bash -n "${script}"
done

python3 - <<'PY'
import ast
from pathlib import Path

for raw in (
    "DS9/noesis/pipelines/hooks.py",
    "DS9/noesis/metadata/intrinsics.py",
    "DS9/noesis/telemetry/latency_metrics.py",
    "DS9/noesis/metadata/object_depth.py",
    "DS9/noesis/metadata/pose_features.py",
    "DS9/noesis/metadata/depth_result.py",
):
    path = Path(raw)
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
PY

for mf in DS9/pipelines/nvdsinfer_yolo26_seg/Makefile DS9/pipelines/nvdsinfer_yolo11_seg/Makefile DS9/pipelines/nvdsinfer_yolo_detect/Makefile DS9/pipelines/nvdsinfer_rfdetr/Makefile DS9/pipelines/nvdsinfer_rfdetr_seg/Makefile; do
  rg -q "deepstream-9.0" "${mf}" || fail "${mf} does not default to deepstream-9.0"
  rg -q "check-ds9" "${mf}" || fail "${mf} missing check-ds9 guard"
done

echo "[OK] DS9 static prep checks passed"
