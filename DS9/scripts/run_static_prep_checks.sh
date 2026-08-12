#!/usr/bin/env bash
set -euo pipefail

DS9_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${DS9_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

# Own the import boundary instead of relying on the invoking shell. DS9's
# package must win for successor-runtime modules while shared noesis_core
# contracts resolve from the repository root.
export PYTHONPATH="${DS9_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

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
require_file "DS9/gst-plugins/build_noesisforceidr.sh"
require_file "DS9/gst-plugins/noesisforceidr/gstnoesisforceidr.cpp"
require_file "DS9/gst-plugins/noesisforceidr/CMakeLists.txt"
require_file "DS9/gst-plugins/build_noesiseos.sh"
require_file "DS9/gst-plugins/noesiseos/gstnoesiseos.cpp"
require_file "DS9/gst-plugins/noesiseos/CMakeLists.txt"
require_file "DS9/noesis/ds9_runtime.py"
require_file "DS9/noesis/ds9_runtime_core.py"
require_file "DS9/noesis/runtime_config.py"
require_file "noesis/mosaic_glib_context.py"
require_file "noesis/mosaic_h264_bridge.py"
require_file "noesis/mosaic_webrtc_gateway.py"
require_file "DS9/noesis/telemetry/world_contract_adapter.py"
require_file "noesis_core/runtime_world.py"
require_file "noesis_core/world_service.py"
require_file "noesis/server/health_api.py"
require_file "noesis/server/scene_api.py"
require_file "DS9/docs/runtime_ownership.yaml"
require_file "DS9/docs/runtime_container_boundary.md"
require_file "DS9/asset_manifest.yaml"
require_file "DS9/docs/asset_manifest.schema.json"
require_file "DS9/scripts/validate_runtime_ownership.py"
require_file "DS9/scripts/validate_asset_manifest.py"
require_file "DS9/scripts/secondary_docker.sh"
require_file "DS9/scripts/build_secondary_dev_image.sh"
require_file "DS9/scripts/build_secondary_runtime_image.sh"
require_file "DS9/scripts/stage_canonical_sources.py"
require_file "DS9/scripts/run_canonical_engine_maintenance.sh"
require_file "DS9/scripts/build_v3dt_tracker_engine.py"
require_file "DS9/scripts/validate_runtime_secrets_container.sh"
require_file "DS9/scripts/run_canonical_runtime_container.py"
require_file "DS9/docker/Dockerfile"
require_file "DS9/docker/Dockerfile.runtime"
require_file "DS9/docker/requirements.lock.txt"
require_file "DS9/csrc/nvdsroiexclude/gstnvdsroiexclude.cpp"
require_file "DS9/csrc/nvdsroiexclude/CMakeLists.txt"
require_file "DS9/config/infer.yaml"
require_file "DS9/config/infer_v3dt.yaml"
require_file "DS9/config/cameras_v3dt.yaml"
require_file "DS9/config/v3dt/nvtracker_v3dt.yaml"
require_file "DS9/config/v3dt/caminfo_baseline/camInfo_living-room.yml"
require_file "DS9/config/v3dt/caminfo_baseline/camInfo_kitchen.yml"
require_file "DS9/config/v3dt/caminfo_baseline/camInfo_family-room.yml"
require_file "DS9/config/config_nvdsanalytics_post.ini"
require_file "DS9/config/nvtracker.yaml"
require_file "DS9/config/depth_registration.json"
require_file "DS9/config/dewarper_g3_instant_charuco_1080.txt"
require_file "DS9/pipelines/config_preproc.ini"
require_file "DS9/pipelines/config_infer_primary_yolo11_seg.ini"
require_file "DS9/pipelines/config_infer_primary_yolo26_m.ini"
require_file "DS9/pipelines/config_infer_primary_yolo26_seg.template.ini"
require_file "DS9/pipelines/config_infer_primary_rfdetr_seg.ini"
require_file "DS9/pipelines/config_infer_primary_deimv2_wholebody49_masks.template.ini"
require_file "DS9/pipelines/config_infer_primary_deimv2_wholebody49_boxes.template.ini"
require_file "DS9/pipelines/config_infer_secondary_yolo26_pose.ini"
require_file "DS9/pipelines/config_infer_secondary_reid_swin.ini"
require_file "DS9/pipelines/config_infer_secondary_depth_tracking_da2.ini"
require_file "DS9/models/onnx/reid_swin_tiny_market1501_aicity156_featuredim256.onnx.provenance.json"
require_file "DS9/models/onnx/bodypose3dnet_accuracy.onnx.provenance.json"
require_file "DS9/models/tracker_reid/resnet50_market1501.etlt.provenance.json"
require_file "DS9/pipelines/config_infer_secondary_mapanything.ini"

if [[ -e "DS9/noesis/ds8_runtime.py" ]]; then
  fail "Stale DS8 runtime copy exists under DS9/noesis"
fi

rg -q "from noesis.ds9_runtime_core import main" DS9/noesis/ds9_runtime.py \
  || fail "DS9 launcher does not delegate to noesis.ds9_runtime_core"

if rg --no-ignore -n "from noesis\\.ds8_(runtime|preflight)|import noesis\\.ds8_(runtime|preflight)|noesis/ds8_runtime\\.py" \
  DS9/noesis/ds9_runtime.py DS9/noesis/ds9_runtime_core.py DS9/scripts -g '*.py' \
  >/tmp/ds9_old_runtime_refs.txt; then
  cat /tmp/ds9_old_runtime_refs.txt >&2
  fail "DS9 executable code still references a DS8 runtime/preflight helper"
fi

if [[ -d "DS9/pipelines/nvdsinfer_yolo26_pose" ]]; then
  fail "No-op YOLO26 pose parser is still in active DS9 parser path"
fi
require_dir "DS9/archive/nvdsinfer_yolo26_pose"

if rg -n "custom-lib-path|parse-bbox" DS9/pipelines/config_infer_secondary_yolo26_pose.ini >/tmp/ds9_pose_parser_refs.txt; then
  cat /tmp/ds9_pose_parser_refs.txt >&2
  fail "Pose SGIE config still references a parser/custom lib"
fi

if rg -ni "osnet" \
  DS9/config/infer.yaml \
  DS9/pipelines/config_infer_secondary_reid_swin.ini \
  DS9/scripts/rebuild_engines.py \
  DS9/scripts/stage_canonical_sources.py \
  DS9/scripts/run_canonical_engine_maintenance.sh \
  DS9/asset_manifest.yaml \
  >/tmp/ds9_active_reid_osnet_refs.txt; then
  cat /tmp/ds9_active_reid_osnet_refs.txt >&2
  fail "Active DS9 ReID surfaces still expose the OSNet substitute"
fi

static_prep_paths=(
  DS9/config
  DS9/pipelines
)

if rg -n "/home/mayor" "${static_prep_paths[@]}" -g '!**/archive/**' >/tmp/ds9_local_path_refs.txt; then
  cat /tmp/ds9_local_path_refs.txt >&2
  fail "Active DS9 prep configs contain machine-local /home/mayor paths"
fi

if rg -n "/opt/nvidia/deepstream/deepstream/lib" "${static_prep_paths[@]}" -g '!**/archive/**' >/tmp/ds9_active_symlink_refs.txt; then
  cat /tmp/ds9_active_symlink_refs.txt >&2
  fail "Active DS9 prep configs bind to the mutable DeepStream symlink instead of deepstream-9.1"
fi

if rg -n "deepstream-9\\.0|cuda-13\\.1" "${static_prep_paths[@]}" -g '!**/archive/**' >/tmp/ds9_stale_toolchain_refs.txt; then
  cat /tmp/ds9_stale_toolchain_refs.txt >&2
  fail "Active DS9 prep configs still reference the retired DeepStream 9.0/CUDA 13.1 toolchain"
fi

rg -q "NOESIS_DS9_ALLOW_PYDS_COMPAT" DS9/noesis/pipelines/hooks.py \
  || fail "hooks.py missing raw PyDS compatibility gate"
if rg -q "NOESIS_DS9_ALLOW_NATIVE_TENSOR_COMPAT" DS9/noesis/pipelines/hooks.py; then
  fail "hooks.py still exposes the retired MapAnything native tensor fallback gate"
fi
rg -q "capture_mapanything_tensor_layers_exact" \
  DS9/noesis/pipelines/hooks.py DS9/native/noesis_depth_tracking_tensor_ext.cpp \
  || fail "DS9 MapAnything exact native tensor capture is missing"
if rg -q '"capture_tensor_layers",' DS9/native/noesis_depth_tracking_tensor_ext.cpp; then
  fail "DS9 native extension still exports the generic MapAnything tensor capture"
fi
rg -q "NOESIS_DS9_ENABLE_INTRINSICS_USER_META" DS9/noesis/metadata/intrinsics.py DS9/noesis/pipelines/hooks.py \
  || fail "intrinsics user-meta gate missing"
rg -q "NOESIS_DS9_ENABLE_INPROCESS_LATENCY" DS9/noesis/telemetry/latency_metrics.py \
  || fail "in-process latency gate missing"
rg -q "nvds_remove_obj_meta_from_frame" DS9/csrc/nvdsroiexclude/gstnvdsroiexclude.cpp \
  || fail "nvdsroiexclude source does not call official object-meta removal API"
rg -q "gst_nvevent_enc_force_idr" DS9/gst-plugins/noesisforceidr/gstnoesisforceidr.cpp \
  || fail "noesisforceidr source does not call NVIDIA's official force-IDR event API"
rg -q "gst_pad_push_event" DS9/gst-plugins/noesisforceidr/gstnoesisforceidr.cpp \
  || fail "noesisforceidr source does not push the force-IDR event downstream"
rg -q "gst_event_new_eos" DS9/gst-plugins/noesiseos/gstnoesiseos.cpp \
  || fail "noesiseos source does not create a standard GStreamer EOS event"
rg -q "gst_pad_push_event" DS9/gst-plugins/noesiseos/gstnoesiseos.cpp \
  || fail "noesiseos source does not push EOS downstream"
rg -q "g_atomic_int_set.*eos_accepted.*TRUE" DS9/gst-plugins/noesiseos/gstnoesiseos.cpp \
  || fail "noesiseos source does not enter terminal state before serializing EOS"
if rg -q "GST_PAD_STREAM_LOCK" DS9/gst-plugins/noesiseos/gstnoesiseos.cpp; then
  fail "noesiseos must not take a sink-pad stream lock from Service Maker Node.set"
fi
rg -q "g_thread_try_new" DS9/gst-plugins/noesiseos/gstnoesiseos.cpp \
  || fail "noesiseos property setter does not dispatch EOS asynchronously"
rg -Fq "g_object_ref(G_OBJECT(self))" DS9/gst-plugins/noesiseos/gstnoesiseos.cpp \
  || fail "noesiseos worker does not hold an element self-reference"
rg -q "GST_BASE_TRANSFORM_FLOW_DROPPED" DS9/gst-plugins/noesiseos/gstnoesiseos.cpp \
  || fail "noesiseos source does not drop post-EOS input buffers"
rg -q "transform_ip_on_passthrough = TRUE" DS9/gst-plugins/noesiseos/gstnoesiseos.cpp \
  || fail "noesiseos post-EOS drop is not active in passthrough mode"

for script in DS9/scripts/*.sh; do
  bash -n "${script}"
done

python3 DS9/scripts/validate_runtime_ownership.py
python3 DS9/scripts/validate_asset_manifest.py
python3 - <<'PY'
from pathlib import Path

from noesis.v3dt_assets import validate_v3dt_assets

validate_v3dt_assets(
    Path("DS9/config/infer_v3dt.yaml"),
    require_engines=False,
    require_sources=False,
)
PY

rg -Fq 'runtime="ds9"' DS9/noesis/ds9_runtime_core.py \
  || fail "DS9 canonical world service is not stamped with runtime=ds9"
rg -Fq 'health_monitor=capability_monitor' DS9/noesis/ds9_runtime_core.py \
  || fail "DS9 canonical capability monitor is not bound to tracking telemetry"
rg -Fq 'app.include_router(health_api.router)' DS9/noesis/ds9_runtime_core.py \
  || fail "DS9 REST app is missing the canonical capability-health router"
rg -Fq 'app.include_router(scene_api.router)' DS9/noesis/ds9_runtime_core.py \
  || fail "DS9 REST app is missing the shared scene-release router"

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
    "DS9/noesis/runtime_config.py",
    "DS9/noesis/v3dt_assets.py",
    "DS9/noesis/telemetry/publishers.py",
    "DS9/noesis/telemetry/world_contract_adapter.py",
    "DS9/scripts/validate_runtime_ownership.py",
    "DS9/scripts/validate_asset_manifest.py",
    "DS9/scripts/stage_canonical_sources.py",
    "DS9/scripts/build_v3dt_tracker_engine.py",
    "DS9/scripts/sv3dt_meta_smoke_test.py",
    "DS9/scripts/run_canonical_runtime_container.py",
):
    path = Path(raw)
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
PY

for mf in DS9/pipelines/nvdsinfer_yolo26_seg/Makefile DS9/pipelines/nvdsinfer_yolo11_seg/Makefile DS9/pipelines/nvdsinfer_yolo_detect/Makefile DS9/pipelines/nvdsinfer_rfdetr/Makefile DS9/pipelines/nvdsinfer_rfdetr_seg/Makefile DS9/pipelines/nvdsinfer_rfdetr_keypoint/Makefile DS9/pipelines/nvdsinfer_deimv2_wholebody49/Makefile; do
  rg -q "deepstream-9.1" "${mf}" || fail "${mf} does not default to deepstream-9.1"
  rg -q "cuda-13.2" "${mf}" || fail "${mf} does not default to cuda-13.2"
  rg -q "NVDS_VERSION_MINOR.*1" "${mf}" || fail "${mf} does not reject pre-9.1 headers"
  rg -q "check-ds9" "${mf}" || fail "${mf} missing check-ds9 guard"
done

echo "[OK] DS9 static prep checks passed"
