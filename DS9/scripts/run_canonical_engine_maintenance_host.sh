#!/usr/bin/env bash
# Native-host DS9.1 engine and plugin maintenance. Does not use Docker.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DS9_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${DS9_ROOT}/.." && pwd)"
# shellcheck source=ds9_build_env.sh
source "${SCRIPT_DIR}/ds9_build_env.sh"

REQUIRED_DRIVER_VERSION="595.58.03"
CANONICAL_ENGINES=(yolo26_m reid_swin yolo26_pose_n depth_anything_v2_tracking mapanything)
declare -Ar ENGINE_FILES=(
  [yolo26_m]="yolo26m_b3_fp16.engine"
  [reid_swin]="reid_swin_tiny_aicity156_dyn_b16_fp16.engine"
  [yolo26_pose_n]="yolo26n-pose_b3_fp16.engine"
  [depth_anything_v2_tracking]="depth_anything_v2_metric_hypersim_vits_294x518_b3_fp16.engine"
  [mapanything]="mapanything_images_294x518_b3_fp32.plan"
)
declare -Ar ENGINE_IDS=(
  [yolo26_m]="engine.yolo26_detect_m"
  [reid_swin]="engine.reid_swin_tiny"
  [yolo26_pose_n]="engine.pose_yolo26"
  [depth_anything_v2_tracking]="engine.depth_tracking_dav2"
  [mapanything]="engine.mapanything"
)

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  NOESIS_DS9_ARTIFACT_ROOT=<absolute-artifact-root> \
    DS9/scripts/run_canonical_engine_maintenance_host.sh [--direct-load]

  NOESIS_DS9_ARTIFACT_ROOT=... \
    DS9/scripts/run_canonical_engine_maintenance_host.sh --only <engine> [--plan]

  NOESIS_DS9_ARTIFACT_ROOT=... \
    DS9/scripts/run_canonical_engine_maintenance_host.sh --all [--plan]

Native-host backend only. Docker is not used. MV3DT/AMC engines are not built.
EOF
}

ONLY=""
ALL=0
PLAN=0
DIRECT_LOAD=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --direct-load) DIRECT_LOAD=1; shift ;;
    --only)
      [[ $# -ge 2 ]] || fail "--only requires an engine name"
      ONLY="$2"
      DIRECT_LOAD=0
      shift 2
      ;;
    --all) ALL=1; DIRECT_LOAD=0; shift ;;
    --plan) PLAN=1; shift ;;
    --v3dt) fail "MV3DT/AMC remain disabled; native maintenance will not build that matrix" ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unsupported argument: $1" ;;
  esac
done

[[ -n "${NOESIS_DS9_ARTIFACT_ROOT:-}" ]] || fail "NOESIS_DS9_ARTIFACT_ROOT must be set"
[[ -z "${NOESIS_DS9_DOCKER_ROOT:-}" ]] || fail "native maintenance refuses NOESIS_DS9_DOCKER_ROOT"
ARTIFACT_ROOT="$(readlink -f -- "${NOESIS_DS9_ARTIFACT_ROOT}")"
[[ -d "${ARTIFACT_ROOT}" ]] || fail "artifact root missing: ${ARTIFACT_ROOT}"
ENGINE_DIR="${ARTIFACT_ROOT}/models/engines"
REALIZATION="${ARTIFACT_ROOT}/asset_realization.json"

DS_HOME="$(ds9_require_deepstream_home)"
CUDA_HOME="$(ds9_require_cuda_home)"
export NOESIS_DEEPSTREAM_HOME="${DS_HOME}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${DS_HOME}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export GST_PLUGIN_PATH="${DS_HOME}/lib/gst-plugins${GST_PLUGIN_PATH:+:${GST_PLUGIN_PATH}}"

HOST_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d '[:space:]')"
LOWEST="$(printf '%s\n' "${REQUIRED_DRIVER_VERSION}" "${HOST_DRIVER}" | sort -V | head -n1)"
[[ "${LOWEST}" == "${REQUIRED_DRIVER_VERSION}" ]] \
  || fail "host NVIDIA driver ${HOST_DRIVER} is below ${REQUIRED_DRIVER_VERSION}"

command -v trtexec >/dev/null || fail "trtexec not found"
trtexec --help >/dev/null 2>&1 || true
NVCC_VERSION="$("${CUDA_HOME}/bin/nvcc" --version | awk '/release/{print}')"
[[ "${NVCC_VERSION}" == *release\ 13.2* ]] || fail "nvcc is not CUDA 13.2: ${NVCC_VERSION}"

PYTHON_BIN="${NOESIS_DS91_NATIVE_ROOT:+${NOESIS_DS91_NATIVE_ROOT}/venv/bin/python}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
[[ -x "${PYTHON_BIN}" ]] || fail "python3 not found"
PYTHON_ABI="$("${PYTHON_BIN}" -c 'import sys; print("cp%d%d" % (sys.version_info[0], sys.version_info[1]))')"
COMPILER="$("${CUDA_HOME}/bin/nvcc" --version | tr '\n' ' ')"
export NOESIS_DS9_MAINT_BACKEND="native_host"
export NOESIS_DS9_MAINT_COMPILER="${COMPILER}"
export NOESIS_DS9_MAINT_PYTHON_ABI="${PYTHON_ABI}"
export NOESIS_DS9_MAINT_IMAGE="native_host"
export NOESIS_DS9_MAINT_IMAGE_ID="native_host"
export NOESIS_DS9_MAINT_BASE_DIGEST="native_host"
export NOESIS_DS9_MAINT_TRT_VERSION="10.16.0.72"
export NOESIS_DS9_MAINT_CUDA_VERSION="13.2"

direct_load() {
  echo "[INFO] native direct-load of selected DS9.1 runtime components"
  "${PYTHON_BIN}" - "${DS9_ROOT}" "${REPO_ROOT}" "${ARTIFACT_ROOT}" <<'PY'
import ctypes, hashlib, json, os, subprocess, sys
from pathlib import Path

ds9_root = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
artifact_root = Path(sys.argv[3])
os.environ.setdefault("NOESIS_NATIVE_EXT_DIR", str(ds9_root / "native_extensions"))
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(ds9_root))

from noesis.runtime_paths import (
    configure_ds9_runtime_import_paths,
    load_ds9_native_extensions,
    service_maker_system_site,
)

native_dir = configure_ds9_runtime_import_paths(
    ds9_root=ds9_root,
    repo_root=repo_root,
    system_site=service_maker_system_site(),
)
origins = load_ds9_native_extensions(native_dir)
print(f"[OK] native extensions: {len(origins)}")

parsers = [
    ds9_root / "pipelines/nvdsinfer_yolo_detect/libnvdsparsebbox_yolo.so",
    ds9_root / "pipelines/nvdsinfer_yolo26_seg/libnvdsinfer_yolo26_seg.so",
    ds9_root / "pipelines/nvdsinfer_deimv2_wholebody49/libnvdsinfer_deimv2_wholebody49.so",
]
for parser in parsers:
    if not parser.is_file():
        raise SystemExit(f"[FAIL] parser missing: {parser}")
    ctypes.CDLL(str(parser), mode=ctypes.RTLD_GLOBAL)
    print(f"[OK] parser: {parser.name}")

plugin = ds9_root / "plugins/libnvdsinfer_custom_impl_Yolo_seg.so"
if not plugin.is_file():
    raise SystemExit(f"[FAIL] TensorRT plugin missing: {plugin}")
ctypes.CDLL(str(plugin), mode=ctypes.RTLD_GLOBAL)
print(f"[OK] TensorRT plugin: {plugin.name}")

gst_env = os.environ.copy()
gst_env["GST_PLUGIN_PATH"] = os.pathsep.join(
    [str(ds9_root / "gst-plugins"), gst_env.get("GST_PLUGIN_PATH", "")]
)
for element, soname in (
    ("nvdsroiexclude", "libgstnvdsroiexclude.so"),
    ("noesisforceidr", "libgstnoesisforceidr.so"),
    ("noesiseos", "libgstnoesiseos.so"),
):
    binary = ds9_root / "gst-plugins" / soname
    if not binary.is_file():
        raise SystemExit(f"[FAIL] GStreamer plugin missing: {binary}")
    proc = subprocess.run(["gst-inspect-1.0", element], capture_output=True, text=True, env=gst_env)
    if proc.returncode != 0 or str(binary) not in (proc.stdout or ""):
        raise SystemExit(f"[FAIL] GStreamer factory {element} did not resolve to {binary}")
    print(f"[OK] GStreamer plugin: {element}")

realization_path = artifact_root / "asset_realization.json"
realization = json.loads(realization_path.read_bytes())
required = {
    "engine.yolo26_detect_m": "DS9/models/engines/yolo26m_b3_fp16.engine",
    "engine.reid_swin_tiny": "DS9/models/engines/reid_swin_tiny_aicity156_dyn_b16_fp16.engine",
    "engine.pose_yolo26": "DS9/models/engines/yolo26n-pose_b3_fp16.engine",
    "engine.depth_tracking_dav2": "DS9/models/engines/depth_anything_v2_metric_hypersim_vits_294x518_b3_fp16.engine",
    "engine.mapanything": "DS9/models/engines/mapanything_images_294x518_b3_fp32.plan",
}
for artifact_id, relative in required.items():
    overlay = realization["artifacts"][artifact_id]
    expected = overlay["provenance"]["output_sha256"]
    path = artifact_root / relative.removeprefix("DS9/")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"[FAIL] engine digest mismatch: {artifact_id}")
    print(f"[OK] engine bytes: {artifact_id} sha256={digest}")
print("[OK] native direct-load complete")
PY
}

rebuild_one() {
  local engine="$1"
  [[ -n "${ENGINE_FILES[${engine}]:-}" ]] || fail "unknown engine: ${engine}"
  local extra=()
  [[ "${engine}" == "mapanything" ]] && extra+=(--include-mapanything)
  (( PLAN )) && extra+=(--dry-run)
  if (( ! PLAN )); then
    local owners
    owners="$(nvidia-smi --id=0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true)"
    [[ -z "${owners//[[:space:]]/}" ]] || fail "GPU compute owner(s) present; native rebuild refuses to start: ${owners}"
  fi
  echo "[INFO] native rebuild backend for ${engine}"
  "${PYTHON_BIN}" "${REPO_ROOT}/DS9/scripts/rebuild_engines.py" \
    --only "${engine}" \
    --validate-load \
    --evidence-root "${ARTIFACT_ROOT}/models/engine_maintenance" \
    "${extra[@]}"
}

if (( DIRECT_LOAD )); then
  [[ -z "${ONLY}" && "${ALL}" -eq 0 ]] || fail "direct-load cannot combine with --only/--all"
  direct_load
  exit 0
fi

if [[ -n "${ONLY}" ]]; then
  rebuild_one "${ONLY}"
  exit 0
fi

if (( ALL )); then
  for engine in "${CANONICAL_ENGINES[@]}"; do
    rebuild_one "${engine}"
  done
  exit 0
fi

fail "specify --direct-load, --only, or --all"
