# DS9.1 native rebuild and smoke gates

Build and test only the artifact family whose inputs changed. All commands run
on the pinned native host stack; Docker is not an accepted backend.

## Common prerequisites

```bash
NATIVE_ENV_FILE="$(
  systemctl --user show noesis-appliance.service \
    -p EnvironmentFiles --value --no-pager |
    awk '$1 ~ /\/native\.env$/ { print $1; exit }'
)"
test -n "$NATIVE_ENV_FILE" && test -f "$NATIVE_ENV_FILE"
set -a
. "$NATIVE_ENV_FILE"
set +a

PYTHON="${NOESIS_DS91_NATIVE_ROOT}/venv/bin/python"
```

The host must resolve DeepStream 9.1, CUDA 13.2, TensorRT 10.16.0.72, Python
3.12, and driver 595.58.03 or newer. Run the supervisor `check` once when those
inputs changed; do not repeat it after an unrelated test edit.

## Native extensions

Relevant sources live in `DS9/native/`. Use the specific build helper where one
exists; use `DS9/scripts/build_all_native_ds9.sh` only when shared headers or
the toolchain invalidate all extensions.

Minimum gate:

1. Compile the affected extension.
2. Import it from the native venv and confirm its origin is the native DS9.1
   path, not root legacy output or a container layer.
3. Run its focused tensor/metadata test.
4. Exercise the direct hook/consumer once if behavior changed.

## Inference parsers and TensorRT plugin

Rebuild only the parser/plugin whose source, headers, ABI, model output, or
toolchain changed.

Minimum gate:

1. Compile against the installed DS9.1 headers and CUDA 13.2.
2. Use `ldd` to confirm required libraries resolve from the host stack.
3. Load the shared object and run the relevant parser output-shape test.
4. Deserialize one engine that consumes it.

## GStreamer plugins

The canonical custom plugins are ROI exclusion, force-IDR, and orderly EOS.

Minimum gate:

1. Rebuild the affected plugin.
2. Inspect it with `gst-inspect-1.0` using the native registry/path.
3. Run the specific reload, late-viewer, or shutdown test that exercises it.

Do not validate all three for a one-plugin edit.

## TensorRT engines

Use `DS9/scripts/run_canonical_engine_maintenance_host.sh`. It refuses Docker
authority and records native host compatibility/provenance.

Canonical runtime engines:

- `engine.yolo26_detect_m`
- `engine.reid_swin_tiny`
- `engine.pose_yolo26`
- `engine.depth_tracking_dav2`
- `engine.mapanything`

Minimum gate for a changed engine:

1. Build/finalize it once from the reviewed model/config inputs.
2. `trtexec --loadEngine` with the expected input profile.
3. Run its output/parser/profile contract test.
4. Exercise its direct runtime consumer on one representative input.
5. Refresh dependent hardened fingerprints only when exact content changed;
   never weaken their checks.

V3DT/MV3DT, segmentation, Wholebody49, RF-DETR, and alternate manual-depth
engines are not canonical merely because they are realized. Build them only for
an explicitly scoped experiment. Building MV3DT assets does not enable MV3DT.

## Pipeline smoke

After an artifact that affects the selected lane changes, run one short live or
recorded smoke and inspect only the affected outputs. Examples include
detections for a detector, StableID for ReID, keypoints for pose, world depth
for DAv2, or one manual depth/floorplan request for MapAnything.

Add WebRTC decode only if the graph/output path changed. Add performance
measurement only if the hot path or runtime environment changed. Do not rerun a
complete parity suite or create a release candidate.
