# Noesis DeepStream 9 Port

Status: DS9 runtime ownership is in place. The launch path is
`DS9/noesis/ds9_runtime.py` -> `DS9/noesis/ds9_runtime_core.py`; DS9 executable
code no longer imports or spawns `noesis/ds8_runtime.py`. Core production gates
pass with DeepStream 9 / TensorRT 10.14.1.48, including preflight, engine load,
DS9-native pose metadata, world/BEV parity, WebRTC, RTSP, MapAnything,
floorplan, ReID, zero-copy, REST depth, ROI prune/restore, object-depth bridge,
depth-tensor bridge, ReID native extraction, YOLO11 detect-only, YOLO26
detect-only (`n/s/m/l/x`), YOLO26 segmentation, RF-DETR segmentation, and
RF-DETR detect-only profile startup. Host occupied-camera validation also proves
live-RTSP ReID/BEV/bridge behavior and clean enough bounded shutdown.

This folder is runtime-standalone from the DS8 entrypoint, DS8 engines, and DS8
native extension binaries. It is not yet a hermetic standalone repository:
the launcher still uses parent-repo app data and shared Python modules such as
`config/cameras.yaml`, calibration/geometry helpers, WebSocket server code, and
some smoke clients. Full DS8 option-surface parity is not yet certified where
V3DT is required. The DS8 native pose metadata bridge remains unsafe under DS9
and is not part of the production path.

## What Changed

- DS9-specific runtime entrypoint: `DS9/noesis/ds9_runtime.py`
- DS9-owned runtime implementation: `DS9/noesis/ds9_runtime_core.py`
- DS9-specific pipeline config: `DS9/config/infer.yaml`
- DS9-specific nvinfer/preprocess configs: `DS9/pipelines/`
- DS9 parser build output target: `DS9/pipelines/*/*.so`
- DS9 custom GStreamer plugin output target: `DS9/gst-plugins/`
- DS9 TensorRT plugin output target: `DS9/plugins/`
- DS9 native extension output target: `DS9/native_extensions/`
- DS9 ONNX/engine staging targets: `DS9/models/onnx/`, `DS9/models/engines/`
- Shared runtime materializers now honor:
  - `NOESIS_MODEL_DIR`
  - `NOESIS_ONNX_DIR`
  - `NOESIS_ENGINE_DIR`
  - `NOESIS_PIPELINE_DIR`
  - `NOESIS_BUILD_DIR`
  - `NOESIS_NATIVE_EXT_DIR`
  - `NOESIS_NATIVE_BUILD_SCRIPT_DIR`
  - `NOESIS_GST_PLUGIN_DIR`
  - `NOESIS_RFDETR_TRT_PLUGIN_LIB`

The DS9 launcher sets these variables before importing the DS9-owned Service
Maker runtime core. It also pins `pyservicemaker` to the DS9 system install
under `/usr/local/lib/python3.12/dist-packages` so a stale user-site wheel cannot
shadow the DeepStream 9 binding. It refuses DS8 installs through
`DS9/scripts/ds9_preflight.py`.

## Configuration Decisions

- 2026-06-30: Detection-wake performance work was ported to the DS9 runtime
  without relaxing DS9 compatibility gates. DS9 keeps the raw-`pyds` and native
  tensor compatibility quarantines, but now uses cache-first pose/object-depth
  processing, bounded per-frame telemetry budgets, publish gates, and stage
  timing counters to reduce CPU/GPU spikes when detections appear.
- 2026-06-30: DS9 depth-tensor native builds now compile DS9-owned native
  sources from `DS9/native/`, link the sibling CUDA ROI/stat sampler kernels,
  and stage outputs under `DS9/native_extensions/`. Root DS8 native binaries
  and root native source paths are not used for DS9 rebuilds.
- 2026-06-30: DS9 YOLO26 pose SGIE now targets a DS9-scoped batch-3 asset pair,
  `DS9/models/onnx/yolo26n-pose_b3.onnx` and
  `DS9/models/engines/yolo26n-pose_b3_fp16.engine`. The repository currently
  fails fast until that DS9 ONNX/engine is staged; the root DS8 b3 engine is not
  reused.
- 2026-06-16: Family-room dewarping preserves the full 1920x1080 rectified
  destination frame, including black border regions. The G4 family-room RTSP
  source is 1280x720, but `nvdewarper` `[surface0] width/height` are
  destination-surface dimensions, so `DS9/config/dewarper_g4_instant_charuco_720_to_1080.txt`
  keeps `output-width/output-height` and `[surface0] width/height` at 1920x1080
  while retaining the 1280x720 source K/D values. This matches the DS8
  full-FoV policy and avoids the top-left zoomed/cropped family-room mosaic.

## Runtime Independence Scope

DS9 is independent from the DS8 runtime entrypoint:

- DS9 runs through `DS9/noesis/ds9_runtime.py`.
- DS9 app logic lives in `DS9/noesis/ds9_runtime_core.py`.
- DS9 preflight checks DS9-native extension and parser outputs under `DS9/`.
- DS9 engines and ONNX/model staging live under `DS9/models/`.
- DS9 generated configs live under `DS9/build/`.
- DS9 executable code is guarded against references to `noesis/ds8_runtime.py`
  by `DS9/scripts/run_static_prep_checks.sh`.

DS9 still shares parent-repo application context:

- Camera inventory defaults to `config/cameras.yaml`.
- Some common Python helpers remain imported from the parent repo until a later
  packaging pass vendors or extracts them cleanly.
- A few validation clients still live in root `scripts/` and are used as
  DS9 evidence only when pointed at a DS9 runtime.

Do not treat those shared helpers as DS8 fallbacks. They are shared app code and
data. The remaining packaging goal is to make that boundary explicit or vendor
the required helpers when DS9 becomes its own repository.

## Pose Metadata Path

DS9 does not use `pyds` and does not reuse the unsafe DS8 object-metadata
unwrap path. The supported path is:

1. `noesis/pipelines/hooks.py` reads YOLO26 SGIE tensor metadata from Service
   Maker `tensor_items`.
2. `DS9/native_extensions/noesis_pose_meta_ext*.so` exposes
   `attach_pose_features_with_batch(...)`, allocating `NvDsUserMeta` from the
   active DS9 batch and appending the compact `NOESIS.POSE_FEATURES` JSON
   payload to each object.
3. Telemetry/world hooks consume the same payload contract. A bounded
   latest-real-pose cache covers sparse SGIE tensor emission without synthetic
   keypoints.

## Verified DS9 Requirements

From NVIDIA DS9 docs:

- DeepStream 9 targets Ubuntu 24.04, Python 3.12, CUDA 13.1, TensorRT 10.14.1.48, and driver 590.48.01 or later.
- DeepStream Python bindings are deprecated; `pyservicemaker` is the recommended Python interface.
- DS8 apps are documented as compatible with DS9, but compiled apps must be rebuilt with DS9 (`NVDS_VERSION=9.0` for Makefile-based apps).
- NVIDIA documents a DS8 library-symlink method for old compiled apps. This port does not use that method because it would make DS9 run against DS8 compatibility paths.

## Host Cutover Snapshot

The 2026-06-16 UTC host validation used the DS9-staged Docker-built resources
without rebuilding host engines, parsers, plugins, or native extensions.

Observed host stack:

- Driver `595.71.05` on `NVIDIA GeForce RTX 3060`.
- DeepStream `9.0.0`.
- CUDA runtime `13.1`.
- TensorRT `10.14.1.48`.
- Torch `2.12.0+cu130` with CUDA available.

Host fresh-start passes:

- `python3 DS9/scripts/ds9_preflight.py`.
- `python3 DS9/noesis/ds9_runtime.py --cameras-config config/cameras.yaml`
  loaded all five engines and opened RTSP `:8554`, WS/WebRTC `:6008`, and REST
  `:8080`.
- WebRTC decoded 602 frames from 6945 RTP packets.
- RTSP mosaic decode held until bounded timeout with no decode error.
- Zero-copy stats passed with `max_boundary_p99_ms=1.002647` and zero
  violations.
- REST-backed zero-copy passed with `max_boundary_p99_ms=2.549196`, nine REST
  refresh successes, and zero violations.
- Floorplan RPC passed.
- Native bridge counters moved for depth tensor, object-depth attachment, and
  ReID extraction with zero core-path CPU-copy violations.
- Focused MP4 ReID stable-ID smoke passed on host with a validation-only
  depthless ReID graph using local file sources and
  `DS9/build/infer_reid_mp4.yaml`.

Host caveats:

- The occupied-camera live-RTSP evidence window on 2026-06-16 proved production
  public tracking, ReID stable IDs, BEV/track parity, native bridge counters,
  media output, MapAnything depth RPC, floorplan RPC, zero-copy stats, and live
  shutdown. The RTSP mosaic sustained decode held for 60 seconds; this is not a
  multi-hour soak.
- Runtime logs include expected noise from short-lived smoke clients closing
  WebSocket connections without a close frame.
- Shutdown/native cleanup: stale user-site `pyservicemaker` shadowing caused
  native heap corruption and constructor/destructor crashes; that shadow install
  has been removed on the host. DS9 preflight and launcher prefer the DS9 system
  binding, and script-mode DS9 shutdown exits immediately after Noesis teardown
  to avoid Python GC destructing live Service Maker native objects. Finite MP4
  EOS and MP4 SIGINT shutdown exit `0` with no heap corruption; looping MP4
  sources and the live RTSP graph can still log `Wait thread did not terminate
  cleanly`. Live RTSP SIGINT shutdown exited `0`, closed ports, left no runtime
  process behind, and showed no fatal Python, malloc, double-free, segfault, or
  heap-corruption markers. One `source_2` reconnect warning appeared during
  teardown after EOS was posted.

## Built-In Elements Used

- `nvurisrcbin` / `nvmultiurisrcbin`: RTSP/file ingestion.
- `nvstreammux`: batching into the three-camera batch.
- `nvdspreprocess`: tensor preparation for detector profiles.
- `nvinfer`: PGIE, ReID SGIE, pose SGIE, DAv2 depth SGIE, MapAnything SGIE.
- `nvtracker`: NvDCF / V3DT tracking.
- `nvdsroiexclude`: pre-analytics exclusion pruning.
- `nvdsanalytics`: ROI and occupancy analytics.
- `nvmultistreamtiler`: mosaic composition.
- `nvdsosd`: masks, labels, keypoints, trails.
- `nvrtspoutsinkbin`: RTSP mosaic output consumed by the WebRTC gateway.

Relevant local sample pattern: `docs/deepstream-docs/03_Sample_Applications.md` cites `deepstream-rtsp-in-rtsp-out/` for processed RTSP output and `deepstream-test4/` for `nvdsanalytics` ROI metadata.

## Build Order

Run inside a DeepStream 9 container or host install:

```bash
cd <repo>
export NOESIS_DEEPSTREAM_HOME=/opt/nvidia/deepstream/deepstream-9.0

python3 DS9/scripts/ds9_preflight.py --env-only
bash DS9/scripts/build_gst_plugins.sh
bash DS9/scripts/build_trt_plugins.sh
bash DS9/scripts/build_custom_parsers.sh
bash DS9/scripts/build_native_extensions.sh
python3 DS9/scripts/rebuild_engines.py
```

MapAnything requires a DS9-compatible ONNX export first. The export may use external ONNX tensor sidecar files; keep those files beside `DS9/models/onnx/mapanything_images_294x518_b3.onnx`.

```bash
python3 utils/onnx2trt/export_ma_onnx/export_to_onnx.py \
  --repo external/map-anything \
  --outdir DS9/models/onnx \
  --h 294 --w 518 \
  --skip-ort --skip-simplify --skip-shape-inference
# Copy or rename the exported images-input ONNX to:
# DS9/models/onnx/mapanything_images_294x518_b3.onnx
NOESIS_MAPANYTHING_PYTHON=/path/to/export-venv/bin/python \
NOESIS_MAPANYTHING_GPU_GUARD_MB=7000 \
NOESIS_MAPANYTHING_GUARD_POLL_SECONDS=1 \
DS9/scripts/build_mapanything_guarded.sh
```

Then run full preflight:

```bash
python3 DS9/scripts/ds9_preflight.py
```

## Run

```bash
python3 DS9/noesis/ds9_runtime.py --cameras-config config/cameras.yaml --enable-rest
```

Optional profiles:

```bash
python3 DS9/noesis/ds9_runtime.py --pgie-profile yolo26_seg --size s --disable-rest
python3 DS9/noesis/ds9_runtime.py --pgie-profile yolo26 --size m --disable-rest
python3 DS9/noesis/ds9_runtime.py --pgie-profile yolo11 --disable-rest
python3 DS9/noesis/ds9_runtime.py --pgie-profile rfdetr_seg --size m --disable-rest
python3 DS9/noesis/ds9_runtime.py --pgie-profile rfdetr --size s --disable-rest
```

YOLO11/YOLO26 detector profiles use DS9-staged ONNX sources under
`DS9/models/onnx/` and DS9-built TensorRT engines under `DS9/models/engines/`.
They use the DS9-compatible custom bbox parser at
`DS9/pipelines/nvdsinfer_yolo_detect/libnvdsparsebbox_yolo.so`, which exports
`NvDsInferParseYolo`. `NOESIS_YOLO_DETECT_PARSER_LIB=/path/to/lib.so` can still
override it for investigation.

RF-DETR detect-only uses DS9-staged ONNX sources under `DS9/models/onnx/` and
DS9-built TensorRT engines under `DS9/models/engines/`. Engines exist for
`n`, `s`, and `m`; the latest bounded startup smoke covered the `s` profile.

The latest detector-profile matrix also covered `yolo11`, `yolo26 n/s/m/l/x`,
`yolo26_seg n/s/m`, `rfdetr n/s/m`, and `rfdetr_seg n/s/m` materialization.

## Graph

See the full DS9 pipeline map:

- [DS9/PIPELINE_GRAPH.md](PIPELINE_GRAPH.md) — Mermaid diagram + component table + all hook attachment points (world_observation_stage, MapAnything gating, native DS9 pose/depth bridges, trails/keypoints, etc.).

Quick textual summary (see the detailed doc for the complete graph):

```text
RTSP sources (w/ dewarpers) -> nvurisrcbin -> nvstreammux -> nvdspreprocess -> yolo11_pgie
  -> main_tee
       ├─ analytics_exclude (nvdsroiexclude) -> nvtracker -> reid_osnet -> yolo26_pose
       ├─ depth_tracking_queue -> depth_tracking_fullframe (DAv2) -> fakesink
       └─ mapanything_queue -> mapanything_valve (gated) -> mapanything_fullframe -> fakesink
  -> world_observation_stage (pose features + object depth fusion via DS9 natives)
  -> tracking_telemetry_stage (analytics telemetry)
  -> tiler -> osd (keypoints + trails injected here)
  -> sink_tee -> (fakesinks + RTSP mosaic branch for WebRTC)
```

Hook attachment points and DS9-specific native metadata paths are documented in the full map.

## Config Knobs Touched

- `streammux.batch-size`: stays `3` to match three sources and fixed batch engines.
- `models.*.engine`: moved to `DS9/models/engines/*`; DS8 engines are not used.
- `models.*.config-file-path`: moved to `DS9/pipelines/*`.
- `tracker.ll-lib-file`: points at `/opt/nvidia/deepstream/deepstream-9.0/lib/libnvds_nvmultiobjecttracker.so`.
- `mosaic_output.*`: unchanged behavior, RTSP to WebRTC remains canonical.
- `analytics.exclude.config-file`: still uses shared ROI config because ROI geometry is app data, not SDK-version-specific.
- `bev.frame`: `backend_world_m`.
- `NOESIS_DEPTH_RPC_TIMEOUT_SECONDS`: WebSocket-side MapAnything depth RPC
  provider timeout, default `12.0` seconds.
- `NOESIS_DEPTH_RPC_ENABLE_SECONDS`: on-demand MapAnything burst length,
  default `4` seconds.

## Acceptance Criteria

- `python3 DS9/scripts/ds9_preflight.py` passes.
- Preflight reports `pyservicemaker` from `/usr/local/lib/python3.12/dist-packages`,
  not the user-site package.
- `python3 DS9/noesis/ds9_runtime.py --enable-rest` starts and reaches the Service Maker wait loop.
- `python3 DS9/scripts/ma_depth_rpc_smoke_test.py --no-spawn` passes against DS9 runtime.
- `python3 DS9/scripts/zero_copy_stats_smoke_test.py --no-spawn` passes against DS9 runtime.
- `python3 DS9/scripts/zero_copy_smoke_test.py --no-spawn` passes against DS9 runtime with REST enabled.
- `python3 scripts/floorplan_rpc_smoke_test.py --no-spawn` passes against DS9 runtime.
- `python3 DS9/scripts/reid_stable_id_smoke_test.py --no-spawn` observes stable IDs on tracking telemetry.
- `NOESIS_MOSAIC_RTSP_ENABLED=1 NOESIS_MOSAIC_WEBRTC_ENABLED=1 python3 scripts/webrtc_gateway_smoke_test.py --ws ws://127.0.0.1:6008 --duration 5 --pt 103` returns decoded frames.
- `python3 DS9/scripts/ds9_live_validation_runner.py` is the preferred
  occupied-camera host evidence bundle for live-RTSP ReID, BEV/track, bridge,
  RTSP/WebRTC, depth/floorplan, zero-copy, and shutdown checks. It writes
  `summary.json`, `summary.md`, `runtime.log`, and per-gate logs under
  `DS9/build/live_validation/<timestamp>/`.
- Focused ROI, V3DT, alternate-profile, and bridge-specific smokes pass where
  those DS8 option paths are required.

Bench report must include command, resolution `1920x1080`, batch `3`, device, fps, latency_ms, and PASS/FAIL against the selected target.

## Current Validation

DS9 Docker validation currently passes these gates:

- `DS9/models/onnx/mapanything_images_294x518_b3.onnx` exists and references 3.43 GiB of ONNX external tensor sidecars.
- `DS9/models/engines/mapanything_images_294x518_b3_fp16.plan` was built in the DS9 container with guarded TensorRT BF16 settings. The filename remains `fp16` for config compatibility.
- `trtexec --loadEngine=DS9/models/engines/mapanything_images_294x518_b3_fp16.plan --skipInference` succeeds in the DS9 container.
- `python3 DS9/scripts/ds9_preflight.py` succeeds in the DS9 container after installing `PyYAML` into the ephemeral container environment.
- DS9 runtime loads all five TensorRT engines: YOLO11 PGIE, ReID, YOLO26 pose, DAv2 depth, and MapAnything.
- RTSP mosaic starts at `rtsp://localhost:8554/mosaic`.
- Live pose activation passed: `pose_present_true=25`, `world_valid_true=5137`, and `backend_world_m=5137` in a 90-second sample.
- BEV/track parity passed with `track_world_valid=2769`, `bev_world_frame_backend_world_m=1632`, `comparisons=1887`, and `p95_err_m=0.0`.
- `DS9/scripts/ma_depth_rpc_smoke_test.py --no-spawn --camera family-room` passed cache-first plus fresh MapAnything depth RPC.
- `scripts/floorplan_rpc_smoke_test.py --no-spawn --max-age-sec 1200` passed.
- `NOESIS_REID_ENABLED=1 DS9/scripts/reid_stable_id_smoke_test.py --no-spawn --duration 20` passed for `kitchen:1`.
- `scripts/webrtc_gateway_smoke_test.py --ws ws://127.0.0.1:6008 --duration 6 --pt 103 --min-rtp 10 --min-decoded 1` passed with `rtp_packets=2287` and `decoded_frames=184` after installing/reinstalling `gstreamer1.0-nice`, `gstreamer1.0-libav`, and codec runtime libraries in the DS9 container.
- RTSP mosaic decode passed with `rtph264depay ! h264parse ! avdec_h264 ! fakesink`.
- DS8-vs-DS9 config parity check passed for model coverage (`pgie`, `reid`, `pose`, `depth_tracking`, `mapanything`), shared top-level stages, matching batch size, and DS9-scoped model config/engine paths.
- `DS9/scripts/zero_copy_stats_smoke_test.py --no-spawn` passed with
  `samples=43`, `max_zero_copy_violations=0`,
  `max_boundary_p99_ms=1.034443`, `ws_depth_requests=43`, and
  `ws_depth_responses=5`.
- `DS9/scripts/zero_copy_smoke_test.py --no-spawn` passed against a REST-enabled
  DS9 runtime with `samples=62`, `max_zero_copy_violations=0`,
  `max_boundary_p99_ms=1.146929`, `rest_refresh_attempts=8`, and
  `rest_refresh_success=8`.
- ROI unit/API tests passed. Focused MP4 ROI hot-restore passed on host:
  full-frame exclusion pruned active tracks from `2` to `0` and same-runtime
  restore recovered `4` active tracks after wiring `NOESIS_ANALYTICS_EXCLUDE_CONFIG`
  into the initial `analytics_exclude` element config.
- YOLO26 segmentation profile `--pgie-profile yolo26_seg --size s` passed
  bounded startup with DS9 engine/config paths and live pose attachments.
- RF-DETR segmentation profile `--pgie-profile rfdetr_seg --size m` passed
  bounded startup after fixing generated DS9 parser/label paths.
- RF-DETR detect-only profile `--pgie-profile rfdetr --size s` passed bounded
  startup after staging DS9 ONNX assets and building
  `DS9/models/engines/rfdetr_s_512_b3_fp16.engine`; the runtime stayed live
  until `timeout` returned `RUNTIME_RC=124`.
- `DS9/scripts/ds9_bridge_contract_smoke_test.py --ws ws://127.0.0.1:6008 --duration 75 --require-embedding-track`
  passed for object-depth, depth-tensor, and ReID native extraction bridges:
  `depth_tracking_device_frames_total=1533`,
  `object_depth_gpu_roi_copies_total=2246`, `object_depth_attach_total=2612`,
  `object_depth_status_total.ok=2246`, `tensor_host_copies_total.reid=354`,
  `depth_ok_tracks=2248`, `embedding_tracks=2609`, and
  `core_path.cpu_copy_violation.total=0`.
- Host shutdown/native cleanup smoke passed with MP4 inputs:
  `DS9/noesis/ds9_runtime.py` reported `pyservicemaker` from
  `/usr/local/lib/python3.12/dist-packages`, received SIGINT, posted EOS, logged
  `Wait thread did not terminate cleanly`, and exited `0` with no
  `Fatal Python error`, segmentation fault, malloc, double-free, or heap
  corruption markers. Finite-source MP4 EOS with `streammux.live-source=0` also
  exits `0`.
- Occupied-camera live-RTSP host validation passed on 2026-06-16:
  - BEV/track parity:
    `track_total=11435`, `track_world_valid=11426`,
    `bev_world_frame_backend_world_m=6980`, `comparisons=11065`,
    `p95_err_m=0.0`, and lagged p95 `0.11610091475856413`.
  - ReID stable-ID smoke: `family-room:16`.
  - Bridge smoke with `--require-embedding-track`:
    `tracking_messages=5373`, `tracks_seen=11436`,
    `depth_ok_tracks=11066`, `embedding_tracks=11398`,
    `depth_tracking_device_frames_total=4833`,
    `object_depth_gpu_roi_copies_total=10980`,
    `object_depth_attach_total=11350`,
    `object_depth_status_total.ok=10980`,
    `tensor_host_copies_total.reid=971`,
    `core_path.cpu_copy_violation.total=0`, and `pipeline_errors=[]`.
  - WebRTC smoke: `rtp_packets=1438`, `decoded_frames=92`.
  - RTSP mosaic decode held for 25 seconds and sustained decode held for
    60 seconds.
  - Zero-copy stats:
    `samples=7`, `max_zero_copy_violations=0`,
    `max_boundary_p99_ms=1.123324`, `ws_depth_requests=38`, and
    `ws_depth_responses=4`.
  - MapAnything depth RPC passed cache-first plus fresh for `family-room`.
  - Floorplan RPC passed for `kitchen` and then `family-room`; the first
    `family-room` floorplan attempt timed out before succeeding on retry.
  - Live RTSP SIGINT shutdown exited `0`, closed ports `6008`, `8080`, and
    `8554`, and left no DS9 runtime process. The shutdown tail logged
    `Wait thread did not terminate cleanly` and one `source_2` reconnect warning
    after EOS, with no native heap/fatal markers.

Still pending before full option-surface parity:

- DS9-native V3DT pipeline/camera/tracker staging and metadata extraction
  smoke if DS9 is expected to run a V3DT profile.
- V3DT bridge behavior smoke after DS9-native V3DT staging exists.
- Longer host RTSP soak evidence, if production acceptance requires more than
  the occupied-camera validation window captured here.
- Packaging cleanup if DS9 needs to become a hermetic repository instead of a
  DS9-owned runtime folder inside the Noesis monorepo.

Historical parity blocker:

- Enabling the DS8 native pose metadata bridge under DS9 caused a segmentation
  fault at the first analytics batch. DS9 keeps that bridge disabled unless
  explicitly forced with `NOESIS_POSE_NATIVE_EXTRACT_ENABLED=1` for debugging.
  The production path is the DS9-native Service Maker/native-extension path
  described above.
