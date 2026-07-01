# DS9 Migration State

Last updated: 2026-06-30

## Goal

Migrate the canonical Noesis DeepStream 8 app to DeepStream 9 without fallbacks.
The DS8 entrypoint remains `noesis/ds8_runtime.py`; the DS9 entrypoint is
`DS9/noesis/ds9_runtime.py`.

`DS9/noesis/ds9_runtime.py` sets DS9-specific environment variables, runs
`DS9/scripts/ds9_preflight.py`, injects `DS9/config/infer.yaml` when no explicit
pipeline config is provided, then calls the DS9-owned runtime implementation in
`DS9/noesis/ds9_runtime_core.py`. This keeps the Service Maker app logic
available under DS9-specific config and assets without importing the root DS8
runtime entrypoint.
The DS9 launcher and preflight prefer the DS9 system `pyservicemaker` binding
under `/usr/local/lib/python3.12/dist-packages` before any user-site wheel.
DS9 still shares parent-repo application helpers and app data such as
`config/cameras.yaml`, calibration/geometry helpers, WebSocket server code, and
some validation clients. That is the current packaging boundary; it is not a DS8
runtime fallback.

## Target Environment

- Docker image validated so far:
  `nvcr.io/nvidia/deepstream:9.0-triton-multiarch`
- DeepStream: 9.0
- TensorRT: 10.14.x, specifically 10.14.1.48 in local DS9 docs
- DS9 SDK root expected by scripts:
  `/opt/nvidia/deepstream/deepstream-9.0`

## Completed Migration Work

- Staged DS9 implementation resources under `DS9/`.
- Added the DS9 launcher at `DS9/noesis/ds9_runtime.py`.
- Added the DS9-owned runtime core at `DS9/noesis/ds9_runtime_core.py`, copied
  from the current Service Maker runtime and retargeted to DS9-scoped defaults.
- Added DS9-specific runtime config at `DS9/config/infer.yaml`.
- Added DS9-specific depth-registration fingerprints at
  `DS9/config/depth_registration.json`.
- Added DS9 runtime Python dependencies at `DS9/requirements-runtime.txt`.
  REST-enabled validation requires both `fastapi` and `uvicorn[standard]`.
- Added DS9 model labels at `DS9/models/coco_labels.txt`.
- Added DS9 preflight script at `DS9/scripts/ds9_preflight.py`.
- Added DS9 build scripts for custom parsers, native extensions, GStreamer
  plugins, TensorRT plugins, TensorRT engine rebuilds, and guarded MapAnything
  plan generation.
- Rebuilt DS9 parser/native/plugin artifacts in DS9-target locations.
- Rebuilt TensorRT engines for DS9/TensorRT 10.14.x.
- Added the DS9 YOLO detect-only parser under
  `DS9/pipelines/nvdsinfer_yolo_detect/` and rebuilt
  `DS9/pipelines/nvdsinfer_yolo_detect/libnvdsparsebbox_yolo.so` for YOLO11 and
  YOLO26 detect-only profiles.
- Built DS9 TensorRT engines for `yolo11` and YOLO26 detect-only sizes
  `n/s/m/l/x`.
- Added guarded MapAnything ONNX/export/build flow. The expected ONNX is
  `DS9/models/onnx/mapanything_images_294x518_b3.onnx`; keep any ONNX external
  tensor sidecars beside it.
- Updated shared path resolution so DS9 can resolve repo-relative paths through
  DS9 environment variables such as `NOESIS_ONNX_DIR`,
  `NOESIS_ENGINE_DIR`, `NOESIS_PIPELINE_DIR`, `NOESIS_BUILD_DIR`,
  `NOESIS_NATIVE_EXT_DIR`, `NOESIS_NATIVE_BUILD_SCRIPT_DIR`,
  `NOESIS_GST_PLUGIN_DIR`, and `NOESIS_RFDETR_TRT_PLUGIN_LIB`.
- Installed WebRTC runtime dependencies in the validation container:
  `gstreamer1.0-nice`, `gstreamer1.0-libav`, and codec libraries needed by the
  smoke client.

## Recent Fixes And Follow-up Results

- Ported the detection-wake performance work from DS8 to DS9 while preserving
  DS9-specific compatibility guards:
  - `DS9/noesis/pipelines/hooks.py` now exposes stage timing counters and
    detection-wake counters for pose, ReID, object-depth, tracking, and BEV work.
  - Pose extraction is cache-first and derives cache age from the configured
    `secondary-reinfer-interval`; stale entries are bounded by bbox shift and
    frame age.
  - Object-depth fusion now uses cache/cadence/budget gates and prefers native
    CUDA ROI/stat samplers before falling back to bounded CPU ROI sampling.
    Detector-only profiles use bbox-band sampling when no segmentation mask is
    available.
  - Analytics telemetry has bounded ReID/pose-anchor work and publish gates for
    tracking and BEV payloads, reducing the work triggered by one or a few
    detections.
  - `DS9/noesis/ds9_runtime_core.py` reports
    `zero_copy_core.stage_timings` in stats payloads and defaults ReID embedding
    refresh to `NOESIS_REID_EMBED_INTERVAL_S=1.0`.
- Reworked DS9 depth-tensor native rebuilds for the CUDA sampler path:
  - `DS9/native/noesis_depth_tracking_tensor_ext.cpp` exposes
    `sample_roi_stats`, `sample_masked_roi_stats`, and
    `sample_masked_person_roi_stats` from `AlignedDepthFrameDevice`.
  - `DS9/native/noesis_depth_tracking_tensor_kernels.cu` contains the DS9-local
    CUDA ROI/stat kernels.
  - Both `DS9/scripts/build_native_extensions.sh` and the per-module
    `DS9/scripts/build_native_ext_ds9.sh` build from `DS9/native/`, link sibling
    CUDA kernel objects when present, and stage outputs under
    `DS9/native_extensions/`.
- Updated DS9 SGIE cadence and tracking defaults for detection-wake load:
  - YOLO26 pose SGIE now uses batch size 3 and `secondary-reinfer-interval=8`.
  - ReID SGIE now uses `secondary-reinfer-interval=6` and keeps synchronous
    tensor metadata extraction.
  - DS9 NvDCF defaults are trimmed for home-scale scenes: lower target cap,
    shorter shadow age, HOG disabled, smaller feature image size, and internal
    NvDCF ReID disabled.
- Updated the DS9 YOLO26 pose asset contract to fail fast on DS9-owned batch-3
  assets: `DS9/models/onnx/yolo26n-pose_b3.onnx` must be staged before
  `DS9/models/engines/yolo26n-pose_b3_fp16.engine` can be rebuilt. The root
  DS8 pose engine is not reused.
- Implemented the DS9-native pose metadata path:
  - `DS9/noesis/ds9_runtime.py` prepends `DS9/native_extensions` so DS9 imports
    DS9-built native extensions, not root DS8 `.so` artifacts.
  - `DS9/native/noesis_pose_meta_ext.cpp` exposes a Service Maker-safe API marker
    and `attach_pose_features_with_batch(...)` so DS9 can allocate
    `NvDsUserMeta`/`UserMetadata` from the DS9 batch and append
    `NOESIS.POSE_FEATURES` to object metadata without `pyds`.
  - `DS9/noesis/pipelines/hooks.py` decodes YOLO26 pose tensor output from
    Service Maker `tensor_items`, attaches the existing compact
    `NOESIS.POSE_FEATURES` contract, and keeps a bounded DS9 latest-real-pose
    cache per track so telemetry can consume sparse SGIE tensor metadata without
    synthetic keypoints.
- Rebuilt MapAnything with DS9 TensorRT using BF16 precision. The engine path is
  still `DS9/models/engines/mapanything_images_294x518_b3_fp16.plan` for config
  compatibility, but the guarded builder invokes `trtexec --bf16`.
- Set DS9 BEV output to `backend_world_m` in `DS9/config/infer.yaml`.
- Increased on-demand MapAnything RPC burst/wait timeouts and made the
  WebSocket depth-provider timeout configurable with
  `NOESIS_DEPTH_RPC_TIMEOUT_SECONDS`.
- Corrected DS9 label paths from `../../models/coco_labels.txt` to
  `../models/coco_labels.txt` in:
  - `DS9/pipelines/config_infer_primary_yolo11_seg.ini`
  - `DS9/pipelines/config_infer_primary_yolo11.ini`
  - `DS9/pipelines/config_infer_primary_rfdetr_seg.ini`
  - `DS9/pipelines/config_infer_primary_rfdetr.template.ini`
  - `DS9/pipelines/config_infer_primary_rfdetr_seg.template.ini`
- Updated `.gitignore` so `DS9/models/coco_labels.txt` can be tracked while
  large DS9 model artifacts remain ignored.
- Moved the pose feature probe attachment in `noesis/pipelines/hooks.py` from
  the pose SGIE component to `world_observation_stage`. DS9 telemetry showed the
  pose SGIE engine loaded, but `pose_present=0` and no pose debug logs appeared
  when the probe was attached directly to the SGIE.
- Follow-up DS9 parity work proved the above probe relocation was not enough for
  full parity. Re-enabling the DS8 native pose metadata bridge in DS9 caused a
  segmentation fault at the first analytics batch, so that unsafe bridge is
  disabled by default in DS9.
- Corrected DS9 runtime/model materialization isolation:
  - `DS9/noesis/ds9_runtime.py` now sets `NOESIS_MODEL_DIR` to `DS9/models`,
    not root `models`.
  - `DS9/noesis/yolo26_assets.py` now uses `NOESIS_ONNX_DIR` for YOLO26 ONNX
    paths while keeping labels under `NOESIS_MODEL_DIR`.
  - RF-DETR materialized PGIE configs now write DS9-scoped absolute label and
    parser library paths, so generated INIs under `DS9/build` do not resolve
    parser libraries relative to root `build/`.
- Corrected YOLO11/YOLO26 detect-only materialization:
  - Generated detector PGIE configs now use the DS9 YOLO parser path instead of
    resolving `custom-lib-path` relative to `DS9/build`.
  - YOLO26 detect-only preprocess configs set `tensor-name=images`.
  - The materialization matrix covers `yolo11`, `yolo26 n/s/m/l/x`,
    `yolo26_seg n/s/m`, `rfdetr n/s/m`, and `rfdetr_seg n/s/m`.

## Current Validation State

The 2026-06-30 detection-wake DS9 port passed focused static/native checks on
the host:

- `python3 -m py_compile DS9/scripts/rebuild_engines.py
  DS9/noesis/pipelines/hooks.py DS9/noesis/ds9_runtime_core.py
  DS9/noesis/ds9_runtime.py`
- `bash -n DS9/scripts/build_native_ext_ds9.sh
  DS9/scripts/build_native_extensions.sh DS9/scripts/build_all_native_ds9.sh
  DS9/scripts/build_noesis_depth_tracking_tensor_ext.sh`
- `DS9/scripts/run_static_prep_checks.sh`
- `git diff --check`
- `DS9/scripts/build_noesis_depth_tracking_tensor_ext.sh`
- DS9-staged import/symbol check proving
  `sample_roi_stats`, `sample_masked_roi_stats`, and
  `sample_masked_person_roi_stats` are exposed from
  `DS9/native_extensions/noesis_depth_tracking_tensor_ext*.so`.

Full `DS9/scripts/ds9_preflight.py` is currently blocked on this host because
`/usr/local/bin/trtexec` reports TensorRT 10.13.3 instead of DS9-required
10.14.x, and because DS9 model/parser/plugin artifacts are not staged. The
native depth-tensor extension was rebuilt into `DS9/native_extensions/`, but the
full preflight still requires the remaining DS9 artifacts. The DS9 YOLO26 pose
rebuild correctly stops until `DS9/models/onnx/yolo26n-pose_b3.onnx` exists.

The latest DS9 core parity pass proved these checks in
`nvcr.io/nvidia/deepstream:9.0-triton-multiarch`:

- All five engines deserialize/load:
  - `mapanything_fullframe`
  - `depth_tracking_fullframe`
  - `yolo26_pose`
  - `reid_osnet`
  - `yolo11_pgie`
- RTSP mosaic port `8554` is open.
- WebRTC signaling port `6008` is open.
- Live DS9 pose activation passed. A 90-second telemetry sample observed
  `pose_present_true=25`, `world_valid_true=5137`, and
  `backend_world_m=5137`; the first pose-positive track used
  `world_source=pose_floor_only`.
- BEV/track parity passed:
  `track_total=2769`, `track_world_valid=2769`,
  `bev_world_frame_backend_world_m=1632`, `comparisons=1887`,
  `p95_err_m=0.0`.
- MapAnything depth RPC passed cache-first plus fresh against `family-room`.
- Floorplan RPC passed.
- ReID stable-ID smoke passed for `kitchen:1`.
- WebRTC smoke passed after reinstalling the libav codec dependencies in the
  container. Passing run: `rtp_packets=2287`, `decoded_frames=184`.
- RTSP mosaic decode passed with `rtph264depay ! h264parse ! avdec_h264 !
  fakesink`.
- DS8-vs-DS9 config parity passed for model coverage, batch size, shared top-level
  stages, and DS9-scoped model config/engine paths.
- Zero-copy stats smoke passed against DS9 runtime:
  `samples=43`, `max_zero_copy_violations=0`, `max_boundary_p99_ms=1.034443`,
  `ws_depth_requests=43`, `ws_depth_responses=5`.
- Zero-copy REST depth smoke passed against DS9 runtime launched with
  `--enable-rest`: `samples=62`, `max_zero_copy_violations=0`,
  `max_boundary_p99_ms=1.146929`, `rest_refresh_attempts=8`,
  `rest_refresh_success=8`, `rest_refresh_last_status=ok`.
- ROI unit/API coverage passed:
  `pytest -q tests/test_exclude_prune_hook.py tests/test_analytics_api.py`.
  Live full-frame exclusion update also pruned pose/debug object counts to zero
  after `nvdsroiexclude` reload, proving pre-tracker exclusion behavior.
- Focused host MP4 ROI hot-restore passed after the pipeline builder was updated
  to honor `NOESIS_ANALYTICS_EXCLUDE_CONFIG` for the initial
  `analytics_exclude` element config:
  `scripts/roi_reload_smoke_test.py --no-spawn --hot-restore --pipeline-config
  DS9/build/infer_reid_mp4.yaml --baseline-timeout 45 --excluded-timeout 45
  --restore-timeout 60` observed
  `baseline_total=2`, `excluded_total=0`, and `restored_total=4`.
- Bridge-specific object-depth, depth tensor, and ReID native extraction smoke
  passed with:
  `python3 DS9/scripts/ds9_bridge_contract_smoke_test.py --ws ws://127.0.0.1:6008 --duration 75 --require-embedding-track`.
  The run observed `stats_samples=76`, `tracking_messages=1472`,
  `tracks_seen=2614`, `depth_ok_tracks=2248`, `embedding_tracks=2609`,
  `depth_tracking_device_frames_total=1533`,
  `object_depth_gpu_roi_copies_total=2246`, `object_depth_attach_total=2612`,
  `object_depth_status_total.ok=2246`, `tensor_host_copies_total.reid=354`,
  `core_path.cpu_copy_violation.total=0`, and `pipeline_errors=[]`.
- Alternate profile smokes:
  - `--pgie-profile yolo11 --disable-rest`: passed. DS9 loaded
    `DS9/models/engines/yolo11m_b3_fp16.engine` and the DS9 YOLO detector
    parser.
  - `--pgie-profile yolo26 --size m --disable-rest`: passed. DS9 loaded
    `DS9/models/engines/yolo26m_b3_fp16.engine`, selected
    `tensor-name=images`, activated the runtime, and delivered first mosaic
    frames before bounded timeout shutdown.
  - `--pgie-profile yolo26_seg --size s --disable-rest`: passed. DS9 loaded
    `DS9/models/engines/yolo26s-seg_fused_b3_fp16.engine` and pose attachments
    appeared in live debug counters.
  - `--pgie-profile rfdetr_seg --size m --disable-rest`: passed after the
    generated RF-DETR parser path fix. DS9 loaded the DS9 TensorRT plugin,
    parser, and `DS9/models/engines/rfdetr_seg_m_432_b3_fp16.engine`.
  - `--pgie-profile rfdetr --size s --disable-rest`: passed after staging
    detect-only ONNX files under `DS9/models/onnx/` and building DS9 TensorRT
    engines for `n`, `s`, and `m`. DS9 loaded
    `DS9/models/engines/rfdetr_s_512_b3_fp16.engine` and the materialized
    `DS9/build/config_infer_primary_rfdetr_s.ini`; the bounded command stayed
    alive until `timeout` returned `RUNTIME_RC=124`.

No DS8 artifact fallback, compatibility shim, synthetic keypoints, or disabled
pose workaround was used for these passing gates.

## Host Cutover Validation State

Host validation on 2026-06-16 UTC used the DS9-staged resources already built
under `DS9/`; no host rebuild of engines, parsers, plugins, or native
extensions was performed before validation.

Host install observed during validation:

- NVIDIA driver: `595.71.05` on `NVIDIA GeForce RTX 3060`.
- DeepStream: `9.0.0`.
- CUDA runtime reported by DeepStream: `13.1`.
- TensorRT: `10.14.1.48`.
- Python runtime: `torch 2.12.0+cu130`, CUDA available.

Host-specific runtime fixes made during cutover:

- `DS9/noesis/ds9_runtime.py` preloads Torch and warms CUDA before handing off
  to the Service Maker runtime. This avoids late Torch/CUDA initialization after
  GStreamer threads are active. Set `NOESIS_DS9_PRELOAD_TORCH=0` only for
  debugging.
- `DS9/noesis/ds9_runtime.py` now defaults `--storage-base` to
  `DS9/data/depth`, avoiding root-owned legacy `data/depth` directories created
  by earlier Docker runs.
- `DS9/noesis/ds9_runtime.py` and `DS9/scripts/ds9_preflight.py` pin
  `pyservicemaker` to the DS9 system install before the shared runtime imports
  Service Maker. A stale user-site `pyservicemaker` copy was shadowing the DS9
  binding and caused minimal constructor/destructor crashes plus native heap
  corruption during shutdown; that user-site package has been removed on the
  host.
- Script-mode `DS9/noesis/ds9_runtime.py` now exits immediately with the runtime
  return code after Noesis teardown completes. This avoids Python interpreter GC
  destructing Service Maker native objects while a native wait thread is still
  alive after EOS.
- `DS9/noesis/ds9_runtime_core.py` treats Service Maker EOS as expected when
  shutdown is already requested or when the built graph is a finite-source graph
  (`streammux.live-source=0`). This prevents MP4 validation EOS from poisoning
  the runtime exit code while preserving surprise live-source EOS as a failure.
- `DS9/noesis/ds9_runtime_core.py` preserves DS9-specified depth-tracking
  `config-file-path` and `engine` entries instead of rematerializing root DS8
  depth assets when `NOESIS_DEEPSTREAM_MAJOR=9`.
- `DS9/noesis/ds9_runtime_core.py` prebuilds the REST app and preloads
  uvicorn/logging modules before DeepStream pipeline startup, avoiding the host
  late-import segfault seen when REST started after the native pipeline.

Host gates proved on a fresh DS9 host runtime:

- `python3 DS9/scripts/ds9_preflight.py`: passed.
- DS9 runtime startup: all five engines loaded, RTSP `:8554`, WebSocket `:6008`,
  and REST `:8080` opened.
- WebRTC smoke passed with `rtp_packets=6945` and `decoded_frames=602`.
- RTSP mosaic decode stayed live until the bounded `timeout` with no
  GStreamer decode error.
- Zero-copy stats smoke passed with `samples=41`,
  `max_zero_copy_violations=0`, `max_boundary_p99_ms=1.002647`,
  `ws_depth_requests=44`, and `ws_depth_responses=5`.
- REST-backed zero-copy smoke passed with `samples=56`,
  `max_zero_copy_violations=0`, `max_boundary_p99_ms=2.549196`,
  `rest_refresh_attempts=9`, `rest_refresh_success=9`, and
  `rest_refresh_last_status=ok`.
- Floorplan RPC passed.
- MapAnything depth RPC passed immediately after moving DS9 storage to
  `DS9/data/depth`; later fresh MapAnything requests timed out once the live
  RTSP sources entered reconnect loops and `depth_fps` fell to zero.
- Focused bridge evidence on host showed DS9 native object-depth/depth/ReID
  counters moving with zero core-path CPU-copy violations:
  `depth_tracking_device_frames_total=4389`,
  `object_depth_gpu_roi_copies_total=31`, `object_depth_attach_total=46`,
  `object_depth_status_total.ok=31`, `tensor_host_copies_total.reid=5`,
  `core_path.cpu_copy_violation.total=0`.
- Focused MP4 ReID stable-ID smoke passed after launching a validation-only
  DS9 graph from `DS9/build/infer_reid_mp4.yaml` with local file sources,
  `streammux.live-source=0`, pose/depth/MapAnything disabled, and the explicit
  `validation.reid_smoke_depthless=true` marker. The
  `DS9/scripts/reid_stable_id_smoke_test.py --no-spawn --duration 35` gate passed
  for `family-room:18` against `DS9/build/infer_reid_mp4.yaml`.

Occupied-camera live-RTSP host gates proved on 2026-06-16:

- BEV/track parity:
  `track_total=11435`, `track_world_valid=11426`,
  `track_world_frame_backend_world_m=11426`, `bev_total=6980`,
  `bev_world_frame_backend_world_m=6980`, `comparisons=11065`,
  `p95_err_m=0.0`, and lagged p95 `0.11610091475856413`.
- ReID stable-ID smoke: `family-room:16`.
- Strict bridge smoke with `--require-embedding-track`:
  `tracking_messages=5373`, `tracks_seen=11436`, `depth_ok_tracks=11066`,
  `embedding_tracks=11398`, `depth_tracking_device_frames_total=4833`,
  `object_depth_gpu_roi_copies_total=10980`,
  `object_depth_attach_total=11350`, `object_depth_status_total.ok=10980`,
  `tensor_host_copies_total.reid=971`,
  `core_path.cpu_copy_violation.total=0`, and `pipeline_errors=[]`.
- WebRTC smoke: `rtp_packets=1438`, `decoded_frames=92`.
- RTSP mosaic decode held for 25 seconds and sustained decode held for
  60 seconds.
- Zero-copy stats: `samples=7`, `max_zero_copy_violations=0`,
  `max_boundary_p99_ms=1.123324`, `ws_depth_requests=38`, and
  `ws_depth_responses=4`.
- MapAnything depth RPC passed cache-first plus fresh for `family-room`.
- Floorplan RPC passed for `kitchen` and then `family-room`; the first
  `family-room` attempt timed out before succeeding on retry.
- Shutdown/native cleanup was reproduced with MP4 file inputs and fixed for the
  DS9 launcher path: preflight now reports the system DS9 `pyservicemaker`,
  SIGINT posts EOS, and the process exits `0` with no fatal Python, segfault,
  malloc, double-free, or heap-corruption markers. Looping MP4 sources can still
  log `Wait thread did not terminate cleanly` because Service Maker `wait()` does
  not unwind after EOS in that graph. Live RTSP SIGINT shutdown also exited `0`,
  closed ports `6008`, `8080`, and `8554`, left no DS9 runtime process, and
  showed no native heap/fatal markers. The shutdown tail logged the wait-thread
  warning and one `source_2` reconnect warning after EOS was posted.

## Broader Migration Validation Results

The checks above prove the core production path, not full DS8 feature and
option-surface parity. Current broader-gate status:

- Passed: `DS9/scripts/zero_copy_stats_smoke_test.py --no-spawn`.
- Passed: `DS9/scripts/zero_copy_smoke_test.py --no-spawn` against a DS9 runtime
  launched with REST enabled.
- Passed: focused ROI pruning and hot restore.
- Passed: enabled YOLO11 detect-only, YOLO26 detect-only `n/s/m/l/x`,
  YOLO26-seg, RF-DETR-seg, and RF-DETR detect-only profile materialization and
  focused startup coverage.
- Blocked: V3DT is not yet DS9-native. The DS9 V3DT smoke script now refuses to
  spawn unless an explicit DS9 V3DT pipeline config is provided, and the
  available V3DT configs still reference root DS8 engines, local `/home/...`
  clips, and non-DS9 tracker paths.
- Passed: bridge-specific object-depth, depth tensor, and ReID native extraction
  smoke. V3DT bridge evidence remains out of scope until DS9-native V3DT staging
  exists.
- Passed: focused host MP4 ReID stable-ID smoke using the validation-only
  depthless ReID config, plus occupied-camera live-RTSP production ReID evidence.

## Verified Local Paths

These paths existed when this handoff was written:

- `DS9/noesis/ds9_runtime.py`
- `DS9/noesis/ds9_runtime_core.py`
- `DS9/config/infer.yaml`
- `DS9/config/depth_registration.json`
- `DS9/models/coco_labels.txt`
- `DS9/scripts/ds9_preflight.py`
- `DS9/scripts/rebuild_engines.py`
- `DS9/scripts/build_gst_plugins.sh`
- `DS9/scripts/build_trt_plugins.sh`
- `DS9/scripts/build_custom_parsers.sh`
- `DS9/scripts/build_native_extensions.sh`
- `DS9/scripts/build_mapanything_guarded.sh`
- `DS9/scripts/ds9_bridge_contract_smoke_test.py`
- `DS9/scripts/ds9_live_validation_runner.py`
- `DS9/pipelines/config_infer_secondary_yolo26_pose.ini`
- `DS9/models/onnx/rfdetr_n_384.onnx`
- `DS9/models/onnx/rfdetr_s_512.onnx`
- `DS9/models/onnx/rfdetr_m_576.onnx`
- `DS9/models/engines/rfdetr_n_384_b3_fp16.engine`
- `DS9/models/engines/rfdetr_s_512_b3_fp16.engine`
- `DS9/models/engines/rfdetr_m_576_b3_fp16.engine`
- `noesis/pipelines/hooks.py`
- `scripts/menon_bev_track_parity_smoke_test.py`
- `scripts/webrtc_gateway_smoke_test.py`
- `DS9/scripts/ma_depth_rpc_smoke_test.py`
- `scripts/floorplan_rpc_smoke_test.py`
- `DS9/scripts/reid_stable_id_smoke_test.py`
- `DS9/pipelines/nvdsinfer_yolo_detect/nvdsinfer_yolo_detect.cpp`
- `DS9/pipelines/nvdsinfer_yolo_detect/Makefile`

## Remaining Work

- Stage DS9-native V3DT pipeline/camera/tracker assets before attempting V3DT
  metadata parity. Do not run V3DT smokes against root DS8 configs as DS9
  evidence.
- Decide whether DS9 should remain a folder in the Noesis monorepo that shares
  app helpers, or become a hermetic standalone repository with vendored/extracted
  shared modules.
- Keep `DS9/scripts/ds9_bridge_contract_smoke_test.py` in the regression set
  when object-depth, depth tensor, or ReID native bridge code changes.
- Keep the DS9-native pose path and MapAnything BF16/runtime timeout behavior
  covered by future regression gates.
- Collect longer host RTSP soak evidence if production acceptance requires more
  than the occupied-camera validation window already captured. Use
  `DS9/scripts/ds9_live_validation_runner.py` for future full-bundle host
  evidence.
- If the DS9 validation container is recreated, reinstall or verify the WebRTC
  decode dependencies (`gstreamer1.0-libav` plus codec runtime libraries) and
  clear the GStreamer registry if plugins remain blacklisted.
- Keep DS9 artifacts under `DS9/`; do not reintroduce DS8 fallback paths.
