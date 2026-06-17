# DS9 Known Blockers

Last updated: 2026-06-16

## Resolved Blocker: Pose Features Not Reaching World Telemetry

Previous symptom:

- DS9 runtime loads the `yolo26_pose` SGIE engine.
- Live detections include pose-eligible objects.
- Tracking/world telemetry still reports `pose_present=0`.
- BEV/track parity smoke fails with `track_world_valid=0`.
- World quality reports:
  `pose_keypoints_unusable,depth_meta_missing,height_lock_missing`.

Previous confirmed issue:

- Pose tensor metadata is not reaching the world/telemetry contract through a
  safe DS9 path. The pose SGIE loaded, but the DS9 run still published
  `pose_present=0` and `world_valid=0`.

Attempts before resolution:

- `noesis/pipelines/hooks.py` now attaches the pose feature probe to
  `world_observation_stage` when that component exists, instead of attaching
  directly to the pose SGIE component.
- Re-enabling the DS8 native pose metadata bridge under DS9 caused a
  segmentation fault at the first analytics batch.
- DS9 now keeps the unsafe DS8 native pose extraction bridge disabled by default;
  it should only be forced for debugging with `NOESIS_POSE_NATIVE_EXTRACT_ENABLED=1`.

Resolution:

- Implemented. DS9 now uses the DS9-built `noesis_pose_meta_ext` Service Maker
  API to allocate object user metadata from the active DS9 batch and attach the
  existing `NOESIS.POSE_FEATURES` payload. Python decodes the YOLO26 SGIE tensor
  through Service Maker `tensor_items`; the unsafe DS8 native extraction path
  remains disabled unless explicitly forced for debugging.
- Live validation proved `pose_present=true`, `world_valid > 0`, and BEV/world
  frames in `backend_world_m`. BEV/track parity passed.

Do not work around this by disabling pose, using DS8 metadata paths, injecting
synthetic keypoints, or treating no-pose world tracks as equivalent.

## Current Gate Status

Core production-path gates from the latest DS9 run:

- `python3 DS9/scripts/ds9_preflight.py`: passed.
- DS9 runtime launch through `DS9/noesis/ds9_runtime.py`: passed.
- DS9 runtime entrypoint ownership: passed. DS9 executable code no longer
  imports or spawns `noesis/ds8_runtime.py`; `DS9/scripts/run_static_prep_checks.sh`
  guards that boundary.
- DS9-native pose feature activation: passed.
- BEV/track parity smoke: passed.
- WebRTC smoke: passed after codec dependencies were reinstalled in the
  validation container.
- MapAnything depth RPC smoke: passed.
- Floorplan RPC smoke: passed.
- ReID stable-ID smoke: passed.
- RTSP mosaic decode on port `8554`: passed.
- DS8-vs-DS9 config/artifact parity review: passed.
- Zero-copy stats smoke: passed.
- Zero-copy REST depth smoke against a REST-enabled DS9 runtime: passed.
- YOLO26 segmentation alternate-profile startup smoke: passed.
- RF-DETR segmentation alternate-profile startup smoke: passed after correcting
  generated DS9 parser/label paths.
- RF-DETR detect-only alternate-profile startup smoke: passed for
  `--pgie-profile rfdetr --size s` after staging DS9 ONNX sources and building
  DS9 TensorRT engines for `n`, `s`, and `m`.
- YOLO11 and YOLO26 detect-only profiles: passed. DS9 now has a DS9-built
  YOLO detector bbox parser at
  `DS9/pipelines/nvdsinfer_yolo_detect/libnvdsparsebbox_yolo.so`, and the
  materialization matrix covers `--pgie-profile yolo11` plus
  `--pgie-profile yolo26 --size n/s/m/l/x`.
- Focused ROI hot-restore behavior: passed on host MP4 input. Full-frame
  exclusion pruned active tracks from `2` to `0`, and restoring the original
  ROI config in the same runtime recovered `4` active tracks.
- Bridge-specific object-depth, depth tensor, and ReID native extraction smoke:
  passed. The smoke observed fresh DS9 runtime counters for native DAv2 device
  frame capture, object-depth GPU ROI copies/attachments, and ReID native
  embedding extraction with zero core-path CPU-copy violations.
- Shutdown/native cleanup smoke on host MP4 input: passed. DS9 preflight and
  launcher now prefer the system DS9 `pyservicemaker` binding instead of a stale
  user-site copy, and script-mode shutdown exits after Noesis teardown to avoid
  Python GC destructing live Service Maker native objects. The smoke exited `0`
  with no fatal Python, segfault, malloc, double-free, or heap-corruption
  markers.
- Occupied-camera live-RTSP host validation: passed. Production public tracking
  was present, BEV/track parity passed with `comparisons=11065` and
  `p95_err_m=0.0`, ReID stable-ID persisted for `family-room:16`, the strict
  bridge smoke saw `embedding_tracks=11398` with zero core-path CPU-copy
  violations, WebRTC decoded `92` frames, RTSP decode held for 25 seconds plus a
  60-second sustained check, MapAnything depth and floorplan RPCs passed, and
  live RTSP SIGINT shutdown exited `0` with closed ports and no native heap/fatal
  markers.

No active core production-path blocker is known from the latest DS9 parity run.

Broader migration-plan blockers/caveats still prevent claiming full DS8
option-surface parity:

- V3DT is not DS9-native yet. `DS9/scripts/sv3dt_meta_smoke_test.py` now refuses
  to spawn unless an explicit DS9 V3DT pipeline config is provided, and the
  available V3DT configs still reference root DS8 engines, local `/home/...`
  clips, and non-DS9 tracker paths.
- V3DT bridge behavior smoke remains blocked/out of scope until DS9-native V3DT
  staging exists.
- DS9 is runtime-standalone from the DS8 entrypoint, DS8 engines, and DS8 native
  extension binaries, but it is not yet a hermetic standalone repository. It
  still shares parent-repo application modules and data such as camera config,
  calibration/geometry helpers, WebSocket server code, and some validation
  clients. Treat this as packaging work, not a runtime fallback.

Host cutover caveats from the 2026-06-16 UTC validation run:

- The host install reaches DeepStream 9.0.0 / TensorRT 10.14.1.48 and starts the
  DS9 runtime with reused DS9 resources.
- Host validation now includes fresh-start gates, focused MP4 ReID, and
  occupied-camera live-RTSP production gates. The core production path is no
  longer blocked by missing public tracking/BEV or live shutdown evidence.
- Remaining caveat: V3DT is still out of scope until DS9-native V3DT staging
  exists. Longer host RTSP soak evidence can be collected with
  `DS9/scripts/ds9_live_validation_runner.py` if production acceptance requires
  more than the occupied-camera validation window captured here.

## Known Validation Caveats

- DS9-copied smoke scripts spawn `DS9/noesis/ds9_runtime.py` by default. For a
  shared already-running validation runtime, pass `--no-spawn` to those scripts.
- `DS9/scripts/zero_copy_smoke_test.py` calls the REST depth-refresh endpoint, so
  it requires DS9 to run with REST enabled.
- REST-enabled DS9 validation requires `fastapi` and `uvicorn[standard]` from
  `DS9/requirements-runtime.txt`. Without `uvicorn`, the runtime accepts
  `--enable-rest` but logs that REST was disabled.
- The successful WebRTC smoke depended on extra GStreamer/libnice/libav/codec
  packages installed into the DS9 validation container. If `avdec_h264` is
  missing even after installing `gstreamer1.0-libav`, force-reinstall the codec
  runtime libraries listed in the validation runbook and remove
  `~/.cache/gstreamer-1.0/registry*.bin`.
- MapAnything ONNX may use external tensor sidecar files. Missing sidecars are a
  hard blocker, not a reason to rebuild from DS8 artifacts.
- Large model, ONNX, and engine artifacts may be ignored by git. Verify their
  presence in the workspace/container before running preflight.
