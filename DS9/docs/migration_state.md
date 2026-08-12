# DS9 Migration State

Last updated: 2026-08-12

## Goal

Directly upgrade the canonical Noesis DeepStream 9.0 app to DeepStream 9.1
without fallbacks while preserving the current application graph, contracts,
and operator process. The executable ownership established during the earlier
DS8-to-9.0 migration remains unchanged.

The active execution checklist is
`DS9/docs/deepstream_9_1_direct_upgrade_plan.md`. Source/toolchain pinning and
its focused contract tests are complete. The exact 9.1 base image has been
pulled and inspected; the derived image has not been built, and no 9.1 native
binary, parser, plugin, TensorRT engine, recorded-media smoke, or live-camera
result is accepted. The 9.0 results retained below are historical comparison
evidence only.

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

- Canonical base image:
  `nvcr.io/nvidia/deepstream:9.1-triton-multiarch@sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994`
- DeepStream: 9.1
- CUDA: 13.2.0.046
- TensorRT: 10.16.0.72
- DS9 SDK root: `/opt/nvidia/deepstream/deepstream-9.1`
- Minimum driver: 595.58.03; current host: 595.71.05

The previous 9.0 target and its exact artifacts remain recorded in dated
sections below. DS8/DS9.0 engines and compiled libraries are incompatible
inputs for this target and must be rebuilt.

## MapAnything Depth-Panel Quality Acceptance — 2026-07-27

The active appliance deployment
`deploy-20260727-mapanything-depth-floorplan-v10-ds9` retains the selected
correctness-first FP32 MapAnything model and canonical DS9 baseline lane. Fresh
manual Refresh completed for living room, kitchen, and family room in
32.6–34.3 seconds. All three responses were floorplan contract v9 products from
the exact captured snapshot, with 1.60–1.68 million points and strict temporal
consensus coverage of 47.14%, 53.94%, and 60.96% against the 40% gate.

Floorplan v9 separates dominant horizontal surfaces, furniture, vertical wall
support, room footprint and boundary, and exact-snapshot RGB instead of
collapsing them into one averaged height cell. Observed metric bounds replace
near-horizon calibration expansion, and quarantine-aware fusion voting uses
only the surviving cohort while still failing closed below three usable frames.
The resulting 0.10 m live grids span 15×7 m, 17×8.5 m, and 15×10.5 m. This
closes the fresh multi-camera MapAnything depth/floorplan quality gate; it is
not a claim of full DS9 option-surface parity or surveyed absolute-scale
acceptance.

## Runtime Ownership And Drift Governance

The current ownership contract is `DS9/docs/runtime_ownership.yaml`. It classifies
SDK-neutral shared modules, DS8/DS9 adapter modules, every copied module that
still needs convergence, and parity across CLI, configuration, API, telemetry,
metadata, media, tracking, and artifact surfaces.

`DS9/scripts/validate_runtime_ownership.py` currently classifies all 23 files
duplicated between `noesis/` and `DS9/noesis/`. Its structural mode passes while
reporting declared gaps. `--require-parity` makes those gaps cutover blockers.
The canonical artifact graph is promoted from its exact external realization.
The three remaining dynamic-evidence gaps are Wholebody49 runtime quality, Swin
ReID runtime quality, and V3DT runtime/world quality. The selected engines and
artifact profiles, including the correctness-first FP32 MapAnything
replacement, are realized. Fresh three-camera MapAnything depth/floorplan
quality is accepted as described above. Household identity APIs and
person-ground product behavior are shared/parity.
The guarded MapAnything build specification now uses correctness-first FP32 and
requires a real-inference functional admission receipt after the loadable FP16
plan produced non-finite depth, zero masks, and confidence sentinels. The
guarded FP32 build, artifact realization, and fresh multi-camera runtime
acceptance now pass. Mosaic WebRTC readiness now matches DS8: one SHM feeder
must prove its socket, PLAYING state, and first complete H.264 AU; one warm
gateway slot is bounded, and additional gateways are allocated on demand.
Optional RTSP output is not consumed by those gateways.

DS9 configuration materialization now owns its PGIE-derived OSD helper in
`DS9/noesis/runtime_config.py`. DS9 executable code no longer imports
`noesis.ds8_preflight`; both DS8 runtime and DS8 preflight imports are forbidden
by the drift gate.

The DS9 runtime also passes its active `camera_labels` map into the shared
stable-ID manager. The shared fail-fast camera-topology validation therefore
applies equally to DS8 and DS9 instead of allowing successor-only drift.

Identity-v2 runtime product behavior is now shared at source level. DS8,
V3DT, and DS9 construct `noesis/identity_v2_service.py` with the canonical world
producer run ID, an SHA-256 of the actual active ReID engine bytes, an explicit
tensor layer/dimension, and a strictly matching camera topology. Each adapter
walks transient SDK metadata once and submits one detached complete source-frame
batch. The shared authenticated `/api/v2/reid` enrollment path consumes exact
server evidence only; clients cannot submit embeddings. Default mode is shadow,
and authoritative startup requires an artifact-backed scoring calibration.
In authoritative mode DS9 also bypasses legacy StableID mutation and resolves
SID-dependent analytics only after its complete identity-v2 frame batch. The
one-shot object metadata pass remains neutral, then a dedicated tiler-sink
BatchMetadataOperator joins the resolved same-frame identity result and stamps
the authoritative OSD label. Tracking/world telemetry and OSD therefore share
the same resolved v2 authority without a stale legacy lookup.

The first sealed occupied baseline attempt on 2026-07-11 exposed three real
parity defects and was not promoted: one camera-local tracker changed visitor
subjects within 0.683 seconds under joint-assignment contention, legacy cached
`embedding_present` values escaped without exact persisted association, and
DS9 legacy StableID performed one pressure auto-merge while identity-v2 was
shadow-only. The shared coordinator now hard-locks an accepted subject until a
fresh-evidence gap expires the tracker state without bypassing open-set gates;
the shared service owns exact-frame embedding/provenance truth; and DS9 admits
and verifies the same default household no-auto-merge/no-blanket-sharing policy
as DS8. CPU replay/parity tests pass. Fresh same-session occupied evidence is
still required; the failed transcript remains immutable evidence, not a pass.

Recovery validation on 2026-07-10: focused identity/calibration/DS9 parity
suites and the complete containerized DS9 suite passed, and
`DS9/scripts/run_static_prep_checks.sh` passed. At that checkpoint strict
ownership parity reported five capability gaps, sixteen native/plugin/parser
artifacts were provenance-complete, and strict canonical asset validation
reported only the five then-unbuilt baseline engines. This closed household identity API/runtime,
person-ground copied-product-logic drift, and the ReID source/config/tensor
mismatch, but not the DS9 ReID/live identity-quality gate, Wholebody49 live
parity evidence, or V3DT runtime/world evidence.

Current 2026-07-11 artifact validation supersedes the build-status portion of
that recovery snapshot. The external realization contains ten engines and
passes canonical, V3DT, and Wholebody49 file/provenance profiles with no errors
or blockers at
`6fab7d456c031490f640ee2c3ce5a38922a96ed86a965020ca3051820306dce4`.
This closes selected-engine construction and canonical-graph promotion, not the
four live-session dynamic-evidence gates.

Canonical world publication is now at DS8 parity. DS9 constructs the shared
`CanonicalWorldService` with `create_runtime_world_service(...)` from the
effective pipeline configuration, content evidence for selected model/tracker
files, and the active calibration provider. `TrackingTelemetryPublisher`
publishes versioned person observations plus a separate world snapshot and
advances the shared capability monitor. The same monitor is exposed at
`GET /api/v1/health/capabilities`; readiness therefore reflects compatible
producer progress rather than port reachability.

DS9 also mounts the shared `noesis/server/scene_api.py` router. Scene release
registration, promotion, rollback, current payload, and history therefore use
the same owner-governed store and contracts as DS8; DS9 has no private scene
state implementation.

DS9 also mounts the shared owner-authenticated alignment-walk controller used
by DS8. It captures the successor runtime's own calibration and per-camera
tracking stream, exposes only image geometry plus run-local tracklet keys to
Menon, and produces FIT-only fixed-center rotation and physical-depth
candidates that must pass untouched HOLDOUT checks. It never applies a
candidate or uses the current producer world point as calibration truth.

Both DS9 public-track adapters stamp processing-time `observed_at_us`, explicit
`capture_time_status: estimated`, and a nonnegative stream-relative
`media_pts_ns`. Stable-ID diagnostics also carry `visitor_generation`, keeping
recycled visitor IDs distinct in canonical world entity IDs.

The artifact contract is schema version 2:

- `DS9/docs/asset_manifest.schema.json` defines the machine-readable shape.
- `DS9/asset_manifest.yaml` records role, DS9-owned output/source paths, builder,
  required profiles, compatibility, staging state, and provenance.
- `DS9/scripts/validate_asset_manifest.py` validates structure by default and
  adds file/profile/provenance gates for rebuild and cutover phases.

The tracked manifest remains declarative and does not claim realized output
hashes. The owner-private external realization records ten selected TensorRT
engines with exact maintenance provenance. Structural governance and canonical,
V3DT, and Wholebody49 file/provenance validation pass; dynamic acceptance stays
in the external evidence registry.

The DS9 baseline also declares the same calibrated circular dewarper-validity
regions as DS8. `DS9/noesis/pipelines/hooks.py` applies the shared calibrated
geometry mask after aligning MapAnything tensors, writes invalid depth as NaN,
zeros invalid confidence, and rejects empty evidence rather than emitting a
full-frame zero-depth substitute.

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
- Wired the shared canonical world/fusion service, content-addressed runtime
  provenance, and capability-progress health route into DS9 telemetry.
- Added DS9 model labels at `DS9/models/coco_labels.txt`.
- Added DS9 preflight script at `DS9/scripts/ds9_preflight.py`.
- Added DS9 build scripts for custom parsers, native extensions, GStreamer
  plugins, TensorRT plugins, TensorRT engine rebuilds, and guarded MapAnything
  plan generation.
- Rebuilt DS9 parser/native/plugin artifacts in DS9-target locations.
- Ported the Wholebody49 `s` mask and `x` box profile through the shared semantic
  materializer with DS9-owned templates, labels, parser, ONNX staging, CLI,
  preflight, guarded engine specs, and independently realized engines. Live
  quality evidence remains pending.
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

- Replaced the failed DS9 MapAnything compatibility gate with one exact,
  DS9-owned native tensor contract. The 2026-07-11 baseline evidence proved
  that the Python Service Maker wrapper at `mapanything_fullframe` exposed only
  the sibling DAv2 UID twice, so wrapper IDs no longer select MapAnything.
  Native capture now requires one raw UID 2 record and exact per-frame
  `depth/conf/mask` shape; Python validates batch/source identity without
  applying another batch offset because DS9 nvinfer already attaches
  frame-offset pointers. The metadata-lifetime copy is fixed at three
  `294x518` float32 maps (`1,827,504` bytes), timed, and releases the GIL while
  dereferencing nvinfer-owned storage. The owned arrays enter a bounded
  asynchronous postprocessor; runtime shutdown closes capture admission,
  drains accepted jobs, and joins its non-daemon worker before closing depth
  storage. Ambiguity, attachment failure, queue saturation, final-job poison,
  or unresolved teardown is fatal.
  Source tests pass. The canonical DS9 build image rebuilt the native binary
  without GPU access on 2026-07-11; manifest provenance now binds output SHA-256
  `fcd38dceadcbc62c14e257efc5c997d1bb1cc924490541cf452061be09065423`,
  the exact export imports from the staged DS9 path, the retired generic export
  is absent, and canonical/V3DT/Wholebody49 artifact-realization profiles pass.
  Fresh live depth/floorplan gates remain pending.

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
  - ReID SGIE uses the shared TAO Swin-Tiny `fc_pred/256` contract,
    `secondary-reinfer-interval=12`, and synchronous tensor metadata extraction.
  - DS9 NvDCF defaults are trimmed for home-scale scenes: lower target cap,
    shorter shadow age, HOG disabled, smaller feature image size, and internal
    NvDCF ReID disabled.
- Updated the DS9 YOLO26 pose asset contract to fail fast on DS9-owned batch-3
  assets. `DS9/models/onnx/yolo26n-pose_b3.onnx` is now CPU-exported and
  hash-verified; `DS9/models/engines/yolo26n-pose_b3_fp16.engine` is independently
  realized for DS9. The root DS8 pose engine is not reused.
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
- Historical validation rebuilt MapAnything with DS9 TensorRT using BF16 while
  naming the engine `mapanything_images_294x518_b3_fp16.plan`; that artifact is
  not accepted. The current guarded realization omits reduced-precision flags,
  records precision `fp32`, and requires a replayable real-inference functional
  receipt before installation. Fresh depth/floorplan runtime quality remains
  mandatory before cutover.
- 2026-07-11: DS9 inline BEV is wired to the shared renderer and the DS9-owned
  active-floorplan registry/provider/fatal callback with
  `frame=camera_local_ground_m`; the separate canonical global world remains
  `backend_world_m`. Before the first valid authority, `None` is
  `startup_pending` and emits no local BEV. Invalid first authority and every
  post-ready loss fail closed without config/auto-extents substitution. Exact
  empty frames count as successful active renders. Tracking/BEV publication is
  same-frame and tracking-first at
  `max(selected tracking interval, BEV interval)`, with count/lifecycle changes
  forcing the pair and tracking failure suppressing BEV. “Tracking-first” now
  means one immutable finite-only ordered tracking/world/event batch is bounded
  and frozen behind a one-shot sender gate; synchronous exact-count journal and
  private world/lifecycle authority commit before release. Paired BEV carries
  the exact sequence/submission cohort with a later admission ID. Pre-admission
  failure is exactly retryable; post-admission authority failure aborts with
  zero delivery and poisons publication. Registered-depth
  coherence and calibration-image scaling are shared with DS8 and protected
  V3DT. Static and focused tests pass; fresh
  `mapanything_depth_quality_v4` live evidence remains required before release.
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

## Validation State And Historical Evidence

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

Host-SDK preflight remains intentionally unsuitable because the host
`/usr/local/bin/trtexec` reports TensorRT 10.13.3. It now explicitly accepts the
installed `595.71.05` driver against the exact `590.48.01` floor before failing
the independent TensorRT gate. The isolated derived image passes the DS9 9.0 /
TensorRT 10.14 environment, Service Maker, GStreamer, native-import, static,
and test gates without GPU devices. Current authoritative artifact preflight
passes the canonical, V3DT, and Wholebody49 profiles against the ten-engine
external realization. Native/plugin/parser outputs and the CPU-exported
`DS9/models/onnx/yolo26n-pose_b3.onnx` plus the other selected source inputs are
staged and hash-verified in the explicit artifact workspace.

The latest DS9 core parity pass proved these checks in
`nvcr.io/nvidia/deepstream:9.0-triton-multiarch`:

- All five engines deserialize/load:
  - `mapanything_fullframe`
  - `depth_tracking_fullframe`
  - `yolo26_pose`
  - `reid_osnet` (historical evidence only; it does not validate the currently
    selected `reid_sgie` Swin-Tiny engine)
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
- The earlier script-mode immediate-exit workaround is superseded. DS9 now
  injects acknowledged EOS through its repo-owned `noesiseos` transform
  immediately after `streammux`, requires the expected EOS callback and Service
  Maker `wait()` return, and only then closes callback-owned resources. The
  launcher exits normally through `SystemExit`; native quiescence may not be
  replaced by interpreter-GC bypass.
- Finite-source completion remains narrowly classified by local-file source,
  non-live stream mux, disabled loop, and pipeline-EOS propagation. Surprise
  live-source EOS remains a failure.
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
- Historical June shutdown/native cleanup was reproduced with MP4 file inputs
  under the prior launcher workaround: preflight reported the system DS9 `pyservicemaker`,
  SIGINT posts EOS, and the process exits `0` with no fatal Python, segfault,
  malloc, double-free, or heap-corruption markers. Looping MP4 sources can still
  log `Wait thread did not terminate cleanly` because Service Maker `wait()` does
  not unwind after EOS in that graph. Live RTSP SIGINT shutdown also exited `0`,
  closed ports `6008`, `8080`, and `8554`, left no DS9 runtime process, and
  showed no native heap/fatal markers. The shutdown tail logged the wait-thread
  warning and one `source_2` reconnect warning after EOS was posted. This result
  is retained only as regression provenance; the current canonical runner must
  reject the wait warning and requires acknowledged EOS plus `wait()` return.

## 2026-07-11 Canonical Baseline Validation Repair

Session `baseline-final2-20260711-1832` is preserved failure evidence, not an
acceptance or promotion. Its EOS request/acknowledgement/callback/Service Maker
wait/completion markers were present and ordered, but the supervisor correctly
rejected the run because the complete log still contained `ERROR` severity.
Two sources were real runtime defects: capability health used the aggregate
world observation-window endpoint as its monotonic progress clock even though
interleaved camera/entity removal can legitimately move that endpoint backward,
and closed validation clients were caught inside the request loop and processed
again, generating repeated send/receive errors through shutdown.

The shared DS8/DS9 publisher now uses the advancing world publication timestamp
for health progress and retains aggregate `world_observed_end_us` as evidence.
Closed clients leave the handler through its lifecycle path, normal/abrupt
disconnects no longer become runtime `ERROR` messages, and WebSocket teardown
must prove both listener and thread quiescence. The behavior evidence leaf is
created as exact mode `0700`. The unchanged 3 ms zero-copy limit now measures
the complete local boundary: producer-to-event-loop dispatch,
serializer-worker dispatch, conversion, single admission-freeze encoding, and
WebSocket send dispatch. Sync producers carry the exact pre-encoded bytes to
delivery, so caller mutation and a second unmeasured encode are impossible.
Those stages remain separately visible; only explicitly configured
non-canonical coalescing dwell and authority-gate persistence dwell are
excluded. Canonical tracking/world/event/BEV never
coalesce. Large JSON work uses a dedicated prewarmed owned
executor rather than competing with provider work. Gate output includes bounded
WS/REST and total-stage attribution with no payload content. The sender owns a
256 MiB global frozen-byte cap in addition to its 256-submission cap, counts
explicit gate aborts, and proves zero reserved bytes at shutdown. Telemetry
fanout defaults to 8 authenticated
clients, is hard-clamped to 16, rejects excess clients before snapshots, and
does not charge `/healthz` against that set. Focused
cross-runtime tests passed (228), followed by dedicated assembled-boundary and
shutdown regressions.

Blocking depth, floorplan, and auto-calibration WebSocket callbacks now use an
owned admission-tracked provider executor. Shutdown rejects new calls, drains
and joins admitted calls, stops the listener, closes depth admission, proves
pipeline EOS/wait, joins MapAnything postprocess, and only then releases depth
storage. Cancellation cannot counterfeit provider completion; a bounded drain
failure preserves native/store resources until the watchdog exits nonzero.
A fresh supervised live run is still required; none of these static results is
runtime acceptance.

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
- Artifact-complete, runtime blocked: V3DT now has DS9-owned pipeline, cameras, camInfo,
  tracker, source provenance, native bridge, NvMOT helper, preflight/runtime
  materialization, and smoke surfaces. Large bytes resolve through the explicit
  artifact root; no root DS8 binary or machine-local clip is used. Its three
  selected DS9 TensorRT 10.14 engines are realized; fresh live behavior evidence
  remains absent. Static coordinate parity is complete: the active shared
  calibration has separated camera centers, DS8/DS9 locked camInfo bytes match,
  and the producer converts `xzy` tracker coordinates to Y-up
  `backend_world_m`. Promotion accepts only the same-session privacy-safe v2
  replay; MV3DT overlap/time-sync/fusion remains a separate claim.
- Passed: bridge-specific object-depth, depth tensor, and ReID native extraction
  smoke. Fresh V3DT bridge evidence remains pending until its runtime gate is
  scheduled.
- Passed: focused host MP4 ReID stable-ID smoke using the validation-only
  depthless ReID config, plus occupied-camera live-RTSP production ReID evidence.

## Verified Local Paths

These source/governance paths exist in the current checkout:

- `DS9/noesis/ds9_runtime.py`
- `DS9/noesis/ds9_runtime_core.py`
- `DS9/noesis/runtime_config.py`
- `DS9/config/infer.yaml`
- `DS9/config/depth_registration.json`
- `DS9/models/coco_labels.txt`
- `DS9/docs/runtime_ownership.yaml`
- `DS9/asset_manifest.yaml`
- `DS9/docs/asset_manifest.schema.json`
- `DS9/scripts/validate_runtime_ownership.py`
- `DS9/scripts/validate_asset_manifest.py`
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
- `noesis/pipelines/hooks.py`
- `scripts/menon_bev_track_parity_smoke_test.py`
- `scripts/webrtc_gateway_smoke_test.py`
- `DS9/scripts/ma_depth_rpc_smoke_test.py`
- `scripts/floorplan_rpc_smoke_test.py`
- `DS9/scripts/reid_stable_id_smoke_test.py`
- `DS9/pipelines/nvdsinfer_yolo_detect/nvdsinfer_yolo_detect.cpp`
- `DS9/pipelines/nvdsinfer_yolo_detect/Makefile`

Large TensorRT bytes live outside the checkout and their truth comes from the
external realization, not the declarative manifest or this source-path list.
Native extensions, parsers, and plugins are staged with provenance. Run the
artifact validator rather than inferring readiness from filenames.

## Remaining Work

- Build the normal derived images from the inspected 9.1 base; the Dockerfile
  installs the bundled Service Maker wheel explicitly.
- Rebuild every selected native bridge, parser, GStreamer/TensorRT plugin, and
  TensorRT engine against DeepStream 9.1 / CUDA 13.2 / TensorRT 10.16.0.72.
  Validate actual 9.1 bytes and never relabel 9.0 provenance.
- Run the focused static, import/load, engine-deserialization, recorded-media,
  and bounded live-baseline checks listed in the direct-upgrade plan. Activate
  through the existing selector only after those direct checks pass.
- Keep AMC deferred. Keep MV3DT distinct and disabled: Kitchen/Family Room is
  the only prospective edge, Living Room has no MV3DT edge, and activation is
  blocked on corrected Kitchen geometry plus synchronized occupied overlap,
  peer-association, and fused-position evidence.
- Decide whether DS9 should remain a folder in the Noesis monorepo that shares
  app helpers, or become a hermetic standalone repository with vendored/extracted
  shared modules.
- Keep `DS9/scripts/ds9_bridge_contract_smoke_test.py` in the regression set
  when object-depth, depth tensor, or ReID native bridge code changes.
- Keep the DS9-native pose path and validated-FP32 MapAnything functional
  admission/runtime timeout behavior covered by future regression gates.
- Collect longer host RTSP soak evidence if production acceptance requires more
  than the occupied-camera validation window already captured. Use
  `DS9/scripts/ds9_live_validation_runner.py` for future full-bundle host
  evidence.
- If the DS9 validation container is recreated, reinstall or verify the WebRTC
  decode dependencies (`gstreamer1.0-libav` plus codec runtime libraries) and
  clear the GStreamer registry if plugins remain blacklisted.
- Keep DS9 artifacts under `DS9/`; do not reintroduce DS8 fallback paths.
