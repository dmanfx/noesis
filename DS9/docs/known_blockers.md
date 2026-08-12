# DS9 Known Blockers

Last updated: 2026-07-27

## Current MapAnything Status

The July 11 live `no_depth` blocker and the separate non-finite FP16 engine
defect are superseded by accepted 2026-07-27 evidence. Active deployment
`deploy-20260727-mapanything-depth-floorplan-v10-ds9` uses the selected FP32
MapAnything engine and returned fresh floorplan contract v9 products for all
three cameras. Manual Refresh completed in 32.6–34.3 seconds with 1.60–1.68
million points per camera, strict consensus coverage above the 40% gate, and
structural floorplan layers that retain furniture, wall, room-boundary, and
exact-snapshot RGB evidence.

This closes MapAnything live depth/floorplan quality as a current DS9 blocker.
Surveyed anchors and held-out-scene evaluation remain future confidence work
for absolute scale and unseen geometry; they do not invalidate the accepted
manual depth-panel path.

## Current Checkout Blockers

The selected DS9 artifact profiles and canonical baseline lane are runnable, but
the current checkout is not ready to cut over:

- The packaged driver migration is complete. The GPU, loaded/on-disk module,
  DKMS, and userspace all report `595.71.05`, above NVIDIA's DeepStream 9 floor
  of `590.48.01`; the current boot has no Xid, API mismatch, GPU-fallen-off, or
  `RmInitAdapter` event. The canonical authenticated DS8 YOLO26m gate has since
  passed its 30-second active window and acknowledged-EOS shutdown with exact
  exit `0` and no forced kill, so native lifecycle is no longer the DS9-build
  blocker. Remaining DS8 product/soak canaries and profile-specific DS9 live
  acceptance still govern cutover. Rebuilding any realized DS9 engine still
  requires an announced exclusive-GPU window.
- The host `trtexec` is TensorRT 10.13.3 and remains unsuitable for DS9 builds.
  The isolated, digest-pinned DS9 image provides TensorRT 10.14.1.48 without
  altering the host SDK.
- The owner-private external realization now records ten selected TensorRT
  engines. Canonical, V3DT, and Wholebody49 file/provenance profiles all validate
  with no errors or blockers against realization
  `6fab7d456c031490f640ee2c3ce5a38922a96ed86a965020ca3051820306dce4`.
  Declarative manifest entries outside those selected profiles may remain
  `missing`; they are not evidence for or against a selected runtime lane.
- The canonical artifact realization is promoted into the external registry.
  The strict ownership gate therefore reports three remaining live-session
  promotions: Swin ReID runtime quality, Wholebody49 runtime quality, and V3DT
  runtime/world quality. MapAnything runtime quality passed fresh three-camera
  acceptance on 2026-07-27. Shared household identity, person-ground behavior,
  and H.264 SHM/WebRTC readiness ownership are static parity.
- DS9-native V3DT source/config/build staging is complete. Runtime parity remains
  blocked on fresh live bridge/global-world-v2/identity/resource/shutdown
  evidence, not engine construction or shared-calibration creation.
- Baseline session `baseline-final2-20260711-1832` failed the MapAnything
  acceptance contract: every on-demand burst stayed at 0 FPS, all three
  floorplans returned `no_depth`, and fresh depth timed out. Its runtime log
  proved the MapAnything operator saw only sibling DAv2 wrapper UID `5`. The
  source now selects one exact raw UID 2 tensor through the DS9-owned native
  bridge, bounds and times the required `1,827,504`-byte probe-local copy, and
  gives the bounded postprocess worker explicit drain/join ownership. The
  native binary and manifest provenance are content-attested in the current
  realization. This historical failed session is not promotion evidence; the
  fresh 2026-07-27 three-camera v10 acceptance above closes the blocker.
- A second independent defect was found at the engine boundary. The currently
  installed FP16 MapAnything candidate deserializes but produces all-NaN depth,
  an all-zero mask, and invalid confidence sentinels on a pinned real input.
  An isolated FP32 diagnostic is finite, positive, dense, and batch-consistent,
  so the reviewed successor target is correctness-first FP32. Guarded canonical
  transaction `20260712T052425739603Z` installed and realized that FP32 engine
  only after a sealed real-inference receipt; both finalization and realization
  reconciliation revalidated it. The fresh 2026-07-27 v10 acceptance closes
  both the engine/tooling defect and the live N/N-camera depth/floorplan gate.

Current read-only gates:

```bash
python3 DS9/scripts/validate_runtime_ownership.py
python3 DS9/scripts/validate_asset_manifest.py
```

Cutover gates intentionally fail until the above work is complete:

```bash
python3 DS9/scripts/validate_runtime_ownership.py --require-parity
python3 DS9/scripts/validate_asset_manifest.py --check-files --profile canonical \
  --require-provenance --require-realization \
  --artifact-root "$NOESIS_DS9_ARTIFACT_ROOT"
```

## Resolved At Source: Person-Ground State Parity

DS8 and DS9 now use the one runtime-neutral estimator in
`noesis/telemetry/person_ground_state.py` for posture, source hysteresis,
constant-velocity filtering, idle locking, and committed trail history. Each
runtime keeps only its SDK metadata/projection adapter. DS9 BEV consumes the
same public motion/posture/trail fields instead of growing a second path state.

The DS8-vs-DS9 characterization gate and existing ground-state regressions pass.
This closes the source/CPU-contract blocker without changing or restarting the
live DS8 process.

## Wholebody49 Source Port Complete; Runtime Quality Still Blocked

The shared `s` mask and `x` box profile contract now resolves runtime-owned
paths. DS9 owns matching templates, labels, a DS9-header-built parser, staged
ONNX sources, CLI/preflight wiring, manifest entries, and guarded TensorRT 10.14
build specs. CPU tests verify ONNX input/output shapes, parser entrypoints,
strict tensor/parser/mask semantics, and derived mask/bbox OSD policy.

Artifact realization is complete, but runtime parity is not. The DS9 `s` and
`x` engines exist, validate, and deserialize independently. The S engine is
25,327,772 bytes at SHA-256
`1fb95225e8258af13ac96de5136b85dadb60419a8c49058e70536eef06dd6bdf`;
the X engine is 108,579,580 bytes at SHA-256
`a5f4322d7e123461a1bbc64388b6f0e30c9359091e76ac7a95144648841138c3`.
The remaining exit is fresh tensor/parser/mask/OSD, throughput, resource, and
occupied-scene validation. A successfully serialized engine alone does not
clear this blocker.

The acceptance path is now executable without arbitrary launch input:
`run_canonical_runtime_container.py` exposes exact `wholebody49-s` and
`wholebody49-x` lanes, while `wholebody49_occupied_scene_smoke_test.py` requires
an occupied scene, error-free stats, zero core CPU-copy violations, and the
correct mask-versus-bbox sampling path. The v2 gate now also requires the exact
configured source inventory, a 30-second advancing-frame window, and the
provisional 2 fps / 2.5-second liveness floors. A separate typed gate proves
direct RTSP and WebRTC decoded frames, while the generic v2 300-second resource
gate binds the exact container/checkout/realization/primary-engine/config
cohort and enforces OOM, leak, cgroup, GPU-memory, and PID ceilings. These are
static/tested contracts until fresh live lane evidence is captured; an empty
house remains an honest blocking result.

## V3DT Source Port Complete; Runtime And World Quality Still Blocked

DS9 now owns a locked unpadded SV3DT pipeline, exact camera order, camera
inventory, camInfo files, tracker config, native bridge, source provenance,
runtime/preflight materialization, NvMOT engine helper, and bbox3d smoke. The
large BodyPose3DNet and tracker-ReID inputs and all three selected V3DT engines
are hash-verified under the explicit artifact root. Both isolated no-GPU engine
plans pass, and the NvMOT helper compiles against DS9 headers with warnings
treated as errors.

This is not a live V3DT claim. Engine construction and the static coordinate
correction are complete; the profile must still pass fresh bbox3d/native-meta
coverage, global-world-v2 replay, identity, throughput/GPU memory, and clean
shutdown. The active shared calibration has separated camera centers, the DS8
and DS9 camInfo sets are byte-identical, and DS9 now converts the locked `xzy`
tracker tuple before publishing Y-up `backend_world_m`. None of those static
facts may be inferred from engine startup or substituted for live evidence.

The acceptance launcher now has a hard-pinned `v3dt` lane with the exact V3DT
pipeline, cameras, `yolo26_seg/s`, six-engine realization set, container
boundary, and ordered lifecycle contract. The bbox and world clients now use
required bearer authentication. `v3dt_world_contract_smoke_test.py` now emits a
privacy-safe v2 source/report pair bound to the exact sealed launcher session,
runtime identity, and effective config hashes. It requires all three cameras,
per-camera continuity, bbox3d/visibility/native-foot coverage,
`backend_world_m`/`bbox3d`, old-raw-tuple rejection, floor consistency, native
image-foot reprojection at or below 40 px p95, and independent image-base
replay. Ownership rejects v1, skipped companion gates, stale sessions, and
mutated bindings. This closes the static/orchestration gap, not the pending
occupied live-evidence gap, and it makes no MV3DT overlap/fusion claim.

## Resolved At Source: Shared Identity-v2 Runtime

DS9 now mounts the same authenticated v2 API and constructs the same shared
store/runtime/whole-frame coordinator as DS8 and V3DT. The DS9 hook supplies one
detached source-frame batch, and overlap permits require fresh topology, world,
time, and appearance proof. Default shadow mode preserves legacy public labels;
authoritative mode cannot start without an artifact-backed scoring calibration.

DS9 now selects the same provenance-locked NVIDIA TAO Swin-Tiny profile as DS8:
RGB ImageNet preprocessing, direct `256x128` resize, dynamic batch `1..16`, raw
`fc_pred` tensor metadata, and a 256-dimensional normalized embedding. The
shared contract rejects parser properties and the active DS9 surfaces contain
no OSNet substitute.

This is source/static and artifact parity only. The selected DS9 TensorRT 10.14
Swin engine is realized, but resident enrollment/migration decisions,
calibration evidence, and a live DS9 shadow replay remain. Historical OSNet
StableID smoke evidence does not satisfy the selected Swin profile or identity-v2
quality gates.

## Resolved: Canonical World And Capability Health Parity

DS9 now uses the same shared observation normalization, artifact fingerprinting,
global world fusion, and capability monitor as DS8. Tracking telemetry includes
versioned observations and a world snapshot, emits the snapshot as a separate
WebSocket message, and advances `tracking_observations` and `global_world`
health only after compatible producer progress.

The selected native and engine artifacts are available. Static integration and
focused contract tests pass; neither is presented as a substitute for fresh DS9
runtime evidence under the current checkout and evidence matrix.

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

## Historical Core Gate Status (2026-06-16)

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
- Historical shutdown/native cleanup smoke on host MP4 input exited `0` after
  selecting the system DS9 `pyservicemaker`, but used the now-removed
  interpreter-exit workaround and therefore does not clear current acceptance.
  DS9 now requires repo-owned downstream-EOS acknowledgement, the expected EOS
  callback, Service Maker `wait()` return, normal `SystemExit`, and no native,
  GStreamer, wait-thread, or forced-termination signature.
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

- V3DT now has DS9-owned pipeline/camera/camInfo/tracker configs, a strict asset
  validator, external source staging, runtime materialization, native bridge,
  smoke entrypoint, and guarded BodyPose/NvMOT-ReID build plans. No root DS8
  engine, native binary, machine-local clip, or mutable SDK symlink is used.
- V3DT bridge behavior remains blocked on fresh runtime evidence even though its
  selected DS9 TensorRT 10.14 engines are realized and its source profile now
  publishes canonical `backend_world_m`. Promotion requires the same-session v2
  replay described above. MV3DT overlap/time-sync/peer-fusion acceptance remains
  a distinct later blocker.
- DS9 is runtime-standalone from the DS8 entrypoint, DS8 engines, and DS8 native
  extension binaries, but it is not yet a hermetic standalone repository. It
  still shares parent-repo application modules and data such as camera config,
  calibration/geometry helpers, WebSocket server code, and some validation
  clients. Treat this as packaging work, not a runtime fallback.

The 2026-07-10 post-reboot recovery audit now separates the satisfied driver
gate from the intentionally different host SDK. The installed DeepStream 9
README requires driver 590+, CUDA 13.1, and TensorRT 10.14.1.48; the active
driver is `595.71.05`, while host CUDA/TensorRT remain at the DS8-owned 13.0 and
10.13.3 versions by design. `ds9_preflight.py --env-only` therefore accepts the
driver and still rejects host TensorRT. The digest-pinned isolated image reports
DeepStream 9.0 / CUDA 13.1 / TensorRT 10.14.1, and all canonical, Wholebody49,
and V3DT no-GPU build plans pass. The canonical DS8 lifecycle gate is accepted;
actual builds still require an announced exclusive-GPU window, and each DS9
profile must pass its own runtime-quality gates before cutover.

Historical host cutover evidence from the 2026-06-16 UTC validation run (not
current readiness):

- The host install reaches DeepStream 9.0.0 / TensorRT 10.14.1.48 and starts the
  DS9 runtime with reused DS9 resources.
- Host validation now includes fresh-start gates, focused MP4 ReID, and
  occupied-camera live-RTSP production gates. The core production path is no
  longer blocked by missing public tracking/BEV or live shutdown evidence.
- Remaining caveat: V3DT source staging is complete, but runtime evidence is
  still out of scope until its two DS9 engines exist. Longer host RTSP soak
  evidence can be collected with
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
