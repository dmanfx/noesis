# MapAnything depth
_Status: canonical native DS9.1 manual-depth lane, updated 2026-08-15._

MapAnything is a first-class DS9 depth component, but it is not the only
depth-related path in the runtime.

## Roles

DS9 currently uses two distinct depth lanes:

- `models.mapanything`
  - Full-frame SGIE branch in the live DS9 pipeline.
  - On-demand and gate-controlled.
  - Owns the `depth_result` WebSocket payload and `get_ma_depth` RPC contract.
  - Persists dense full-frame snapshots below the native runtime root selected
    by `NOESIS_DS9_RUNTIME_ROOT`.
- `models.depth_tracking`
  - Full-frame Depth Anything V2 metric lane used by the active baseline
    world tracking.
  - Always on in baseline mode.
  - Does not publish a second full-frame depth WebSocket stream.
  - Contributes through `NOESIS.OBJECT_DEPTH` and the fused backend world
    estimator in `DS9/noesis/pipelines/hooks.py`.

MapAnything also acts as the offline reference source for the DAv2 room
registration artifact used by baseline tracking.

## Runtime Topology

MapAnything stays on its own DS9 branch:

- `main_tee -> mapanything_queue -> mapanything_valve -> mapanything_fullframe -> mapanything_rgb_convert -> mapanything_rgb_caps -> mapanything_fullframe_sink`

`mapanything_fullframe` accepts the canonical NV12/RGBA NVMM buffer and attaches
tensor metadata. The downstream GPU converter produces explicit NVMM RGB, and
the exact-capture buffer probe on `mapanything_rgb_caps` reads both that RGB
surface and the preserved tensor metadata from one buffer. The valve is upstream
of inference and conversion, so the branch remains idle outside a startup
negotiation prime or an explicit manual capture.

Config source:

- `DS9/config/infer.yaml`
- `models.mapanything.*`
- default `gie_id=2`

Gate control:

- REST: `GET /api/v1/depth/refresh?seconds=N`
- WebSocket RPC: `get_ma_depth` can open a short refresh window
- Runtime: the DS9 pipeline builder controls `mapanything_valve.drop`

The gate applies only to MapAnything. It does not control the always-on baseline
DAv2 tracking lane.

## Postprocess, Storage, and RPC

Live MapAnything processing is owned by `MapAnythingProcessor` in
`DS9/noesis/pipelines/hooks.py`.

DS9 has one tensor ownership path. The DS9-built
`noesis_depth_tracking_tensor_ext` walks public Service Maker frame user
metadata, requires exactly one tensor record with the configured `gie_id=2`,
requires exactly `depth`, `conf`, and `mask` at per-frame shape
`1x294x518`, and copies each frame-local tensor once. Python separately
requires a valid `FrameMetadata.batch_id` and configured source identity but
does not use either as a second tensor offset. The generic native capture
export and the
`NOESIS_DS9_ALLOW_NATIVE_TENSOR_COMPAT` switch are retired. The Python
`frame_meta.tensor_items` wrapper is not a MapAnything authority: the first
2026-07-11 baseline acceptance attempt exposed only sibling DAv2 UID `5` there
even at the MapAnything output, while the raw typed frame metadata remains the
SDK-owned source. This follows DS9 nvinfer's
`attach_tensor_output_meta`: it advances each frame's device/host pointers by
the batch position before attaching metadata while retaining per-frame
`inferDims`. Manually slicing again by `batch_id` would double-offset the
tensor and is forbidden by regression tests.

The frame-local MapAnything tensors are an intentional, exactly bounded
CPU-edge copy for dense snapshot/RPC construction: three float32 maps at
`294x518`, exactly `1,827,504` owned host bytes per captured frame. The native
selector must dereference nvinfer-owned tensor pointers synchronously while the
frame metadata callback owns their lifetime; no SDK-supported clone/retention
contract currently permits deferring that dereference to another thread. The
native bridge releases the Python GIL during each synchronous host copy, and
copy duration plus bytes are recorded under
`mapanything.exact_native_tensor_to_host` and the
`mapanything_probe_local_d2h_*` counters. Only the resulting owned arrays enter
the bounded asynchronous postprocess queue, so alignment, masking, storage,
and publication do not run on the Service Maker operator.

This is the sole documented callback-copy exception in the canonical graph. It
is justified by NVIDIA metadata lifetime, is request-gated and measured, and
must not be generalized into full-frame host staging or blocking work on an
always-on branch. See `../../docs/performance_invariants.md`.

The worker is a runtime-owned non-daemon resource. Shutdown first drains
external REST/WebSocket providers and stops WebSocket publication, then closes
depth admission and proves pipeline EOS/wait so no probe can enter. Worker
shutdown defensively waits any capture already holding the metadata-lifetime
lease, FIFO-drains every accepted job, appends a stop sentinel, and joins before
storage is released. A poisoned final job is surfaced
during that same teardown even when no later frame arrives. A full queue,
malformed layer/shape set, ambiguous UID, missing batch/source identity,
attachment failure, unresolved worker teardown, or worker error fails the path.
None selects a wrapper, raw-PyDS, source-ID-as-batch, or cached-depth substitute.

Every publishable result also crosses one mandatory transactional storage
boundary. The worker waits for its exact `WriteHandle` receipt before recording
the frame or constructing/publishing `DepthResult`.
`NOESIS_DEPTH_STORE_COMMIT_TIMEOUT_S` defaults to 30 seconds, requires a finite
value, and clamps to 0.1–60 seconds. Timeout or write failure poisons the
worker, signals the runtime failure callback, and publishes nothing. The
retired `NOESIS_DEPTH_STORE_ENABLED` switch cannot select a `memory://` path;
there is no direct-write or memory fallback.

Responsibilities:

- capture and decode the exact MapAnything UID/layer/batch contract
- align depth/conf/mask to camera frame geometry
- intersect the tensor validity mask with the calibrated per-camera fisheye
  dewarper-validity mask declared in `DS9/config/infer.yaml`
- store invalid/out-of-FoV depth as NaN and confidence as zero; an empty result
  is rejected instead of being expanded into a full-frame zero-depth product
- store dense snapshots through `geometry.depth_source.DepthStorageManager`
- publish `DepthResult`
- serve `get_ma_depth` via the runtime provider path

The full-frame `depth_result` and `ma_depth_response` contracts remain
MapAnything-specific. They are not reused for the baseline DAv2 tracking lane.

## Offline DAv2 -> MapAnything Registration

Baseline world tracking requires a prebuilt registration artifact
that maps raw DAv2 anchor range into MapAnything-aligned room range.

Canonical pieces:

- builder: `DS9/scripts/build_depth_registration.py`
- artifact: `DS9/config/depth_registration.json`
- schema/loader: `DS9/noesis/calibration/depth_registration.py`
- fitter: `DS9/noesis/calibration/depth_registration_builder.py`

Operational rules:

- The artifact is generated offline and loaded read-only by DS9 at startup.
- DS9 does not auto-generate, auto-refresh, or auto-download this artifact.
- Missing or stale entries are a fatal startup error in baseline mode.
- Empty-room RTSP captures are preferred, but the builder now filters samples to
  temporally stable pixels so minor/static occupancy does not automatically
  poison the fit.

Typical workflow:

```bash
bash DS9/services/mapanything_svc/run.sh
env CUDA_VISIBLE_DEVICES='' python3 DS9/scripts/build_depth_registration.py \
  --output DS9/config/depth_registration.json
timeout 45s python3 DS9/noesis/ds9_runtime.py --pgie-profile yolo26 --size m --disable-rest
```

The builder uses live RTSP sources from `DS9/config/infer.yaml` by default.

## MapAnything Service Notes

The local MapAnything service is still an active tool for offline registration
work. The deprecated part is the old live-runtime microservice/adapter depth
path; the canonical runtime depth path is the DS9 SGIE branch.

Current DS9 service ownership:

- startup script: `DS9/services/mapanything_svc/run.sh`
- app: `DS9/services/mapanything_svc/server.py`

The service is primarily used to produce reference dense depth for registration
builds, not to replace the DS9 in-pipeline MapAnything branch.

## What MapAnything Does Not Own

MapAnything is not the canonical owner of baseline person world tracking.

It does not:

- own `track.world` in baseline mode
- emit `NOESIS.OBJECT_DEPTH`
- replace the pose-first anchor chain
- act as an automatic fallback when DAv2 registration is missing; the dense
  reference-depth product is generated only by an explicit manual
  refresh/capture

Baseline room-relative tracking remains:

- pose-first anchor authority
- DAv2 object depth on the same pose ray
- optional offline DAv2 -> MapAnything registration correction
- one fused backend world estimator

## Quick Validation

- Source-contract checks (no live inference):
  `python3 -m pytest -q DS9/tests/test_mapanything_exact_native_capture.py DS9/tests/test_public_metadata_native_sources.py`
  - proves one native capture surface, exact UID/layers/batch ownership,
    exact probe-local copy bound/timing, fail-fast attachment, bounded async
    handoff, poisoned-final-job detection, and joined teardown
- Direct live contract gate against the managed native runtime:
  `python3 DS9/scripts/ma_depth_rpc_smoke_test.py --no-spawn`
  - manually requests one live `ma_depth_response` and validates its exact
    component descriptors and request identity
- `python3 DS9/scripts/build_depth_registration.py --help`
  - validates builder surface

Use one `scene_prior_only` floorplan request when PCF presentation changed, or
one fresh request when capture/storage changed. Do not rerun every camera or
start a second runtime unless the result is ambiguous.

The retained 2026-07-25 live gate passed all three configured cameras with
`1920x1080` `rgb8` evidence and cache-only zero mutation. It proves exact
capture identity and request behavior, not metric ground-truth accuracy.
