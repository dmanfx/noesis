# DS9 BEV and Capture-Event Integration Boundary

_Status: shared runtime contract implemented; this note does not assert live DS9 promotion._

DS9 uses the same capture-event, transactional storage, active-floorplan, and
BEV contracts as DS8 and protected V3DT. It must not regain a private BEV fork,
open a second camera reader, or answer a failed exact capture with stale data.

## Shared ownership

- `noesis/telemetry/bev.py` is the sole BEV renderer. DS9 extends the shared
  `noesis` namespace and has no local `bev.py` shadow.
- `noesis_core/capture_event_fusion.py` owns typed raw-only cohort admission,
  deterministic event identity, optional timestamped RGB association, and
  sealed source/result evidence.
- `noesis/depth_capture_event.py` adapts transactional depth storage without
  weakening that contract. It exposes committed raw descriptors and validates
  portable references, write IDs, timestamps, sequences, manifest/content
  digests, roles, and fusion levels.
- `noesis/capture_event_controller.py` is the single capture owner for depth,
  floorplan, and calibration callers. Runtime-specific providers in
  `noesis/capture_event_runtime.py` turn its outcomes into WebSocket payloads.
- `noesis_core/active_floorplan.py` owns one bounded, deeply copied record per
  canonical camera. The runtime and BEV renderer consume that registry; the
  metadata thread never generates a floorplan or opens depth.
- `noesis_core/depth_contract.py` owns registered-depth coherence, and
  `noesis_core/tracking_continuity.py` owns the pair-safe tracking/BEV cadence.
  DS9 and protected V3DT adapt their SDK metadata to those shared rules rather
  than redefining them.

The DS9 ownership validator also performs an isolated DS9-first import and
requires `noesis.telemetry.bev` to resolve to the shared renderer. Message-type
string matching is not ownership evidence.

## Capture transaction and admission

The graph has one process-wide MapAnything valve. A runtime therefore creates
one shared controller, not one controller per RPC or camera. Each request first
takes a non-blocking per-camera alias/capture lease and then one process-wide
gate lease. Same-camera and cross-camera contention return
`capture_event_busy`; cohorts cannot overlap through the global valve.

An admitted non-cache request performs one exact transaction:

1. Close the MapAnything valve, prove the MapAnything worker idle, flush the
   transactional storage frontier, and record the camera's latest raw
   timestamp.
2. Open one bounded depth burst, then close the valve and repeat the worker-idle
   and storage-frontier barriers.
3. Select only typed raw snapshots newer than that baseline and fuse one
   `capture_event_fused` / `intra_capture` cohort.
4. Revalidate the committed descriptor before releasing admission.

Every barrier has a stable failure code. A gate mismatch, invalid idle receipt,
unclean storage frontier, mutated descriptor, mixed/duplicate cohort, or fused
identity mismatch fails closed and enters runtime failure handling where the
code is classified fatal. No partial cohort becomes public.

## Exact response and cache semantics

A successful fresh depth response reloads the fused artifact by its portable
store-relative reference, transactional write ID, and content digest. It then
matches timestamp, role, fusion level, sequence, manifest digest, event ID, and
source write IDs against the sealed controller outcome. It never performs a
`latest` lookup after capture.

A non-cache floorplan request is always a fresh capture and is pinned to the
same fused descriptor; it cannot be satisfied by a generic latest/raw
floorplan. Exact generation
and its in-memory cache key include the fused write ID; the returned
`snapshot_ref`, `snapshot_id`, `snapshot_content_sha256`, and `snapshot_ts` must
match the controller outcome before the response can become active. An exact
generation failure returns its stable error code and never substitutes the
stale pre-burst payload or a newer unrequested snapshot.

Cache-only requests are resolved before capture admission:

- depth returns an already-valid payload or `no_cached_depth`;
- floorplan returns an already-valid memory/disk cache entry or
  `no_cached_floorplan`.

A cache-only miss does not read a depth snapshot, generate or persist a
floorplan, fill a cache, open the valve, query a frame provider, select/fuse a
cohort, write depth storage, or mutate the active-floorplan registry.
Exact floorplan success atomically publishes independent bounded in-memory
copies under its write-ID key and the explicit cache-only `latest` alias; it
does not overwrite the reusable generic disk cache.

Public failures are machine codes, not exception prose. In addition to cache,
shutdown, rate/capacity, and `capture_event_busy` errors, exact integrity uses
codes such as `exact_depth_identity_mismatch`,
`floorplan_snapshot_integrity_failed`, and
`active_floorplan_contract_failed`. A failed response carries no stale success
payload.

## RGB truth

There is no reviewed DS8/DS9 graph callback that offers a full RGB frame to
capture-event fusion. The current runtime therefore makes an explicit
depth-only request. Full sealed fusion evidence records
`rgb.status=not_requested`, while compact public evidence declares
`capture_mode=depth_only` and carries the full-evidence digest.

If a caller later requires RGB without a legitimate pipeline-owned timestamped
frame, the result is `rgb_frame_unavailable`. An RGB provider may only be fed
from a reviewed frame already present in the active GPU graph/native metadata
boundary. It must not add an appsink, CPU decode branch, source-locator lookup,
or independent camera open.

## Active floorplan, BEV, and world frames

Only a successful floorplan with bounded geometry, exact snapshot identity,
`frame=camera_local_ground_m`, `units=meters`, and the current calibration
fingerprint can update the active registry. Older versions do not replace the
record; a same-version conflict or malformed payload fails closed. The registry
supplies BEV bounds, grid shape/resolution, capture/floorplan timestamps, and
reviewed ray-to-floorplan alignment.

DS9 inline floorplan BEV is `camera_local_ground_m`. That is a display frame,
not a competing world model: canonical observations and the global world
snapshot remain `backend_world_m`, and world-to-local projection happens only
at the BEV boundary. The configured active-floorplan provider is the sole local
bounds authority; config bounds and auto extents are not substitutes.

Before a camera has accepted its first valid active-floorplan record, a provider
result of `None` records `floorplan_authority_state=startup_pending` and emits
neither a camera-local `bev-frame` nor an error `bev-status`. That bootstrap
state does not invoke the fatal callback and cannot satisfy active-floorplan
N/N acceptance. A malformed payload or provider exception fails immediately.
After readiness, any missing, malformed, stale, or conflicting authority is a
fatal `active_floorplan` failure with state `lost`; the renderer never reuses
last-known bounds or a last-known transform to hide it. Current homography or
publication failures are likewise fatal and update renderer health.

Every successfully emitted exact frame counts as renderer activity, including
an empty frame with `footpoints: []` and no trails. Such a camera is
`active_ready`. `inactive_ready` means no successful exact BEV frame has been
published yet (including a pre-authority startup camera); it is not an empty-
occupancy state.

Tracking and BEV are one exact publication pair when the renderer is active.
The effective gate interval is
`max(selected tracking interval, configured BEV interval)`, where the selected
tracking interval is the occupied or empty-heartbeat interval for that frame.
Tracking publishes first as one immutable finite-only ordered
tracking/world/event batch. Typed admission freezes and reserves its bytes, but
client delivery remains gated until synchronous exact-count journal and private
world authority commit. Lifecycle/tombstones and cadence advance after release.
BEV receives that exact
source/frame/time/sequence/submission cohort, returns an explicit `admitted`,
`startup_pending`, or `failed` receipt, and an admitted BEV submission ID must
follow the tracking batch. A pre-admission failure leaves authority unchanged
for exact retry; a post-admission authority failure aborts before delivery and
poisons publication. Count
transitions and tracker lifecycle/key-set changes bypass cadence and force the
pair; a tracking publication failure suppresses BEV. This rule is shared by
DS8, protected V3DT, and DS9. Sender admission is not a client delivery ACK.

Registered depth is eligible for canonical semantics and local BEV only when
`depth_status=ok`, `depth_registration_status=ok`, and
`depth_registered_m` is positive and finite. A present `depth_used_m` must also
be positive, finite, and equal within relative and absolute tolerance `1e-6`;
a mismatch rejects the depth instead of selecting a different interpretation.
Protected V3DT scales image anchors, candidate points, and bboxes from their
declared source image size into the active calibration `image_size` before BEV
projection, matching the DS8 and DS9 adapters.

Calibration mutation invalidates geometry explicitly. A camera-extrinsics
change clears that camera's active record and floorplan cache; a shared
alignment change clears every record and in-memory cache. Persisted caches bind
their calibration fingerprint and are rejected after a mismatch.

Runtime stats expose:

- `pipeline.bev.frame=camera_local_ground_m` plus `noesis.bev.health` v2,
  which separates renderer/config readiness, pre-first-success
  `inactive_ready`, exact-frame `active_ready`, authority-pending/ready/lost,
  and failed camera state;
- `pipeline.active_floorplan` with configured/active/missing cameras, rejection
  counters, and exact per-camera snapshot/calibration evidence;
- `pipeline.capture_event_fusion` with process-gate ownership, bounded counters,
  per-camera admission state, and the last fatal barrier code.

A missing configured-camera floorplan cannot satisfy the active-floorplan
health gate. An actual BEV render/publication failure or an active/fatally
failed capture controller is not healthy. Empty occupancy needs no synthetic
person, but the exact empty frame itself is a successful active render.

## Validation

GPU-free contract checks:

```bash
python3 -m pytest -q \
  tests/test_capture_event_controller.py \
  tests/test_depth_capture_event_adapter.py \
  tests/test_exact_snapshot_floorplan.py \
  tests/test_active_floorplan_registry.py \
  tests/test_bev_renderer_world_smoothing.py \
  tests/test_tracking_continuity.py \
  tests/test_v3dt_world_ground_state.py \
  DS9/tests/test_capture_event_fusion.py \
  DS9/tests/test_bev_parity.py \
  DS9/tests/test_runtime_ownership.py
```

These checks establish the shared code contract only. Live DS9 acceptance must
still pass the version-4 floorplan gate under the normal release gates:

```bash
python3 DS9/scripts/ds9_floorplan_live_gate.py \
  --ws ws://127.0.0.1:6008 \
  --pipeline-config DS9/config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --max-age-sec 120 \
  --session-id "$SESSION" \
  --runtime-lane baseline \
  --runtime-instance-id "$RUNTIME_INSTANCE_ID" \
  --runtime-run-id "$RUNTIME_RUN_ID" \
  --out "$EVIDENCE/mapanything-depth-quality.json" \
  --source-out "$EVIDENCE/mapanything-depth-quality-source.json"
```

The gate serializes fresh transactions for every configured camera, validates
the immutable snapshot and capture-event receipt (including
`rgb.status=not_requested`), and still requires N/N active-floorplan readiness.
BEV renderer readiness is checked separately: every exact frame, occupied or
empty, is active; a camera with no successful frame yet may remain
inactive-ready; and any actual failed camera rejects the gate. One cache-only
read per camera must leave the active registry and capture controller
unchanged. The bounded private source transcript makes those claims
independently replayable by the ownership validator. The promoted behavior ID
is `mapanything_depth_quality_v4`; pre-v4 reports do not satisfy this gate. This
document does not claim that live promotion has passed.
