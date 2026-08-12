# DS8 Testing & Validation Guide
_Status: current as of 2026-07-12._

This guide describes how to validate DS8 changes using the current runtime and contracts.

For calibration, BEV, tracking-world, virtual-twin, or Menon-facing spatial
changes, also consult `plans/noesis_menon_validation/`. That workstream defines
the layered validation toolbox, report schema, cross-space acceptance gates, and
agent instructions for proving the full chain from camera pixels to Menon
reprojection.

Validation tiers:

- Tier 1: unit, schema, and math checks with no GPU or runtime.
- Tier 2: offline fixture and regression reports with visual artifacts.
- Tier 3: DS8 runtime or saved WebSocket telemetry from `noesis/ds8_runtime.py`.
- Tier 4: DS8 plus Menon cross-space traces or browser evidence.

See `plans/noesis_menon_validation/validation_tiers.md` and
`plans/noesis_menon_validation/validation_asset_inventory.md` for command
selection and reusable existing smoke tests.

## Canonical CPU contract gate

The hosted and local Tier 1 contract is:

```bash
python3 -m pip install \
  --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
python3 -m pip install -r requirements-ci.txt
bash scripts/run_cpu_contract_gates.sh
```

The gate requires Python 3.12 and runs generated-schema drift, AGENTS/docs
consistency, DS9 static ownership/artifact checks, the canonical Python suite,
the explicit DS9 suite, and whitespace checks. It intentionally excludes
DeepStream, CUDA, TensorRT, `pyds`, and hardware codecs. Runtime operator
modules remain importable for contract tests but raise if instantiated without
their installed Service Maker base.

The DS9 owner-RTSP orderly-EOS characterization is still collected by this
suite, but it reports a hardware skip when `/dev/nvidia0` is unavailable. That
case exercises `nvurisrcbin` and therefore belongs to the later exclusive-GPU
runtime tier; a CPU-gate skip never substitutes for its live DS9 evidence.

### DS8/DS9 person-ground and Wholebody49 parity

After changing person support-point/path behavior or the Wholebody49 profile,
run the GPU-free symmetry and contract gates:

```bash
python3 -m unittest DS9.tests.test_person_ground_parity -v
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 -m pytest -q \
  tests/test_person_ground_state.py \
  tests/test_v3dt_world_ground_state.py \
  tests/test_deimv2_wholebody49_assets.py \
  DS9/tests/test_wholebody49_profile.py \
  DS9/tests/test_engine_build_specs.py \
  DS9/tests/test_asset_manifest.py \
  DS9/tests/test_runtime_ownership.py
```

These gates prove shared algorithm/config/tensor/parser/OSD contracts, not live
DS9 quality. Follow `DS9/DS9_REBUILD_AND_SMOKE_GATES.md` for isolated parser
builds and exclusive-GPU engine/occupied-scene validation. Never reuse a DS8
engine to satisfy a DS9 check.

Private-state changes must also run:

```bash
python3 -m pytest tests/test_private_state_paths.py -q
```

That gate proves state code rejects shared/system directories, insecure files,
symlinks, and hardlinks without repairing their permissions. Real DS8/GPU
evidence is dispatched only through the labeled appliance hardware workflow;
a hosted CPU pass is never hardware readiness.

### Selector-driven appliance contracts

After changing deployment selectors, state releases, health binding, runtime
materialization paths, or the DS9 supervisor adapter, run the GPU-free contract
set before any release activation:

```bash
python3 scripts/export_noesis_core_schemas.py --check
python3 -m pytest -q \
  tests/test_appliance_binding.py \
  tests/test_appliance_health.py \
  tests/test_health_api.py \
  tests/test_internal_rest_auth.py \
  tests/test_inference_runtime_contract.py \
  DS9/tests/test_appliance_supervisor.py

(cd ../Menon && node --test \
  tests/appliance-deployment.test.mjs \
  tests/appliance-release-stage.test.mjs)
```

The cross-repository tests require byte-identical
`contracts/fixtures/v1/noesis_runtime_snapshot_vector.json`. They prove the
closed canonical JSON contracts, `noesis-runtime-v1` digest parity, private
selector/state paths, exact DS8 tuple or DS9 lane selection, selector-bound v2
health, and the no-fallback runtime dispatch. They do not activate a service,
start Docker, or use the GPU.

For DS9, `appliance-check --deployment-selector <file>
--selector-sha256 <digest>` is the final non-launching admission check. It
accepts no lane, image, state, lease, or fallback override. `appliance-run`
uses the same two arguments and is reserved for the deployment orchestrator,
which must already hold the selected release's full-lifetime shared lease.

### Shared observation, empty-frame, and depth-bridge contracts

After changing tracking publication, identity evidence linkage, strict
observations, DAv2 fusion, or native public metadata attachment, run:

```bash
python3 -m pytest -q \
  tests/test_identity_v2_service.py \
  tests/test_noesis_core_contracts.py \
  tests/test_noesis_core_world_service.py \
  tests/test_tracking_continuity.py \
  tests/test_analytics_telemetry_hook.py \
  tests/test_bev_renderer_world_smoothing.py \
  tests/test_v3dt_world_ground_state.py \
  tests/test_depth_tracking_frame_processor.py \
  DS9/tests/test_tracking_empty_frames.py \
  DS9/tests/test_world_snapshot_runtime.py \
  DS9/tests/test_bev_parity.py \
  DS9/tests/test_depth_bridge_rendezvous.py \
  DS9/tests/test_public_metadata_hook_batch.py \
  DS9/tests/test_semantic_observation_gate.py \
  DS9/tests/test_live_validation_runner.py
```

These tests prove immediate nonempty-to-empty publication plus bounded
advancing heartbeats, typed tracking/world/BEV sender receipts, exact ordered
cohorts, pre-admission rollback/retry, post-admission commit abort plus poison,
lifecycle/gate
commit after receipt, release-gated zero-delivery abort on journal/authority
failure, synchronous exact journal acknowledgement, non-mutating immutable
world reads, exact source clearing in the canonical world, the
explicit ROI-versus-camera-default zone authority contract (including DS8/DS9
producer parity and fail-closed canonical room derivation), the
persisted embedding-provenance all-or-none triad, truthful `depth_present`,
exact source/frame/PTS rendezvous before bounded lag, the 20 ms default/250 ms
hard wait bound, bridge/attachment counters, failure observability,
owning-batch native attachment, and fail-closed semantic report validation.
They do not prove that an occupied live home scene has exercised those paths.

#### Synchronous world-journal durability and cadence

The release-gated world journal uses storage contract v2: one process-owned,
RLock-serialized SQLite WAL connection with `synchronous=FULL`. The database,
WAL, and shared-memory files remain owner-private and inode-bound while open.
Startup verifies the exact schema, state row, retained sequence/hash chain,
canonical payload bytes, contract columns, and configured retention before any
append can be acknowledged. A valid legacy v1 rollback journal migrates once
to WAL and atomically records v2; a v2 database that is no longer WAL is
rejected without repair. Close requires an exact successful truncate-checkpoint
receipt and then rejects further use.

After changing the journal, canonical world service, or publication release
boundary, run:

```bash
python3 -m pytest -q \
  tests/test_noesis_core_journal.py \
  tests/test_private_state_paths.py \
  tests/test_noesis_core_runtime_world.py \
  tests/test_noesis_core_world_service.py
```

The three-source cadence regression warms three aggregate cycles and measures
31. Both median and arithmetic mean must remain below 27 ms, preserving the
approximately 12 Hz/source budget without treating one scheduler or storage
outlier as a semantic failure. A unit test deliberately has no maximum-latency
assertion: filesystem tail latency and sustained throughput belong to bounded
live-soak evidence, where the complete distribution and runtime health can be
evaluated together.

Against an already-running occupied DS9 baseline, the live bridge counter gate
is:

```bash
python3 DS9/scripts/ds9_bridge_contract_smoke_test.py \
  --ws ws://127.0.0.1:6008 \
  --duration 60 \
  --require-embedding-track \
  --auth-token-file "$HOME/.local/state/noesis/gateway-token"
```

It requires fresh DAv2 device frames, GPU object-depth ROI work, successful
object-depth attachment with `status=ok`, ReID extraction, a public embedding
track, and zero core CPU-copy violations. In an empty house the person-dependent
checks are blocked; they must not be waived into a pass.

### Engine-only runtime contract

Production startup consumes prebuilt TensorRT and native-extension artifacts;
it is not an engine or compiler workflow. Source-rich nvinfer and NvMOT files
remain available to explicit offline maintenance, but DS8 and DS9 derive atomic
runtime copies below `NOESIS_BUILD_DIR/runtime_inference/`. Those copies contain
the exact selected nonempty engine and no ONNX, ETLT, UFF, calibration, or
custom engine-builder inputs. Parser libraries may remain only when symbol
inspection proves they do not export TensorRT builder entrypoints.

Run the GPU-free contract gate after changing a model profile, graph builder,
tracker materializer, depth materializer, or native-extension preflight:

```bash
pytest -q tests/test_inference_runtime_contract.py
```

The gate proves DS8/DS9 parity, immutable source configs, atomic engine-only
outputs, source-absent startup materialization, missing/empty-engine rejection,
`force_engine_rebuild` rejection, NvMOT source stripping, and no subprocess or
native compilation from runtime checks. Missing or stale native extensions must
be built with their explicit repository build scripts before startup. Required
TensorRT plugins are selected by the runtime profile/engine contract; runtime
must not open ONNX bytes to decide whether a plugin is needed.

The current depth-registration v1 artifact predates engine-content hashing. Its
MapAnything profile binds normalized semantic fields and the engine path, not
the engine bytes. Validation tolerates the derived runtime config path and
cadence/source plumbing only; engine, name, batch, GIE, input, scope, and other
recorded semantic changes still fail. A future versioned builder should add
engine/config content hashes without silently reinterpreting v1 artifacts.

## Canonical live DS8 lifecycle gate

After graph, runtime-lifecycle, authentication, source, or GPU artifact changes,
run the bounded production canary on the DeepStream host:

```bash
umask 077
: "${NOESIS_DS8_CANARY_STATE_ROOT:?set a dedicated external canary state root}"
RUN_ID="baseline-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/ds8_runtime_30s_gate.py \
  --state-root "$NOESIS_DS8_CANARY_STATE_ROOT" \
  --duration-s 30 \
  --auth-token-file "$HOME/.local/state/noesis/gateway-token" \
  --log-path "$NOESIS_DS8_CANARY_STATE_ROOT/evidence/$RUN_ID.log" \
  --report-path "$NOESIS_DS8_CANARY_STATE_ROOT/evidence/$RUN_ID.json"
```

The state root is mandatory, absolute, owner-owned mode `0700`, outside both
the checkout and operator home, and neither inside nor above a sealed source
checkpoint. It must be empty on first use; the gate creates an exact private
cohort marker and thereafter accepts only that gate-owned layout. Reuse the
same root intentionally for a sequential baseline/V3DT identity-continuity
cohort, but give every log and report a fresh name. The gate never clears or
re-seeds gallery, resident, visitor, identity-v2, world, or scene databases on
reuse.

Before changing `HOME`, the gate captures and validates only the existing
owner-private camera registry, MapAnything key, and internal bearer **file
paths**. Secret bytes never enter the child environment, argv, or report, and
the bearer is load-only: a missing token is an error, not a request to create
one. The child receives a narrow OS/GPU environment plus canary-owned
`HOME`, Python user base, XDG roots, temporary/CUDA/GStreamer caches, generated
build configs, MapAnything depth/floorplan storage, analytics YAML/derived INI,
calibration/alignment copies, calibration audit, StableID/identity-v2 state,
world journal, scene database, virtual-twin root, and diagnostics. This is what
makes a canary from a sealed checkout non-mutating.

Because required runtime packages are installed in user-site directories on
the DeepStream host, the gate derives one absolute read-only `PYTHONPATH` from
the already active interpreter (including expanded `.pth` entries), excludes
the checkout and canary state tree, ignores inherited `PYTHONPATH`, and sets
`PYTHONNOUSERSITE=1`. This preserves dependency imports without restoring a
writable operator-home authority.

The 30-second interval begins only after authenticated REST and WebSocket
health, RTSP DESCRIBE, and required capability readiness succeed. The gate then
requires tracking/world sequence advancement and sends SIGTERM. A pass requires
all of these lifecycle markers in the listed order, exact process exit `0`, and
no forced kill:

- `Orderly pipeline EOS accepted:`
- `EOS received on pipeline (reason=shutdown_requested)`
- `pyservicemaker wait() returned (pipeline stopped)`
- `Shutdown complete`

The token file is owner-only authority state and must never be copied into a
command-line value, URL, report, or environment value. The runtime keeps RTSP
reconnect enabled; its repo-owned `noesiseos` element immediately after
`streammux` emits EOS asynchronously downstream and acknowledges the exact
request. Do not replace this proof with `Pipeline.stop()`, a timeout, SIGKILL,
or immediate interpreter exit. If acknowledgement, the EOS callback, clean
`Pipeline.wait()` completion, or wait-thread success is missing, teardown
deliberately fails closed; an exception from `wait()` is never accepted merely
because its thread returned.

Accepted engine-only lifecycle evidence is retained below appliance checkpoint
`/mnt/noesis_storage/checkpoints/20260710T203037Z-driver595-postboot/`:

- `ds8-runtime-v3dt-30s-engine-only-orderly-eos.log`: startup `8.104s`, active
  `30.0s`, tracking sequence `42→1590`, shutdown `1.467s`, exit `0`.
- `ds8-runtime-baseline-30s-engine-only-orderly-eos-rerun.log`: startup
  `10.569s`, active `30.0s`, tracking sequence `36→895`, shutdown `1.567s`,
  exit `0`.

Both passed without forced kill, error signatures, missing markers, residual
process/listener/GPU ownership, or generated inference source keys. Each active
nvinfer component reported engine deserialization and `Load new model` from its
`build/runtime_inference/...` config, followed by the required orderly EOS,
shutdown callback, Service Maker wait return, and `Shutdown complete` sequence.

Those accepted lifecycle runs prove startup, deserialization, sequence
advancement, and orderly shutdown. They are not occupied semantic acceptance
and must not be cited as proof of persisted embedding linkage, pose, usable
DAv2 depth, or backend-world agreement.

The final identity-continuity proof runs V3DT and then baseline without deleting
gallery data, resetting state, or waiting between runtimes. The accepted reports
are `/mnt/noesis_storage/checkpoints/20260710T203037Z-driver595-postboot/ds8-runtime-v3dt-30s-swin-shared-final.json`
(SHA-256 `2710f73c...d997`, startup `8.56s`, sequence `81→1789`, shutdown
`1.617s`, exit `0`) and `/mnt/noesis_storage/checkpoints/20260710T203037Z-driver595-postboot/ds8-runtime-baseline-after-v3dt-swin-final.json`
(SHA-256 `58db8a33...0010`, startup `10.565s`, sequence `29→774`, shutdown
`1.917s`, exit `0`). Both logs bind model SHA-256 `7a15727d...f6830` and
`fc_pred/256`.

Runtime credential and source-reference changes must additionally run:

```bash
python3 -m pytest \
  tests/test_runtime_secrets.py \
  tests/test_ma_service.py \
  tests/test_noesis_core_runtime_world.py \
  tests/test_depth_registration.py -q
```

These tests require no real secret values. Pytest provisions random owner-only
fixtures at collection and removes them at process exit. Live DS8/DS9 instead
use the appliance paths and provisioning workflow in `Runtime_Secrets.md`.

The first reusable fixture runner is available for GPU-free checks:

```bash
python3 scripts/noesis_validation_runner.py \
  --fixture-registry plans/noesis_menon_validation/fixture_registry.json \
  --fixture-id minimal_validation_fixture
python3 -m pytest tests/test_validation_toolbox.py -q
```

The runner writes JSON, Markdown, and visual artifacts under
`diagnostics/validation/<run_id>/`.
The fixture report now includes generated-scene coordinate-system checks and
room geometry-constraint checks when the fixture provides those fields.

### Transactional MapAnything snapshot storage

After changing depth persistence, fusion, retention, or runtime shutdown
wiring, run:

```bash
python3 -m pytest -q \
  tests/test_depth_storage_transactions.py \
  tests/test_depth_capture_event_adapter.py \
  tests/test_mapanything_postprocess.py \
  tests/test_depth_normals.py \
  tests/test_floorplan_agl_normalization.py \
  tests/test_floorplan_orientation.py
```

The transaction suite proves bounded admission without a direct-write
fallback, unique camera/timestamp reservations, immutable first-failure poison,
exact write/flush/shutdown receipts, atomic manifest-backed publication,
timestamp ordering under reordered workers, explicit legacy migration, and
read leases that keep fusion inputs alive during pruning. A stored path is
durable and safe to publish only after its `WriteHandle` yields a
`CommitReceipt`. Public depth results use the strict finite
`NOESIS_DEPTH_STORE_COMMIT_TIMEOUT_S` wait (default 30 seconds, clamped to
0.1–60 seconds); timeout or commit failure publishes no depth reference.
Runtime shutdown must reject new admission, inspect the flush receipt, and
require a completed `CLOSED` shutdown receipt; a timeout deliberately retains
live thread and queue references for another bounded drain attempt.
Capture-event consumers use `describe_snapshot()` for a validated immutable
camera/timestamp/write/content identity. Fusion fails if any requested member
is missing, corrupt, derived, mismatched, or pruned; its return evidence lists
the complete sealed source cohort and the exact derived commit identity.
Startup reports and ignores pre-contract trees without a manifest, but rejects
any tree that claims the commit contract and fails its manifest, hash, dataset,
or attribute validation.

The AGL regression drives both the storage-owned and direct MapAnything
floorplan generators. It requires finite offset metadata and proves the emitted
AGL raster is the result of one bounded low-percentile correction, preventing
both duplicate normalization and an undefined payload offset.

For a read-only A/B on an exact preserved raw cohort, run:

```bash
python3 scripts/mapanything_fusion_quality_report.py \
  --fused-snapshot /path/to/capture_event_fused.zarr
```

Repeat with `--without-scale-normalization` to compare the same source frames.
The report compares raw and robust-capped confidence weighting, temporal
median, and one-sample-trimmed fusion; it records eligible/retained support,
scale factors, temporal residuals, edge statistics, and confidence/error
correlation without changing storage or opening the depth gate.

For virtual-twin plane reconstruction changes that touch MapAnything normals,
run the focused unit/bundle checks before live Menon validation:

```bash
python3 -m pytest \
  tests/test_mapanything_normals_fusion.py \
  tests/test_virtual_twin_geometry.py \
  tests/test_virtual_twin_builder.py \
  -q
```

These tests prove that dense MapAnything depth-derived normals are persisted in
revision evidence, surfaced in `planes.json`, accepted when they agree with the
fitted stream plane, and rejected when they clearly contradict the plane.

Saved or live WebSocket telemetry can also be validated through the same report
schema:

```bash
python3 scripts/noesis_validation_telemetry_report.py \
  --input plans/noesis_menon_validation/minimal_telemetry.ndjson \
  --run-id minimal_telemetry

# Against a running DS8 WebSocket server:
python3 scripts/noesis_validation_telemetry_report.py \
  --ws ws://127.0.0.1:6008 \
  --duration 20 \
  --run-id live_tracking_bev_window
```

This emits tracking/BEV contract checks, motion checks, BEV/track agreement, a
copied telemetry NDJSON, and `tracking/track_audit.json` under the run directory.
The audit includes identity, current room, named world position, projection,
temporal, and ReID confidence, doorway/occlusion history, impossible-motion
events, speed summary, and warnings. The telemetry report also validates
occlusion bridges when `occluded` samples include uncertainty evidence.

For a measured Menon-scene waypoint walk, declare every calibration waypoint
with `expected_scene_xyz` and `split: fit|holdout`, then record the usual
owner-private alignment walk and mark arrival while the actor is paused. Use at
least three non-collinear fit waypoints and one holdout that is not used by the
solver. After capture verification, build the exact-frame calibration report:

```bash
python3 scripts/noesis_alignment_walk.py waypoint-calibration \
  --run-dir "$ALIGNMENT_WALK_RUN_DIR"
```

The report deterministically selects the assigned tracklet sample nearest the
marker without using the known waypoint to choose a frame. It binds camera,
run-local tracklet, frame, media PTS, image foot, explicit image size, K, E,
active similarity digest, raw and registered depth, raw floor/depth candidates,
prefilter/filter/final world points, and the measured Menon XYZ. It emits
per-stage fit and untouched-holdout metric errors and coverage, plus an
advisory-only similarity candidate. For each camera, complete FIT rows then
drive a fixed-center proper Wahba/Kabsch rotation solve from observed optical
rays to the known backend-world directions. The emitted candidate E preserves
the captured optical center exactly; degenerate direction sets and reflective
unconstrained solutions fail closed. Using that candidate E, the verified
monotonic piecewise fitter maps at least 32 real post-marker raw DAv2 samples
to physical camera optical Z without extrapolation. Both camera solvers exclude
HOLDOUT rows and report FIT and HOLDOUT angular, image-reprojection,
optical-depth, and reconstructed/producer position distributions separately in
`waypoint_camera_calibration_candidates.json`. All candidate artifacts are
digest-bound to the source capture, calibration, similarity, and solver inputs.
The command never writes active calibration.

For the strict inline floorplan BEV boundary, validate the selected runtime
configuration and exact-frame geometry directly:

```bash
python3 scripts/menon_bev_track_parity_smoke_test.py \
  --no-spawn \
  --ws ws://127.0.0.1:6008 \
  --pipeline-config DS9/config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --calibration-config config/camera_calibration.json \
  --alignment-config config/ply_alignment.json \
  --auth-token-file "$HOME/.local/state/noesis/gateway-token"
```

The gate requires the reviewed config and every BEV frame to declare
`camera_local_ground_m`/`camera_local`/meters, joins points to tracking by exact
camera/source/frame/observation time, validates tracker identities,
`displaySource`, bounds, and trails, and derives comparison points with the
appropriate independent world-to-camera transform, registered-depth
unprojection, or image-ray/floor intersection. It never subtracts
`backend_world_m` coordinates directly from camera-local BEV coordinates.

Captured Menon placement traces can be validated against the same report schema:

```bash
python3 scripts/noesis_validation_menon_trace_report.py \
  --trace plans/noesis_menon_validation/minimal_menon_trace.json \
  --run-id minimal_menon_trace

# When a live Menon checkout is required for acceptance evidence:
MENON_ROOT=../Menon python3 scripts/noesis_validation_menon_trace_report.py \
  --trace plans/noesis_menon_validation/minimal_menon_trace.json \
  --require-menon-root \
  --run-id minimal_menon_trace_with_checkout
```

This emits world-to-Menon transform, placement, floor-contact, trail-agreement,
BEV-to-Menon trail agreement, avatar scale/collision/orientation, timestamp, and
transform-audit checks. If Menon is required but no valid checkout is provided,
the report records blocked Menon evidence instead of treating a Noesis-only run
as complete.

When Menon is running in a browser, capture its live debug state and validate it
through the same Menon trace contract:

```bash
umask 077
MENON_VALIDATION_RUN_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/noesis/menon-tier4/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$MENON_VALIDATION_RUN_DIR"
python3 scripts/noesis_validation_capture_menon_trace.py \
  --url http://127.0.0.1:5175 \
  --storage-state "$MENON_PLAYWRIGHT_STORAGE_STATE" \
  --output "$MENON_VALIDATION_RUN_DIR/trace.json" \
  --screenshot \
  --validate
```

This writes `0600` raw snapshot, converted trace, optional screenshot, and
reports inside a new owner-owned `0700` directory. The storage-state file must
be regular, owner-owned, unlinked, valid JSON, and mode `0600`. Capture proves a
fresh same-origin `/api/auth/session`, binds the final page URL, and admits only
coherent current DS8/DS9 canonical state/entities, presentation/debug cursor,
render paths, active scene cohort, identity/lifecycle/timestamps/positions, and
one authored transform. Missing or mismatched evidence is failed or blocked,
not a substitute fixture pass.

To run the registered fixtures as a small regression suite:

```bash
python3 scripts/noesis_validation_regression_runner.py \
  --fixture-registry plans/noesis_menon_validation/fixture_registry.json \
  --run-id minimal_regression
```

This writes `regression_summary.json` plus per-case reports under
`diagnostics/validation/<run_id>/`. The registry can declare expected status,
minimum check counts, and maximum failure/warning/blocked counts; expectation
misses are reported as regression failures. Each run directory and descendant
directory is mode `0700`, every artifact is mode `0600`, and an existing unsafe
tree is rejected instead of repaired or overwritten.

## 1. Quick Sanity Checks

### DS8 runtime smoke test

Once `noesis/ds8_runtime.py` and `config/infer.yaml` are wired:

```bash
python3 noesis/ds8_runtime.py \
  --pipeline-config config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --enable-rest
```

YOLO26 segmentation profile (size defaults to `m` when omitted):

```bash
python3 noesis/ds8_runtime.py \
  --pipeline-config config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --pgie-profile yolo26_seg \
  --size m
```

Notes:

- The YOLO26 DS8 profile uses fused assets with a single output tensor (`output0`) and `disable-output-host-copy=1` to keep mask work on-GPU:
  `models/yolo26{n,s,m}-seg_fused.onnx` and `models/engines/yolo26{n,s,m}-seg_fused_b3_fp16.engine`.
- The fused YOLO26 engines compose masks for the top 30 detections (matching YOLO11) to keep ROIAlign cost bounded.
- Performance note: ROIAlign cost scales with the number of detections composed. With the unfused/top-300 path, `roi_align_proto` dominated YOLO26n (~18.6 ms) and drove very high GPU utilization; after capping to 30 detections, `roi_align_proto` dropped to ~3.8 ms and YOLO26n GPU usage in the DS8 pipeline dropped dramatically (reported ~10% on the nano model; exact numbers depend on stream FPS and host/GPU).

YOLO26-seg engines must be maintained explicitly; do not use runtime startup as
an engine builder. Review the GPU-free plan first:

```bash
python3 scripts/ds8_yolo26_seg_engine_maintenance.py --plan --sizes n,s,m
pytest -q tests/test_ds8_yolo26_seg_engine_maintenance.py
```

The guarded build command, exact DS8 TensorRT/CUDA/GPU profile, prior-engine
preservation contract, bounds, and false-positive `trtexec` load regression are
documented in `docs/DS8_yolo26_seg_engine_maintenance.md`. A real `--build`
requires an announced exclusive-GPU window and must be followed by the focused
runtime/parser/mask gate; never accept `trtexec` exit zero or its final `PASSED`
banner without positive deserialization markers and no TensorRT error signature.

Check logs for:

- Successful DS8 pipeline build (no missing elements or config errors).
- WebSocket server startup on the configured host/port.
- REST server startup (if enabled).
- Mosaic delivery toggles are consumed **at build time**: set `NOESIS_MOSAIC_WEBRTC_ENABLED=1` before startup to build the H.264 SHM output and WebRTC gateway. WebRTC does not enable or consume RTSP; `NOESIS_MOSAIC_RTSP_ENABLED=1` is an independent tooling-only output.
- If MapAnything SGIE gating is enabled, expect a startup log like `MapAnything gate primed; closed valve after 1.00s` (tunable via `NOESIS_MAPANYTHING_GATE_PRIME_SECONDS`) to confirm the SGIE branch prerolls and then closes when depth is disabled.
- Baseline non-`v3dt` startup now also requires a valid depth-registration artifact (`depth_registration.path`, default `config/depth_registration.json`). DS8 should fail fast before activation if the artifact is missing or stale for any enabled camera.

**Important:** Appliance DS8 tests should use the real RTSP camera inputs referenced
by configuration files and resolved through owner-only state:

- Prefer `config/infer.yaml` `sources` entries for DS8.

Do **not** put URIs in config or environment variables. Add a stable
`uri_secret` reference to public config and provision its value through
`scripts/provision_runtime_secrets.py`.

### Bounded production lifecycle canaries

Use the shared canary for an authenticated active window followed by a strict
orderly-EOS shutdown check:

```bash
umask 077
: "${NOESIS_DS8_CANARY_STATE_ROOT:?set a dedicated external canary state root}"
# Canonical DS8 baseline: YOLO26m detection + baseline tracking.
RUN_ID="baseline-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/ds8_runtime_30s_gate.py \
  --state-root "$NOESIS_DS8_CANARY_STATE_ROOT" \
  --log-path "$NOESIS_DS8_CANARY_STATE_ROOT/evidence/$RUN_ID.log" \
  --report-path "$NOESIS_DS8_CANARY_STATE_ROOT/evidence/$RUN_ID.json"

# Canonical DS8 V3DT lane: isolated V3DT runtime + YOLO26s segmentation.
RUN_ID="v3dt-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/ds8_runtime_30s_gate.py \
  --state-root "$NOESIS_DS8_CANARY_STATE_ROOT" \
  --runtime-profile v3dt \
  --duration-s 30 \
  --shutdown-timeout-s 90 \
  --log-path "$NOESIS_DS8_CANARY_STATE_ROOT/evidence/$RUN_ID.log" \
  --report-path "$NOESIS_DS8_CANARY_STATE_ROOT/evidence/$RUN_ID.json"
```

The profiles are fixed contracts rather than free-form runtime launchers. Both
require owner-token authentication, live `uri_secret` sources, advancing
tracking and global-world capability sequences, REST and WebSocket health,
SHM feeder first-AU readiness, decoded WebRTC media, synchronous orderly-EOS
acceptance, the expected EOS callback, Service Maker wait completion, exact
exit code zero, and no forced kill. The V3DT profile additionally pins
`noesis/ds8_runtime_v3dt_reimpl.py`,
`config/infer_v3dt_reimpl_fast1056_mp4.yaml`, the V3DT tracker, metadata
extraction, and the repo-built patched `nvtracker`. Despite the historical
config filename, its active sources are live secret references; an inline file
or network URI makes the gate refuse to start. The V3DT gate also refuses to
start unless `scripts/build_patched_nvtracker.sh` has produced the regular
repo-owned plugin at
`build/gst-plugins-deepstream/libnvdsgst_tracker.so`.
The canary forces exactly one warm gateway (`max=1`, `initial=1`), forces RTSP
off, assigns an isolated SHM socket, and requires the exact warm-capacity and
`MosaicH264ShmFeeder ready` logs. Its active probe runs
`scripts/webrtc_gateway_smoke_test.py` and requires RTP plus at least one
decoded frame; socket existence or an SDP answer alone is not media readiness.

### Build or refresh the depth-registration artifact

YOLO26 detection profile with the largest upstream checkpoint:

```bash
python3 noesis/ds8_runtime.py \
  --pipeline-config config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --pgie-profile yolo26 \
  --size x
```

Baseline pose+depth world tracking depends on a prebuilt DAv2->MapAnything
registration artifact. The canonical operational flow is:
- YOLO26 detection supports `n`, `s`, `m`, `l`, and `x` checkpoints. The `x`
  variant uses `models/yolo26x.onnx` and
  `models/engines/yolo26x_b3_fp16.engine` when materialized through the DS8
  runtime profile.

```bash
bash services/mapanything_svc/run.sh
env CUDA_VISIBLE_DEVICES='' python3 scripts/build_depth_registration.py \
  --output config/depth_registration.json
```

Notes:

- The builder uses the live RTSP sources from `config/infer.yaml` by default.
- Empty-room captures are preferred, but the builder now filters to temporally
  stable pixels so minor/static occupancy does not automatically invalidate the
  fit.
- Rebuild `config/depth_registration.json` when the registration inputs change:
  camera intrinsics or image size. Pose-only extrinsics edits, floor-height
  edits, and scene-unit updates do not invalidate the registration mapping
  because the fit is image-space DAv2 depth vs MapAnything reference depth.
- The runtime does not generate this artifact automatically.

### Registration-backed runtime smoke

After the artifact is built:

```bash
timeout 25s python3 noesis/ds8_runtime.py --pgie-profile yolo26_seg --size s --disable-rest
```

Expected:

- DS8 starts cleanly without a depth-registration startup error.
- `mapanything_fullframe` and `depth_tracking_fullframe` both load.
- The runtime enters the main loop and reaches timeout without registration
  validation failures.

### StableID pose integration (unit)

```bash
python3 -m pytest tests/test_stable_id_manager_pose.py
```

Notes:
- Rebuild `noesis_pose_meta_ext` before runtime tests that read pose meta.

### ReID Swin profile and DS9 cutover gate

Run the GPU-free source/config/tensor contract checks after changing ReID
configuration, metadata extraction, DS9 staging, or engine build specs:

```bash
python3 -m pytest -q \
  DS9/tests/test_reid_swin_profile.py \
  DS9/tests/test_engine_build_specs.py \
  tests/test_reid_tensor_contracts.py \
  tests/test_identity_v2_runtime_integration.py \
  tests/test_pipeline_build.py
```

These checks require exact TAO Swin-Tiny ONNX provenance, preprocessing,
`fc_pred/256` native extraction arguments, DS8/DS9 semantic equality, and no
active DS9 OSNet substitute. They do not validate a DS9 TensorRT engine or
identity quality. Build only in an announced exclusive-GPU window with
`DS9/scripts/run_canonical_engine_maintenance.sh --only reid_swin`, then require
manifest provenance, separate deserialization, and occupied-scene open-set
resident/visitor quality gates.

### Identity v2 calibration, API, and OSD gates

Run the complete GPU-free identity gate before changing authority, calibration,
enrollment, or migration review behavior:

```bash
python3 -m pytest \
  tests/test_identity_v2_calibration.py \
  tests/test_identity_v2_osd.py \
  tests/test_identity_v2_service.py \
  tests/test_identity_v2_runtime.py \
  tests/test_identity_v2_coordinator.py \
  tests/test_identity_v2_store.py \
  tests/test_identity_v2_resolver.py \
  tests/test_identity_v2_scoring.py \
  tests/test_identity_v2_migration.py \
  tests/test_identity_v2_cli.py \
  tests/test_identity_v2_migration_review.py \
  tests/test_reid_v2_api.py \
  tests/test_identity_v2_runtime_integration.py \
  tests/test_analytics_telemetry_hook.py -q
```

Also run `python3 scripts/export_noesis_core_schemas.py --check`. These gates
cover deterministic evidence/artifact bytes, disjoint split/stratum leakage,
insufficient-evidence refusal, person-worst benchmark and encounter-worst local
FAR/FRR/unknown/prior validation, active semantic/artifact pins, and calibrated
gallery-envelope refusal, loaded native/Python transform provenance, and the
separate public-runtime cutover artifact,
server-only enrollment keys, biometric-free migration review, exact OSD joins,
and DS8/V3DT/DS9 hook attachment parity.

The authoritative calibration gate is two-stratum and correlation-aware.
Artifact v2 rejects legacy v1 labels, datasets, and artifacts. A
provenance-locked subject-disjoint benchmark needs 50 resident/50 unknown
challenge-covered train people and 300 resident/300 unknown challenge-covered
holdout people. Outcomes are worst-case across encounters for each person. Its holdout exact
one-sided 95% FAR and misidentification upper bounds must each be no higher than
1%. A separately bound household dataset needs 10 known/10 unknown train and 20
known/20 unknown holdout encounters; each household partition must have zero
false accepts, zero misidentifications, FRR no higher than 35%, and no harmful
or rejection-rescuing resident-prior changes. Household tuning may only keep or
raise the benchmark rejection gates.

Observations are grouped deterministically by session/run/source/camera/tracker
and five-second bucket, capped at 300 per unit, and fit through bounded
person/encounter-balanced representatives. Outcomes aggregate first at
physical-encounter worst case and then at benchmark truth-person worst case.
Repeated frames, tracklets, windows, or encounters for one person never increase
benchmark confidence denominators. Household unknown people are split-disjoint
even when residents repeat across visits. Calibration and replay files are private
newline-complete bounded records; linked, shared-mode, truncated, or tampered
files must fail rather than be repaired in place. Identity evidence v2 also
requires contiguous sequence/previous-event links and an exact private
head/tail checkpoint, including after bounded contiguous-prefix pruning.

Every authority encounter exposes non-empty hard-allowed resident and visitor
competition plus an impostor; training also needs the genuine candidate.
Evidence rows bind the semantic digest derived from the active engine, model
YAML, nvinfer semantics, actually loaded ReID native bridge, and executable
Python transform/scoring components. Authoritative startup requires independent
artifact-byte and semantic pins. The calibrated gallery envelope is enforced
at startup and before mutations. Its declared authority scope is the open-set
scorer only, not whole-frame coordinator or occupied-scene authority.

Public promotion is a separate fail-closed gate. A byte-pinned
`noesis.identity.authority_cutover` v1 artifact must match the exact scorer,
model semantics, DS8/DS9 executable authority profile, topology, and camera set,
then verify two distinct owner-private report files: whole-frame coordinator
replay and occupied-scene runtime. Focused tests must cover a changed native
transform, missing cutover artifact, wrong pin/profile/runtime/topology, linked
or altered reports, and successful exact evidence. Unit fixtures do not satisfy
the real gate.

For a real calibration, obtain a provenance-reviewed benchmark scored by the
exact active model profile and collect owner-labeled score-only home evidence
from independent encounters, then follow
`plans/household_identity/calibration_and_enrollment.md`. Do not substitute the
unit fixture artifact for household evidence. Do not enable authoritative mode,
apply migration, or restart the live runtime as part of a unit/offline gate.

### Occupied semantic observation acceptance

The shared public contract has a fail-closed DS9 live gate. Against the running
baseline session, with a person deliberately visible and the exact owner-only
identity evidence file from that same session:

```bash
export SESSION="replace-with-supervised-runtime-session-id"
python3 DS9/scripts/ds9_semantic_observation_smoke_test.py \
  --ws ws://127.0.0.1:6008 \
  --pipeline-config DS9/config/infer.yaml \
  --duration 45 \
  --session-id "$SESSION" \
  --runtime-lane baseline \
  --runtime-instance-id "$RUNTIME_INSTANCE_ID" \
  --runtime-run-id "$RUNTIME_RUN_ID" \
  --identity-evidence \
    "$NOESIS_DS9_RUNTIME_ROOT/evidence/$SESSION/runtime/identity_v2.jsonl" \
  --snapshot-out \
    "$NOESIS_DS9_RUNTIME_ROOT/evidence/$SESSION/launcher/semantic-identity-evidence.jsonl" \
  --source-out \
    "$NOESIS_DS9_RUNTIME_ROOT/evidence/$SESSION/launcher/semantic-observation-source.json" \
  --out \
    "$NOESIS_DS9_RUNTIME_ROOT/evidence/$SESSION/launcher/semantic-observation.json" \
  --auth-token-file "$HOME/.local/state/noesis/gateway-token"
```

A pass requires one exact public-track to `noesis.observation.person` to private
hash-chain embedding anchor with a non-null immutable identity-v2 resident or
visitor. Pose, finite positive usable depth, and finite matching
`backend_world_m` may arrive on separate inference frames, but their complete
cohort must span no more than 1.5 seconds and retain the exact runtime run,
source, camera, tracker, identity-v2 subject/compatibility SID/resident UUID or
visitor generation, and calibration/model/config fingerprints. Any observed
tracker absence, null or conflicting identity, tracker-ID reuse, artifact
change, replay, or out-of-bound endpoint splits the cohort. Exact-frame evidence
is the zero-duration case. The checksum-covered source seals the exact
acquisition start/end. Every tracking frame, including an empty frame, must
remain inside the observed/capture bounds; capture timestamps may precede its
start only by the fixed two-second
receive/processing allowance and may not follow its end. A live partial capture
does not claim that its first publisher sequence or lifecycle generation began
at zero: their origins are explicitly externally unanchored, while every later
publication is contiguous within the window. A source's first received frame
may carry and count a tombstone whose predecessor predates attachment; every
later tombstone must repeat the actual in-window last published frame/time for
that lifecycle. Each canonical observation additionally proves
`observed_at_us <= published_at_us <= acquisition_finished_at_us`.

The v3 report exposes pose, usable-depth, and backend-world availability
independently from cohort acceptance, so a failed join does not falsely report
that an independently scheduled component never ran. It also seals a redacted
source transcript and exact private evidence snapshot for ownership replay;
raw embedding vectors and authentication material are forbidden, and a privacy
failure seals only a bounded redaction marker rather than the offending bytes.
Raw WebSocket input, reports, source replay, private snapshot rows, and ownership
replay all reject duplicate JSON keys before projection or model validation.
Missing people is `status="blocked"`, not pass. Serializer-only output, partial
provenance, nonfinite components, stale/private evidence, or pipeline errors
fail.

`DS9/scripts/ds9_live_validation_runner.py --lane baseline` includes this gate
when supplied `--identity-evidence`; for ownership-attached runs it writes the
semantic trio directly to the launcher directory. Baseline ReID promotion
requires both `reid_open_set_occupied_v1` and the lane-neutral
`semantic_gate_v3`. Post-gate validation passes only identity arguments to the
ReID validator and the reviewed pipeline plus exact identity snapshot to the
semantic validator. The report validator requires every check above and at
least one accepted bounded identity cohort. The implementation and GPU-free
tests pass, but no occupied live pass is recorded in this guide yet.

### Immutable scene-release boundary

Run the GPU-free release/store/API/builder gate after changing scene contracts,
virtual-twin release materialization, current-scene serving, or DS9 REST mounts:

```bash
python3 -m pytest \
  tests/test_noesis_core_scene_store.py \
  tests/test_scene_release_builder.py \
  tests/test_scene_release_adversarial.py \
  tests/test_scene_api.py -q
python3 scripts/export_noesis_core_schemas.py --check
```

This gate covers traversal, symlink/hardlink/non-regular substitution,
replacement during read, oversized manifests/assets/text, OBJ/MTL dependency
bombs, malformed UTF-8/directives, atomic no-replace failure, immutable tree
inventory, selected-file serving cost, and shared DS8/V3DT/DS9 router mounts.
The artifact-route spy must continue to show zero whole-cohort validations and
one selected-file verification per binary request. Promotion, `/current`, and
`/current/payload` must continue to validate the entire cohort.

For the preserved real candidate, also run the read-only store validation and
the Menon consumer validator recorded in `plans/spatial_os/validation.md`.
Neither command authorizes registration in the live store or promotion.

## 2. REST Endpoint Validation

### REST boundary serialization contract

After changing depth, analytics, ReID, FastAPI mounting, or boundary metrics,
run the CPU-only contract gate:

```bash
python3 -m pytest -q \
  tests/test_rest_boundary_metrics.py \
  tests/test_rest_product_boundary_coverage.py \
  tests/test_zero_copy_boundary_budget.py \
  tests/test_depth_api.py \
  tests/test_analytics_api.py \
  tests/test_reid_api.py \
  tests/test_reid_v2_api.py \
  tests/test_health_api.py \
  tests/test_scene_api.py \
  tests/test_virtual_twin_store_api.py
```

The adversarial probe must retain its response-model OpenAPI snapshot, 201
status, custom header, alias output, `exclude_none` behavior, and filtered
internal field while invoking FastAPI serialization once. Its measured payload
length must equal the actual returned body length. The rolling-window test uses
an injected monotonic clock to prove deterministic 10/60-second expiry and to
prove that a sparse 10 ms route fails max-path p99 even when 200 fast samples
hide it from pooled p99. Cardinality/high-rate cases additionally prove
overflow-max authority, hard 4096-sample caps, saturation drop evidence and
expiry, bounded ranked compact diagnostics, and tagged boundary-only error
counters. `tests/test_reid_ds9_parity.py` locks household resident/health source,
OpenAPI, and wire behavior across DS8 and DS9.
`tests/test_rest_product_boundary_coverage.py` additionally proves that all 46
mounted product routes use the measured route class, all 41 successful JSON
routes declare response-model timing, exactly five pre-rendered/file routes
declare explicit exemptions, and an unmarked success fails closed.

### WebSocket boundary and zero-copy collection contract

After changing WebSocket producers, stats collection, MapAnything RPCs, WebRTC
signaling, BEV/world publication, or any runtime shutdown sequence, run:

```bash
python3 -m pytest -q \
  tests/test_websocket_boundary_metrics.py \
  tests/test_noesis_core_world_service.py \
  tests/test_noesis_core_runtime_world.py \
  tests/test_zero_copy_boundary_budget.py \
  tests/test_zero_copy_stats_contract.py \
  tests/test_runtime_shutdown_contract.py \
  tests/test_mosaic_webrtc_gateway_shutdown.py \
  tests/test_mapanything_worker_lifecycle.py \
  DS9/tests/test_mapanything_exact_native_capture.py
```

The JSON contract is fail-closed: response assembly must be timed before a
dict reaches send, broadcast, or explicitly configured non-canonical
coalescing. Sync sends must freeze finite size-bounded JSON into immutable bytes
before returning a typed receipt, encode exactly once, and reject an entire
batch before admission when any member is invalid. A canonical tracking/world/
event batch remains behind a one-shot gate until synchronous journal/world
commit; commit failure aborts before delivery. Canonical tracking/world/event/
BEV never coalesces, even if a generic interval is injected for that type.
Tests also saturate the 256 MiB global frozen-byte budget, prove exact release
on success/failure/abort/cancellation, unresolved-gate shutdown blocking, and
zero bytes at shutdown, exercise the telemetry-client 1–16 clamp and capacity close,
and prove `/healthz` bypasses the telemetry set. Provider wait and network flow
control and authority-gate persistence dwell remain outside the 3 ms CPU
boundary; publisher/executor handoff,
conversion, the single admission-freeze encoding, and local send dispatch
remain inside it. The
three live zero-copy collectors reject missing p99/error-counter samples,
nonzero or growing serialization-error totals, pipeline errors, and over-budget
p99. Their reports retain first/final/max error totals and bounded, label-only
offender diagnostics without payload content.

The synchronous world-journal cadence regression models three serialized
sources at the pair-safe approximately 12 Hz/source cadence: six simultaneous
people on every camera must complete one aggregate prepare+journal/authority
cycle below 27 ms median. The separate 64-entity test is capacity stress for
fusion preparation, not a claimed steady-state journal workload.

#### Explicit synthetic lifecycle backend

`--stub` on the zero-copy stats or soak tools selects a pure-Python protocol
backend for fast API, WebSocket, response-boundary, and shutdown-harness tests.
It is not a DeepStream, Service Maker, GPU, media, inference, zero-copy, or
production-readiness run. DS8 selects it only with
`NOESIS_DS8_STUB_PIPELINE=1`; DS9 owns the separate
`NOESIS_DS9_STUB_PIPELINE=1` selector.

The backend models readable node properties, analytics reload receipts, a
strictly monotonic asynchronous `noesiseos` request, a typed EOS callback, and
a blocking `wait()` that returns only after that callback succeeds. Every stub
tool result and backend-selection log carries this exact evidence:

```json
{"backend":"synthetic_stub","native_runtime":false,"promotable":false}
```

That record is test evidence only. It cannot replace the canonical native
lifecycle markers or be submitted to DS9 ownership promotion. Native canaries
and the DS9 live runner scrub both stub selectors before launch and still
require real camera ingest, decoded WebRTC media, inference, GPU, EOS, and
Service Maker evidence.
Spawned stub tools pin the existing owner-only camera/key inputs by absolute
path, then relocate `HOME`, build output, analytics exclusion state, identity,
world journal, gallery, aliases, and depth output into one mode-0700 temporary
root. That root is removed after process exit, so the test cannot rewrite live
household identity or tracked analytics state.

The focused CPU check is:

```bash
python3 -m pytest -q \
  tests/test_synthetic_stub_lifecycle.py \
  tests/test_servicemaker_shutdown.py
```

For a three-second control-plane smoke, use
`python3 scripts/zero_copy_stats_smoke_test.py --stub --skip-cuda-preflight
--duration-s 3 --auth-token-file <owner-only-token-file>`. A passing result
still remains explicitly non-promotable. The DS9 equivalent additionally needs
its external artifact root so normal config and engine-only materialization
contracts remain testable:

```bash
NOESIS_DS9_ARTIFACT_ROOT=<private-ds9-artifact-root> \
python3 DS9/scripts/zero_copy_stats_smoke_test.py \
  --stub --skip-cuda-preflight --duration-s 3 \
  --auth-token-file <owner-only-token-file>
```

The 2026-07-11 isolated-state runs passed with DS8 `samples=4`, boundary p99
`0.724547 ms`, and DS9 `samples=4`, boundary p99 `0.739423 ms`. Both exited
`0`, used no forced kill, had zero boundary errors/zero-copy violations, and
contained every required synthetic lifecycle marker. These numbers are
control-plane regression evidence only, not native performance measurements.

### Depth API

With DS8 runtime running:

```bash
curl -X POST "http://127.0.0.1:8080/api/v1/depth/refresh?seconds=20"
```

If the standalone virtual-twin artifact API is already using `8080`, run DS8
with `--rest-port 8082` and use
`curl -X POST http://127.0.0.1:8082/api/v1/depth/refresh?seconds=20` for this check.

Expect a JSON payload matching `DepthRefreshResponse` and see depth bursts (depth telemetry and stored maps) during the enabled window.

### Analytics ROI API

The only canonical exclusion path is the repo-owned pre-tracker
`nvdsroiexclude` element. The REST layer writes the durable analytics YAML and
its derived INI as one fail-closed transaction, while the native element removes
object metadata before the tracker. Do not add a Python pruning hook or accept a
successful file write without the exact native receipt.

Run the GPU-free API, graph, shutdown-lease, and occupied-gate contract tests:

```bash
python3 -m pytest -q \
  tests/test_analytics_api.py \
  tests/test_pipeline_build.py \
  tests/test_runtime_shutdown_contract.py \
  tests/test_roi_reload_smoke_test.py
```

These tests cover strict ROI schema/bounds, disabled-empty versus enabled-
nonempty streams, atomic rollback, fatal ambiguous native commits, exact
receipt publication, DS8/V3DT/DS9 shutdown quiescence, mandatory restore, and
blocked unoccupied scenes. The size/restart contract is exercised by the
native and supervisor suites below.

Rebuild the SDK-owned binaries only with their exact-major build entrypoints:

```bash
# On the canonical DeepStream 8 host:
bash gst-plugins/build_nvdsroiexclude.sh

# Inside the pinned DeepStream 9 image/install:
NOESIS_DEEPSTREAM_HOME=/opt/nvidia/deepstream/deepstream-9.0 \
  bash DS9/scripts/build_nvdsroiexclude_ds9.sh
```

The DS8 and DS9 C++ sources are intentionally byte-identical owned mirrors, but
their CMake/build roots reject the wrong DeepStream major and produce separate
binaries. Validate both plugin origins, exact property surface, strict startup,
synchronous hash/sequence acknowledgement, prior-config retention after a bad
reload, disabled-empty policy, and Service Maker node access:

```bash
python3 -m pytest -q \
  DS9/tests/test_nvdsroiexclude_plugin.py \
  DS9/tests/test_preflight_plugin_origin.py
```

When the DS9 supervisor boundary changes, also run:

```bash
python3 -m pytest -q DS9/tests/test_runtime_container_boundary.py
```

That suite proves the appliance-persistent `persistent/analytics` pair is
seeded once, survives a new session, retains exact source coverage, accepts a
valid YAML larger than 1 MiB when its INI remains within 1 MiB, rejects corrupt
or partial state, mounts only the nested analytics directory persistently, and
captures private before/after session evidence.

The final acceptance gate requires a real person in the selected camera and an
authenticated running runtime (or lets the script spawn canonical DS8):

```bash
python3 scripts/roi_reload_smoke_test.py \
  --hot-restore \
  --camera <occupied-camera> \
  --auth-token-file "$HOME/.local/state/noesis/gateway-token" \
  --evidence "$HOME/.local/state/noesis/diagnostics/roi-hot-restore.json"
```

Use `--no-spawn` to attach to an already supervised runtime. The gate snapshots
the exact stage and native receipt, installs a temporary full-frame exclusion
for only the target stream, requires advancing real tracker frames to disappear
while video continues, requires `objects_removed_count` to rise, and restores
from `finally`. A pass additionally requires a second exact native receipt, the
original active hash and semantic REST stage, and the real person returning on
advancing frames. Exit `2` means the scene was unoccupied and is blocked, not a
pass. HTTP 200, a counter-only check, synthetic tracks, or an older hot-reload
smoke does not close this live gate. Keep the checklist item open until this
occupied restore evidence is captured for the target runtime family.

## 3. WebSocket Validation

With DS8 runtime active, connect a WS client (your UI or a small script) to the configured WS host/port.

Check for:

- `stats` messages at regular intervals with fields `prepared`, `activated`, `depth_enabled`, `depth_fps`, `errors`.
- Mosaic video rendered in the UI via WebRTC (not WebSocket JPEG frames).
- BEV frames and overlays updating in response to BEV config/overlay messages.
- Depth (`depth_result`) and tracking (`tracking`) messages present and well-formed.

### Depth drawer RPC gates (no log grepping)

These scripts exercise the WS RPC contracts used by `DepthDrawer` in `oai2-fe/`:

```bash
python3 scripts/ma_depth_rpc_smoke_test.py
python3 scripts/floorplan_rpc_smoke_test.py
```

Expected:

- `ma_depth_rpc_smoke_test.py` prints `[PASS]` and proves both cache-first and fresh depth retrieval (`served_from_cache=false` on the fresh call, and `ts_us` increases).
- `floorplan_rpc_smoke_test.py` prints `[PASS]` and returns a non-error `floorplan_response` with required fields (density/height/distance layers).
- The dashboard manual-refresh path must pass the successful fresh depth
  response's exact `snapshot_ref`, `snapshot_id`, and
  `snapshot_content_sha256` into `get_floorplan`. The floorplan response must
  repeat that identity with `exact_snapshot_reused=true`; this proves one
  capture cohort feeds both views and no second inference burst was opened.
- These RPCs validate only the gated MapAnything lane. Baseline DAv2 tracking
  depth is observed through `tracking`, `stats.payload.cameras[*].tracking.active_tracks[]`,
  and the on-screen `z=` label.

### SV3DT/MV3DT 3D meta (tracking payload)

This checks that `bbox3d` appears in tracking telemetry when SV3DT is enabled:

```bash
python3 scripts/sv3dt_meta_smoke_test.py
# Live RTSP validation:
# python3 scripts/sv3dt_meta_smoke_test.py --pipeline-config config/infer_v3dt_baseline.yaml
# then run DS8 with --tracking-mode v3dt and live RTSP sources in that config
```

Expected:

- `[PASS]` after 3D tracking is active (BodyPose3DNet assets + SV3DT tracker config required).
  - Tracking mode must be `v3dt` (the smoke test defaults `NOESIS_TRACKING_MODE=v3dt`).
  - The script defaults to `config/infer_v3dt_sample.yaml` (offline Retail02 clip) when present to avoid depending on live camera occupancy.
  - When testing live RTSP (`config/infer_v3dt_baseline.yaml` with real sources), ensure a person is visible; otherwise the script may report “no tracking messages observed”.

### V3DT Forensics Toolkit (Snapshot + Telemetry + Panel)

Use this when you need concrete calibration math + per-frame tracking metrics in one place:

```bash
# 1) Snapshot calibration + camInfo math
python3 scripts/v3dt_forensics.py snapshot --pipeline-config build/effective_pipeline_yolo11_seg.yaml

# 2) Enable per-frame tracking log (NDJSON) while DS8 runs
export NOESIS_V3DT_DIAG_LOG=1
export NOESIS_V3DT_DIAG_DIR="$HOME/.local/state/noesis/diagnostics"
python3 noesis/ds8_runtime.py --tracking-mode v3dt

# 3) Analyze the log (optionally pass the snapshot)
python3 scripts/v3dt_forensics.py analyze \
  --log "$NOESIS_V3DT_DIAG_DIR/v3dt_frames_<session>.ndjson" \
  --snapshot "$NOESIS_V3DT_DIAG_DIR/v3dt_snapshot_<timestamp>.json"

# 4) Generate the HTML panel
python3 scripts/v3dt_forensics.py panel \
  --snapshot "$NOESIS_V3DT_DIAG_DIR/v3dt_snapshot_<timestamp>.json" \
  --report "$NOESIS_V3DT_DIAG_DIR/v3dt_report_<timestamp>.json"
```

All artifacts default to the owner-only state directory above (directory mode
`0700`, files `0600`). Runtime NDJSON is bounded to 64 MiB per session with
eight retained sessions by default; use `NOESIS_V3DT_DIAG_MAX_BYTES` and
`NOESIS_V3DT_DIAG_MAX_FILES` only when a longer forensic capture is required.
Snapshot headers retain only non-secret runtime switches and public camera
source references. The panel server is loopback-only and serves panel HTML,
not snapshots, reports, or raw tracking logs.

See `docs/DS8_v3dt_forensics.md` for details.

### Auto-calibration RPC (wiring + contract)

```bash
python3 scripts/auto_calibrate_rpc_smoke_test.py
```

Expected:

- `[PASS]` output with a valid `auto_calibrate_result` message (contract only; calibration may still fail if depth is unavailable).
- `error` is not `no_handler` (regression guard).
- By default the script requests a non-existent camera id to avoid persisting to `config/camera_calibration.json`; to exercise the real calibrate-all path use `--calibrate-all` (may persist extrinsics).

Config:

- `NOESIS_AUTOCALIB_ENABLE_SECONDS` controls the depth burst when auto-calibration needs to open the depth gate (default: 10, clamped to 1–15).

### WebRTC mosaic (H.264 AU SHM → WebRTC gateway)

With `NOESIS_MOSAIC_WEBRTC_ENABLED=1`, DS8 encodes the GPU mosaic once and
publishes byte-stream, AU-aligned H.264 to `shmsink`. One
`MosaicH264ShmFeeder` proves the SHM control socket, PLAYING state, and first AU,
then fans complete AU copies to bounded per-peer `appsrc → rtph264pay →
webrtcbin` pipelines. This is the only RTP packetization step for browser
delivery. `NOESIS_MOSAIC_RTSP_ENABLED` defaults to `0` and controls only the
optional tooling branch.

Canonical `mosaic_output` quality/recovery defaults for the 3840x720 mosaic:

- `video_bitrate_kbps: 12000` (CBR)
- `h264_iframeinterval: 10`
- `h264_idrinterval: 10`
- `encoder: nvv4l2h264enc`

These values are strict: explicit zero/blank bitrate, encoder, GOP, preset, or
SHM-size values fail pipeline construction instead of being replaced with a
default. Retired `rtsp_*` H.264 keys also fail with a rename error.

Twelve Mbps is the initial quality floor rather than an assertion that eight
Mbps can never work. The 3840x720, nominally 30 fps composite combines three
independently moving scenes, OSD edges, and a short GOP, so 12 Mbps provides
useful motion headroom while the transport is being proven. Reduce it only
after occupied-scene appliance captures show equal quality and the delivered
bitrate tracks the encoder output.

On the installed NVIDIA encoder, `10/10` was verified to emit IDR NALs at
frames 0, 10, 20, 30, and 40. Do not restore `idrinterval=1`: with the same
encoder it emitted an IDR for every input frame. A separate hardware probe set
the natural GOP to `100/100`, advanced `noesisforceidr.request-sequence` after
output frame 14, observed the exact acknowledgement, and saw frame 15 become
the only additional non-delta frame in 60 outputs. Force-IDR therefore remains
effective in the SHM path without an RTSP sink.

ICE configuration (server-side `webrtcbin`):

- `NOESIS_MOSAIC_WEBRTC_STUN_SERVER` (optional; disabled by default for LAN-only operation)
- `NOESIS_MOSAIC_WEBRTC_TURN_SERVER` (optional; can include credentials, so avoid logging it)

The runtime never persists or logs full SDP, ICE candidates, DTLS fingerprints,
or TURN credentials. Use bounded state/payload/counter summaries in the normal
service log and pass an owner-protected captured offer explicitly to the replay
tool when SDP-level diagnosis is required.

Headless gate (proves SDP + RTP + decode):

```bash
NOESIS_MOSAIC_RTSP_ENABLED=0 NOESIS_MOSAIC_WEBRTC_ENABLED=1 python3 noesis/ds8_runtime.py --log-level INFO
python3 scripts/webrtc_gateway_smoke_test.py --ws ws://127.0.0.1:6008 --duration 5 --pt 103
```

Browser-offer replay gate (SDP-level; no ICE/DTLS required):

```bash
python3 scripts/webrtc_gateway_browser_offer_replay_test.py --ws ws://127.0.0.1:6008 --offer "$HOME/.local/state/noesis/webrtc_offers/offer_example.sdp"
```

Expected:

- Smoke test JSON includes `answer_video_direction="sendonly"`, `rtp_packets>0`, `decoded_frames>0`.
- Replay test asserts the returned answer is `sendonly` (guards against `a=inactive` answers for browser offers).

Workspace validation on 2026-08-08 exercised the complete transport and
signaling boundary with synthetic AU-aligned H.264. One client received 159 RTP
packets and decoded 158 frames; two simultaneous clients each received 160 RTP
packets and decoded 159 frames in five seconds. This proves SHM fanout,
per-client ownership, RTP packetization, ICE, and decode locally. It does not
replace the occupied-camera appliance checks below because the synthetic source
does not reproduce the production mosaic's motion complexity or NVENC rate
control.

Live appliance transport acceptance on 2026-08-09 promoted
`deploy-20260809-mosaic-shm-v2-ds9` on top of semseg Small/Large v12. The
AU-aligned SHM stream produced 240 AUs in 9.822 seconds (24.43 fps, 9.82
Mbit/s, 24 keyframes), confirming a realized 10-frame keyframe cadence. A
single peer completed ICE and decoded 307 frames in 10 seconds; two concurrent
peers decoded 221 and 220 frames. All six appliance units remained active,
gateway reset/retirement completed without a severe media signature, and no
listener remained on UDP 5400 or RTSP 8554. This proves the production
transport, fanout, and decode path, but not the subjective occupied-person
motion check in step 3 below.

If the dashboard shows ICE “connected” and `bytesReceived>0` but
`framesDecoded=0` / `videoWidth=0`, inspect the bounded feeder AU and gateway
frame/keyframe/RTP counters. Startup must contain `MosaicH264ShmFeeder ready`;
absence is a transport failure, not a signaling success. A peer drop rejects a
whole incoming AU for that peer and requests a new IDR. Receiving RTP with no
keyframes still indicates an encoder/event/caps fault and must not be hidden by
enabling RTSP. A gateway bus error or unexpected EOS must retire and stop only
that slot; healthy peers and the shared feeder remain live, and the next offer
may allocate a replacement within the configured capacity.

Before accepting the dashboard artifact fix on an appliance:

1. Run the bounded canary and require `ok=true`, `rtp_packets>0`, and
   `decoded_frames>0` in `webrtc_media`.
2. Confirm no listener or receive queue exists on the former localhost UDP
   handoff (`ss -ulnp` must show no Mosaic owner on port 5400).
3. Exercise the dashboard with occupied motion for at least two minutes and
   verify that moving edges do not leave sticky blocks across frames.
4. Test two simultaneous peers, disconnect/reconnect each peer, and verify
   decoded frames resume within the 10-frame recovery window.
5. Constrain one peer long enough to trigger whole-AU drops; verify the other
   peer and analytics continue advancing and the service log does not report
   mid-GOP/RTP queue drops.

### Mosaic aspect sanity (avoid stretched tiles)

The DS8 pipeline logs a structured tiler configuration event during build:

- `{"event":"ds8_mosaic_tiler_config", ... "square_seq_grid": true, ...}` in the normal service log.

Default behavior enables `nvmultistreamtiler square-seq-grid=true` to preserve per-tile aspect within a fixed 1920×1080 mosaic output. Override only if you explicitly want a non-square layout:

- Disable square tiling: `NOESIS_MOSAIC_TILER_SQUARE_SEQ_GRID=0`
- Explicit layout: `NOESIS_MOSAIC_TILER_COLUMNS=<N>` and/or `NOESIS_MOSAIC_TILER_ROWS=<N>`

## 4. Cross-Run Consistency Checks

For a curated set of test streams, run repeated DS8 sessions and capture telemetry for offline comparison.

### Suggested approach

1. Run DS8 session A for N seconds on a fixed set of streams and record:
   - WebSocket telemetry (e.g., via a client that logs `stats`, `tracking`, `depth_result`, `bev-frame`).
   - Any key logs about analytics and occupancy.
2. Run DS8 session B on the same streams and record the same data.
3. Compare:
   - Object counts and classes per frame (tolerate small differences if model configs differ, but investigate large discrepancies).
   - Zone occupancy over time.
   - Depth min/max distributions and burst timing.
   - BEV overlays/footpoints (visual and numeric.

## 5. Focused Component Tests

When changing a specific DS8 component, perform focused tests:

- **Pipeline config (`ds8_pipeline` / `infer.yaml`):**
  - Verify DS8 pipeline can start and process frames with no errors.
  - Check logs for any configuration warnings (paths, batch size, unique IDs).

- **Hooks (`noesis/pipelines/hooks.py`):**
  - Add temporary debug logging to confirm calibration-bundle
    intrinsics/extrinsics, MapAnything tensors, and analytics user meta are
    functioning. Validate exclusion separately through the native plugin and
    exact reload receipt; it is not a Python hook.

- **Telemetry (`noesis/telemetry/*`):**
  - Confirm WS messages have the expected `type` and field structure.

- **REST APIs (`noesis/server/*`):**
  - Exercise endpoints manually with `curl` or HTTP clients and validate error handling.

### MapAnything engine rebuild (ONNX → TensorRT)

If MapAnything depth outputs become sparse/mostly-zero (e.g., black heatmap), rebuild the ONNX export and TensorRT engine using the DS8-compatible `forward()` export path.

Export ONNX (torch-only; outputs `depth/conf/mask`):

```bash
python3 export_ma_onnx/export_to_onnx.py --repo external/map-anything --outdir /tmp/ma_onnx_out_forward --h 294 --w 518
```

Build TensorRT engine (this repo’s DS8 config uses `batch_size=3`, so the engine is built fixed-batch=3):

```bash
trtexec --onnx=/tmp/ma_onnx_out_forward/model.onnx --bf16 \
  --minShapes=images:3x3x294x518 --optShapes=images:3x3x294x518 --maxShapes=images:3x3x294x518 \
  --saveEngine=models/mapanything_depth/1/model.plan
```

Notes:

- If you change `config/infer.yaml` `batch_size` or the number of sources, rebuild the engine with matching shapes.
- TensorRT 10.13 has hit a Myelin internal error when attempting dynamic batch 1–3 for this model; fixed batch=3 is the validated path here.

## 6. Automation & CI

If you introduce automated tests (e.g., pytest), follow these principles:

- Keep tests narrowly focused on DS8 logic (metadata parsing, config normalization, simple dry-run pipeline constructs) that can run without GPUs.
- Avoid introducing heavyweight integration tests that require full DeepStream runtime unless the CI environment explicitly supports it.
