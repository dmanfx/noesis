# Noesis DeepStream 9.1 Runtime

> **Historical snapshot:** this document records the superseded container-era
> transition state as of 2026-08-15. Current operation is native-host only; see
> `DS9/docs/README.md`.

Status: direct 9.0-to-9.1 upgrade in progress. The launch path remains
`DS9/noesis/ds9_runtime.py` -> `DS9/noesis/ds9_runtime_core.py`; DS9 executable
code does not import or spawn `noesis/ds8_runtime.py` or import DS8 preflight
helpers. The host driver `595.71.05` satisfies the DeepStream 9.1 floor, while
the SDK, CUDA, TensorRT, compiled artifacts, and runtime remain isolated in the
secondary-Docker boundary.

The source/toolchain pins are updated. The focused source suite passed 29 tests
with 1 skipped, and the four direct 9.1 pin assertions also pass. The exact 9.1
base has been pulled and inspected as image
`sha256:c41fa01c8657a7476a4b252261d9277f5117c33083c399a49b2a993ef9f6ac70`.
The derived image is not built, and no 9.1 native library, parser, plugin,
TensorRT engine, recorded-media smoke, or live-camera result is accepted. The
previous 9.0 realization
`6fab7d456c031490f640ee2c3ce5a38922a96ed86a965020ca3051820306dce4`
and its dated canary/performance results remain historical comparison and
rollback evidence only. They are not reusable 9.1 artifacts. Follow
`DS9/docs/deepstream_9_1_direct_upgrade_plan.md` for the active execution state.

AMC is deferred. MV3DT remains a separate disabled capability: the only future
edge is Kitchen/Family Room, Living Room has no MV3DT edge, and activation is
blocked on corrected Kitchen geometry plus synchronized occupied overlap,
peer-association, and fused-position evidence. The existing `v3dt` lane remains
SV3DT and must not be presented as MV3DT.

This folder is runtime-standalone from the DS8 entrypoint, DS8 engines, and DS8
native extension binaries. It is not yet a hermetic standalone repository:
the launcher still uses parent-repo app data and shared Python modules such as
`config/cameras.yaml`, calibration/geometry helpers, WebSocket server code, and
some smoke clients. Full DS8 option-surface parity is not yet certified where
V3DT is required: the DS9-owned source/config/build/runtime boundary and its
three selected engines are realized, but fresh runtime/world behavior evidence
is still required. The DS8 native pose metadata bridge remains unsafe under DS9
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
- Machine-readable ownership/capability matrix: `DS9/docs/runtime_ownership.yaml`
- Ownership drift gate: `DS9/scripts/validate_runtime_ownership.py`
- Owner-private dynamic evidence registry:
  `DS9/scripts/runtime_ownership_registry.py`
- Offline validated evidence recorder:
  `DS9/scripts/promote_runtime_ownership_evidence.py`
- Versioned artifact schema/manifest: `DS9/docs/asset_manifest.schema.json` and
  `DS9/asset_manifest.yaml`
- Separate immutable image authorities: `target.build_image` remains the exact
  TensorRT engine-build owner, while `runtime.image` identifies the exact
  two-layer runtime derivative used by the canonical supervisor. Runtime-only
  package changes never rewrite engine build evidence.
- Artifact staging/provenance gate: `DS9/scripts/validate_asset_manifest.py`
- Canonical observation/world fusion: shared `noesis_core/runtime_world.py` and
  `noesis_core/world_service.py`, fed by DS9 public tracking rows.
- Capability-progress health: `GET /api/v1/health/capabilities`, backed by
  successful canonical observation and world-snapshot publication.
- Atomic owner-governed scene releases: shared `/api/v1/scenes` registration,
  promotion, rollback, current-release, payload, and history routes.
- Shared household identity-v2 service: DS9 uses the same process-owned
  store/runtime/frame coordinator and authenticated `/api/v2/reid` router as
  DS8/V3DT. Its hook contributes one complete source-frame primitive batch;
  product scoring and visitor/enrollment state are not copied into `DS9/`.
- Shared person-ground estimator and BEV renderer: posture, source hysteresis,
  human motion filtering, idle lock, path commit, floorplan display selection,
  smoothing/trails, diagnostics, and BEV failure health stay single-sourced.
  DS9 owns its Service Maker metadata/projection adapter and runtime
  composition, including the active-floorplan registry/provider and fatal
  failure callback. Forced DS9 import resolves the canonical BEV owner; DS9
  does not carry a renderer fork.
  The DS9 adapter owns the same constant-velocity prior wrapper used by DS8 so
  prefilter diagnostics and accepted shared-filter world updates cannot be
  lost to a missing SDK-local adapter method.
- Wholebody49 `s` masks and `x` boxes: one shared semantic materializer feeds
  DS8 and DS9, while DS9 owns its templates, labels, parser binary, ONNX staging,
  engine paths, builder, preflight, and OSD policy. The S builder uses
  optimization level 0, 6 GiB WORKSPACE, and 2 GiB TACTIC_DRAM. Sealed 25 ms
  NVML evidence proved that both 8 GiB and 6 GiB WORKSPACE builds still consume
  about 9.48 GiB of full-device memory, so WORKSPACE is treated as tactic
  admission rather than a process-peak control. The reviewed S outer guard is
  therefore 10,000 MiB, leaving more than 2 GiB free on the 12 GiB target GPU;
  X retains optimization level 3, 4 GiB WORKSPACE, 2 GiB TACTIC_DRAM, and an
  11,000 MiB guard. Committed transaction `20260711T152338477896Z` realized the
  25,327,772-byte S engine at SHA-256
  `1fb95225e8258af13ac96de5136b85dadb60419a8c49058e70536eef06dd6bdf`
  with 2,738 samples, a 27.742316 ms maximum gap, and 9,481 MiB maximum
  observed. Transaction `20260711T152552934749Z` realized the 108,579,580-byte X
  engine at SHA-256
  `a5f4322d7e123461a1bbc64388b6f0e30c9359091e76ac7a95144648841138c3`
  with 12,750 samples, a 30.858598 ms maximum gap, and 1,017 MiB maximum
  observed. Candidate and installed-path deserialization passed for both; live
  mask/box runtime quality remains a separate gate.
- DS9-native SV3DT profile: DS9 owns the pipeline, camera inventory, locked
  camInfo set, tracker config, native bridge, source provenance, NvMOT engine
  helper, fail-closed preflight, and smoke entrypoint. Large source/engine bytes
  resolve through `NOESIS_DS9_ARTIFACT_ROOT`; the repository keeps only virtual
  `DS9/models/...` paths and provenance. All three selected V3DT engines are
  realized; fresh runtime-quality and world-contract evidence remain required.
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

- DS9 shares the fail-closed inference boundary with DS8. Every active nvinfer
  and NvMOT node receives an atomic engine-only config below
  `NOESIS_BUILD_DIR/runtime_inference/`; model sources remain offline
  maintenance inputs and are neither required nor passed to production
  plugins. Missing/empty engines and missing or provenance-mismatched native
  extensions stop startup instead of triggering ONNX export, `trtexec`, native
  compilation, or an SDK engine rebuild. Before any native import, DS9 verifies
  the exact Python ABI filename plus every declared source-bundle and binary
  SHA-256 against `DS9/asset_manifest.yaml`; filesystem timestamps are not
  treated as build provenance.

The DS9 launcher sets these variables before importing the DS9-owned Service
Maker runtime core. It also pins `pyservicemaker` to the DS9 system install
under `/usr/local/lib/python3.12/dist-packages` so a stale user-site wheel cannot
shadow the DeepStream 9 binding. It refuses DS8 installs through
`DS9/scripts/ds9_preflight.py`.

## Ownership And Artifact Gates

Run the governance gates without loading DeepStream or disturbing DS8:

```bash
python3 DS9/scripts/validate_runtime_ownership.py
python3 DS9/scripts/validate_asset_manifest.py
python3 -m unittest discover -s DS9/tests -v
```

The tracked ownership matrix contains static contract truth and repository
assertions only. Realized and runtime selectors are promoted into the fixed
hash-chained registry below an explicit private runtime root; missing current
promotions remain effective blockers even when the implemented surface has
static `parity` status. A registry promotion cannot override a real tracked
`known_gap` or `blocked` status.

The non-strict ownership gate validates that every copied/shared/adapter module
is classified and that DS9 executable code has no DS8 runtime/preflight import.
It reports effective static/evidence blockers as warnings. Cutover work uses strict mode:

```bash
python3 DS9/scripts/validate_runtime_ownership.py --require-parity
python3 DS9/scripts/validate_asset_manifest.py --check-files --profile canonical
python3 DS9/scripts/validate_asset_manifest.py --check-files --profile full --require-provenance
```

Strict failures are expected until the recorded parity gaps are closed and the
DS9-native artifact set is rebuilt, checksummed, and given build provenance.

## Configuration Decisions

- 2026-08-12: A camera-bound prior-conditioned fusion (PCF) Scene Prior is the
  canonical presentation source for the oai2-fe Depth drawer. Explicit
  `scene_prior_only` floorplan requests bypass DS9 static capture and caches and
  derive the Heatmap diagnostics, textured floorplan, and established four 3D
  representations from that immutable prior. Manual Refresh retains the normal
  fresh static-frame capture path as comparison evidence, but the frontend
  validates and discards its response from visible drawer state so it cannot
  replace PCF.

- 2026-07-11: DS9 depth, analytics, and ReID REST routes share the same
  single-render metric boundary as DS8. Handlers retain their declared FastAPI
  response models and mark model-assembly timing only; the route wrapper
  observes the completed response bytes after FastAPI's one real render. Exact
  returned byte length, true monotonic 10/60-second windows, and the worst
  budgeted path p99 replace the former duplicate surrogate JSON timing. This is
  wire-neutral and the DS9 implementation remains a DS9-owned source mirror.
  All metric sample/detail/error stores are hard-bounded; live saturation fails
  closed until expiry, and the compact stats getter retains only eight ranked
  budget paths. The previously missing v1 household resident CRUD and identity
  health routes are now exact source/OpenAPI/wire parity with DS8.

- 2026-07-12: DS8 and DS9 share one bounded WebSocket sender/fanout contract.
  The 256-submission queue also caps exact frozen in-flight bytes at 256 MiB,
  releases reservations on every completion/failure/explicit abort/cancellation,
  and requires zero bytes at shutdown. Canonical tracking/world/event batches
  remain release-gated until synchronous exact-count journal acknowledgement
  and private fusion commit; aborted gates begin no client delivery.
  Authenticated telemetry clients default to 8 and are
  hard-clamped to 16; excess connections close with `1013` /
  `telemetry_capacity_reached` before snapshots, while `/healthz` remains
  available and does not consume a telemetry slot.

- 2026-07-11: Occupied baseline evidence made tracker continuity and household
  parity executable invariants. Once identity-v2 accepts a subject for a
  camera-local tracker, alternatives are hard-masked until a real evidence gap
  expires that state; the locked subject is still rejected unless it passes all
  open-set gates. The v2 service now owns exact-frame `embedding_present` and
  persisted provenance instead of inheriting legacy cache diagnostics. DS9 also
  applies and verifies the same default household StableID policy as DS8, so
  legacy pressure auto-merge and blanket multi-camera activity remain disabled
  even if old environment knobs request them.

- 2026-07-12: The unpromoted semantic-v2 draft was superseded by semantic v3.
  DS8, protected V3DT, and DS9 now stamp every processed tracker with a
  lifecycle generation, force tracking publication on tracker-set transitions,
  publish exact tombstones, and expose publisher-owned contiguous per-source
  sequences. Those sequences now use typed bounded sender admission plus a
  one-shot authority gate: a rejected admission reuses the exact candidate,
  while journal/authority failure aborts before delivery and poisons the
  publisher. DS9
  acceptance binds both capture and observation time, requires
  registered depth only, uses the canonical runtime-world artifact roles, and
  seals a strict semantic projection that rejects embedding-shaped public data.
  The live runner exactly replays the report, projection, and private identity
  snapshot; fresh occupied promotion evidence remains pending.

- 2026-07-12: Runtime behavior evidence is immutable per session. Identity,
  floorplan, semantic, V3DT world, Wholebody occupied, and decoded-media
  producers now use the same owner-private create-if-absent publication path,
  publish sources before reports, and reject any existing or interrupted
  output cohort instead of replacing or repairing it. Semantic report/source
  bytes and each identity JSONL line are canonical. Ownership independently
  binds Wholebody source inventory and every V3DT session/build path to the
  reviewed canary and inspected mount. `runtime_session` promotion remains
  bounded ephemeral-canary evidence; `appliance-run` evidence is explicitly a
  separate, non-promotable lifecycle.

- 2026-07-11: DS9 tracking no longer drops zero-person frames before the
  WebSocket/world boundary. A count transition to zero publishes immediately;
  sustained emptiness carries an advancing camera frame ID at the shared
  default 2 Hz heartbeat. Empty frames clear that exact source's canonical
  world evidence without resetting observation ordering, matching DS8 and
  V3DT while avoiding frame-rate empty journals.

- 2026-07-11: DS9 camera-local BEV now has the same exact authority and
  publication boundary as DS8 and protected V3DT. Before the first valid active
  floorplan, a missing record is healthy `startup_pending` and emits no local
  BEV; malformed authority and every post-ready loss are fatal, with no bounds
  fallback. Empty exact frames are successful renders. Tracking publishes first
  at the pair-safe `max(tracking interval, BEV interval)` cadence, and BEV is
  emitted only from the matching typed receipt; both messages carry the same
  source/frame/time/sequence cohort and the BEV sender-admission ID must follow
  the ordered tracking/world batch. Count/lifecycle changes force the pair.
  Registered-depth coherence and calibration-image scaling are shared
  across all three producers. The replayable promotion contract is
  `mapanything_depth_quality_v4`. A 2026-07-25 isolated canonical-container
  session passed fresh exact-RGB capture and cache-only zero mutation for all
  three configured cameras. That validates the capture contract, but it was
  not a production release promotion and did not measure absolute depth
  accuracy; those acceptance decisions remain separate.

- 2026-07-11: MapAnything tensor ownership is one DS9-native exact path. Live
  baseline evidence showed the Python Service Maker tensor wrapper at the
  MapAnything output carrying only the sibling DAv2 UID, so DS9 no longer
  treats wrapper metadata or an opt-in generic native reader as a capture
  alternative. The DS9-built bridge requires one raw UID 2 record and exact
  per-frame `depth/conf/mask` shape. Python validates public batch/source
  identity, but native capture copies the already frame-offset tensors once;
  DS9 nvinfer applied the batch offset before attaching frame metadata. The
  probe-local copy is exactly three `294x518` float32 maps (`1,827,504` bytes),
  timed, and performed while the nvinfer-owned pointer lifetime is valid; the
  bridge releases the Python GIL during D2H. A post-inference GPU conversion
  produces explicit NVMM RGB; the `BufferOperator` attached to
  `mapanything_rgb_caps` extracts that batch surface and the preserved tensor
  metadata from the same buffer. Owned arrays enter a bounded asynchronous
  postprocessor whose non-daemon worker is drained and joined by runtime
  shutdown. Missing or duplicate UID state, alternate layers/shapes,
  attachment failure, queue saturation, final-job poison, or unresolved worker
  teardown fails closed. Fresh inference remains manual-only; passive and
  cache-only requests do not open the valve. DS8 remains unchanged because
  this defect was observed only in the DS9 wrapper surface.

- 2026-07-11: DS9 native startup uses content provenance, never source/output
  modification-time ordering. A fresh release worktree exposed the old mtime
  heuristic by rejecting binaries whose source and output hashes exactly
  matched the reviewed manifest. DS9 now attests all six extensions before
  import using strict manifest parsing, exact source membership, validator-
  identical aggregate hashing, the active CPython ABI filename, one unambiguous
  regular output, and stable owner-only file identities. A controlled no-GPU
  rebuild reproduced five outputs byte-for-byte and independently demonstrated
  NVCC salt nondeterminism in the CUDA/NPP tensor binary, confirming that a
  blind rebuild is not a stronger identity contract.

- 2026-07-11: TensorRT engine-build provenance and live service dependencies
  use separate immutable image identities. `DS9/docker/Dockerfile` and
  `noesis-ds9-dev:9.0-20260710` remain the build authority for existing engine
  evidence. `DS9/docker/Dockerfile.runtime` starts from that exact image ID and
  adds the Ubuntu PyGObject/GstWebRTC bindings required by the canonical
  gateway. The supervisor validates exact IDs plus the 70-layer parent / 72-
  layer runtime prefix before launch. The manifest realization may rebase its
  manifest anchor for this exact runtime-image adoption only after proving all
  engine records and build authority unchanged.

- 2026-07-10: Canonical behavior acceptance no longer treats one repeated
  numeric stable ID or the first successful floorplan RPC as semantic proof.
  The DS9 identity gate authenticates health before/after, pins shadow mode,
  requires fresh server-produced ReID observations plus tracker-subject
  continuity, keeps public authority blocked, and records cross-camera and
  open-set mechanism observations separately from unlabeled accuracy. Either
  observation can be made mandatory for a staged/natural session without fake
  resident labels. The floorplan gate requires fresh, non-empty, contract-valid
  grids for every active configured camera.
- 2026-07-10: The canonical live-validation bundle has no anonymous client
  lane. Its WebRTC, ReID, BEV, native-bridge, floorplan, MapAnything, and
  zero-copy clients accept `--auth-token-file`, default through
  `NOESIS_INTERNAL_AUTH_TOKEN_FILE`, load existing owner-only state read-only,
  and send Bearer authorization only in HTTP or WebSocket handshake headers.
  The runner passes only the path; unavailable credential state fails the gate
  before client traffic. Tokens never enter URLs, command lines, or reports.
- 2026-07-12: Repository validation helpers are an explicit Python package.
  Without `scripts/__init__.py`, Python's namespace-package resolution allowed
  an unrelated user-site package also named `scripts` to shadow
  `scripts.internal_auth_client` when gates were executed by filename from an
  immutable checkout. Every authenticated gate then exited before connecting,
  despite a healthy runtime. The explicit package marker makes the inserted
  repository root authoritative; a subprocess regression verifies both package
  and helper origins, and all 17 authenticated command surfaces pass `--help`
  import smokes under the host interpreter.
- 2026-07-10: DS9 shares the strict runtime-secret contract with DS8. Public
  pipeline YAML contains `uri_secret` references; owner-only camera locators
  are resolved in memory, stripped before plugin property application, and
  excluded from generated YAML, provenance, and fingerprints. The MapAnything
  RPC service has no built-in key. Missing, weak, linked, or insecure secret
  state blocks startup rather than selecting a fallback.
- 2026-07-11: DS9 V3DT reproduces the locked DS8 SV3DT semantics without
  reusing DS8 engines or native binaries. The profile pins an unpadded
  1920x1080 stream, three ordered DS9 camInfo files, tracker-internal TAO ReID,
  BodyPose3DNet, and the shared `config/camera_calibration.json`. Its locked
  `xzy` tracker tuple is converted at the producer boundary before publishing
  Y-up `backend_world_m`; malformed axis/bbox state fails closed. The exact
  shared root camera inventories are accepted as app data, while tracker,
  config, model, native binary, and engine ownership remains DS9-only. This
  static global-world correction is not MV3DT overlap-fusion evidence.
- 2026-07-10: Identity-v2 defaults to shadow in DS8 and DS9. Startup hashes the
  actual selected ReID engine bytes and requires an explicit output layer and
  embedding dimension. Authoritative mode requires an artifact-backed scoring
  calibration. DS9 now selects the same provenance-locked TAO Swin-Tiny
  `fc_pred/256` source/config/tensor contract as DS8. The DS9 TensorRT engine is
  now realized; source/artifact parity still does not clear the live
  identity-quality blocker.
- 2026-07-10: DS9 now creates the same canonical world service and capability
  monitor as DS8. Effective runtime config plus content-addressed model/tracker
  files and per-camera calibration snapshots form observation provenance.
  Public track rows carry estimated wall-clock observation time, nonnegative
  media PTS evidence, and visitor generation without treating stream PTS as an
  epoch timestamp.
- 2026-07-10: Person-ground product behavior is one shared implementation with
  explicit DS8/DS9 adapters. Wholebody49 profile semantics are likewise shared,
  but SDK binaries and engines remain runtime-owned; DS9 never reuses a DS8
  engine. Source/parser/config parity does not substitute for an occupied-scene
  runtime-quality gate.
- 2026-07-11: Wholebody49 engine construction no longer relies on a
  workspace-only `trtexec` contract. TensorRT 10.14 does not expose tactic DRAM
  through that CLI, so the DS9 adapter uses a version-pinned C++ builder with
  exact S/X output sets, a fixed batch-three profile, variant-specific 6/4 GiB
  workspaces, and a shared 2 GiB tactic-DRAM limit, checked by pool-specific
  assertions, error-recorder, getter, and exact transcript receipts. Private
  source and ONNX snapshots, exclusive candidate creation, and independent
  `trtexec` candidate/final deserialization remain mandatory. The current outer
  guards are 10,000 MiB for S and 11,000 MiB for X; tactic sources and auxiliary
  streams are unchanged. The
  builder getter-pins optimization level 0 for S and level 3 for X, with an
  exact variant-specific transcript receipt. Transaction
  `20260711T080353252694Z` proves the intervening
  3 GiB setting violated TensorRT's `hasSingleBit(poolSize)` API requirement;
  transaction `20260711T081443440868Z` proves the legal 2 GiB tactic ceiling
  alone still peaked at 9,441 MiB; and transaction
  `20260711T083837156365Z` proves the subsequent 2 GiB S workspace still peaked
  at 9,436 MiB. Transaction `20260711T130350917674Z` proves level 0 reduced the
  peak to 506 MiB across 19 samples but left the final mask Myelin node without
  an admitted tactic: half requested 4,571,136,000 bytes and float requested
  9,142,272,000 bytes. An 8 GiB policy admitted the half requirement while
  excluding the float requirement.
  Transaction `20260711T131135058479Z` proves that policy generated the engine
  in 55.7403 seconds. Its 4,880 MiB result came from 19 coarse host samples and
  is not an authoritative peak; TensorRT's 4,603 MiB result covers only its
  allocator. Publication remained fail-closed because one of
  55,191 VERBOSE diagnostics was overlong; there were 25 INFO, three WARNING,
  and no ERROR or INTERNAL_ERROR messages. The successor ignores only VERBOSE
  before copy/truncation bookkeeping and provenance-pins INFO as the minimum
  captured severity, captured truncation as fatal, and errors as sticky fatal.
  Transaction `20260711T132034937854Z` subsequently built and deserialized the
  25,348,956-byte S engine at both candidate and installed paths, but the host
  finalizer rejected a missing caller-known maintenance-manifest path and
  removed the no-prior candidate. Its 4,846 MiB coarse poll is explicitly
  withdrawn as peak evidence because a contemporaneous operator observation
  was roughly 9,472 MiB. The replacement 25 ms NVML-v2 guard is now part of
  realization acceptance. Its first 8 GiB run captured 9,481 MiB across 2,368
  samples; a 6 GiB run still captured 9,476 MiB, proving WORKSPACE is tactic
  admission rather than a full-device peak control. The reviewed 10,000 MiB S
  guard leaves more than 2 GiB free on the 12 GiB target GPU. Final transactions
  `20260711T152338477896Z` and `20260711T152552934749Z` independently realized
  and loaded the S and X engines, respectively; fresh live quality remains a
  separate gate.
- 2026-07-10: Added explicit runtime ownership/capability and artifact contracts.
  DS9 static checks now classify every copied module, reject DS8 runtime and
  preflight imports, and distinguish structural validity from strict cutover
  parity. The DS9 baseline also applies the shared calibrated dewarper-validity
  geometry to MapAnything outputs with DS9-owned dewarper paths.
- 2026-06-30: Detection-wake performance work was ported to the DS9 runtime
  without relaxing DS9 compatibility gates. DS9 keeps the raw-`pyds`
  compatibility quarantine, but now uses cache-first pose/object-depth
  processing, bounded per-frame telemetry budgets, publish gates, and stage
  timing counters to reduce CPU/GPU spikes when detections appear.
- 2026-06-30: DS9 depth-tensor native builds now compile DS9-owned native
  sources from `DS9/native/`, link the sibling CUDA ROI/stat sampler kernels,
  and stage outputs under `DS9/native_extensions/`. Root DS8 native binaries
  and root native source paths are not used for DS9 rebuilds.
- 2026-06-30: DS9 YOLO26 pose SGIE now targets a DS9-scoped batch-3 asset pair,
  `DS9/models/onnx/yolo26n-pose_b3.onnx` and
  `DS9/models/engines/yolo26n-pose_b3_fp16.engine`. The ONNX source is now
  CPU-exported and hash-verified, and the DS9 10.14 engine is independently
  realized. The root DS8 b3 engine is not reused.
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
  or `noesis.ds8_preflight` by `DS9/scripts/run_static_prep_checks.sh` and the
  ownership validator.

DS9 still shares parent-repo application context:

- Camera inventory defaults to `config/cameras.yaml`.
- Canonical contracts, artifact fingerprinting, global fusion, capability
  monitoring, and the capability-health router live under `noesis_core/` and
  `noesis/server/health_api.py` as shared product code.
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

## Verified DS9.1 Requirements

The direct-upgrade target is:

- Base image: `nvcr.io/nvidia/deepstream:9.1-triton-multiarch@sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994`.
- DeepStream 9.1 at `/opt/nvidia/deepstream/deepstream-9.1`.
- CUDA `13.2.0.046`, TensorRT `10.16.0.72`, and driver `595.58.03` or later.
- DeepStream Python bindings are deprecated; `pyservicemaker` is the recommended Python interface.
- The Service Maker wheel is bundled but must be installed explicitly in the
  derived image.
- Every DS8/DS9.0 native library, parser, plugin, and TensorRT engine must be
  rebuilt. This port does not use compatibility symlinks.

## Current Host Platform Gate

The host audit clears the driver prerequisite: GPU, loaded kernel module,
on-disk module, DKMS, and userspace all report `595.71.05`, which is newer than
the DeepStream 9.1 minimum of `595.58.03`. The current boot has no
Xid, API-mismatch, GPU-fallen-off, or `RmInitAdapter` event. The host deliberately
keeps `/usr/local/cuda` on CUDA 13.0, Python and `trtexec` on TensorRT 10.13.3.9,
and the canonical `deepstream` link on DS8 so DS8 remains recoverable.

This clears only the driver/platform prerequisite. Host preflight now reports
the 595 driver as compatible and still fails closed on host TensorRT 10.13.
DS9.1 CUDA 13.2 / TensorRT 10.16 builds and runtime validation use the pinned
isolated image. The
canonical authenticated DS8 YOLO26m gate has now passed a 30-second advancing
world-state window followed by acknowledged EOS, the expected EOS callback,
Service Maker `wait()` return, exact exit `0`, and no forced kill. That clears
the DS8 native-lifecycle blocker. The selected DS9 engines are realized; any
rebuild still requires an announced exclusive-GPU window. V3DT remains blocked
on fresh live bridge/world/identity/resource/shutdown gates, not engine bytes.

## Historical Host Cutover Snapshot

The 2026-06-16 UTC host validation used the DS9-staged Docker-built resources
without rebuilding host engines, parsers, plugins, or native extensions.

This section is historical evidence, not the current host state. The host was
subsequently restored to driver `580.167.08`, CUDA 13.0, TensorRT 10.13.3, and
DS8 after DS9 MapAnything FP16 plans produced invalid outputs and the FP32
substitute regressed height/floorplan quality. It was migrated back to the
packaged 595 driver on 2026-07-10 without changing the host CUDA/TensorRT stack.
Archived and current kernel evidence do not attribute a failure to driver 595.

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

- `DS9/scripts/zero_copy_stats_smoke_test.py --stub` is an explicit
  control-plane test backend selected by `NOESIS_DS9_STUB_PIPELINE=1`. Its
  result is stamped `backend=synthetic_stub`, `native_runtime=false`, and
  `promotable=false`. It exercises the Python lifecycle/API harness only; it
  does not execute the DeepStream/TensorRT graph, RTSP/WebRTC media, or GPU
  inference and cannot satisfy any acceptance or ownership-promotion item
  below. DS9 does not consume the DS8 stub selector. Config, engine-only
  materialization, and owned-extension import contracts still run, so the
  explicit external DS9 artifact root and its selected files must exist. The
  2026-07-11 isolated-state three-second run observed four stats samples, `0.739423 ms`
  boundary p99, zero errors/violations, all lifecycle markers, exit `0`, and no
  forced kill; that remains non-promotable control-plane evidence.

- The occupied-camera live-RTSP evidence window on 2026-06-16 proved production
  public tracking, repeated stable-ID output, BEV/track parity, native bridge
  counters, media output, MapAnything depth RPC, floorplan RPC, zero-copy stats,
  and live shutdown. It did not prove identity-v2 cross-camera/open-set
  semantics or person-level accuracy, and its floorplan smoke did not cover the
  full camera inventory. The RTSP mosaic sustained decode held for 60 seconds;
  this is not a multi-hour soak.
- Runtime logs include expected noise from short-lived smoke clients closing
  WebSocket connections without a close frame.
- Shutdown/native cleanup: stale user-site `pyservicemaker` shadowing caused
  native heap corruption and constructor/destructor crashes; that shadow install
  has been removed on the host. The June evidence below used an immediate-exit
  workaround and tolerated a live wait thread, so it is historical evidence only
  and no longer satisfies acceptance. DS9 now uses its repo-owned `noesiseos`
  bridge immediately after `streammux` and releases callback-owned state only
  after exact bridge acknowledgement, the expected EOS callback, and Service
  Maker `wait()` return. The launcher returns through normal `SystemExit`; there
  is no GC-bypass environment switch. The canonical runner rejects any wait
  warning, native/GStreamer error, forced kill, nonzero exit, or missing/ordered
  lifecycle marker.

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

Run inside the pinned DeepStream 9.1 container:

```bash
cd <repo>
export NOESIS_DEEPSTREAM_HOME=/opt/nvidia/deepstream/deepstream-9.1
export DS9_CUDA_HOME=/usr/local/cuda-13.2

python3 DS9/scripts/ds9_preflight.py --env-only
bash DS9/scripts/build_gst_plugins.sh
bash DS9/scripts/build_trt_plugins.sh
bash DS9/scripts/build_custom_parsers.sh
bash DS9/scripts/build_native_extensions.sh
```

Large model inputs and engines belong in an explicit filesystem with adequate
headroom, not on the root checkout. The direct engine builder fails closed
unless `NOESIS_MODEL_DIR` or `NOESIS_DS9_ARTIFACT_ROOT` explicitly selects that
external model root. Stage and verify the inputs first:

```bash
export NOESIS_DS9_ARTIFACT_ROOT=<absolute-large-filesystem-artifact-root>
python3 DS9/scripts/stage_canonical_sources.py
python3 DS9/scripts/validate_asset_manifest.py \
  --artifact-root "$NOESIS_DS9_ARTIFACT_ROOT"
```

The staging gate preserves 10 GiB of residual free space by default. The V3DT
engines must be built in order during an announced exclusive-GPU window. The
same commands with `--plan` use isolated `runc` with no GPU devices:

```bash
export NOESIS_DS9_DOCKER_ROOT=<absolute-secondary-docker-root>
DS9/scripts/run_canonical_engine_maintenance.sh --only bodypose3dnet --plan
DS9/scripts/run_canonical_engine_maintenance.sh --only v3dt_tracker_reid --plan

# Exclusive-GPU window only, after every compute owner has stopped:
DS9/scripts/run_canonical_engine_maintenance.sh --only bodypose3dnet
DS9/scripts/run_canonical_engine_maintenance.sh --only v3dt_tracker_reid
```

`--plan` is strictly write-free, including filesystem metadata. It requires
the artifact engine/evidence/log directories and the owner-only transaction
lock to exist already with the reviewed owner and modes; it refuses missing or
unsafe paths instead of creating or chmod-repairing them. Only an actual
maintenance invocation may prepare those paths.

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
NOESIS_MAPANYTHING_GPU_GUARD_MB=9000 \
NOESIS_MAPANYTHING_GUARD_POLL_SECONDS=1 \
DS9/scripts/build_mapanything_guarded.sh
```

Then run full preflight:

```bash
python3 DS9/scripts/ds9_preflight.py
```

## Run

Host execution reads the owner-only paths documented in
`../docs/Runtime_Secrets.md`. Container execution must mount the camera registry
and MapAnything key individually, read-only, and set their in-container paths
with `NOESIS_CAMERA_SECRETS_FILE` and
`NOESIS_MAPANYTHING_API_KEY_FILE`; the exact uid/mode-safe command is in
`docs/validation_runbook.md`. Never mount the entire host secrets directory.

Production canary acceptance uses only the enumerated secondary-container
supervisor. It hard-pins profile/config/engine provenance and owns resource,
shutdown, cleanup, and evidence contracts:

```bash
python3 DS9/scripts/run_canonical_runtime_container.py plan --lane baseline
python3 DS9/scripts/run_canonical_runtime_container.py plan --lane v3dt
python3 DS9/scripts/run_canonical_runtime_container.py plan --lane wholebody49-s
python3 DS9/scripts/run_canonical_runtime_container.py plan --lane wholebody49-x
```

Rerun the selected lane with `run --authorize-gpu-runtime` only after its plan
is green and the exclusive GPU owner has been released. See
`docs/runtime_container_boundary.md` and `docs/validation_runbook.md` for the
exact canary and occupied-scene sequences.

The direct host commands below remain developer diagnostics; they do not prove
the pinned image, external realization, resource ceiling, or ordered lifecycle
boundary:

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
python3 DS9/noesis/ds9_runtime.py --pgie-profile rfdetr_keypoint --disable-rest

# Requires the complete V3DT artifact profile and a real DS9 runtime:
NOESIS_DS9_ARTIFACT_ROOT=<absolute-artifact-root> \
python3 DS9/noesis/ds9_runtime.py --tracking-mode v3dt --disable-rest
```

In the current checkout these direct commands stop before profile preflight
because `DS9/noesis/ds9_runtime_core.py` imports the absent
`noesis_core.runtime_publication` module. The RF-DETR assets and configs are
materialized and attested, but no live-runtime smoke is claimed; see
`DS9/docs/rfdetr_1_8_3_assets.md`. No deployed snapshot is used as an import
fallback.

An omitted tracking mode, or the explicit value `auto`, selects the documented
baseline. Any other nonempty value must resolve to a supported baseline/V3DT
alias; an unknown `NOESIS_TRACKING_MODE` value is fatal and never falls back to
baseline.

YOLO11/YOLO26 detector profiles use DS9-staged ONNX sources under
`DS9/models/onnx/` and DS9-built TensorRT engines under `DS9/models/engines/`.
They use the DS9-compatible custom bbox parser at
`DS9/pipelines/nvdsinfer_yolo_detect/libnvdsparsebbox_yolo.so`, which exports
`NvDsInferParseYolo`. `NOESIS_YOLO_DETECT_PARSER_LIB=/path/to/lib.so` can still
override it for investigation.

RF-DETR 1.8.3 uses versioned runtime ONNX sources and DS9-built TensorRT
engines under the configured artifact root. The selected runtime profile is
FP16 with TF32 enabled (`fp16_tf32`), matching the current YOLO precision
policy; rendered PGIE configs require `network-mode=2`. Detection supports
`n/s/m/l`, segmentation supports `n/s/m/l/x/2x`, and keypoints use the sole
reviewed Preview variant. Every selected runtime ONNX, receipt, engine, and
engine receipt is SHA-256-bound in the model matrix; materialization also
attests the selected parser source and binary. The selection is grounded in
the local 5,000-image person-only evaluation under
`NOESIS_DS9_ARTIFACT_ROOT/models/benchmarks/head-to-head/`
`coco-person-full-20260725T215428672011Z/`, not published checkpoint metrics.

The latest detector-profile matrix covers `yolo11`, `yolo26 n/s/m/l/x`,
`yolo26_seg n/s/m`, `rfdetr n/s/m/l`, `rfdetr_seg n/s/m/l/x/2x`, and
`rfdetr_keypoint` materialization.

## Graph

See the full DS9 pipeline map:

- [DS9/PIPELINE_GRAPH.md](PIPELINE_GRAPH.md) — Mermaid diagram + component table + all hook attachment points (world_observation_stage, MapAnything gating, native DS9 pose/depth bridges, trails/keypoints, etc.).

Quick textual summary (see the detailed doc for the complete graph):

```text
RTSP sources (w/ dewarpers) -> nvurisrcbin -> nvstreammux -> nvdspreprocess -> yolo11_pgie
  -> main_tee
       ├─ analytics_exclude (nvdsroiexclude) -> nvtracker -> reid_sgie (TAO Swin-Tiny fc_pred/256) -> yolo26_pose
       ├─ depth_tracking_queue -> depth_tracking_fullframe (DAv2) -> fakesink
       └─ mapanything_queue -> mapanything_valve (gated) -> mapanything_fullframe
            -> nvvideoconvert -> NVMM RGB caps -> fakesink
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
- `tracker.ll-lib-file`: points at `/opt/nvidia/deepstream/deepstream-9.1/lib/libnvds_nvmultiobjecttracker.so`.
- V3DT streammux remains exactly unpadded `1920x1080` with one surface per
  frame so the locked projection matrices stay valid.
- `NOESIS_DS9_ARTIFACT_ROOT`: external physical owner for virtual
  `DS9/models/...` paths. Source staging and engine maintenance retain at least
  10 GiB of free space by default.
- Actual DS9.1 engine maintenance requires the installed DeepStream 9.1 driver
  floor (`595.58.03` or newer). The current `595.71.05` driver satisfies that
  prerequisite. Plan mode remains CPU-only and is not build-readiness evidence.
- `mosaic_output.*`: unchanged behavior, RTSP to WebRTC remains canonical.
- `analytics.exclude.config-file`: still uses shared ROI config because ROI geometry is app data, not SDK-version-specific.
- `bev.frame`: `backend_world_m`.
- `NOESIS_DEPTH_RPC_TIMEOUT_SECONDS`: WebSocket-side MapAnything depth RPC
  provider timeout, default `110.0` seconds.
- `NOESIS_DEPTH_RPC_ENABLE_SECONDS`: on-demand MapAnything burst length,
  default `20.0` seconds.

## Acceptance Criteria

For manual behavior-producer commands, set `RUNTIME_INSTANCE_ID` and
`RUNTIME_RUN_ID` from authenticated `/api/v1/health/capabilities`, or from the
canonical supervisor session's owner-private `runtime-identity.json`. Those
values bind the report to the runtime it actually observed.

- `python3 DS9/scripts/ds9_preflight.py` passes.
- Preflight reports `pyservicemaker` from `/usr/local/lib/python3.12/dist-packages`,
  not the user-site package.
- `python3 DS9/noesis/ds9_runtime.py --enable-rest` starts and reaches the Service Maker wait loop.
- `python3 scripts/ma_depth_rpc_smoke_test.py --no-spawn` passes against DS9 runtime.
- `python3 scripts/zero_copy_stats_smoke_test.py --no-spawn` passes against DS9 runtime.
- `python3 scripts/zero_copy_smoke_test.py --no-spawn` passes against DS9 runtime with REST enabled.
- `python3 DS9/scripts/ds9_floorplan_live_gate.py --session-id manual-session --runtime-lane baseline --runtime-instance-id "$RUNTIME_INSTANCE_ID" --runtime-run-id "$RUNTIME_RUN_ID" --out DS9/build/live_validation/manual/mapanything-depth-quality.json --source-out DS9/build/live_validation/manual/mapanything-depth-quality-source.json`
  validates the v4 fresh/cache-only, active-authority, and reconciled local-BEV
  contract for every active configured camera. Run it only against an
  intentionally launched validation session: its fresh phase manually opens
  the MapAnything valve, while its passive/cache-only phase must produce zero
  snapshot mutation. Passing this gate alone does not promote a release or
  establish absolute depth accuracy.
- `python3 DS9/scripts/ds9_identity_shadow_live_gate.py --session-id manual-session --runtime-lane baseline --runtime-instance-id "$RUNTIME_INSTANCE_ID" --runtime-run-id "$RUNTIME_RUN_ID" --out DS9/build/live_validation/manual/identity-open-set-occupied.json`
  validates authenticated shadow health, fresh embedding-backed observation
  keys, and tracker-subject continuity while requiring public authority to stay
  blocked. Its report must leave semantic accuracy `not_evaluated`; use
  `--require-cross-camera` and `--require-open-set` only for a real scene that
  should exercise those mechanisms.
- `NOESIS_MOSAIC_RTSP_ENABLED=1 NOESIS_MOSAIC_WEBRTC_ENABLED=1 python3 scripts/webrtc_gateway_smoke_test.py --ws ws://127.0.0.1:6008 --duration 5 --pt 103` returns decoded frames.
- `python3 DS9/scripts/ds9_live_validation_runner.py` is the preferred
  occupied-camera host evidence bundle for live-RTSP identity shadow health,
  BEV/track, bridge, RTSP/WebRTC, depth/floorplan, zero-copy, and shutdown
  checks. Every behavior report carries the runner's exact session/lane
  envelope. It writes `summary.json`, `summary.md`, `runtime.log`, canonical
  behavior reports, and per-gate logs under
  `DS9/build/live_validation/<timestamp>/`. The owner-only bearer must already
  exist; select a non-default provisioned file with `--auth-token-file`.
- `--lane v3dt` hard-pins `infer_v3dt.yaml`, `cameras_v3dt.yaml`,
  `yolo26_seg/s`, and tracking mode `v3dt`; it runs authenticated bbox3d plus
  global-world v2 gates and never substitutes the baseline profile.
- `--lane wholebody49-s` and `--lane wholebody49-x` run separate occupied-scene
  parser gates. S requires consumed instance-mask samples; X requires bbox
  samples and forbids mask-path activation. Each promotion session also
  requires the typed direct-RTSP plus WebRTC decoded-media report and the
  generic v2 300-second resource report, all sealed with the same runtime
  identity and exact container/checkout/realization/engine/config binding.
  Source-complete 2 fps and 2.5-second cadence bounds and the 10,000 MiB S /
  11,000 MiB X GPU ceilings are provisional fail-closed policy, not empirical
  model performance. Empty-house evidence remains an explicit occupied-gate
  failure rather than being synthesized or forced.
- Focused ROI, V3DT, alternate-profile, and bridge-specific smokes pass where
  those DS8 option paths are required.
- V3DT acceptance additionally requires the authenticated
  `DS9/scripts/sv3dt_meta_smoke_test.py` and
  `DS9/scripts/v3dt_world_contract_smoke_test.py`. The latter accepts only its
  privacy-safe v2 source/report pair from the exact same supervisor session. It
  binds the sealed launch plan plus effective pipeline/tracker/camera/
  calibration/alignment/camInfo hashes, requires all three cameras and advancing
  per-camera continuity, rejects the old raw `xzy` tuple, and validates
  `world_frame=backend_world_m`, `world_source=bbox3d`, floor consistency, native
  image-foot reprojection, and independent `image_base` replay. V1 or
  camera-local evidence cannot be promoted. Source staging or engine
  deserialization alone is not runtime parity, and this SV3DT gate makes no
  MV3DT overlap/time-sync/fusion claim.

Bench report must include command, resolution `1920x1080`, batch `3`, device, fps, latency_ms, and PASS/FAIL against the selected target.

## Historical Validation Evidence

DS9 Docker validation currently passes these gates:

- `DS9/models/onnx/mapanything_images_294x518_b3.onnx` exists and references 3.43 GiB of ONNX external tensor sidecars.
- Historical validation built `mapanything_images_294x518_b3_fp16.plan` with
  BF16 despite its filename, and `trtexec --loadEngine ... --skipInference`
  succeeded. That result is retained only as historical evidence and no longer
  satisfies cutover. The guarded builder now requires real `--fp16`; a fresh
  provenance-complete artifact plus depth/floorplan quality validation remains
  a blocker.
- `python3 DS9/scripts/ds9_preflight.py` succeeds in the DS9 container after installing `PyYAML` into the ephemeral container environment.
- Historical DS9 runtime evidence loaded all five then-selected TensorRT engines,
  including OSNet ReID. That evidence does not validate the currently selected
  TAO Swin-Tiny engine, which is now realized but lacks fresh identity-quality
  evidence.
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
- The historical 2026-07-25 RF-DETR 1.8.3 runtime-input media gate passed all
  11 open variants across 88 ONNX Runtime and 88 TensorRT cases using raw RGB01
  input and the then-selected full-FP32, TF32-disabled engines. All retention,
  candidate precision, box, score, mask, and keypoint gates passed. The later
  local labeled head-to-head measured every FP16-minus-FP32 primary AP change
  within one tenth of a point and selected `fp16_tf32`; see
  `DS9/docs/rfdetr_1_8_3_assets.md`.
- Historical RF-DETR segmentation profile `--pgie-profile rfdetr_seg --size m`
  passed
  bounded startup after fixing generated DS9 parser/label paths.
- Historical RF-DETR detect-only profile `--pgie-profile rfdetr --size s`
  passed bounded
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
- Historical June host shutdown/native cleanup smoke with MP4 inputs (retained
  for regression provenance, not current acceptance):
  `DS9/noesis/ds9_runtime.py` reported `pyservicemaker` from
  `/usr/local/lib/python3.12/dist-packages`, received SIGINT, posted EOS, logged
  `Wait thread did not terminate cleanly`, and exited `0` with no
  `Fatal Python error`, segmentation fault, malloc, double-free, or heap
  corruption markers. Finite-source MP4 EOS with `streammux.live-source=0` also
  exits `0`. The current gate must reject that wait-thread warning.
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
  - Historical live RTSP SIGINT shutdown exited `0`, closed ports `6008`, `8080`, and
    `8554`, and left no DS9 runtime process. The shutdown tail logged
    `Wait thread did not terminate cleanly` and one `source_2` reconnect warning
    after EOS, with no native heap/fatal markers. This older result does not
    clear the current acknowledged-EOS shutdown gate.

### Portable reviewed inference sources (2026-07-12)

- The canonical and V3DT pipeline YAMLs now select the tracked
  `DS9/pipelines/config_infer_secondary_depth_tracking_da2.ini` source contract
  instead of requiring a generated file below `DS9/build/`. Runtime still
  derives an engine-only config under the selected `NOESIS_BUILD_DIR` and uses
  DS9-owned artifact roots for the engine.
- DS9 static preparation requires and scans that reviewed source. It no longer
  treats mutable build output as source-checkpoint content.

### Zone authority parity (2026-07-19)

- DS9 emits the same additive `zone_source` and `zone_authoritative` fields as
  DS8. NvDsAnalytics ROI labels are spatially authoritative; camera-name
  fallbacks remain occupancy/dwell diagnostics and cannot derive canonical
  `room_id`.

Still pending before full option-surface parity:

- Run one fresh occupied, same-session DS9-native V3DT
  bbox3d/global-world/identity/resource/shutdown bundle and promote only its
  replayed v2 evidence. All selected engines are already realized.
- Prove MV3DT separately with synchronized kitchen/family-room overlap, peer
  association, and fused-position evidence; the global SV3DT frame cannot stand
  in for that proof.
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
