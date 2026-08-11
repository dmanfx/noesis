# Spatial OS Architecture Decisions

## SOD-001 - Local appliance boundary

- Date: 2026-07-09
- Decision: One single-home LAN appliance, with one authenticated same-origin
  gateway at `TauntonMainframe.local`; no cloud or multi-tenant dependency.

## SOD-002 - Resident-biased open-set identity

- Date: 2026-07-09
- Decision: Unknown remains explicit. A bounded resident prior applies only
  after absolute evidence and ambiguity gates and cannot force acceptance.

## SOD-003 - Backend owns global world truth

- Date: 2026-07-09
- Decision: Noesis publishes time-aligned global entities with uncertainty and
  provenance. Menon renders them and does not perform production fusion, range
  compression, or room-boundary correction.

## SOD-004 - One product core, two SDK adapters

- Date: 2026-07-09
- Decision: Identity, world, scene, telemetry, API contracts, policy, and replay
  are shared. DS8/DS9 graph, metadata access, native ABI, assets, and lifecycle
  remain adapter-specific.

## SOD-005 - DS9 is the intended successor

- Date: 2026-07-09
- Decision: Every accepted product contract is mandatory for DS8 and DS9. DS8
  remains production and rollback until DS9 passes canary and soak gates.

## SOD-006 - Deterministic action authority

- Date: 2026-07-09
- Decision: Models propose; server policy authorizes; users approve where
  required; adapters execute; readback and receipts determine truth.

## SOD-007 - Coherent scene promotion

- Date: 2026-07-09
- Decision: Menon renders one atomic promoted multi-camera scene release. It
  never assembles product truth from independently newest camera revisions.

## SOD-008 - Modular monolith before services

- Date: 2026-07-09
- Decision: Establish explicit in-process domain contracts and replay first.
  Split processes only when an operational boundary is demonstrated.

## SOD-009 - Privacy by local minimization

- Date: 2026-07-09
- Decision: Resident identity is opt-in; visitor/provisional evidence expires;
  sensitive files use owner-only permissions; raw video stays memory-only unless
  an owner explicitly promotes evidence.

## SOD-010 - No compatibility through hidden fallback

- Date: 2026-07-09
- Decision: Migrations may use explicit shadow comparison and rollback, but a
  failed canonical path never silently selects a legacy or unauthenticated path.

## SOD-011 - Contract truth precedes runtime integration

- Date: 2026-07-09
- Decision: Product observations, identity decisions, world snapshots, scene
  releases, capability health, and actions are immutable versioned contracts in
  `noesis_core`. DS8 and DS9 adapters normalize transient SDK metadata into
  those contracts; consumers never retain SDK wrapper objects.

## SOD-012 - Calibrated image validity is shared evidence

- Date: 2026-07-09
- Decision: MapAnything output validity is derived from explicit nvdewarper
  calibration and source geometry and is applied identically in canonical and
  V3DT processors. Missing configured data fails visibly; invalid pixels become
  NaN with zero confidence and are never promoted through a full-frame mask.

## SOD-013 - Global fusion exposes disagreement

- Date: 2026-07-09
- Decision: Compatible contemporaneous camera observations are fused by
  declared uncertainty. Contradictory observations are not averaged; the best
  supported observation is retained and all rejected evidence is published with
  a conflict reason.

## SOD-014 - Wire semantics are never inferred

- Date: 2026-07-10
- Decision: Public payloads must explicitly carry contract name/version and
  coordinate frame/units. Generated JSON Schema and TypeScript come from the
  same strict model source; missing or unsupported semantics fail closed.

## SOD-015 - Replay evidence is deterministic and integrity chained

- Date: 2026-07-10
- Decision: Characterization, identity, world, and adapter parity evidence use
  the versioned `noesis.replay` NDJSON format. Every payload is contract-
  validated, sequences are contiguous, and each record hashes the prior record.

## SOD-016 - Scene content is immutable; promotion is separate state

- Date: 2026-07-10
- Decision: A scene release never mutates from candidate to promoted. Immutable
  release content is registered by hash; one atomic compare-and-swap pointer and
  append-only promotion history own current/rollback state.

## SOD-017 - World output is content-addressed and journaled

- Date: 2026-07-10
- Decision: DS8 and DS9 normalize adapter evidence into the same observation,
  world snapshot, and semantic event contracts. Active calibration, effective
  model/tracker files, and pipeline configuration are content-addressed. A
  private bounded SQLite chain journals canonical output; transport success is
  not the only retained evidence.

## SOD-018 - Scene bundles separate shared and per-camera identity

- Date: 2026-07-10
- Decision: A scene release fingerprints the selected calibration/model bundle
  while retaining each camera revision's individual calibration, model,
  manifest, and artifact hashes. Requiring individual camera hashes to be equal
  would erase rather than prove multi-camera provenance.

## SOD-019 - Appliance control endpoints are exact

- Date: 2026-07-10
- Decision: Noesis REST and WebSocket controls default to loopback and either
  bind their configured endpoint or fail. The runtime never selects a nearby
  port, assumes an existing listener is Noesis, or silently omits required
  WebRTC/depth startup capability.

## SOD-020 - Semantic world events are derived from canonical snapshots

- Date: 2026-07-10
- Decision: Appeared, held, resumed, lost, conflict-started, and
  conflict-cleared events are deterministic diffs of backend-owned world
  snapshots. Browser arrival time and local disappearance timers do not define
  household presence truth.

## SOD-021 - Journal retention preserves a verifiable prefix boundary

- Date: 2026-07-10
- Decision: The canonical contract journal may discard only a contiguous oldest
  sequence prefix and anchors the retained chain to the final discarded hash.
  Age-based retention never punches holes in the chain when producer timestamps
  arrive out of order, and writer failure remains fatal while still allowing a
  deterministic thread shutdown.

## SOD-022 - A scene release closes over rendered bytes

- Date: 2026-07-10
- Decision: Scene immutability covers every camera artifact, the authored OBJ,
  its transitive rendered MTL/texture set, and the validation report, not only
  revision manifests. The contract carries unique release-owned normalized
  paths, lengths, and SHA-256 values; promotion and current reads revalidate the
  bytes. The authored home, dependencies, and role-addressed camera artifacts
  are served from current-release same-origin endpoints rather than the
  unrestricted revision catalog. CLI validation output uses the same canonical
  bytes named by the release hash, and an existing release ID can never be
  overwritten with different content.

## SOD-023 - Identity transitions replace tracklet ownership atomically

- Date: 2026-07-10
- Decision: Global fusion owns an exact `(camera, source, run, tracker)` to
  entity mapping. When identity evidence changes the subject, the prior entity's
  camera observation is removed before the replacement is ingested. Explicit
  open-set unknown is retained as `unknown`, not rewritten as provisional.

## SOD-024 - Disabled internal auth is structurally loopback-only

- Date: 2026-07-10
- Decision: Explicit development auth disablement is not sufficient authority
  to expose a listener. REST and WebSocket startup accept disabled mode only on
  `localhost` or a literal loopback address and reject wildcard, LAN, and
  arbitrary hostname binds across DS8, V3DT, and DS9.

## SOD-025 - Storage paths never cross the telemetry boundary

- Date: 2026-07-10
- Decision: Frame-level depth may retain its real storage reference inside the
  runtime, but WS telemetry replaces it with a deterministic opaque SHA-256
  artifact identifier. Public clients receive neither filesystem structure nor
  a backend storage URI, and the identifier is not advertised as a fetch URL.

## SOD-026 - Product world frame and local BEV frame are separate contracts

- Date: 2026-07-10
- Decision: DS8 and DS9 must agree on the strict canonical
  `backend_world_m` observation/snapshot contract. Their adapter-local diagnostic
  BEV renderers may retain different proven display frames; that difference is
  not a cutover blocker and does not authorize browser-side product fusion.

## SOD-027 - Media readiness means usable RTSP, not an open port

- Date: 2026-07-10
- Decision: DS8 and DS9 require a successful RTSP DESCRIBE for the exact mosaic
  mount before starting WebRTC, require an upstream keyframe requester, keep a
  bounded warm gateway set, and allocate additional slots on demand. A TCP
  accept alone is not media readiness.

## SOD-028 - Build intent and artifact evidence are separate gates

- Date: 2026-07-10
- Decision: Correcting the DS9 MapAnything builder from BF16 to true FP16 does
  not make the named engine available or validated. Cutover remains blocked
  until a newly built engine carries complete provenance and passes tensor,
  NaN, depth, and floorplan parity gates; a historically loadable BF16 engine
  with an FP16 filename is not qualifying evidence.

## SOD-029 - Identity authority is whole-frame, provenance-bound, and calibrated

- Date: 2026-07-10
- Decision: DS8, V3DT, and DS9 construct one shared identity-v2
  store/runtime/coordinator for the canonical world run. Hooks detach a complete
  source-frame primitive batch and resolve it once. Shadow is the default;
  authoritative mode requires both an artifact-backed scoring calibration and
  a separate byte-pinned runtime cutover artifact covering coordinator replay
  and occupied-scene reports. Model semantics bind actual engine and loaded
  ReID extractor bytes, output layer/dimension, and the Python normalization,
  gallery-similarity, and scoring implementations. Enrollment consumes exact
  server evidence only, and every dual-camera exception requires fresh
  topology, world, time, and appearance proof. DS8 evidence cannot authorize
  DS9.

## SOD-030 - Authoritative identity has one owner and truthful display timing

- Date: 2026-07-10
- Decision: Authoritative v2 bypasses legacy StableID mutation and performs
  SID-dependent analytics only after whole-frame resolution. Fresh-evidence
  rejection is `unknown`; missing evidence without a held resolution is
  `provisional`. The one-shot SDK metadata walk uses an explicit neutral OSD
  override instead of stale legacy identity. A verified tiler-sink Service Maker
  operator then renders only an exact bounded camera/frame/tracker decision;
  every missing or stale join remains `#XX`.

## SOD-031 - Private state is create-private or validate-private

- Date: 2026-07-10
- Decision: Contract journals, scene promotion state, identity state, internal
  tokens, and integration secrets may create a dedicated owner-only leaf and
  file, or validate an existing owner-only single-link path. They never chmod a
  pre-existing directory or file to make it acceptable. Shared parents,
  symlinks in any existing path component, hardlinks, ownership drift, and
  permission drift fail closed. The shared path primitive performs this check,
  so journals, scenes, identity, diagnostics, calibration audit, and tokens do
  not implement weaker local variants.

## SOD-032 - Live and offline geometry share one calibration authority

- Date: 2026-07-10
- Decision: DS8, V3DT, DS9, reconstruction builders, and depth-registration
  builders all construct the exact-parity `CalibrationManager` through one
  factory using the same cameras mapping, streammux image space, extrinsics,
  and scene alignment. The former private runtime provider is removed rather
  than retained as a compatibility path.

## SOD-033 - Hosted CI and appliance hardware evidence are distinct

- Date: 2026-07-10
- Decision: Hosted Python 3.12/Node 22 CI runs deterministic schemas, contracts,
  unit/replay/security/browser/static DS9 gates and production builds in
  pinned CPU environments. DeepStream, TensorRT, decoded media, GPU resource,
  and live shutdown evidence runs only through explicitly dispatched labeled
  appliance gates; a CPU pass never claims hardware readiness.

## SOD-034 - Immutable scene bytes are descriptor-verified and bounded

- Date: 2026-07-10
- Decision: Scene roots and relative components are opened without following
  links. Every manifest, authored dependency, validation report, and camera
  artifact must be a non-empty, single-link regular file and is read through a
  bounded inode/path-checked descriptor before its length and SHA-256 are
  accepted or served. Builders parse bounded strict UTF-8 OBJ/MTL input once,
  materialize those exact bytes into a staged exact tree, and publish with
  atomic no-replace semantics; existing state is validated, never overwritten
  or chmod-repaired. Promotion and current payload reads validate the whole
  cohort, while role-addressed binary reads verify only their selected file
  against the integrity-checked release row to avoid O(N-squared) hashing.

## SOD-035 - Continuous appliance health is a fail-closed lifecycle authority

- Date: 2026-07-10
- Decision: Ordered one-shot readiness gates remain the startup authority, but
  a separate long-lived guard continuously checks authenticated producer
  capability health, CA-validated Menon HTTPS, and a bounded health-only
  WebSocket path. Core and WebSocket failure budgets are independent. When
  either budget is exhausted the guard exits and the target's `BindsTo=` stops
  the whole appliance; the target never upholds or silently restarts the guard.
  Every orchestration mutation is serialized by a kernel-held owner-only lock,
  and interrupted upgrade/cutover journals require explicit fail-closed
  recovery before another lifecycle action.

## SOD-036 - DS9 V3DT separates source parity, engine evidence, and world truth

- Date: 2026-07-10
- Decision: DS9 owns its V3DT pipeline, camera/camInfo/tracker configs, native
  bridge, source provenance, build helper, preflight, runtime materialization,
  and smoke. Large model/engine bytes live in an explicit external artifact
  root with residual-capacity and atomic-install gates. Source/config parity
  does not clear the platform, two-engine, or live-runtime gates; actual builds
  also require the installed DeepStream 9 driver floor. The locked profile
  stays `camera_local`; only separately accepted shared metric calibration may
  promote it into the canonical `backend_world_m` fusion contract.

## SOD-037 - DS9 host capability is a reversible driver-only migration

- Date: 2026-07-10
- Decision: Move the restored host from the APT/DKMS-managed 580.167.08 open
  driver to Ubuntu's exact 595.71.05 open-driver package only after a private
  full-source/runtime/package checkpoint. Noble's 590 package is transitional,
  and NVIDIA's 590 runfile must not be mixed into this package-managed host.
  Keep host CUDA 13.0 and TensorRT 10.13.3 for DS8; DS9 CUDA/TensorRT engine
  work remains isolated. A coherent post-reboot driver is only platform
  eligibility: DS8 engine/runtime/media/identity/MapAnything/desktop gates must
  pass before any DS9 build, and exact cached 580 packages remain the rollback
  authority. The June 595 rollback was an application/model-quality rollback,
  not evidence of a driver compute failure.

## SOD-038 - Rollback rehearsal is exact, local-only, and not execution

- Date: 2026-07-10
- Decision: The 595-to-580 rollback authority is one versioned 33-archive
  package/control manifest plus the private checkpoint checksum manifest. A
  qualifying rehearsal rehashes every declared checkpoint byte, verifies exact
  package identities, versions, architectures, payload hashes, and dependency/
  conflict relationship hashes, checks current module/DKMS/boot coherence, and
  runs APT only as `--simulate --no-download` against the exact local archives.
  The transaction explicitly removes the held conflicting 595 cohort. A
  non-595 removal is forbidden unless the cached package relationships prove it
  is solver-required; `nvidia-prime` is the sole exception because the exact
  580 driver metapackage declares `Conflicts`/`Replaces`, and the accepted 580
  baseline records it removed. Persisted rehearsal evidence is atomically
  fsynced into an owner-only directory as a new single-link 0600 file; symlinks
  and replacement are rejected. Rehearsal never claims rollback execution.
  Actual execution requires an exclusive GPU/display maintenance window,
  initramfs regeneration, reboot, and the complete post-reboot DS8 engine,
  authenticated runtime/media/world/identity/depth/resource/desktop acceptance
  contract.

## SOD-039 - DS9 build intent is portable; machine realization is transactional

- Date: 2026-07-10
- Decision: Keep `DS9/asset_manifest.yaml` as the portable, reviewable artifact
  declaration and record machine-local TensorRT results only in the external
  owner-private `asset_realization.json`. Every real engine build is a
  one-engine transaction under the shared artifact lock: a pinned source and
  tensor contract, immutable DS9 image ID, exact GPU/driver identity,
  pre-launch host snapshot, candidate and installed cold-load proofs, atomic
  engine/realization commit, and deterministic rollback are all required.
  Interrupted containers may clean only residue proven to have been created by
  their prepared transaction. A committed artifact remains
  `staged_unverified` until its runtime lane passes; serialization alone never
  promotes readiness.

## SOD-040 - DS9 acceptance uses enumerated lanes, not free-form profiles

- Date: 2026-07-10
- Decision: The canonical container supervisor exposes only the reviewed
  `baseline`, `v3dt`, `wholebody49-s`, and `wholebody49-x` lanes. Each lane
  hard-pins its pipeline/camera configuration, model profile and size, tracking
  mode, required realized engines, container identity, behavior gate, and
  lifecycle evidence. Unknown combinations and YOLO11 substitutes fail closed.
  V3DT acceptance proves its locked `camera_local` plus `bbox3d` contract; it
  does not claim `backend_world_m` fusion without separately accepted shared
  metric calibration. Wholebody49-s must exercise masks, while Wholebody49-x
  must exercise boxes and prove the mask path stayed inactive.

## SOD-041 - MapAnything correctness is admitted by real inference, not precision intent

- Date: 2026-07-12
- Decision: Supersede SOD-028's provisional FP16 build intent for DS9
  MapAnything after the actual batch-three FP16 engine loaded successfully but
  emitted non-finite depth, zero masks, and confidence sentinels on a pinned
  real camera tensor. The canonical DS9 authority now requires a
  correctness-first FP32 build and a pre-install real-inference receipt binding
  the exact engine, private fixture, TensorRT platform, command, tensor shapes,
  finite/positive/coverage/distribution bounds, and identical-batch behavior.
  Deserialization alone cannot admit the engine. FP32 functional admission is
  necessary but does not claim all-camera floorplan parity or acceptable
  full-runtime GPU headroom; those remain separate fail-closed live gates.
