# Noesis + Menon Spatial OS Implementation Plan

Status: execution approved 2026-07-09.

This workstream converges Noesis and Menon into a secure, measurable, local
spatial operating system for one home. Noesis owns perception, identity, world
truth, and scene evidence. Menon owns the household experience, device control,
and governed workflows. DS8 remains the live production baseline while DS9 is
developed as the near-term successor against the same product contracts.

## Accepted product decisions

1. The deployment is a single-home LAN appliance, not a cloud or multi-tenant
   service.
2. Identity is open-set against an intentionally enrolled resident roster.
   Unknown is always a valid outcome.
3. Resident evidence receives a bounded household prior only after clearing an
   absolute open-set evidence floor and ambiguity checks. The prior may change
   ranking among admissible candidates; it may not turn a rejection into a
   resident acceptance.
4. Resident names are exposed and identity-sensitive automations are authorized
   only after a confirmed resident decision. Tentative hypotheses stay internal.
5. DS9 is the intended near-term runtime successor. Every product behavior and
   contract change is shared or explicitly implemented and validated for both
   SDK adapters.
6. DS8 stays immediately restartable until DS9 passes replay, media, resource,
   persistence, shutdown, canary, and soak gates.
7. The working DS8 GPU graph, native bridges, Three.js authored home, HomeSeer,
   Cast integration, and versioned reconstruction artifacts are retained unless
   measured evidence requires replacement.
8. No hidden fallbacks or client-side geometry corrections may mask a failed
   canonical path.

## Plan evolution

### Iteration 1: strategic dependency order

The first pass established the sequence: baseline and security, shared
contracts, identity, backend world fusion, coherent digital twin, safe agentic
control, DS9 convergence, then operational cutover.

### Iteration 2: acceptance and migration depth

The second pass added concrete contracts, typed producer metadata, local
authentication, action receipts/audit, visitor generations, atomic scene
promotion, Menon service boundaries, CI, replay, and explicit rollback gates.

### Iteration 3: execution contract

Three independent read-only audits refined the final plan:

- identity must begin with truth-preservation defects and real negative replay,
  not threshold tuning;
- DS8 and DS9 need one product core plus explicit SDK adapters, not copied
  application algorithms;
- the LAN product needs one authenticated same-origin gateway and a deterministic
  action coordinator before agent authority expands.

## Target architecture

```text
camera sources
  -> DS8 or DS9 SDK adapter
  -> versioned observation envelopes
  -> identity resolver + global world fusion
  -> world snapshots/events + coherent scene releases
  -> authenticated appliance gateway
       -> Menon household experience
       -> Noesis operator experience
       -> engineering console (owner/loopback only)
       -> policy/approval/action coordinator
            -> HomeSeer / Cast / Noesis adapters
            -> readback + immutable audit
```

Shared product logic must be imported through an unambiguous regular Python
package. SDK-specific metadata access, graph construction, engines, parsers,
native ABI, and lifecycle behavior remain behind DS8 and DS9 adapters.

## Phase 0 - Governance and baseline ledger

### Deliverables

- Track this plan, `work_order.md`, `decisions.md`, `validation.md`, and
  `handoff.md`.
- Record both dirty worktrees, active processes/listeners, SDK/runtime versions,
  current identity health, current contract versions, and test baselines.
- Classify existing modifications by owner and contract boundary before staging
  or committing anything.
- Create a DS8/DS9 ownership and capability matrix covering modules, configs,
  CLI flags, APIs, telemetry, models, native extensions, and validation.
- Add an automated drift check that rejects an unclassified duplicated product
  module or contract difference.
- Define clean checkpoint boundaries and exact rollback behavior for every
  state/schema migration.

### Gate

No implementation phase starts without a preserved baseline and explicit file
ownership. The live DS8 process is not restarted in this phase.

## Phase 1 - Truth and safety repair

### Identity truth repairs

- Fix the two pre-existing alias-suite failures.
- Prevent generic ghost expiry or SID recycling from purging resident galleries.
- Mark restored visitor slots used after restart and add generation-safe visitor
  identity.
- Derive gallery counts from persisted truth rather than stale enrollment
  metadata.
- Validate camera-topology source IDs against the active camera inventory.
- Stop accepting enrollment by a recyclable integer alone; require a versioned
  run/camera/tracker/observation reference.
- Reconcile documented resident/visitor persistence with the real store.
- Make identity state directories `0700` and sensitive files `0600`; add
  deletion and retention tests.

### Product safety repairs

- Remove side-effecting GET routes, beginning with Menon catalog generation and
  Noesis depth refresh.
- Remove the page-load catalog regeneration trigger and arbitrary output/backend
  parameters.
- Remove prompt-derived success acknowledgements; displayed success must come
  from provider readback.
- Repair full pytest discovery, including the removed dewarper-mask import, and
  stop selectively ignoring most tests.
- Make capability/readiness health report actual producer progress, contract
  compatibility, and stale data rather than open ports alone.

### Gate

All focused suites and full collection pass. Current live behavior remains
available until migrated clients are validated.

## Phase 2 - Runtime-neutral product core and contracts

### Package boundaries

- Introduce a regular runtime-neutral core package with no DeepStream imports.
- Keep launchers as composition roots.
- Normalize SDK frame/object/tensor evidence into safe immutable observations;
  never retain transient Service Maker metadata wrappers.
- Migrate shared identity, world, telemetry schema, configuration semantics,
  API assembly, and validation logic into the product core incrementally.

### Versioned contracts

Define Python models, JSON Schema, fixtures, compatibility checks, and generated
or mechanically synchronized TypeScript types for:

- `ObservationEnvelope`
- `IdentityObservation` and `IdentityDecision`
- `Subject` and lifecycle/generation identity
- `WorldSnapshot` and semantic world events
- `SceneRelease`
- `CapabilityHealth`
- `ActionIntent`, `Approval`, `ExecutionReceipt`, and `AuditEntry`

Every observation includes producer/run ID, camera/source ID, frame number,
capture/observation/publish timestamps, sequence, coordinate frame, units,
calibration/model/config fingerprints, evidence quality, uncertainty,
provenance, and freshness.

### Gate

DS8 and DS9 characterization fixtures serialize equivalent product contracts;
Menon rejects unsupported major versions instead of guessing.

## Phase 3 - Local appliance gateway and control boundary

### Gateway

- Use `TauntonMainframe.local` as the default appliance name.
- Serve Menon and the operator application through one Node gateway.
- Use local HTTPS for LAN mode with generated local-CA tooling and support
  owner-provided certificates. HTTP development mode is loopback-only.
- Proxy Noesis REST and WebSocket through authenticated same-origin routes.
- Bind backend development servers and Noesis control surfaces to loopback after
  migration; keep the engineering console owner/loopback-only.

### Authentication and authorization

- First-run owner pairing with scrypt password hashing.
- SQLite-backed, server-side, revocable sessions.
- `Secure`, `HttpOnly`, `SameSite=Strict` cookies; exact Host/Origin checks;
  Fetch Metadata enforcement; session-bound CSRF tokens for mutations.
- One-use, short-lived WebSocket tickets.
- Three default-deny roles:
  - `viewer`: non-identifying home/device status only;
  - `operator`: live presence/video and routine reversible device/media actions;
  - `owner`: identity, calibration, scene, credentials, runtime, and destructive
    administration.
- Scoped internal bearer credentials for the gateway-to-Noesis boundary and
  approved non-browser tools.

### Action coordinator

All writes use:

```text
authorize -> read current state -> prepare -> confirm if required
          -> execute once -> provider readback -> receipt -> append audit
```

Confirmation tokens are actor-bound, short-lived, and single-use. Agents and
automations may propose but may never self-confirm.

### Gate

Cross-origin/CSRF attacks fail; unauthorized WebSockets fail; all mutations have
an actor, idempotency key, readback, receipt, and audit record; only the gateway
is LAN-reachable for browser/control traffic.

## Phase 4 - Household identity v2

### Resolver semantics

- Candidate families: enrolled residents, active/recent visitors, and one
  explicit unknown option per physical tracklet group.
- Hard masks: model/dimension mismatch, evidence quality, topology,
  exclusivity, time, world reachability, and overlap geometry.
- Absolute resident open-set similarity floor and top-1/top-2 ambiguity margin.
- A bounded resident prior can break close admissible ties but cannot lower the
  absolute open-set floor.
- Established residents use hysteresis and multiple observations; one
  contradictory crop cannot flip identity.
- Keep raw similarity, calibrated confidence, ambiguity margin, prior
  contribution, evidence count, and reject reason for every decision.

### Global assignment

- Group legitimate simultaneous overlap-camera observations before assignment.
- Build the full tracklet-group by subject score matrix.
- Add one unknown dummy per group.
- Apply hard masks before Hungarian assignment.
- Production hooks make one safe primitive-copy pass and submit batches; no
  greedy production assignment remains.
- Shadow comparison is diagnostic-only and time-bounded; cutover is explicit.

### Subject and persistence model

- Tracker-local key: `(run_id, camera_id, tracker_id)`.
- Durable resident `subject_id`: resident UUID.
- Visitor `subject_id`: session UUID plus non-repeating generation.
- Provisional/unknown `subject_id`: tracklet UUID.
- Numeric `stable_id` remains a compatibility display field during migration.
- Use a versioned SQLite identity store loaded into memory for hot matching.
- Preserve immutable enrollment anchors; online exemplars enter quarantine and
  require repeated corroboration before promotion.
- Visitor/provisional embeddings expire and are not included in cutover backups.

### Enrollment and migration

- Two-step propose/confirm enrollment with exact observation identity and stale
  checks.
- Normalize names and surface existing-resident updates rather than silently
  creating duplicates.
- Produce an idempotent dry-run migration report for duplicate names, empty
  galleries, model mismatches, visitor collisions, and stale metadata.
- Archive before migration and require owner confirmation for ambiguous resident
  records.

### Gate

Assignment is order-invariant. The visitor-negative replay has zero forced
resident decisions. Resident convergence, handoff, overlap, restart, latency,
and DS8/DS9 parity gates pass. Identity health is non-green when resident
anchors are absent or embedding evidence collapses.

## Phase 5 - Canonical global world

### Backend fusion

- Consume versioned per-camera observations by producer time, not browser
  arrival time.
- Group by durable subject identity while preserving raw observations.
- Fuse compatible contemporaneous positions by declared uncertainty.
- Never average contradictory cameras; choose the best supported observation,
  emit a conflict, and preserve provenance.
- Publish entity lifecycle, position/covariance, velocity, room/topology state,
  identity state/confidence, contributing sources, freshness, and conflict
  evidence in `WorldSnapshot`.
- Persist a bounded local replay/event history with explicit retention.

### Consumer cutover

- Menon consumes global entities as its production presence path.
- Raw per-camera tracks remain visible only in operator diagnostics.
- Remove source-camera election, browser-arrival freshness, range compression,
  and room-boundary clamping from production placement.
- BEV and Menon validate against the same backend entity state.

### Gate

Replay and live Tier 3/4 checks prove timestamp ordering, bounded world error,
no wall teleports, consistent BEV/Menon placement, clean empty-frame removal,
and explainable multi-camera conflict handling.

## Phase 6 - Coherent scene releases

- Use one calibration service/library for live tracking and offline builds.
- Build immutable multi-camera scene releases containing selected reconstruction
  revisions, calibration/model/config fingerprints, time cohort, authored-scene
  alignment, the complete OBJ/MTL/texture dependency graph, metrics, and
  validation evidence.
- Validate all selected revisions as one compatible cohort.
- Promote and roll back releases atomically; never infer product truth from an
  individually newest per-camera revision.
- Require owner promotion for retained RGB evidence and apply the documented
  retention policy.
- Make Menon render one promoted release and surface incompatibility as a hard
  error.

### Gate

Atomic promotion, rollback, fingerprint mismatch, mixed-cohort rejection,
artifact path safety, and Menon browser readback tests pass.

## Phase 7 - Menon product services and governed agent workflows

### Menon boundaries

- Typed services/stores for authentication, world state, devices, media,
  scenes, rendering, and workflows.
- Remove global browser state from the new canonical paths before touching
  unrelated legacy UI.
- Keep developer, operator, and household controls visibly separated.
- Preserve the authored home and current provider integrations behind adapters.

### Agent broker

- Launch the reasoning process with an allowlisted environment, dedicated
  profile/work directory, read-only sandbox, concurrency/time/output bounds, and
  server-owned conversation IDs.
- Give the model typed read capabilities and action-proposal capabilities only.
- Route proposed mutations through the action coordinator.
- Derive the visible outcome from execution receipts and provider readback, not
  generated acknowledgement text.
- Persist workflow state, approvals, receipts, failure/uncertain results, and
  audit events.

### Gate

The agent cannot mutate without coordinator authorization, cannot self-approve,
cannot inherit unrelated host secrets, and cannot claim success without a
matching successful readback receipt.

## Phase 8 - DS8/DS9 convergence and artifact provenance

- Add an automated ownership/parity matrix and reject unclassified drift.
- Remove namespace-path selection as an implicit ownership mechanism.
- Keep SDK adapters responsible only for graph construction, metadata access,
  native ABI, assets, and lifecycle.
- Reproduce prior cutover failures as tests: WebRTC refresh ownership, RTSP
  late-viewer keyframes, stale user-site bindings, missing fresh depth,
  reconnect loops, native bridge incompatibility, and shutdown hangs.
- Build an immutable DS9 environment without changing the live DS8 host stack.
- Manifest every engine/parser/plugin/native artifact with SDK, CUDA/TensorRT,
  source checksum, output checksum, precision, batch, tensor contract, and build
  command.
- Rebuild all DS9 binaries; never reuse DS8 engines or native objects.
- Add DS9-native V3DT and run the same identity/world contracts.
- Package and validate DS8 as the rollback runtime on the future DS9-capable
  driver before host cutover.

### Gate

No production capability is missing, skipped, or silently imported from a
DS8-specific adapter. Replay quality is at least DS8 baseline; real RTSP,
decoded WebRTC, identity, world, depth, floorplan, V3DT, resource, persistence,
and shutdown gates pass.

## Phase 9 - CI, canary, soak, and cutover

- Track the complete relevant test suite.
- Add Python and Node environment contracts; Menon builds/tests on Node 22.
- Add schema/drift, unit, replay, security, browser, docs, and packaging CI.
- Keep hardware tests as explicit host/container gates, never fake CI passes.
- Run independent adversarial reviews for identity, security, world geometry,
  DS9 native boundaries, and rollback.
- Execute DS9 on canonical ports only in scheduled single-runtime canaries.
- Canary progression: supervised two hours, 24-hour soak, then seven-day
  burn-in.
- Automatic rollback triggers include native crash, media freeze, repeated
  source resets, stale world state, incompatible persistence, confirmed identity
  false-share, or a failed critical gate.
- Rollback stops DS9, preserves its evidence bundle, starts the validated DS8
  runtime on the same ports, and verifies media/world/API health.

## Subagent orchestration

The root agent is the architecture integrator and owns shared contracts,
cross-lane decisions, merges, live processes, and final validation. Three
parallel lanes are used per wave.

### Wave A - completed read-only audits

- Identity architecture and live truth.
- DS9 ownership, artifact, prior-cutover, and parity audit.
- LAN security, gateway, action, privacy, and agent-control audit.

### Wave B - truth repair and foundations

- Identity owner: `reid/`, identity tests, and identity-specific plans.
- Security owner: Menon gateway/security/control modules and focused tests.
- DS9 owner: ownership matrix, drift tooling, DS9 docs, and parity
  characterization tests.
- Root: product-core contracts, workstream docs, test discovery, and integration.

### Wave C - product truth

- Identity owner: scorer, resolver, state migration, and enrollment contracts.
- World owner: observation envelopes, fusion, semantic events, and telemetry.
- Menon owner: authenticated clients, typed world store, and canonical render
  cutover.
- Root: contract compatibility, replay integration, and cross-space gates.

### Wave D - twin and actions

- Scene-release owner: atomic release store/API and reconstruction integration.
- Control owner: provider adapters, action coordinator, audit, and agent broker.
- DS9 owner: shared-core adapters, artifact manifests/builds, and V3DT.
- Root: integration, threat review, and live migration.

### Wave E - independent verification and cutover

- Adversarial identity/replay reviewer.
- Security and authorization reviewer.
- DS8/DS9 media/native/resource reviewer.
- Root: final Tier 1-4 evidence, canary, soak, rollback/cutover, docs, and
  handoff.

Only one agent may edit `reid/stable_id_manager.py` at a time. Only one agent
may edit either giant hook implementation at a time. Agents own disjoint paths,
return evidence before integration, and re-read changed contracts before the
next wave.

## Checkpoint boundaries

1. Baseline ledger and workstream governance.
2. Truth/safety repairs with green baseline tests.
3. Product-core contracts and drift gates.
4. Gateway/auth/action foundation.
5. Identity v2 shadow and replay evidence.
6. Identity v2 production cutover.
7. Global world snapshot and Menon cutover.
8. Atomic scene releases.
9. Governed agent workflows.
10. DS9 parity/artifact/container completion.
11. DS9 canary and soak evidence.
12. Final docs, migration, rollback, and maintainer handoff.

Every checkpoint stages only its owned files, runs `git diff --cached --check`,
records exact validation, and leaves unrelated pre-existing work untouched.

## Principal risks

- Empty or poisoned resident galleries make threshold tuning meaningless.
- Real household ground truth requires opt-in local capture and human labels.
- The current live runtime predates some dirty fixes, so code and process truth
  must be distinguished during every measurement.
- Moving browser clients behind one gateway is a coordinated contract migration.
- DS9 currently lacks its required host SDK version and complete artifact set.
- The RTX 3060 should not run both full graphs concurrently.
- Calibration correction can create visually plausible but physically false
  output; acceptance must use external anchors and reprojection evidence.
- A driver/SDK cutover can invalidate DS8 rollback unless the rollback runtime is
  packaged and tested first.

## Explicitly avoided work

- No detector, ReID backbone, Three.js, DeepStream graph, or HomeSeer replacement
  before evidence identifies it as the bottleneck.
- No microservice decomposition for its own sake.
- No client-side smoothing, range constants, or room clamps used to conceal
  backend world defects.
- No agent permission expansion before the action boundary is deterministic.
- No DS9 promotion by bulk-copying DS9 into the root tree.
- No DS8 binary or config fallback in DS9.
