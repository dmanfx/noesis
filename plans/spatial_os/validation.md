# Spatial OS Validation Contract

Every phase records command, artifact, status, failure category, runtime, and
whether evidence is fixture, replay, live DS8, live DS9, or live Menon.

## Baseline gates

- Full relevant Python collection succeeds.
- Focused identity, world, virtual-twin, REST/WS, security, and validation tests pass.
- Menon tests, type check, and production build pass on Node 22.
- Docs consistency and `git diff --check` pass.

## Security gates

- Only the appliance gateway is browser/control reachable from the LAN.
- All mutations require an authenticated authorized actor and CSRF-safe request.
- WebSocket upgrades require an authenticated one-use ticket.
- Cross-origin, stale ticket, replayed confirmation, and role escalation tests fail closed.
- Every mutation has idempotency, provider readback, receipt, and audit evidence.
- Agent processes receive an allowlisted environment and cannot self-authorize.

## Identity gates

- Assignment is invariant to input ordering.
- Curated visitor-negative replay produces zero forced resident decisions.
- A resident prior never changes an otherwise rejected observation into acceptance.
- No two incompatible physical groups receive one subject.
- Valid overlap co-visibility can share one resident subject.
- Resident enrollment anchors survive ghost expiry, restart, and visitor pressure.
- Visitor subject generations never collide across slot reuse.
- Empty/mismatched galleries and missing embeddings make health non-green.
- Resolver p50/p95 latency and detection-wake cost stay within recorded budgets.
- DS8 and DS9 return equivalent identity decisions for identical observations.

## World gates

- Producer sequence and observation timestamps are monotonic per source/run.
- Unsupported contract versions fail visibly.
- Fused positions preserve contributing observations, uncertainty, and conflicts.
- Contradictory cameras are not silently averaged.
- Empty source frames remove presence according to producer lifecycle.
- BEV and Menon consume the same entity state and agree within declared tolerance.
- Camera reprojection, known anchors, room bounds, walls, and doorway checks pass.

## Scene-release gates

- Mixed calibration/model/time cohorts are rejected.
- Promotion and rollback are atomic.
- Artifact traversal, unsafe path components, symlink/hardlink/non-regular
  substitution, replacement during reads, and extra release-tree content are
  rejected without chmod repair.
- Manifest/artifact/OBJ/MTL size, line, camera, artifact, dependency, total-file,
  and total-byte bombs fail before publication.
- Release directories publish from a complete staged tree with atomic
  no-replace semantics; an injected publication failure leaves no visible
  partial release.
- Promotion/current metadata/current payload validate the complete cohort.
  Individual binary routes verify only their selected exact bytes and never
  reopen a path after verification.
- DS8, V3DT, and DS9 mount the same shared scene router.
- Menon readback identifies exactly one promoted release and its fingerprints.

Focused gate:

```bash
python3 -m pytest \
  tests/test_noesis_core_scene_store.py \
  tests/test_scene_release_builder.py \
  tests/test_scene_release_adversarial.py \
  tests/test_scene_api.py -q
python3 scripts/export_noesis_core_schemas.py --check
node ../Menon/scripts/validate-scene-release-consumer.mjs \
  --release data/virtual_twin/releases/home_rgbmesh_20260623T2158_v1.json \
  --virtual-twin-root data/virtual_twin
```

The last command is a read-only candidate/consumer gate. It does not register
or promote the candidate and must continue to report 34 verified requests, 3
cameras, 27 camera artifacts, and 5 authored dependencies.

_2026-07-10 result: 31/31 focused scene tests passed; 92/92 broader
core/contracts/private-path/catalog tests passed; shared DS9 mount checks
passed 5/5; generated schemas and docs consistency passed. The unmodified
`home_rgbmesh_20260623T2158_v1` candidate passed strict store validation and the
Node 22 consumer reported 34 verified requests, 3 cameras, 27 camera artifacts,
1 MTL, 4 textures, and no diagnostic virtual-twin route._

## DS9 gates

- DS9 imports no DS8 runtime/preflight implementation as implicit fallback.
- All native/engine artifacts match the DS9 manifest and SDK ABI.
- Real RTSP DESCRIBE, RTP, decoded WebRTC frames, fresh depth/floorplan, identity,
  world, V3DT, resource, persistence, and clean shutdown pass.
- Prior cutover failure regressions pass.
- DS9 replay quality is no worse than the accepted DS8 baseline.
- DS8 rollback is verified on the future DS9-capable platform before cutover.

## Cutover gates

- Two-hour supervised canary passes.
- Twenty-four-hour soak passes.
- Seven-day burn-in passes.
- Automatic rollback triggers and evidence preservation are rehearsed.
