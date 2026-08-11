# Spatial OS Work Order

Execution authority: `implementation_plan.md`.

## Phase 0 - Governance and baseline

- [x] Record live process, listener, SDK, worktree, contract, health, and test baselines.
  _2026-07-09 (Codex): Captured `baseline_2026-07-09.md`; focused spatial and Menon suites passed, identity had 2 pre-existing failures, full collection stopped after 443 tests on a stale post-rollback dewarper-mask import, and live identity/security/runtime evidence was recorded without restarting services._
- [x] Classify dirty-tree ownership and checkpoint boundaries.
  _2026-07-09 (Codex): Recorded the pre-existing Noesis and Menon dirty groups in `baseline_2026-07-09.md`; all checkpoints use explicit path staging and the live DS8 process remains outside source-validation claims._
- [x] Add DS8/DS9 ownership matrix and initial drift report.
  _2026-07-10 (Codex): Added a machine-readable matrix covering 46 modules, 27 capabilities, and all 23 duplicated modules, plus AST/regex drift enforcement. Structural validation passes; strict parity currently reports seven documented blockers. The former BEV-frame blocker was corrected to adapter-specific, and DS9 now has DS8-equivalent DESCRIBE readiness plus bounded on-demand WebRTC gateway allocation with 25 DS9 tests passing._
- [x] Record initial ADRs, SLOs, risks, and rollback rules.
  _2026-07-09 (Codex): `decisions.md`, `implementation_plan.md`, and `validation.md` now define the appliance, open-set identity, world ownership, action, scene-release, DS9, privacy, validation, and rollback contracts._

## Phase 1 - Truth and safety repair

- [x] Repair full test discovery and pre-existing identity failures.
  _2026-07-10 (Codex): Restored an explicit calibrated dewarper-validity contract shared by canonical and V3DT MapAnything processors; the two alias/copresence regressions are fixed. Added a root pytest contract so the default canonical suite is `tests/`, while vendored/prototype/DS9 suites remain explicit gates instead of being accidentally imported together. Focused validity tests: 5 passed; focused identity selection: 83 passed._
- [x] Repair resident retention, visitor restart/generation, health truth, and topology validation.
  _2026-07-09 (Codex): Household maintenance no longer purges resident galleries; restored visitor slots are reconciled with persisted galleries and carry generations; resident counts derive from live gallery truth; runtime camera maps can be validated fail-fast against topology._
- [x] Secure identity persistence modes, deletion, and retention.
  _2026-07-10 (Codex): Added the versioned owner-only SQLite identity store with cascading biometric deletion, immutable resident anchors, quarantined adaptive exemplars, generation-safe visitor sessions, row-derived health, explicit TTL retention, migration inspection/application, restart tests, and 34 passing identity-v2 tests._
  _2026-07-10 (Codex): Hardened the shared private-path authority to reject
  symlinks in every existing path component, not only the leaf. Cross-store
  tests cover journals, scenes, identity, and atomic private files without
  chmod repair or writes through the linked path._
- [x] Remove embedded integration credentials and define fail-closed activation.
  _2026-07-10 (Codex): Verified MQTT/Influx publishing is dormant in DS8, V3DT,
  and DS9; removed source defaults for both credentials; added owner-only,
  no-follow secret-file reads at the optional publisher boundary, file-path-only
  deployment overrides, transactional startup failure, and 22 focused tests.
  Exact retired values were also redacted from a stale ignored crash log while
  preserving its other forensic content/timestamp and setting mode `0600`.
  Influx was rotated into separate operator and bucket-scoped depth-writer
  credentials, MQTT was rotated, replacement authentication was verified, and
  both retired credentials were verified unusable without exposing values._
- [x] Remove side-effecting GET routes and prompt-derived success.
  _2026-07-10 (Codex): Catalog generation and depth refresh are explicit POST
  mutations; browser HomeSeer access is server-typed; generic Noesis proxies
  are read-only; and generated acknowledgements cannot become action success.
  Governed mutations require an exact coordinator adapter and provider
  readback._
- [x] Add truthful capability/readiness health.
  _2026-07-10 (Codex): The shared progress monitor is wired to real observation/world publication in DS8, V3DT, and DS9 and exposed as strict `GET /api/v1/health/capabilities`; stale/expired/unknown and API tests pass, and open ports never produce health._

## Phase 2 - Product core and contracts

- [x] Add runtime-neutral product-core package and adapter interfaces.
  _2026-07-09 (Codex): Added the regular `noesis_core` package with no DeepStream imports and immutable identity, observation, world, scene, action, audit, and health contracts plus a deterministic global-world fusion core._
- [x] Add versioned Python/JSON/TypeScript contracts and compatibility gates.
  _2026-07-10 (Codex): Strict Pydantic contracts now require explicit name/version and coordinate semantics; deterministic JSON Schema and read-only TypeScript are generated from the same source, checked for drift, and exercised by a v1 characterization fixture._
- [x] Add producer/run/sequence/timestamp/fingerprint/uncertainty metadata.
  _2026-07-09 (Codex): Observation and world contracts now require producer/run identity, monotonic sequence, capture status/time, frame and source identity, coordinate frame, artifact fingerprints, provenance, covariance, confidence, and freshness._
- [x] Add deterministic replay format and fixtures.
  _2026-07-10 (Codex): Added owner-only atomic NDJSON replay archives with explicit format version, validated contract payloads, contiguous sequencing, deterministic bytes, and a SHA-256 integrity chain; tamper/unsupported-contract tests pass._

## Phase 3 - Appliance gateway and action boundary

- [x] Add local TLS, owner pairing, sessions, roles, CSRF/origin, and WebSocket tickets.
  _2026-07-10 (Codex): Menon now has scrypt owner bootstrap, hashed SQLite sessions, exact-origin/CSRF/Fetch-Metadata enforcement, throttling, viewer/operator/owner policy, one-use confirmation challenges, persistent idempotency, one-use WebSocket tickets, TLS-only production mode, and browser authentication gates. Adversarial gateway/security tests pass._
- [x] Proxy Menon, operator UI, and Noesis through one same-origin gateway.
  _2026-07-10 (Codex): Fixed REST/WS policy routes inject only the internal
  token, strip browser credentials, revoke media on session loss, expose only
  sanitized current-scene data, and deny uncoordinated mutations._
- [x] Add internal Noesis authentication and route/message policy.
  _2026-07-10 (Codex): DS8, V3DT, and DS9 REST/WS require an owner-only internal bearer by default; wildcard/regex CORS and URL/query credentials are forbidden. Shared control defaults are loopback and exact WS/REST endpoints now fail instead of selecting alternate ports or trusting unrelated listeners. Explicit disabled development auth is executable only on literal loopback/localhost, and depth telemetry replaces internal storage paths with opaque artifact IDs before transport; focused auth/parity/privacy tests pass._
  _2026-07-10 (Codex): Closed the validation-client side of the boundary. The
  DS9 live orchestrator now supplies its existing owner-only token file to all
  HTTP/WS smokes, every client sends a Bearer header and fails closed, and no
  token enters a URL or command. Focused client/runner tests and all 80 DS9
  tests pass._
- [x] Add provider adapters, action coordinator, readback receipts, and audit.
  _2026-07-10 (Codex): Menon mutations now enter a durable coordinator with actor, persistent idempotency, one-use confirmation, execution, readback verification, explicit success/failed/uncertain receipts, and redacted owner-only audit. HomeSeer, scene, Cast, identity-v2, and approved diagnostics have typed adapters; the generic proxy is read-only._
- [ ] Close direct LAN browser/control listeners after client migration.
  _2026-07-10 (Codex): Staged the hardened six-unit user-systemd graph,
  generated and validated the appliance CA/server certificate, provisioned a
  non-disclosing one-time owner seed, and installed the content-addressed units
  disabled/inactive. Real wrappers pass preflight. Read-only live audit confirms
  `eno1` / `192.168.3.0/24` and a conflict-free UFW/Avahi plan; no firewall,
  Avahi, listener, or live-process change has occurred. Client CA trust and the
  coordinated maintenance-window cutover remain the gate._
  _2026-07-10 (Codex): Added authenticated continuous capability/HTTPS and
  health-only WebSocket supervision with independent failure budgets. The
  target is fail-closed on guard loss, and fresh install, staged upgrade,
  interrupted upgrade, cutover, and interrupted-cutover operations are
  kernel-lock-serialized and journal-recoverable. Adversarial review cleared
  the real v1-to-v2 inactive upgrade; bundle v2 is installed with 13 exact
  files, all six units remain inactive, the target remains disabled, and the
  four live manual processes are unchanged. A disposable systemd-255 runtime
  test also proved uphold restart, guard-to-target stop propagation, and
  target-to-child `PartOf` stop propagation; all transient probes were removed._

## Phase 4 - Household identity v2

- [x] Add resident-biased open-set scorer and calibrated decision evidence.
  _2026-07-10 (Codex): Added a dependency-free scoring kernel with quality, absolute appearance, hard-constraint, calibrated-confidence, and pre-prior ambiguity gates. The bounded resident prior affects assignment utility only and cannot rescue rejected evidence; raw/calibrated/margin/prior/reason evidence is retained._
- [x] Add batch overlap grouping and Hungarian assignment with unknown dummies.
  _2026-07-10 (Codex): Added deterministic exact joint assignment with one independent unknown option per tracklet, one-to-one identity claims, and identity-specific pairwise overlap permits; 16 scorer/resolver tests pass. Physical overlap grouping and safe one-pass DS8/DS9 hook integration remain._
  _2026-07-10 (Codex): Completed the shared one-pass DS8/V3DT/DS9 source-frame
  adapter. Physical share permits now require strict topology plus fresh world,
  time, and appearance proof; the same proof is never reused as a sticky exemption._
- [x] Add durable subject IDs and generation-safe visitor sessions.
  _2026-07-10 (Codex): Identity v2 stores durable resident UUIDs, generation-safe visitor sessions, and provisional state independently of recyclable display SIDs; migration and restart coverage pass._
- [x] Add protected enrollment anchors, quarantine, and versioned identity store.
  _2026-07-10 (Codex): Standard-library SQLite v1 enforces model/dimension compatibility, immutable float32 anchors, independently corroborated quarantine promotion, private modes, atomic transactions, deletion, retention, and row-derived health._
- [x] Add stale-safe two-step enrollment and migration report.
  _2026-07-10 (Codex): Idempotent legacy inspection/apply and provenance blocking are complete; exact-observation two-step enrollment and typed API are active Wave E work._
  _2026-07-10 (Codex): The authenticated v2 API now accepts only an exact
  server observation key plus naming intent, consumes bounded immutable evidence
  once, and confirms against the same key/digest. Client biometric vectors are
  rejected; no partial legacy migration was applied._
- [ ] Pass replay, live health, performance, and DS8/DS9 parity gates.
  _2026-07-10 (Codex): Unit/API/hook/static parity gates pass. Default mode is
  shadow; authoritative remains gated on an artifact-backed scoring calibration
  plus live/replay evidence, so this rollout gate correctly remains open._
  _2026-07-10 (Codex): Score-only tamper-evident evidence, disjoint held-out
  calibration, exact post-resolution OSD, owner UI, and migration review are
  complete. Identity selection passed 125 tests and parity/private/benchmark
  selection passed 17; real multi-session household labels remain required._

## Phase 5 - Canonical global world

- [x] Publish versioned per-camera observation envelopes.
  _2026-07-10 (Codex): DS8, V3DT, and DS9 now normalize public person tracks into strict observation v1 with run/source/frame/time/PTS, coordinate, uncertainty, and content-addressed calibration/model/config provenance._
- [x] Add backend global-person fusion, uncertainty, conflicts, and semantic events.
  _2026-07-10 (Codex): Shared global fusion enforces source/run ordering, uncertainty-weights compatible evidence, preserves contradictory cameras without averaging, and emits appeared/held/resumed/lost/conflict lifecycle events. Exact tracklet ownership now makes provisional/unknown-to-resolved identity transitions atomic, preventing the prior subject and new resident from coexisting until TTL; explicit open-set unknown remains distinct from provisional. Eighteen focused world tests pass._
- [x] Publish canonical world snapshots and bounded replay history.
  _2026-07-10 (Codex): Both runtimes publish strict world snapshots plus events over WS and persist observations/snapshots/events in a private bounded integrity-chained SQLite journal. Retention now prunes only a contiguous chain prefix even when evidence timestamps arrive out of order, and failed asynchronous writers are stopped cleanly during teardown; 19 focused world/replay/journal tests pass._
  _2026-07-12 (Codex): Superseded asynchronous runtime-world retention with a
  synchronous exact-count acknowledgement and release-gated outbox. A world
  batch cannot begin client delivery before journal and private fusion commit;
  commit failure aborts with zero delivery, and inspection exposes only the
  immutable committed snapshot._
- [x] Cut Menon and BEV to canonical entity state.
  _2026-07-10 (Codex): Menon now accepts only strict monotonic `noesis.world.snapshot` v1, removes entities immediately when absent, presents each entity through exactly one explicit world-to-scene transform, and keeps per-camera tracks as diagnostics rather than presence truth._
- [x] Retire production browser fusion/range/clamp behavior.
  _2026-07-10 (Codex): Source election, browser-arrival freshness, range correction, clamping, smoothing, and per-camera occupancy were removed from the authoritative presentation path. The browser boundary tests prove they cannot silently re-enter production placement._

## Phase 6 - Coherent scene releases

- [x] Unify live/offline calibration authority.
  _2026-07-10 (Codex): DS8, V3DT, DS9, reconstruction, and depth-registration
  paths now use one exact-parity CalibrationManager factory. The alternate
  private providers and duplicate camera-label loaders were removed. All three
  real camera snapshots match the former proven K/E/image-space numerically,
  while the canonical bundle also retains scene similarity; 35 focused tests
  passed._
- [x] Add immutable multi-camera scene release schema/store/API.
  _2026-07-10 (Codex): Immutable scene v1, private SQLite registry, owner-routed REST surface, current payload, atomic compare-and-swap promotion/rollback, and exact current-release URLs are implemented for DS8/DS9. The release now closes over every rendered camera artifact plus the release-owned authored OBJ, every MTL/texture dependency, and validation report by safe path, byte length, and SHA-256; promotion/current reads fail closed on mutation and bundle output refuses release-ID drift._
  _2026-07-10 (Codex): Adversarial hardening made registration, promotion,
  building, hashing, and serving share bounded no-follow/inode-checked file
  semantics; rejects links, non-regular/empty files, unsafe components, read
  replacement, oversized/dependency-bomb input, and extra tree content; and
  publishes complete release directories atomically without overwriting or
  chmod-repairing existing state. Binary routes verify one selected snapshot
  instead of rehashing the complete cohort per request. The focused scene gate
  passes 31 tests; the broader core/contracts/private-path/catalog gate passes
  92; schema/docs checks pass; and the unchanged real candidate passes all 34
  read-only Menon consumer requests._
- [x] Add cohort validation, atomic promotion, rollback, and retention.
  _2026-07-10 (Codex): The explicit-revision builder separates shared bundle and per-camera fingerprints, recursively inventories OBJ material/texture references, verifies cohort/manifest identity and all selected bytes, and rebuilt the unpromoted three-camera candidate with 27 camera artifacts, one OBJ, one MTL, and four textures. Its embedded and top-level validation bytes exactly match SHA-256 `a5773a3a443b416ff2e9d2f75ea61a6002082d9017ab01f941853609296abeb2`; 28 focused scene contract/store/API/builder tests pass._
- [x] Cut Menon reconstruction rendering to one promoted release.
  _2026-07-10 (Codex): Menon now loads one strict current scene release, verifies SHA-256 and length for the OBJ, all MTL/textures, and all camera artifacts, decodes the base and every enabled overlay off-scene, then commits one synchronous application cohort with rollback. It performs no legacy `/virtual-twin/` fetch. The real unpromoted candidate passed 34 verified requests for 3 cameras, 27 camera artifacts, and 5 authored dependencies._

## Phase 7 - Menon services and agent workflows

- [x] Add typed bounded world/device/media/scene/render/workflow services.
  _2026-07-10 (Codex): Menon now owns strict canonical-world, HomeSeer, Cast,
  scene-release, identity, rendering-cohort, conversation, proposal, receipt,
  and audit services with schema, byte, count, time, and concurrency bounds._
- [x] Separate household, operator, and developer surfaces.
  _2026-07-10 (Codex): The authenticated gateway enforces default-deny
  viewer/operator/owner roles, keeps engineering surfaces loopback/owner-only,
  strips identity and raw calibration from operator payloads, and emits no
  private household assets in the public production bundle._
- [x] Add isolated read-only agent broker and typed proposals.
  _2026-07-10 (Codex): The production broker uses a dedicated Codex profile,
  bubblewrap, an allowlisted environment, disposable bounded turn state, typed
  reads, and typed proposals only. Host secrets, tools, plugins, MCP, shell,
  repository, and direct provider mutation paths are absent._
- [x] Route all mutations through policy/approval/execution/readback/audit.
  _2026-07-10 (Codex): HomeSeer, Cast, scene, identity, and approved runtime
  writes pass through actor-bound idempotency, one-use confirmation where
  required, exact provider readback, durable receipts, and redacted bounded
  audit. Full Menon validation after orchestration/owner-seed review passed 199 tests,
  TypeScript/Vite production build, and full/production dependency audits with
  zero vulnerabilities._

## Phase 8 - DS8/DS9 convergence

- [ ] Converge shared product behavior and explicit SDK adapters.
  _2026-07-10 (Codex): World observations/events/journaling, health, security, scene-release behavior, and RTSP late-viewer readiness now share product semantics across DS8 and DS9; exact publisher drift is gated. Seven declared identity/ground-state/model/V3DT/artifact blockers remain._
  _2026-07-10 (Codex): Person-ground posture/support/motion/path behavior now has one shared owner with explicit DS8/DS9 SDK adapters and parity characterization. Wholebody49 source/config/parser parity is also ported through one semantic owner with DS9-owned artifacts; its two engines and occupied-scene quality evidence remain gated. Strict ownership now reports five blockers._
- [ ] Reproduce prior cutover failures as regression tests.
- [ ] Build provenance-complete DS9 container and artifacts.
  _2026-07-10 (Codex): Added artifact manifest v2, JSON Schema, ownership/compatibility/checksum/provenance validator, and static gates. The MapAnything spec now requests true FP16 and has a regression gate, but the blocker remains until a provenance-complete engine passes depth/floorplan quality validation. The manifest truthfully reports missing artifacts/incomplete provenance; no live-host build or unverified reuse occurred._
  _2026-07-10 (Codex): Staged and hash-verified both Wholebody49 ONNX sources in DS9-local and external artifact ownership, built the 40,928-byte parser against DS9 headers in an isolated CPU-only `runc` container, recorded provenance, and dry-ran both guarded TensorRT commands with no GPU devices. The manifest now records 14 staged binaries and 22 missing engines; no live DS8 process or GPU owner was disturbed._
  _2026-07-12 (Codex): Superseded the earlier FP16 intent after exact real-frame inference proved the loadable DS9 FP16 MapAnything engine emitted non-finite depth, zero masks, and confidence sentinels. The reviewed authority now requires correctness-first FP32 plus a sealed, independently replayed real-inference functional-quality receipt before atomic installation. The guarded replacement realization and all-camera live floorplan/resource gates remain open._
- [ ] Implement and validate DS9-native V3DT.
  _2026-07-10 (Codex): Completed the CPU/source boundary: DS9 now owns an unpadded locked SV3DT pipeline, camera/camInfo/tracker configs, strict external-artifact validation, runtime materialization, native bridge binding, NvMOT engine helper, and default smoke. Both isolated no-GPU engine plans pass; the helper compiles against DS9 headers with `-Werror`; focused V3DT tests pass. This item remains open until an exclusive-GPU window builds BodyPose3DNet then tracker ReID and live bbox3d/world/identity/resource/shutdown gates pass. Global-world parity separately requires accepted shared metric calibration._
- [x] Package/test DS8 rollback on the DS9-capable platform.
  _2026-07-10 (Codex): The pre-driver recovery boundary now contains complete
  Noesis/Menon Git bundles, exact dirty source snapshots, the active DS8 engine
  and native-bridge bytes, boot/DKMS/package/process state, prior DS9 upgrade
  evidence, 21 verified Ubuntu 595 target packages, and all 33 verified 580/EGL
  rollback packages on private NVMe storage. Historical evidence proves the
  June rollback followed MapAnything output/quality regression rather than a
  demonstrated 595 compute failure. This item remains open until the new
  595 reboot passes DS8 regression and an offline 580 rollback is rehearsed or
  executed against the DS9-capable platform._
  _2026-07-10 (Codex): Closed the reversible platform gate without mutating the
  host. The post-reboot 595 platform passed all configured DS8 engine loads,
  the full 785-test canonical suite, and the authenticated 30-second baseline
  lifecycle gate. The exact offline rollback rehearsal then rehashed all 845
  checkpoint files/9,061,703,053 bytes, all 33 Debian archives plus dependency
  relationships and architectures, current 595 module/DKMS/boot coherence,
  and a local-only no-download APT transaction of 25 installs/21 removals with
  no CUDA/TensorRT/DeepStream change. This is validated rehearsal evidence, not
  an executed rollback; a future triggered rollback still requires the
  disruptive reboot and complete post-rollback DS8 acceptance contract._

## Phase 9 - Operational cutover

- [x] Add CI, environment contracts, replay/security/browser gates, and hardware runbooks.
  _2026-07-10 (Codex): Added pinned CPU contract dependencies, Python 3.12
  schema/docs/static/full-suite gates, Node 22 test/build/audit CI, and an
  explicitly dispatched labeled-appliance workflow for disruptive real
  DeepStream/GPU gates. A clean NVMe-hosted CPU environment collected all 662
  tests without Service Maker installed; runtime operators remain fail-loud._
- [ ] Complete independent adversarial reviews.
- [ ] Run DS9 two-hour canary, 24-hour soak, and seven-day burn-in.
- [x] Execute or rehearse evidence-preserving rollback.
  _2026-07-10 (Codex): Rehearsed the evidence-preserving 595-to-580 rollback
  non-mutating and CPU-only. The validator fails closed on checkpoint/package/
  dependency/platform/solver drift, emits the exact ordered recovery command,
  preserves the post-rollback DS8 acceptance boundary, and explicitly reports
  `rollback_executed: false`; focused tests passed 10/10. The real owner-only
  atomic evidence file is SHA-256
  `94620d38ac34f8864de27e0ddcb3d10cc650677c8f3eebe005495a10a3adcde4`._
- [ ] Cut over, update all docs, and publish final maintainer handoff.

## Validation log

- _2026-07-09: Pre-implementation baseline: 82 focused spatial tests pass; 41 Menon tests and Node 22 build pass; identity selection reports 2 failures/76 passes; full Noesis collection finds 443 tests and stops on the removed dewarper-mask helper; docs consistency and diff whitespace checks pass._
- _2026-07-10: Foundation checkpoint: calibrated dewarper validity 5 passed; household/StableID selection 83 passed; `noesis_core` contract/world fusion 14 passed. A subsequent root invocation exposed accidental vendored/prototype collection; `pytest.ini` now makes the canonical default suite deterministic and separately gated suites explicit._
- _2026-07-10: World/scene/security checkpoint: world/replay/journal focused suite 22 passed; scene contract/store/API/builder suite 18 passed; exact endpoint/zero-copy invariants 8 passed; DS9 suite 22 passed plus static preparation. A real three-camera 2026-06-23 RGB-mesh cohort validated at 30.445214 seconds capture spread, with every declared artifact hashed; it remains an unpromoted candidate until the Menon consumer cutover._
- _2026-07-10: Menon canonical world/scene checkpoint: Node 22 full suite 112/112, TypeScript/Vite production build passed, production dependency audit found zero vulnerabilities, and the real scene-release consumer verified all 34 current-release requests with no diagnostic-catalog route usage. No live service was restarted._
- _2026-07-10: Menon appliance checkpoint: post-adversarial full suite 199/199,
  TypeScript/Vite production build, full and production dependency audits (zero
  vulnerabilities), source/systemd audit, real wrapper preflights, local
  CA/server chain and key validation, private owner-seed provisioning, and
  disabled/inactive content-addressed unit installation all passed. Live manual
  DS8/Menon/Vite processes remain unchanged; network apply/cutover did not run._
- _2026-07-10: Menon appliance supervision checkpoint: health-only WS and
  lifecycle recovery gates passed 23 focused Node 22 tests plus 24 Noesis
  auth/WS tests; post-staging full Menon validation passed 209 tests, the
  TypeScript/Vite production build passed, and both dependency audits reported
  zero vulnerabilities. Adversarial review approved and the
  transactionally upgraded v2 unit bundle remains disabled/inactive; no live
  process, listener, firewall, Avahi, scene, credential, or GPU state changed._
- _2026-07-10: DS9 person-ground/Wholebody49 checkpoint: person-ground characterization 3/3 and shared regressions 13/13 passed; focused Wholebody49/manifest/ownership/runtime-config suite 26/26 passed; both isolated guarded engine plans verified exact FP16 batch-3 commands without GPU access. Wholebody49 remains a truthful runtime-quality blocker pending DS9 engines and occupied-scene validation._
- _2026-07-10: DS9 V3DT CPU-safe checkpoint: the complete DS9 suite passed
  64/64 and the focused V3DT suite passed 12/12; three locked camera/camInfo
  inventories, 604 staged source files, both no-GPU engine plans, native helper
  compilation, external-capacity enforcement, static preparation, formatting,
  and docs checks passed. The live DS8 GPU owner was not disturbed. Engine
  construction and occupied-scene gates remain blocked on a DS9-compatible host
  driver and an exclusive GPU maintenance window._
- _2026-07-10: Final pre-driver CPU checkpoint: the canonical Noesis suite
  passed 745 tests with 15 intentional skips; the independently hardened
  identity matrix passed 146/146; the explicit DS9 import-path suite passed
  64/64; Menon passed 209/209 plus a production build and zero-vulnerability
  full/production audits. DS9 static preparation, schema/TypeScript drift,
  docs consistency, and whitespace checks passed. Five GPU/runtime parity
  blockers remain explicit rather than being converted into CPU claims._
- _2026-07-10: Offline rollback rehearsal: the complete private checkpoint
  (845 files, 9,061,703,053 bytes), 33 exact 580/EGL archives (23 amd64, 9 i386,
  1 all), package relationship digests, two-kernel 595 DKMS/module state, and
  the exact no-download APT solver transaction passed. The corrected plan is 25
  local installs and 21 explicit/solver-required removals, with no protected
  SDK action. The rollback was not executed; focused validator tests passed
  10/10 and the private atomic JSON evidence hash is
  `94620d38ac34f8864de27e0ddcb3d10cc650677c8f3eebe005495a10a3adcde4`.
  A real future rollback still requires maintenance reboot plus DS8
  product acceptance._
