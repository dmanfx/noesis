# Household Identity — Work Order

Master execution checklist. Update checkboxes with dated notes as work lands.

## Program status

- Planning docs: **complete** (2026-07-08)
- Implementation: **household default-on** (2026-07-09)
- Identity v2: **correlation-aware offline gates complete; qualifying benchmark
  and real household calibration evidence pending**
- Audit open items closed; see `composer_audit_2026-07-09.md`
- Opt out: `NOESIS_HOUSEHOLD_IDENTITY=0`

## Gate overview

- [x] **G0** Phase 0 complete (`validation.md`)
  _2026-07-08: exclusivity+overlap, pose/world wire, archive, telemetry, tests._
- [x] **G1** Phase 1 complete
  _2026-07-08: provisional/visitor/resident spaces + REST enroll stub._
- [x] **G2** Phase 2 complete (unit level)
  _2026-07-08: Swin default, quality gallery, MNN; live perf spot-check still open; SOLIDER deferred._
- [x] **G3** Phase 3 complete (MVP)
  _2026-07-08: residents CRUD + health + FE name display._
  _2026-07-09: oai2-fe People drawer for enroll/rename/remove + health/suggest._

---

## Phase 0 — Stop the bleeding

Reference: `phase0_stop_the_bleeding.md`

- [x] Household mode construction + legacy state archive-on-enable
  _2026-07-08: `reid/household_state.py` + `_build_stable_id_manager` in `ds8_runtime.py` / `ds8_runtime_v3dt_reimpl.py`._
- [x] `config/camera_topology.yaml` + loader
  _2026-07-08: `reid/camera_topology.py`._
- [x] Overlap permit + exclusivity in all match paths
  _2026-07-08: kitchen↔family-room geometry permit; fail-closed otherwise._
- [x] Wire pose + world into `StableIDManager.update()` from hooks
  _2026-07-08: both hook variants pass pose/world; prior-frame world cache._
- [x] Confidence / overlap telemetry on tracking payloads
  _2026-07-08: reid_confidence/required, overlap_permit, identity_kind/state._
- [x] Embedding extract budget + priority for new/provisional tracks
  _2026-07-08: household setdefault max=4; pre-pass priority extract._
- [x] Disable pressure auto-merge in household mode
  _2026-07-08: forced off in household_mode._
- [x] Unit tests: exclusivity, overlap, archive (+ provisional fixtures)
  _2026-07-08: suite updated for visitor confirmation gates._
- [x] Docs: WS contracts + design decision entry
  _2026-07-08: WS/REST docs + `ds8_design_decisions.md` + master work orders._
- [x] Focused pytest pass recorded
  _2026-07-08: see Validation section below._

---

## Phase 1 — Closed-world

Reference: `phase1_closed_world.md`

- [x] `identity_kind` / provisional suppression
  _2026-07-08: provisional 9000+; no gallery until confirmed._
- [x] Resident vs visitor galleries + visitor TTL recycle
  _2026-07-08: visitors 1000–1031; residents 1..N via enrollment._
- [x] REST stub list/enroll residents
  _2026-07-08: GET/POST `/api/v1/reid/residents*`._
- [x] Disable V3DT `_public_stable_id` remap under household mode
  _2026-07-08: public_max_id forced 0 when household env on._
- [x] Unit tests for mint/recycle
  _2026-07-08: `test_stable_id_resident_visitor.py`, `test_stable_id_provisional.py`._
- [x] REST contract doc updates
  _2026-07-08: planned/implemented resident endpoints documented._

---

## Phase 2 — SOTA matching

Reference: `phase2_sota_matching.md`

- [x] Harden Swin default + portable ReID ini paths
  _2026-07-08: Swin default in infer.yaml; OSNet paths portable._
- [x] Quality-gated gallery + multi-exemplar clusters
  _2026-07-08: `reid/gallery_quality.py`._
- [x] Global MNN/Hungarian assignment with constraint masks
  _2026-07-08: `reid/assignment.py` MNN + frame claims._
- [x] Topology-aware handoff
  _2026-07-08: handoff relax only for topology-adjacent pairs._
- [ ] Optional SOLIDER path only if needed
  _Deferred: Swin-Tiny already default; enable only if live confusion remains._
- [x] Assignment/quality unit tests (+ perf spot-check pending live)
  _2026-07-08: unit tests green; live perf spot-check still open._

---

## Phase 3 — Enrollment product

Reference: `phase3_enrollment_product.md`

- [x] Complete enroll/list/patch/delete REST
  _2026-07-08: GET/POST/PATCH/DELETE residents + GET identity_health._
- [x] `display_name` / `resident_uuid` on wire
  _2026-07-08: hooks publish from id_diag; FE prefers display_name._
- [x] Minimal FE name display + enroll
  _2026-07-08: LegendPanel/App show display_name; enroll via REST (no heavy UI)._
- [x] Identity health endpoint + report script
  _2026-07-08: `/api/v1/reid/identity_health` + `scripts/household_identity_health_report.py`._
- [x] Docs finalization; supersede soft-merge-as-primary for household mode
  _2026-07-08: REST contracts list household endpoints; auto-merge disabled in household mode._

---

## Cross-cutting

- [x] Register workstream in `plans/DS8/ds8_master_work_orders.md`
- [x] `plans/DS8/ds8_design_decisions.md` entry for household identity program
- [x] `./scripts/check_agents_docs_consistency.py` after AGENTS/docs edits

## Identity v2 runtime cutover adapter

- [x] Add one shared DS8/V3DT/DS9 service factory with actual engine-byte
  fingerprint, explicit tensor layer/dimension, strict topology, and world run ID.
  _2026-07-10: Added process-owned store/runtime/coordinator construction;
  default shadow mode and calibrated-authoritative startup gate fail loudly._
- [x] Batch complete source-frame primitives exactly once in all three hooks.
  _2026-07-10: Hooks retain their one-shot SDK metadata walk, extract detached
  server evidence, and call the shared coordinator once including empty frames._
- [x] Add exact server-only enrollment evidence and authenticated v2 REST routes.
  _2026-07-10: `/api/v2/reid` is mounted in all authenticated main apps; proposal
  requests reject embeddings and consume bounded exact observation keys._
- [x] Add proof-only overlap permits and authoritative field replacement.
  _2026-07-10: Topology + fresh world distance + appearance are all required;
  authoritative unknown clears SID/name/UUID/generation, visitors remain unnamed
  and generational, and residents publish durable UUID/name plus compatibility SID._
  _2026-07-10: Authoritative hooks now bypass every legacy StableID mutation and
  defer SID-dependent dwell/transition work until whole-frame v2 resolution.
  Evidence rejection remains explicit unknown, missing evidence is provisional,
  and the initial metadata walk is neutral before an exact downstream OSD join._
- [x] Close the v2 store on runtime shutdown and gate DS8/DS9 parity.
  _2026-07-10: DS8, V3DT, and DS9 share lifecycle/API/product behavior; focused
  adapter/API/hook/parity tests passed without restarting the live runtime._
- [x] Close occupied-runtime tracker continuity, provenance, and DS9 household-policy drift.
  _2026-07-11: A tracker-local accepted subject is now hard-locked until its
  complete state expires, without bypassing open-set rejection; exact-frame v2
  evidence owns `embedding_present` and persisted provenance; DS9 now admits
  the same default-on no-auto-merge household StableID policy as DS8. Focused
  coordinator/service/DS9 parity regression tests passed without launching a
  runtime or modifying the sealed failed evidence._

## Identity v2 calibration and owner product

- [x] Add strict score-only shadow/replay evidence capture and validation.
  _2026-07-10: Added owner-only JSONL capture with deterministic event digests,
  exact model/session provenance, duplicate/tamper rejection, and no vectors._
- [x] Add deterministic disjoint calibration and authoritative artifact gate.
  _2026-07-10: Added subject/session leakage checks, sample/independence minima,
  logistic fitting, policy search, train/holdout FAR/FRR/unknown/prior metrics,
  and strict runtime artifact validation._
  _2026-07-10 adversarial hardening (superseded by the correlation correction
  below): Product-owned authority minima were 50
  known, 300 unknown, and three independent groups per partition; FAR and
  misidentification are at most 1%, FRR at most 35%, and harmful prior changes
  zero. Artifacts and CLI arguments may only be stricter._
  _2026-07-10 correlation correction: Superseded artifact v1 because its
  frame-level rates and three-group minimum could not support the stated 1%
  claim. Artifact v2 now requires a provenance-locked subject-disjoint benchmark
  plus a separately bound household domain-verification dataset, deterministic
  capped session/tracklet/time units, person/encounter-balanced fitting,
  truth-person-worst-case benchmark confidence, encounter-worst-case local
  metrics, active semantic/artifact pins, and a conservative runtime gallery
  envelope. Household tuning can only tighten the benchmark policy; the
  artifact authorizes the scorer only. No live authority artifact was generated
  or installed._
- [x] Enforce a separate executable/public-runtime cutover artifact.
  _2026-07-10: Model semantics now bind the loaded ReID native bridge and Python
  transform/scoring implementations. `authoritative` additionally requires an
  independently pinned `noesis.identity.authority_cutover` v1 artifact whose
  exact runtime/code/topology/camera/scorer bindings and distinct private
  coordinator-replay plus occupied-scene report bytes all verify before the
  store opens. No real cutover artifact was generated or installed._
- [x] Add owner-only Menon propose/confirm enrollment and migration review.
  _2026-07-10: Added isolated identity service/panel with fresh observation
  selection, stale/conflict UX, exact action confirmation, resident listing,
  evidence status, duplicate-name/empty-gallery review, and no apply path._
- [x] Replace neutral-only authoritative video with an exact post-resolution OSD stage.
  _2026-07-10: DS8/V3DT/DS9 use a verified tiler-sink metadata probe and bounded
  exact decision cache; stale camera/frame/tracker joins remain neutral `#XX`._

## Validation log

- _2026-07-08: 21 household identity unit tests passed; docs consistency PASSED._
- _2026-07-09: Household default-on cutover; unset env enables closed-world path; opt out with `NOESIS_HOUSEHOLD_IDENTITY=0`. First start archives legacy `~/.noesis` identity files._
- _2026-07-09: Truth-preservation repair: aliases and copresence guards fixed;
  residents survive maintenance; visitor generations and restart reconciliation
  added; health counts use live galleries; resident deletion purges biometric
  state; persistence is owner-only. Focused household/StableID selection: 83
  passed. The live DS8 process has not yet been restarted onto these changes._
- _2026-07-10: Added the identity-v2 open-set kernel. Absolute quality,
  appearance, constraint, calibrated-confidence, and ambiguity gates run before
  a bounded resident assignment prior; each tracklet has an explicit unknown
  option and joint assignment is deterministic/one-to-one unless an exact
  overlap permit exists. Scorer/resolver validation: 16 passed. Runtime hook
  cutover remains gated on a safe one-pass primitive batch adapter and replay._
- _2026-07-10: Added the safe one-pass runtime adapter in default shadow mode.
  Actual ReID engine bytes, declared layer/dimension, exact world run ID, strict
  topology, server-only enrollment cache, frame coordinator, visitor lifecycle,
  and proof-only overlap behavior now have shared DS8/V3DT/DS9 integration.
  Authoritative mode remains intentionally unavailable until a scoring
  calibration artifact is supplied and live shadow/replay evidence passes._
- _2026-07-10: Internet-loss recovery and adversarial authoritative audit:
  restored the DS9 hook from Git plus exact recorded diffs, preserved explicit
  unknown vs provisional semantics through canonical world fusion, eliminated
  authoritative legacy StableID mutations, deferred SID analytics until v2
  resolution, and made the initial OSD pass truthfully neutral. Identity/API/hook
  selection: 105 passed; DS9/parity/ReID selection: 31 passed; DS9 static prep
  passed. Strict DS9 parity and artifact/provenance gates still report only the
  seven declared capability gaps and canonical missing artifacts._
- _2026-07-10: Completed the operable identity-v2 product layer: tamper-evident
  score-only capture, disjoint held-out calibration, exact post-resolution OSD,
  biometric-free migration review, and owner-only Menon proposal/confirmation
  with gateway idempotency. Final identity selection: 125 passed; additional
  DS9/parity/private-state/benchmark selection: 17 passed; Menon full suite:
  133 passed; production build, generated-schema check, Python compilation,
  diff checks, and AGENTS/docs consistency all passed. No live runtime restart,
  migration apply, or authoritative cutover was performed._
- _2026-07-10: Adversarial identity/storage audit made evidence and replay files
  private, newline-complete, and bounded; added deterministic evidence pruning,
  fixed asymmetric product authority gates, recursively allowlisted migration
  review output, and made closed runtime APIs return 503. No live process,
  migration, enrollment, or authority mode was changed. Full identity,
  calibration, enrollment, world/API, replay, contract, and private-path gate:
  162 passed; DS9 parity/benchmark follow-up: 4 passed; generated schemas and
  AGENTS/docs consistency checks passed._
- _2026-07-10: Calibration-authority adversarial closure moved the generic
  claim to challenge-covered truth-person-worst-case evidence, bound every row
  to active model semantics, required licensed/consented real-human truth,
  excluded blocked/empty-gallery fit evidence, pinned artifact bytes and active
  semantics, and enforced the minimum calibrated gallery envelope before
  mutation. Focused new adversarial cases: 10 passed; service plus deterministic
  calibration: 15 passed; the broader selected identity matrix reached 154
  passes before one stale test assertion, whose corrected isolated rerun passed.
  Schema check, Black, Ruff, and AGENTS/docs consistency passed. No live runtime,
  gallery, mode, evidence, or artifact was changed._
- _2026-07-11: Replayed the sealed baseline subject-flip shape in a deterministic
  two-track/two-subject coordinator regression. Continuity now retains the
  independently admissible accepted subject, emits unknown for contradictory
  open-set evidence, and permits reassignment only after the evidence-gap TTL.
  Service regressions prove stale legacy embedding diagnostics are cleared and
  fresh public/canonical provenance is exact. DS9 source-level builder execution
  proves `NOESIS_REID_AUTO_MERGE_ENABLED=1` cannot override household policy.
  Focused selection: 32 passed; no Docker/GPU/runtime/evidence mutation._
