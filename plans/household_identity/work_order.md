# Household Identity — Work Order

Master execution checklist. Update checkboxes with dated notes as work lands.

## Program status

- Planning docs: **complete** (2026-07-08)
- Implementation: **household default-on** (2026-07-09)
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

## Validation log

- _2026-07-08: 21 household identity unit tests passed; docs consistency PASSED._
- _2026-07-08: Enable with `NOESIS_HOUSEHOLD_IDENTITY=1` (archives legacy `~/.noesis` identity files on first start)._
