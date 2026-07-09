# Composer Subagent Audit — Household Identity (2026-07-09)

Audit of work done by composer-model subagents during the household identity
rework. BEV regression root cause and follow-up fixes recorded here.

## BEV / tracking regression + segfault (FIXED)

**Root cause (BEV empty):** Composer ReID embedding pre-pass iterated
`frame_meta.object_items` before the main tracking/BEV loop. Service Maker
`object_items` is a **one-shot iterator**. The pre-pass consumed it; the main
loop saw zero people → empty footpoints.

**Root cause (SIGSEGV, 2026-07-09):** The follow-up “fix” of
`object_items = list(...)` then re-using stored wrappers is **unsafe**. Those
wrappers are transient native views; after the iterator advances, prior
`ObjectMetadata` objects dangle. Accessing `class_id` segfaults in
`deepstream::ObjectMetadata::classId()` on the `tracking_telemetry_stage`
thread (reproduced with `./ds8_runtime.py --pgie-profile yolo26 --size m
--tracking-mode v3dt`).

**Correct fix:** Single-pass iteration only — extract ReID embeddings inline
while walking `object_items`; never `list()`/store wrappers for a second pass
(`hooks.py`, `hooks_v3dt_reimpl.py`). Keep household pose/world-before-StableID
wiring inside that same pass. Also stop permanently setting
`_stable_id_enabled = False` on single update/maintenance exceptions.

**Regression test:** `tests/test_object_items_oneshot_regression.py`
(asserts hooks do not contain `object_items = list(`).

_2026-07-09 (Grok): Confirmed segfault via gdb; restored single-pass; 45s
yolo26m+v3dt smoke reached Main Loop + REST without core dump._

## Production safety

| Mode | Status |
|------|--------|
| Default (`NOESIS_HOUSEHOLD_IDENTITY` unset) | **Household ON** (cutover 2026-07-09) |
| Opt-out (`NOESIS_HOUSEHOLD_IDENTITY=0`) | Legacy open-world StableID |
| V3DT gallery persist | Pre-existing: V3DT builder never set `gallery_persist_file` before this work (not a new regression) |

## Composer findings (severity)

### Critical / high (household mode)
1. **Shared sid/visitor pool file** — `resolve_household_paths` pointed both at `sid_pool.json` (different JSON schemas). **Fixed** → `visitor_pool.json`.
2. **Provisional early-return without `active_tracks`** — emits 9000+ SIDs without creating active records; next frame re-treats as new; metrics inflate. **Fixed** (active provisional records + stable SID across frames).
3. **enroll/delete remaps incomplete** — zones/ghosts/gallery not fully remapped. **Fixed** (`_remap_sid`).
4. **World computed after StableID on baseline hooks** — overlap permits often deny on first frames. **Fixed** → `_ensure_world_before_stable_id` runs before `_maybe_assign_stable_id` in `hooks.py` (and V3DT fallback when bbox3d world missing).
5. **Pose-only gallery match skips MNN/frame claims** in household mode. **Fixed** (pose path uses `_gallery_match_ok`).

### Medium
- `_maintain_stable_ids` used to permanently disable StableID — **fixed** (log only).
- `stable_id is None` still skips person from BEV/tracks (pre-existing pattern).
- Dead helpers: `assign_tracklets_mnn` batch path unused; `filter_kwargs_for_init` unused by runtime.
- Thin FE: display_name only; no enroll UI.

### What looks solid
- Household gated off by default
- `use_extractor=False` preserved (no torchreid on DS8)
- Overlap permit structure + topology loader
- Gallery quality gates legacy-permissive when household off
- REST shape for residents/health
- Unit tests for exclusivity/overlap/provisional/visitor (with confirm fixtures)

## Do not enable household mode in production until
- [x] Provisional lifecycle creates active records or withholds public SID
  - _2026-07-09 (Grok): Provisional mint inserts `active_tracks` + `active_zones`; same track keeps SID across frames; promote via `_household_finalize_new_sid` + `_remap_sid`. Metrics: `provisional_count`/`provisional_active_count` = live population; `provisional_event_count` = cumulative first mints._
- [x] enroll/delete fully remaps zones/ghosts/gallery
  - _2026-07-09 (Grok): Added `_remap_sid`; `enroll_resident` remaps tracks/zones/ghosts/gallery/pose and releases visitor; `delete_resident` mints visitor and remaps (no orphan resident SID)._
- [x] Baseline world available before StableID update
  - _2026-07-09 (Grok): `_ensure_world_before_stable_id` fills world via existing `_augment_track_with_world` before StableID; cache still updated after. Validated with `py_compile` + `tests/test_world_before_stable_id_ordering.py`._
- [x] Pose-only path uses household claim/MNN guards
  - _2026-07-09 (Grok): Pose-only gallery match now goes through `_gallery_match_ok(..., require_mnn_emb=False)` so exclusivity + frame claims apply._

## Cutover (completed 2026-07-09)

1. [x] Default `NOESIS_HOUSEHOLD_IDENTITY=1` in `is_household_identity_enabled()`.
2. [x] Opt-out via `NOESIS_HOUSEHOLD_IDENTITY=0` for legacy debugging.
3. [x] First start archives poisoned legacy gallery/aliases (D8).
4. [x] Recorded in `work_order.md` + `ds8_design_decisions.md`.
