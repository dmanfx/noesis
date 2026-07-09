# Phase 0 — Stop the Bleeding

**Goal:** Eliminate false cross-camera SID shares (except true FoV overlap), stop
identity-state pollution, wire pose, and expose confidence telemetry — without
yet fully splitting resident/visitor spaces.

**Depends on:** `decisions.md` D2–D4, D6–D8; `camera_topology.md`; `performance.md`

## Scope

### In
- Household mode flag / env to select new behavior
- Archive legacy gallery/aliases/pool on enable
- Disable pressure auto-merge in household mode
- Geometry-aware exclusivity + overlap permits
- Wire pose features into `StableIDManager.update()` from hooks
- Raise embedding extract budget carefully
- Emit `reid_confidence`, `reid_required`, `overlap_permit`, richer `id_event`s
- Unit tests for exclusivity/overlap/pose
- `config/camera_topology.yaml` scaffold

### Out
- Full resident/visitor enrollment UX (Phase 1/3)
- Backbone swap beyond confirming Swin default (Phase 2)
- Hungarian assignment (Phase 2)

## Technical design

### Household mode activation

Env (preferred for DS8 runtime parity):
- `NOESIS_HOUSEHOLD_IDENTITY=1` enables household behavior
- `NOESIS_HOUSEHOLD_ARCHIVE_STATE=1` (default on) archives legacy files once
- `NOESIS_CAMERA_TOPOLOGY_FILE=config/camera_topology.yaml`

When enabled, `_build_stable_id_manager` sets:
- `auto_merge_enabled=False`
- `allow_multi_zone_active=False` as the *base* flag; dual-active only via overlap permit API
- Higher gallery threshold defaults (e.g. cos_sim_high 0.78)
- Smaller handoff margin or topology-gated handoff only
- Persistence paths under `~/.noesis/household/` (or keep legacy paths but empty after archive — prefer new dir)

### Exclusivity change in manager

Replace “`allow_multi_zone_active or not active_zones`” checks with:

```text
can_dual = overlap_permit(sid, sensor_id, world_xy, ts, appearance_sim?)
allow = (not sid_active_elsewhere) or can_dual
```

Implement `overlap_permit(...)` using loaded topology + optional world point
passed into `update()` (new optional args: `world_xy`, `world_valid`).

Hooks already compute world/footpoints nearby — pass them into `_maybe_assign_stable_id` → `update()`.

### Pose wire

In `hooks.py` / `hooks_v3dt_reimpl.py` `_maybe_assign_stable_id`:
- Extract pose features/quality already available on the object/track
- Pass `pose_features=` / `pose_quality=` into `mgr.update(...)`
- Ensure `NOESIS_REID_POSE_ENABLED` defaults on when household mode is on

### Embedding budget

Change default `NOESIS_REID_EMBEDS_PER_FRAME_MAX` from 2 → 4 (or dynamic
`min(8, max(4, n_persons))` if cheap). Document in performance.md.

Prioritize extracts for: new tracks, provisional tracks, tracks lacking emb.

### State archive

On first household-mode start:
1. Create `~/.noesis/household/backups/UTC_TIMESTAMP/`
2. Move/copy `reid_gallery.npz`, `reid_aliases.json`, `sid_pool.json`, optional `stable_id_state.json`
3. Start with empty household galleries
4. Log loudly what was archived

### Telemetry

Populate public_track fields per `contracts.md` (Phase 0 subset OK if
`resident_uuid`/`display_name` null).

## Files to touch

- `reid/stable_id_manager.py` — exclusivity, overlap, metrics, household paths
- `noesis/ds8_runtime.py` — build knobs, archive helper
- `noesis/pipelines/hooks.py` — pose + world into update; confidence fields
- `noesis/pipelines/hooks_v3dt_reimpl.py` — same
- `config/camera_topology.yaml` — new
- `tests/test_stable_id_*.py` — new
- `docs/DS8_api_contracts_ws.md` — additive fields
- `plans/DS8/ds8_design_decisions.md` — pointer entry

## Action checklist

- [ ] Add household mode construction + archive-on-enable
- [ ] Implement topology load + `overlap_permit`
- [ ] Enforce exclusivity with overlap exception in all match paths
- [x] Wire pose + world into `update()` from both hook variants
  _2026-07-08 (Codex): hooks.py + hooks_v3dt_reimpl.py — see work_order.md hooks section._
- [x] Emit confidence / overlap telemetry fields
  _2026-07-08 (Codex): public_track + diagnostics via `_apply_household_id_diag_fields`._
- [x] Adjust embedding budget + priority
  _2026-07-08 (Codex): priority pre-pass in DS8 handle_frame_ds8; budget from env._
- [ ] Disable auto-merge under household mode
- [ ] Add unit tests (exclusivity, overlap, pose, archive)
- [ ] Update WS contract docs + design decision entry
- [ ] Run pytest focused suite; note results under work_order

## Risks

- Wrong `source_id` mapping in topology → false deny/grant. Mitigate: log
  camera name↔id at startup; fail closed.
- Missing world points → no overlap permits → brief dual-view ID splits in
  overlap zone. Mitigate: Phase 0 degraded appearance-only path only if
  explicitly enabled; prefer fix world quality.
- Raising emb budget increases detection-wake cost. Measure.
