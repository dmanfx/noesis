# Household Identity — Camera Topology

Defines when the same physical person may legally hold one SID on two cameras.

## Known layout

- Cameras include **kitchen** and **family-room** with a real FoV overlap region.
- A third camera (often living-room / other) may have little or no overlap.
- Exact `source_id` indices depend on the active infer/sources YAML — verify at
  implementation time; do not assume 0/1/2 without reading the running config.

## Overlap permit algorithm

Given candidate assignment of identity `I` already active on camera `A`, and a
new/updated track on camera `B` wanting `I`:

1. If `(A,B)` not in enabled overlap pairs → **DENY** (exclusivity).
2. If either track lacks a valid world/footpoint → **DENY** unless
   `overlap_allow_appearance_only=false` (default false in household mode).
   Phase 0 may temporarily allow appearance-only with high sim (≥ 0.85) and
   log `overlap_permit_degraded=appearance_only` for metrics — record if used.
3. If `|tA - tB| > max_time_delta_s` → **DENY**.
4. If `||worldA - worldB|| > max_world_dist_m` → **DENY**.
5. Optional: if appearance sim < `require_appearance_sim` → **DENY**.
6. Else **GRANT** `overlap_permit=true` and allow dual-active SID.

Handoff without overlap (person left A, appears on B):
- Not a dual-active case. Use ghost (same cam) or gallery match after A releases
  activity (track lost / grace). Do not require overlap permit.

## Config file

Create `config/camera_topology.yaml` (see `contracts.md`). Load from
`ds8_runtime` / StableID manager construction. Missing file → exclusivity
without permits (safe default: no dual-active).

## Calibration dependency

World distances require existing Menon/DS8 world projection to be sane. If world
quality is `invalid`, do not grant overlap permits. Prefer failing closed
(unique IDs per camera) over false shares.

## Validation scenes

1. Person standing in kitchen–family overlap → same SID both cams.
2. Two different people, one in kitchen / one in family (no shared body) →
   different SIDs even if similarly dressed.
3. Walk kitchen → family through overlap → continuous SID, brief dual-active OK.
4. Walk kitchen → living-room with no overlap → SID continues after release,
   never dual-active during transit gap.
