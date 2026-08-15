# DS8 V3DT Enhancements Plan (post-bring-up)

This document captures **follow-on** work once SV3DT and MV3DT are stable (per `plans/DS8/v3dt/integration_plan.md`). The goal is to fully exploit **metric 3D state**, **overlap fusion**, and **pose-assisted tracking** in your house deployment.

## Guiding principles (Noesis-specific)

- **Keep external identity stable:** `stable_id` remains the only user-visible ID. MV3DT IDs are internal hints (`mv3dt_id`) used to strengthen StableID and analytics.
- **MV3DT only for true overlap:** keep MV3DT neighbor graph restricted to cameras with overlapping FoVs (kitchen ↔ family-room).
- **Adjacent cameras:** living-room ↔ kitchen transitions are handled via StableID + geometry gating (not MV3DT).

## Enhancement backlog (recommended order)

### 1) MV3DT-assisted StableID (overlap robustness)

**Problem**
- Appearance embeddings degrade under occlusion and viewpoint changes, causing occasional StableID switches in the overlap.

**Approach**
- When MV3DT is enabled, treat `mv3dt_id` as a strong constraint:
  - maintain a short-lived mapping `mv3dt_id → stable_id` (people-only)
  - immediately reuse `stable_id` when a known `mv3dt_id` reappears (even if embeddings are weak)
  - never merge across non-neighbor cameras using `mv3dt_id`

**Acceptance**
- >20% reduction in overlap ID switches on kitchen↔family-room test clips.
- No increase in false merges (different people collapsed).

### 2) Multi-view occupancy dedup (HomeSeer/MQTT)

**Problem**
- Per-camera occupancy double-counts the same person when they are visible in both kitchen and family-room simultaneously.

**Approach**
- Compute “house occupancy” by deduplicating across cameras:
  - primary: `stable_id`
  - secondary: `mv3dt_id` (if present) as a cross-check for overlap duplicates
- Publish:
  - per-camera occupancy (unchanged)
  - deduped “whole house” occupancy
  - per-zone occupancy (kitchen vs family-room vs living-room) using world footpoints

**Acceptance**
- Total occupancy never exceeds the true number of distinct people in overlap scenarios.
- Zone occupancy is consistent and stable (no rapid flicker) when people hover near boundaries.

### 3) Metric velocity + heading analytics (world-space)

**Problem**
- Pixel velocity is not comparable across depth/cameras and is less useful for automation.

**Approach**
- Use `NvDsObj3DBbox.{xVel,yVel,zVel}` to compute:
  - horizontal speed `sqrt(xVel^2 + zVel^2)` (m/s)
  - heading `atan2(zVel, xVel)` (house/world frame)
- Feed into:
  - “moving vs stationary” classification
  - dwell-time analytics (standing in kitchen island zone, etc.)

**Acceptance**
- Speed/heading are stable (no spikes) and match visual intuition on test clips.

### 4) Fusion confidence scoring (MV3DT health signal)

**Problem**
- MV3DT can fuse incorrectly under bad calibration, poor sync, or marginal overlap; we need a health metric.

**Approach**
- Compute a per-track fusion confidence score from signals we can access in DS8 Python:
  - cross-camera agreement in world footpoint (kitchen vs family-room) when both see the same `mv3dt_id`
  - temporal smoothness of 3D velocity
  - optional: visibility gating if we later expose it in Python
- Use the score to:
  - suppress aggressive StableID merges when confidence is low
  - raise “calibration drift suspected” warnings when confidence degrades persistently

**Acceptance**
- When calibration is intentionally perturbed, confidence drops quickly and reliably.
- When calibration is good, confidence stays high in stable overlap scenes.

### 5) Adjacent handoff events (living-room ↔ kitchen)

**Problem**
- Cameras are adjacent but not overlapping, making identity handoff and “room transitions” noisy.

**Approach**
- Emit explicit “transition events” when StableID moves between zones/cameras:
  - require time-window continuity + boundary proximity in world coords (3D gating)
  - optionally incorporate direction-of-travel using world velocity

**Acceptance**
- Transition events match reality in representative clips (few false transitions).

### 6) Calibration drift detection (continuous monitoring)

**Problem**
- Small camera bumps or timestamp drift can silently break MV3DT fusion quality over time.

**Approach**
- Monitor long-running metrics:
  - overlap agreement residuals (same `mv3dt_id` observed in two cameras should land near the same world footpoint)
  - sudden shift in per-camera “floor contact” distribution (foot y deviates from floor)
- Raise:
  - warnings in logs/telemetry
  - optional: MQTT status topic updates (so HomeSeer can alert)

**Acceptance**
- Drift detector triggers on injected calibration offsets and remains quiet in stable operation.

### 7) 3D trajectory heatmaps (BEV analytics)

**Problem**
- Hard to reason about typical paths and occupancy patterns without aggregated views.

**Approach**
- Accumulate world footpoints into heatmaps:
  - per-zone
  - per-time-of-day window
- Render as BEV overlays and/or export periodic snapshots.

**Acceptance**
- Heatmaps are stable over time and clearly reflect common movement patterns (kitchen↔family-room paths, etc.).

## Notes on “dogs”

SV3DT/MV3DT are primarily documented around a human model (height/radius) and optional pose anchors. “Dog support” is only safe once:
- you confirm the tracker supports per-class `modelInfo` (or a second object model type), and
- you have an appropriate detector+tracker class mapping and (optionally) a dog-specific ReID strategy.
