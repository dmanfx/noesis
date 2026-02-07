# DS8 Pose-Assisted StableID Integration
_Status: current as of 2026-02-02._

This note records how pose features are used to strengthen StableID assignment in the DS8 pipeline.

## Overview

- **Source:** YOLO26 pose SGIE → `PoseFeatureProcessor` attaches `NOESIS.POSE_FEATURES` object meta.
- **Transport:** Native bridge `noesis_pose_meta_ext.extract_pose_features(obj_meta)` reads JSON from `NvDsObjectMeta`.
- **Consumer:** `_AnalyticsTelemetryProcessor` passes pose features into `StableIDManager.update(...)`.
- **Goal:** Use pose ratios as a **secondary identity signal** to reduce ID switches, while keeping
  appearance (ReID embeddings) as the primary matcher.

## Data Flow

1. Pose SGIE produces keypoints and ratio features per person.
2. `PoseFeatureProcessor` attaches JSON to each object:
   `type="pose_features"`, `features`, `kpt_mean_conf`, `kpt_valid_frac`.
3. `_AnalyticsTelemetryProcessor` extracts pose JSON (native bridge) and supplies:
   - `pose_features`: dict of float ratios
   - `pose_quality`: dict with `kpt_mean_conf`, `kpt_valid_frac`
4. `StableIDManager.update(...)` fuses pose similarity with ReID similarity
   or uses pose-only matching when embeddings are missing.

## StableID Behavior

- **Gated blend:** pose contributes only when pose quality is good and pose similarity passes
  thresholds. ReID thresholds still gate final acceptance.
- **Pose-only fallback:** if ReID embeddings are missing, pose similarity alone can match
  (higher threshold).
- **Ghost re-association:** pose vectors are stored in ghost records to recover IDs after
  short occlusions when embeddings are missing.

## Memory & Storage Control

- **Bounded RAM:** per-stable-id deque with a size cap + global cap.
- **TTL pruning:** pose entries older than `pose_max_age_s` are removed during `prune_ghosts`.
- **No persistence:** pose features are not written to disk.

## Configuration (env vars + defaults)

Enable/disable:
- `NOESIS_REID_POSE_ENABLED` (auto-on if pose SGIE enabled and pose features not disabled)
- `NOESIS_POSE_FEATURES_ENABLED=0` disables pose features globally

Fusion + thresholds:
- `NOESIS_REID_POSE_WEIGHT=0.15`
- `NOESIS_REID_POSE_SIM_THRESHOLD=0.55`
- `NOESIS_REID_POSE_SIM_HIGH_THRESHOLD=0.65`
- `NOESIS_REID_POSE_ONLY_THRESHOLD=0.80`
- `NOESIS_REID_POSE_MIN_VALID_FRAC=0.45`
- `NOESIS_REID_POSE_MIN_MEAN_CONF=0.50`
- `NOESIS_REID_POSE_MIN_FEATURES=6`
- `NOESIS_REID_POSE_INTERVAL_S=0.75`

Memory caps:
- `NOESIS_REID_POSE_GALLERY_SIZE=8`
- `NOESIS_REID_POSE_MAX_AGE_S=30`
- `NOESIS_REID_POSE_MAX_TOTAL_ENTRIES=0` (auto-derived from `max_total_ids`)

## Validation

- Unit tests:
  - `python3 -m pytest tests/test_stable_id_manager_pose.py`
- Runtime:
  - Ensure `noesis_pose_meta_ext` is built.
  - Run DS8 pipeline and confirm stable_id persistence when pose is present.

## Notes for Agents

- Service Maker Python does not expose `obj_user_meta_list`; pose extraction **must**
  go through the native bridge.
- This integration is DS8-only; do not add legacy pad-probe or CPU fallback branches.
