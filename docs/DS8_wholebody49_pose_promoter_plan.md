# DS8 Wholebody49 X Pose Promoter Plan
_Status: implementation plan drafted 2026-06-24._

This plan describes how to promote DEIMv2 Wholebody49 X label-only keypoint detections into the existing DS8 `NOESIS.POSE_FEATURES` object metadata contract when Wholebody49 is selected as the PGIE. The goal is to remove the redundant YOLO26 pose SGIE later, while preserving keypoint overlay, pose-derived world anchors, and pose feature payloads for identity work. This plan intentionally stays boxes-only and does not depend on Wholebody masks.

## Scope

In scope:

- Wholebody49 `x` / DINOv3-X boxes profile only.
- Runtime path under the canonical DS8 stack: `noesis/`, `pipelines/`, `config/`, `tests/`.
- Raw `label_xyxy_score` tensor rows with shape `[N, 1240, 6]` or `[1240, 6]`, where each row is `[class_id, x1, y1, x2, y2, score]`.
- Body objects stay as the only promoted detector objects for tracker, analytics, ReID, and public telemetry.
- Wholebody keypoint rows are grouped to those body objects and attached as `NOESIS.POSE_FEATURES` JSON user meta.
- The resulting payload must be consumable by the existing pose keypoint overlay and baseline fused world estimator.

Out of scope:

- Wholebody mask promotion, mask rendering, or mask-based depth sampling.
- Replacing OSNet ReID embeddings.
- Replacing DAv2 object-depth tracking or MapAnything.
- Adding deprecated-stack code paths, appsinks, GI pad probes, or CPU frame branches to DS8.
- Treating failed tensor/grouping evidence as a silent fallback to YOLO26 pose. If Wholebody pose promotion is required and unavailable, fail loudly or keep YOLO26 pose explicitly enabled by policy.

## Current State

The current DS8 graph supports optional PGIE profiles, including `--pgie-profile wholebody49 --size x`.

Key current files:

- `noesis/ds8_runtime.py`
  - Materializes Wholebody49 profile overlays through `noesis.deimv2_wholebody49_assets`.
  - Attaches pose, depth, MapAnything, and tracking hooks.
- `noesis/deimv2_wholebody49_assets.py`
  - Defines `WHOLEBODY49_SIZES = ("s", "x")`.
  - Resolves `x` to `models/engines/deimv2_wholebody49_dinov3_x_boxes_640_b3_fp16.engine`.
  - Materializes `build/config_infer_primary_wholebody49_x.ini`.
- `pipelines/config_infer_primary_deimv2_wholebody49_boxes.template.ini`
  - Uses `network-type=0`.
  - Uses `parse-bbox-func-name=NvDsInferParseDeimv2Wholebody49Boxes`.
  - Has `output-blob-names=label_xyxy_score`.
  - Currently has `output-tensor-meta=0`, which prevents the runtime from seeing raw Wholebody keypoint rows.
- `pipelines/nvdsinfer_deimv2_wholebody49/nvdsinfer_deimv2_wholebody49.cpp`
  - The boxes parser filters to `class_id == 0`.
  - This is good for tracker cleanliness: only body detections become `NvDsObjectMeta`.
  - Non-body rows, including keypoints and bones, are not emitted as detector objects.
- `noesis/pipelines/ds8_pipeline.py`
  - Builds PGIE, tracker, optional ReID SGIE, optional YOLO26 pose SGIE, `world_observation_stage`, `tracking_telemetry_stage`, tiler, and OSD.
  - If `models.pose.enable=true`, YOLO26 pose still runs after the tracker/ReID chain.
- `noesis/pipelines/hooks.py`
  - `PoseFeatureProcessor` currently reads YOLO26 pose tensor metadata and attaches `NOESIS.POSE_FEATURES`.
  - `_AnalyticsTelemetryProcessor._extract_pose_keypoints_for_anchor(...)` consumes either native YOLO26 tensor extraction or attached pose feature payloads.
  - `PoseKeypointOverlayProcessor` reads attached pose feature payloads for skeleton drawing.
  - `_ObjectDepthFusionProcessor` attaches DAv2 object-depth data at `world_observation_stage`.
- `native/noesis_pose_meta_ext.cpp`
  - Attaches and extracts `NOESIS.POSE_FEATURES` JSON user meta from `NvDsObjectMeta`.
  - Must remain the object-user-meta bridge because Service Maker Python wrappers do not expose arbitrary `obj_user_meta_list`.
- `docs/DS8_metadata_contracts.md`
  - Documents `NOESIS.POSE_FEATURES`.
- `docs/DS8_pose_stable_id_integration.md`
  - Documents pose-assisted StableID intent and thresholds.

Prototype reference only:

- `testpipelines/deimv2-wholebody49/wholebody_overlay.py`
  - Decodes all Wholebody rows from `label_xyxy_score`.
  - Applies class-aware thresholds.
  - Keeps keypoints near body context.
  - Applies keypoint NMS.
  - Assigns keypoints and bones back to body instances.
  - Draws skeletons from the grouped detections.

Do not import prototype modules into production runtime. Use the prototype as an algorithm reference and move production-grade, tested helpers into `noesis/`.

## Target Behavior

When `--pgie-profile wholebody49 --size x` and `models.wholebody_pose.enable=true`:

1. The Wholebody X PGIE emits body objects as it does now.
2. The same PGIE also exposes raw `label_xyxy_score` tensor metadata.
3. A DS8 metadata operator reads the raw tensor rows for each frame.
4. The operator groups keypoint rows back onto body objects produced by the PGIE/parser/tracker.
5. For each body object with enough grouped keypoints, the operator attaches a normal `PoseFeatureResult` payload using `noesis_pose_meta_ext.attach_pose_features(...)`.
6. Downstream code continues using the existing pose contract:
   - `PoseKeypointOverlayProcessor` draws skeletons from attached payloads.
   - `_AnalyticsTelemetryProcessor._extract_pose_keypoints_for_anchor(...)` resolves pose anchors from attached payloads.
   - The fused world estimator can produce `world_source="pose_depth_fused"` or `world_source="pose_floor_only"` with Wholebody-derived keypoints.
7. YOLO26 pose SGIE can later be disabled by explicit policy once validation shows parity.

The first implementation should not remove the existing YOLO26 pose path. Add Wholebody promotion beside it behind a config gate, validate it, then introduce policy that disables YOLO26 pose only when Wholebody promotion is explicitly active and healthy.

## Proposed Config Shape

Add a new optional model subsection:

```yaml
models:
  wholebody_pose:
    enable: false
    source_gie_id: 1
    model: deimv2-wholebody49-x-boxes
    object_score_threshold: 0.50
    attribute_score_threshold: 0.75
    keypoint_threshold: 0.50
    keypoint_candidate_threshold: 0.40
    class_aware_filtering: true
    max_detections: 300
    min_keypoints_for_payload: 4
    attach_replace_existing: true
```

Suggested later policy, after validation:

```yaml
models:
  pose:
    enable: false
  wholebody_pose:
    enable: true
```

Policy rules:

- `wholebody_pose.enable=true` should require Wholebody49 PGIE X or another validated Wholebody label-only profile.
- If `wholebody_pose.enable=true` but the active PGIE is not Wholebody49, fail startup with a clear error.
- If `wholebody_pose.enable=true` but `output-tensor-meta` is not available at the chosen attach point, fail loudly with a diagnostic that names the component and available tensor IDs.
- Keep YOLO26 pose enabled until an explicit config or runtime overlay disables it.

## Required Production Changes

### 1. Expose raw Wholebody X tensor metadata

Update `pipelines/config_infer_primary_deimv2_wholebody49_boxes.template.ini`:

```ini
output-tensor-meta=1
```

Reason:

- The production parser should keep emitting only class `0` bodies.
- The promoter needs the raw `label_xyxy_score` rows from frame tensor metadata.
- For X boxes, this tensor is small compared with mask outputs. The plan intentionally avoids mask tensors.

Also consider adding a comment near this setting:

```ini
# Required by DS8 Wholebody pose promoter; X boxes emits only label_xyxy_score.
```

### 2. Add a pure Wholebody pose decoding module

Recommended new module:

- `noesis/metadata/wholebody49_pose.py`

Responsibilities:

- Define constants for the 49 Wholebody classes.
- Decode `label_xyxy_score` into production detections.
- Apply class-aware thresholds.
- Apply keypoint NMS.
- Assign keypoints to body detections.
- Convert selected Wholebody keypoints to COCO-style 17-keypoint arrays.
- Compute ROI-space and absolute-space keypoint arrays for `PoseFeatureResult`.

Keep this module pure and independent of DeepStream objects so it can be heavily unit-tested without DS8 runtime.

Suggested data structures:

```python
WholebodyDetection(
    class_id: int,
    class_name: str,
    score: float,
    bbox_xyxy: tuple[float, float, float, float],
    source_idx: int,
    group: str,
    attributes: dict[str, str],
)

WholebodyPoseAssignment(
    body_source_idx: int,
    keypoints_abs: np.ndarray,  # shape (17, 3)
    keypoints_roi: np.ndarray,  # shape (17, 3)
    score: float,
    selected_class_ids: dict[int, int],
    debug: dict[str, Any],
)
```

Suggested public functions:

```python
decode_wholebody49_detections(...)
refine_wholebody49_detections(...)
assign_wholebody49_keypoints_to_bodies(...)
wholebody49_pose_for_body(...)
wholebody49_to_pose17(...)
```

The production module can reuse the algorithmic structure from `testpipelines/deimv2-wholebody49/wholebody_overlay.py`, but should not import it.

### 3. Wholebody class groups and class IDs

Use zero-based class IDs from `models/deimv2_wholebody49/classes.txt`.

Primary classes:

```text
0  body
1  adult
2  child
3  male
4  female
5  body_with_wheelchair
6  body_with_crutches
7  head
16 face
17 eye
18 nose
19 mouth
20 ear
21 collarbone
22 shoulder
23 shoulder_left
24 shoulder_right
25 solar_plexus
26 elbow
27 elbow_left
28 elbow_right
29 wrist
30 wrist_left
31 wrist_right
32 hand
33 hand_left
34 hand_right
35 abdomen
36 hip_joint
37 hip_joint_left
38 hip_joint_right
39 knee
40 knee_left
41 knee_right
42 ankle
43 ankle_left
44 ankle_right
45 foot
46 foot_left
47 foot_right
48 bone
```

Class groups:

- Body: `0`
- Attributes: `1,2,3,4,8,9,10,11,12,13,14,15`
- Keypoints: `21,22,23,24,25,26,27,28,29,30,31,35,36,37,38,39,40,41,42,43,44`
- Objects/body parts: `0,5,6,7,16,17,18,19,20,32,33,34,45,46,47`
- Bone: `48`

Left-side class IDs:

```python
{23, 27, 30, 33, 37, 40, 43, 46}
```

Right-side class IDs:

```python
{24, 28, 31, 34, 38, 41, 44, 47}
```

### 4. Convert Wholebody keypoints to the existing 17-point pose shape

The DS8 pose contract uses COCO-style ordering:

```text
0 nose
1 left_eye
2 right_eye
3 left_ear
4 right_ear
5 left_shoulder
6 right_shoulder
7 left_elbow
8 right_elbow
9 left_wrist
10 right_wrist
11 left_hip
12 right_hip
13 left_knee
14 right_knee
15 left_ankle
16 right_ankle
```

Recommended direct mapping:

```python
WHOLEBODY_TO_POSE17 = {
    18: 0,   # nose
    23: 5,   # shoulder_left
    24: 6,   # shoulder_right
    27: 7,   # elbow_left
    28: 8,   # elbow_right
    30: 9,   # wrist_left
    31: 10,  # wrist_right
    37: 11,  # hip_joint_left
    38: 12,  # hip_joint_right
    40: 13,  # knee_left
    41: 14,  # knee_right
    43: 15,  # ankle_left
    44: 16,  # ankle_right
}
```

Optional anchor helpers:

- Use `46` / `47` foot detections only as lower-confidence substitutes for missing `43` / `44` ankles.
- Do not use generic side-less `22`, `26`, `29`, `36`, `39`, or `42` as left/right keypoints unless a later validation pass proves a safe side assignment rule.
- Leave unknown face side points at confidence `0.0` rather than inventing left/right eye or ear assignments.

Coordinate conversion:

- Wholebody rows are bbox boxes. Use the bbox center as the keypoint coordinate:
  - `u = (x1 + x2) * 0.5`
  - `v = (y1 + y2) * 0.5`
  - `conf = detection.score`
- `keypoints_abs` is in the same frame space as the current `ObjectMetadata.rect_params`.
- `keypoints_roi` subtracts the body bbox origin:
  - `x_roi = u - body_left`
  - `y_roi = v - body_top`
- Clamp ROI points to `[0, body_width]` and `[0, body_height]`.
- Missing keypoints must remain `[0.0, 0.0, 0.0]`.

### 5. Grouping algorithm

Start from the prototype's successful X grouping behavior.

Recommended grouping order:

1. Decode all rows above coarse group thresholds.
2. Sort by score descending.
3. Apply class-aware filtering:
   - Keep body rows at `object_score_threshold`.
   - Keep keypoints at `keypoint_candidate_threshold` if they are near/inside a body.
   - Drop orphan keypoints unless they exceed a high orphan threshold.
   - Keep attributes only if they can be related to a body or body part.
   - Keep bone rows only for optional skeleton-line diagnostics, not for the pose17 payload initially.
4. NMS keypoint boxes per class ID.
5. Assign candidate keypoints to containing body boxes:
   - First criterion: keypoint center inside body bbox with a small padding, e.g. `2 px`.
   - If multiple bodies contain it, choose the smallest-area body, then higher body score, then lower source index.
   - Do not assign a keypoint to two bodies.
6. For each body, select the best keypoint for each class/side:
   - Prefer smaller keypoint area ratio to body area.
   - Prefer higher keypoint score.
   - Prefer closer normalized center distance to the body center only as a tie-breaker.
7. Convert selected keypoints to pose17.
8. Emit a payload only if enough useful pose points exist.

Recommended initial `min_keypoints_for_payload`:

- `4` for world anchoring: shoulders/hips/ankles may be sparse in real scenes.
- StableID pose-vector contribution should use its existing quality gates, not this minimum alone.

Recommended quality fields:

- `kpt_mean_conf`: mean of all 17 confidence values, including zeros.
- `kpt_min_conf`: minimum of all 17 confidence values.
- `kpt_valid_frac`: fraction of 17 keypoints with `conf >= keypoint_threshold`.
- `score`: mean or max of selected Wholebody keypoint scores. Use one definition consistently and document it in code.

### 6. Add a DS8 metadata processor and hook

Recommended additions in `noesis/pipelines/hooks.py`:

```python
def attach_wholebody_pose_promoter_hook(
    pipeline: "DS8Pipeline",
    *,
    camera_labels: Optional[Mapping[int, str]] = None,
) -> None:
    ...
```

Recommended classes:

```python
class WholebodyPosePromoterProcessor:
    def handle_frame_ds8(self, frame_meta: Any) -> None:
        ...

class _WholebodyPosePromoterOperator(BatchMetadataOperator):
    def handle_metadata(self, batch_meta: Any) -> None:
        ...
```

Processor responsibilities:

- Read `models.wholebody_pose` from `pipeline.config`.
- Require `enable=true`.
- Require `noesis_pose_meta_ext.attach_pose_features` to exist.
- Read `frame_meta.tensor_items`.
- Convert tensor items through `as_tensor_output()` when available.
- Find `unique_id == source_gie_id` and layer name `label_xyxy_score`.
- Convert the layer to a CPU numpy array. For X boxes this is expected to be small.
- Select the correct batch slice for the frame if the tensor is `[B, 1240, 6]`.
- Get frame dimensions from the same attributes used elsewhere: `pipeline_width` / `source_width` / `frame_width` / `width`.
- Decode and group Wholebody detections.
- Match grouped body detections to `frame_meta.object_items` class `0` objects by IoU or center containment.
- Attach `PoseFeatureResult(..., model="deimv2-wholebody49-x-boxes")`.

Object matching:

- The raw tensor body boxes are in PGIE/network/frame coordinates; current body objects are the parser/tracker objects.
- Match only class `0` object metas.
- Convert `rect_params` to `[left, top, width, height]` with `_rect_to_bbox`.
- Convert body detection `bbox_xyxy` to `[left, top, width, height]`.
- Prefer highest IoU.
- Require a minimum IoU, e.g. `0.30`, or center containment plus reasonable scale ratio.
- If no match is found, do not attach pose meta for that body and increment a diagnostic counter.
- Do not create new object metadata for keypoints.

Payload attachment:

- Use `PoseFeatureResult` from `noesis/metadata/pose_features.py`.
- Use `noesis_pose_meta_ext.attach_pose_features(obj_meta, payload_json, replace_existing=True)`.
- Use `_serialize_compact_json_with_metrics(..., metric="pose_features.user_meta_json")`.
- Enforce `_pose_meta_payload_limit_bytes()`.
- Increment existing-style counters:
  - `tensor_host_copies_total.wholebody_pose`
  - `tensor_boundary_copy_bytes_total.pose_meta`
  - `wholebody_pose_frames_total`
  - `wholebody_pose_objects_total`
  - `wholebody_pose_attached_total`
  - `wholebody_pose_missing_tensor_total`
  - `wholebody_pose_no_match_total`

### 7. Attach-point and graph sequencing

Preferred first implementation:

- Attach the Wholebody pose promoter to `world_observation_stage`.
- Call `attach_wholebody_pose_promoter_hook(...)` before `attach_object_depth_fusion_hook(...)` and before `attach_analytics_telemetry_hook(...)` in `noesis/ds8_runtime.py`.

Reason:

- `world_observation_stage` is after tracker, analytics, and optional ReID/pose SGIE chain.
- Object IDs should exist by this point.
- The fused world estimator and tracking telemetry run after this stage.
- Pose overlay runs later near tiler/OSD and can consume attached payloads.

Validation requirement:

- Verify that PGIE tensor metadata with `unique_id=1` is still visible on `frame_meta.tensor_items` at `world_observation_stage`.
- If it is not visible, do not silently switch behavior. Stop and document the observed attach point, available tensor IDs, and the next approved graph change.

If a deterministic separate stage is needed:

- Add a queue component such as `wholebody_pose_stage` immediately before `world_observation_stage` in `noesis/pipelines/ds8_pipeline.py`.
- Attach the promoter to that component.
- Keep this DS8-only and Service Maker based.

### 8. Runtime profile integration

In `noesis/ds8_runtime.py`:

- When `pgie_profile == "wholebody49"` and `size == "x"`, materialize the existing X PGIE/preprocess configs.
- If `models.wholebody_pose.enable=true`, validate:
  - active PGIE profile is `wholebody49`,
  - active Wholebody size is `x`,
  - `models.pgie.gie_id` or materialized PGIE unique ID is `1`,
  - `models.wholebody_pose.source_gie_id == 1` unless explicitly configured otherwise,
  - native pose meta bridge is importable before activation.
- Log the active pose source:
  - `pose_source=yolo26_pose`
  - `pose_source=wholebody49_x`
  - `pose_source=none`
- Do not disable `models.pose` automatically in the first implementation. Make YOLO26 disabling an explicit follow-up config or CLI policy after validation.

### 9. Existing consumer compatibility

Keypoint overlay:

- `PoseKeypointOverlayProcessor` reads attached pose payloads through `noesis_pose_meta_ext.extract_pose_features`.
- It should work without knowing whether the payload was YOLO26 or Wholebody as long as `keypoints_roi` or `keypoints_abs` exists.
- Keep `visualization.display_keypoints=true` for overlay validation.

World anchoring:

- `_AnalyticsTelemetryProcessor._extract_pose_keypoints_for_anchor(...)` first tries native YOLO26 tensor extraction and then attached payloads.
- If YOLO26 pose SGIE is disabled and Wholebody payloads are attached, it should read `keypoints_abs` from the payload.
- Expected good outcome: tracks regain pose-derived anchor paths such as `pose_depth_fused` or `pose_floor_only`.

StableID pose feature fusion:

- `StableIDManager` already supports `pose_features` and `pose_quality` arguments.
- Audit the current `_AnalyticsTelemetryProcessor._maybe_assign_stable_id(...)` call path before claiming pose-assisted StableID parity. Current code may need a small plumbing change to extract `features`, `kpt_mean_conf`, and `kpt_valid_frac` from `NOESIS.POSE_FEATURES` and pass them to `StableIDManager.update(...)`.
- This audit is part of the Wholebody promoter work because the payload must be useful for both world anchors and identity features.

### 10. Tests

Add focused tests before or alongside implementation.

Recommended unit tests:

- `tests/test_wholebody49_pose_promoter.py`
  - Decode accepts `[1240, 6]` and `[B, 1240, 6]`.
  - Normalized boxes convert to frame pixels.
  - Absolute pixel boxes remain unchanged.
  - Class grouping matches `classes.txt`.
  - Keypoint NMS keeps the best detection per keypoint class.
  - Context filtering keeps body-near keypoints and drops orphan keypoints.
  - Two nearby body instances keep separate keypoint assignments.
  - Left/right keypoint class IDs map to the correct pose17 indices.
  - Missing keypoints remain zero-confidence rows.
  - Foot detections can fill missing ankle points only when configured.
  - `PoseFeatureResult` output includes `model="deimv2-wholebody49-x-boxes"`, `keypoints_abs`, `keypoints_roi`, `features`, and quality fields.

Recommended hook tests:

- Extend or mirror `tests/test_pose_tensor_contracts.py`.
- Use fake `frame_meta.tensor_items` and fake `object_items`.
- Monkeypatch `hooks.noesis_pose_meta_ext` with a fake object exposing `attach_pose_features`.
- Assert attached payload JSON can be read back and has shape-compatible keypoints.
- Assert missing tensor increments/logs without attaching payloads.
- Assert non-Wholebody profile with `wholebody_pose.enable=true` raises a clear startup/config error.

Recommended config tests:

- Extend `tests/test_deimv2_wholebody49_assets.py`.
- Assert X template or materialized X config contains:
  - `output-blob-names=label_xyxy_score`
  - `output-tensor-meta=1`
  - no mask output names
- Assert S/mask config does not accidentally become the X boxes path.

Recommended parser tests:

- Keep the current boxes parser emitting only body objects.
- Add a regression test or parser contract check stating keypoint rows are consumed by the promoter, not emitted as tracked objects.

### 11. Runtime validation

Minimum non-live validation:

```bash
python3 -m pytest tests/test_deimv2_wholebody49_assets.py tests/test_pose_tensor_contracts.py tests/test_wholebody49_pose_promoter.py -q
python3 -m compileall -q noesis tests/test_wholebody49_pose_promoter.py
./scripts/check_agents_docs_consistency.py
```

If native pose meta bridge changes:

```bash
./scripts/build_noesis_pose_meta_ext.sh
python3 -m pytest tests/test_pose_tensor_contracts.py -q
```

Runtime smoke, when a live or recorded scene with visible people is available:

```bash
python3 noesis/ds8_runtime.py --pipeline-config config/infer.yaml --pgie-profile wholebody49 --size x
```

Runtime evidence to collect:

- Startup logs show `pose_source=wholebody49_x` when Wholebody promoter is active.
- Logs show Wholebody pose tensor IDs and attachment counts.
- `pose_present=true` appears in tracking diagnostics for visible people.
- Keypoint overlay draws skeletons without the YOLO26 pose SGIE enabled.
- Tracking telemetry includes pose-derived `world_source` values:
  - `pose_depth_fused`
  - `pose_floor_only`
  - or a clear quality reason when pose anchor is unavailable.
- StableID diagnostics show pose feature presence only after the StableID handoff audit is complete.
- GPU utilization and FPS are compared against the same scene with YOLO26 pose enabled.

Quality acceptance:

- One visible standing/walking person produces shoulders, hips, knees, and at least one ankle/foot anchor in most occupied frames.
- Multiple people in the same camera view do not cross-assign keypoints between bodies.
- Far-camera partial bodies degrade by leaving low-confidence missing keypoints, not by inventing geometry.
- Turning off YOLO26 pose does not remove keypoint overlay when Wholebody promoter is active.
- World anchor quality does not regress relative to YOLO26 pose on the same occupied-scene window.

### 12. Diagnostics and failure modes

Add concise diagnostics because most failures will otherwise look like "pose missing."

Recommended log/counter fields:

- `wholebody_pose.enabled`
- `wholebody_pose.source_gie_id`
- `wholebody_pose.tensor_layers`
- `wholebody_pose.available_tensor_ids`
- `wholebody_pose.decoded_rows`
- `wholebody_pose.body_count`
- `wholebody_pose.keypoint_count`
- `wholebody_pose.assigned_body_count`
- `wholebody_pose.attached_count`
- `wholebody_pose.no_tensor_count`
- `wholebody_pose.no_body_match_count`
- `wholebody_pose.payload_too_large_count`

Fail-fast conditions:

- `wholebody_pose.enable=true` while active PGIE is not Wholebody49 X.
- Native pose meta bridge is missing.
- The configured source GIE ID is absent for a sustained startup window.
- `label_xyxy_score` layer is missing.
- Tensor shape is not compatible with `[*, 1240, 6]` or `[1240, 6]`.

Non-fatal per-frame conditions:

- No people in frame.
- People present but no qualifying keypoints.
- A single frame has no tensor metadata during startup or interval behavior, as long as the configured mode allows skipped frames and diagnostics show it.

Do not add a hidden fallback to YOLO26 pose. If a runtime operator wants YOLO26 as a backup, that must be an explicit policy such as `pose_source_policy: yolo26` or `pose_source_policy: wholebody_required`.

### 13. Rollout Plan

Phase 1: pure helper and tests

- Add `noesis/metadata/wholebody49_pose.py`.
- Port the prototype X grouping behavior into pure functions.
- Add synthetic tests with one body, two bodies, orphan keypoints, and left/right joints.
- Do not touch runtime graph yet.

Phase 2: tensor-meta exposure

- Set `output-tensor-meta=1` for the Wholebody X boxes template.
- Update tests to assert materialized X config exposes tensor metadata.
- Validate the materialized config under `build/` after running the profile materializer.

Phase 3: DS8 hook, gated off by default

- Add `models.wholebody_pose.enable=false` default behavior.
- Add `WholebodyPosePromoterProcessor`.
- Attach it only when the config enables it.
- Verify it is inert for all existing PGIE profiles.

Phase 4: Wholebody X active validation with YOLO26 still enabled

- Enable `wholebody_pose`.
- Keep `models.pose.enable=true`.
- Compare Wholebody payloads with YOLO26 payloads in the same frames.
- Confirm no object/tracker pollution.
- Confirm no increase in public object count from keypoint rows.

Phase 5: Wholebody X active validation with YOLO26 disabled

- Set `models.pose.enable=false`.
- Keep `wholebody_pose.enable=true`.
- Confirm overlay, world anchors, and diagnostics still work.
- Compare FPS/GPU against the YOLO26 pose baseline.

Phase 6: explicit pose source policy

- Add a small policy layer only after validation:
  - `auto`
  - `yolo26`
  - `wholebody`
  - `off`
- `wholebody` should require active Wholebody pose promotion and fail if unavailable.
- `auto` should log its selected source.

### 14. Documentation Updates When Implemented

Update these docs after implementation, not before:

- `docs/DS8_Baselines.md`
  - Add Wholebody X pose promoter status and expected profile behavior.
- `docs/DS8_metadata_contracts.md`
  - Change `NOESIS.POSE_FEATURES` producer from YOLO26-only to YOLO26 or Wholebody promoter.
  - Add `model="deimv2-wholebody49-x-boxes"` example.
- `docs/DS8_pose_stable_id_integration.md`
  - Clarify pose source options and whether StableID pose feature handoff is active.
- `docs/DS8_testing_guide.md`
  - Add Wholebody pose promoter validation command(s).
- `plans/DS8/ds8_design_decisions.md`
  - Record the decision to keep keypoint rows out of tracker objects and promote them as object user meta.

## Open Questions

- Does PGIE `label_xyxy_score` tensor metadata survive to `world_observation_stage` with `output-tensor-meta=1` in the current Service Maker runtime?
- Should foot detections `46` / `47` fill ankle pose slots by default, or only under a config flag after occupied-scene validation?
- Should StableID pose-feature handoff be repaired in the same implementation slice, or treated as a follow-up after world anchoring and overlay parity are proven?

## Agent Notes

- Read `AGENTS.md`, `noesis/AGENTS.md`, and `docs/AGENTS.md` before implementation.
- Before DS8 code changes, read `plans/DS8/ds8_master_work_orders.md` and the relevant checklists under `plans/DS8/`.
- Use only verified DeepStream/Service Maker APIs.
- Keep the parser body-only; do not emit keypoints as tracker objects.
- Do not import from `testpipelines/` in production code.
- Do not add appsinks, deprecated runtime paths, or hidden fallbacks.
- If changing native pose metadata plumbing, rebuild `noesis_pose_meta_ext` and run focused metadata tests.
