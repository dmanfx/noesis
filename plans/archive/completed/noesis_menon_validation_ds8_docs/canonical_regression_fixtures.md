# Canonical Regression Fixtures

Status: defined as of 2026-05-27; capture/asset population is incremental.

This file defines the canonical fixture set for long-term Noesis, BEV, and Menon
validation. Entries are intentionally explicit about required evidence so future
agents do not mistake a narrow smoke test for full coverage.

## Naming Rules

- Fixture IDs use `domain_room_camera_scenario_vN` when room/camera-specific.
- Saved clips, snapshots, and traces should live under a repo-relative fixture
  root declared in `fixture_registry.json`.
- If required evidence is unavailable, the regression case should report
  `blocked`, not pass through a substitute path.
- Every fixture should declare expected status, minimum check count, maximum
  warnings/failures, and visual artifact expectations when visuals are part of
  acceptance.

## Scene Generation Fixtures

| Fixture ID | Purpose | Required evidence | Required checks |
|---|---|---|---|
| `scene_empty_room_static_v1` | Static geometry and empty-room stability. | Empty-room image/clip, room anchors, plane export, depth anchors. | Coordinate system, known anchors, planes, room dimensions, depth static variance. |
| `scene_known_anchor_set_v1` | Scale, origin, and object-anchor correctness. | 5-10 measured anchors, camera mount, doorway edges, static object corners. | Known anchors, scene scale, semantic object anchors, camera reprojection overlay. |
| `scene_low_light_room_v1` | Robustness in low-light conditions. | Low-light clip, confidence map, depth/reconstruction outputs. | Depth confidence weighting, data-quality flags, regression failure categorization. |
| `scene_ir_night_room_v1` | IR/night behavior. | IR/night clip, camera calibration declaration, depth/plane outputs. | Intrinsics scope, depth ordering, plane-depth agreement, static-scene stability. |
| `scene_occluded_furniture_v1` | Depth/semantic resilience around occluded static objects. | Occluded furniture clip, object anchors, free-space mask. | Semantic support, known-object anchors, free-space/walkability, depth object agreement. |
| `scene_wide_dewarped_camera_v1` | Dewarp validation and wide-angle stress. | Raw/dewarped declaration, dewarped image, anchors at image edges. | Intrinsics raw/dewarped scope, anchor reprojection, floor grid overlay. |
| `scene_changed_furniture_v1` | Scene update and persistent-object drift. | Before/after scene exports and anchor measurements. | Scene diff artifact, semantic static-shift, mesh/asset sanity, failure taxonomy. |

## Tracking Fixtures

| Fixture ID | Purpose | Required evidence | Required checks |
|---|---|---|---|
| `tracking_straight_walk_v1` | Projection smoothness for simple motion. | DS8 telemetry with `backend_world_m`, bbox/footpoint evidence, BEV samples. | Detection-world projection, projection confidence, speed/acceleration, BEV agreement. |
| `tracking_stationary_person_v1` | Jitter and idle stability. | Saved telemetry over a standing interval. | Idle jitter, temporal confidence, BEV path smoothness, projection confidence. |
| `tracking_room_crossing_v1` | Cross-room containment and path quality. | Room-labeled track sequence plus room polygons. | Room containment, path smoothness, zone consistency, wall crossing. |
| `tracking_doorway_exit_v1` | Doorway transition logic. | Room transition through a known doorway segment. | Doorway transition, speed/acceleration, no wall crossing. |
| `tracking_couch_occlusion_v1` | Occlusion bridge behavior. | Track sequence with occlusion flags and uncertainty. | Occlusion bridge, uncertainty widening, identity continuity, ReID agreement. |
| `tracking_two_people_cross_v1` | ID switch/split/merge stress. | Two-person crossing telemetry with StableID/ReID fields. | Identity continuity, ReID/geometry contradiction, split/merge evidence. |
| `tracking_near_camera_v1` | Perspective and footpoint stress near camera. | Near-camera detections, bbox/mask/pose/depth footpoints. | Footpoint agreement, bbox aspect, projected bbox overlap, height sanity. |
| `tracking_far_camera_v1` | Small-detection stress at distance. | Far-field detections and confidence values. | Projection confidence, depth/object agreement, data-quality flags. |
| `tracking_ir_night_movement_v1` | Low-light movement tracking. | IR/night tracking telemetry and source-frame evidence. | Detection confidence, projection confidence, temporal continuity, data-quality flags. |
| `tracking_camera_edge_cases_v1` | Per-camera calibration edge cases. | Camera-specific clips near image edges and doorway/room boundaries. | Reprojection overlay, floor-ray, known anchors, room bounds. |

## Menon Fixtures

| Fixture ID | Purpose | Required evidence | Required checks |
|---|---|---|---|
| `menon_static_avatar_placement_v1` | Static world-to-scene placement. | Menon trace with `backend_world_m`, `world_to_menon_col_major`, rendered position. | Transform audit, world/Menon round-trip, placement agreement, floor contact. |
| `menon_bev_path_agreement_v1` | BEV/Menon path shape parity. | BEV trail, Menon trail, world-to-BEV and world-to-Menon transforms. | BEV world round-trip, BEV/Menon trail agreement, trail consistency. |
| `menon_camera_view_overlay_v1` | End-to-end camera reprojection. | Source frame, Menon camera render, overlay layers, bbox/avatar/mask metrics. | Menon camera reprojection, anchor/grid/edge pixel error, bbox/avatar/mask overlap. |
| `menon_room_mesh_collision_v1` | Room mesh and collision correctness. | Scene mesh, collision proxies, avatar/object positions. | Menon asset sanity, avatar collision, floor contact, doorway blocking. |
| `menon_multi_camera_frustums_v1` | Shared world alignment across cameras. | Multiple camera frustums, shared room geometry, trace placements. | Camera frustum coverage, BEV overlay, transform audit, camera marker sanity. |
| `menon_timestamp_sync_v1` | Live display freshness and latency. | Noesis frame timestamps, Menon display timestamps, render/update timestamps. | Timestamp alignment, latency summary, sync failure categorization. |

## Promotion Path

1. Add the fixture artifact files under the declared repo-relative fixture root.
2. Register the fixture in `fixture_registry.json` with expected status and
   artifact expectations.
3. Run the lowest tier that proves the fixture, then the registry runner.
4. Promote the fixture to the adoption gate only after it passes or intentionally
   reports blocked evidence for unavailable live dependencies.
