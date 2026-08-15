# Validation Artifact Contracts

Status: planning scaffold as of 2026-05-27.

All validators should emit one common machine-readable format and optional visual
or Markdown companions. The common format lets future agents compare failures
without reverse-engineering one-off logs.

## Validation result levels

| Level | Meaning |
|---|---|
| verified | Multiple independent checks agree. |
| likely | Good confidence, but not fully cross-validated. |
| tentative | Usable, but weak or partially contradicted. |
| untrusted | Should not drive automation or analytics. |
| invalid | Known-bad result. |

## Check status values

| Status | Meaning |
|---|---|
| pass | Check met its threshold. |
| warning | Check is usable but outside the good band. |
| fail | Check violated a required threshold. |
| blocked | Required evidence was unavailable. |
| skipped | Check was intentionally out of scope for this run. |

## Failure taxonomy

| Type | Examples |
|---|---|
| calibration_failure | Bad intrinsics, bad extrinsics, wrong dewarp model. |
| transform_failure | Matrix convention wrong, axis flipped, scale mismatch. |
| projection_failure | Footpoint outside room, high reprojection error. |
| scene_failure | Warped room, invalid mesh, wrong scale. |
| temporal_failure | Teleport, ID switch, impossible velocity. |
| semantic_failure | Person through wall, object floating, doorway mismatch. |
| sync_failure | Menon displays stale track position or mismatched timestamp. |
| model_failure | Detector, pose, depth, or reconstruction hallucination. |
| data_quality_failure | Motion blur, glare, occlusion, low light, missing frame. |
| regression_failure | Previously passing scenario now fails. |
| infrastructure_failure | Tool, fixture, Menon checkout, DS8 runtime, or dependency unavailable. |

Each failure must include:

- what failed;
- where it failed;
- severity;
- likely subsystem;
- supporting evidence;
- suggested next diagnostic.

## Confidence categories

Reports should keep separate scores for:

- detection confidence;
- tracking confidence;
- ReID confidence;
- projection confidence;
- geometry confidence;
- temporal confidence;
- semantic confidence;
- end-to-end confidence.

Avoid replacing these with a single global score. If an aggregate is useful, it
must also list the category scores and any contradiction flags.

## JSON report skeleton

```json
{
  "schema_version": 1,
  "run_id": "20260527T120000_noesis_menon_validation",
  "created_at": "2026-05-27T12:00:00Z",
  "source": {
    "repo": "Noesis_Devel",
    "git_revision": "<sha-or-dirty-marker>",
    "pipeline_config": "config/infer.yaml",
    "cameras_config": "config/cameras.yaml",
    "menon_available": true,
    "menon_revision": "<sha-or-null>"
  },
  "scope": {
    "rooms": ["family-room"],
    "cameras": ["family-room"],
    "fixtures": ["known_anchors_family_room_v1"],
    "tiers": ["unit", "fixture", "runtime", "menon"]
  },
  "summary": {
    "status": "warning",
    "level": "likely",
    "end_to_end_confidence": 0.78,
    "failure_count": 0,
    "warning_count": 1,
    "blocked_count": 0
  },
  "confidence": {
    "detection": 0.91,
    "tracking": 0.87,
    "reid": 0.74,
    "projection": 0.78,
    "geometry": 0.86,
    "temporal": 0.90,
    "semantic": 0.82,
    "end_to_end": 0.78
  },
  "checks": [
    {
      "id": "CAM-REPROJECT-001",
      "domain": "camera",
      "name": "known_anchor_reprojection",
      "status": "warning",
      "level": "tentative",
      "failure_type": "projection_failure",
      "severity": "warning",
      "camera": "family-room",
      "metric": {
        "mean_error_px": 18.5,
        "max_error_px": 42.0
      },
      "threshold": {
        "good_max_px": 12.0,
        "fail_max_px": 50.0
      },
      "evidence": [
        "visual/camera_family_room_anchor_overlay.png"
      ],
      "detail": "East wall anchors are shifted right in camera reprojection.",
      "suggested_next_diagnostic": "Run extrinsics floor-ray and Menon camera-view comparison for family-room."
    }
  ],
  "artifacts": {
    "markdown_report": "report.md",
    "visual_index": "visual/index.json",
    "telemetry_excerpt": "telemetry/tracking.ndjson"
  }
}
```

## Visual artifacts

Validators should support these artifact types when source data exists:

- camera image with projected floor grid;
- camera image with projected room mesh;
- camera image with detected footpoints;
- camera image with 3D bbox or avatar reprojection;
- BEV with tracks, trails, frustums, walls, doorways, and confidence overlays;
- Menon top-down screenshot;
- Menon camera-view render;
- projection or geometry error heatmap;
- track timeline chart;
- scene diff overlay between revisions.

Visual artifacts should be indexed under `visual/index.json` inside the run
directory so agents can find supporting images without parsing prose reports.

The fixture runner currently writes:

- `visual/camera_<camera_id>_reprojection.png` for floor-grid, anchor,
  room-outline, mesh-edge, footpoint, detected-bbox, projected-bbox/avatar, and
  mask-polygon camera reprojection evidence;
- `visual/bev_diagnostic.png` for room-polygon, wall, doorway, frustum,
  track-trail, raw-footpoint, uncertainty ellipse, StableID, ReID confidence,
  and projection-confidence BEV evidence.

## Numeric artifacts

- Geometry validation report.
- Calibration validation report.
- Reprojection report.
- Track validation report.
- Confidence report.
- Regression report.
- Failure triage report.
- Versioned benchmark summary.

The registry regression runner writes `regression_summary.json` with one row per
fixture case, including kind, status, level, check counts, and report paths.
Rows also include `regression_status` and `regression_failures` when a case
misses the expected thresholds declared in `fixture_registry.json`.
Each case and the suite summary include `failure_categories`,
`dominant_failure_category`, and `suggested_diagnostic_focus` so agents can route
failures toward calibration, transform, projection, scene, temporal, semantic,
sync, model, data quality, infrastructure, or regression diagnostics.
For artifact-level checks, rows include `artifact_comparisons`; image entries can
compare an actual artifact against a `golden_path` and write a diff image under
`regression_diffs/`.

## Track audit object

Per-track reports should include:

```json
{
  "track_id": 17,
  "stable_id": 4,
  "current_room": "family-room",
  "world_position": {"x": 3.2, "y": 0.0, "z": 1.4},
  "projection_confidence": 0.81,
  "temporal_confidence": 0.92,
  "reid_confidence": 0.76,
  "last_doorway_transition": null,
  "last_occlusion_age_s": 2.4,
  "last_impossible_motion_event": null,
  "warnings": ["slight BEV jitter near couch edge"]
}
```

The saved/live telemetry runner writes this family of evidence to
`tracking/track_audit.json`.

The current audit also includes `identity`, `camera_id`, timestamp bounds,
`p95_speed_m_s`, `max_speed_m_s`, optional `reid_identity`, `appearance_id`,
`appearance_key`, `appearance_keys_observed`, and both object-style
`world_position` and legacy list-style `current_world_position` so older
consumers can keep reading the artifact while newer checks use named axes.

Tracking fixtures and telemetry may include `occluded` plus
`occlusion_uncertainty_m` on track samples. These fields drive the
`TRACK.occlusion_bridge` check, which validates that occluded spans bridge
visible samples with plausible duration/speed and that uncertainty is not
understated. Room-labeled track samples plus `rooms[].doorways_xz` drive the
`TRACK.doorway_transitions` check, which verifies room changes pass through a
known opening.

Ordered track samples also drive `BEV.path_smoothness`, which compares BEV X/Z
trail shape against path-length inflation and abrupt heading changes. This
complements speed and acceleration checks by catching zig-zag paths that can
look locally plausible but are visually unstable in the top-down view.

Tracking fixtures and telemetry may also include `reid_identity`,
`appearance_id`, `appearance_cluster`, and `reid_confidence`. These fields drive
the `TRACK.identity_continuity` and `TRACK.reid_geometry_consistency` checks,
which flag StableID switches, simultaneous duplicate StableID placement,
split/merge evidence, continuous-geometry ReID contradictions, and low
appearance confidence.

## Semantic scene object fields

Scene fixtures may include `scene.semantic_objects` rows with:

- `bbox_min`, `bbox_max`, and `support_y` for support and dimension checks;
- `max_wall_intersection_m` for object-wall collision evidence;
- `allowed_rooms` plus `room` for object-room compatibility;
- `doorway_clearance_m` for doorway blockage checks;
- `free_space_clearance_m` and `walkable_area_blocked_ratio` for walkability
  evidence;
- `max_static_shift_m` and `known_anchor_error_m` for persistent-object and
  known-object anchor stability.

These fields drive `SCENE.semantic_objects` and report separate counters for
room incompatibility, doorway clearance, free-space clearance, walkable-area
blockage, known-anchor error, static-object drift, wall intersection, support
error, and implausible object dimensions.

## Detection projection sample fields

Fixture `projection_samples` rows may include:

- `world_point` plus `bbox_foot_world`, `mask_foot_world`, `pose_foot_world`,
  and `depth_world` for footpoint agreement;
- `floor_y`, `person_height_m`, `ray_floor_valid`, and `room_polygon_xz` for
  physical and room-bound checks;
- `image_bbox_xyxy` for the source image detection box in `[x1, y1, x2, y2]`
  pixels;
- `projected_bbox_xyxy` for the reprojected 3D box/image extent in the same
  pixel convention;
- `projected_bbox_iou` or `bbox_center_error_px` when a producer has already
  computed those metrics;
- `reprojection_score`, `temporal_smoothness_score`, and
  `semantic_validity_score` for explicit projection confidence components.

These fields drive `TRACK.detection_world_projection` and contribute optional
components to `TRACK.projection_confidence`.

## Menon asset sanity fields

Scene fixtures may include `scene.menon_assets` rows with:

- `asset_type` values such as `room_mesh`, `floor_mesh`, `wall_mesh`,
  `camera_marker`, and `debug_overlay`;
- `scale_error_m`, `orientation_error_deg`, `origin_error_m`, and
  `collision_mesh_error_m` for room mesh and collision parity;
- `floor_flatness_m`, `floor_alignment_error_m`, and `walkable` for floor mesh
  validation;
- `doorway_blocked` for wall/doorway clearance evidence;
- `camera_position_error_m` and `camera_aim_error_deg` for camera marker
  placement;
- `overlay_rendered_layers` and `required_overlay_layers` for debug overlay
  renderability.

These fields drive `SCENE.menon_assets`.

## Mesh quality fields

Scene fixtures may include `scene.meshes` rows with topology and rendering
quality evidence:

- bounding boxes, triangle counts, watertight expectation, non-manifold edges,
  inverted faces, duplicate faces, and UV overlap;
- `lod_max_error_m` for decimated mesh preservation;
- `texture_alignment_error_px` and `texture_stretch_ratio` for texture atlas
  bake quality;
- `bvh_valid` and `collision_bvh_error_m` for collision/BVH parity against the
  visible mesh.

These fields drive `SCENE.mesh_quality`.

## Menon trace audit object

Captured Menon placement traces should include enough evidence to explain every
world-to-scene placement:

```json
{
  "entity_id": "stable-1",
  "camera_id": "fixture-camera",
  "room": "fixture-room",
  "world_point": [0.45, 0.0, 0.15],
  "menon_point": [0.45, 0.0, 0.15],
  "noesis_ts_s": 10.5,
  "menon_ts_s": 10.53,
  "transform_audit": [
    {"stage": "source_payload", "frame": "backend_world_m"},
    {
      "stage": "scene_similarity",
      "from_frame": "backend_world_m",
      "to_frame": "menon_scene"
    },
    {"stage": "render_anchor", "frame": "menon_scene"}
  ]
}
```

Traces can also include `world_to_bev_col_major` and `world_bev_points`:

```json
{
  "world_to_bev_col_major": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  "world_bev_points": [
    {
      "entity_id": "stable-1",
      "world_point": [0.45, 0.0, 0.15],
      "bev_point_xz": [0.45, 0.15]
    }
  ]
}
```

This drives `BEV.world_round_trip`, which checks world-to-BEV agreement and
BEV-to-world round-trip error so scale/origin drift is visible before Menon
placement is inspected.

The Menon trace runner writes a copied trace to `menon/trace.json`, summary
counts to `menon/trace_audit.json`, and the common JSON/Markdown reports at the
run root.

The browser capture path first writes `browser_snapshot.json` beside the
converted trace, then converts declared browser debug evidence into this same
trace object. A placement produced from browser state must still contain a
declared `backend_world_m` source point, a Menon scene point, and a
world-to-scene transform. Missing fields should produce failed or blocked
checks, not inferred placement evidence.
When browser debug state exposes camera-view reprojection evidence, the adapter
promotes it into `camera_reprojections` so fixture and live modes share the same
`MENON.camera_reprojection` gate.

Optional cross-space trail and avatar evidence can extend the same trace:

```json
{
  "bev_trails": [
    {
      "entity_id": "stable-1",
      "floor_y_world": 0.0,
      "bev_points_xz": [[0.0, 0.0], [0.45, 0.15]],
      "menon_points": [[0.0, 0.0, 0.0], [0.45, 0.0, 0.15]]
    }
  ],
  "avatars": [
    {
      "entity_id": "stable-1",
      "menon_point": [0.45, 0.0, 0.15],
      "radius_scene": 0.18,
      "expected_height_m": 1.72,
      "observed_height_scene": 1.70,
      "heading_deg": 18.4,
      "movement_vector_xz": [0.45, 0.15]
    }
  ],
  "objects": [
    {
      "object_id": "fixture-couch",
      "category": "couch",
      "world_point": [0.45, 0.0, 0.15],
      "menon_point": [0.45, 0.0, 0.15],
      "bbox_min_scene": [0.25, 0.0, -0.05],
      "bbox_max_scene": [0.95, 0.8, 0.35],
      "expected_dimensions_m": [0.70, 0.80, 0.40],
      "observed_dimensions_scene": [0.70, 0.80, 0.40],
      "support_y_scene": 0.0,
      "allowed_rooms": ["fixture-room"]
    }
  ],
  "collision": {
    "wall_segments_xz": [[[0.0, 0.0], [1.0, 0.0]]],
    "obstacle_boxes_xz": [
      {"bbox_min_xz": [1.35, -0.45], "bbox_max_xz": [1.70, -0.15]}
    ]
  },
  "camera_reprojections": [
    {
      "camera_id": "family-room",
      "source_frame": "visual/source_frame_family-room.png",
      "menon_render": "visual/menon_camera_family-room.png",
      "overlay_path": "visual/menon_camera_family-room_overlay.png",
      "layers": [
        "source_frame",
        "menon_render",
        "detected_bbox",
        "projected_avatar",
        "room_mesh_edges",
        "floor_grid",
        "anchors"
      ],
      "anchor_mean_error_px": 8.0,
      "anchor_max_error_px": 18.0,
      "floor_grid_mean_error_px": 9.5,
      "room_edge_mean_error_px": 11.0,
      "bbox_iou": 0.62,
      "avatar_iou": 0.66,
      "mask_iou": 0.58
    }
  ],
  "latency_samples": [
    {
      "entity_id": "stable-1",
      "noesis_ts_s": 10.5,
      "telemetry_ts_s": 10.51,
      "menon_update_ts_s": 10.53,
      "menon_render_ts_s": 10.55,
      "menon_display_ts_s": 10.56
    }
  ]
}
```

These fields drive BEV/Menon trail agreement, avatar scale, collision, and
movement-orientation checks, plus static/generated object placement checks.
`latency_samples` drives `MENON.latency_alignment`, and `camera_reprojections`
drives `MENON.camera_reprojection`, which blocks when source/render/layer
evidence is missing and warns or fails when pixel-error or overlap metrics fall
outside the declared thresholds.

## Required blocked evidence

When a required check cannot run, the report must say why. Examples:

- Menon checkout unavailable.
- DS8 runtime did not start.
- Fixture lacks known anchors.
- Calibration bundle missing dewarped resolution.
- Source clip has no visible person.
- Required visual frame was not captured.

A blocked check is not a pass. It is evidence that acceptance is incomplete.

## Guided waypoint calibration evidence

`scripts/noesis_alignment_walk.py waypoint-calibration` writes owner-private
`noesis.alignment.waypoint_evidence` v1 rows. A complete row binds one marked
waypoint to one deterministic nearest-marker sample and preserves:

- the fit or untouched-holdout role and known Menon scene XYZ;
- exact camera, run-local tracklet, frame, media PTS, image foot, and image size;
- the captured calibration bundle, K, world-to-camera E, and active similarity
  digest;
- camera center and the observed unit ray in camera and backend-world frames;
- raw, registered, and used depth plus the raw-depth-to-physical-optical-depth
  fit pair;
- raw floor/depth candidates, prefilter measurement, filter prediction, and
  final world position with active and advisory-candidate metric errors.

`noesis.alignment.waypoint_metrics` v1 reports binding and per-stage coverage
plus error distributions separately for fit and holdout. The candidate
similarity is fit only from complete non-collinear fit rows and is advisory;
the report writer never mutates active calibration.

`noesis.alignment.waypoint_camera_calibration_candidates` v1 groups evidence by
camera. It uses only complete FIT waypoint rows to solve a proper
camera-to-backend-world rotation while holding the captured optical center
fixed, then emits the equivalent candidate world-to-camera E. Parallel or
otherwise degenerate directions block the solve. An unconstrained reflective
solution is corrected to a determinant-+1 diagnostic matrix but is explicitly
rejected for admission so mirrored evidence cannot be hidden by the correction.
With an admissible rotation, at least 32 coherent post-marker FIT samples feed
the repository's verified monotonic piecewise fitter from raw DAv2 range to
physical candidate-camera optical Z; values outside the learned raw domain are
not extrapolated. HOLDOUT waypoints and samples enter neither solver.

Per-camera FIT/HOLDOUT output reports coverage plus median, p95, and maximum
angular-ray, image-reprojection, registered/mapped optical-depth, reconstructed
floor/depth position, and producer-stage position errors. Source capture,
calibration bundle, active and candidate similarity, K/E, rotation candidate,
and exact fit-input digests bind the advisory outputs. A derived artifact index
hashes every report output and binds them to the original capture artifact
index.

## Depth anchor evidence

Generated-scene and monocular-depth fixtures can declare depth evidence under
`scene.depth_anchors`:

```json
{
  "scene": {
    "depth_anchors": [
      {
        "anchor_id": "floor_midpoint",
        "expected_depth_m": 2.0,
        "observed_depth_m": 2.05,
        "order_group": "family-room-camera-depth-order",
        "expected_order": 1,
        "plane_residual_m": 0.03,
        "object_depth_error_m": 0.04,
        "temporal_std_m": 0.02,
        "static_std_m": 0.02,
        "edge_alignment_error_px": 3.0,
        "confidence": 0.82,
        "fusion_weight": 0.70
      }
    ]
  }
}
```

These fields drive `SCENE.depth_consistency`: metric scale, relative ordering,
plane-depth agreement, object-depth agreement, temporal stability, static-scene
stability, depth-edge alignment, and low-confidence fusion-weight checks.

## Scene coordinate and room-geometry evidence

Generated scene fixtures can declare coordinate-system and room-constraint
measurements directly under `scene`:

```json
{
  "scene": {
    "coordinate_systems": [
      {
        "scene_id": "family_room_scene",
        "units": "m",
        "axis_convention": "x_right_y_up_z_forward",
        "camera_pose_convention": "world_to_camera",
        "origin_delta_m": 0.02,
        "transform_col_major": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "round_trip_points": [[0, 0, 0], [1, 0, 1]],
        "scale_anchor_expected_m": 3.0,
        "scale_anchor_observed_m": 3.02
      }
    ],
    "room_geometry_constraints": [
      {
        "constraint_id": "main_doorway",
        "constraint_type": "doorway",
        "doorway_width_m": 0.86,
        "doorway_height_m": 2.03,
        "doorway_bottom_gap_m": 0.0,
        "surface_gap_m": 0.02
      },
      {
        "constraint_id": "north_window",
        "constraint_type": "window",
        "window_width_m": 1.10,
        "window_height_m": 0.80,
        "window_sill_height_m": 0.92,
        "surface_gap_m": 0.02
      }
    ]
  }
}
```

These fields drive coordinate unit/axis/pose/handedness/origin/scale checks and
room wall-junction, doorway, window cutout, orthogonality, and
surface-continuity checks. Window rows report `impossible_window_count`
separately from `impossible_opening_count` so agents know whether to inspect
window segmentation or doorway/floor-contact evidence first.
