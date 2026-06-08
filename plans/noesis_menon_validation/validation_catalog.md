# Validation Catalog

This catalog is the target checklist for the Noesis/Menon validation toolbox.
Each validator should produce a common report row with status, severity,
confidence, failure category, evidence, and suggested next diagnostic.

## Core chain validators

| ID | Layer | Checks |
|---|---|---|
| CORE-001 | Input validity | Inputs present, camera IDs valid, timestamps valid, resolutions declared, units declared, confidence values present when expected, calibration files loaded, stale files detected. |
| CORE-002 | Numeric safety | Finite values, bounded outputs, no NaN/inf in transforms, coordinates, depths, confidences, velocities, or report fields. |
| CORE-003 | Transform validity | Matrix shape, invertibility, convention declared, scale preserved, axes correct, handedness not mirrored, round-trip error below threshold. |
| CORE-004 | Output validity | Schema match, physical bounds, confidence categories, debug fields, explicit failure state, downstream-safe values. |
| CORE-005 | Cross-system validity | Noesis agrees with BEV, Menon, camera reprojection, scene geometry, semantic logic, and historical state where applicable. |

## Scene generation validators

### Coordinate system

- Units are meters unless a field explicitly declares another unit.
- X/Y/Z axis convention matches the active Noesis/Menon contract.
- World origin is known, stable, and persisted between runs.
- Camera pose convention says world-to-camera or camera-to-world.
- Handedness checks detect mirrored spaces, inverted yaw, and flipped vertical
  axes.
- World -> camera -> world and world -> Menon -> world round trips stay within
  tolerance.
- Known scale anchors match measured room distances.
- Implemented fixture evidence lives in `scene.coordinate_systems` and produces
  `SCENE.coordinate_system` report rows.

### Known-anchor tolerances

| Item | Good | Warning | Fail |
|---|---:|---:|---:|
| Room width/length | <= 0.10 m | <= 0.20 m | > 0.20 m |
| Camera position | <= 0.15 m | <= 0.30 m | > 0.30 m |
| Doorway position | <= 0.10 m | <= 0.20 m | > 0.20 m |
| Object placement | <= 0.20 m | <= 0.40 m | > 0.40 m |

### Plane and room geometry

- Floor plane flatness and RMS deviation.
- Wall verticality and max angular error.
- Ceiling/floor parallelism.
- Wall/floor orthogonality.
- Room height consistency.
- Wall intersection validity.
- Door/window cutout plausibility.
- Plane normal consistency.
- Surface continuity across expected continuous walls/floors.
- Implemented fixture evidence for wall/opening/junction checks lives in
  `scene.room_geometry_constraints` and produces
  `SCENE.room_geometry_constraints` report rows.
- Window cutouts may declare `window_width_m`, `window_height_m`, and
  `window_sill_height_m`; impossible width, height, or sill placement is counted
  separately from doorway failures.

### Mesh quality

- Watertightness where expected.
- Non-manifold geometry.
- Inverted normals.
- Duplicate vertices/faces.
- Triangle-density bounds.
- LOD preservation.
- Texture atlas alignment.
- UV overlap.
- Bounding box sanity.
- Collision/BVH agreement with visible mesh.
- Implemented fixture evidence lives in `scene.meshes` and produces
  `SCENE.mesh_quality`. Mesh rows may declare `lod_max_error_m`,
  `texture_alignment_error_px`, `texture_stretch_ratio`, `bvh_valid`, and
  `collision_bvh_error_m` in addition to topology, normal, triangle, UV, and
  bounding-box fields.

### Menon asset checks

- Room mesh scale, orientation, origin, and collision mesh.
- Texture atlas mapping and stretching.
- Floor flatness, walkability, and alignment with tracking floor.
- Wall verticality and known-doorway clearance.
- Camera markers positioned and aimed correctly.
- Debug overlays can render anchors, frustums, floor grid, and track trails.
- Implemented fixture evidence lives in `scene.menon_assets` and produces
  `SCENE.menon_assets`. Assets may declare scale, orientation, origin,
  collision mesh, floor flatness, floor alignment, walkability, doorway blocking,
  camera marker position/aim, and required/rendered debug overlay layers.

### Depth and monocular reconstruction

- Relative depth ordering.
- Metric scale alignment from anchors.
- Temporal depth consistency for static regions.
- Plane-depth agreement for floors/walls.
- Object-depth agreement for detections.
- Depth discontinuities align with object/wall edges.
- Confidence weighting suppresses low-confidence depth in fusion.
- Static scene stability across repeated empty-room frames.
- Implemented fixture evidence lives in `scene.depth_anchors` and produces
  `SCENE.depth_consistency`. Depth anchors may declare `order_group`,
  `expected_order`, `object_depth_error_m`, `edge_alignment_error_px`,
  `confidence`, `fusion_weight`, and `static_std_m` in addition to metric depth,
  plane residual, and temporal variance.

### Semantic scene logic

- Objects rest on supporting surfaces.
- Objects do not intersect walls unless explicitly allowed by fixture metadata.
- Doorways remain traversable.
- Furniture sizes are plausible.
- Object-room compatibility is plausible.
- Persistent static objects remain stable between runs.
- Known-object anchors are used to lock scale.
- Free space remains walkable.
- Implemented fixture evidence lives in `scene.semantic_objects` and produces
  `SCENE.semantic_objects`. Objects may declare `allowed_rooms`,
  `doorway_clearance_m`, `free_space_clearance_m`,
  `walkable_area_blocked_ratio`, and `known_anchor_error_m` in addition to
  support, wall-intersection, size, room, and static-shift fields.

## Camera calibration validators

### Intrinsics

- Resolution matches inference/dewarped frame resolution.
- Focal lengths are plausible for the camera/lens model.
- Principal point is near the expected image center unless intentionally offset.
- `fx/fy` and FOV are consistent with aspect ratio and pixel aspect.
- Distortion model matches raw or dewarped stream.
- Dewarped virtual camera calibration explicitly declares raw/dewarped scope and
  inference resolution.
- Implemented fixture evidence for this contract lives in `cameras[]` with
  `stream_kind`, `applies_to_raw`, `applies_to_dewarped`, and
  `expected_resolution`; ambiguous dewarped ownership produces
  `CAM.intrinsics.dewarped_scope` failure.

### Extrinsics

- Camera position and height match physical mount expectations.
- Yaw, pitch, and roll point into the room.
- Frustum intersects expected floor/wall surfaces.
- Bottom-center and footpoint rays hit plausible floor locations.
- World-to-camera matrix convention is explicit.
- Known world points project to expected pixels.
- Known pixels backproject/raycast to expected floor points.

### Required visual debug artifacts

- Camera image with projected floor grid.
- Camera image with projected room outline.
- Camera image with projected known anchors.
- Camera image with detected bbox, projected bbox/avatar extent, mask polygon,
  and detected footpoint when fixture evidence exists.
- Menon view with camera frustum.
- BEV view with camera frustum and visible floor footprint.

## Tracking projection validators

### Detection to world

- Bbox bottom-center footpoint maps to plausible floor point.
- Segmentation lower-mask footpoint agrees with bbox footpoint when available.
- Pose ankles/feet agree with projected footpoint.
- Depth-assisted footpoint agrees with floor-plane intersection.
- 3D cuboid rests on the floor, not floating or sinking.
- Reconstructed person height is plausible.
- Person bbox aspect ratio is plausible for distance and perspective.
- Camera ray intersects floor within room bounds.
- Implemented fixture evidence lives in `projection_samples` and produces
  `TRACK.detection_world_projection`. Samples may include `image_bbox_xyxy`,
  `projected_bbox_xyxy`, `projected_bbox_iou`, and `bbox_center_error_px` to
  validate source-image aspect and 3D bbox reprojection overlap. Samples may
  also include `reprojection_score`, `temporal_smoothness_score`, and
  `semantic_validity_score` for projection-confidence components.

### Projection confidence inputs

- Bbox-foot agreement.
- Floor-contact score.
- Room-bounds score.
- Reprojection score.
- Image bbox aspect and projected 3D bbox overlap when present.
- Temporal-smoothness score.
- Semantic-validity score.

Suggested classification:

| Score | Level |
|---:|---|
| 0.85-1.00 | strong |
| 0.65-0.85 | usable |
| 0.45-0.65 | weak |
| < 0.45 | untrusted |

### BEV

- Track inside room polygon.
- No wall crossing.
- Doorway transitions only through known openings.
- Speed sanity.
- Acceleration sanity.
- Stationary jitter.
- Path smoothness.
- Camera coverage.
- Occlusion-aware uncertainty.
- Zone consistency.
- Implemented fixture evidence for doorway transitions uses room-labeled tracks
  plus `rooms[].doorways_xz` and produces `TRACK.doorway_transitions`.
- Implemented fixture/telemetry evidence for BEV path smoothness uses ordered
  `tracks` world X/Z samples and produces `BEV.path_smoothness`, which flags
  abrupt zig-zags and severe path-length inflation.

### BEV debug overlays

- Room polygons.
- Wall segments.
- Doorways.
- Camera frustums.
- Track trail.
- Confidence ellipse.
- Raw footpoints.
- StableID label.
- ReID confidence.
- Projection confidence.

### Temporal tracking

- Position continuity.
- Velocity limits.
- Acceleration limits.
- Idle stability.
- Occlusion bridge logic.
- Room containment.
- Door transition logic.
- Identity continuity.
- ReID contradiction detection.
- Track split/merge detection.
- Implemented fixture/telemetry evidence for occlusion bridges uses `occluded`
  and `occlusion_uncertainty_m` and produces `TRACK.occlusion_bridge`.
- Implemented fixture/telemetry evidence for identity continuity uses
  `stable_id`, `tracker_id`, `reid_identity`, `appearance_id`, and
  `reid_confidence` when available. These fields produce
  `TRACK.identity_continuity` and `TRACK.reid_geometry_consistency` rows that
  catch StableID switches, simultaneous duplicate StableID placement,
  split/merge evidence, ReID contradictions over continuous geometry, and weak
  appearance evidence.

## Menon projection validators

- Menon object position matches Noesis `backend_world_m` after the declared
  world-to-scene transform.
- Person/avatar scale matches real-world height assumptions.
- Avatar feet rest on Menon floor mesh.
- Avatar does not pass through walls/furniture unless explicitly allowed.
- Menon camera-view render aligns with the original camera detection.
- BEV and Menon top-down positions agree.
- Orientation/facing/movement direction is plausible.
- Menon display timestamp matches Noesis frame timestamp within threshold.
- Menon trail shape matches BEV trail shape.
- Transform audit logs every transform used to place each entity.
- Static/generated Menon objects match declared Noesis world anchors, rendered
  dimensions, support surfaces, room compatibility, and wall/furniture collision
  constraints.

Saved Menon traces should include:

- `world_to_menon_col_major` or `align.scene_similarity.world_to_scene_col_major`;
- optional `world_to_bev_col_major` plus `world_bev_points` for explicit
  world-to-BEV agreement and round-trip validation;
- placement pairs with `world_point`, `menon_point`, `noesis_ts_s`, and
  `menon_ts_s`;
- trail pairs with Noesis world points and Menon scene points;
- optional `bev_trails` with BEV X/Z points and Menon scene points for direct
  BEV-to-Menon path agreement;
- optional `avatars` plus `collision` geometry for rendered height, wall/furniture
  collision, and movement-orientation checks;
- optional `objects` with world/Menon positions, scene bboxes, expected/observed
  dimensions, support surfaces, allowed rooms, and collision evidence for static
  object placement checks;
- optional `latency_samples` with Noesis frame, telemetry, Menon update, Menon
  render, and Menon display timestamps for display freshness checks;
- transform-audit stages that explicitly declare `backend_world_m` input and
  `menon_scene` output frames.

Live Menon browser captures should be converted into the same trace shape using
browser debug evidence only when the snapshot exposes both declared
`backend_world_m` source points and Menon scene placement points. The capture
tool should fail or block if Playwright, Menon, browser debug globals, or the
world-to-scene transform is missing.

Required end-to-end feature:

- Camera reprojection mode in Menon or an equivalent validation runner that
  renders from a selected camera pose and compares the Menon room/avatar/anchor
  projection against the source frame.
- Implemented saved-trace evidence for this mode lives in
  `camera_reprojections` and produces `MENON.camera_reprojection`. A valid
  sample declares source frame, Menon render, detected bbox, projected avatar,
  room mesh edges, floor grid, and anchor layers, then reports pixel error and
  overlap metrics where available.

## Cross-space validators

For every tracked point when data is available, store:

- pixel point;
- camera ray;
- world point;
- BEV point;
- Menon point;
- reprojected pixel;
- reprojection error.

Required checks:

- Pixel -> world -> pixel has low reprojection error.
- World -> BEV -> world has no scale/origin drift.
- World -> Menon -> world has no scale/origin drift.
- Menon camera render aligns with source camera image.
- BEV trail and Menon trail have the same path shape.
- Track room assignment and Menon room label agree.

## Reprojection metrics

### Scene reprojection

- Mean reprojection error.
- Max reprojection error.
- Edge alignment score.
- Anchor alignment score.
- Floor-grid alignment score.
- Generated mesh edge support against source image edges when available.

### Tracking reprojection

- Footpoint reprojection error.
- Bbox center reprojection error.
- 3D bbox reprojection overlap.
- Segmentation mask overlap.
- Pose skeleton/foot overlap.

## Physical plausibility validators

- Gravity: person or object does not float.
- Collision: person or object does not pass through walls.
- Speed limit.
- Acceleration limit.
- Reachability.
- Support.
- Containment.
- Line of sight.
- Persistence.
- Doorway transition logic.

These checks should produce flags, not always hard failures. Reports must
distinguish impossible, unlikely, uncertain, and temporarily unresolved.

## Regression fixtures

### Scene generation

- Empty room clip.
- Known anchor image set.
- Low-light room clip.
- IR/night clip.
- Occluded furniture clip.
- Wide-angle/dewarped clip.
- Changed furniture clip.

### Tracking

- Person walks straight line.
- Person stands still.
- Person crosses room.
- Person exits doorway.
- Person occluded by couch or furniture.
- Two people cross paths.
- Person near camera.
- Person far from camera.
- IR/night movement.
- Camera-specific edge cases.

### Menon

- Static avatar placement.
- BEV/Menon path agreement.
- Camera-view render overlay.
- Room mesh collision.
- Multiple camera frustums.
- Timestamp sync.
