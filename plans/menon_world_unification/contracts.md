# Contracts - Menon World Unification

## 1) Canonical world contract

- Canonical backend frame: `backend_world_m`.
- Derived Menon render frame: `menon_scene`.
- Coordinate handedness: right-handed.
- Up axis: `+Y`.
- Phase scope: baseline (non-V3DT) tracking, BEV, floorplan, and Menon reprojection/render consumers.

## 2) Pose schema (`PoseV1`)

Expected location:
- Baseline and `v3dt` default: `config/camera_calibration.json` -> `cameras.<cameraId>.pose`
- Legacy compatibility artifact only: `config/camera_calibration_menon_obj.json`

Required fields:
- `position`: `[x, y, z]`
- `yaw_pitch_roll_deg`: `[yaw, pitch, roll]` in degrees
- `rotation_order`: `"YXZ"`
- `frame`: `"backend_world_m"` for canonical runtime calibration

Legacy compatibility:
- `frame: "menon_scene"` remains accepted on ingest and is converted to backend world meters using alignment and unit metadata.

Optional field:
- `source`: free-form source tag (for diagnostics only)

Validation rules:
- `position` length exactly 3, finite numbers only.
- `yaw_pitch_roll_deg` length exactly 3, finite numbers only.
- `rotation_order` must be exactly `"YXZ"`.
- `frame` must be exactly `"backend_world_m"` or `"menon_scene"`.

## 3) Pose-to-extrinsics conversion contract

- Build camera pose matrix `Twc` from `position` and Euler angles in backend world meters.
- Euler order is fixed to `YXZ`.
- If the input pose is `menon_scene`, convert position and rotation through alignment + unit metadata into backend world meters before synthesis.
- Convert to extrinsics with `E = inv(Twc)`.
- Persist `E` as 16 floats in column-major order.

## 4) Calibration source selection contract

- Runtime override: `NOESIS_CALIBRATION_EXTRINSICS` (if set) is authoritative for all tracking modes.
- Baseline mode (`tracking_mode != "v3dt"`) default source is `config/camera_calibration.json`.
- `v3dt` default source remains `config/camera_calibration.json`.
- `config/camera_calibration_menon_obj.json` is legacy compatibility only and is not the runtime default.

## 5) Strict mode contract

Environment flag:
- `NOESIS_CALIBRATION_POSE_ONLY` (default: enabled / truthy)

Behavior:
- Pose is mandatory for every configured camera.
- Missing or invalid pose for any configured camera aborts startup.
- No fallback to legacy `E` or `Twc` payload authority.
- No hidden world/scene fallback authority.
- Runtime `set_extrinsics` in strict mode accepts only valid `pose` payload authority.

## 6) World output contract

- Baseline track world outputs must emit:
  - `world`: `[x, y, z]`
  - `world_valid`: `true|false`
  - `world_frame`: `"backend_world_m"` when valid
- BEV world mode must use the same `backend_world_m` coordinate frame and match track-world orientation.
- BEV local mode must emit `world_frame: "camera_local_ground_m"` and use full world->camera projection, not yaw-only rotation.
- Floorplan / height / distance products must be labeled `frame: "camera_local_ground_m"` with metric bounds.
- Menon derives `menon_scene` exactly once from backend world meters before rendering. Preferred path is the explicit `align.scene_similarity.world_to_scene_col_major` transform when present; otherwise legacy alignment + unit metadata may be used only as a compatibility path.

## 7) Validation evidence contract

Every completed validation item must include:
- command or method
- file path(s) touched
- pass/fail outcome

## 8) Canonical Menon world consumer

- Production input is only `noesis.world.snapshot` contract version 1.
- Producer runtime/instance/run identity, sequence, observation/publish timestamps, entity/source bounds, and `backend_world_m`/`meters` coordinates are validated before commit.
- A snapshot is authoritative: absent and `lost` entities are removed immediately.
- Resident, visitor, unknown, and provisional subjects remain explicit; Menon does not re-identify or merge them.
- Per-camera `tracking` is diagnostic-only and cannot become a render or occupancy fallback.

## 9) Renderer transform boundary

- The explicit `align.scene_similarity.world_to_scene_col_major` matrix is required.
- Each backend-world entity or reconstruction artifact crosses that matrix exactly once.
- Canonical person visuals mount at Three.js scene root in `menon_scene`; they are not children of an additionally transformed environment.
- Missing, incompatible, or repeated transforms fail closed.

## 10) Promoted scene consumer

- Production scene metadata is read only from `/api/v1/scenes/current/payload`.
- Authored geometry is read only from `authored_scene_url`.
- Authored MTL/textures are read only from declared `authored_scene_dependency_urls`.
- Camera assets are read only from the declared current-release role URLs.
- Release/camera/revision/role inventories, cohort fingerprints, same-origin URL paths, byte lengths, and SHA-256 values are validated before atomic commit.
- OBJ/MTL/texture references must match the dependency inventory exactly; missing, extra, ambiguous, or escaping references fail closed.
- No latest, revision scan, newest-per-camera, local authored geometry, local material, or untextured fallback is permitted.

## 11) Canonical occupancy

- Menon derives occupancy once from canonical world entities with `room_id`.
- Per-camera analytics occupancy remains diagnostic-only and is never summed in the browser.
