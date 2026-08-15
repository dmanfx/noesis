# Decisions - Menon World Unification

## DEC-001 - Canonical frame

- Date: 2026-02-10
- Decision: Superseded on 2026-03-31. Baseline canonical backend world outputs now use `backend_world_m`; `menon_scene` is a derived Menon render frame only.
- Rationale: One producer-owned metric world frame removes scene-unit leakage, hidden flips, and duplicate pose authority between Noesis and Menon.

## DEC-002 - Strict pose-only mode

- Date: 2026-02-10
- Decision: Enable strict mode via `NOESIS_CALIBRATION_POSE_ONLY=1` with startup hard-fail on missing/invalid pose.
- Rationale: Eliminates hidden fallback behavior and makes validation deterministic.

## DEC-003 - No meter conversion in this phase

- Date: 2026-02-10
- Decision: Superseded on 2026-03-31. Backend contracts now use meters canonically and Menon derives scene units explicitly from alignment metadata.
- Rationale: Relative depth, BEV, and world tracking all become simpler and more auditable once meters are the only backend unit.

## DEC-004 - No fallback calibration authority

- Date: 2026-02-10
- Decision: In strict mode, pose is the only accepted calibration authority for this feature path.
- Rationale: Prevent mixed-source drift and ambiguous debugging.

## DEC-005 - Baseline defaults for world outputs

- Date: 2026-02-10
- Decision: Superseded on 2026-03-31. Baseline tracking and BEV world mode now default to `backend_world_m`; camera-local BEV/floorplan products use `camera_local_ground_m`.
- Rationale: Canonical backend world meters plus explicit local-ground products remove mixed scene/local semantics and make BEV parity testable.

## DEC-006 - Shared image-axis flip inference for world rays

- Date: 2026-02-10
- Decision: Superseded on 2026-03-31. Canonical world-ray projection no longer uses inferred image-axis flips; any remaining flip hints are diagnostic-only.
- Rationale: Hidden image flips were compensating for a pose-basis mismatch and became a source of mirrored BEV/world outputs once the pose boundary was corrected.

## DEC-007 - Baseline calibration artifact lock to Menon OBJ units

- Date: 2026-02-10
- Decision: Superseded on 2026-03-31. Baseline DS8 now defaults to `config/camera_calibration.json`; the OBJ-unit artifact remains legacy compatibility only.
- Rationale: The runtime default should match the canonical backend meter contract rather than baking scene-unit assumptions into baseline startup.

## DEC-008 - Menon world consumer does not rescale backend positions

- Date: 2026-02-10
- Decision: Superseded on 2026-03-31. Menon converts backend `track.world` from `backend_world_m` into `menon_scene` exactly once before render/update consumption.
- Rationale: Backend positions remain canonical meters, but Menon render systems still operate in scene space; explicit one-time conversion preserves both truths.

## DEC-009 - Camera pose authority

- Date: 2026-03-31
- Decision: Calibration pose/extrinsics are the single camera authority for backend world tracking, BEV, and Menon reprojection. Device-layer camera poses are no longer the default reprojection authority.
- Rationale: Split camera pose ownership was making backend world placement and Menon reprojection impossible to reconcile deterministically.

## DEC-010 - Canonical local-ground frame

- Date: 2026-03-31
- Decision: `camera_local_ground_m` is the only local BEV/floorplan frame. Local products must use full world->camera projection onto the ground plane, not a yaw-only approximation.
- Rationale: Yaw-only local conversion produces diagonal forward/back motion when cameras are pitched, which is exactly the misalignment seen in living-room BEV.

## DEC-011 - Backend global entities are Menon's only production person authority

- Date: 2026-07-10
- Decision: Menon accepts only `noesis.world.snapshot` v1 for production people and occupancy. Per-camera tracking projection is retained solely as labeled diagnostics and never as fallback.
- Rationale: Camera election, client fusion, range repair, smoothing, and stale holds duplicate backend world ownership and can silently invent a different household state.

## DEC-012 - One explicit world-to-scene boundary

- Date: 2026-07-10
- Decision: The explicit authored scene-similarity matrix is applied exactly once at the renderer boundary; canonical visuals mount at scene root and promoted authored geometry is not transformed again.
- Rationale: Parent transforms and compatibility matrices were producing untraceable coordinate drift and double transforms.

## DEC-013 - Promoted scene release is the only production scene authority

- Date: 2026-07-10
- Decision: Environment, virtual-twin surface, and room reconstruction consumers use only the coherent promoted release and exact current-release URLs. Revision scans and newest-per-camera selection are non-production and removed from these consumers.
- Rationale: A visually plausible mixture of independent camera revisions is not a coherent world model.

## DEC-014 - Authored scene dependencies are first-class immutable release artifacts

- Date: 2026-07-10
- Decision: OBJ, MTL, and every referenced texture must be inventoried, content-addressed, fetched, and verified together before Menon loads the environment.
- Rationale: Hashing only OBJ geometry leaves the visible scene dependent on mutable out-of-band material files.
