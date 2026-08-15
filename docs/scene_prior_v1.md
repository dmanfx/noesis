# Scene Prior v1

Scene Prior v1 turns a verified, world-aligned room walk into immutable spatial
evidence that Noesis can compare with live tracking and floorplan observations.
It is generic across sites, physical spaces, cameras, and scan revisions. The
current deployed catalog covers the Living Room and Family Room, but no
contract or builder path is room-specific.

## Authority and ontology

The prior follows the existing Spatial OS ownership boundaries:

- `site_id` identifies one deployed home.
- `space_id` identifies a physical space inside that site. It is independent of
  camera identity.
- Authored room labels and OBJ groups remain the semantic authority for room
  membership. Scan geometry cannot invent room labels, doorways, or object
  classes.
- `camera_id` is only a sensor binding to an exact space revision. Multiple
  cameras may bind to the same space, and one space may receive later immutable
  scan revisions.
- All prior geometry is stored in `backend_world_m`. Camera-local floorplan
  layers are derived at response time from the current calibrated extrinsics.
- The standalone PNG is a review artifact in the source bundle's fixed-camera
  ground frame: screen-right is calibrated camera-right and image-up is
  calibrated camera-forward. It does not rotate or mirror the canonical grid.
- Live producer-owned `track.world` remains canonical. V1 is `shadow` only and
  never moves, clamps, rejects, or creates a track.

The strict product contracts are:

- `noesis.scene_prior.revision` v1: one immutable derived revision, including
  exact source/alignment/authored-scene fingerprints, semantic bindings, grid
  definition, derivation thresholds, quality, and artifact inventory.
- `noesis.scene_prior.catalog` v1: one mutable site catalog that selects exact
  revisions and binds cameras to spaces in explicit `shadow` mode.

Generated JSON Schema and TypeScript definitions live with the other Noesis
product contracts under `contracts/schema/` and `contracts/typescript/`.

## Derived evidence

The builder clips aligned scan points to the selected authored room surfaces and
derives a deterministic 2.5D grid with these layers:

- `authored_walkable`: authored semantic floor support.
- `observed`: at least one admitted scan point in the cell.
- `evidence_confidence`: bounded scan-density and floor-support confidence.
- `floor_supported`: scan evidence near the declared floor.
- `obstacle_mask`: a scan-derived static obstacle candidate, not a semantic
  object label.
- `walkable_candidate`: advisory intersection of authored floor, observation,
  floor support, and no obstacle candidate.
- `floor_height_m` and `height_agl_p95_m`: robust floor and surface height.
- `boundary_signed_distance_m` and `obstacle_signed_clearance_m`: metric
  diagnostic distance fields.
- point, floor-support, and obstacle-support counts.

The immutable revision contains `manifest.json`, `grid.npz`, `metrics.json`, a
camera-facing top-down `preview.png`, and a bounded review `points.glb`. The
manifest records the exact camera-calibration and target-revision fingerprints,
camera position, ground-plane axes, and preview bounds. Rebuilding unchanged
inputs and thresholds produces the same prior ID and bytes.

## Building any room

The input must be a cold-storage bundle using
`noesis.reference.room_scan_bundle.v1`, with a passed alignment report targeting
backend world meters. The builder verifies every source file it consumes,
including the fixed-camera calibration and target reconstruction metadata,
against both bundle inventories. It refuses failed or mismatched alignment,
camera, authored-scene, room-map, or coordinate evidence.

```bash
python3 scripts/build_scene_prior.py \
  --source-bundle <verified-room-scan-bundle> \
  --site-id <portable-site-id> \
  --space-id <portable-space-id> \
  --semantic-room '<exact authored room label>' \
  --authored-scene data/virtual_twin/releases/<release-id>/<scene>.obj \
  --room-group-map config/authored_scene_room_groups.json \
  --world-to-scene config/ply_alignment.json \
  --bind-camera <camera-id>
```

Repeat `--semantic-room` only when one physical prior intentionally spans more
than one authored room. Repeat `--bind-camera` for multiple sensors occupying
the same space. Do not broaden the semantic selection merely because scan
points leak through a doorway; the authored map owns that decision.

The output root defaults to `data/scene_priors/`:

```text
data/scene_priors/
├── catalog.json
└── revisions/
    └── <prior-id>/
        ├── manifest.json
        ├── grid.npz
        ├── metrics.json
        ├── preview.png
        └── points.glb
```

This directory is runtime data and is intentionally ignored by Git. Deployment
or backup tooling must copy the exact catalog and referenced immutable revision
directories together. The original room-scan bundle remains cold evidence and
is never modified.

## Runtime behavior

When `scene_priors.path` is configured, the native DS9.1 runtime verifies the catalog,
manifest, every artifact digest/size, and the bounded NPZ inventory before
startup. Invalid configured evidence aborts startup; there is no substitute
revision or legacy path.

For a bound camera:

- `tracks[].scene_prior` reports pass, warning, fail, unknown, or error against
  the exact backend-world prior. It includes room containment, observed-state,
  confidence, boundary distance, obstacle clearance, static height, and stable
  reason codes. It cannot alter the adjacent world fields.
- A normal fresh `floorplan_response` retains every live layer and adds static
  and composite layers in the same current `camera_local_ground_m` grid. Live
  observed cells always win; the prior fills only live unknown cells. This
  remains the static-camera comparison lane.
- A `get_floorplan` request with `scene_prior_only=true` is the read-only PCF
  presentation lane. It bypasses static capture and caches, and derives its
  full diagnostic raster family and 3D inputs directly from the immutable,
  camera-bound Scene Prior. It fails explicitly when no enabled prior is bound.
- The Depth drawer keeps its four established 3D representations: Obstacles,
  Heightfield, Point cloud, and Visible floor. It does not add source-specific
  duplicate modes. PCF is the canonical and sole reconstruction source for
  those representations and for the standard Heatmap diagnostics, textured
  floorplan, derived normals, confidence histogram, and room metrics. Opening
  the drawer loads PCF automatically. Refresh may still run a fresh static
  capture for later comparison, but oai2-fe does not admit that response into
  the displayed drawer state or allow it to replace PCF. Export metadata records
  the exact prior identity.

V1 deliberately does not infer named furniture, doorway topology, navigation
policy, occlusion correction, or tracking authority. Those require separate
reviewed contracts and evidence before promotion beyond shadow diagnostics.

## Current deployed revisions

The deployed `tanglewood-manor` catalog binds two conditioned-fusion revisions
in `shadow` mode:

- Living Room: `sceneprior_living-room_20260802T202254Z_a3a77e7dcada`,
  bound to camera and space `living-room`. At 2.5 cm resolution it contains
  150,901 source points, 128,401 authored-room selections, 62,252 authored
  cells, 39,066 observed cells (62.75%), 22,734 floor-supported cells (36.52%),
  and 9,350 obstacle cells.
- Family Room: `sceneprior_family-room_20260811T015847Z_b2e023e59271`,
  bound to camera and space `family-room`. At 2.5 cm resolution it contains
  119,554 source points, 90,521 authored-room selections, 49,762 authored
  cells, 24,960 observed cells (50.16%), 11,920 floor-supported cells (23.95%),
  and 8,588 obstacle cells.

Both revisions use the prior-conditioned MapAnything + DA3 consensus with the
DA3 trajectory as pose carrier. Phone-walk inputs remain phone-only; the
calibrated static reconstruction is the alignment authority and independent
validation target. For the Family Room, the imported target cloud had a
verified 180-degree local-X/Z convention mismatch. The alignment process
rotated only its working target-cloud copy about the calibrated camera center
and world-up axis, preserved the global camera pose, and recorded the action in
the passed alignment report.

The Living Room preview uses camera-right `(-0.973939, -0.226811)` and
camera-forward `(-0.226811, 0.973939)` in backend-world X/Z. The Family Room
preview uses camera-right `(0.929727, 0.368249)` and camera-forward
`(0.368249, -0.929727)`. These values describe review orientation, not
production tracking acceptance.

## Focused validation

```bash
python3 -m pytest -q tests/test_scene_prior.py
python3 scripts/export_noesis_core_schemas.py --check
python3 -m py_compile \
  noesis/scene_prior_builder.py \
  noesis_core/contracts/scene_prior.py \
  noesis_core/scene_prior.py \
  DS9/noesis/ds9_runtime_core.py \
  DS9/noesis/pipelines/hooks.py
(cd oai2-fe && npm run build)
```
