# WO-1A/B: shared metric frame and render transport

Status: implementation complete for the Noesis and Menon frame transport
surfaces. The capture/VIO endpoint producer now emits the exact accepted
registration edge when manifest-backed endpoint evidence is present; physical
metric admission still remains owned by this report's v2 binding contract.

## Contract and authority

`ScenePriorFrameBinding` remains explicitly versioned. Existing v1 bindings
retain the original full artifact hash, frame revision, rigid transform, floor
plane, and catalog checks. V2 adds these separate identities:

- `metric_frame`: accepted `backend_world_m` frame identity, in meters, with a
  floor plane and physical provenance for camera calibration and world
  alignment. Its revision is derived from the frame ID, floor plane, and the
  physical artifact hashes.
- `artifact_revision_id`: the scene-prior/map artifact revision used by the
  catalog. It may differ from the metric coordinate revision.
- `source_camera_calibration_physical_sha256` and
  `source_world_alignment_physical_sha256`: physical calibration provenance;
  authored presentation metadata is excluded from these identities.
- `presentation`: an accepted proper positive uniform-similarity mapping from
  the target metric frame to `menon_scene`, with a content digest and a bounded
  allow-list of accepted source transform hashes, plus a separate authored
  scene revision.

V2 catalog bindings match `artifact_revision_id` to the catalog prior ID and
require all V2 bindings in one catalog to carry the same accepted metric frame,
including its floor plane. A presentation-only change therefore cannot relabel
the metric frame, while a physical calibration or floor change produces a new
metric revision. Current room joins in `data/scene_priors/catalog.json` were
not changed or promoted.

## Accepted registration import

`build_accepted_room_to_home_binding` and
`import_accepted_room_to_home_binding` require an existing producer report.
The accepted report shape is intentionally strict:

```json
{
  "schema": "noesis.pcf.connector_multianchor_pose_graph.v1",
  "status": "passed",
  "accepted_for_canonical_use": true,
  "disposition": "validated_cross_session_registration",
  "pose_graph": {"accepted": true, "reason_codes": []},
  "holdout": {"source": "pose_graph", "status": "passed"},
  "bound_transform": {
    "source_frame": {"frame_id": "backend_world_m", "revision": "..."},
    "target_frame": {"frame_id": "backend_world_m", "revision": "..."},
    "target_from_source_col_major": [16],
    "target_from_source_sha256": "<sha256>",
    "source_floor_plane": {"normal": [3], "offset_m": 0.0},
    "target_floor_plane": {"normal": [3], "offset_m": 0.0}
  }
}
```

The builder compares every endpoint, matrix, digest, and floor-plane value to
the binding it constructs. It accepts only the known
`validated_cross_session_registration` disposition and a passed pose-graph
holdout; review-only, rejected, unknown-schema, missing, stale, or mismatched
reports fail closed. The multianchor producer now emits `bound_transform` only
after checking source/target manifest identities, deriving baseline revisions
from their exact bytes, and validating any nonidentity endpoint map against an
accepted registration report. Intermediate reconstruction-frame evidence does
not mint physical calibration provenance; the final v2 binding still requires
the true physical calibration and world-alignment artifacts.

The normal scene-prior CLI accepts `--accepted-frame-binding` together with
`--registration-acceptance-report` and imports that exact producer-owned v2
edge. V2 derives the source physical revision from the bundle's E matrices and
alignment transform, applies the accepted source-to-target transform to points
before grid construction, reframes the camera preview, and preserves the
target-owned presentation mapping. Missing, rejected, mismatched, or
self-labelled inputs fail closed. The older `--metric-frame` and
`--world-presentation` options are accepted only when they exactly equal the
imported binding; omitting all v2 inputs preserves the existing v1 output path.

## Runtime and renderer transport

`CalibrationManager` computes physical frame hashes independently of
presentation fields, validates v1/v2 bindings, and publishes the binding table
plus a revision-keyed `world_frame_presentations` table in the calibration
bundle. Each target-owned presentation is derived once as
`scene_from_calibration * inverse(world_from_calibration)` when an explicit v2
mapping is not supplied. The scene revision is derived from the effective
target-to-Menon mapping and is checked against any explicit presentation;
changing the authored mapping while retaining the metric frame rejects a stale
explicit presentation. A target-owned authored registration may be supplied
through `scene_similarity.target_world_to_scene_col_major` and
`scene_similarity.target_registration_sha256`; this mapping is checked against
explicit presentations independently of every source edge. Multiple cameras
may therefore contribute different source transform hashes to the same target
revision when their target-owned mapping agrees; the hashes are unioned into
that mapping's bounded allow-list.

Menon's `CalibrationManager` preserves both tables, validates each mapping's
content digest and scene revision with WebCrypto once at calibration-bundle
ingress, and exposes getters. The digest v2 canonicalization uses explicit
little-endian IEEE-754 Float64 matrix bytes plus frame/render identity, so
Python and JavaScript agree for signed zero, tiny exponents, and values such as
2^-18. `CanonicalWorldPresentation` then validates the
target revision, accepted mapping, proper uniform similarity, finite affine
values, and source-hash allow-list structurally at the render boundary. It
applies a target-owned mapping directly when a fused entity has
`world_transform_sha256: null`, and retains the existing v1 composed binding
path for entities with one source edge hash. Velocity uses the mapping's
linear component and covariance uses `A Sigma A^T`; translation is excluded.
Missing target revisions, stale/unadmitted source hashes, malformed mappings,
and missing revision provenance fail closed.

## Direct evidence and validation

The Noesis test constructs `GlobalWorldFusion` output from two accepted
sources with target revision `home-r1` and distinct edge hashes, serializes it
through `WorldSnapshot.model_dump_json()`, and restores the serialized object.
The resulting entity preserves both source hashes and deliberately has no
single entity transform hash. That serialized fixture is ingested by Menon's
`CanonicalWorldState` and rendered through `CanonicalWorldPresentation` with
the actual target-owned presentation emitted by the Noesis calibration bundle
fixture. The test verifies rendered position, velocity, covariance, null entity
hash, target revision, and presentation application; separate cases reject
missing targets, shear/reflection mappings, and an unadmitted source hash. A
Menon ingress test changes the mapping translation without changing either
claimed digest and verifies that the bundle is rejected before state mutation.

Focused checks passed:

```text
export NOESIS_NATIVE_PYTHON="${NOESIS_NATIVE_PYTHON:-python3}"
export PHONE_SCAN_PYTHON="${PHONE_SCAN_PYTHON:-$NOESIS_NATIVE_PYTHON}"
export NODE22_BIN="${NODE22_BIN:-node}"
export MENON_REPO="${MENON_REPO:-../Menon}"

"$PHONE_SCAN_PYTHON" -m pytest -q \
  tests/test_scene_prior_frame_authority.py \
  tests/test_calibration_manager.py \
  tests/test_noesis_core_world_fusion.py
57 passed (including the normal builder -> catalog -> CalibrationManager
nonidentity-edge roundtrip) in the system/phone-scan environment. The native
Noesis environment runs the frame-authority subset as 3 passed, 1 dependency
skip because it does not carry the offline trimesh builder dependency.

"$NODE22_BIN" --check "$MENON_REPO/src/services/CanonicalWorldPresentation.js"
"$NODE22_BIN" --check "$MENON_REPO/src/features/calibration/CalibrationManager.js"
"$NODE22_BIN" --test \
  "$MENON_REPO/tests/canonical-world-state.test.mjs" \
  "$MENON_REPO/tests/canonical-world-producer-fixture-presentation.test.mjs" \
  "$MENON_REPO/tests/environment-coordinate-authority.test.mjs"
28 passed

git diff --check
passed in both repositories
```

No runtime, GPU, service, active catalog, or manual extrinsic input was
changed. Real phone calibration and accepted cross-room endpoint evidence remain
dependencies for later metric calibration; the endpoint producer's graph to
WO-1 builder proof is covered by the focused capture/VIO test.
