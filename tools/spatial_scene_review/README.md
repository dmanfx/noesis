# Offline static scene review

This tool turns the verified September SpatialLM experiment into an editable
Living Room proposal layer and a strict `noesis.scene_semantics` v1 export.
It runs separately from Noesis and Menon. It has no production activation path.
The model weights used for the source experiment are CC BY-NC 4.0; this remains
a non-commercial local experiment.

The review supports 3D/floor views, measured RGB clouds (140,000 display points
by default, full cloud on demand), original versus reviewed geometry, selection,
metric center/dimension/heading edits, opening-wall association, labels,
accept/reject, manual objects, undo/redo, immutable saved change records, reload,
revision-bound draft import/download and accepted-only exports. A floor-plane
measurement tool measures display-picked endpoints; those picks are not survey
evidence.

Set `SPATIALLM_EXPERIMENT` and `SEMANTIC_REVIEW_ROOT` to the existing experiment
and a new output directory on the host. `PYTHON` must provide NumPy and Pydantic 2
(the existing isolated experiment environment does). `THREE_ROOT` identifies
the installed Three.js package; no CDN or external service is used.

```bash
"$PYTHON" tools/spatial_scene_review/build_package.py \
  --experiment-root "$SPATIALLM_EXPERIMENT" \
  --output-root "$SEMANTIC_REVIEW_ROOT" \
  --dimensions data/scene_priors/review/living-owner-dimensions-20261004.json \
  --floor-rise config/calibration/living-room-support-measurements.json \
  --window-record "$WINDOW_COHORT_RECORDING" \
  --hearth-record "$HEARTH_COHORT_RECORDING"

"$PYTHON" tools/spatial_scene_review/serve_review.py \
  --package-root "$SEMANTIC_REVIEW_ROOT" --three-root "$THREE_ROOT" --port 8873
```

Open `http://127.0.0.1:8873`. Save a draft before exporting; accepted geometry
with unresolved floor-contact/opening diagnostics cannot export. The server
admits same-origin JSON writes, a bounded request size and optimistic revision
checks. It exposes only specific assets and writes only its fixed draft/export
directories. Original proposals remain immutable; rejection retains history.
One accepted element has a review actor/reason but remains
`not_independently_qualified`. Review acceptance is not calibration, authored
room membership, a claim of metric accuracy, or permission for live tracking
constraints. Source/frame identities cannot be changed through the edit API.

The builder verifies the original deliverables hashes, reconstruction and exact
source alignment. It undoes the shared yaw and `(x,-z,y)` axis permutation only.
It does not repeat phone alignment or the model's min-point translation and
does not rescale to assumed room dimensions. Each box retains its full rotated
basis and eight verified corners. Zero-width walls remain planes.

The persisted owner tape record is **15 ft 4 in × 21 ft 9 in** for the main lower
rectangle excluding the foyer; the independently measured foyer rise is
**7.5 in**. Endpoint correspondences and numerical uncertainty remain unresolved.
Neither full-cloud bounds nor model wall spans qualify as those measured spans.
The default 0.15 m floor, 0.10 m opening-plane and 0.05 m opening-extent checks
are internal diagnostic tolerances, not physical acceptance tolerances.

The advisory consumer measures distance to accepted object ground footprints
without moving canonical people. It rejects expired, future, predicted, held,
lost, conflicting, mismatched or unbound evidence. It preserves identity
generation, continuity, observation/source evidence and `room_id:null` from the
recorded snapshots. Evaluate historical data at its recorded clock:

```bash
"$PYTHON" tools/spatial_scene_review/query_context.py \
  --map "$SEMANTIC_REVIEW_ROOT/exports/accepted-map.json" \
  --snapshot-record "$SEMANTIC_REVIEW_ROOT/evidence/window.source-record.ndjson"
"$PYTHON" -m unittest tools.spatial_scene_review.test_scene_semantics -v
```

Generated schema: `contracts/schema/scene_semantics.schema.json`. TypeScript is
in the standard `contracts/typescript/noesis-contracts.ts`. Use the existing
`scripts/export_noesis_core_schemas.py` to regenerate in a temporary directory
and compare the affected outputs. Unrelated pre-existing schema drift must not
be silently rewritten by this task.

Local files and screenshots contain private room/visitor data. They are not
uploaded, published or connected to a cloud service by this tool.
