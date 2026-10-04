# Structural world-to-scene registration

Status: implementation available, updated 2026-10-04; an explicitly **owner-selected
intermediate** is active, not structurally qualified. See the
[assessment](../plans/structural_registration_results.md).

Noesis remains the position authority. PCF world, camera calibration, depth,
tracking and BEV do not change to accommodate Menon's authored model. Fit
controls are measured structural geometry, not hand-placed camera devices,
people or phone optical centres. The solution is one proper positive uniform
similarity: yaw, horizontal translation and scene units/metre, with gravity and
the admitted lower floor determining Y. Menon's existing target-frame wire
presentation remains unchanged, including XYZ, velocity and covariance.

## Workflow for any room

1. Retain the actual calibration bundle, authored OBJ/semantic parent, room map
   and admitted PCF catalog. A source image requires its rectified capture
   manifest, calibration and pixel-space metadata, bound to that same revision.
2. Use `scripts/register_structural_presentation.py inspect` with a recipe
   selecting the current room manifest and observed spatial bounds. It extracts
   supported PCF planes and actual authored floor boundaries, faces and
   openings. A plane is not automatically a particular physical wall. The
   scale hint only conditions tolerances; it does not constrain fitted scale.
3. Review corresponding physical features in the source raster and both
   geometries. Do not substitute trim for jambs, disconnected surfaces for a
   continuous wall, or floor-slab sides/furniture for architectural controls.
   Reserve independent holdouts before fitting. Two wall directions plus
   sufficient independent offsets/contacts must determine all parameters.
4. Run `solve`. Recipe root is `noesis.structural_registration.recipe`, version
   1, with `rooms` keyed by camera ID. Room fields include `manifest_path`,
   `bounds_xz_m`, selected source/authored segment IDs and correspondences
   marked `fit`/`heldout`. Image points require `pixel`, reviewed `pixel_bounds`,
   `identity_review`, image/capture paths, raster size and verified plane or
   measured support height. The bounds must include half-pixel quantization
   and the actual wider annotation ambiguity; there is no default ±3-pixel
   assumption. Authored points select exact opening endpoints or floor
   vertices. See [the CLI](../scripts/register_structural_presentation.py) for
   exact fields. Byte provenance cannot prove physical feature identity.
5. For an upper measured level, use
   `scripts/derive_authored_floor_level.py --lower-room ...`. Lower world and
   authored heights are explicit evidence, not assumed zero. A
   `--structural-review` supplies the replayed fitted scale for a review-only
   preview; otherwise use the admitted target-frame scale, not the inherited
   global scale. Derive from the bound parent. Reinspect and solve against the
   derived OBJ; geometry IDs can change. Repeat until scale and measured level
   agree. A preview is not live admission.
6. Run `build` with the review, model, catalog, room map, support sets and
   measured floor-group mapping (camera → surface ID → authored group list).
   Every current measured level must appear at the fitted scale. Every room
   requires its own structural evidence; inherited camera-marker registrations
   and omitted-room retention are retired. An updated authored export may use
   explicit reviewed semantic rebinding without changing admitted PCF manifests.
7. Check coordinates and corresponding source/BEV/model views, including
   occupied retained cohorts. Only after acceptance explicitly select the
   matching model and registration through existing scene/runtime controls.
   The tool never edits an active catalog, release, selector or service.

```bash
python scripts/register_structural_presentation.py inspect \
  --bundle "$BUNDLE" --authored-scene "$MODEL" --recipe "$RECIPE" --output "$INSPECTION"
python scripts/register_structural_presentation.py solve \
  --bundle "$BUNDLE" --authored-scene "$MODEL" --recipe "$RECIPE" --output "$SOLVE_REVIEW"
python scripts/register_structural_presentation.py build \
  --bundle "$BUNDLE" --authored-scene "$MODEL" --review "$SOLVE_REVIEW" \
  --room-group-map "$ROOM_MAP" --scene-prior-catalog "$CATALOG" \
  --parent-authored-scene "$PARENT_MODEL" --support-surfaces "$MEASURED_SUPPORTS" \
  --measured-floor-groups "$MEASURED_GROUPS" \
  --output "$REGISTRATION"
```

Omit support arguments only when genuinely inapplicable; omitting
current measured supports fails publication rather than creating a fallback.

## Admission and cost

Fit evidence alone selects orientation. Untouched holdouts then check it.
Coplanar heldout walls, corners derived only from fitted walls, reused pixels
and aliased measurements cannot supply independent evidence. Numeric gates:
fit walls 15 cm/6 degrees, heldout walls 20 cm/8 degrees, fit points 15 cm,
heldout points 20 cm, plus overlap, rank and orientation checks. These are
rejection bounds, **not** centimetre-accuracy claims.

Build and startup check every joint centre/corner combination of reviewed
raster intervals: at most four points and 625 combinations. Larger sets fail
explicitly. This finite stress check does not certify nonlinear interior
extrema, semantic identity or independent physical accuracy. Reconstruction
bias and wrong authored dimensions still require independent observed controls.

The strict `noesis.scene_presentation.structural_registration` v1 artifact
binds exact catalog metadata, PCF geometry, model/parent, calibrated rays,
measured supports, frozen solve and uncertainty checks. Offline build
re-extracts source features. Native startup verifies immutable bytes and
replays compact math/rays without Open3D or trimesh. Bundle publication only
checks current identities; no cloud fitting, file I/O or GPU work enters the
media callback. Failed registration has no camera/person/offset fallback.

## Explicitly selected intermediate presentation

An owner may deliberately select an exact, visually reviewed intermediate
despite a recorded heldout-quality failure. This is a separate strict
`noesis.scene_presentation.owner_selected_structural_registration` v1 contract,
not a passed structural registration. Its bounded owner-selection receipt names
the exact matrices by digest, authorization and reason, and requires
`structurally_qualified: false`. The frozen rejected solve and joint raster
check remain intact. Only heldout wall/point quality failures are eligible;
rank, source identity, revision, calibrated rays, metric transform, measured
support and geometry-validity failures still reject. No automatic selection,
localization fallback or threshold change is introduced.

Native initialization supplies configured camera labels before validating the
current image-ray binding. This prevents a valid selected artifact from being
checked against an empty startup camera map.

Pass `--owner-selection "$SELECTION_RECEIPT"` to `build` only for this explicitly
authorized selection contract. It does not convert a rejected fit into a pass.

## Retained carrier evidence

Where a reviewed point comes from retained phone reconstruction, an explicit
`source_feature_evidence` record binds the exact admitted PCF artifact, source
bundle/manifest, recorded phone-only similarity, retained view arrays and RGB
rasters. Offline verification re-extracts the selected PCF wall supports and
replays measured wall planes, source height lines and calibrated image rays.
Selection bounds and support digests are evidence data, not generic floor or
camera-height assumptions. Startup checks the immutable sources and bounded
compact geometry; reconstruction libraries are offline-only dependencies.

An authored `wall_segment_endpoint` selector identifies the exact start/end of
a named architectural wall segment on its verified wall plane. It is not
silently treated as an aperture corner. Carrier points without their matching
room evidence, changed geometry, different revisions and substituted images
fail closed. Byte/ray agreement still cannot establish semantic correspondence
or independently correct physical scale.

## Updated authored model and retirement of marker alignment

The owner's corrected export is mapped by
`config/authored_scene_room_groups_20261004.json`. An
`--authored-room-rebinding` record binds the previous semantic map and reviewed
room identities; it does not replace PCF geometry, world edges or camera K/E.
Exact reviewed mesh-face and wall-vertex selectors may identify an exposed
architectural face omitted by floor-boundary extraction. Retained carrier
source proof survives model-only refits, but every new authored correspondence
and solve is verified independently.

The all-room selection is recorded in
`config/structural_scene_alignment_20261004.json`. Living passes the current
structural checks; Family and Kitchen retain explicitly accepted heldout-window
failures. The combined owner-selected presentation remains unqualified. The
derived foyer top preserves the measured 0.1905 m rise at the fitted Living
scale, without using the flattened parent floor to determine horizontal fit.

Marker builders and their runtime fallback are removed. Historical alignment
file bytes remain because admitted frame bindings fingerprint them; their
`scene_similarity` metadata is ignored and never published or used. Camera
device objects and calibrated camera rays remain diagnostic presentation only.
Menon consumes exact revision-owned room mappings without requiring a global
marker matrix; a missing or mismatched mapping fails closed.
