# WO-5A/B — PCF room reintegration and review surface evidence

Status: implemented and locally validated on DS9/Noesis.  The resulting
artifacts remain review-only.

The room join now treats an X/Z ownership envelope as a candidate for overlap,
rather than as proof that a later voxel is a duplicate.  Exact voxel keys are
still owned by the existing room.  Other candidate voxels are suppressed only
when the source and existing clouds have compatible three-dimensional points
within `0.06 m`, and both sides have valid retained RGB-D support.  A source
point with a cross-room free-space contradiction is retained with
`geometry_status=1` whenever it cannot be safely suppressed, giving review
tooling an explicit uncertain class and preserving its room/view provenance;
the `0.30 m` conflict band remains reported for bounded proximity diagnostics.
A different height or an X/Z column shared by itself therefore cannot discard a
measured complementary surface.  The old exact and column candidate reasons
remain in `suppressed_moving_overlap_evidence.npz`.

The overlap check projects points into the exact raw views named by each room's
manifest.  It uses the recorded camera pose and full 3x3 intrinsics, requires
positive Z and an in-image pixel retained by the source mask, and compares the
projected Z to `depth_z` with `0.05 m + 3%` tolerance.  It classifies supported
surface, occluded/unknown, and free-space contradiction.  Work is processed in
bounded 8192-point chunks.  A reference point with free-space contradiction is
excluded from the compatibility tree; a query point with that contradiction is
never suppressed.  For a candidate inside the X/Z envelope, uncertainty also
requires a cross-room opposite-view free-space contradiction (the reverse ray
is checked at the nearest valid existing point); opposite-room occlusion or an
out-of-image ray stays unknown.  Extension consumes the same manifest-named
views for the existing Kitchen/Family join and for Living/connector sources,
with old NPZs without `geometry_status` mapped to neutral status `0`.

The unconditional `agreement_weight_bonus=.25` was removed from the effective
weight.  Agreement remains reported as provenance, while confidence and the
raw MapAnything/DA3 reliability fields determine the voxel centroid and
weight.  Nonzero legacy bonus input is rejected so a caller cannot silently
reintroduce correlated-model double weighting.

Review meshing remains per-owner.  Poisson runs with one bounded worker, then
vertices and triangles are screened by support distance and density.  Each
triangle is additionally checked at its centroid and three edge midpoints
against the retained source camera/depth/mask views.  It must have centroid
support, at least two supported samples, and no free-space contradiction.  The
mesh report and publication descriptor carry the coordinate frame, source
frame identity, observed/uncertain/unknown counts, and explicit triangle/ray
check flags.  `WholeHomePcfOverlay` exposes this support metadata on the review
object/state; it does not use it to change canonical person or world geometry.

## Validation

Commands use the configured environment variables so the same checks can run
from another checkout:

```bash
export NOESIS_NATIVE_PYTHON="${NOESIS_NATIVE_PYTHON:-python3}"
export PHONE_SCAN_PYTHON="${PHONE_SCAN_PYTHON:-$NOESIS_NATIVE_PYTHON}"
export NODE22_BIN="${NODE22_BIN:-node}"
export MENON_REPO="${MENON_REPO:-../Menon}"

PYTHONPATH=. "$PHONE_SCAN_PYTHON" -m pytest -q tests/test_pcf_multiroom_boundaries.py
"$PHONE_SCAN_PYTHON" tools/mapanything_phone_scan/reintegrate_pcf_rooms.py --self-test
"$NODE22_BIN" --test "$MENON_REPO/tests/scene-review-assembly-service.test.mjs"
```

The focused Python boundary suite passes 9 tests.  It covers same-X/Z,
different-height retention, compatible overlap suppression, free-space and
unknown projection classifications, removal of the agreement bonus, triangle
opening rejection, support/frame metadata in the ordinary mesh builder, and
support metadata in the publication helper.  The reintegration self-test
passes.  The Menon scene-review assembly suite passes 6 tests.  The Noesis
frame-authority suite also passes 4 tests under the system interpreter and
3 passed plus 1 dependency skip under the native interpreter.

A bounded two-room run used the existing Kitchen/Family review manifest and
its exact fixed and moving raw roots.  It wrote
`wo5_sample_two_views_v7` under the configured reconstruction work root.  The
run produced 1,195,696 output voxels and remained `review_only`; 22,626 exact
shared voxels were suppressed, 15,121 non-exact voxels were suppressed by
3-D-compatible overlap after cross-room ray checks, 25,121 candidate-column
complementary voxels were retained, and 545,509 moving voxels carried
cross-view or own-view contradictory geometry status; 121,210 fixed voxels
also carried own-view contradiction status.  The report records valid RGB-D
support for 375,196 fixed and 514,687 moving voxels, plus 510,362 cross-room
free-space contradictions and 226,668 cross-room occluded/unknown samples.  No
active catalog, accepted registration, runtime service, GPU process, or
rejected three-room join was changed.

The recorded v7 surfel artifact was then passed through the ordinary review
mesh builder with `voxel_size_m=0.08`, `poisson_depth=7`, and one Poisson
worker.  It emitted
`wo5_sample_two_views_v7_mesh/multiroom_surface_mesh.glb` with 30,218 vertices
and 50,007 triangles.  The mesh report shows 20,125 triangles rejected by the
camera/depth support test and 107 rejected by geometric support; the remaining
review geometry has separate `pcf_surface_{room}_observed`,
`pcf_surface_{room}_uncertain`, and small unknown point nodes.  Source and mesh bounds were compared
under the same X/Z extent and in X/Y and Z/Y side slices.  The resulting review
figures are `same_bounds_height_slices.png` and `side_slices_xy_zy.png` in
that output directory.  The slices retain the room envelope and measured
openings while showing the uncertain geometry in its separate tinted class.

The larger Kitchen/Living/connector input remains review-only because its
registration acceptance gates have not passed.  This work does not infer a
connector transform, promote a rejected join, add a live-person geometry
constraint, or make surface meshing a measured-world authority.
