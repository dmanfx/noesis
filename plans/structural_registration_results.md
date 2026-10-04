# Structural registration: implementation and direct assessment

2026-10-03. Baseline/non-SV3DT, scene presentation only. **The physical
alignment correction is not complete.** The owner-reviewed intermediate is now
selected live; existing localization, BEV and camera calibration remain intact.

Implemented: generic PCF/authored structural extraction, calibrated image
evidence, gravity-locked uniform fitting, origin-independent rank, fit-only
orientation selection, independent holdouts, duplicate measurement rejection,
joint raster-ambiguity checks, strict artifact/current-binding validation and
CalibrationManager publication. Measured floor derivation uses actual lower
world/authored planes and the target's fitted scale. No room offsets or
hard-coded step heights were introduced. The
[workflow](../docs/structural_scene_registration.md) is reusable across rooms.

## Observed baseline

The window comparison has 595 distinct measured observations, no holds or
predictions. Example canonical XYZ `[1.27179747,0,3.84417282]` becomes
`[162.30689227,0,369.39859509]` under the existing mapping. Source/BEV locate
the person beside the window/perimeter; the authored model puts them inward
near a different corner. The hearth comparison has 272 measured and two
predicted observations, no holds; all 274 mapped X/Z positions overlap the
authored hearth while the source shows the person on wood flooring beside/in
front of it. The consumer applies the declared transform once. Closely timed
stationary screenshots are not an exact shared video-PTS claim. People are
observations for checking presentation, never fitting controls or survey truth.

## Structural trials

| Room | Direct result | Admission |
| --- | --- | --- |
| Living | Three wall faces produce scale 109.62723, yaw approximately -3.95 degrees. Left heldout window error is 0.19875 m; reviewed ±3-pixel ambiguity raises it to 0.21996 m, above the unchanged 0.20 m gate. | Rejected |
| Family | After correcting arbitrary-origin rank sensitivity, the opposite heldout wall misses by approximately 0.365 m. | Rejected |
| Kitchen | Supported planes are nearly coplanar and do not independently determine scale and both translations. | Underdetermined |

The Living offline builder was exercised against actual catalog/PCF, retained
rectified capture, authored model and measured supports. It rejects after six
joint uncertainty combinations with `heldout_point_residual`; it does not
write an admitted registration. At that trial, the review-only candidate
represented the measured 0.1905 m rise as 20.883987 scene units. The owner
subsequently selected those same derived floor/model bytes, as recorded below.

Doorway-trim-as-jamb trials were discarded for incorrect feature identities.
The reconstructed stone plane has disconnected hearth/door-pier support and
cannot be relabelled as the complete outer wall. Lower-floor PCF points occupy
much of the authored hearth's front ~20 cm, but its precise physical edge is
not clear enough to resize safely. The saved phone cloud was aligned to the
separate paired-static point set, not directly to admitted PCF, and was not
promoted; it is not an independent quantitative wall holdout.

Private evidence: `$MENON_REPO_ROOT/artifacts/structural-registration-20261003/`,
`window-visual-review-20261003/attempt-4/` and
`fireplace-visual-review-20261002/occupied/`. In-checkout geometry reviews:
`data/scene_priors/review/`, including
`structural-contract-smoke-20261003/contract-check.json` and
`annotation-stability.json`. These are diagnostic artifacts, not live assets.

## Follow-up structural assessment

The exact retained Family DA3 carrier and its recorded world transform were
checked before using image rays; the earlier raw-carrier gauge hypotheses are
not evidence. Correcting solid-wall/finish identities improves the central fit,
but leaves an opposite-wall residual of about 0.355 m. Read-only planar-affine
trials can reduce selected central residuals, but independent views and reviewed
jamb bands still exceed the unchanged 0.20 m gate. A bright opaque window-side
panel is approximately 0.24–0.26 m from the recessed/glass plane; the two cannot
be treated as one wall face. Affine trials were not added to the v1 uniform
similarity contract or selected live.

Living planar-affine trials also fail to resolve all independent structures.
The apparent step residual from earlier trials is conditional, not a proven
floor-height defect: detailed source crops show nosing, casing and baseboard
overlap, so the proposed pixels are not demonstrated counterparts of the
authored bare-floor vertices. They were not refitted or admitted. A locally
horizontal PCF ceiling candidate suggests a vertical scale near 109.1 scene
units per metre if its authored counterpart is correct; that semantic match
remains unproven and is not a new height authority.

Direct low-height support checks in both admitted PCFs do not establish
continuous opaque wall–floor contacts. The structural extractor fits points
above 0.25 m; its tilted-plane intersections at floor height are extrapolations.
Living has no sustained compatible vertical base strip. Family base points are
sparse, outside the observed solid-panel extent, or floor/clutter-coloured.
Those points do not justify replacing the source wall planes. No correction
was fitted to the occupied tracks, phone path, hand-placed camera, or mesh.

The independent RGB PCF tracking review is now served in Menon, with retained
occupied canonical-consumer checks and a direct live presentation check. It
uses exact current frame bindings and recorded metric mesh-build edges, not an
admitted replacement for the authored-home registration. Its cross-room joins
remain explicitly review-only. Menon commits `1164967` and `185efbd` contain
that view and its RGB/layout correction; no push was made.

## Remaining input before the owner's room measurement

More walking is not the missing evidence. The smallest targeted Living check
is three labelled architectural spans: bare window-aperture width, distance
along its wall from the rear-wall junction to one bare jamb, and perpendicular
distance from that wall to the hearth's stone face. Equivalent independently
bound structural imagery/geometry is also suitable. These distinguish authored
dimensions/feature identity from registration error; width alone cannot locate
and orient the room. No broader depth survey is required.

## Owner-selected intermediate and further iteration

The owner explicitly selected the displayed improvement despite its recorded
quality failures. The active scene is `home_owner_structural_20261003_v1`, with
Living scale 109.6272308 and Family scale 103.2656373; Kitchen is unchanged.
The foyer model height is derived from its measured 0.1905 m rise at the new
Living scale (20.8839875 scene units). The separate owner-selection contract
retains rejected evidence and publishes `structurally_qualified: false`.
Direct configured-producer readback confirms unchanged camera K/E and complete
frame bindings, exact selected matrices, and matching served model SHA.

Further Family checks used visually selected solid wall regions in multiple
retained phone views, then repeated the extraction on the exact admitted PCF
point artifact. Front white-panel residuals are small, but a three-opaque-wall
uniform fit fails independent opening controls: about 0.473/0.308 m. Reviewed
pixel changes cannot bring those errors within 0.20 m. The mismatch runs mainly
along the window wall; source/model feature identity and authored opening
placement remain unresolved. A passing wall-only trial was not installed.

A Living diagnostic excludes the disconnected stone/hearth/pier plane as an
asserted whole-wall control. Two walls plus one opening endpoint, holding out
the other endpoint, pass the central and all 25 reviewed pixel-bound checks
(maximum 0.1840 m). This is additional evidence, not yet a claim that the hearth
or entire room is correct. Observed-height comparisons also remain diagnostic,
not permission to force inferred tilted walls upright.

The numerical opening pass also depends on feature identity: the selected
static pixels are outer top-trim corners, not proven bare-aperture jambs. Their
source tangent span is approximately 1.259 m versus about 1.105 m for the
authored opening/surround at that candidate scale. Numeric admission alone
does not prove those physical counterparts, so this trial was not selected.

The owner supplied a lower-room rectangle of 15 ft 4 in (4.6736 m) by 21 ft
9 in (6.6294 m). Short endpoints are camera-wall floor trim to the foyer step
aligned with the TV wall, independently repeated trim-to-trim on the other
side. Long endpoints are window-wall trim bottom-left of the window to the
wood-floor step-up edge right of the hearth. This is not a hearth-face or
whole-floor bounding-box measurement. At the active scale the authored lower
floor's short axis spans only about 4.152–4.164 m. Its near-matching long floor
extent has different endpoints and is not a passed measurement check.
Shared floor, wall/window, Kitchen and foyer topology must be preserved by any
dimension-driven authored derivative; a stretched person/world transform is
not a correction. See `data/scene_priors/review/living-owner-dimensions-20261004.json`
and the `living_feature_geometry_audit_20261004` report/figure.

The high-detail RGB review artifact is separate from the original low-resolution
mesh and preserves identical metric room edges. Photo texture adds visible RGB
detail, not independent depth accuracy or acceptance of rejected room joins.

Two dimension-driven Living model derivatives were tested with shared vertices
preserved: pinning the TV/rear wall or the camera/front trim. Three-wall fits
retain the measured short span but miss independent window controls by about
0.36–0.40 m. Fits using two walls plus one window control change the uniform
scale enough to reduce the short span back to approximately 4.20 m and worsen
step-endpoint checks to approximately 0.95–1.10 m. The rear-pinned derivative also
moves Kitchen geometry; the front-pinned derivative changes Family and foyer
geometry. Neither derivative was selected. The repeatable tape measurement
demonstrates a dimensional disagreement, but it does not identify all shared
model vertices or justify a global piecewise stretch.

A final floor-level source review used the exact paired static keyframe and
unchanged bound rays. The rear/window junction is chair-occluded, the hearth
stone-to-wood contact is hidden by the screen/ledge, and the camera-side trim is
outside useful view or sofa-occluded. Plausible points on the visible window
baseboard produce distances of 6.31–6.97 m to the threshold, encompassing the
measured 6.6294 m without isolating its tape endpoint. Existing upper-threshold
annotations use the measured 0.1905 m plane but do not establish surveyed OBJ
endpoint identities. No fabricated floor control or additional fit was made.
See `living_bound_floor_contacts_20261004` and
`living_authored_z_derivative_trial_20261004` under
`data/scene_priors/review/`. Structural qualification remains incomplete.

The high-detail v3 geometry derivative, served with corrected v4 provenance,
preserves all 18 original v2
non-image GLB buffer views and embeds the exact saved photo atlas JPEGs without
exporter recompression. It adds only source-supported, unoffset triangles from
the retained low mesh where the high mesh has gaps. The result contains
1,097,926 triangles and 903,242 vertices in 54,899,928 bytes. Same-view Living
coverage increases from 345,668 to 439,333 pixels; original-low coverage is
471,889. Added faces are explicitly low-prior presentation evidence, often
single-view supported, not newly measured surfaces. Holes, texture seams and
coarse gap polygons remain; this is sharper and more complete, not a
photorealistic or independently physically qualified reconstruction.
The initial v3 full report failed the existing consumer's contract/frame/atlas
checks and was not activated. V4 keeps identical GLB/atlas bytes and supplies
the required full-report schema and gauge identity; all four affected
tracking-review tests pass. Consumer validity gates were not loosened.

Direct current strict Menon consumers accepted 1,099 retained occupied world
snapshots (825 window, 274 hearth); transformed position, velocity and
covariance agree exactly with the declared active matrices. Low/high PCF
consumers return identical positions and uncertainty. Entity position states
are 1,097 measured and two predicted, not inferred from lifecycle or source
quality. These files contain no held entity samples and no `support_surface`
records, so they do not prove upper-floor playback or fresh current occupancy.

No Noesis commit or push was made during the preceding review iterations.

## 2026-10-04 — Latest reviewed alignment selected by the owner

The owner requested activation and scoped commits of the rightmost/latest fits
from the fresh comparison. Living now uses the two-wall/right-opening trial at
108.07193691708451 scene units/metre; Family uses the exact admitted opaque-face
trial at 112.33409691445456. Kitchen is retained unchanged. The matching scene
is `home_owner_structural_20261004_v2`; the registration digest is
`8c40ed9e44e6c00ee0a98554c1b99c731d7db4f9d2e78e92e58ca27b76b66557` and
authored OBJ digest is
`ffc0e566f2a7a5f2ec27b294298747694722869220cfea3e5ae293df7168b017`.
The foyer elevation is derived from the measured 0.1905 m rise at that Living
scale, not a generic display lift.

Family carrier evidence now replays the exact admitted GLB wall support,
phone-only source registration, retained rasters/rays and measured plane
intersections. Authored endpoint selectors retain exact physical feature kinds.
Source, revision, metric, measured-support and raster gates pass without changes
to their thresholds. Structural qualification remains false: Family's independent
window errors remain approximately 0.473/0.308 m, and Living's outer-trim versus
aperture identity remains unresolved. Owner selection is not physical accuracy.

The existing native baseline service publishes the exact selected matrices and
matching served model, with unchanged camera intrinsics, extrinsics and world
frame bindings. No localization, depth registration or BEV correction is made.
Focused structural checks pass (38 tests plus three carrier-evidence tests);
the actual CalibrationManager also loads and publishes the selected artifact.
Live readback contains 20 empty snapshots, not new occupied accuracy evidence.
