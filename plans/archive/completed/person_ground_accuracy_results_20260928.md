# Baseline world-position and BEV results — 2026-09-28

Historical first delivery. The owner's subsequent 0.1905 m measurement and
PCF-relative depth course correction supersede this report's earlier statement
that no upper-floor artifact can be installed. The source results below remain
unchanged evidence of that earlier run; current scope/status is in the
[implementation work orders](../../person_ground_accuracy_implementation.md) and the
[PCF-relative reassessment results](person_ground_pcf_reassessment_results_20260929.md).

## Assessment

The feasible implementation and direct replay work is delivered in the checkout.
**The requested physical-accuracy outcome is not yet established.** No new live
floor surface or metric depth calibration was enabled: the retained evidence
does not justify their geometry. The code now preserves measured surfaces and
uncertainty correctly when supplied, and a confirmed depth-fitting defect is
fixed. These are correctness improvements, not evidence that every current dot
has become more accurate.

All task inference used native DS9.1 **baseline/non-SV3DT** mode. The existing
selected runtime was restored afterward. No Menon review/implementation, native
SV3DT implementation, gait/body-cylinder work, ID repair, commit or push occurred.
Unrelated untracked workspace files were preserved.

The [seven detailed work orders](../../person_ground_accuracy_implementation.md)
record evidence, dependencies, contracts, direct checks and completion criteria.

## What changed

1. **Measured support geometry:** `SupportSurfaceSet` binds finite convex plane
   patches to camera/calibration/world/transform revisions and evidence. A
   review-only reconstruction cannot silently become live localization. No
   room-specific offset, nominal six-inch step or phone carrying height is used.
2. **Baseline localization:** calibrated rays intersect the actual bounded
   planes. Ambiguous hits remain separate. The resolver and global fusion do
   not combine incompatible support identities or extrapolate a fused point
   beyond measured extent. Covariance follows the plane slope and retains
   residual uncertainty; it does not pretend local fit scatter bounds camera
   calibration error.
3. **Canonical continuation:** selected plane, full XYZ and provenance survive
   PersonGroundState and world-service proof checks. Predictions cannot create
   a new surface; accepted support changes break the trail. Current covariance
   is absent on holds/predictions rather than copied from a past measurement.
4. **Depth fitting:** support-weighted pool-adjacent-violators replaces a
   cumulative maximum. Previously a high noisy bin could raise every farther
   knot. Genuine flat intervals remain flat and runtime-ineligible; neither
   the supported domain nor metric-evidence gates were widened. Active depth
   artifacts were left unchanged.
5. **BEV:** full XYZ and 3×3 covariance are transformed before dropping height.
   Normal footpoints carry `worldElevationM`, `covarianceXZ` and optional finite
   `supportSurface`; diagnostics use the same transform. No browser estimator,
   room correction or extra display smoothing was introduced.
6. **Direct review:** bounded native capture and exact-cohort review utilities
   retain source frames, image anchors, world/BEV heads, trails and measured/
   predicted/held labels. The capture helper rejects a source raster that does
   not match its calibrated dewarper before stopping the live service.

Main implementation files: `DS9/noesis/pipelines/hooks.py`,
`noesis/calibration/{manager,depth_registration_builder}.py`,
`noesis/telemetry/{person_ground_state,bev}.py`,
`noesis_core/contracts/support_surface.py`,
`noesis_core/world/{support_surfaces,resolver,fusion}.py`,
`noesis_core/world_service.py`, and affected schemas/contracts.

## Geometry: a real local step measurement, not a complete floor survey

The visible riser can be measured independently of people: lower-edge rays
intersect the calibrated lower floor; gravity lines from those base points
intersect the upper image edge. The five edge samples give a median **0.153285 m**
and span **0.14997–0.15675 m**. Perturbing one endpoint coordinate at a time by
1.5 pixels gives median sensitivity **0.14721–0.15934 m**, not a confidence
interval. Nosing and common camera/calibration errors are not bounded.

Projecting the same top edge onto the wrong lower plane moves it **0.341–0.363 m
in X/Z**. This supports retaining multi-surface localization as a meaningful
correction. It is not a measured person accuracy gain.

The edge cannot determine the upper plane's perpendicular slope or full extent.
The six static learned-depth views give incompatible lower-floor tilts
(0.078–0.146 m/m), foreground contamination and inconsistent cross-level
separation. Phone views have almost no independently supported upper-floor
coverage. Inventing a flat plane at the measured edge height would still be an
assumption, so no such artifact was installed.

Evidence: [surface review](../../../data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/surface-measurements-20260928/review.md),
`riser-edges.json`, `riser-measurement-v2.json`; reproducible calculation in
`scripts/measure_image_riser.py`.

## Depth: inference availability is not metric registration availability

The active Living Room mapping has raw camera-Z domain approximately
8.385–11.797 m and registered knots beginning 3.252 m followed by a roughly
3.591 m plateau. Its review fails independent-label and observable-domain
requirements. Historical fields named `*_range_m` in this fitting evidence
actually hold optical camera Z, not Euclidean ray length; the contract notes
now make this explicit.

In the paired after capture, 2,127 person observations carried raw depth, while
only 22 explicitly carried `depth_registration_status=ok`. Missing status does
not itself mean inference failed: the posture/anchor/localization path may not
consume registered depth. Nine explicit plateau rejections and eleven
out-of-domain/invalid rejections remain. Increasing DAv2 cadence cannot repair
the metric meaning of those values.

The fitter fix is directly tested. A replacement mapping needs independently
measured synchronized camera-Z at the actual occupied anchor, spanning the
intended near/far and seated range with separate holdout observations. Empty
floor behind a person, another Noesis estimate, or the phone optical-center
trajectory cannot substitute for that evidence.

Evidence: `depth-registration-review.json` in the replay artifact directory.

## Direct changed-producer + BEV checks

Artifact root (all original inputs preserved):
`data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/world-bev-implementation-20260928/`.

Counts below are **person observations**, not unique people. Valid-world means
an eligible finite `track.world` in that exact publication, not independent
physical truth. BEV heads are joined by camera/source/epoch/frame/time/publication
sequence and tracker lifecycle; no last-seen join is used.

| Complete capture | Person observations | Fresh / predicted / held | No valid world | Valid world → BEV on matched cohorts |
| --- | ---: | ---: | ---: | ---: |
| `after-paired-v2` | 4,443 | 2,493 / 441 / 156 | 1,353 | 3,090 → 3,090 |
| `after-multi` | 2,553 | 1,304 / 141 / 61 | 1,047 | 1,505 → 1,505 |
| `after-july` | 1,034 | 521 / 223 / 124 | 166 | 868 → 832 |

The multi-person capture ends with one additional valid tracking observation
whose BEV lies beyond the capture boundary, excluded from the matched count.
The paired capture has one boundary tracking cohort with no valid world; July
has one boundary BEV without its tracking packet. These are recorder boundaries,
not evidence of canonical publication drops. All native runs exited cleanly
and restored the selected service.

For all **5,427 emitted matched heads**, the canonical point transformed into
the captured PCF display basis matches published BEV X/Z exactly at the stored
numeric precision (checked tolerance 1e-6 m). This checks coordinate transport,
not whether the original world point is physically right. The named raised/
sloped-surface path is exercised in focused producer/consumer checks, not by
pretending these unchanged single-floor recordings contain admitted surfaces.

The canonical `world_snapshot` consumer was checked separately: all **5,463
valid-world observations in matched cohorts** have exact current source evidence
(paired 3,090; multi-person 1,505; July 868). No position, frame, revision or
transform mismatch was found. This joins current observation ID, camera,
source/tracker IDs and observation time, not the global entity's last aggregate
position or StableID alone. Source-fusion acceptance and current metric-measurement
acceptance have different semantics and were not conflated.

All 36 July omissions are Living Room `canonical_world_outside_display_guard`:
9 measured points, 18 image-motion predictions, 8 holds and 1 CV prediction.
Their coordinates exceed the existing display rectangle by a median 0.523 m,
maximum 0.596 m. There is no nonfinite, behind-camera or revision mismatch among
these cases. The display guard was not expanded and these points were not
clamped into the room. Exact records and bounds are saved in
`after-july-guard-analysis-20260928/analysis.json`.

### Visual findings and replay limitations

- Corrected July Family Room boxes now align with the person in the source
  image. The original video is 2688×1512 whereas its selected calibrated source
  raster is 1280×720. The first attempt is invalid for Family Room placement.
  The final run uses a full-FoV 1280×720 derivative preserving all 2,562 frames,
  not a crop or world correction. LR/Kitchen use their correct 1920×1080 raster.
- Paired source frames show the adult by the Living Room window and child at
  the threshold with corresponding heads/trails. A Kitchen example has one
  located adult and a heavily occluded background person with no valid world
  point. Universal coverage is therefore **not achieved**.
- Historical Living Room views have visibly different background framing from
  the September paired reference. A saved review-only feature-match diagnostic
  finds median unregistered inlier displacement around 65 px (May multi-person)
  and 57 px (July), in the same 920-pixel-wide review panel. Door/window/wall
  correspondences and their images were visually inspected. A homography fits
  local image relationships but does not prove a metric camera pose or identify
  whether mount, lens or framing changed. **It is not applied to any position.**
  These old clips are valuable robustness/transport checks but cannot establish
  current-calibration absolute accuracy without their own measured registration.
- Source anchors still vary with gait and occlusion. That body-footprint idea
  remains deferred as requested; no new gait filter was hidden in this work.

Use complete reviews `after-paired-v2-review-20260928/`,
`after-multi-final-review-20260928/`, `after-july-review-20260928/` and
`after-july-display-guard-review-20260928/`. Earlier partial multi-person and
wrong-raster July reviews are retained as rejected attempts, not final evidence.
Each complete review has `analysis.json`, exact-cohort rows, `samples.json` and
source/BEV JPGs. `background-framing.json` and matching JPGs retain the historical
framing diagnostic and its explicit non-authority caveat.

Representative visual checks:

- [Paired Living Room: adult and child](../../../data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/world-bev-implementation-20260928/after-paired-v2-review-20260928/living-room-1123-232.jpg).
- [Multi-person Kitchen: occupied scene](../../../data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/world-bev-implementation-20260928/after-multi-final-review-20260928/kitchen-1384-427.jpg).
- [July Family Room: corrected source raster](../../../data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/world-bev-implementation-20260928/after-july-review-20260928/family-room-1879-246.jpg).
- [July Living Room: out-of-display-guard example](../../../data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/world-bev-implementation-20260928/after-july-display-guard-review-20260928/living-room-174-47.jpg).

### Before/after interpretation

`before-paired-v2` ran the starting baseline producer; `after-paired-v2` reran
the paired video after changes. The before recorder retained tracking/world
but did not request BEV. Thus this is a matched **producer position comparison**
plus direct after-producer/BEV checks, not a claimed before/after published-BEV
accuracy benchmark. Source PTS and image boxes, not StableID equality, associate
observations. See the saved paired comparison for pairing limits and numbers.
Neither Noesis version is treated as truth. No supported real surface or depth
mapping was replaced, so ordinary position differences must not be advertised
as an accuracy improvement from the new surface machinery.

The reproducible comparison is
`paired-before-after-comparison-20260928/comparison.json`, generated by
`scripts/compare_world_bev_replays.py`. It verifies identical per-source video
URIs, uses one-to-one PTS pairing within 100 ms and one-to-one boxes with IoU
at least 0.5; neither video cohort is reused. Both-valid pairs have the following
**before/after output difference**, not error against truth:

| Camera | Both-valid pairs | X/Z displacement median / p95 |
| --- | ---: | ---: |
| Living Room | 1,192 | 0.0168 / 0.2142 m |
| Kitchen | 786 | 0.0040 / 0.4909 m |
| Family Room | 384 | 0.0055 / 0.1044 m |

Of 3,377 matched boxes, 988 have no valid world in either run, 23 have one only
before and 4 only after. Pairing permits nonzero time offsets (median 0–33 ms,
maximum 100 ms depending on camera), so this is not a frame-identical deterministic
replay or proof of causal improvement/regression. The output does not support
claiming improved localization coverage from these changes.

## Focused verification and remaining work

Changed-file syntax and focused surface, calibration binding, resolver/fusion,
world-service proof, BEV and depth-fit checks pass. Domain, flatness, incompatible
surface, finite extent, full XYZ/covariance and hold/prediction tests target the
changed behavior. The analytic riser utility reproduces a known 0.2 m fixture.
An agent also ran the existing broader world-service file: 277 passed and two
timing-budget tests failed during concurrent GPU replay. Those performance
tests were not rewritten or repeatedly rerun; this is not a full-suite pass or
throughput qualification. The final scoped docs consistency and whitespace
checks pass. Unrelated pre-existing ScenePrior schema export drift was left
outside this change; the three changed schemas and named TypeScript definitions
are checked against their current models.

To complete the physical outcome safely, the missing evidence is specific:

1. Independently metric upper-floor points distributed in two dimensions, with
   observed boundary coverage and another view/held-out check, registered to
   the static camera's existing frame. The measured riser supplies one useful
   boundary, not this entire plane.
2. Synchronized independent metric measurements at occupied torso/ankle depth
   anchors over the necessary ranges/postures. These qualify the repaired
   fitter without circularly using Noesis or the phone path as labels.
3. For absolute tests on May/July recordings, independent historical camera
   registration; otherwise compare robustness and transport only. Then repeat
   the same concise source-frame/BEV checks with admitted geometry/depth.

These are measurement limits, not a request for more unit-test infrastructure.
No claim is made that tracking is now universally accurate or always visible.
