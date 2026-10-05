# PCF-relative BEV reassessment — 2026-09-29

## Scope and result

The owner's course correction is accepted: PCF supplies the room scale. DAv2
needs a bound registration into that scale, not intrinsically correct absolute
depth or a new manual room survey. Existing surface-aware and canonical-output
work is retained. This report supersedes the acquisition prerequisites in the
[first delivery](person_ground_accuracy_results_20260928.md), not its retained results.

The non-SV3DT baseline now loads a measured, finite upper-floor patch. The
selected live SV3DT configuration is not changed. Menon, gait-aware footpoint
logic, identity changes, commits and pushes remain out of scope.

## Measured floors, without moving the map

- The owner measured **0.1905 m at both ends** and specified practically flat,
  parallel floors. The observations are stored in
  `config/calibration/living-room-support-measurements.json`.
- `scripts/build_parallel_floor_surfaces.py` preserves the lower normal and
  computes the upper plane offset. It projects the annotated visible foyer
  perimeter through the bound camera and clips it to the PCF coverage hull.
  It does not extend the rise across the rest of the room. The convex coverage
  hull is a coarse outer bound, not proof that every enclosed cell was scanned.
- `DS9/config/support_surfaces.json` binds the result to the Living Room PCF
  revision, calibration fingerprint, transform and measured-source digest.
  `DS9/config/infer.yaml` loads it; the replay renderer retains its exact path.
  There is no room-name branch or hard-coded rise in the tracking algorithm.
- The local lower-floor RMS scatter is **0.0388 m**. This is reconstruction
  scatter, not a measured bound on camera pose, annotation or absolute accuracy.
  The 4,326 covered PCF cells are not 4,326 independent physical measurements.
- For the same annotated foyer rays, using the measured upper plane instead
  of Y=0 changes horizontal intersections by **0.437–0.873 m** across the
  perimeter vertices. That is the geometric consequence of the correction,
  not a measured person-localization error improvement.

The earlier image-derived rise was 0.153285 m, **37.2 mm lower** than the tape
measurement. Reprojecting the measured rise from the annotated lower edge lands
4.53–4.97 pixels beyond the annotated upper line. The two drawn edges trace the
visible wood face, but their vertical correspondence is not established: nosing
or a hidden floor/riser junction can invalidate that construction. The image
is also a six-frame pixelwise median. Raster size, destination intrinsics and
matrix composition agree with the retained binding; no definite projection
bug was found. The dewarper's inherited source-camera comment is a provenance
question, not proof that the wrong camera was used. **The exact discrepancy is
not resolved**, and neither the map nor a floor slope was adjusted to hide it.

Review artifacts beneath
`data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/`:

- `parallel-floor-reassessment-20260929/source-pcf-floor-model.png`
- `parallel-floor-reassessment-20260929/build-review.json`
- `riser-reassessment-20260929/review.md`

## Producer repairs exposed by real raised-floor replay

1. Recompute the upright body-plane solve on each admitted horizontal support,
   rather than solving on the lower floor and changing only the output Y.
   Multiple compatible levels remain ambiguous. The measured seated torso
   reference is already an absolute calibrated height; do not add the rise
   to it a second time.
2. Retain support identity in the actual publication watermark. A measured
   level change must publish its own exact tracking/world/BEV cohort, even if
   it matures between ordinary scalar ticks. A prediction cannot relabel the
   old published point as belonging to the new floor.
3. Preserve full XYZ in inferred-motion raw origins, trusted origins, deltas,
   current process samples and proofs. Previously the emitted point was raised
   while the proof still used Y=0; the canonical service correctly rejected it.
   Plane geometry now supplies the vertical delta, including analytic sloped
   fixtures. The service's existing proof checks were not weakened.
4. Fix a variable-shadowing exception in registered-depth covariance: the
   nested projection used `height` for both image height and support elevation.
   Some otherwise usable depth candidates were consequently discarded.
   Bounded exception diagnostics now identify the failing function/line.

These changes are in `DS9/noesis/pipelines/hooks.py` and
`noesis/telemetry/person_ground_state.py`. Temporary replay tracing used to
pinpoint the two proof failures was removed from production code.

## DAv2 registration: useful information, broken old mapping

`noesis/calibration/pcf_depth_alignment.py` verifies exact camera, PCF revision,
transform, source frame/PTS and artifact binding. It fits on **person-excluded
static pixels**, then applies the room registration only to actual same-frame
MapAnything person samples. It never substitutes PCF depth behind a person.

On the six retained paired-static frames, 640 static fit pixels give a
MapAnything-to-PCF scalar of **1.079818**. Separate frame/tile holdouts improve
median optical-Z disagreement from 0.360/0.354/0.324 m to
**0.103/0.096/0.092 m**. This demonstrates consistency with the admitted PCF,
not independent physical accuracy.

The selected native TensorRT DAv2 engine was run on all six exact source frames.
Raw and rectified frame hashes match the retained MapAnything manifest. The
diagnostic follows native input geometry/normalization, but does not claim
byte-exact NVMM interpolation. Four visually identified torso pixels have valid
same-frame MapAnything values; two other visible torso pixels are invalid and
remain invalid. The four valid raw/registered-target pairs are:

| Raw DAv2 | PCF-scaled MapAnything person optical Z (m) |
| ---: | ---: |
| 8.633 | 3.739 |
| 10.281 | 4.496 |
| 12.227 | 5.955 |
| 16.234 | 8.021 |

That small sample has a strong monotonic relationship (r=0.997, n=4). A positive
affine fit on the **separate static training pixels**, not the people, gives
`Z = 0.521526 × raw − 0.948111 m`. On 796 held-out static pixels its median/p95
disagreement is **0.181/0.563 m**; on the four person pixels it is
**0.344/0.523 m**. Those are diagnostic errors against reconstruction, not
ground-truth person errors. Raw depths 13–17 have no static training pixels;
the farthest person lies in that gap and remains unqualified.

The existing v3 mapping has raw domain about 8.385–11.797 and is flat near
3.591 m for most of it. Its old occupied fit pairs have little useful raw/target
correlation, use bounding-box ROI sampling, and lack explicit admission to the
current PCF revision. Isotonic pooling alone cannot manufacture information
missing from those pairs. The new exact-frame check supports a registration
repair, not abandoning DAv2 or demanding intrinsic metric scale.

The historical fields named `*_range_m` mean camera-forward optical Z in this
contract, not Euclidean ray length. Both old and new fits intend the same depth
axis. The older service exposes MapAnything `depth_z`; the paired-static
reference converts `depth_along_ray` using captured-calibrated rays. Capture
era, normalization, ray calibration and revision binding still differ, so the
new numerical curve cannot replace the old one without occupied validation.

**No replacement mapping was installed.** One recording and four manual pixels
do not satisfy the existing occupied-support/capture-disjoint/domain gates.
The practical next step is separate, revision-matched occupied capture cohorts
with exact DAv2/MapAnything frame association and actual person silhouette or
pose-capsule support. This needs ordinary occupied camera recordings, not a
board moved around the room or a broad physical survey. Reuse only captures
whose camera binding can be established; splitting this one video into four
windows does not create four independent captures. Physical measurements can
then check a particular residual rather than block this path.

Depth artifacts are under `path_review/depth-reassessment-20260929/`:
`pcf-static-scale-review-verified.json`, `six-frame-dav2/diagnostic.json`,
`da2-pcf-static-fit-v4.json`, `da2-pcf-static-fit-v4.png`, and
`da2-pcf-static-assessment.md`. All fitted scalars/curves there are review-only.

## Direct replay review

The corrected-code captures and source/BEV comparisons are recorded under
`path_review/parallel-floor-reassessment-20260929/`. All three native baseline
runs exited cleanly. Their `run.json` files record the actual source-code and
calibration hashes; the Git revision alone does not describe this dirty tree.
Before means the first delivery's `after-paired-v2`, `after-multi`, or
`after-july`, not the much older retained original Noesis run.

| Capture | Valid world → BEV heads, before | Valid world → BEV heads, corrected | Corrected fresh / predicted / held | No valid world |
| --- | ---: | ---: | ---: | ---: |
| Paired walk plus occupied Kitchen/Family Room sources | 3,090 → 3,090 | 3,207 → 3,207 | 2,497 / 471 / 239 | 1,604 |
| Multi-person room clips | 1,506 → 1,505 | 1,383 → 1,382 | 1,144 / 158 / 81 | 850 |
| July walking clips | 868 → 832 | 1,046 → 993 | 562 / 299 / 185 | 211 |

These are **publication observations**, not unique people or complete video
frames. A fresh row means a current accepted localization candidate, not
independently verified physical accuracy. Replay publication cadence differs;
raw count changes alone are not an improvement rate. Multi-person before and
after each end with one captured tracking cohort whose companion BEV was not
captured at the time boundary. There are no unexplained missing heads inside
their exact joined cohorts. The paired and July captures each have one extra
BEV-only cohort, also excluded from exact joined-person analysis. Runtime
shutdown reports admitted/completed publication counts equal in all three runs.

All emitted heads match the canonical XYZ-to-PCF display transform exactly
(reported maximum discrepancy 0). **All three corrected captures have zero
`world_estimation_error` observations.** None of this means every detection has
a valid localization: weak cold starts, occlusion, physical continuity gates,
unsupported surface transitions and depth-domain rejection still occur.

In the paired Living Room, the named upper surface has **427 valid outputs**:
270 fresh measurements, 121 predictions and 36 holds. They preserve Y=0.1905
through canonical output; four additional rows retain the support reference
but have no valid world point and are not counted as displayed measurements.
An observed upper-to-lower departure holds briefly, rejects two unmeasured
transition rows, then resumes on a fresh lower-floor measurement. Predictions
did not acquire authority to choose a new floor.

The canonical journal still has six image-motion and five bounded-process
provenance rejections in the full paired run. Four exact baseline matches
replace earlier held/predicted outputs, so these must not be counted as an
unqualified continuity win. They remain unresolved boundary cases; the direct
evidence did not isolate another safe producer correction. The strict gate is
retained. See the
[upper-surface audit](../../../data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/parallel-floor-reassessment-20260929/paired-corrected-review/upper-surface-audit.md).

The July Living Room has **53** valid-world rows excluded by the unchanged
`canonical_world_outside_display_guard`, versus 36 before. Of the 53, 30 are
image-motion predictions, 13 holds, one CV prediction and nine fresh rows.
Their overshoot beyond the display bounds has median 0.523 m, maximum 0.718 m.
This is a visible remaining limitation, not hidden by expanding the guard or
clamping dots to the map. Correct input raster does not prove that an older
recording shares the current physical camera pose and scene binding.

One-to-one source-PTS/bounding-box comparisons use at most 100 ms separation
and IoU ≥0.5, never StableID alone. They include 1,405 Living Room matches in
the paired walk: 1,262 valid both times, 35 valid only after, 20 only before and
88 invalid both times. In the multi-person Kitchen there are 49 after-only
versus two before-only valid matches; July Living Room has nine versus 12.
Thus there is **no supported claim of universal coverage improvement**.
Exact paired-walk lower/upper examples below show the intended geometric
change more clearly than these asynchronous aggregate counts.

### Source-frame / BEV visual checks

All links below open a source image next to its own **captured producer BEV**,
with the actual image foot/base anchor, trail and localization state. The
source PTS is in `samples.json`; a requested review time may select the nearest
occupied cohort and must not be mistaken for its actual timestamp.

1. **Paired walk, about 60 s:** adult beside the TV/rug stays on the lower
   floor at approximately (3.64, 0, 3.93). The child in the foyer is explicitly
   a prediction, not fresh depth. Its point changes from approximately
   (7.95, 0, 8.32) to (7.84, 0.1905, 7.72), about 0.61 m horizontally toward
   the camera. The source frame is identical; the child remains in the foyer
   and the adult's position does not acquire a room-wide offset.
   [Before](../../../data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/parallel-floor-reassessment-20260929/paired-before-review/living-room-1792-504.jpg)
   / [corrected](../../../data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/parallel-floor-reassessment-20260929/paired-corrected-review/living-room-1792-459.jpg).
2. **Paired walk, about 105 s:** the adult is just beyond the step. The fresh
   floor-ray point changes from approximately (7.37, 0, 5.51) to
   (7.25, 0.1905, 4.97), moving about 0.55 m toward the visible threshold.
   This is consistent with the measured floor correction; it is not a
   surveyed 55 cm error reduction. The source PTS differs by about 67 ms.
   [Before](../../../data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/parallel-floor-reassessment-20260929/paired-before-review/living-room-3139-955.jpg)
   / [corrected](../../../data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/parallel-floor-reassessment-20260929/paired-corrected-review/living-room-3141-912.jpg).
3. **Multi-person Kitchen, exactly 42 s:** the foreground adult beside the
   island/table now has a fresh floor-ray point in that aisle; the older run
   had no valid world point there. The far, heavily occluded person still has
   no valid dot in either run. This is not two-person coverage success.
   [Before](../../../data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/parallel-floor-reassessment-20260929/multi-before-review/kitchen-1260-385.jpg)
   / [corrected](../../../data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/parallel-floor-reassessment-20260929/multi-corrected-review/kitchen-1260-381.jpg).
4. **July Living Room, about 20 s:** the adult is between the sectional and
   coffee table; the fresh gravity-derived point is in that part of the room
   before and after, with little change (about 5 cm). The trail remains
   visibly irregular, so this is not a claim that gait/occlusion is solved.
   [Before](../../../data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/parallel-floor-reassessment-20260929/july-before-review/living-room-599-199.jpg)
   / [corrected](../../../data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/parallel-floor-reassessment-20260929/july-corrected-review/living-room-598-199.jpg).
5. **July Family Room, 68 s:** the fresh floor-ray point is beside the table
   on the couch-facing walkway; the source anchor lies between the visible
   legs. The Kitchen at 67.958 s has a fresh person-anchor floor point by the
   far passage. Neither example supplies surveyed foot coordinates.
   [Family Room](../../../data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/parallel-floor-reassessment-20260929/july-corrected-review/family-room-2039-299.jpg)
   / [Kitchen](../../../data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/parallel-floor-reassessment-20260929/july-corrected-review/kitchen-2035-317.jpg).
6. **Remaining visual mismatch:** in the older multi-person Living Room clip
   at 8 s, the person is beside the TV but the fresh gravity-derived dot lies
   beyond the corresponding PCF wall. The Family Room at about 100 s shows a
   long jump toward the kitchen doorway. Older scene/framing differences and
   incomplete occupied depth qualification prevent declaring those accurate.
   These examples are retained, not corrected with a room offset.
   [Living Room mismatch](../../../data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/parallel-floor-reassessment-20260929/multi-corrected-review/living-room-240-70.jpg)
   / [Family Room transition](../../../data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/parallel-floor-reassessment-20260929/multi-corrected-review/family-room-2999-511.jpg).

The `*-corrected-review/analysis.json` files contain full counts and exact
world/BEV parity; `cohort_metrics.ndjson` retains per-person states. The
`*-corrected-comparison/comparison.json` files retain individual before/after
image matches. There are 47 corrected source/BEV stills across the three sets.
Intermediate `paired-final` and `paired-proof-probe*` folders are diagnostic
runs superseded by **`paired-corrected`**, not final result inputs.

## Validation, remaining limits and next work

- **281 focused tests passed** across support geometry/producer proofs,
  PCF registration, replay config, PersonGroundState, resolver hooks and BEV.
  Offline helpers use the existing host Python environment; native producer
  replays use the selected DS9.1 environment and existing engines. No model
  packages or engines were replaced. Changed-file syntax, the AGENTS/docs
  consistency checker and `git diff --check` pass. Captured source/artifact
  hashes match the final checkout for all three corrected runs.
- The measured foyer patch and canonical propagation are implemented in the
  baseline checkout and exercised on occupied video. This is not a change of
  the selected live SV3DT configuration. The original managed service and
  supervising target were restored active after every replay.
- Depth work is **partial**, not complete: the fitter and revision-bound
  registration review are delivered, but the active flat mapping remains.
  Next use distinct revision-matched occupied recordings, exact per-frame
  model pairs and real person-support masks across the unsupported ranges.
  The existing capture/foreground/domain gates still decide eligibility.
- Local measured height is established; finite-boundary annotation precision,
  the old 37.2 mm riser discrepancy, historical camera binding and absolute
  person localization accuracy are not independently resolved. Targeted
  physical checks can investigate those specific residuals if needed.
- Gait/body-footprint work remains deferred. Menon and SV3DT implementation,
  commits and pushes were not performed. Concurrent unrelated files remain
  untouched. This is a BEV review delivery, not a claim of impeccable tracking.
