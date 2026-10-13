# Universal person world localization

Status: current canonical DS9.1 baseline, 2026-10-04.

## Purpose

Noesis localizes each tracked person once as a revision-bound
`ground_footprint` in `backend_world_m`.  Every downstream spatial view—strict
world observations, BEV, dashboard trails, OSD reprojection, and Menon
3D rendering—consumes that same filtered point and may only apply an explicit
view transform.

The localization strategy is universal.  Camera calibration and depth
registration remain camera-specific evidence, but camera names, room names,
sites, and homes never select different estimator algorithms.

## Authoritative flow

```text
detector/tracker lifecycle + pose + compact object depth
                         |
                         v
        current-frame geometric hypotheses (max 4)
          floor_ray | registered_depth | pose_scale |
                    gravity_reconstruction
                         |
             UniversalWorldMeasurementResolver
       evidence score + covariance + compatibility + PCF
                         |
             one current ground measurement
                         |
                 PersonGroundState
       physical gate + CV filter + lock + reacquisition
                         |
       track.world + covariance + exact revision/provenance
                         |
       canonical world service / BEV / OSD / Menon views
```

`PersonGroundState` remains the only temporal person-ground filter.  The
universal resolver is the measurement side feeding it; it is not a second
tracker, smoother, lifecycle manager, or identity system.

## Measurement contract

`noesis_core.contracts.world_measurement` defines an immutable exact-cohort
contract.  One `WorldMeasurementSet` contains at most four current metric
hypotheses.  Every hypothesis has the same:

- camera, source, tracker, tracker-lifecycle generation, frame, and observation
  time;
- active `backend_world_m` revision;
- the exact source-to-world transform SHA-256;
- camera-calibration revision;
- optional PCF revision.

Each hypothesis carries:

- a finite 3D position and full row-major 3x3 PSD covariance;
- typed kind and anatomical/image anchor;
- posture, support state, occlusion, motion and support evidence;
- ray-incidence or depth-support evidence when applicable;
- optional revision-matched PCF evidence and a bounded rejection reason.
- an optional typed `support_surface`, preserving surface ID/revision, plane
  normal/offset and finite convex XZ extent in the same metric world frame.

Prediction and hold are process continuations, not measurements.  They cannot
be placed in the hypothesis list or presented as fresh geometric evidence.

The current authoritative quantity is `ground_footprint`.  `body_root_3d` is a
future separate quantity; an ankle-floor contact and a seated pelvis must never
silently share one field.

## Candidate construction

The DS9 analytics hook constructs each candidate independently:

- `floor_ray`: the current accepted lower-body/person contact pixel intersected
  with the active revision's calibrated floor plane.  Near-horizon and
  out-of-range geometry is rejected before resolver entry.
- `registered_depth`: the exact current object-depth anchor UV projected with
  that camera's validated DAv2-to-room optical-Z mapping, then represented as a
  ground footprint at the active floor elevation.  Pose UV is never substituted
  for the depth sample's own anchor. The object-geometry and GPU depth-tensor
  frame/PTS must both match; a current bbox sampled against a lagged tensor is
  retained as diagnostics rather than mislabeled as current evidence.
  Independent ankle, orientation-independent torso, and partial-shoulder patches
  retain their own UV/range/support. Missing feet or hips do not invalidate a
  qualified shoulder observation. The measured body point is gravity-projected
  to the selected support surface; it is not an observed covered-foot contact
  or an inferred hip. If only shoulders are visible and posture remains unknown,
  qualified exact shoulder depth can establish this distinct body footprint
  when no verified image floor contact exists. Cold shoulder admission requires
  the existing three-current-sample consensus and detector semantic gate;
  no posture, population height, or old standing position is fabricated.
- `pose_scale`: an exact-current, typed body-to-floor projection for people
  whose physical floor contact is hidden. A standing solve intersects calibrated
  nose, shoulder, and hip rays with their anatomical height planes and solves
  the common height/footprint without using detector-box bottom. Five retained
  planes spanning all three height bands are required for one-frame strong
  proof. One three- or four-plane row never establishes canonical world state.
  As the narrow cold-start exception, exactly two compatible current
  four-plane solves may establish the first coordinate on the second solve
  when they remain on the same lifecycle/body-plane basis, arrive within
  `reacquire_max_gap_s`, and pass the ordinary physical-step bound. The first
  solve may survive intervening no-measurement callbacks, but its timestamp is
  never renewed. Three-plane, mixed, incompatible, expired, or basis-changing
  evidence cannot form this proof. For an established relocation, moderate
  rows may accumulate same-basis trajectory evidence, but only a current
  verified five-plane row may finalize the reanchor. A seated solve projects
  the observed torso to the support footprint only after the same tracker
  lifecycle has measured that exact torso anchor from registered depth. It
  never derives the plane from the prior standing footprint, uses a
  population-average body height, or relabels the pelvis itself as the ground
  point. Exact-current
  registered torso depth supplies the corresponding typed body footprint for
  sitting (`seat`) or lying (`couch`) posture; lying has no pose-only plane.
- `gravity_reconstruction`: a deliberately weak upright-height reconstruction
  used only when the existing PersonGroundState evidence permits it.  It is not
  a seated/lying floor contact.

A failure in one source does not suppress an independently valid source.
Cached depth is not current evidence. Bbox-only, untyped torso-range,
contaminated, or unsupported samples remain diagnostic rather than becoming
authoritative floor contacts. When an exact upright body-plane candidate is
available, the resolver does not also add the learned-height gravity hypothesis
from the same current body; correlated body estimates cannot masquerade as
independent fusion evidence.

An inferred-height `gravity_reconstruction` also cannot become a contributor
to measured range or observed support merely because both use `unknown`
support and agree numerically. It remains an explained alternate. This keeps
a qualified partial-body range from acquiring a diagnostic-only contributor
and subsequently failing the all-contributor canonical admission gate.

Shoulder depth cannot seed a torso-height reference. A registered torso sample
may seed one only at the exact corresponding torso UV. Shoulder and torso
depth have distinct anatomical continuity bases: switching between them cannot
silently blend a different body point into the accepted trajectory. Without
independent coherent image motion, the existing valid coordinate remains held;
an accepted same-basis reacquisition is explicit and breaks the trail. Cached
rows never increase that consensus or renew last-good measurement time.

## Measured finite floor surfaces

`support_surfaces.path` loads exact camera/calibration/world/transform-bound
`measured_calibration` sets. `review_only` sets cannot place people. The current
baseline selects one Living Room upper patch derived from the owner's measured
0.1905 m rise at both ends, the existing lower-plane normal, calibrated visible
perimeter and PCF coverage. The rise is room-specific evidence; generic
localization contains neither that constant nor a floor chosen by room name.
See [the measurement builder](../scripts/build_parallel_floor_surfaces.py).

Floor rays must have one unique bounded intersection. Upright body-plane solves
are recalculated on each eligible level before selecting one; solving at the
lower level and changing Y afterward would corrupt X/Z. A measured seated torso
height is already absolute and must not receive the rise again. Fusion never
averages different or unknown support identities into a named surface. Plane
projection retains finite extent and propagates covariance conservatively.

Accepted measured support changes reset motion and break trails. A continuation
keeps the committed surface and cannot establish or cross into another level.
Outside the measured patch, the original calibrated lower-plane path remains.
Local PCF residual scatter is not an absolute camera/annotation uncertainty
bound. The earlier image-derived 0.153 m rise remains a recorded unresolved
edge-correspondence discrepancy, not a competing runtime floor.

## Depth registration boundary

DAv2's sampled scalar is camera-forward optical Z, not distance along a unit
ray. Some historical field names contain `range`; projection must retain the
declared optical-Z meaning. Registration needs trustworthy alignment to the
admitted PCF scale and frame, not intrinsically perfect DAv2 metric scale.

The fitter uses support-weighted isotonic pooling rather than cumulative maxima.
Flat/dead intervals, unsupported domains and stale or mismatched frames still
reject. The offline `noesis.calibration.pcf_depth_alignment` review fits scale
on person-excluded stationary PCF correspondences with disjoint frame/tile
holdouts. Applying that scale to an occupied sample requires actual same-frame
MapAnything person support at the identical DAv2 anchor. Empty-room depth behind
a person is not a person measurement. The current replacement remains
review-only pending adequate occupied evidence; a successful static fit is not
runtime admission or independent physical proof.

## Uncertainty and resolution

Candidate covariance is derived from available geometry instead of selecting
one of three fixed quality buckets:

- floor-ray pixel uncertainty is propagated through the calibrated ray/plane
  projection; shallow incidence, occlusion, and non-upright posture increase
  uncertainty;
- registered-depth covariance includes anchor pixel support, within-anchor
  spread, the camera's occupied-person registration residuals, posture, and
  occlusion;
- body-plane covariance includes plane scatter, solved-height uncertainty, and
  a conservative floor-plane term; side-profile foreshortening is not rejected
  merely because left/right joints overlap in image space;
- floor-plane uncertainty remains explicit;
- covariance is finite, symmetric, PSD, bounded, and transformed with the same
  Jacobian as the point at every view boundary.

The resolver ranks valid hypotheses using their direct evidence, covariance,
posture/support compatibility, motion consistency, and conservative PCF
likelihood. Floor support remains first for standing/unknown posture. For a
person explicitly classified as sitting or lying, a typed current `seat` or
`couch` body footprint ranks ahead of an ankle/floor ray; untyped body range
does not. No camera or room identifier participates in the ranking.

Two candidates contribute mathematically only when both their Mahalanobis
agreement and absolute metric separation pass the universal compatibility
limits against every existing contributor. Pairwise agreement with the
primary alone is insufficient. Compatible candidates use covariance
intersection, which does not assume their errors are independent. A literal
duplicate is not reported as fusion. A substantially disagreeing candidate is
never averaged: the best supported candidate is selected, the other is
retained as an alternate, and the output covariance is inflated to expose the
unresolved ambiguity.

After PersonGroundState accepts and filters a current measurement, the public
covariance is enlarged by the resolver-to-emitted displacement.  Prediction
and hold do not inherit a stale current-measurement covariance; downstream
compatibility behavior uses conservative continuation uncertainty.

The canonical global-world service admits measured observations only with a
complete revision and transform identity. `cv_prediction`,
`image_motion_prediction`, and `anchor_hold` can preserve guarded, exact
producer continuity in the snapshot; they cannot become new global fusion
evidence or renew last-good measurement time. Simultaneous camera
observations can fuse only in the same proven target-frame revision. Mixed or
incomplete target identities remain typed conflict evidence. Cross-camera
fusion uses full-matrix covariance intersection, preserving anisotropy and
correlation without assuming independent camera errors.

## PCF evidence

The active Scene Prior contributes only revision-matched soft evidence:

- measured extent;
- authored-space membership and boundary signed distance;
- observed-space confidence;
- floor elevation.

PCF never clamps, snaps, or manufactures a person position.  Unobserved space
is neutral by itself.  Authored walls and extent can reduce an otherwise
ambiguous candidate's plausibility.  The current unlabeled obstacle layer is
diagnostic only and cannot push a seated person away from a couch or chair.
The prior also reports metric distance beyond its source-grid rectangle. This
is distinct from the tighter camera-local raster bounds derived from authored
and observed PCF cells. A person visible through a doorway can therefore be
correctly located outside the current room-only raster without implying a
frame-transform failure.

Because unobserved PCF space is neutral, it cannot by itself reject every
mirror ray. Before a lifecycle owns a queue-published metric point, any selected
`floor_ray` must also have ray-incidence sine of at least `0.20`; shallower
geometry is intrinsically too sensitive to pixel error to establish first
metric authority. Body projection, registered depth, or a later adequately
conditioned floor ray may still establish the lifecycle.

An authored-boundary or source-grid violation remains uncertainty-aware. It
becomes a strong contradiction only when it exceeds both the universal margin
and three horizontal standard deviations. The candidate stays at its measured
coordinate for diagnostics but is typed weak and quarantined before
PersonGroundState. No point is clamped or snapped, and two compatible
hypotheses cannot fuse away a shared strong contradiction. A continuation
outside a room-only raster is shown as predicted/held rather than mislabeled as
a fresh measurement; a future registered multi-room prior can display that
same canonical coordinate without re-estimation.

A cold ankle ray that strongly contradicts the bound prior cannot normally
bootstrap from repetition alone because reflections can reproduce coherent
ankle motion. The narrow exception is two bounded, mutually consistent exact
ankle-pair samples under one immutable lifecycle/revision/calibration binding:
every contributing sample must have admitted plausible floor contact, tight
silhouette/range agreement, adequate ray incidence, current detector semantic
confidence, and no current no-ground contradiction. The second current sample
supplies the coordinate; the saved sample is corroboration, not a position.
Independent registered person-floor depth may also establish the lifecycle;
after that, current ankle evidence can override an imperfect prior through the
normal physical gate.

## Canonical and diagnostic outputs

The accepted track publishes:

- `world`, `world_valid`, `world_frame`, `world_frame_revision`, and
  `world_transform_sha256`;
- `world_quantity="ground_footprint"`;
- `world_covariance` for an accepted current measurement;
- `world_posture` and `world_support_state`;
- `world_support_surface` when a measured bounded surface was selected;
- the existing PersonGroundState admission, motion, hold, prediction, and
  trail fields;
- compact `world_resolver_*` decision scalars (confidence, selected ID, fusion,
  and disagreement) on normal tracking/statistics publications.

The canonical snapshot additionally publishes entity `position_state`
(`measured`, `predicted`, `held`) and `continuity_segment_id`. Source evidence
preserves its own `position_source`, `position_quality`, `position_state` and
support reference. Presence/lifecycle does not imply a fresh measurement. An
accepted support/frame/source-lifecycle discontinuity breaks the entity segment;
switching contributors must not hide a reset previously seen on another source.
Menon preserves these distinctions and uses exact held samples rather than
inferring freshness from confidence or animation.

BEV footpoints normally expose `worldElevationM`, transformed `covarianceXZ`
and optional full `supportSurface`; rich candidate diagnostics remain gated.
The renderer projects full canonical XYZ and full covariance through the same
display basis, then drops height for the top-down view. Menon applies one
validated target-world-to-scene mapping, with velocity transformed linearly and
covariance as A Sigma A^T. Its delayed interpolation is presentation only;
support/continuity changes reset playback, never interpolate through a step.

Canonical `world`, not the diagnostic candidates, moves the BEV dot or trail.
The BEV renderer accepts resolver details only when camera, source, tracker,
tracker-lifecycle generation, exact track key, frame, observation time, world
revision, transform SHA-256, calibration revision, and active PCF revision
match the rendered cohort. It projects candidate positions and covariance for
display without re-estimating them.

When a callback transitions a lifecycle from no queue-published metric output
to a valid accepted current metric point, that exact tracking/world/BEV cohort
is forced through the publication cadence immediately. Predictions, holds, and
saved proof samples do not trigger this rule.

An accepted support transition likewise forces its exact cohort immediately;
an older point must not be reclassified onto a new level between publications.

The full exact-cohort candidate tree is never copied into the canonical
tracking publication. The normal dashboard has a `Localization details`
toggle. It is off by default and asks BEV to carry the current final point,
candidates, uncertainty ellipses, disagreement, and PCF evidence for the exact
current cohort. It also reconstructs the retired room-policy's current-frame
choice from those same candidates. This reproduces the old selection
rule—including the old "fused" label whose numeric X/Z came wholly from
registered depth—but deliberately does not run a second temporal filter.
Turning it on or off cannot change tracking, world estimation, BEV admission,
or trails. The historical policy is validated at startup into a separate
diagnostic-only field and is never an estimator fallback. The toggle is a
connection-scoped capability
request: the native WebSocket server keeps the rich diagnostic payload
disabled until at least one connected dashboard requests it, broadcasts the
effective state to all connected dashboards, and revokes it when the final
requesting dashboard disconnects.

## Performance invariants

- Candidate count is capped at four and contains compact scalars only.
- No frame, image, mask, tensor, SDK object, or raw depth map crosses the
  resolver boundary.
- Candidate construction reuses already-produced pose/depth metadata; it adds
  no full-frame host copy, CUDA synchronization, model invocation, I/O, or
  persistence.
- Scene Prior lookups and resolver work are bounded per admitted person.
- Canonical tracking/world/BEV remains one exact ordered publication cohort.

## Practical validation

Use direct, goal-specific tests:

1. **Rigid geometry:** project known PCF/floor locations through the exact
   active calibration and confirm the point and covariance use the same frame
   and revision.
2. **Candidate separation:** on paced recorded clips, confirm floor and depth
   are both present when independently valid, invalid sources do not suppress
   valid ones, compatible candidates actually fuse, and large disagreements
   remain alternates.
3. **Human continuity:** walk, stop, sit, become partly occluded, and reappear;
   confirm the canonical point follows physical motion, holds do not grow
   trails, prediction is typed, and a relocation breaks the trail.
4. **World/BEV identity:** for every rendered dot, match camera, source,
   tracker, lifecycle generation, track key, frame, observation time,
   calibration, world revision, transform SHA-256, and PCF revision to its
   tracking row.
5. **PCF behavior:** verify wall/outside evidence changes candidate
   preference/quality without changing its coordinates, and that two agreeing
   candidates cannot override the same strong revision-matched boundary or
   extent contradiction.
6. **Performance:** in occupied paced replay, measure per-source progress,
   analytics/resolver callback tails, publication overflow, encoded access-unit
   cadence/drops, and WebRTC decoded frames.  Dashboard FPS alone is not proof.

Recorded clips can prove cohort integrity, continuity, physical plausibility,
and performance.  They cannot establish centimeter absolute accuracy without
independently surveyed real-world marks; that limitation must be stated rather
than replaced by visual confidence.
