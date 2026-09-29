# Person position accuracy: observations and deferred work

Status: proposed investigation and idea backlog, 2026-09-26. The user explicitly
deferred gait implementation. This note changes no estimator, calibration, or
presentation behavior.

## Evidence and revised worklist

The paired Living Room review is available in the local scan artifact:
Supporting alignment review: `data/mapanything_phone_scans/20260922-120808-2713a4ce/path_review/bev-video-alignment-20260926/alignment_review.md` (local ignored artifact; not included in this repository).
It combines the retained static recording, full phone video, recorded canonical
Noesis positions, and the independently registered phone optical-center path.
The phone optical center and the person's ground footprint are different
physical quantities; their separation is diagnostic rather than a position
error label. The recording also predates the latest non-upright localization fix.

Keep the following worklist visible; it is an investigation order, not a claim
that every item has already been demonstrated as a runtime defect:

1. **Reference alignment:** the synchronized review and a better global image
   registration are complete. Isolate the remaining time/view-dependent reference
   errors, especially repeated locations on outbound and return passes.
2. **Position calculation:** at selected frames, compare observed contact anatomy,
   body projection, admitted metric depth, ray/surface intersections, and emitted
   world coordinates to identify which calculation causes each discrepancy.
3. **Metric depth:** investigate the retained review's rejected far-range sample
   and flat registration curve using independently registered same-ray geometry
   over the occupied range. Preserve camera-Z versus ray-distance semantics.
4. **Measured floor elevations:** retain this as an explicit general improvement.
   Fit actual support surfaces and their spatial extents, then intersect each
   observation with the supported surface. Do not hard-code two rooms, a step
   height, or a regional position offset. This item was deprioritized by the
   reference mismatch, not ruled out by it.
5. **Presentation comparison:** re-render the calculated candidates and canonical
   output through this same BEV review. Later Menon work uses its canonical
   world-snapshot presentation and preserves the supplied 3D position.

## Deferred idea: gait-aware body footprint

The user observed the image anchor alternate among the trailing foot, leading
foot, and an apparently body-centered position while walking, especially when
one foot was occluded. The current pose-anchor constructor supports this
mechanism: it averages the two ankle image coordinates when both are available,
then uses the remaining ankle when only one is available
([`person_ground_state.py`](../../noesis/telemetry/person_ground_state.py),
`resolve_pose_floor_anchor`, ankle-pair and single-ankle branches). This is a source
of measurement variation to investigate, not proof that all visible jitter has
that cause.

The proposed direction is a body-centered ground footprint with an observed
support envelope, conceptually the footprint of the user's vertical cylinder:

- Keep actual contact points and the desired body-centered ground footprint
  explicitly distinguished. A projected torso/pelvis center is not a measured
  center of mass. Preserve the existing `ground_footprint` contract and introduce
  a separately typed quantity if the chosen semantics require it.
- Use current calibrated pose, admitted depth, visibility, and measured motion
  to estimate the body center and extent. Avoid a population-average cylinder
  radius, assumed body height, or a fixed percentage that masks real motion.
- Evaluate stance versus swing evidence and contact changes. A visible ankle
  need not be touching the floor. When valid contacts are available, compare
  combining their world projections with the current image-midpoint approach;
  perspective means projecting the average image pixel is not generally the
  same as averaging the projected world positions.
- Treat plausible within-body gait excursions through measurement uncertainty
  and the existing `PersonGroundState` temporal filter. A hard rule that holds
  the dot until it exits a cylinder can cause delayed starts, stair-step motion,
  and missed slow movement. A foot can also legitimately extend beyond the
  torso's vertical projection during a stride. The body envelope should inform
  evidence weighting, not clamp a measured point, reject every contact outside
  the envelope, or add a second frontend smoother.
- When revisited, use the existing occupied recording to check walking both
  directions, one-foot occlusion, starts/stops, slow motion, turns, and steps.
  Compare jitter reduction with body-position lag and displacement accuracy;
  smooth-looking video alone is insufficient.

## Outbound versus return: useful drift evidence

The user observed closer phone/Noesis agreement after returning from the hallway
than during the initial outbound walk. With the current scene registration,
median phone-center/ground-point separation is 0.406 m in the 23–60 s outbound
span (64 matches) and 0.249 m in the 110–140 s return span (29 matches). These
spans differ in location, direction, and visibility; the numbers do not measure
position accuracy. Retain this observation and compare
repeated physical locations, visibility, source/contact basis, speed, and carry
orientation before attributing it to drift. A timing offset produces an error
related to velocity that reverses with walking direction; phone/body offset and
which ankle is visible can also change with direction. Static scene landmarks
can test reconstruction drift independently of those person-related effects.

The current registration's held-out scene reprojection medians are approximately
17.3 phone-model pixels for early views 1–35 and 24.9 for return views 231–255.
Thus better agreement between the two dots does not by itself show improved
phone-reference accuracy. Independently solved view yaw medians differ by about
4.8 degrees between these spans; viewpoint/depth bias and reconstruction drift
remain competing explanations. The saved refinement's 19 accepted revisit
constraints connect only early views 1–31, so they do not establish whole-walk
loop closure. Its 39 withheld early-to-return comparisons have median rotation
residual 4.49 degrees and translation residual 0.130 m, providing another reason
to investigate the changing orientation. Sources: `scene_alignment.json`
beside the review and
`offline-20260926-living-room-pcf/refinement/path_refinement_report.json`.

## Raised foyer floor: retain and measure

The user's approximate step-height report is 6 inches (0.1524 m), not an admitted
phone-derived measurement. With a fixed image ray, a floor-height change changes
horizontal range by `delta_height / tan(ray_downward_angle)`. Using the captured
camera center at 2.08688 m above its reference plane and the reviewed 9.03 m
front-door floor-ray range, raising the support plane by the reported amount
would shorten that range by about 0.66 m. This is a sensitivity calculation, not
a correction to apply or a complete explanation of the remaining discrepancy.

Measure the level difference from separate reconstruction/depth floor patches
on both sides of the transition, retaining fit uncertainty and the common frame.
Phone-camera height changes alone are insufficient because carrying height and
reconstruction pose error also change Y. A future generic implementation should
carry the selected support elevation through canonical 3D world state and use
surface extent/contact evidence to select among plausible intersections. Keep
ambiguity explicit at the step rather than averaging two levels into a fictitious
surface. Static reconstruction geometry must have its declared calibration/world
authority before influencing live position calculations.

## Opus review: checked-out presentation details

The older Menon `src/features/path-visualization/TrackProjection.js` does flatten
positions to `__environmentFloorY` or zero and applies its own X/Z smoothing.
However, the current checked-out `src/services/WebSocketClient.js` routes
`world_snapshot` through `CanonicalWorldIngress`; tracking messages do not feed
that older projector. `CanonicalWorldPresentation` transforms the supplied XYZ
position once and preserves height. This is checked-out code evidence, not a
claim about which bundle is currently deployed. Check future Menon alignment in
that canonical path and account for the visualizer's small decorator lift when
comparing actual rendered geometry.

The BEV footpoint payload publishes planar X/Z as JSON `x`/`y`; it has no separate
height field. Covariance already reaches the optional `resolverDiagnostics`
through `covarianceXZ` while localization details are requested
([BEV implementation](../../noesis/telemetry/bev.py) and
[WebSocket contract](../../docs/api_contracts_ws.md)). That supports the current
diagnostic ellipses. Always-visible uncertainty would need a compact normal
footpoint field. Per-surface ambiguity additionally needs support-surface
identity/revision, candidate weights or another explicit ambiguity representation,
and corresponding positions/uncertainties; one 2x2 covariance cannot identify
which floor is being considered. Preserve 3D geometry through the calibrated
transform before reducing it to a top-down display.
