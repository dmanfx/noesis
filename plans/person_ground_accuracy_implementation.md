# Measured world-position and BEV implementation

Status: feasible course-corrected implementation and direct replays completed
2026-09-29 UTC, consolidated in Git 2026-10-04; WO5 remains partial pending
qualified occupied capture evidence. This plan retains the original producer/BEV
scope, not a prohibition on the subsequently authorized Menon integration.
PCF-relative registration is the target; intrinsic metric DAv2 and
a new room survey are not prerequisites. The owner measured a 0.1905 m rise at
both ends and specified practically flat parallel floors. Prior surface-aware
and canonical-output work is retained. The owner authorized coherent commits on
2026-10-04; no push is authorized.

See [dated results and remaining limits](archive/completed/person_ground_pcf_reassessment_results_20260929.md)
and the [historical first delivery](archive/completed/person_ground_accuracy_results_20260928.md).
Current scene integration is documented in
[structural registration](../docs/structural_scene_registration.md) and its
[assessment](structural_registration_results.md). The configured native service
now runs non-SV3DT baseline with the L detector; the starting V3DT process below
is historical evidence. Backend gait/body-footprint and standing-jitter polish
remain deferred.

## Objective and boundaries

Improve the physical placement of canonical person ground footprints and their
BEV presentation using measured, revision-bound geometry and current evidence.
Keep visibility/continuity improvements already made. Do not optimize IDs, add a
second display estimator, or turn validation into a release process.

In scope: work orders 1–7 below, native DS9.1 producer, shared world contracts,
calibration/depth, BEV, and direct recorded comparisons. Out of scope: **all
Menon review and implementation**, gait/body-cylinder estimation, model swaps,
hard-coded step heights or room offsets, phone-path/Noesis-track fitting, and
weakened validity gates. Phone optical centers are reference diagnostics, not
person-ground truth. Independent reconstruction can measure geometry without
automatically acquiring live tracking authority.

The primary agent owns this plan, shared-file integration, native runtime
operations, and final assessment. SOL xhigh investigates difficult geometry;
Luna xhigh handles bounded evidence and direct checks. Only one owner edits each
shared file. No inference jobs run concurrently without a GPU/resource check.
Fast service is used only if supported by the available tool; no unsupported
mode is assumed.

## Execution order and evidence baseline

Begin with WO1 and independent WO2/WO5 evidence inspection. Implement the
surface contract and ray math in WO3 before integration in WO4. WO5 can proceed
independently where metric labels permit it. Publish/inspect the same canonical
output in WO6, then run WO7; use concrete replay failures to revisit the owning
work order. This is a dependency order, not an approval/promotion sequence.

Starting Git revision: `a599334` on `DS9`; tracked tree initially clean.
Unrelated untracked `.codex/config.toml`, `.scope/`, and the RoomWalk APK signature
are excluded. Recheck diffs before each shared edit: concurrent work may land.
The observed live process uses `infer_v3dt.yaml`, `cameras_v3dt.yaml`, YOLO26-m
and `tracking-mode=v3dt`. The user subsequently clarified that this task targets
the **non-SV3DT baseline pipeline**; SV3DT remains untouched and is restored
after bounded baseline replays. The older retained recording is not evidence
of either current process. Establish source/config/frame authority before replay.

Common retained evidence (local ignored artifacts):

- `data/mapanything_phone_scans/20260922-120808-2713a4ce/` contains the handoff,
  prepared phone views, paired static reference, and offline reconstruction.
- Its `path_review/bev-video-alignment-20260926/` contains the synchronized
  continuous-phone/static/BEV review, independent scene fit and timing report.
- `data/mapanything_phone_scans/.companion-captures/companion-20260922-120318-4e394d2f/`
  contains the occupied static video and original tracking/world records.
- July single-person and multi-person room recordings will be resolved from
  the existing video directory without copying or inventorying large trees.

## WO1 — Establish current authority and local registration

- [x] Authority, local registration evidence and before capture reviewed.
- **Evidence:** selected process, CLI/config/model and native environment;
  applicable calibration/frame bindings; retained timing and scene registration;
  stationary landmarks in early/return views; source-contact/BEV exemplars.
- **Work:** capture a bounded current-code before baseline; distinguish native
  SV3DT positions from universal-resolver positions and presentation transforms.
  Scope implementation and replay to the baseline producer, leaving native
  SV3DT untouched per the user's clarification. Check static camera-to-world registration independently of people/phone
  trajectories. Quantify local versus whole-walk registration limitations.
  Do not make whole-phone-path perfection a prerequisite for qualified local
  surfaces. Preserve every source recording and original result.
- **Likely files/contracts:** existing alignment/review scripts, calibration
  binding inspection, `DS9/scripts/render_recorded_replay_config.py` only if the
  current mode requires a confirmed repair; no calibration replacement merely
  to make the trajectories agree.
- **Dependencies:** none; this establishes inputs for all remaining work.
- **Direct validation:** reprojection of withheld stationary features, exact
  source/frame/PTS association, and matched world-to-BEV coordinates. Report
  timing uncertainty and reference-point differences explicitly.
- **Done:** reproducible before inputs/config/mode and representative frames are
  recorded; each candidate correction has a known owner and supported frame;
  uncertain global alignment is not misrepresented as measured accuracy.

## WO2 — Measure bounded support surfaces

- [x] Owner-measured parallel upper patch implemented and exercised in native
  paired, July and multi-person replays (2026-09-29 UTC). Boundary precision and
  the earlier image-height discrepancy remain explicitly limited.
- **Evidence:** owner-measured 7.5 inches (0.1905 m) at both ends, explicit
  parallel-floor constraint, calibrated visible foyer perimeter, admitted PCF
  coverage and exact camera/world binding. The old image-derived 0.153 m is a
  discrepancy diagnostic, not the calibration height.
- **Work:** retain the calibrated lower plane and calculate a parallel upper
  patch from the measured rise. Project visible perimeter vertices through the
  bound camera, limit to PCF coverage, store a separate explicit calibration
  artifact, and retain local uncertainty/provenance. Do not infer a slope from
  noisy learned depth, extend the step across the room, or move the PCF map.
  Investigate annotated-edge vertical correspondence, projection and camera
  binding; broader physical measurements are optional targeted final checks.
- **Likely files/contracts:** an SDK-neutral support-surface contract/module,
  calibration loading/binding in `noesis/calibration/manager.py`, a small offline
  extraction utility using existing reconstruction outputs. Avoid changing PCF
  reconstruction or its live authority merely to satisfy localization.
- **Dependencies:** WO1 local frame registration; can investigate in parallel.
- **Direct validation:** measured rise and parallel normal, exact binding,
  finite extents on source/PCF views, ray reprojection and native person
  transitions. Report local PCF scatter separately from absolute calibration
  uncertainty; two equal tape readings do not establish survey precision.
- **Done:** every admitted surface has measured geometry and explicit binding;
  ambiguous or unsupported surfaces remain review evidence, not live offsets.
  If the retained data cannot establish a level, identify the exact missing
  measurement rather than fabricate it.

## WO3 — Calculate and select surface-aware measurements

- [x] Implemented and directly tested with analytic measured-plane fixtures.
  WO2's explicitly measured upper patch now exercises this code in real replays.
- **Evidence:** WO2 surfaces, exact current anchor rays, pose/contact visibility,
  eligible current registered depth, existing hypothesis admission rules.
- **Work:** intersect rays with bounded surfaces using their full plane equations;
  reject negative/near-parallel/outside-extent intersections. Propagate image and
  plane uncertainty. Select using current supported geometry; keep ambiguous
  levels separate, never average them into an invented intermediate floor.
  An existing single calibrated floor remains the explicit behavior where no
  additional admitted measured surfaces exist, not an inferred extra surface.
- **Likely files/contracts:** `noesis_core/contracts/world_measurement.py`,
  `noesis_core/world/resolver.py`, new bounded surface geometry helpers, and
  current-measurement construction in `DS9/noesis/pipelines/hooks.py`.
- **Dependencies:** WO1 authority and WO2 schema; mathematical work can use
  analytically known planes before measured artifacts are qualified.
- **Direct validation:** ray/plane intersections and reprojections, boundaries,
  incompatible surfaces, transformed covariance and binding mismatch checks;
  feed an exact real retained observation through candidate construction.
- **Done:** current candidates identify one measured surface or explicit
  ambiguity; same-support fusion requires compatible surface identity; no
  room-name branch, clamping, validity relaxation or extra temporal filter.

## WO4 — Preserve support geometry through canonical output

- [x] Implemented; full producer-proof/world-service/BEV contract checks pass.
- **Evidence:** current full producer path, native SV3DT frame conventions,
  resolver output, PersonGroundState continuations and world-service ingress.
- **Work:** remove affected accidental global-floor reflattening. Carry selected
  surface ID/revision/elevation and full XYZ through temporal output and public
  world provenance. Preserve correct seat/couch ground-footprint semantics and
  lifecycle-local torso depth. Holds/predictions retain their true provenance,
  not a fresh surface measurement; never smooth a level change through air.
  Leave native SV3DT implementation untouched.
- **Likely files/contracts:** `DS9/noesis/pipelines/hooks.py`,
  `noesis/telemetry/person_ground_state.py`, `noesis_core/world_service.py`,
  shared world-observation schemas and `docs/metadata_contracts.md`.
- **Dependencies:** WO3. Native SV3DT limitations are outside this work order.
- **Direct validation:** accepted measurement → temporal output → canonical
  world → BEV with full identity/revision; raised support, sloped support,
  sitting/standing transitions, hold/prediction and stale-binding examples.
- **Done:** no affected consumer silently replaces measured Y with one global
  floor, incompatible surfaces do not fuse, and ordinary single-level behavior
  and existing non-upright continuity remain valid.

## WO5 — Repair metric depth registration at its evidence boundary

- [ ] Partial: fitter repaired and PCF-bound scale/DAv2 reassessment completed
  (2026-09-29 UTC). Static held-out consistency and four exact occupied pixels
  support this approach, but do not qualify a replacement active mapping.
  Far-domain and capture-disjoint occupied support remain insufficient;
  domain/flatness/foreground/cohort gates are unchanged.
- **Evidence:** raw DAv2 values and anchor UV/frame/PTS, tensor coordinate map,
  camera-Z versus range conventions, fit/holdout data, existing curve domains,
  and same-frame occupied person depth samples. PCF is the room's metric scale;
  DAv2 does not need to be intrinsically metric.
- **Work:** diagnose the flat curve and far-range rejection; repair any wrong
  sample association, domain, units, range conversion, fit or validation logic.
  First bind MapAnything optical Z to the admitted PCF using stationary,
  person-excluded correspondences and spatial/temporal holdout. Apply only the
  resulting registration to actual occupied MapAnything samples paired with
  DAv2 at the same anchor and source frame; retain fit shape/range gates.
  Empty-room depth behind a person, phone camera position and Noesis output are
  not person depth labels. Keep unsupported ranges rejected and explain why.
- **Likely files/contracts:** `noesis/calibration/depth_registration.py`, builder
  and review helpers, native compact object-depth fields only if association is
  defective, calibration record provenance and affected focused tests.
- **Dependencies:** WO1 and admitted PCF frame/scale; may run alongside WO2–4,
  feeding valid depth into WO3 without making depth mandatory. A manual room
  survey is not a prerequisite. Independently proven physical accuracy remains
  a distinct claim from demonstrated PCF-relative alignment.
- **Direct validation:** numeric camera-Z/range round trip at actual anchor UV,
  exact tensor/object cohort and sampling region, near/mid/far held-out errors,
  seated torso samples, fit shape/domain and actual producer output.
- **Done:** confirmed implementation defects fixed and tested; accepted depth
  represents the sampled person anchor in the correct metric frame. A new
  accuracy claim or widened range requires evidence, not a convenient curve.

## WO6 — Publish and inspect the unchanged canonical point in BEV

- [x] Implemented and inspected on exact native replay cohorts. The historical
  July Living Room still has existing display-guard exclusions, reported below.
- **Evidence:** full canonical XYZ/surface/covariance/provenance and the exact
  paired `tracking`, `world_snapshot` and `bev-frame` cohorts.
- **Work:** add compact surface ID/revision/elevation and covarianceXZ to normal
  footpoints; keep rejected alternatives optional diagnostics. Transform full
  3D coordinates before dropping Y, including diagnostic candidates. Rebuild
  review views with source image/contact, canonical dot/trail, support/depth and
  fresh/hold/prediction labels. Keep optical-center overlay visibly distinct.
- **Likely files/contracts:** `noesis/telemetry/bev.py`,
  `docs/api_contracts_ws.md`, existing dashboard footpoint types if required,
  offline retained-video renderer. No new browser localization or smoothing.
- **Dependencies:** WO4; review renderer scaffolding can proceed earlier.
- **Direct validation:** exact world-to-BEV coordinates/covariance in both normal
  and details modes, visible trails and lifecycle breaks, source-frame/BEV
  visual checks at the same cohort. No details-toggle-dependent position.
- **Done:** BEV presents the producer's point and support provenance faithfully;
  output distinguishes fresh measurement, hold, prediction and invalid absence.

## WO7 — Direct before/after producer and BEV comparison

- [x] Feasible direct checks completed across all three video classes. Absolute
  accuracy and universal dot coverage are not established; see result limits.
  Corrected-code runs and before/after source/BEV stills refreshed on
  2026-09-29 UTC after real raised-floor propagation repairs.
- **Evidence:** unchanged source videos/config/models, baseline and changed
  outputs, paired walk plus July single-person and multi-person room clips.
- **Work:** actually run changed native producer, not just redisplay old world
  rows; use the same BEV consumer on both outputs. Inspect near/far/step,
  outbound/return, sitting/occlusion, multi-person and other-room examples.
  Compare localization coverage, freshness, hold/prediction age, coordinate
  shifts, trail breaks, jitter/lag and source/BEV visual placement. Keep optional
  work bounded and inspect queue/source progress for the replay.
- **Likely files/contracts:** bounded replay/capture/review utilities and a
  concise result artifact; regression tests only for affected behavior.
- **Dependencies:** all feasible preceding corrections; revisit the responsible
  work order when evidence reveals a defect rather than expanding test scope.
- **Direct validation:** changed-file syntax, focused geometry/resolver/depth/BEV
  tests, recorded producer+consumer smoke on all requested video classes, visual
  source/BEV comparisons. Independent known floor locations can support the
  existing 20 cm p95/30 cm maximum clear-contact target; phone/track agreement
  alone cannot. Report unavailable absolute labels honestly.
- **Done:** review artifacts show what changed and why, direct checks pass or
  have explicit evidence-backed limits, unrelated work/runtime is preserved,
  no commit/push performed, and the BEV result is ready for user review before
  any Menon work begins.

## Work log

- 2026-09-28: inspected the tracked-clean starting checkout and observed explicit
  SV3DT process; started independent measured-surface and depth/video evidence
  investigations. All task replays used baseline mode, not native SV3DT.
- Implemented bounded support contracts, ray/plane calculation, compatible
  surface selection, plane-aware covariance, temporal/proof propagation,
  finite-extent fusion checks and full-XYZ BEV presentation. Default active
  single-floor calibration is unchanged.
- Replaced cumulative-maximum depth fitting with support-weighted isotonic
  pooling. Independent-label and unobservable-domain requirements remain.
- Measured the visible threshold riser at approximately 0.153 m using image
  edges and current calibrated lower-floor geometry. Its top edge projected
  onto the wrong lower floor moves about 0.34–0.36 m horizontally. This is not
  sufficient evidence for an entire upper support plane.
- Captured current-code before and changed-code after paired-walk baseline
  runs, then changed-code July and multi-person runs. Added exact-cohort BEV
  review images, freshness states, transform checks and a source-raster check.
  The first July Family Room attempt had a wrong input raster and is explicitly
  superseded by the full-FoV resized replay; originals were retained.
- Native services restored to their original selected mode. No Menon review,
  SV3DT implementation, gait changes, new calibration activation or Git
  publication occurred. Result report distinguishes delivered code from the
  physical accuracy outcome that still needs independent measurement.
- 2026-09-29 UTC, course correction: stored the owner's measured 0.1905 m rise
  as room calibration data, projected the visible upper-floor perimeter and
  bound it to the admitted PCF revision. The baseline config now loads this
  finite parallel plane; the selected SV3DT config is unchanged. The earlier
  work-log statement about no new calibration applies only to first delivery.
- Real replays exposed body-plane solving on the wrong floor, publication
  watermark support mismatch, flattened inferred-motion proof origins/deltas,
  and depth-covariance variable shadowing. Corrected those owning producer
  paths without weakening canonical validation or moving the map.
- PCF-bound static holdouts and same-frame occupied pixels demonstrate useful
  DAv2 registration information, not intrinsic scale or physical accuracy.
  The active flat mapping remains unchanged until independent occupied
  capture/domain gates pass; a manual room survey is not a prerequisite.
- Final three native baseline runs exited cleanly; 281 focused tests passed.
  Retained 47 corrected source/BEV stills and before/after exact-cohort analysis.
  Valid world-to-BEV transform parity is exact, with 53 July display-guard
  exclusions and meaningful occlusion/placement limits documented. No Menon
  review, SV3DT implementation, gait work, commit or push was performed.
