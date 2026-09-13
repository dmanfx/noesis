# Reconstruction and world unification implementation plan

Status: implementation and final review complete, 2026-09-04. Coordinator: root agent;
implementation: GPT-5.6-Luna agents at xhigh reasoning. VI3 experiments are deferred.

## Outcome and scope

Improve the existing DA3-carried, prior-conditioned MapAnything reconstruction
with an explicit shared metric world, physically grounded capture and camera
calibration, verified trajectory constraints, honest uncertainty, and
visibility-aware surface fusion. Preserve one Noesis world authority and one
world-to-Menon transform. Deliver usable application paths, not disconnected
helper libraries or tests that only mirror implementation.

The user confirms that existing static-camera extrinsics came from manually
placed SweetHome3D camera objects and visually estimated yaw/tilt. They are
initial estimates, not immutable physical truth. Back up those exact inputs
before any replacement. Better room-walk-derived poses may replace them after
geometric validation and dependent-consumer regeneration. This authorization
supersedes earlier advice to preserve raw extrinsics indefinitely. It does not
make rejected room joins or guessed physical measurements accepted evidence.

Runtime downtime is authorized when needed. The coordinator owns GPU/service
scheduling. No agent changes services or consumes the GPU without coordinating
first. Heavy dependencies, recordings, intermediate arrays, and reports belong
under the configured large-storage root, not the nearly full root filesystem.

Capture device: the user identifies a new Fold 8 Ultra running Android.
Implement the Android native recorder/import workflow. Device-reported Camera2
timestamp support and calibration are checked from capture evidence; the model
name alone does not establish synchronized acquisition or camera-to-IMU geometry.

## Execution rules

- Read root and applicable subtree AGENTS.md. Native-host DS9.1 is canonical.
  Read the routed NVIDIA skill/references before SDK-facing changes.
- Preserve the initial dirty tracking/resolver lane and Menon changes. Do not
  reset, checkout, stage, commit, push, or change branches. Shared Noesis stays
  on DS9. Explicitly coordinate any edit to another agent's file.
- The coordinator writes work orders, reviews diffs/results, and assigns
  corrections. Agents implement. Coordinator implementation is reserved for
  repeated agent failure.
- Reproduce each affected behavior, change the smallest useful surface, run
  focused checks, then exercise the changed producer and direct consumer.
  Do not add full-suite, long-soak, release, bundle, selector, state-clone,
  promotion, sealing, or independent-reviewer machinery.
- Keep optional reconstruction/VIO work offline or in existing bounded workers.
  Preserve models, resolution, cadence, enabled outputs, exact world cohorts,
  and existing rejection thresholds unless evidence supports an explicit change.
- No AMC, extra MV3DT edges, browser person-position corrections, identity
  association in global fusion, or hard PCF constraints on live people.
- Never fabricate IMU samples, calibration, physical dimensions, loop matches,
  or accepted registrations. Existing RGB-only recordings remain explicitly
  RGB-only. A missing capture is a stated evidence dependency, not a test pass.
- Record implementation, direct test results, real-data before/after metrics,
  and remaining evidence dependencies separately. No speed/accuracy claim from
  schema acceptance or synthetic tests alone.

## Improvement 1: Shared metric frame from producer through renderer

### WO-1A — Separate frame identity from artifact identity

1. Trace ScenePriorCatalog, ScenePriorFrameBinding, ScenePrior builder/runtime,
   calibration manager/bundle, WorldPositionObservation, GlobalWorldFusion,
   and Menon frame lookup. Retain exact legacy inputs through explicit versioned
   validation; never weaken the current revision/hash checks wholesale.
2. Introduce the smallest explicit contract needed for independently versioned
   reconstruction artifacts to reference one accepted metric frame. A map
   revision and a coordinate revision are different identities. Verify source,
   target, units, proper rigid transform, floor plane, digest, and acceptance.
3. Separate metric calibration dependencies from authored presentation metadata.
   A presentation-only change must not redefine physical-world coordinates;
   real changes to calibration/floor/world registration must invalidate the
   appropriate dependent objects. Supply an explicit migration path for current
   catalogs instead of silently changing their interpretation.
4. Provide a usable builder/import path for accepted room-to-home transforms.
   Rejected review assemblies must continue to be rejected. Do not relabel the
   three current room frames as identical without accepted geometry.
   Require the accepted binding/report in the normal scene-prior build path,
   not only in a standalone helper. Apply the transform to retained geometry,
   camera/floor outputs, and presentation before building the grid, and exercise
   a nonidentity build -> catalog -> CalibrationManager roundtrip. A v2 build
   must use the physical source revision expected by that runtime consumer.

### WO-1B — Make fused provenance renderable

1. Preserve individual calibration/transform hashes in WorldSourceEvidence.
   A fused entity with two accepted edges into the same target frame may have
   no single source-edge hash. Do not invent or choose an arbitrary one.
2. Make the world-to-scene transform belong to the target frame revision, with
   explicit provenance supplied by Noesis and exact validation in Menon.
   Preserve the existing composed binding path for existing room revisions.
3. Exercise the actual Noesis producer -> serialized snapshot -> Menon state ->
   renderer boundary with two cameras, different edge hashes, one target frame.
   Verify position, velocity, covariance, missing/stale-frame rejection, and
   continuity across map-only changes. A correctly fused entity must render.

Primary files: noesis_core/contracts/scene_prior.py, noesis/scene_prior_builder.py,
noesis_core/scene_prior.py, noesis/calibration/manager.py, calibration bundle,
noesis_core/world/fusion.py, and Menon CanonicalWorldPresentation/CalibrationManager.

## Improvement 2: Synchronized capture, VIO, and measured calibration

### WO-2A — Preserve trustworthy phone sensor capture

1. Inspect the existing upload/prepare flow and the actual target phone.
   Define a bounded capture input containing video, frame timestamps mapped to
   IMU time, timestamped accelerometer/gyro samples, units/axes, device/sensor
   identity, phone intrinsics/distortion, camera-to-IMU transform, timing offset,
   and stabilization/crop/orientation information needed for interpretation.
2. Use an existing suitable recorder or a native recorder when browser APIs
   cannot provide acquisition timestamps/calibration. Browser callback arrival
   timestamps must never be represented as exact sensor capture times.
3. Add usable upload/import and UI status for sensor recordings. Preserve
   original streams, associate prepared views with exact source frame times,
   and validate finite values, monotonicity, gaps, coverage, video association,
   size/count bounds, and safe archive paths. Incomplete calibration may be
   retained but cannot be admitted as a metric VIO result.
4. Verify recorder/import/selection with a small real public sensor recording
   and, when available, the user's phone. Do not claim phone recording verified
   from a synthetic browser event stream. Existing video upload still works.

### WO-2B — Establish an executable conventional visual-inertial baseline

1. Select an established visual-inertial estimator from primary documentation
   after checking native dependencies; prefer a bounded offline OpenVINS path
   where appropriate. Do not implement naive double integration as VIO.
2. Install/build only necessary dependencies in large storage. Pin the source
   used and record a reproducible native command. No Docker, VI3, or model
   weight adaptation. An adapter that only checks for a missing executable is
   not completion: run an actual estimator on a small real synchronized dataset.
3. Emit poses, velocities, gravity/bias estimates, quality/uncertainty, calibrated
   frame/time identities, reset/segment boundaries, and explicit failure reasons.
   Require sufficient excitation and calibrated camera/IMU timing. Preserve
   relative gauge; inertial data does not determine house origin/global yaw.
4. Connect accepted VIO results to DA3 carrier refinement/conditioning through
   a documented common contract shared with WO-3. Transform translation/depth
   scale consistently. Keep the current RGB workflow explicit for old captures.
5. Compare metric scale/trajectory against dataset ground truth excluded from
   estimation. Separately list phone-specific calibration/capture dependencies.

### WO-2C — Replace manual static poses only with supported geometry

1. Add or reuse multi-view static-camera-to-map PnP with complete-view and
   temporal validation, proper frame conventions, floor/scale checks, and
   reprojection on exact calibrated static-camera imagery.
   The current localizer sets `accepted_for_canonical_use` from the
   `full_pcf_pose` option. Replace that coupling with measured acceptance;
   requesting a complete pose is not proof that it is a valid calibration.
   Keep entire excluded views/temporal groups out of the fitted consensus and
   evaluate the resulting pose on them. Check the static image's intrinsics,
   distortion/rectification, resolution, and source frame before PnP.
2. Use independent measured dimensions or independently calibrated sensor
   evidence to diagnose scale. Do not force the PCF into authored dimensions
   using unconstrained per-room scales. A Family-only structural fit cannot
   establish whole-home metric accuracy.
3. Before accepted extrinsics replacement, retain a timestamped exact backup
   of manual inputs and a portable provenance record. Regenerate only direct
   dependent calibration/depth/frame-binding artifacts, and exercise those
   consumers. Unsupported cameras stay explicitly unverified.

Primary files: tools/mapanything_phone_scan app/static/processing, new capture/VIO
adapters and focused tests, alignment/localize_pcf_static_camera tooling, and
calibration consumers coordinated with WO-1.

## Improvement 3: Verified loops and global trajectory consistency

### WO-3A — Build evidence-backed nonadjacent constraints

1. Reuse retained prepared images and current feature/matching primitives.
   Retrieve plausible nonadjacent revisits; require geometric verification,
   sufficient support, spatial distribution, and complete view identities.
2. Keep bounded candidate/match budgets. Do not force the last pose to the first
   solely because a capture is called a loop. Remove or explicitly gate the
   generic joint solver's unconditional endpoint closure; DA3-carrier behavior
   must be analyzed separately.
3. Admit calibrated VIO relative constraints from WO-2 without inventing loop
   closure or cross-room identity. Preserve right-handed metric frames.

### WO-3B — Refine and rebuild affected geometry

1. Extend the existing window/pose-graph mechanism rather than creating a
   parallel reconstruction stack. Optimize verified constraints jointly across
   affected windows with bounded work and explicit gravity/scale treatment.
2. Apply the resulting pose changes to raw geometry and all retained camera
   outputs consistently, then re-run dependent conditioning/fusion where needed.
   Do not modify camera poses while leaving old world points in the former frame.
   Integrate before DA3-prior conditioning in the existing `pcf.py` path, using
   a separate refinement output directory. Revalidate the static-world mapping
   after a physical scale or gauge change; an old alignment must not silently
   undo the inertial scale correction or reinterpret new poses in the old frame.
3. Compare the same retained capture with the current result, alternate window
   boundaries, and withheld temporal visits. Report closure, pose disagreement,
   scale consistency, surface residuals, and coverage. Preserve room-join gates.
4. Use the accepted transform path from WO-1 for final room registration. If
   current footage still lacks sufficient connector evidence, provide an exact
   short recapture requirement and keep rejected joins review-only.

Primary files: windowed_inference.py, windowed_da3_inference.py,
run_mapanything_prior_variants.py, pcf_multiroom_pose_graph.py, matching tools,
build_consensus_fusion.py (coordinate edits with WO-4), and pcf.py integration.

## Improvement 4: Honest confidence, support, and evaluation

### WO-4A — Correct confidence and validation semantics

1. Keep model confidence percentiles as ranking scores; do not label empirical
   CDF values calibrated probabilities. Preserve raw scores/provenance.
2. Account explicitly for DA3-derived conditioning when combining MapAnything
   and DA3 evidence. Do not grant an unexplained independent-evidence bonus.
   Qualify any changed selection threshold on matched retained data; do not
   silently reduce useful coverage merely to improve residuals.
3. Label even/odd reprojection correctly as internal consistency when both
   views participated in inference. Count large errors instead of censoring
   errors above 2 m. Report coverage and catastrophic-error fractions alongside
   residual quantiles, with finite/empty-input behavior explicit.
4. Add a practical independent-evaluation input for withheld measurements or
   separately excluded views/visits, and separate fitting from evaluation.
   An external measurements file must include its frame/unit/source identity.
   Match its exact coordinate revision or equivalent static-target/calibration
   binding to the candidate before computing distances. The shared string
   `backend_world_m_stream_points` alone cannot identify a room's coordinates.
5. Add empirical reliability/error calibration only when sufficient independent
   residual data exists; otherwise publish explicitly uncalibrated scores.

### WO-4B — Require actual multi-view surfel evidence

1. Fix initial surfel support so multiple pixels from one image do not satisfy
   a multi-view requirement. Retain distinct-view identities/counts and bounded
   accumulation; maintain compatibility with downstream raw/reintegration data.
2. Exercise one-view rejection, two-view acceptance, correlated evidence,
   large-error reporting, and missing reference behavior. Run one matched
   retained raw reconstruction comparison and report accuracy/coverage tradeoffs.

Primary files: build_consensus_fusion.py, evaluate_mapanything_prior_variants.py,
test_consensus_fusion.py, evaluation tests and downstream provenance consumers.

## Improvement 5: Visibility-aware measured surfaces and unified presentation

### WO-5A — Fuse compatible 3D observations, preserve complementary surfaces

1. Replace whole-column X/Z suppression with conservative 3D compatibility and
   visibility decisions where evidence supports them. Preserve separate surfaces
   at different heights and source-room/view provenance. Ambiguous seams remain
   uncertain; do not blend contradictory walls or accepted/rejected worlds.
2. Reuse retained rays/depth/poses, distinct-view support, normals/view direction,
   and registration uncertainty. Keep finite memory/chunking; no unconditional
   large-cloud copy or recursive artifact scan.
   Carry WO-4's uncalibrated evidence semantics through reintegration; remove
   the downstream unconditional correlated-agreement weight bonus as well.
   Apply the same policy in `extend_pcf_multiroom_with_connector.py`, which
   currently repeats column suppression when adding a third room.
3. Keep provenance for excluded observations and quantify retained/complementary
   coverage. Different ownership does not itself mean geometric contradiction.

### WO-5B — Derive surfaces and views from the same accepted geometry

1. Produce a measured surface artifact with observed/unknown/uncertain support,
   preventing unsupported mesh triangles across openings or incompatible joins.
   Reuse current mesh tools and visibility support before considering a new
   meshing algorithm. Keep authored scene semantics separate from measurements.
2. Connect measured surface and uncertainty to the existing review/scene paths
   using WO-1 frame identity. Do not promote a rejected assembly or introduce
   alternate person-position transforms. Menon, BEV, and reconstruction views
   must consume the same declared frame relationship.
3. Exercise distinct-height complementarity, contradictory overlap, unsupported
   openings, and one retained multiroom sample. Review concrete before/after
   geometry and coverage, not only successful GLB generation.

Primary files: reintegrate_pcf_rooms.py, build_pcf_review_surface_mesh.py,
related publication/rendering tools and Menon measured-layer consumer.

## Assignment and integration sequence

1. First wave: frame contracts/renderer (WO-1); capture and VIO (WO-2A/B);
   confidence/evaluation and support (WO-4).
2. Coordinator reviews each implementation and requests corrections. Work
   continues in independent lanes while waiting for phone information or builds.
3. Subsequent wave: loops/refinement (WO-3), measured surface fusion (WO-5),
   and static calibration work (WO-2C). Assign precise file ownership before edits.
4. Integrate through normal application entrypoints. Use real stored captures
   and a real synchronized benchmark; no synthetic result substitutes for
   metric improvement evidence.
5. A final Luna integration pass updates current docs, decisions, upgrade
   history, and this work-order status under coordinator review. Run docs
   consistency, diff whitespace checks, affected tests, and bounded direct smokes.
6. Coordinator compares before/after outputs and dispatches corrective work
   when a change regresses coverage, alignment, observability, or direct-consumer
   behavior. Stop expanding validation when the affected behavior is proven.

## Acceptance and evidence dependencies

Implementation acceptance requires a connected usable feature, focused checks,
direct-consumer exercise, and clear failure behavior. Reconstruction improvement
requires matched real-data results. A home-global calibration additionally needs
accepted cross-room geometry and validated metric scale. Phone recording/VIO
additionally needs the actual phone and synchronized sensor evidence. These
criteria must be reported separately rather than marking unavailable evidence
complete. No rejected current artifact is promoted to satisfy a checklist.

Each agent writes a concise result under plans/reconstruction_work_orders/
with changed files, exact validation commands/results, before/after findings,
and unresolved dependencies. The coordinator maintains the status below.

| Work order | Status | Result |
| --- | --- | --- |
| WO-1A/B | reviewed | Normal build with existing nonidentity floor correction and two numeric camera edges passes; point/camera transforms agree, one target presentation, exact digests. Actual graph producer composes proven endpoint conversions and feeds the normal binding importer |
| WO-2A/B | reviewed | Real monocular OpenVINS: 313 initialized states from 500 frames, camera-origin SE(3) ATE RMSE 0.0690 m. Actual HTTP import/preparation/VIO produced 30 selected poses and 29 consumed relative constraints; Fold calibration and capture remain unavailable |
| WO-2C | reviewed | Measured PnP/temporal acceptance, exact backup, E/PoseV1 consistency, actual rectified-image provenance, replacement depth/static revision and normal builder/catalog/manager consumers exercised. Retained Living and Family inputs admit no replacement |
| WO-2D | implementation/deployment complete; review evidence complete | Deployed HTTP/HTTPS Room Walk path is live and hostname-verified. Chrome loads the secure origin and shows the sensor panel; the desktop has no rear camera. The corrected duration rerun imported `root12s-browser-smoke.tar` as scan `20260905-180446-7df21485`, retained 2 accel + 2 gyro rows, and reached ready with 15 adaptive views from 48 candidates. Prepared state and frame manifest report 11.967s with `duration_source=capture_import.video.encoded_duration_s`; Chrome displays 12s rounded and both sensor counts. All prepared `capture_time_ns` and `source_frame_index` values remain null. Narrow duration regression passed 2 tests; live initiate-VIO correctly returned HTTP 409; both synthetic scans were deleted via API 204 after review and evidence retained. Fold 8 Ultra capture remains unverified |
| WO-2E | implemented and directly verified | Paired static RTSP/tracking capture is live in Room Walk. Actual Living Room video decoded at 1080p/30 fps, with 26 consecutive tracking/world pairs, 11 clock probes, one marker and complete runtime/calibration evidence. Browser-controller lifecycle and synthetic phone-bundle import/retry/conflict paths passed; Chrome exposed saved paired artifacts. Test walks were removed after evidence retention. Physical Fold/occupied timing and pose accuracy remain unverified. See `plans/reconstruction_work_orders/WO-2E-static-paired-capture.md` |
| WO-3A/B | reviewed | Two actual 48-view normal PCF runs include fresh MapAnything conditioning. Verified visual refinement improves consensus p80 reprojection 0.10454 to 0.09951 m. Larger VIO scale changes now invoke existing fixed-scale alignment on the actual refined carrier before conditioning; recorded producer and PCF consumer checks passed |
| WO-4A/B | reviewed | Matched 48-view geometry reviewed; raw validity unchanged and support classes retain 95.98% X/Z coverage. Independent evaluation now requires exact target revision and registration fingerprint |
| WO-5A/B | reviewed | Real retained RGB-D rays distinguish observed, contradictory and unknown geometry. Recorded 1.20M-point assembly and 50,007-triangle mesh reviewed in matching height and side slices; complementary structure remains available and uncertainty is visible |
| Integration and final review | complete | Native DS9.1 restored and phone service reloaded. Authenticated health ready, 102 tracking/snapshot pairs received, and actual phone HTTP/VIO/trajectory path passed. Live sample was empty; occupied cross-source fixtures exercise frame presentation |

### Matched normal PCF evidence

The coordinator ran `run_pcf_review_candidate` twice on the same 48 retained
Living Room images, DA3 outputs, and previously passed static-world alignment.
Both runs perform fresh MapAnything inference and the normal consensus and
static-world evaluation. The input is an isolated bounded fixture preserving
exact source image/manifest hashes; active scans and calibration are untouched.
Outputs are under `$NOESIS_RECONSTRUCTION_WORK_ROOT/wo3_normal_pcf_run_v1`
and `wo3_normal_pcf_run_v2`, with the comparison in
`wo3_normal_pcf_before_after.json`.

The refined run uses nine geometrically verified nonadjacent constraints and
no VIO input, with maximum camera changes of 0.1068 m and 2.936 degrees.
Six verified temporal-withheld edges remain outside fitting. Their translation
p80 increases by 0.000714 m (0.226805 to 0.227520 m), within the declared
0.02 m no-regression tolerance tied to half the 4 cm surfel voxel. Absolute
error and deformation gates remain active. This is internal withheld-constraint
evidence, not an independent household accuracy measurement.

| Same-data measure | Control | Refined |
| --- | ---: | ---: |
| Valid fused depth fraction | 0.88677 | 0.89141 |
| Multi-view accepted surfels | 83,817 | 85,820 |
| Internal reprojection median / p80 (m) | 0.04310 / 0.10454 | 0.04140 / 0.09951 |
| Internal even/odd errors above 2 m | 0.5585% | 0.4875% |
| Internal even/odd pixel coverage | 0.39434 | 0.39566 |
| Static-camera-visible median residual (m) | 0.07891 | 0.07754 |
| Static-camera-visible overlap within 0.20 m | 0.92447 | 0.93381 |
| Full static target-to-phone median residual (m) | 0.13443 | 0.14615 |
| Fixed-camera overlapping depth-delta median / p80 (m) | 0.23811 / 0.81403 | 0.26414 / 0.84999 |
| Visible static structure plane residual p80 (m) | 0.10628 | 0.10907 |

The geometry remains coherent in the generated review images. These are
useful, modest consistency improvements with mixed static depth/structure results;
they do not establish physical scale or justify replacing current manual
extrinsics. Independently measured household validation was not supplied.

### Recorded surface review and calibration boundary

The coordinator inspected the actual WO-5 merged point output and generated
mesh using matching top-down height slices and vertical side slices. The
review output preserves complementary surfaces at different heights instead
of deleting every point in a shared X/Z column. Exact counts are 528,939
observed, 666,719 uncertain, and 38 unknown points. The mesh retains support
classes through named geometry and visible colors; ray-contradictory triangles
are removed, and unresolved geometry remains visible for review. These
artifacts retain the original assembly's review-only authority.

Both retained 48-view static-camera localization runs admitted zero candidates.
The Family diagnostic reached solved PnP in four views but their feature
coverage was concentrated in only two or three image-grid bins. Relaxing that
gate would not provide a reliable extrinsics set. A wider walk with distributed
static-camera features, early/late revisits, calibrated image provenance, and
independent scale evidence is the remaining data requirement. Existing manual
extrinsics are unchanged. The replacement path has been exercised through
exact backup, E/PoseV1, regenerated static depth and revision, normal Scene Prior
builder/catalog, and CalibrationManager in an isolated fixture.

### Phone service and live transport integration

A calibrated public EuRoC recording was imported through the actual HTTP
sensor-bundle endpoint. Normal preparation selected 39 frames; the configured
native OpenVINS executable returned 30 initialized selected camera poses.
The actual trajectory consumer accepted 29 consecutive relative constraints
with exact service-produced index/hash frame IDs and acquisition timestamps,
covariance present, and no skipped reset/gap. The browser showed completed
metric VIO. An uncalibrated sensor bundle was also preserved successfully but
its metric VIO request correctly returned HTTP 409. The two temporary test
scans were removed after evidence was saved; the four original scans remain.

This service exercise uses public benchmark imagery and calibration. It is
not a recording from the Fold 8 Ultra. Evidence is retained under
`$NOESIS_RECONSTRUCTION_WORK_ROOT/benchmarks/MH_01_easy_mono_subset/http_public_euroc_e2e_20260905/`.

The native DS9.1 runtime was restored after the two GPU reconstruction runs.
Authenticated health was ready and an 18-second WebSocket sample delivered
102 tracking/world-snapshot pairs. The live calibration bundle was consumed
successfully by Menon's actual CalibrationManager for all three cameras.
The sample was empty; occupied cross-source fusion and position/velocity/
covariance presentation were exercised by the bounded producer/consumer
fixtures. No active camera calibration or rejected room join was changed.

### Final correction work order: align the actual rescaled geometry

Status: completed and reviewed. The recorded 48-view source was deliberately
scaled by 1.03 and passed the existing alignment gates in 22.34 seconds. A
separate PCF consumer smoke injected that precomputed carrier, ran the actual
alignment callback, and verified that the conditioning command received the
new raw directory and the newly computed rigid world transform. That smoke
stopped at the GPU subprocess boundary; the earlier paired normal PCF runs
provide the full inference/fusion evidence. Source paths remain stable, with
no staging promotion or renamed provenance. Seven trajectory checks and two
alignment checks passed in the coordinator's focused runs.

Review found that an exported matrix and a passed report about earlier geometry
cannot establish alignment after a new inertial scale correction. Complete the
existing producer connection rather than introducing another acceptance format:

1. The capture agent adds explicit optional source-raw and output-manifest
   inputs to the normal alignment loader/runner, preserving existing defaults,
   fixed scale of one, and all geometric gates. Record the exact consumed source
   artifacts and return paths for the requested output directory.
2. The trajectory agent materializes a proposed refined carrier, including
   consistently scaled depth, poses and world points. For a scale change above
   two percent, the normal PCF caller runs existing world alignment on that
   actual carrier before conditioning. Only its passed result supplies the new
   world transform; an old report or a caller-labelled matrix is insufficient.
3. Alignment failure must explicitly reject that refined review or use the
   original geometry with a recorded VIO rejection. It must never silently
   combine rescaled points with the old mapping. Keep all outputs in the
   existing isolated run directory.
4. Exercise the changed raw producer, existing alignment loader, and PCF
   transform consumer with focused checks. Confirm nonunit scaling reaches
   depth and camera geometry together, and a failed alignment cannot authorize
   conditioning. The already exercised RGB path requires no further GPU run.
