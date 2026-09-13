# Room reconstruction fitting

This is the reusable working procedure for fitting a phone reconstruction to
an independent static-camera reference. It records experiments and admission
evidence separately. Living Room is the primary investigation, with Kitchen
and Family Room replays recorded below. Each room needs its own evidence.

## Inputs and authority

- Name the phone scan, camera, prepared-view manifest, reconstruction provider,
  static reference revision, and calibration/frame-binding revision.
- A paired walk uses its finalized static recording and captured calibration.
  An unpaired walk uses its explicitly validated saved reference. Preserve the
  source videos and original reconstruction outputs.
- Keep static geometry independent of phone reconstruction. Apply the captured
  calibration-to-world binding once. Tracking positions are not calibration.
- Save experiments outside the scan's accepted alignment directory. Record the
  input manifests, code identity, command, parameter choices, and measured
  results. A passing experiment does not change live Noesis world authority.

## Fitting sequence

1. Reproduce the current result with unchanged inputs and quality gates. Save
   visual-anchor support, floor orientation, candidate transforms, visible and
   full-cloud residuals, overlap, and failure reasons.
2. Test visibility-aware refinement against the same fixed-camera visibility
   and occlusion model used for validation. Keep gravity, scale, transform
   bounds, and acceptance thresholds unchanged. Compare support as well as
   residuals so hiding a difficult surface cannot count as improvement.
3. Inspect residuals by wall, image region, and phone view/window. Distinguish
   global pose, floor orientation, metric scale, local surface distortion,
   moving objects, and correspondence failures. Repeated static predictions
   measure repeatability; they do not establish absolute metric accuracy.
4. If the fit remains inconsistent, compare an independent reconstruction on
   the same prepared phone views before changing frame selection or models.
   Use the established MapAnything/DA3 consistency workflow and preserve
   disagreements. Do not condition the independent static reference on phone
   geometry merely to lower its validation residual.
5. For unresolved scale/calibration ambiguity, request a small set of measured
   room dimensions or surveyed image correspondences. Do not invent a room
   offset, alter camera calibration from one fit, or relax quality thresholds.
6. Once an implementation change is supported, run focused geometric tests,
   replay the actual producer and direct consumer, inspect served review
   artifacts, and record remaining limitations. Reuse the same procedure on
   another room with that room's own captured camera binding and evidence.

## Experiment ledger

Evidence root for this investigation:
`$FIT_EVIDENCE_ROOT/fitting_process_20260906/`, where `FIT_EVIDENCE_ROOT` is the
chosen local reconstruction-work directory on durable storage.

| Experiment | Inputs and change | Result | Decision |
| --- | --- | --- | --- |
| Paired reference baseline | Living Room scan `20260906-050646-107f244c`; static revision `paired_static_reference_2811d75e0520af6c`; MapAnything phone output, 256 views | Static reference: 99,198 points; visual consensus: 29 phone views; visible vertical median: 0.122825 m against 0.10 m limit | Rejected; original phone output and static reference retained |
| Static repeatability | Six recorded observations on their common valid pixel set | Median temporal depth absolute deviation: 0.0134 m across 74,747 pixels | Repeatability evidence only; systematic depth/calibration bias remains possible |
| Visibility-aware refinement | Same paired inputs, transform bounds, and quality gates | Gated vertical median: 0.085703 m; all 15 checks pass. Untrimmed median over the same 6,400 baseline-visible points improves from 0.164513 to 0.142606 m | Implemented; no scale, gravity, calibration, or threshold change |
| Spatial holdout | Withhold 914 source points in fixed image regions from fitting | Gated median: 0.094713 m. Withheld-point untrimmed median improves from 0.197122 to 0.152223 m relative to baseline | Supports visibility change beyond its fitting points; no surveyed-accuracy claim |
| Image-balanced holdout | Equalize image-cell weights in addition to visibility | Gated median: 0.112395 m | Rejected; no additional weighting introduced |
| Static floor/plane diagnostic | Same six static arrays; no phone geometry | Inferred floor is roughly 0.17–0.24 m above declared floor. Final observation has lower confidence and a median depth ratio of 0.961 | Track model/metric uncertainty separately from relative wall-fit admission |
| Camera-height authority check | Captured runtime camera config versus captured calibration-to-world composition | Configured height is 2.60 m; `target_from_calibration * inverse(E)` implies 2.086880 m above the declared floor, a 0.513120 m discrepancy. Physical measurement hashes are absent | Neither value is an independent scale anchor; retain both facts and require a measured lens-center-to-floor distance before metric correction |
| Kitchen replay | Existing 48-view walk and validated saved reference; original input hashes match | All 15 checks pass; gated vertical median improves from 0.081716 to 0.054205 m | Confirms the same implementation works without a Living Room-specific parameter |
| Family Room replay | Existing 48-view walk and historical June static reference | Camera-forward fraction is zero; rejection occurs before refinement | Reference/calibration preflight blocks this old pair; obtain accepted matching reference evidence |
| Independent DA3 comparison | Identical 256 upright prepared views and timestamps, eight windows, no static conditioning | All internal window gates pass; paired fit fails the wall gate at 0.120824 m, with untrimmed comparable median 0.192566 m | Retain as disagreement evidence; standalone DA3 does not improve this fit |
| Initial consistency fusion | Same MapAnything/DA3 views, existing common-ray fusion and joint poses | 286,552 surfels; internal even-to-odd consistency median improves from 0.110020 m (MapAnything) to 0.067681 m, with coverage decreasing from 78.0% to 69.6%. Static wall residual is 0.072919 m, but candidate separation fails and RGB consensus has only two views | Rejected despite cleaner surfaces; investigate loss of RGB anchor support before another fit |
| Isolated RGB anchor | Original consensus geometry/poses/intrinsics; use only MapAnything RGB remapped to the existing common rays | RGB support increases from two solved views / 21 inliers to 35 solved views / 31 consensus views / 1,940 inliers; visual anchor passes | Implement single-reference RGB projection with explicit manifest provenance; retain all geometric gates |
| RGB-only full fit | Original consensus geometry with the explicit single-RGB anchor override | All 15 checks pass; wall residual 0.079614 m, untrimmed median 0.112638 m, visible source overlap 93.7%. Yaw refinement reaches its existing 5-degree bound | Supports the RGB correction; retain the transform bound and metric uncertainty |
| Rebuilt fusion geometry equivalence | Execute the normal fusion producer with single-reference RGB | All non-RGB arrays match across all 256 raw views; camera solution and non-color surfel arrays also match exactly. All 286,552 surfels are retained | Confirms the correction changes image/color evidence without changing fused geometry or depth admission |
| Rebuilt fusion through normal fitter | Standard saved corrected-fusion raw files and manifest, with no experimental anchor override | All 15 checks pass; wall residual 0.079048 m, untrimmed comparable median 0.113527 m; 30 RGB consensus views / 1,907 inliers | Working Living Room consensus fit retained in the evidence directory; the browser's MapAnything fit remains separately available |

The existing 0.10 m wall gate is a robust statistic: use neighbors within 0.80 m,
retain the lowest 75% of absolute plane residuals, then take their median.
Reports also expose the untrimmed median and p80 across all comparable points.
Passing the fit gate does not establish surveyed scale or correct a static
reference's floor/model bias.

The final consensus review artifacts are under
`consensus_single_rgb_paired_alignment/` in the evidence root. Its
`alignment_report.json`, `noesis_phone_comparison.glb`,
`fixed_camera_reprojection.jpg`, and `alignment_topdown.png` describe the
normal rebuilt result. The earlier `consensus_rgb_override_alignment/` was
an isolation experiment. Neither output changes live Noesis world authority.

### Metric measurement for a later session

Record the static camera's lens-center height above the actual floor, measurement
method, approximate uncertainty, date, and camera identity. Add two room spans
visible in the imagery, preferably on different axes. Keep these observations
separate from runtime configuration and reconstructed geometry. Use them to
test the captured calibration and depth scale; any accepted correction needs a
new reference revision and another fixed-scale phone fit. A height from a
configuration file alone is not a surveyed measurement.

## Reproduce a room fit

Run from the repository root with an unused output directory on durable storage:

```bash
python3 -m tools.mapanything_phone_scan.diagnose_alignment \
  --scan-dir "$SCAN_DIR" \
  --camera-id "$CAMERA_ID" \
  --target-revision "$STATIC_REFERENCE_DIR" \
  --calibration-path "$CAPTURED_CALIBRATION_FILE" \
  --output-dir "$FIT_RUN_DIR" \
  --save-snapshot
```

For a paired walk, resolve `STATIC_REFERENCE_DIR` and
`CAPTURED_CALIBRATION_FILE` from its saved `alignment.static_reference` paths
relative to the scan. For an unpaired walk, use its explicitly accepted target
revision/calibration pair. Do not carry either path from another room.

The command leaves scan state and accepted alignment artifacts untouched. It
stores `run_summary.json`, and after the fitting stage stores
`diagnostic_report.json` even when the quality gate rejects the result.
`--save-snapshot` additionally retains the bounded arrays in
`diagnostic_inputs.npz` for CPU-only refinement experiments. Passed replays
also produce the usual review artifacts inside the experiment directory.
An early input/calibration rejection retains its reason in `run_summary.json`.
Exit status is 0 for passed, 2 for rejected, and 1 for an execution error.

When comparing another reconstruction of the same prepared views, provide both
`--source-raw-root` and `--source-output-manifest`. This explicitly binds the
comparison inputs instead of silently loading the scan's original provider.

## Compare providers on one walk

Preserve the prepared-view manifest and require identical source-frame paths,
timestamps, and pixel transforms for both providers. Obtain exclusive GPU use
for the existing DA3 runner; retain its normal window gates and use
`anchor_image=None`. Save its outputs to a new experiment directory. A static
frame must not enter either phone reconstruction. This investigation retains
the runner and its settings in
`experiments/run_same_prepared_da3.py` under the evidence root, along with the
prepared-manifest hash and the complete run log.

After both independent reconstructions complete, run the established fusion
with explicit raw roots and an unused output directory:

```bash
python3 -m tools.mapanything_phone_scan.build_consensus_fusion "$SCAN_DIR" \
  --mapanything-raw "$MAPANYTHING_OUTPUT_DIR/raw" \
  --da3-raw "$DA3_OUTPUT_DIR/raw" \
  --output-dir "$CONSENSUS_OUTPUT_DIR"
```

Then replay each comparison against the same static reference and calibration:

```bash
python3 -m tools.mapanything_phone_scan.diagnose_alignment \
  --scan-dir "$SCAN_DIR" --camera-id "$CAMERA_ID" \
  --target-revision "$STATIC_REFERENCE_DIR" \
  --calibration-path "$CAPTURED_CALIBRATION_FILE" \
  --source-raw-root "$COMPARISON_OUTPUT_DIR/raw" \
  --source-output-manifest "$COMPARISON_OUTPUT_DIR/scan_outputs_manifest.json" \
  --output-dir "$COMPARISON_FIT_DIR" --save-snapshot
```

Compare wall residuals, source support, target coverage, withheld observations,
and visible artifacts. A cleaner cloud or better internal consistency does
not override a rejected static fit. Retain the fusion's provider-selection and
disagreement diagnostics. Passing or failing one walk does not establish a
global provider preference for other rooms.

If fusion loses RGB anchor support that its providers had, inspect the actual
serialized RGB and its associated intrinsics. The raw image must represent one
camera projection. Averaging differently warped versions of a frame can create
ghosted landmarks even when the fused geometry is useful. Diagnose RGB support
separately before changing geometry or weakening candidate-separation gates.

For each completed experiment, retain its report and record the exact reference
revision. Do not transfer a numerical correction from Living Room to Kitchen or
Family Room.

## After fitting: verified trajectory and measured surfaces

The 2026-09-06 follow-through uses the same 256-view Living Room capture and
preserves the passed baseline consensus. Its evidence directory is
`$FIT_EVIDENCE_ROOT/nonintrinsic_completion_20260906/`. Browser IMU data remains
ineligible for metric VIO; these experiments use visual/depth constraints and
keep physical scale unchanged.

Predeclare complete temporal intervals before retrieving matches. The measured
run excludes `[48,64)`, `[112,128)`, `[176,192)`, and `[240,256)` from visual
constraint fitting. Those images still contributed to reconstruction and
odometry, so the result measures internal withheld-constraint consistency.
There is no forced first-to-last loop.

Matching must use `model_rgb` on the retained depth/intrinsics grid. Resizing
prepared RGB can sample unrelated depth pixels after a crop or common-ray
warp. Prepared-image hashes still establish capture identity. Refinement
preserves the source coordinate-frame label, hashes its source views, and
rebuilds poses and world points together in a separate directory.

```bash
python3 -m tools.mapanything_phone_scan.trajectory_refinement \
  --scan-dir "$SCAN_DIR" \
  --source-raw-root "$CONSENSUS_OUTPUT_DIR/raw" \
  --output-dir "$TRAJECTORY_RUN_DIR" \
  --withheld-range 48 64 --withheld-range 112 128 \
  --withheld-range 176 192 --withheld-range 240 256
```

For another capture, select valid temporal ranges for its actual view count
before examining retrieval results. The ordinary finite retrieval, geometry,
deformation, and holdout thresholds remain unchanged. An unavailable or failed
holdout prevents admission. The CLI returns 0 for a materialized admitted
refinement and 2 when no refinement is admitted.

A post-fusion pose change requires fresh fusion admission and surfel support.
Use the original independent model outputs and the passed trajectory report:

```bash
python3 -m tools.mapanything_phone_scan.build_consensus_fusion "$SCAN_DIR" \
  --mapanything-raw "$MAPANYTHING_OUTPUT_DIR/raw" \
  --da3-raw "$DA3_OUTPUT_DIR/raw" \
  --trajectory-refinement-report "$TRAJECTORY_RUN_DIR/trajectory_refinement_report.json" \
  --output-dir "$REFINED_CONSENSUS_OUTPUT_DIR"
```

This route accepts only a passed pose-only consensus refinement with verified
temporal holdouts. It checks the prepared capture, source manifest, raw-view
hashes, camera solution, original poses, common rays, and origin gauge. It
recomputes multiview consistency, depth selection, evidence weights, raw world
points, and distinct-view surfel support. It does not load old surfels under
new poses. Run the diagnostic fixed-scale alignment again using the rebuilt
raw directory and output manifest before meshing it.

### Follow-through results

| Experiment | Trajectory evidence | Static-reference result | Disposition |
| --- | --- | --- | --- |
| DA3 initial refinement replay | Six fitted and seven withheld constraints using the prior resized-capture matcher; withheld translation p80 0.036829 to 0.043305 m, within the unchanged 0.02 m no-regression band | Wall statistic 0.117090 m, above 0.10 m | Rejected; retained as the pre-correction diagnostic |
| DA3 refinement with retained RGB projection | Six fitted and seven withheld constraints; withheld translation p80 0.042445 to 0.043257 m; maximum change 0.183445 m / 5.501879 degrees; scale 1 | Wall statistic 0.119362 m, above 0.10 m; other 14 checks pass | Current CLI producer and fitter exercised; standalone DA3 remains rejected |
| Consensus visual refinement | Two fitted and eight withheld constraints; withheld translation p80 improves from 0.110095 to 0.098891 m; maximum change 0.227101 m / 4.481921 degrees; scale 1 | All 15 checks pass; wall statistic 0.086279 m versus baseline 0.079048 m | Retain the trajectory/static-fit tradeoff; recompute fusion before using the surface |
| Full consensus rebuild with refined poses | Same original providers, rays, metric scale, and fusion gates; all consistency and support recomputed | All 15 checks pass at 0.085710 m; untrimmed visible median 0.118967 m versus baseline 0.113527 m | Retain as a comparison; baseline remains preferred |
| Alternate DA3 window boundaries | Identical 256 prepared images and settings except overlap 16 to 20 views; nine proposed windows instead of eight | Window 5 fails the existing camera-position registration gate: p80 0.556 m, translation 0.328 m, relative scale ratio 0.874 | Rejected; keep original completed output and window settings |

A lower internal residual does not establish absolute accuracy. Keep coverage,
static residuals, and visible geometry beside the trajectory metrics. Retain a
passing baseline when a new candidate has a mixed result.

The full rebuild retains 298,509 multi-view surfels versus 286,552 before
refinement. More surfels are not proof of better geometry. Internal even-to-odd
depth median/p80 changes from 0.067681/0.195464 m to 0.075350/0.212817 m,
while corresponding coverage changes from 69.576% to 69.498%. The fused valid
depth fraction is nearly unchanged (64.627% to 64.601%). Together with the
static-reference residual, this supports retaining the original consensus as
the preferred surface review while preserving the improved trajectory
holdouts as separate evidence.

### Measured surface generation

Use the passed source-specific alignment with the matching consensus output.
The single-room adapter checks the retained raw-view identity, applies the
accepted review transform once, and uses the existing per-owner surface mesh
builder. Single-view surfels stay in a separate withheld-point artifact and
cannot become accepted mesh triangles. Full-height output uses the measured
cloud extent; the optional cutaway is a display choice.

```bash
python3 -m tools.mapanything_phone_scan.build_single_room_review_surface \
  --consensus-dir "$CONSENSUS_OUTPUT_DIR" \
  --alignment-dir "$COMPARISON_FIT_DIR" \
  --room-name living-room --owner 3 \
  --ceiling-mode full_height \
  --output-dir "$ROOM_SURFACE_OUTPUT_DIR"
```

The existing owner schema uses Family Room 1, Kitchen 2, Living Room 3, and
connector 4. A room/owner must match its alignment camera; it cannot relabel
another room's frame. For a separate cutaway output, use `--ceiling-mode
cutaway --ceiling-cutaway-m 1.85` and another unused output directory.
The normal mesh builder retains its one-worker default, support-distance
checks, and source RGB-D ray checks. Holes or sparse patches remain visible
when the evidence does not support triangles.

A Living Room artifact alone does not satisfy Menon's whole-home review
contract, which also requires exact scene/calibration bindings and camera
anchors for the assembly. Keep the mesh available as a room review; do not
invent Family Room gauge or cross-room camera markers to activate it.

### Evidence still needed outside software execution

The complete static-camera pose replay now also matches retained `model_rgb`
to its corresponding raw depth grid. It uses the passed consensus alignment
to put phone points in the bound static world before fitting. Comparing a
candidate pose with the configured camera requires
`world_from_calibration * inverse(camera_from_calibration)`; comparing it
with the raw inverse extrinsic would mix frames.

With the same four predeclared temporal holdouts, the corrected replay solves
35 of the 256 phone views. Only views 0, 4, and 237 meet the fitting gates,
below the unchanged minimum of four. Thirty-two solved views fail spatial
support; eight solved views are excluded from fitting, all from `[240,256)`.
No solved view supports the other three withheld intervals. These counts can
overlap because a view can fail more than one check. The run stops before an
aggregate pose or holdout acceptance is produced.

An insufficient-view rejection now saves
`static_camera_anchor_failure.json`, including exact supplied-input hashes,
the configured minimum, fitted and excluded view counts, per-view poses,
reprojection statistics, and rejection reasons. Private correspondence arrays
are omitted from this bounded report. The recorded run is under
`registration_pose_review/living_complete_pose_pnp/` in the follow-through
evidence directory. The preliminary zero-view result used an unaligned source
frame and is retained as a separate execution diagnostic, not a reconstruction
quality result.

The original Kitchen/Family source registration reports
`insufficient_cross_session_overlap`, with a failed solution and failed held-out
observation/temporal support. Its later floor-constrained reintegration is a
validated review candidate with `accepted_for_canonical_use=false`. The separate
Living/Kitchen bridge also fails observation, deformation, and all four temporal
holdout gates. These reports are distinct; neither provides an accepted edge
for a common home frame.

Retain the existing camera-specific frame bindings until an accepted connector
supplies the missing room-to-home geometry. A new connector should include
sharp translated views of stable structure inside both rooms, continuous
parallax through the doorway, and a return visit that can be withheld as a whole
temporal segment. The two endpoints and the doorway should each have multiple
spatially separated observations. A corridor transit with support at only one
endpoint does not resolve the failed holdouts. The new Living Room-only walk
cannot supply a Kitchen or Family Room endpoint that it did not capture.

Independent lens-center height and room-span measurements remain necessary to
diagnose physical scale and the manual static-camera poses. The browser capture
also still lacks the camera/IMU intrinsic, extrinsic, timing, and noise evidence
needed for metric VIO. Intrinsic work was explicitly excluded from this pass.
No physical measurements were invented, no raw static extrinsics were replaced,
and no camera bindings were relabeled as one accepted household frame.

### Surface comparison on the retained walk

With the same meshing parameters and ray gates, the baseline full-height mesh
contains 3,331 triangles and its 1.85 m cutaway contains 1,997. The rebuilt
refined result contains 9,007 and 6,366 respectively. Both preserve withheld
single-view points as separate artifacts: 63,447 for baseline and 65,653 for
refined. Minimum/median accepted surfel support stays at 2/4 distinct views.
The full-height limit comes from the retained point-cloud extent, not an
independently measured ceiling height.

Matched-bound top and side views are saved under
`measured_surface_refined/{full_height,cutaway_1p85}/baseline_refined_comparison.png`.
The blue mesh patches show retained triangles; orange points are single-view
withheld evidence. The larger refined mesh is useful comparison evidence,
while its higher static and internal depth residuals prevent claiming a
uniform improvement. Confidence remains explicitly uncalibrated. Unknown and
contradictory source-point counts are marked unassessed when the adapter has
not measured them; the mesh builder separately reports its actual ray checks.

### Completion boundary for the original workstreams

| Workstream | Completed using existing evidence | Remaining external evidence |
| --- | --- | --- |
| Shared metric home frame | Reviewed exact Kitchen/Family and Living/Kitchen registrations; preserved rejected edges and current room bindings | New connector observations with both room endpoints and passed temporal holdouts |
| Capture and static-camera poses | Ran the complete-pose localizer against the paired reference, corrected RGB/depth projection, and retained the failed support report | More spatially distributed static-visible views and independent lens-height/room-span measurements; intrinsic and metric-VIO work remains deferred |
| Verified loops and trajectory | Ran refinement on all 256 views, evaluated predeclared temporal holdouts, rebuilt consensus support, and repeated the static fit | No additional software dependency for this comparison; a new capture may provide stronger revisit constraints |
| Confidence and evaluation | Compared static residuals, withheld trajectory constraints, internal depth errors, coverage, and alternate DA3 window boundaries | Independent household measurements to establish absolute accuracy and calibrate reliability |
| Measured surfaces and presentation | Built full-height and cutaway meshes for baseline and refined consensus, preserving single-view evidence separately; checked the existing Menon consumer contract | Accepted common-frame assembly and exact scene/calibration/camera bindings before whole-home presentation |

Forty-three focused trajectory, fusion, surface-adapter, and static-localizer
tests pass. The actual refinement-to-fusion-to-static-fit-to-mesh path was
exercised on the retained capture, including rejected alternatives. The phone
service reload passed its health check and still serves the existing passing
Living Room alignment. These results do not admit calibrated VIO, replace
static calibration, or publish a household world frame.
