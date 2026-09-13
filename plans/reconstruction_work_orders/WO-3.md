# WO-3: verified trajectory refinement and world registration

Status: implementation and final review complete, including visual refinement,
graph-to-world endpoint integration, and direct post-scale world alignment.
The work orders and remaining household evidence dependencies are recorded in
the [official plan](../reconstruction_world_unification_20260904.md).

## Implemented behavior

The normal `run_pcf_review_candidate` path runs bounded trajectory refinement
before fresh DA3-conditioned MapAnything inference. It retrieves at most 64
nonadjacent pairs with an eight-view minimum gap, then verifies actual SIFT
matches, two-dimensional geometric support and spatial distribution, retained
depth correspondences, and robust rigid three-dimensional registration.
Visual loop constraints touching the withheld temporal ranges remain excluded
from fitting. Retained DA3 odometry and any VIO relative measurements still
contain those views, so this is internal constraint validation. Rejected matches
carry explicit reasons. Merely ending a walk no longer creates a loop-to-origin edge.

DA3 remains the single pose carrier. Refinement rebuilds camera poses and world
points together; an accepted physical scale also scales camera-local depth and
visual translation constraints. Conditioned MapAnything and DA3 are correlated
evidence, so their agreement cannot count as two independent pose estimates.
Output manifests record which trajectory report produced the retained raw data.

The VIO consumer uses the actual persisted OpenVINS result from the phone
service. Prepared frame identity is shared by preparation, VIO, and trajectory
loading as `prepared:<index>:<image-sha256>`; the loader verifies image bytes.
Camera-relative constraints require exact acquisition timestamps, consecutive
selected frames, the same initialized segment, and valid covariance. Absolute
OpenVINS world yaw/origin is not treated as the household coordinate frame.
Relative covariance uses the declared camera-origin pose convention and a
conservative bound when state cross-correlation is unavailable.

The multianchor graph producer now emits its exact source-to-target transform
after composing proven endpoint conversions in the correct order. Intermediate
reconstruction frame identities do not masquerade as physical calibration
fingerprints. The final Scene Prior binding still requires the real physical
calibration provenance, passed graph holdouts, matching frame revisions,
transform digest, and floor planes. Missing conversion evidence remains
review-only.

## Matched normal application comparison

The coordinator ran the ordinary PCF function twice on the same 48 retained
Living Room images, DA3 raw views, and passed static-world alignment. Both runs
include fresh MapAnything inference, ordinary consensus fusion, and static
evaluation. No inference function was mocked. Heavy outputs stayed in the
configured large-storage directory, and the native runtime was restored after
the GPU runs.

The bounded input is `wo3_normal_pcf_input_v1`; its
`recorded_smoke_input.json` records exact source/target hashes. Its source is
the retained `20260802-162254-8bcc7dd7` scan and the passed alignment under
`data/mapanything_phone_scans/landscape_da3_alignment_validation/alignment`.
The two output directories are `wo3_normal_pcf_run_v1` and
`wo3_normal_pcf_run_v2`, under `$NOESIS_RECONSTRUCTION_WORK_ROOT`.
The comparison is `wo3_normal_pcf_before_after.json`. These normal runs
supersede the earlier diagnostic comparison that reused conditioned
MapAnything arrays after changing only the DA3 carrier.

The refined run admitted nine nonadjacent visual constraints and no VIO input.
Maximum camera change was 0.1068 m and 2.936 degrees. Six verified temporal
constraints in the excluded range `[34, 40)` remained outside fitting. Their
translation residual p80 was 0.226805 m before and 0.227520 m after: a
0.714 mm increase, within the declared 2 cm tolerance (half the 4 cm surfel
voxel). Absolute residual and deformation gates remain in force. This
temporal comparison is internal evidence, not independent physical accuracy.

| Same-data measure | Control | Refined |
| --- | ---: | ---: |
| Valid fused depth fraction | 0.886771 | 0.891409 |
| Multi-view accepted surfels | 83,817 | 85,820 |
| Internal reprojection median / p80 (m) | 0.04310 / 0.10454 | 0.04140 / 0.09951 |
| Internal even/odd depth median / p80 (m) | 0.06639 / 0.19150 | 0.06723 / 0.18997 |
| Internal even/odd errors above 2 m | 0.5585% | 0.4875% |
| Internal even/odd pixel coverage | 0.39434 | 0.39566 |
| Visible static-cloud median residual (m) | 0.07891 | 0.07754 |
| Visible static-cloud overlap within 0.20 m | 0.92447 | 0.93381 |
| Full static target-to-phone median residual (m) | 0.13443 | 0.14615 |
| Fixed-camera depth-delta median / p80 (m) | 0.23811 / 0.81403 | 0.26414 / 0.84999 |
| Visible static structure plane residual p80 (m) | 0.10628 | 0.10907 |
| Whole PCF elapsed time (s) | 194.43 | 220.71 |

The same-data results show modest improvements in internal consistency and
visible-cloud overlap, with mixed static depth/structure residuals. The
coordinator inspected the generated geometry and found its room structure
coherent. There is no basis here to claim better absolute household dimensions
or replace the manual extrinsics. The independent evaluation correctly reports
`not_supplied`; no withheld physical household measurements were available.

Earlier CPU experiments using different withheld temporal ranges remain under
`wo3_landscape_trajectory_refinement_v3` and
`wo3_landscape_trajectory_refinement_withheld_mid_v2`. They exercise sensitivity
to which visits are excluded; they are not an independent accuracy benchmark.

## Direct VIO and frame-consumer evidence

Actual HTTP sensor import, normal preparation, and native OpenVINS produced
39 selected frames and 30 initialized selected poses on public EuRoC imagery.
The trajectory consumer accepted 29 exact consecutive relative constraints,
with zero skipped reset/gap pairs. Preserved service output and selected image
bytes are under
`$NOESIS_RECONSTRUCTION_WORK_ROOT/benchmarks/MH_01_easy_mono_subset/http_public_euroc_e2e_20260905/`.
This confirms producer/consumer wiring; it is not Fold 8 Ultra field evidence.

The graph producer's focused tests exercise the real optimizer and the WO-1
binding builder with noncommuting rotations and translations. They also reject
stale manifests, arbitrary endpoint conversions, and missing accepted evidence.
The coordinator reran this focused set: five tests passed.

The localizer, source-image producer, confidence, capture/VIO, and trajectory
contracts have focused component coverage. The coordinator's final trajectory
and alignment checks passed seven and two tests respectively; the agent's
adjacent affected-contract set passed 43 tests.
The existing RGB reconstruction, camera calibration, and rejected house joins
are not changed merely by running this review path.

## Post-scale alignment correction

The final connection uses the existing fixed-scale alignment solver on the
actual materialized refined carrier before allowing a larger physical scale
change into conditioning. It does not accept a caller-labelled passed matrix
or reuse an alignment report from the earlier geometry. Depth, camera
translation, visual-edge translation and world points use the same physical
scale. Optimization deformation is measured against the scaled starting
trajectory, and internal visual holdouts use that same scale on both sides of
the comparison. Failed alignment leaves the proposed geometry unadmitted.

The coordinator exercised the actual raw materializer and normal alignment
runner on 48 recorded refined views with a deliberate 1.03 scale perturbation.
All existing alignment quality gates passed, including source/target support,
camera visibility, structural residual, and rigid transform checks. The run
took 22.34 seconds and retained an alignment matrix with unit linear scale.
Its exact inputs, normal report and resulting transform are under
`$NOESIS_RECONSTRUCTION_WORK_ROOT/wo3_scaled_alignment_direct_smoke_v1/`.
This is a controlled input-perturbation integration test, not an IMU-derived
household scale measurement. No GPU inference or active calibration was changed.

The coordinator also exercised the normal PCF callback using this precomputed
scaled carrier. Actual alignment ran again inside the caller, and the next
conditioning command received the new raw path and its newly computed world
transform. The GPU subprocess was intercepted at that boundary. This seam
check is distinct from the two full normal GPU runs above. Its result is
`wo3_scaled_pcf_consumer_smoke_v1/direct_pcf_consumer_result.json` under the
same work root. Proposed raw data and alignment provenance retain stable paths;
rejection leaves those inputs unadmitted without renaming or promoting them.
