# Phone-Walk Room Reconstruction

This LAN-only tool records a phone walk, prepares an adaptive set of views, and
runs either MapAnything or Depth Anything 3 (DA3). It also contains an offline
consensus workflow that combines both model outputs into a cleaner shared room
reconstruction.

The intended long-term room workflow is:

1. Capture a slow, overlapping phone walk to build the room reconstruction.
2. Run the same prepared views through MapAnything and DA3.
3. Build the consistency-gated consensus reconstruction from both outputs.
4. Register that reconstruction to the room's calibrated static-camera data.
5. Use the saved room transform to place live detections and tracks into the
   reconstructed room coordinate system.

The phone walk supplies coverage and room geometry. Static-camera runs supply
the fixed Noesis alignment and validation reference. A static frame is not
silently mixed into phone-only inference or fusion.

This tool saves review candidates. It does not automatically publish them into
the live native DS9.1 depth, tracking, floorplan, or virtual-twin contracts.

The [canonical end-to-end Prior-Conditioned Fusion (PCF) operating and handoff
runbook](../../docs/PCF_Workflow.md) carries the process from capture through
the sealed evidence bundle, immutable Scene Prior, runtime binding, and
dashboard verification. The fusion decision, validation basis, and algorithm
detail are in
[`Phone_Walk_Fusion_Reconstruction.md`](../../docs/Phone_Walk_Fusion_Reconstruction.md).

## Validated reference capture

The consensus result reviewed on 2026-08-09 used this scan:

```text
data/mapanything_phone_scans/20260801-112036-2d9225dd
```

It is a **portrait** capture:

```text
48 prepared views
1080 x 1920 pixels per prepared frame
```

A second stored capture is landscape and was validated on 2026-08-09 with the
same provider, fusion, static-world alignment, and diagnostic rules:

```text
data/mapanything_phone_scans/20260802-162254-8bcc7dd7
48 prepared views
1920 x 1080 pixels per prepared frame
```

Within a walk, consistent orientation, slow motion, repeated overlap, and
deliberate revisits matter more than rotating the phone midway through the
recording.

## Start the service

The deployed user service is:

```bash
systemctl --user status noesis-phone-scan.service
```

It is enabled at login and listens on port 8788. To run the launcher directly
from the repository root:

```bash
./tools/mapanything_phone_scan/run.sh
```

On first start, the launcher prepares an isolated Python environment under
`data/mapanything_phone_scan_runtime/`. It reuses the machine's CUDA-enabled
PyTorch installation and pins the official model dependencies used by this
tool.

The service prints the phone URL, normally resembling:

```text
http://192.168.x.x:8788
```

Connect the phone to the same trusted LAN, open the URL in Chrome, and permit
the local-network and camera prompts. The service has no user authentication;
do not forward its port through the router.

Alignment targets come from one validated Noesis scene release. The default
for this installation is
`data/virtual_twin/releases/home_rgbmesh_20260623T2158_v1.json`; another home
must set `NOESIS_PHONE_SCAN_ALIGNMENT_RELEASE` to its own validated release
manifest before starting the service. The browser lists only cameras whose
release revision, backend-world metadata, RGB keyframe, and calibration row are
present.

## Capture and single-provider workflow

1. Select **New Walk** to clear the current review and enter fresh-capture mode.
   That mode survives a browser refresh; it does not delete or overwrite an
   earlier walk.
2. Give the walk a room name, then record it or choose an existing video. The
   upload is automatically saved under
   `data/mapanything_phone_scans/<scan-id>/phone_walk.mp4`.
   Saved walks appear in the **Walks** list. Press and hold a walk to rename it;
   the new name is saved immediately without renaming or moving its asset
   directory.
3. The server analyzes a dense four-view-per-second candidate stream. It keeps
   frames that add viewpoint information, enforces temporal coverage, and
   inserts bridge views when adjacent selected frames lack reliable visual
   overlap. The default 256 selected-view limit is an emergency ceiling, not a
   requested frame count; preparation reports if it ever constrains a walk.
4. Selection combines relative sharpness, exposure, feature coverage, visual
   motion, and overlap. The manifest records every selected timestamp, reason,
   quality score, adjacent-view connectivity measurement, and any warnings.
5. Once `prepared_frames_manifest.json` is durable, choose **MapAnything** or
   **DA3** and select **Run reconstruction**.
6. The selected provider reconstructs every prepared frame. MapAnything uses
   one joint pass through 80 views. Larger adaptive sets are divided into
   80-view windows with 24 exact duplicate views between
   neighboring windows. Duplicate camera poses initialize a proper Sim(3),
   pixel-corresponding duplicate 3D surfaces robustly refine scale and
   translation, and both camera and dense-surface gates must pass before a
   window enters the saved reconstruction. DA3 uses the same fail-closed
   mechanism with 48-view windows and 16 exact duplicate views. Unique views
   from every accepted
   window are retained; review points are confidence weighted and averaged in
   3.5 cm voxels.
7. The tool saves a review GLB, camera trajectory, RGB, depth, confidence or
   validity data, masks, poses, intrinsics, metric scale, and raw NPZ arrays.
8. Choose the static camera physically installed in the scanned room, then
   select **Align to selected camera**. The target is saved with the walk before
   alignment starts. The tool estimates the phone floor, preserves gravity and
   metric scale, and registers room structure to that camera's validated room
   reconstruction. For an ordinary phone-only walk, shared RGB landmarks across
   multiple phone views and the fixed-camera keyframe choose the room pose by
   depth-backed PnP consensus; vertical geometry then performs only a bounded
   refinement. Source-fit metrics are evaluated over surfaces the single static
   camera could actually observe, while points hidden behind its measured depth
   remain recorded as global diagnostics instead of counting as alignment
   failures. Foreground disagreements are still scored. The tool also verifies
   that the selected camera faces its own target cloud and never rotates or
   rewrites the authoritative cloud or global Noesis calibration.
9. A weak or ambiguous registration fails its quality gate. After a passed DA3
   alignment, select **Generate PCF review** to run the canonical
   `da3_pose_sparse_depth` conditioned MapAnything stage, DA3-carried
   consistency fusion, and static-world evaluation over the same prepared
   views. The app auto-saves the PCF GLB, raw evidence, collaboration image,
   diagnostic layers, 2.5 cm point layers, evaluation metrics, and run log.
   On GPU-constrained installations,
   `NOESIS_PHONE_SCAN_PCF_PAUSE_APPLIANCE=1` makes this button explicitly pause
   the native Noesis/Menon appliance and restore it when the PCF job exits. The
   runtime lease and any restoration error are saved with the job.
10. A completed browser PCF remains a review candidate. The button never seals
    a room-scan bundle, builds or binds a Scene Prior, or changes live Noesis.

## Adding another video to an existing walk

After the first reconstruction completes, select **Add Video** to record another
pass or **Use saved video** to upload one already on the phone. The upload and
its prepared frames are immediately auto-saved below
`<scan-id>/supplements/<addition-id>/`; the original video, prepared views, and
reconstruction outputs are never overwritten.

An addition uses the provider selected for the original reconstruction. Before
using the GPU, the service searches the current reconstruction's retained RGB
views for strong overlap with the new video. It copies up to eight exact prior
views into the new inference batch as bridge views and reserves the remaining
view budget for the new pass. Static-camera images are not inserted into this
phone-to-phone bridge batch.

Publishing an added pass requires both checks below:

1. The duplicate bridge camera poses must agree on one stable metric Sim(3)
   transform from the new joint reconstruction into the original phone frame.
2. Independently matched RGB landmarks from the new views must solve against
   prior reconstructed 3D points with a camera pose consistent with that
   transform.

If either gate fails, the base reconstruction remains current and all uploaded
video and prepared frames remain available for review or retry. If both pass,
the service creates an immutable revision containing the joint inference,
append-to-base transform, source comparison, camera path, confidence-weighted
4 cm surfels, raw provenance counts, report, and manifest. New-only voxels need
support from at least two views; overlapping evidence is confidence weighted,
with the original reconstruction retaining greater authority. Later additions
may bridge through retained frames from any accepted earlier pass.

If the base walk is already registered to Noesis, the saved phone-to-Noesis
transform is composed into a Noesis-world derivative of the merged revision.
If Noesis registration happens later, that derivative is generated when the
alignment completes. Deleting an addition is allowed only from the newest end
of a dependent revision chain.

The browser runs one base provider per scan. A completed, quality-gated DA3
alignment exposes the optional PCF action, which runs conditioned MapAnything
and the selected dual-provider fusion without replacing the base DA3 output.
PCF currently uses the original prepared walk only. If an added-video revision
is active, the app refuses PCF rather than silently omitting those added views.

## Provider semantics

### MapAnything

- Official Apache-licensed `facebook/map-anything-apache` model.
- Jointly predicts multi-view poses, depth, intrinsics, metric scaling, and
  learned confidence.
- The installed RTX 3060 capacity check passes 80 joint views at 10.56 GB peak
  allocated memory; 96 views OOM with DS9.1 stopped. The 80-view default is a
  measured per-window GPU limit, not a capture limit. Use
  `benchmark_mapanything_capacity.py` before increasing it on another GPU.
- Native DS9.1 must be paused while this offline reconstruction owns the GPU;
  the deployed browser action exposes and records its configured automatic
  pause/restore lease rather than competing for memory or silently changing
  runtime authority.

### DA3

- Official Apache-licensed `depth-anything/DA3-BASE` supplies joint any-view
  geometry, poses, intrinsics, and learned multiview confidence.
- DA3Metric-Large supplies metric depth through the validated FP16 TensorRT
  engine for the RTX 3060.
- That DA3Metric engine is coupled to the installed TensorRT version and must
  be rebuilt after a TensorRT upgrade. MapAnything and consensus fusion do not
  use serialized TensorRT plans in this utility.
- The DA3Metric non-sky output is a validity mask, not a confidence map.
- The integration uses DA3 Base confidence for weighting and the DA3Metric
  non-sky mask for validity. It does not invent confidence values.
- Adaptive sequences above 48 views use 16-view-overlap DA3 windows. Exact
  duplicate RGB-D views gate each fixed-scale window registration, and every
  unique accepted view is retained for the conditioned MapAnything and PCF
  stages.

## Consensus fusion workflow

The authoritative implementation is
`tools/mapanything_phone_scan/build_consensus_fusion.py`.

It does not average raw point clouds. The fusion performs every stage below:

1. **Shared pose graph**
   - Aligns the DA3 trajectory to MapAnything with a Sim(3) estimate.
   - Uses relative-pose observations from both models at one-, two-, and
     four-frame spans.
   - Adds a soft positional loop constraint between the first and final view.
   - Does not force the final phone orientation to equal the starting
     orientation.
2. **Common camera rays**
   - Converts the 518x294 MapAnything depth and 504x280 DA3 depth to a shared
     504x280 angular grid derived from both inferred intrinsics.
   - This is geometric reprojection, not image-sized depth averaging.
3. **Separate reliability calibration**
   - Converts each provider's raw confidence distribution to its own empirical
     percentile reliability scale.
   - Combines that reliability with temporal multiview reprojection consistency
     and a depth-boundary penalty.
4. **Agreement fusion**
   - When depths agree within approximately 10-15 cm, selects a weighted median
     surface so an artificial midpoint wall is not created.
5. **Moderate disagreement selection**
   - Chooses the provider with stronger multiview reprojection consistency.
6. **Large-disagreement rejection and hole filling**
   - Rejects major conflicts as uncertain.
   - Accepts a single-provider hole fill only when confidence, multiview
     consistency, and neighboring support clear their thresholds.
7. **Surfel fusion**
   - Integrates accepted depths into confidence-weighted 4 cm surfels.
   - Preserves the provider-selection and disagreement arrays in every raw
     fused view for diagnostics.

The 4 cm surfel GLB is the conservative review artifact. It is not the
highest-density representation available for downstream point models; the
accepted full-resolution evidence remains in the consensus `raw/` directory.

### Reusable command

Once both raw output roots exist for matching prepared views:

```bash
python3 tools/mapanything_phone_scan/build_consensus_fusion.py \
  data/mapanything_phone_scans/<scan-id> \
  --mapanything-raw data/mapanything_phone_scans/<scan-id>/<ma-output>/raw \
  --da3-raw data/mapanything_phone_scans/<scan-id>/<da3-output>/raw \
  --output-dir data/mapanything_phone_scans/<scan-id>/consensus_fusion
```

The input roots must contain matching view counts from the same prepared phone
frames. The builder fails rather than substituting another model or input set.

### Point-preserving 2 cm export for PTv3 and Roomform

To reuse the accepted consensus evidence at PTv3's 2 cm sampling scale without
rerunning either depth model:

```bash
python3 testpipelines/roomform/build_point_preserving_fusion.py \
  data/mapanything_phone_scans/<scan-id>/<consensus-output>/raw \
  data/mapanything_phone_scans/<scan-id>/<consensus-output>/point_preserving_fusion_2cm
```

This export does not reintroduce points rejected by consensus. It removes the
review GLB's every-other-pixel sampling, performs confidence-weighted 2 cm
aggregation, and retains voxels with at least two accepted samples and 0.80
accumulated weight. It writes GLB, PLY, NPZ, and JSON report artifacts. The NPZ
also carries accumulated weights, sample counts, and distinct-view counts.

Use the matching consensus `camera_solution.npz` when supplying this cloud to
Roomform so its point evidence and scanner stations stay in the same frame.

On the validated landscape living-room capture, 3,206,909 accepted samples
produced 417,845 RGB points. Relative to the 89,499-point 4 cm review cloud,
nearest-neighbor spacing improved from 30.5 mm to 15.8 mm, the fraction of
points isolated beyond 5 cm fell from 1.31% to 0.10%, and local roughness fell
from 19.3 mm to 9.5 mm. The raw horizontal DA3 cloud remains slightly denser,
while this fusion retains the cross-model conflict rejection.

For the validated portrait capture, the preserved input and output roots are:

```text
MapAnything: data/mapanything_phone_scans/20260801-112036-2d9225dd/outputs/raw
DA3:         data/mapanything_phone_scans/20260801-112036-2d9225dd/da3_toggle_phone_only_20260808/raw
Consensus:   data/mapanything_phone_scans/20260801-112036-2d9225dd/consensus_fusion_20260809
```

## Heatmap-style diagnostic views

The renderer applies the same floor estimation, camera-ground presentation,
rasterization, height, density, obstacle, walkable, and edge calculations to
MapAnything, DA3, and consensus outputs. It derives +X camera-right and +Z
camera-forward from the first phone view (or the selected static camera for
backend-world evaluation), writes row zero at maximum +Z, and never applies a
room-specific rotate or mirror.

For reusable paths:

```bash
python3 tools/mapanything_phone_scan/render_phone_heatmap_diagnostics.py \
  data/mapanything_phone_scans/<scan-id> \
  --mapanything-raw data/mapanything_phone_scans/<scan-id>/<ma-output>/raw \
  --da3-raw data/mapanything_phone_scans/<scan-id>/<da3-output>/raw \
  --consensus-raw data/mapanything_phone_scans/<scan-id>/consensus_fusion/raw
```

The standard expanded-section label is **Diagnostic layers**. In addition to
the common Heatmap layout, consensus writes
`consensus_collaboration_diagnostics.png`, which shows:

- normalized per-model depth;
- fused depth;
- absolute disagreement;
- calibrated reliability;
- multiview consistency;
- selected provider;
- accepted versus rejected regions;
- original and optimized camera trajectories.

## Validated portrait results

The 2026-08-09 portrait fusion produced:

```text
93,939 confidence-weighted surfels
68.95% fused valid-pixel coverage
44.59% direct agreement among mutually valid model pixels
30.89% moderate disagreements resolved by multiview consistency
24.52% large disagreements rejected
8.21% total pixels filled by a validated single-model observation
```

Internal and held-out phone-view validation:

| Metric | MapAnything | DA3 | Consensus |
| --- | ---: | ---: | ---: |
| Multiview reprojection median | 5.98 cm | 5.35 cm | **4.08 cm** |
| Held-out even-to-odd median | 11.90 cm | 9.15 cm | **8.31 cm** |
| Held-out even-to-odd p80 | 36.88 cm | 28.77 cm | **24.90 cm** |
| Held-out odd-frame coverage | 35.1% | 38.1% | 32.7% |

The soft loop constraint reduced the joint trajectory's first-to-last position
gap from 74.3 cm to 55.1 cm. A stronger constraint reached 24 cm but degraded
held-out depth accuracy, so it was rejected.

## Static-camera alignment findings

Static-camera data was held out of the phone-only fusion and used afterward for
Noesis registration and validation.

The consensus reconstruction passed every existing alignment gate:

```text
Phone-cloud source overlap within 30 cm: 95.93%
Phone-cloud source median residual:      10.19 cm
Vertical plane median residual:           8.35 cm
Fixed-camera target coverage:            51.98%
Fixed-camera depth disagreement median:  31.45 cm
```

This is a mixed result. Consensus is best on internal multiview consistency and
held-out phone views, and it produces the cleanest floorplan representation.
MapAnything alone remains better against the current fixed-camera depth
comparison, whose earlier median disagreement was approximately 21.5 cm.
Therefore:

- keep consensus as the preferred phone-walk room reconstruction candidate;
- keep the static reconstruction as the independent alignment authority;
- retain all provider and disagreement diagnostics;
- do not claim consensus is more accurate on every static-visible surface;
- do not promote a room transform unless the Noesis alignment gate passes.

## Validated landscape DA3-conditioned MapAnything workflow

MapAnything accepts per-view depth, pose, and calibration inputs. The validated
landscape workflow uses this support to condition MapAnything with DA3 rather
than trying to inject an unordered DA3 point cloud.

The prior builder derives a sparse metric-depth input from DA3 only. It keeps
pixels that pass temporal reprojection and depth-boundary checks, then samples
10% of those reliable pixels with a fixed seed. For the validated walk, the
prior retained 6.45% of all pixels. This avoids presenting dense correlated
DA3 errors to MapAnything as ground truth.

Four controlled variants are produced:

| Variant | Purpose | Result |
| --- | --- | --- |
| DA3 pose | Test trajectory conditioning alone | Poor geometry; model-to-carrier pose median 22.21 cm |
| Sparse DA3 depth | Test metric surface conditioning alone | Strong phone consistency, but still needs trajectory registration |
| DA3 pose + sparse depth | Joint trajectory and surface conditioning | Best unfused variant; model-to-carrier pose median 1.99 cm |
| DA3 pose + sparse depth + static view | Test a calibrated static view inside the joint batch | Slightly worse than pose + depth without the static view |

The best operational result is a second consistency-gated fusion of DA3 with
the pose-plus-depth MapAnything output, using the validated DA3 trajectory as
the pose carrier. It is saved as `prior_conditioned_consensus_da3_carrier`.
The pose-plus-depth output remains the unfused comparison/control artifact.

### Landscape validation results

Every candidate below used the same static-world crop, 5 cm diagnostics, 2.5 cm
non-averaged point splat, held-out phone-frame test, and fixed-camera test.

| Candidate | Internal median | Held-out median | Static target median | Target within 30 cm | Fixed-camera depth delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| MapAnything image-only | 10.12 cm | 12.99 cm | 25.96 cm | 53.0% | 49.07 cm |
| DA3 | 5.38 cm | 8.26 cm | 14.76 cm | 65.5% | 33.51 cm |
| Earlier image-only MA + DA3 fusion | 3.57 cm | 5.31 cm | 21.59 cm | 57.1% | 29.63 cm |
| MapAnything + DA3 pose | 11.81 cm | 22.32 cm | 13.88 cm | 72.5% | 77.23 cm |
| MapAnything + sparse DA3 depth | 4.36 cm | 5.82 cm | 16.97 cm | 62.7% | 31.61 cm |
| MapAnything + DA3 pose/depth | 4.60 cm | 7.48 cm | **13.03 cm** | **65.1%** | 26.84 cm |
| MapAnything + DA3 pose/depth/static | 4.82 cm | 7.72 cm | 14.03 cm | 64.7% | 27.15 cm |
| **Prior-conditioned MA + DA3 fusion** | 4.31 cm | 6.58 cm | 13.78 cm | 62.8% | **23.73 cm** |

Compared with the earlier image-only consensus, the prior-conditioned fusion
increased valid fused pixels from 51.28% to 88.68%, increased agreement among
mutually valid pixels from 32.13% to 89.27%, and reduced large disagreements
from 42.97% to 3.23%. The two models are no longer statistically independent
after conditioning, so those agreement numbers are interpreted together with
the independently held-out phone and static-camera improvements.

The calibrated static reconstruction remains the world-frame authority and an
independent validation target. Adding its RGB/depth as a 49th joint inference
view did not improve this walk. Live detections and tracks must continue to use
the exact per-camera calibration; the phone reconstruction supplies the room
surface and floorplan, not a replacement for calibrated track projection.

### Family Room qualification

The same workflow was repeated for landscape scan
`20260810-215847-571c6efe`. Its admitted immutable lineage resolved an earlier
Family Room target-frame mismatch upstream. Current alignment validates the
calibrated camera against the authoritative cloud and refuses a backwards
target; it does not rotate a working target copy or add a Family-only display
correction.

| Candidate | Internal median | Held-out median | Static source median | Static target median | Source within 30 cm | Target within 30 cm | Fixed-camera depth delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DA3 | 3.69 cm | 4.66 cm | 10.99 cm | 7.15 cm | 83.2% | 87.0% | 12.65 cm |
| MapAnything + DA3 pose/depth | 4.41 cm | 5.53 cm | 12.96 cm | 9.29 cm | 81.3% | 77.8% | 13.84 cm |
| **Prior-conditioned MA + DA3 fusion** | **3.35 cm** | **4.40 cm** | 12.25 cm | 9.37 cm | 82.7% | 78.4% | 12.94 cm |

DA3 alone remained slightly closer to the fixed-camera target, while the
prior-conditioned fusion retained the best internal and held-out phone-view
consistency and the cleaner review geometry. The selected scene-prior source is
therefore `prior_conditioned_consensus_da3_carrier`; DA3 and the unfused
pose/depth result remain required comparison evidence.

This qualifies a room-local static scene prior only. It does **not** establish
cross-camera overlap or authorize AMC/MV3DT calibration. The current topology
has no Living Room/Family Room overlap; the only prospective multi-view edge is
Kitchen/Family Room, and that edge remains disabled until the Kitchen geometry
is corrected and synchronized occupied overlap evidence passes review.

### Reusable commands

For the routine path, run the selected pose-plus-depth variant on matching
prepared phone views and DA3 raw data. Omit `--variants` when qualifying all
four variants:

```bash
  data/mapanything_phone_scan_runtime/venv/bin/python \
  tools/mapanything_phone_scan/run_mapanything_prior_variants.py \
  data/mapanything_phone_scans/<scan-id> \
  --da3-raw data/mapanything_phone_scans/<scan-id>/outputs/raw \
  --world-from-da3 \
    data/mapanything_phone_scans/<scan-id>/alignment/phone_ma_to_noesis_world.json \
  --target-revision data/virtual_twin/revisions/<approved-room-revision> \
  --calibration config/camera_calibration.json \
  --camera <camera-id> \
  --variants da3_pose_sparse_depth \
  --output-root <large-storage-root>/<scan-id>/da3_prior_suite
```

Fuse the best unfused variant with DA3 while preserving the validated DA3 pose
carrier:

```bash
python3 tools/mapanything_phone_scan/build_consensus_fusion.py \
  data/mapanything_phone_scans/<scan-id> \
  --mapanything-raw <suite-root>/mapanything_da3_pose_sparse_depth/raw \
  --da3-raw data/mapanything_phone_scans/<scan-id>/outputs/raw \
  --pose-carrier da3 \
  --output-dir <suite-root>/prior_conditioned_consensus_da3_carrier
```

Evaluate the selected result and regenerate the common diagnostic layouts:

```bash
python3 tools/mapanything_phone_scan/evaluate_mapanything_prior_variants.py \
  data/mapanything_phone_scans/<scan-id> \
  --suite-root <suite-root> \
  --da3-raw data/mapanything_phone_scans/<scan-id>/outputs/raw \
  --prior-consensus-raw \
    <suite-root>/prior_conditioned_consensus_da3_carrier/raw \
  --variants da3_pose_sparse_depth \
  --world-from-da3 \
    data/mapanything_phone_scans/<scan-id>/alignment/phone_ma_to_noesis_world.json \
  --target-revision data/virtual_twin/revisions/<approved-room-revision> \
  --calibration config/camera_calibration.json \
  --camera <camera-id> \
  --output-dir <suite-root>/evaluation_static_world
```

The suite manifest records model/package identity, prior thresholds and seed,
input hashes, pose disagreement, output coordinate frames, and the selected
review artifacts. Pose-conditioned raw NPZ files preserve the model-predicted
pose separately as `model_camera_pose`; their `world_points` use predicted
MapAnything depth and intrinsics backprojected through the supplied world pose
carrier.

### Hand off an approved PCF candidate to Noesis

Do not treat a successful evaluation directory as a deployed room. Continue at
**Seal the approved reconstruction** in the
[canonical PCF runbook](../../docs/PCF_Workflow.md). That runbook uses
`build_conditioned_scene_prior_bundle.py` to enforce the alignment and
static-visible admission gates, then uses `scripts/build_scene_prior.py` to
derive and bind the immutable 2.5 cm Scene Prior. This explicit bridge is the
only documented PCF-to-runtime path.

### Align a whole-home PCF review assembly to an authored model

Use `solve_pcf_model_structural_alignment.py` only for a Menon review overlay;
it does not publish calibration or canonical world state. The admitted static
camera supplies a bounded yaw/correspondence seed. The final uniform planar
similarity (yaw, X/Z, and XZ scale) comes from the four dominant Family Room
vertical wall planes and the reviewed Family Room floor polygon in the authored
OBJ. It deliberately does not use whole-cloud ICP or the editable camera-device
position as final geometry authority.

Run it from the repository root so the package imports resolve:

```bash
python3 -m tools.mapanything_phone_scan.solve_pcf_model_structural_alignment \
  --review-manifest "$PCF_REVIEW_MANIFEST" \
  --surfels-npz "$PCF_SURFELS_NPZ" \
  --surfels-manifest "$PCF_SURFELS_MANIFEST" \
  --authored-scene "$MENON_AUTHORED_SCENE_OBJ" \
  --room-group-map config/authored_scene_room_groups.json \
  --seed-yaw-deg "$PCF_CAMERA_SEED_YAW_DEG" \
  --output "$PCF_STRUCTURAL_ALIGNMENT_REPORT"
```

The generated `noesis.pcf.menon_structural_alignment` report is review-only and
fail-closed. Its assembly, GLB, surfel, authored-scene, room-group, and camera-
anchor digests must remain bound, all structural gates must pass, and Menon must
apply the resulting transform once to the complete assembly. `Structural fit`
restores this report; `Camera seed` is only a diagnostic comparison. Internal
room-to-room registration uncertainty remains unchanged by this global model
alignment.

### Derive a surface mesh for Menon review

`build_pcf_review_surface_mesh.py` converts the retained multi-room PCF surfels
into a colored cutaway surface for owner review. It reconstructs each room owner
independently with screened Poisson, trims triangles that are not supported by
nearby measurements, and removes only small disconnected components. Rebuilding
rooms separately prevents an uncertain cross-room registration from inventing
surfaces across a doorway or room join. The default 4 cm voxel size, 10 cm
support limit, and 1.85 m ceiling cutaway favor a readable interior without
claiming unobserved closure.

```bash
python3 tools/mapanything_phone_scan/build_pcf_review_surface_mesh.py \
  --surfels-npz "$PCF_SURFELS_NPZ" \
  --surfels-manifest "$PCF_SURFELS_MANIFEST" \
  --output-glb "$PCF_MESH_GLB" \
  --output-manifest "$PCF_MESH_MANIFEST"
```

To expose that derived GLB through the same bounded Noesis-to-Menon review
route, create a new immutable contract-v4 assembly from the active contract-v3
point assembly:

```bash
python3 tools/mapanything_phone_scan/publish_pcf_review_surface_mesh.py \
  --pcf-storage-root "$NOESIS_PHONE_SCAN_PCF_STORAGE_ROOT" \
  --current-descriptor "$NOESIS_PHONE_SCAN_PCF_STORAGE_ROOT/review-assemblies/current.json" \
  --mesh-glb "$PCF_MESH_GLB" \
  --mesh-manifest "$PCF_MESH_MANIFEST" \
  --source-structural-alignment "$PCF_STRUCTURAL_ALIGNMENT_REPORT" \
  --structural-output "$MENON_PCF_STRUCTURAL_ALIGNMENT_CONFIG" \
  --assembly-id "$PCF_MESH_ASSEMBLY_ID" \
  --activate
```

The publisher verifies the point-assembly, surfel, mesh, and structural-report
digest chain before changing the active review selector. It preserves the
source point assembly, copies the already reviewed transform without changing
its yaw, translation, or scale, and binds Menon to the new mesh digest. Both the
mesh and point variants remain presentation-only; neither changes calibration,
room registration, canonical world state, or tracking authority.

## Detection and tracking integration contract

The intended detection/tracking use is not to rebuild the room continuously.
Instead:

1. Build and approve a room reconstruction from a phone walk.
2. Align it to the calibrated static-camera Noesis world.
3. Save the accepted world transform and its source artifact identities.
4. Apply the existing calibrated camera-to-world projection to detections and
   tracks.
5. Render those world-space tracks against the approved room reconstruction.

Producer and consumer coordinate frames, units, camera calibration identity,
room revision, and transform provenance must match. A visually plausible but
unvalidated transform is not sufficient for live tracking.

## Important artifacts

The curated selected landscape evidence is retained at:

```text
docs/evidence/phone_walk_fusion/20260809-landscape-living-room/
```

For the validated portrait run:

```text
consensus_fusion_20260809/consensus_manifest.json
consensus_fusion_20260809/scan_outputs_manifest.json
consensus_fusion_20260809/camera_solution.npz
consensus_fusion_20260809/surfel_points.npz
consensus_fusion_20260809/consensus_surfel_reconstruction.glb
consensus_fusion_20260809/consensus_collaboration_diagnostics.png
consensus_fusion_20260809/noesis_alignment_validation/alignment_report.json
phone_only_heatmap_diagnostics/consensus_fusion/phone_only_heatmap_diagnostics.png
<consensus-output>/point_preserving_fusion_2cm/point_preserving_fusion_2cm.glb
<consensus-output>/point_preserving_fusion_2cm/point_preserving_fusion_2cm.ply
<consensus-output>/point_preserving_fusion_2cm/point_preserving_fusion_2cm.npz
<consensus-output>/point_preserving_fusion_2cm/point_preserving_fusion_report.json
```

## Configuration

Common optional environment variables:

```text
NOESIS_PHONE_SCAN_PORT=8788
NOESIS_PHONE_SCAN_STORAGE_ROOT=data/mapanything_phone_scans
NOESIS_PHONE_SCAN_PCF_STORAGE_ROOT=/large-storage-root/noesis-phone-pcf
NOESIS_PHONE_SCAN_PCF_PAUSE_APPLIANCE=0
NOESIS_PHONE_SCAN_RUNTIME_ROOT=data/mapanything_phone_scan_runtime
NOESIS_PHONE_SCAN_CANDIDATE_FPS=4
NOESIS_PHONE_SCAN_MAX_CANDIDATE_FRAMES=1200
NOESIS_PHONE_SCAN_MAX_SELECTED_FRAMES=256
NOESIS_PHONE_SCAN_CANDIDATE_EDGE_PX=1280
NOESIS_PHONE_SCAN_FEATURE_EDGE_PX=640
NOESIS_PHONE_SCAN_MIN_KEYFRAME_INTERVAL_S=0.40
NOESIS_PHONE_SCAN_MAX_KEYFRAME_INTERVAL_S=1.25
NOESIS_PHONE_SCAN_MAX_EDGE_PX=1920
NOESIS_PHONE_SCAN_MA_MAX_JOINT_VIEWS=80
NOESIS_PHONE_SCAN_MA_WINDOW_OVERLAP_VIEWS=24
NOESIS_PHONE_SCAN_POINT_BUDGET=600000
NOESIS_PHONE_SCAN_MA_DEVICE=cuda:0
NOESIS_PHONE_SCAN_DA3_DEVICE=cuda:0
NOESIS_PHONE_SCAN_DA3_PROCESS_RES=504
NOESIS_PHONE_SCAN_DA3_REF_VIEW=middle
NOESIS_PHONE_SCAN_DA3_MAX_JOINT_VIEWS=48
NOESIS_PHONE_SCAN_DA3_WINDOW_OVERLAP_VIEWS=16
NOESIS_PHONE_SCAN_DA3_ENGINE=data/ds9_artifacts/models/engines/da3metric_large_294x518_b3_fp16_trt10.16.engine
NOESIS_PHONE_SCAN_LOCAL_FILES_ONLY=1
NOESIS_PHONE_SCAN_STATIC_ANCHOR=0
NOESIS_PHONE_SCAN_ALIGNMENT_CAMERA_ID=living-room
NOESIS_PHONE_SCAN_ALIGNMENT_RELEASE=data/virtual_twin/releases/home_rgbmesh_20260623T2158_v1.json
NOESIS_PHONE_SCAN_ALIGNMENT_CALIBRATION=config/camera_calibration.json
```

PCF retains conditioned MapAnything raw arrays, fused raw arrays, and static-
world review artifacts. Budget roughly 1.5-2 GB for a 180-210-view room and
place `NOESIS_PHONE_SCAN_PCF_STORAGE_ROOT` on durable large storage. Deleting a
walk from the app also deletes that walk's separately stored PCF runs.
Set `NOESIS_PHONE_SCAN_PCF_PAUSE_APPLIANCE=1` when the conditioned MapAnything
window cannot coexist with the native appliance on the installed GPU. The app
stops `menon-appliance.target` only if it was active before PCF and restores it
on both success and failure. An interrupted phone-tool process records the
lease before stopping the target so its next startup can recover the appliance.

The app fails clearly if FFmpeg, CUDA, cached official models, the validated
TensorRT engine, or the local Three.js dependency is unavailable. It does not
select a degraded inference path automatically.

## Validation commands

```bash
pytest -q \
  tools/mapanything_phone_scan/test_phone_scan.py \
  tools/mapanything_phone_scan/test_consensus_fusion.py \
  tools/mapanything_phone_scan/test_prior_variants.py \
  tools/mapanything_phone_scan/test_prior_variant_evaluation.py \
  tools/mapanything_phone_scan/test_point_preserving_heatmap.py

python3 -m py_compile \
  tools/mapanything_phone_scan/build_consensus_fusion.py \
  tools/mapanything_phone_scan/render_phone_heatmap_diagnostics.py \
  tools/mapanything_phone_scan/run_mapanything_prior_variants.py \
  tools/mapanything_phone_scan/evaluate_mapanything_prior_variants.py
```

The consensus tests cover common-ray reprojection, source selection,
large-disagreement rejection, single-model fill validation, and loop-constrained
pose optimization.
