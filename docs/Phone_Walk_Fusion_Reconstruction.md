# Phone-Walk Fusion Reconstruction

Status: validated 2026-08-10 against the current phone-scan tools and the
48-view landscape living-room artifacts.

## Decision

The preferred room-reconstruction path is:

1. capture one slow, overlapping phone walk;
2. run DA3 on every prepared phone view;
3. align the DA3 trajectory to the calibrated Noesis room world;
4. run MapAnything with the DA3 world poses and a sparse, reliability-gated
   subset of DA3 metric depth as priors;
5. consistency-gate the conditioned MapAnything depth against DA3 again, using
   DA3 as the pose carrier;
6. validate the result against held-out phone views and the independent static
   reconstruction; and
7. approve the resulting room artifact separately from live tracking.

The selected artifact is called the **prior-conditioned MA + DA3 fusion with
DA3 pose carrier**. The unfused MapAnything pose-plus-depth result is retained
as its control.

This decision is based on the landscape living-room comparison, not on model
preference. It produced the cleanest useful floorplan, the best fixed-camera
depth agreement of the tested phone candidates, strong held-out phone-view
consistency, and an explicit path into the calibrated backend world.

## Authority boundaries

| Concern | Authority | What it must not replace |
| --- | --- | --- |
| Offline room surface and floorplan | Approved phone-walk fusion artifact | Per-camera tracking calibration |
| Live detection and track projection | Exact calibrated static-camera transforms | A visually fitted phone trajectory |
| Phone-to-room registration and validation | Static-camera room revision plus camera calibration | Static RGB/depth injected into every phone inference batch |
| Manual depth-panel inference | Selected MapAnything or DA3 runtime backend | The offline room-fusion workflow |

Static-camera data is intentionally independent during the selected fusion.
The experiment that added a calibrated static view inside the MapAnything
batch was slightly worse. Static data therefore remains the alignment
authority and validation target instead of becoming correlated reconstruction
input.

The DA3/MapAnything selector described in `DS9/docs/DA3Metric_Large.md` changes
manual depth inference after a controlled restart. It does not select or run
this offline fusion. Likewise, `docs/MapAnything_Depth.md` describes the live
manual depth/floorplan contract, not room-artifact generation.

## Why prior conditioning won

Four MapAnything variants isolated the effect of each prior:

| Variant | Result |
| --- | --- |
| DA3 poses only | Poor geometry; median predicted-pose disagreement with the carrier was 22.21 cm |
| Sparse reliable DA3 depth only | Good phone consistency but still required trajectory registration |
| DA3 poses plus sparse reliable DA3 depth | Best unfused result; median pose disagreement fell to 1.99 cm |
| DA3 poses/depth plus a static view | Slightly worse than poses/depth without the static view; median pose disagreement was 3.54 cm |

The poses and sparse metric depth solve different problems together. The DA3
trajectory supplies a stable world-coordinate carrier. Sparse DA3 depth keeps
MapAnything near the metric surface solution without forcing every correlated
DA3 pixel to be ground truth. MapAnything can then contribute cleaner object
and room structure. The final fusion keeps surfaces supported by both models,
chooses the better-supported provider for moderate conflicts, and rejects
large conflicts.

DA3Metric does not produce a learned confidence map. The combined DA3 provider
uses DA3 Base multiview confidence for weighting and the DA3Metric non-sky mask
for validity. No synthetic confidence is invented for DA3Metric.

## Reconstruction algorithm

### 1. Prepare one shared view set

The video is sampled into a fixed prepared-frame manifest. Every subsequent
provider and variant must consume those same frame identities and order. A
view-count or input mismatch is an error; the tools do not substitute another
scan or model.

The validated landscape capture used 48 views at 1920 x 1080. Forty-eight is
the current configured maximum, not a demonstrated optimum. A more complete
walk should improve coverage when it adds new, sharp, overlapping viewpoints.
Increasing the view limit is a separate experiment because GPU memory,
quadratic multiview work, blur, repeated views, and loop drift can offset the
benefit.

### 2. Establish DA3 reliability and world poses

DA3 supplies any-view geometry, camera poses and intrinsics, DA3 Base
multiview confidence, and DA3Metric-Large metric depth. The phone alignment
maps its trajectory into `backend_world_m_stream_points` using the configured
static room revision and camera calibration.

The alignment filename retains the legacy
`phone_ma_to_noesis_world.json` name, but the transform loader accepts the DA3
world-transform field and the transform is provider-generic.

### 3. Build the sparse metric-depth prior

The prior builder uses DA3 data only:

- temporal multiview reprojection consistency threshold: `0.55`;
- depth-boundary weight threshold: `0.50`;
- deterministic sample: 10% of the reliable pixels;
- random seed: `8675309`.

On the selected walk, 64.32% of all pixels passed the reliability test and
6.45% of all pixels were supplied to MapAnything after sampling. Each view
retained at least 2,373 prior samples. This intentionally sparse prior reduces
the chance that locally correlated DA3 errors dominate MapAnything.

### 4. Run conditioned MapAnything

The selected `da3_pose_sparse_depth` variant gives MapAnything the backend-
world DA3 pose for every view and the sparse metric-depth samples. MapAnything
still predicts its own camera solution for diagnostics. Its predicted metric
depth and intrinsics are back-projected through the supplied DA3 world poses,
so the saved point maps have an explicit coordinate carrier.

For a new room, run only this selected variant for the routine path. Run all
four variants when qualifying a new model, capture protocol, room type, or
materially changed alignment method.

### 5. Fuse the conditioned result with DA3

`tools/mapanything_phone_scan/build_consensus_fusion.py` performs geometric
fusion rather than averaging unordered point clouds:

- both providers are reprojected to common camera rays;
- each provider's confidence distribution is calibrated independently;
- confidence, temporal consistency, and a depth-boundary penalty form the
  per-pixel reliability;
- depths agreeing within a distance-dependent 10-15 cm gate select an actual
  provider surface by weighted median, never a synthetic midpoint wall;
- moderate disagreements within a 25-35 cm gate select the provider with
  stronger multiview consistency, with reliability as the tie-breaker;
- larger disagreements are rejected;
- single-provider holes are accepted only with reliability at least `0.48`,
  consistency at least `0.42`, and temporal support;
- accepted pixels with quality at least `0.35` are integrated into
  confidence-weighted 4 cm review surfels with at least two samples and `0.80`
  accumulated weight.

The selected fusion uses `--pose-carrier da3`. The accepted full-resolution
per-view evidence and provider-selection arrays remain in `raw/`; the 4 cm GLB
is a conservative review cloud.

### 6. Render point-preserving review layers

The evaluation produces both the familiar 5 cm diagnostic layers and 2.5 cm
phone-point layers. The 2.5 cm layers do not average color or height. When
multiple samples collide in one 2D X/Z cell, the highest-confidence actual
phone sample wins, and separate floor, low-object, furniture, and upper-
structure bands avoid collapsing a vertical column into one mean.

These are review rasters, not the stored point-cloud contract. The raw accepted
samples remain available, and the optional 2 cm point-preserving export can be
used by PTv3 or Roomform as described in
`tools/mapanything_phone_scan/README.md`.

The camera-oriented review convention rotates the rendered BEV 180 degrees so
the living-room hallway/foyer appears at the top-left. This changes presentation
only; it does not rotate the saved backend-world geometry.

## Validated landscape result

All candidates used the same static-world bounds, raster rules, held-out
even-to-odd phone test, and fixed-camera comparison.

| Candidate | Internal median | Held-out median | Static target median | Static target within 30 cm | Fixed-camera depth delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| MapAnything image-only | 10.12 cm | 12.99 cm | 25.96 cm | 53.0% | 49.07 cm |
| DA3 | 5.38 cm | 8.26 cm | 14.76 cm | 65.5% | 33.51 cm |
| Earlier image-only MA + DA3 fusion | 3.57 cm | 5.31 cm | 21.59 cm | 57.1% | 29.63 cm |
| MapAnything + DA3 pose | 11.81 cm | 22.32 cm | 13.88 cm | 72.5% | 77.23 cm |
| MapAnything + sparse DA3 depth | 4.36 cm | 5.82 cm | 16.97 cm | 62.7% | 31.61 cm |
| MapAnything + DA3 pose/depth | 4.60 cm | 7.48 cm | **13.03 cm** | **65.1%** | 26.84 cm |
| MapAnything + DA3 pose/depth/static | 4.82 cm | 7.72 cm | 14.03 cm | 64.7% | 27.15 cm |
| **Prior-conditioned MA + DA3 fusion** | 4.31 cm | 6.58 cm | 13.78 cm | 62.8% | **23.73 cm** |

The selected fusion contained 150,901 4 cm surfels. Across 6,773,760 common-
ray pixels, 88.68% had an accepted fused depth. Where both providers were
valid, 89.27% passed the agreement gate and 3.23% were rejected as large
disagreements. Median absolute inter-model depth disagreement was 2.59 cm.

Those agreement numbers are not independent evidence because MapAnything was
conditioned with DA3. The independently held-out phone frames and static-camera
comparison are therefore required alongside them. The remaining 23.73 cm
fixed-camera median depth delta is useful for candidate comparison but is not
tracking-grade calibration.

The derived 2.5 cm TSDF review mesh used all 48 fused RGB-D views, a 10 cm SDF
truncation, a 10.5 m depth limit, no smoothing, small-component filtering, and
a crop to the accepted surfel bounds plus 15 cm. Its final review form had
363,652 vertices and 693,035 triangles. It is a visualization derivative, not
the authoritative point evidence, and it contains no static-camera geometry.

Curated image evidence and hashes are stored in
`docs/evidence/phone_walk_fusion/20260809-landscape-living-room/`.

## Repeat the selected workflow for another room

Run all commands from the repository root. Put large generated artifacts on a
large storage volume, not the root filesystem.

### Prerequisites

- a new phone scan with `prepared_frames_manifest.json` and its original video;
- the official cached DA3 and Apache-licensed MapAnything models;
- the validated DA3Metric-Large TensorRT engine;
- an approved static-camera room revision containing `room_points.npz`,
  `room_points_meta.json`, and an RGB keyframe;
- that camera's entry in `config/camera_calibration.json`; and
- enough storage for raw provider views, variants, evaluations, and retained
  manifests.

Configure the phone-scan service for the target camera and static revision
before capture/alignment:

```text
NOESIS_PHONE_SCAN_ALIGNMENT_CAMERA_ID=<camera-id>
NOESIS_PHONE_SCAN_ALIGNMENT_REVISION=data/virtual_twin/revisions/<room-revision>
NOESIS_PHONE_SCAN_ALIGNMENT_CALIBRATION=config/camera_calibration.json
```

Restart the user service after changing those environment values. Do not reuse
the living-room transform for a different room.

### 1. Capture, run DA3, and align

Use the phone browser to capture a slow walk with:

- the phone held in one orientation for the entire recording;
- slow translation and rotation with minimal motion blur;
- substantial overlap between neighboring views;
- deliberate coverage of doorways, corners, furniture sides, and open floor;
- an early view near the calibrated static camera's view; and
- a return near that area to provide loop evidence.

Select **DA3**, run reconstruction, then select **Align to Noesis**. For a scan
whose DA3 run is the browser's active provider, the expected inputs are:

```text
data/mapanything_phone_scans/<scan-id>/outputs/raw/
data/mapanything_phone_scans/<scan-id>/alignment/phone_ma_to_noesis_world.json
```

Stop if alignment fails its quality gate. Do not silently continue in a local
phone frame or borrow another room's transform.

### 2. Run the selected conditioned MapAnything variant

```bash
data/mapanything_phone_scan_runtime/venv/bin/python \
  tools/mapanything_phone_scan/run_mapanything_prior_variants.py \
  data/mapanything_phone_scans/<scan-id> \
  --da3-raw data/mapanything_phone_scans/<scan-id>/outputs/raw \
  --world-from-da3 \
    data/mapanything_phone_scans/<scan-id>/alignment/phone_ma_to_noesis_world.json \
  --target-revision data/virtual_twin/revisions/<room-revision> \
  --calibration config/camera_calibration.json \
  --camera <camera-id> \
  --variants da3_pose_sparse_depth \
  --output-root <large-storage-root>/<scan-id>/da3_prior_suite
```

The runner refuses to overwrite an existing variant directory. Preserve a
failed or superseded manifest long enough to diagnose it, then choose a new
suite directory for the next attempt.

### 3. Build the selected fusion

```bash
python3 tools/mapanything_phone_scan/build_consensus_fusion.py \
  data/mapanything_phone_scans/<scan-id> \
  --mapanything-raw \
    <suite-root>/mapanything_da3_pose_sparse_depth/raw \
  --da3-raw data/mapanything_phone_scans/<scan-id>/outputs/raw \
  --pose-carrier da3 \
  --output-dir <suite-root>/prior_conditioned_consensus_da3_carrier
```

### 4. Evaluate in the static-camera world

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
  --target-revision data/virtual_twin/revisions/<room-revision> \
  --calibration config/camera_calibration.json \
  --camera <camera-id> \
  --output-dir <suite-root>/evaluation_static_world
```

The evaluator uses identical bounds and calculations for every candidate in
that invocation. It writes the expanded **Diagnostic layers**, 2.5 cm point
layers, static top view, fixed-camera reprojection, overview, and JSON metrics.

For a full qualification, omit `--variants` from both the runner and evaluator
to include all four conditioned variants. Optional evaluator controls can be
added with `--image-only-raw` and `--comparison-consensus-raw` when matching
same-frame artifacts exist.

### 5. Review and approve

Review at least:

- input manifests, view count, hashes, model/package identity, and coordinate
  frame;
- the camera trajectory, first-to-last loop gap, and pose disagreement;
- the collaboration diagnostic's agreement, selection, and rejected regions;
- the 5 cm diagnostic and 2.5 cm point-preserving layers;
- held-out phone reprojection error and coverage;
- static-cloud target distance and coverage;
- fixed-camera reprojection depth disagreement; and
- point cloud and optional mesh for holes, duplicated walls, stretched
  furniture, floating surfaces, and missing doorway coverage.

Do not approve from a clean-looking floorplan alone. Save the selected manifest,
evaluation JSON, transform identity, static revision identity, and a compact
evidence set. Live tracking continues to project through the calibrated static
camera, while the approved phone artifact supplies the shared room surface.

## Artifact retention and cleanup

Keep:

- the original phone video and prepared-frame manifest;
- the selected DA3 raw output;
- the selected conditioned MapAnything raw output and variant manifest;
- the prior-conditioned consensus raw output, point cloud, camera solution,
  collaboration diagnostic, and manifest;
- the final evaluation JSON and selected review images; and
- the exact phone-to-world transform, static revision, and calibration
  identities.

Remove only reproducible scratch renders, Python bytecode caches, incomplete
temporary directories, superseded duplicate mesh exports, and old evaluation
images whose JSON metrics and provenance chain have been retained. Do not
delete provider raw arrays or manifests solely because a later visualization
looks better.

## Known limits

- The selected models are correlated after DA3 conditioning; cross-model
  agreement alone cannot prove accuracy.
- Phone-to-static registration is still weaker than exact camera calibration.
- The static camera sees only part of the room, so its target coverage is not a
  complete accuracy measure.
- Thin, reflective, textureless, occluded, and motion-blurred surfaces remain
  difficult.
- A denser or longer walk can improve coverage only if it adds usable parallax
  and overlap; more redundant or blurred frames can make inference slower or
  less stable.
- The mesh is a derived review surface and can bridge holes or expose TSDF edge
  artifacts. Point evidence and manifests remain authoritative.

## Related references

- `tools/mapanything_phone_scan/README.md` — service, provider, artifact, and
  operator details.
- `docs/Virtual_Twin_Reconstruction.md` — static-camera room-revision contract.
- `docs/MapAnything_Depth.md` — manual runtime depth/floorplan behavior.
- `DS9/docs/DA3Metric_Large.md` — DA3Metric-Large engine and manual-depth
  selector.
