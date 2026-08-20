# PCF multi-room cross-session registration

Status: current lab workflow and fail-closed runbook, 2026-08-16.

Use this document when two independently accepted Prior-Conditioned Fusion
(PCF) room walks overlap and must be placed into one continuous home world.
This is not the per-room PCF build: it begins with two already accepted PCF
revisions and estimates only their cross-session spatial relationship.

## Decision and authority

- Hold one accepted room fixed as the output-world gauge. For the first
  Kitchen/Family Room join, Family Room is fixed.
- Preserve metric scale at exactly one, gravity, handedness, and the accepted
  fixed-room world. The cross-room state is planar SE(2): yaw, X, and Z.
- Derive Y from the two accepted floor planes. Do not let image PnP introduce a
  floor step.
- Use static-camera evidence only through each accepted room's recorded
  local-PCF-to-backend transform. Do not add a static frame to either phone
  inference set.
- Whole-room ICP, geometry-only initialization, nearest-neighbor room fitting,
  reflection, manual visual nudging, and scale fitting are prohibited.
- A local point-to-plane refinement is allowed only after RGB-D registration
  passes and only inside the overlap observed by verified image matches. It is
  a bounded refinement, never an initializer or a way around failed holdouts.
- A visually convincing composite is not automatically canonical. Rejected
  transforms may be rendered as `review_only`, but must not become a Scene
  Prior, tracking-world transform, or dashboard authority.

## Implementation surfaces

| File | Responsibility |
| --- | --- |
| `tools/mapanything_phone_scan/register_pcf_rooms.py` | Accepted-artifact loading, classic feature and RGB-D geometry primitives, diagnostics |
| `tools/mapanything_phone_scan/register_pcf_rooms_lightglue.py` | Exhaustive SuperPoint + LightGlue cross-walk discovery and fail-closed registration report |
| `tools/mapanything_phone_scan/pcf_multiroom_pose_graph.py` | View-balanced planar pose graph, drift field, and complete-view/temporal holdouts |
| `tools/mapanything_phone_scan/pcf_local_overlap_refinement.py` | Optional visually bounded local surface refinement with floor lock |
| `tools/mapanything_phone_scan/reintegrate_pcf_rooms.py` | Provenance-preserving 2.5 cm two-room reintegration and ownership |
| `tools/mapanything_phone_scan/render_pcf_multiroom_presentation.py` | Presentation-only export in a recorded reference-camera ground frame |
| `tools/mapanything_phone_scan/render_pcf_multiroom_bev.py` | Full-extent dashboard-style joined BEV review |

The pose graph never consumes a room cloud, and the reintegrator never
estimates registration. That separation prevents an attractive rendering from
silently becoming its own alignment evidence.

## Required evidence

For each room retain:

1. the original video and immutable `prepared_frames_manifest.json`;
2. every prepared RGB frame;
3. the selected PCF directory
   `prior_conditioned_consensus_da3_carrier`, including all `raw/view_*.npz`
   files and `surfel_points.npz`;
4. the accepted Scene Prior ID and phone scan ID; and
5. the exact accepted `evaluation/static_world_point_cloud_manifest.json`.

The raw PCF points and camera poses are first-view-relative DA3 metric data.
Apply the transform from the accepted static-world manifest before comparing
rooms. Do not substitute the scan's earlier alignment JSON when the accepted
bundle records a later corrected transform.

## Lab sequence

### 1. Discover overlap from the original RGB

Extract SuperPoint features once for every retained view, then exhaustively
match all cross-room view pairs with LightGlue. For each pair:

- require enough learned matches;
- reject fundamental-matrix outliers;
- map feature coordinates back to the PCF model grid;
- require accepted, non-uncertain PCF depth and reliability; and
- retain the complete moving/fixed view identities.

SIFT remains a cheap first pass, but it is not a failure verdict for a
wide-baseline doorway viewed from opposing directions. The Kitchen/Family data
had genuine overlap that SIFT did not recover reliably.

### 2. Solve both PnP directions

For every surviving view pair, solve:

- moving-room PCF depth into the fixed-room image; and
- fixed-room PCF depth into the moving-room image.

Project each result to gravity-preserving yaw and translation, and record
inlier count plus median and p80 reprojection. Bidirectional agreement matters:
one-sided planar fits can look excellent while translation remains weak.

Canonical pair admission defaults are:

| Gate | Requirement |
| --- | ---: |
| PnP inliers | at least 20 |
| median reprojection | no more than 3 px |
| p80 reprojection | no more than 5 px |
| bidirectional yaw disagreement | no more than 2.5 degrees |
| bidirectional translation disagreement | no more than 0.65 m before graph fitting |

The broad bidirectional limits only screen individual hypotheses. The stricter
whole-view and segment gates below decide whether the session transform is
usable.

### 3. Lock the accepted floor plane

Estimate the dominant low horizontal support plane from each accepted PCF
cloud and set:

```text
vertical translation = fixed floor height - moving floor height
```

Then optimize yaw, X, and Z only. Record the raw PnP Y disagreement as a
diagnostic, not as a solved degree of freedom.

### 4. Fit a view-balanced trajectory graph

Keep the fixed room immutable. Fit one global moving-room transform plus a
smooth per-moving-view correction field. Weight complete moving views equally
so one highly textured rug view cannot dominate several independent phone
positions. The correction field reveals RoomWalk drift; it does not authorize
arbitrary per-frame warping.

When available, add verified intra-session RGB-D or VIO edges between moving
views. They constrain trajectory shape but never establish the cross-room
gauge.

### 5. Hold out independent evidence blocks

Validation units are complete phone views and temporal visits, never random
feature points from the same image.

Required gates include:

| Gate | Requirement |
| --- | ---: |
| training planar translation, median / p80 | no more than 0.15 / 0.25 m |
| training yaw, median / p80 | no more than 0.75 / 1.25 degrees |
| leave-one-moving-view-out translation, median / p80 | no more than 0.20 / 0.25 m |
| leave-one-moving-view-out yaw, median / p80 | no more than 0.40 / 0.50 degrees |
| leaveout session-transform p80 | no more than 0.25 m / 0.50 degrees |
| temporal-segment session delta | no more than 0.20 m / 0.50 degrees |
| floor disagreement after transform | no more than 0.03 m |

A temporal segment must leave enough other moving views and cross-room pairs
to solve the graph. If holding out one visit leaves only one moving view, the
evidence topology is insufficient even when its training residual is small.

Depth-to-depth and overlap-surface checks are independent diagnostics:

- held-out paired depth should target median at most 0.10 m and p80 at most
  0.25 m;
- visually bounded mutual surfels should have median at most 0.05 m and p80 at
  most 0.10 m; and
- at least 30 percent of the known-overlap moving surfels should be within
  0.10 m of fixed evidence.

Nearest-neighbor surface statistics never initialize the transform and cannot
override a failed view/segment holdout.

### 6. Reintegrate only after disposition is explicit

For an accepted transform, stream all selected raw PCF pixels from both rooms,
apply their accepted local-to-world transforms and the cross-session
correction, and fuse at 2.5 cm. Preserve:

- room owner and room presence;
- source-view bitmasks and observation counts;
- PCF source-selection counts;
- cross-model agreement and absolute disagreement; and
- MapAnything and DA3 reliability.

The fixed room owns exact shared voxels and its multiview-supported X/Z
coverage. Moving-room evidence outside that overlap remains. The output raster
may hide isolated one-view splats, but the NPZ and GLB keep the complete fused
evidence.

`reintegrate_pcf_rooms.py` defaults to
`caller_supplied_unvalidated`, writes `status: review_only`, and sets
`accepted_for_canonical_use: false`. Only a passed registration report may use
`accepted_cross_session_registration`.

### 7. Render in one recorded camera frame

Backend-world axes are metric authority, not a human-facing orientation. After
registration and reintegration, derive one fixed camera's recorded right and
forward basis and apply it equally to every room. Do not stitch camera-local
rasters and do not add a room-specific flip.

For a Family-fixed review:

```text
Family display  = W_family_camera * Family_backend
Kitchen display = W_family_camera * T_kitchen_to_family * Kitchen_backend
```

The display matrix may have determinant `-1` because it maps OpenCV camera
Y-down into a Y-up presentation basis. That is not a physical reflection and
must never enter PnP, pose-graph optimization, reintegration, Scene Prior
geometry, or tracking transforms.

## Reproducible command pattern

Use a storage-backed virtual environment so model weights do not consume the
root filesystem. The first Kitchen/Family experiment used official LightGlue
commit `eb42fee2d71449efb0aa5c10549752b5d75384d8`, LightGlue package version
`0.0`, PyTorch `2.12.1+cu130`, and these weight hashes:

```text
SuperPoint  52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e
LightGlue   6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a
```

Example registration:

```bash
PCF_REG_PY="replace-with-registration-venv/bin/python"
PCF_TORCH_CACHE="replace-with-storage-backed-torch-cache"
PCF_OUTPUT="replace-with-new-output-directory"

TORCH_HOME="${PCF_TORCH_CACHE}" "${PCF_REG_PY}" \
  tools/mapanything_phone_scan/register_pcf_rooms_lightglue.py \
  --moving-name kitchen \
  --moving-prior-id "replace-kitchen-prior" \
  --moving-scan-id "replace-kitchen-scan" \
  --moving-scan-dir "data/mapanything_phone_scans/replace-kitchen-scan" \
  --moving-pcf-root "replace-kitchen-pcf-root" \
  --moving-world-manifest "replace-kitchen-accepted-static-world-manifest" \
  --fixed-name family-room \
  --fixed-prior-id "replace-family-prior" \
  --fixed-scan-id "replace-family-scan" \
  --fixed-scan-dir "data/mapanything_phone_scans/replace-family-scan" \
  --fixed-pcf-root "replace-family-pcf-root" \
  --fixed-world-manifest "replace-family-accepted-static-world-manifest" \
  --vertical-translation-m 0.0 \
  --output-dir "${PCF_OUTPUT}"
```

The command returns nonzero and omits consumable transforms when support or a
holdout fails. Inspect `registration_report.json`; do not extract a rejected
diagnostic candidate into runtime configuration.

After a passed registration:

```bash
python3 tools/mapanything_phone_scan/reintegrate_pcf_rooms.py \
  --fixed-name family-room \
  --fixed-prior-id "replace-family-prior" \
  --fixed-scan-id "replace-family-scan" \
  --fixed-raw-root "replace-family-pcf-root/raw" \
  --fixed-world-manifest "replace-family-accepted-static-world-manifest" \
  --moving-name kitchen \
  --moving-prior-id "replace-kitchen-prior" \
  --moving-scan-id "replace-kitchen-scan" \
  --moving-raw-root "replace-kitchen-pcf-root/raw" \
  --moving-world-manifest "replace-kitchen-accepted-static-world-manifest" \
  --moving-registration-report "${PCF_OUTPUT}/registration_report.json" \
  --registration-disposition accepted_cross_session_registration \
  --output-dir "replace-with-new-reintegration-directory"
```

Then create a camera-oriented review without altering the fused backend world:

```bash
python3 tools/mapanything_phone_scan/render_pcf_multiroom_presentation.py \
  --source-npz "replace-reintegration/multiroom_pcf_surfels.npz" \
  --source-reintegration-manifest \
    "replace-reintegration/multiroom_pcf_manifest.json" \
  --scene-prior-manifest \
    "data/scene_priors/revisions/replace-fixed-prior/manifest.json" \
  --output-dir "replace-with-new-presentation-directory"
```

## Kitchen and Family Room result, 2026-08-15

Inputs:

- Family Room fixed prior
  `sceneprior_family-room_20260811T015847Z_ffdc144a8f59`, scan
  `20260810-215847-571c6efe`;
- Kitchen moving prior
  `sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483`, scan
  `20260814-190412-ea26c040`; and
- 48 original landscape phone views per room, with no static image mixed into
  either PCF inference batch.

The learned matcher found undeniable shared Family Room evidence through the
opening. The strict screen retained 19 PnP observations over 12 view pairs,
three Kitchen views, six Family views, and seven bidirectional pairs. Both
accepted floor medians were already equal within `0.000018 m`.

The best review-only planar transform is:

```text
yaw = 18.012509 degrees
translation = (-4.518760, -0.000018, 8.871015) m

[[ 0.95098903, 0,  0.30922462, -4.51875994],
 [ 0,          1,  0,          -0.00001780],
 [-0.30922462, 0,  0.95098903,  8.87101457],
 [ 0,          0,  0,           1          ]]
```

It produces a physically plausible continuous review and strong local seam
statistics, but is not canonical:

- training planar translation was `0.118 m` median / `0.179 m` p80;
- leave-one-Kitchen-view-out was `0.307 m` median / `0.363 m` p80 and
  `0.608` / `0.919` degrees;
- the late temporal-segment holdout collapsed to one remaining Kitchen view;
- the separate paired-depth holdout was `0.192 m` median / `0.499 m` p80; and
- evidence was concentrated in Kitchen views 11 and 46.

The output is therefore explicitly `review_only`. Current artifacts are under:

```text
<PCF_STORAGE_ROOT>/pcf_multiroom/
  kitchen_family_cross_session_20260815/
  floor_constrained_review_reintegration_v2/
```

The raw backend-X/Z topdown in that directory is superseded for orientation
review; its source registration and backend NPZ are unchanged. The shared
Family-camera presentation is under
`floor_constrained_review_reintegration_v3_family_presented/`. It places the
Family camera at `(0, 0)`, uses camera-right horizontally and camera-forward
upward, and applies that same presentation basis to both rooms. Its manifest
accounts for all `1,075,656` surfels and both 48-view paths while proving the
source NPZ remained byte-identical.

The canonical solution is a short connector walk, not another full-room walk.

## Minimal connector capture

Use the same landscape orientation as the accepted room walks. A practical
20–40 second capture is enough:

1. begin two to three meters inside Family Room with the doorway, both jambs,
   adjacent wall, floor, and fixed furniture in view;
2. approach slowly while adding a small left/right lateral arc for parallax;
3. cross into Kitchen while keeping the jambs and fixed cabinetry visible;
4. continue two to three meters into Kitchen and pan across fixed Kitchen
   structure; and
5. turn and cross the opening once more before stopping.

Prefer walls, jambs, cabinetry, and floor/wall intersections over the patterned
rug and movable children's furniture. Keep exposure steady and motion slow.
This supplies two independently held-out visits plus views that overlap each
existing room walk. Process the connector as PCF phone evidence, add verified
intra-session/cross-session edges to the pose graph, and rerun the same gates.
It does not need to replace either accepted full-room reconstruction.

## Connector evidence and geometry ownership

A connector walk has two roles that must be evaluated separately:

1. endpoint observations constrain the relative transform between accepted
   room PCFs; and
2. reconstructed transition points may fill geometry neither room measured.

The first role does not authorize the second. Accepted PCFs remain geometry
authority at both endpoints. Limit connector points to the actual temporal
transition and suppress them inside buffered endpoint-room coverage. If the
remaining points are disconnected, duplicate an endpoint room, or fail
trajectory consistency, omit them while retaining verified endpoint matches as
registration constraints. Never append another reconstruction of either
accepted room merely because it came from the connector video.

`extend_pcf_multiroom_with_connector.py` implements this ownership boundary.
It can retain an explicit transition-view interval or use
`--omit-connector-geometry` when the walk supports registration but not new
surface geometry. Its manifest records selected views, endpoint buffers, room
ownership, and whether the connector was constraints-only.

## Living Room to Kitchen connector result, 2026-08-16

Supplement `add-20260816-152331-a80746c9` supplied 72 adaptive frames plus
eight exact Living bridge views. One joint 80-view DA3 pass and one joint
80-view prior-conditioned MapAnything pass produced phone-only PCF evidence
with `0.055 m` median / `0.116 m` p80 multiview disagreement and `0.812`
cross-model agreement.

Endpoint match support improved substantially, but the combined pose graph
remained rejected:

- training translation was `0.252 m` median / `0.379 m` p80;
- leave-one-moving-view-out translation was `0.417 m` median / `0.671 m` p80;
- leave-one-moving-view-out yaw was `1.682` degrees median / `3.442` degrees
  p80; and
- training, complete-view, per-view-deformation, and temporal-segment gates
  failed.

The accepted Living and Kitchen PCFs nevertheless produced a useful
review-only transform. Their final 10 cm occupancy had 347 neighboring columns,
and `99.9281%` of occupied columns belonged to one connected component. The
connector transition surfels formed a separate sliver, so the final review used
the connector only for pose constraints. Ownership is Family `496,413`,
Kitchen `579,243`, Living `465,703`, and connector `0` voxels; the Kitchen /
Family source NPZ remained byte-identical.

This is not a Scene Prior or tracking-world authority. The combined Living
camera placement carries approximately `0.88 m` p80 translation uncertainty
and a `6.71` degree yaw bound. Current artifacts are under:

```text
<PCF_STORAGE_ROOT>/pcf_multiroom/
  kitchen_family_living_bridge_20260816_152331/
  review_reintegration_v3/
  review_bev_v3_with_living_camera/
```
