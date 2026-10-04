# Kitchen structural evidence review — 2026-10-04

Status: offline extraction improvement, not an admitted registration or a
physical-accuracy pass. Active Kitchen mapping, authored home, runtime,
calibration, localization and BEV are unchanged. The owner is checking physical
dimensions and correcting the Sweet Home 3D source independently.

## What changed

The existing global extractor retained only three nearly parallel planes. Its
bounded trace shows 32 of 36 RANSAC attempts selecting near-horizontal
hypotheses, even after the local-normal prefilter. Rejected hypotheses consume
iterations and remove their inliers. This explains missing constraints; it
does not prove that Kitchen lacks perpendicular wall geometry.

Added a generic, offline image-region extraction lane:

- Identify a surface in exact retained carrier RGB, not by its location under
  the current unqualified Menon mapping.
- Keep the existing consensus quality threshold (0.70), consensus selection,
  valid mask and uncertain-point exclusion. This score is not a calibrated
  probability. Reject weak regions instead of accepting single-model fills.
- Replay the retained carrier-to-world transform unchanged. Associate regions
  with existing admitted PCF points; do not introduce replacement points.
- Measure an unconstrained 3D seed plane. Recover connected admitted support
  in the existing 5 cm plane slab and 6 cm voxel grid; require vertical local
  normals agreeing with the measured seed within 30 degrees. The original
  seed stays fixed during selection, preventing iterative plane chasing.
- Refit the measured support without forcing an upright plane. Retain support,
  height, length and tilt bounds, and separately require P90 residual at most
  5 cm for a review candidate. These checks do not admit a registration.

Implementation: `noesis/calibration/structural_roi_planes.py` and
`scripts/review_pcf_wall_regions.py`. Nine focused component checks pass,
including measured tilt, exact-point selection, rejection of weak support and
disconnected coplanar surfaces. Syntax checks and the real Kitchen run pass.

## Exact retained evidence and results

Selected evidence: `data/scene_priors/review/kitchen-wall-review-20261004/`
`normal-supported-review/review.json`; corresponding top-down and fixed-camera
projection images are beside it. Earlier subdirectories are exploratory or
partial attempts, not selected review outputs.

The admitted Kitchen PCF is
`sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483`; point digest
`e4382e3b55aed54041dc6989edb77efff55244e36e57219107e225650107d6cf`.
Source bundle and explicit file inventory match its manifest. All 167,338
admitted points replay from the retained transformed surfels with maximum
nearest-point discrepancy 0.000001051 m. This proves source/frame association,
not independent physical accuracy.

| Reviewed surface | Connected PCF points | 6 cm voxel support | Vertical span | P90 plane residual | Review candidate |
| --- | ---: | ---: | ---: | ---: | --- |
| Right stone facade | 3,233 | 1,712 | 2.286 m | 0.0436 m | Yes |
| Far portrait wall | 2,800 | 1,484 | 2.012 m | 0.0445 m | Yes |
| TV / near-camera wall | 8,943 | 3,833 | 2.442 m | 0.0403 m | Yes |
| Cabinet tile backing | 420 | 275 | 0.629 m | 0.0466 m | No: tilt and height |

Stone/portrait acute angle is 86.73 degrees; stone/TV is 88.27 degrees;
portrait/TV directions differ by 1.54 degrees. This supplies the previously
missing perpendicular direction, without forcing the room square. Small
non-orthogonality may include reconstruction error; it is not proof of a
physical slope or crooked room.

Exact RGB review corrected an initial mistaken correspondence: phone views 8
and 22 show the TV wall, not the portrait wall opposite the fixed camera.
Portrait view 25 supplies the clean seed; view 26's upper patch has zero points
after the unchanged quality gates. View 22's TV-region geometry disagrees with
view 8 and was not combined into the final TV seed. Its rejected combined
result remains in `expanded-planes-v2/review.json`.

The fixed-camera projection uses the retained keyframe and exact bound K/E.
Green support falls on the right stone facade; magenta support falls on the
portrait wall. The near-camera TV plane is behind the camera, so it has no
visible projected points. There is no occlusion test: points behind furniture
can project over a chair/table, and that is not person-depth evidence. The
review overlay is not a fitted Menon preview.

## Authored counterparts and next fit

On the currently selected authored model, camera-facing candidate surfaces
are stone `wall_59_414`, portrait `wall_39_278`, and TV bay `wall_14_100` between
the window and slider. Their opposite faces are different planes. Stone
material is on the opposite authored face, so its exact facing/correspondence
needs checking rather than assuming texture establishes the wall plane.

Connected support endpoints are coverage extents, not observed corners or
aperture edges. Do not use them as independent heldout contacts. Three fit
planes alone do not satisfy the structural solver's independent-holdout gate;
it remains unchanged. Retained views also share one fused reconstruction,
not independently surveyed ground truth. A new mapping is not admitted here.
Directly exercising the unchanged solver against the current exact authored
model returns `rejected: missing_independent_holdout`; the frozen inputs and
result are in `fit-readiness.json` beside the review directories. This is the
expected boundary, not a waived check or a newly qualified fit.

When the owner supplies a corrected SH3D export, re-extract its faces and
openings; group/segment IDs may change. Check finished face versus opposite
face and visible trim versus bare opening identity. Fit the three reviewed
plane constraints and reserve independently identified window/slider or wall
termination evidence before fitting. Useful take-offs record named finished
faces, exact endpoints, window/slider widths and their distance to a named
corner. Avoid treating the obscured cabinet face as the backing wall.

## Reproduction

Set `KITCHEN_SOURCE_BUNDLE` to the retained source bundle directory (its digest
is verified against the admitted prior). Use a fresh output directory:

```sh
PYTHONPATH=. data/mapanything_phone_scan_runtime/venv/bin/python \
  scripts/review_pcf_wall_regions.py \
  --prior-manifest data/scene_priors/revisions/sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483/manifest.json \
  --source-bundle "$KITCHEN_SOURCE_BUNDLE" \
  --recipe data/scene_priors/review/kitchen-wall-review-20261004/regions-final.json \
  --output data/scene_priors/review/kitchen-wall-review-20261004/replayed-review \
  --grow-connected \
  --reference-image data/virtual_twin/revisions/vt_kitchen_stream_rgbmesh_20260623T215835_609188061/keyframes/kitchen_0001.png \
  --reference-alignment-report data/mapanything_phone_scans/20260814-190412-ea26c040/alignment/alignment_report.json
```
