# WO-2C static camera calibration and direct dependents

Status: implementation complete for the bounded review and replacement path;
canonical replacement remains blocked by missing measured static-image metadata,
independent metric scale, and accepted frame bindings in the retained household
inputs.

## Implemented path

`tools/mapanything_phone_scan/localize_pcf_static_camera.py` now validates a
versioned static-camera image contract before geometric estimation. It checks
the exact image digest, camera ID, encoded resolution, intrinsics, distortion
model and coefficients, rectification state, meters, and source frame identity
including its revision. The existing stream reconstruction producer emits
`noesis.room_reconstruction.stream_points.v4`, so an exact declared keyframe can
be adapted for geometric review. The adapter records the input schema and
explicitly marks distortion provenance and rectification as unknown; it does
not claim that the legacy image was rectified. This adapter cannot satisfy the
canonical static-image provenance check.

The localizer keeps complete excluded temporal ranges out of the fitted pose
consensus and evaluates those views afterward. It gates each candidate with
fundamental-matrix support, depth-backed PnP, reprojection residuals,
cheirality, spatial distribution, early/late walk support, SE3 dispersion, and
floor geometry in full-pose mode. `full_pcf_pose` is recorded as a request;
measured acceptance requires every applicable check. Independent scale evidence
is a separate meters-only input with exact source frame ID and revision, at
least two independently sourced measurements, no fit on evaluation points, and
finite agreement. An explicit assembly-to-calibration binding is required for
replacement.

`tools/mapanything_phone_scan/calibration_replacement.py` makes an exact,
scoped backup before writing a separate replacement calibration. It updates E
and the corresponding PoseV1 record together, so `CalibrationManager` cannot
regenerate a stale manual pose over the replacement. The replacement output is
atomic and carries source, replacement, backup, frame, acceptance, and binding
provenance. When retained static depth is supplied, the path reprojects depth
through the replacement pose, writes a bounded world-point artifact, and
materializes a normal `room_points.npz` plus `room_points_meta.json` revision
from an explicit template keyframe. The revision records replacement and frame
binding hashes and is consumed by the existing PCF static-reference loader.
Source calibration and template revisions remain unchanged.

## Retained-data evidence

The exact machine paths and command results are stored under the configured
large-storage root. The portable result records are:

- `$NOESIS_RECONSTRUCTION_WORK_ROOT/wo2c_static_localization_real_v5/real_run_results.json`
- `$NOESIS_RECONSTRUCTION_WORK_ROOT/wo2c_static_localization_real_v6_family/real_run_results.json`

The Living Room review used the retained 48-view scan, the conditioned
MapAnything plus DA3 carrier raw output, the existing Living Room source-world
manifest, and the exact `living-room_0001.png` keyframe declared by its virtual
twin metadata. The legacy metadata adapted successfully, but its 48-view run
admitted 0 static-camera candidates in 64.8 seconds and failed closed. It did
not authorize a replacement.

The Family Room review used the retained 48-view scan, the conditioned
cross-model raw carrier under `$NOESIS_PCF_STORAGE_ROOT`, the exact Family Room
keyframe and metadata, and the corrected global source-world manifest. It
admitted 0 candidates and remained review-only. The run used the corrected raw
carrier with `cross_model_agreement`; it was not rejected for a missing field.
The legacy stream metadata still lacks measured distortion and rectification
provenance, and no independent physical scale evidence or accepted calibration
frame binding was supplied.

These runs are geometric review evidence only. They provide no physical
accuracy claim and no canonical extrinsic replacement. A legacy source that
happens to produce PnP candidates would still fail the static-image provenance
check until a calibrated producer supplies the exact intrinsics, distortion,
rectification, resolution, image digest, source frame ID, and revision.

## Direct-consumer fixture and validation

The bounded fixture exercises replacement calibration -> exact backup -> E and
PoseV1 update -> normal WO-1 `build_scene_prior`/catalog materialization ->
`CalibrationManager` reload with the generated v2 frame binding. It also
exercises replacement depth -> world points -> normal static revision ->
existing `_load_static_reference`, checks that source bytes remain unchanged,
and checks output collision rejection. No active calibration catalog or runtime
file was changed.

The normal stream producer now carries the actual `RgbFrame` through
`FrameCloud`, floor correction, and ceiling clipping. When the live path has a
configured dewarper, `_write_revision` emits the nested
`noesis.pcf.static_camera_image.v1` contract with the dewarper config digest,
source fisheye coefficients, rectified K scaled to the written keyframe size,
source URI/index, keyframe digest, and calibration fingerprint. Persisted
zarr/depth-texture paths remain explicitly unmeasured and do not emit a
rectification claim. The focused producer test runs `_build_frame_clouds` and
the normal revision writer with an actual dewarper config.

Focused validation:

```text
data/mapanything_phone_scan_runtime/venv/bin/python -m pytest -q \
  tools/mapanything_phone_scan/test_localize_pcf_static_camera.py \
  tools/mapanything_phone_scan/test_build_stream_room_reconstruction.py \
  tools/mapanything_phone_scan/test_solve_pcf_connector_multianchor.py \
  tools/mapanything_phone_scan/test_trajectory_refinement.py \
  tools/mapanything_phone_scan/test_consensus_fusion.py \
  tools/mapanything_phone_scan/test_capture_vio.py
40 passed
```

Python compilation passed for the changed localizer, replacement helper,
trajectory/PCF mapping consumer, and stream producer. `git diff --check` passed
for the owned files. No active calibration, service, or runtime state was
changed by this work.

The retained Family diagnostic is stored at
`$NOESIS_RECONSTRUCTION_WORK_ROOT/wo2c_family_root_review_probe.json` and
explains the zero-admission result without
weakening gates: four views reached solved-PnP with at least eight inliers, but
all failed the required spatial support gate. Their image-area hull fractions were
0.004625, 0.006052, 0.014625, and 0.045931, with only 2--3 occupied 4x4 bins
per view against the minimum four. The practical recapture request is a wider
walk with revisits across distributed static-camera features; the current
single local feature cluster cannot authorize a replacement.

## Remaining evidence and integration dependencies

The retained household images still do not provide measured static-image
distortion/rectification, independent physical scale, or an accepted
assembly-to-calibration transform, so they remain review-only. A future
accepted replacement needs those measurements plus active catalog/frame-binding
regeneration coordinated through WO-1. Existing manual poses remain the usable
fallback until those dependencies pass.

The physical calibration fingerprint covers all camera E matrices in the
shared calibration file. Replacing one camera therefore requires rebuilding
the affected frame bindings for that calibration set through the normal
builder before reload, including bindings for cameras whose numeric E did not
change. Existing bindings with the old physical fingerprint cannot be reused.
