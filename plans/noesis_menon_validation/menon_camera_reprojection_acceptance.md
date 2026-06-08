# Menon Camera Reprojection Acceptance

Status: implemented acceptance contract as of 2026-05-27.

Use this gate when a change affects camera calibration, generated Menon room
geometry, Noesis `backend_world_m` placement, BEV-to-Menon transforms, avatar
scale, or any live overlay meant to prove Noesis and Menon agree in camera
space.

## Required Evidence

Each selected camera must provide one `camera_reprojections` row in the shared
Menon trace report with:

- `camera_id` and, when available, `room`;
- `source_frame` from the real camera/video frame;
- `menon_render` rendered from the matching Menon virtual camera pose;
- `overlay_path` or equivalent visual review artifact;
- layers: `source_frame`, `menon_render`, `detected_bbox`,
  `projected_avatar`, `room_mesh_edges`, `floor_grid`, and `anchors`;
- anchor, floor-grid, room-edge, bbox/avatar, or mask alignment metrics.

Missing source/render/layer evidence is `blocked`, not a pass. Numeric
misalignment is a projection warning or failure.

## Thresholds

Default acceptance uses `MENON.camera_reprojection`:

| Metric | Good | Fail |
|---|---:|---:|
| Mean pixel error | <= 12 px | > 50 px |
| Max pixel error | <= 30 px | > 100 px |
| Bbox/avatar/mask IoU | >= 0.50 | < 0.25 |

At least one numeric alignment metric must be present. A visual-only row can be
kept for human review, but it should not be treated as verified.

## Fixture Mode

Use a saved Menon trace when live Menon is unavailable or when reviewing a
deterministic fixture:

```bash
python3 scripts/noesis_validation_menon_trace_report.py \
  --trace plans/noesis_menon_validation/minimal_menon_trace.json \
  --run-id minimal_menon_trace
```

## Live Browser Mode

Use browser capture when Menon is running and exposes debug evidence:

```bash
python3 scripts/noesis_validation_capture_menon_trace.py \
  --url http://127.0.0.1:5173 \
  --output diagnostics/validation/menon_browser_trace/trace.json \
  --screenshot \
  --validate
```

For Menon-required acceptance, run the saved-trace validator with
`--require-menon-root` and pass `--menon-root` or set `MENON_ROOT`. If the
checkout, page, source frame, Menon render, camera pose, or required layers are
unavailable, report the blocked check and stop instead of substituting a
Noesis-only fixture.

## Review Rule

A Menon-facing spatial change is not accepted from top-down agreement alone.
It must either pass `MENON.camera_reprojection` or record a blocked result with
the missing evidence named explicitly.
