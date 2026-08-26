# ChArUco intrinsics calibration
_Status: native DS9.1 workflow, updated 2026-08-26._

This guide shows how to calibrate camera intrinsics using a ChArUco board and
apply the results to canonical `config/cameras.yaml`.

## Board parameters (your print)

- Board: 7 x 5 squares
- Dictionary: `DICT_4X4_50`
- Square length: 30 mm
- Marker length: 22 mm

Print at 100% scale. Measure one square with calipers and use the measured value
if it differs from 30 mm.

## Capture checklist

1. Use the exact camera feed settings you run in DS9.1 (resolution, crop, zoom).
   If DS9.1 uses a substream, calibrate on that substream; scaling a full-res
   calibration can be wrong when the substream is cropped/zoomed.
2. Capture 30 to 80 images with the board at:
   - different positions across the frame (center, corners, edges)
   - different distances (near, mid, far)
   - different tilts (pitch/roll)
3. Avoid motion blur. Lock autofocus/auto-zoom if possible.
4. Save images as JPG or PNG in a single folder.

Tip: If you only have video, you can extract frames:

```
ffmpeg -i /path/to/video.mp4 -vf fps=2 /tmp/charuco/frame_%04d.jpg
```

## Run calibration (example)

```
python3 scripts/charuco_calibrate_intrinsics.py \
  --image-dir /tmp/charuco \
  --squares-x 7 \
  --squares-y 5 \
  --square-length-mm 30 \
  --marker-length-mm 22 \
  --dictionary DICT_4X4_50
```

The script writes a JSON report under `diagnostics/` and prints the intrinsics.
Aim for median reprojection error around 0.5 to 1.5 px; lower is better.

## Update canonical intrinsics

To update the canonical intrinsics file:

```
python3 scripts/charuco_calibrate_intrinsics.py \
  --image-dir /tmp/charuco \
  --squares-x 7 \
  --squares-y 5 \
  --square-length-mm 30 \
  --marker-length-mm 22 \
  --dictionary DICT_4X4_50 \
  --update-cameras-yaml config/cameras.yaml \
  --yaml-model-key unifi_g4_instant_charuco_720_raw_fisheye
```

This updates the selected model entry in `config/cameras.yaml` with its measured
resolution, K parameters, radial coefficients, and calibration metadata. Review
the diff before binding a camera to the new model. DeepStream camInfo generation
uses K but does not consume distortion coefficients.

## Distortion handling (SV3DT)

SV3DT camInfo uses a **pinhole** projection (no distortion). For wide-angle
cameras, you should undistort the video **before** PGIE/tracker using
`nvdewarper`, then treat the dewarped stream as pinhole with the same
fx/fy/cx/cy and **zero** distortion.

Guidelines:

1. Build a dewarper config using the ChArUco K + distortion.
2. Add `nvdewarper` before `nvstreammux` (per-source).
3. Create a rectified intrinsics model (k1/k2/k3 = 0).
4. Regenerate camInfo using the rectified intrinsics.

Example files in this repo (family-room RTSP 1280x720):

- Dewarper config: `config/dewarper_family_room_charuco_rtsp.txt`
- Rectified intrinsics: `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`
- Preview pipeline: `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp.yaml`

Note: the dewarper config now includes `dst-focal-length` and
`dst-principal-point` computed from OpenCV’s `getOptimalNewCameraMatrix`
(`centerPrincipalPoint=True`, alpha=0) to reduce crop/warp artifacts.

## After calibration

1. Update the selected model in `config/cameras.yaml` and bind the camera to it.
2. Run the focused calibration/intrinsics tests and one projection overlay for
   the changed camera.
3. Restart the managed runtime only when accepting the new intrinsics.

Baseline tracking remains the default. The accepted Kitchen/Family Room MV3DT
lane is a separate explicit opt-in; ordinary intrinsics calibration must not
select it or widen its peer topology. If calibration changes either accepted
peer camera, regenerate its camInfo with `scripts/generate_v3dt_caminfo.py` and
revalidate the bound MV3DT profile before selecting that lane.
