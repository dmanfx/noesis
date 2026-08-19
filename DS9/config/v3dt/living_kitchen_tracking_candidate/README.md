# Kitchen optimized SV3DT profile

Status: validated, opt-in per-camera SV3DT profile. MV3DT remains disabled in
the canonical native DS9.1 application. This profile is isolated from the
non-V3DT lane and does not replace its tracker, analytics, identity settings,
or camera assets.

Bounded recorded-input reproduction command:

```bash
NOESIS_CAMERA_CALIBRATION_FILE="$PWD/DS9/config/v3dt/living_kitchen_tracking_candidate/camera_calibration.json" \
NOESIS_V3DT_AUTOGEN_CAMINFO=0 \
python3 DS9/noesis/ds9_runtime.py \
  --tracking-mode v3dt \
  --pipeline-config DS9/config/infer_v3dt_living_kitchen_tracking_candidate.yaml \
  --cameras-config DS9/config/cameras_v3dt.yaml \
  --pgie-profile yolo26_seg \
  --size s
```

## Accepted Kitchen geometry

The Kitchen phone walk is
`data/mapanything_phone_scans/20260814-190412-ea26c040`. Its admitted Scene
Prior is `sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483`: alignment passed,
167,338 points were selected, and the preview is bound to the Kitchen camera in
`backend_world_m`. The resulting camera center and heading agree with the
existing no-image-Y-flip V3DT calibration, so this iteration did not invent a
second Kitchen transform or alter the non-V3DT calibration.

## Corrected one-person result

The July 3 Kitchen clip contains one person. Earlier scoring was invalidated
because wall portraits survived an analytics reload and entered the tracker as
people. The profile now selects a V3DT-coordinate analytics bundle before the
tracker. It removes the portraits without affecting the walking path, and the
YAML and generated INI are checked for exact equivalence.

| July one-pass metric | Non-V3DT | Kitchen SV3DT |
| --- | ---: | ---: |
| Track coverage | 100% | 92.22% |
| Raw tracker IDs | 6 | 8 |
| Canonical StableIDs | 1 | 2 |
| ReID embedding availability | 96.60% | 100% |
| BBox3D / world availability | 0% / 96.27% | 100% / 100% |
| World-step p95 | n/a | 0.237 m |
| World points inside current static bounds | n/a | 92.97% |

The 2026-08-18 accelerated repeated replay is the direct regression check for
the optimized settings. Static detections were zero in Living Room and Family
Room, Kitchen never exceeded one simultaneous person track, and all 673 track
rows carried both BBox3D and valid world output. Twenty-four of 25 raw
tracklets reused StableID 1000. One seven-row tracklet minted StableID 1001 at
a file-loop boundary; the test replays source time much faster than wall time,
so weakening the live anti-merge gate to hide that artifact is not justified.

The profile confirms a household identity after one V3DT embedding. That
override is parsed only for V3DT/SV3DT/MV3DT; the baseline remains at its
existing confirmation policy. The corrected person cuboid is also profile-
local and was visually checked at doorway, mid-room, and near-camera positions
with its bottom-face center beneath the feet.

## MV3DT gate

The synchronized July Kitchen and Family Room clips contain a real common-FOV
doorway interval at approximately 47.0-49.5 seconds. That proves an overlap
edge exists. It does not supply a trustworthy shared world frame: the latest
three-room fusion remains `review_only`, with registration rejected for
training and held-out observation error, per-view deformation, and temporal
holdout failure. Keep MV3DT disabled until a common Kitchen/Family transform
passes those gates; per-room Scene Priors remain reconstruction evidence, not
cross-camera tracking authority.
