# Kitchen V3DT tracking candidate

Status: archived experimental profile. MV3DT is disabled in the canonical
native DS9.1 application. The Kitchen geometry is not accepted, so this profile
must not be launched or promoted.

This profile preserves the accepted living-room work and the best current
Kitchen no-image-Y-flip projection, but it is not an optimized or promoted
Kitchen profile. It remains isolated from the non-V3DT lane.

Historical reproduction command (not a current operating command):

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

## Corrected one-person result

The July 3 Kitchen clip contains one person. Earlier scoring was invalidated
because wall portraits survived an analytics reload and entered the tracker as
people. After synchronizing the exclusion YAML and INI, Kitchen has zero
duplicate-person frames, but this V3DT candidate still trails non-V3DT:

| Metric | Non-V3DT | Kitchen candidate |
| --- | ---: | ---: |
| Track coverage | 100% | 92.22% |
| Raw tracker IDs | 6 | 8 |
| Canonical StableIDs | 1 | 2 |
| ReID embedding availability | 96.60% | 100% |
| BBox3D / world availability | 0% / 96.27% | 100% / 100% |
| World-step p95 | n/a | 0.237 m |
| World points inside current static bounds | n/a | 92.97% |

The Kitchen phone walk is the remaining geometry input. Use the Family Room
workflow as the template: validate the phone scan against fixed-camera world,
correct the full camera rotation if supported, regenerate only Kitchen camInfo,
and rerun this same one-person replay before promotion.
