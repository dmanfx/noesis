# Living-room + family-room optimized V3DT profile

This cumulative DS9 profile keeps the accepted living-room tuning and adds the
Family Room calibration derived from its 2026-08-10 phone walk. It is isolated
from the non-V3DT tracker and from the locked V3DT baseline assets.

Select it with:

```bash
NOESIS_CAMERA_CALIBRATION_FILE="$PWD/DS9/config/v3dt/living_family_phone_optimized/camera_calibration.json" \
NOESIS_V3DT_AUTOGEN_CAMINFO=0 \
python3 DS9/noesis/ds9_runtime.py \
  --tracking-mode v3dt \
  --pipeline-config DS9/config/infer_v3dt_living_family_phone_optimized.yaml \
  --cameras-config DS9/config/cameras_v3dt.yaml \
  --pgie-profile yolo26_seg \
  --size s
```

Keep camInfo autogeneration disabled. Living Room and Family Room use reviewed
no-image-Y-flip matrices, while the untouched non-V3DT lane retains its own
tracker and configuration.

## Family Room geometry

The Family Room phone scan is
`data/mapanything_phone_scans/20260810-215847-571c6efe`. The selected
MapAnything + DA3 workflow uses DA3 poses and reliable sparse depth, followed
by disagreement-aware consensus fusion. The fixed-camera reconstruction
remains the world-alignment authority and the phone cloud is validation
evidence, not a runtime tracker input.

The prior calibration faced away from every saved Family Room reconstruction.
The phone/static refinement restores the full fixed-camera rotation (yaw,
pitch, and roll), preserves the camera center, and regenerates only
`camInfo_family-room.yml`. A fresh alignment with this calibration requires no
orientation workaround and passes all quality gates:

| Geometry metric | Result |
| --- | ---: |
| Phone-to-static median distance | 0.110 m |
| Phone points within 0.30 m of static | 83.44% |
| Static points within 0.30 m of phone | 77.69% |
| Phone-visible fixed-camera overlap | 99.37% |
| Target fixed-camera coverage | 88.08% |

The prior-conditioned MA+DA3 fusion also passes in the corrected frame: 81.85%
of phone points are within 0.30 m of static, phone-visible reprojection overlap
is 99.79%, and its held-out temporal reprojection median is 0.043 m.

## Corrected one-person July replay

The July 3 clips contain one physical person. Analytics exclusions run before
tracking and now remove both Kitchen wall portraits and the Family Room TV
reflection. The authoritative analytics YAML and generated exclusion INI are
kept byte-equivalent by a regression test.

| Family Room metric | Non-V3DT | Pitch-only V3DT | This profile |
| --- | ---: | ---: | ---: |
| Track coverage on baseline-positive frames | 100% | 99.04% | 97.60% |
| Raw tracker IDs | 1 | 4 | 1 |
| Canonical StableIDs | 3 | 2 | 1 |
| ReID embedding availability | 94.73% | 100% | 100% |
| BBox3D / world availability | 0% / 83.71% | 100% / 100% | 100% / 100% |
| World-step p95 | n/a | 0.343 m | 0.332 m |
| World points inside Family Room bounds | n/a | 0% | 100% |
| Concurrent physical-person tracks | 1 | contaminated in old scoring | 1 |

This is comparable to the accepted living-room result: one continuous raw
track, one canonical StableID, complete embeddings and 3D/world output, and
97.6% versus 98.5% tracked coverage. The few large raw-world steps coincide
with abrupt detector-box truncation; the operational BEV path already applies
its bounded motion smoother.

## Kitchen status

Kitchen remains a tracking-only candidate in this bundle. With the corrected
one-person scoring it reaches 92.22% coverage, uses eight raw IDs versus six in
non-V3DT, and produces two canonical StableIDs. Do not treat it as optimized or
promote it yet. Repeat this geometry-first process after the Kitchen phone walk
is available.
