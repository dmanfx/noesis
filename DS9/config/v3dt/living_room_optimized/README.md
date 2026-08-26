# Living-room optimized V3DT profile

Status: archived experimental profile. This Living Room profile is not an
accepted MV3DT lane; Living Room remains local-only. The separate
Kitchen/Family Room lane is available only as an explicit opt-in. Keep this
material as prior tuning evidence and do not launch this archived profile
without explicit user direction.

This directory is an isolated DS9 SV3DT tuning profile. It does not replace
the locked V3DT assets in `caminfo_baseline/` and does not change the
non-V3DT tracker.

Historical reproduction command (not a current operating command):

```bash
NOESIS_CAMERA_CALIBRATION_FILE="$PWD/DS9/config/v3dt/living_room_optimized/camera_calibration.json" \
NOESIS_V3DT_AUTOGEN_CAMINFO=0 \
python3 DS9/noesis/ds9_runtime.py \
  --tracking-mode v3dt \
  --pipeline-config DS9/config/infer_v3dt_living_room_optimized.yaml \
  --cameras-config DS9/config/cameras_v3dt.yaml \
  --pgie-profile yolo26_seg \
  --size s
```

Keep camInfo autogeneration disabled for this profile. The living-room model
uses a phone-walk/static-world rotation refinement and no image-Y flip, while
the kitchen and family-room files remain byte-identical copies of their locked
baseline camInfo. The phone-walk point cloud is validation evidence, not a
runtime tracker input.

Tracker changes are confined to this profile: unreliable pose-height fitting
is gated, VPI NvDCF uses feature level 3 with search padding 2, and StableID
retains a raw V3DT track for 0.5 seconds across short metadata gaps.

## July living-room replay result

The 2026-08-10 comparison scored the same 802 baseline-positive frames from
the early-July three-camera replay, while gating decisions on living room only.

| Metric | Non-V3DT | Previous V3DT | This profile |
| --- | ---: | ---: | ---: |
| Track coverage | 100.0% | 29.3% | 98.5% |
| Raw tracker IDs | 2 | 51 | 3 |
| Median raw-track lifetime | 401 frames | 2 frames | 224 frames |
| Stable IDs | 4 | 2 | 1 |
| ReID embedding availability | 92.27% | 63.40% | 100.0% |
| BBox3D availability | 0% | 100% | 100% |
| World-step p95 | 0.369 m | 4.511 m | 0.215 m |
| Inside phone-walk full X/Z bounds | 70.69% | 37.02% | 95.82% |

The profile sustained 30.5 living-room frames per second. Its three raw IDs
represent one avoidable mid-walk split plus the new ID after the person leaves
and later re-enters; StableID remains `1000` across all tracked scored frames.

## Reusable room-tuning model

Use this sequence for kitchen, family room, and later V3DT camera work:

1. Keep the accepted non-V3DT and V3DT bundles untouched. Copy the pipeline,
   tracker, calibration, and all required camInfo files into a named candidate.
2. Replay the same synchronized recording through non-V3DT, current V3DT, and
   the candidate. Exclude model warm-up and score only the room being tuned.
3. Diagnose geometry before association thresholds. Compare detector boxes to
   V3DT positive and shadow boxes; inspect fitted-height clamps, projected
   image-foot error, world-step continuity, and room-bound violations.
4. Refine only the target camera's extrinsics/camInfo. A phone-walk consensus
   cloud is useful as independent rotation and room-bounds evidence, but is not
   a tracker input and must not replace the static-camera calibration authority.
5. If pose-height fitting is demonstrably unstable, gate it with the documented
   `minPoseConfidence` control while retaining pose output for other consumers.
6. Tune documented NvDCF controls after geometry is credible. Sweep one change
   at a time; for the living room, feature level 3 plus search padding 2 gave
   the best continuity/resource tradeoff. Reject settings that improve coverage
   by publishing implausible 3D motion.
7. Apply StableID gap retention only in V3DT and only through an explicit
   profile value. Keep the default at zero so existing V3DT and non-V3DT
   profiles preserve their lifecycle behavior.
8. Promote a room candidate only when tracking coverage, raw-ID fragmentation,
   StableID switches, embedding/BBox3D availability, world continuity, spatial
   bounds, throughput, and GPU/CPU/RAM use have all been compared directly.

For mixed per-camera profiles, keep `NOESIS_V3DT_AUTOGEN_CAMINFO=0`. The current
generator has one global image-flip convention and must not overwrite a tuned
room while regenerating untouched cameras under a different convention.
