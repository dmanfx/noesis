# Partial-body depth and BEV review — 2026-10-09

## Implemented behavior

Baseline DS9.1 now keeps independent ankle and body depth attempts. A narrow
torso core accepts horizontal anatomy; a fallback central shoulder band needs
two observed shoulders, not covered hips/feet. Sitting/lying prioritizes qualified
body evidence. Unknown posture stays unknown: current qualified shoulder depth
can establish a gravity-projected body footprint after the existing three-sample
cold consensus. No hidden joints, body height, room offset, or new floor plane
is assumed. Registration, detector confidence, exact tensor/geometry provenance,
contamination, physical motion and canonical service gates remain intact.

Shoulder and torso depth retain different continuity bases. A switch without
independent coherent motion holds the accepted coordinate instead of inventing
a translation; an accepted corroborated reanchor breaks the trail. Cached
depth never counts as fresh consensus. Inferred-height gravity is an alternate,
not a fusion contributor to measured support/range.

## Focused checks

262 tests passed across the affected object-depth, hook/resolver, person-ground
state, partial-body and generic measurement-resolver files. Syntax compilation,
`git diff --check` and the active AGENTS/docs consistency check passed.

The independent geometry control projects an observed shoulder point
`[0.45, 1.0, 4.6]` through the calibrated camera, samples its optical-Z at that
same UV, and emits `[0.45, 0.0, 4.6]` after three fresh observations. Covered
hips/ankles are absent, posture remains unknown, and the BEV footpoint transports
that canonical coordinate exactly. This proves the software path, not physical
accuracy in the room.

## Direct occupied producer/BEV evidence

The configured native process, PID 3303633, loaded the new sampling/admission
path. Concurrent work selected `yolo26-watch`, size M, with baseline tracking
and YOLO26 Small pose. That selection and all geometry/configuration were
preserved. We did not launch another runtime.

The ordinary read-only `get_floorplan` request with `scene_prior_only=true`
loaded the existing revision-bound PCF presentation. A 16-second authenticated
capture then received 288 tracking messages, 288 world snapshots and 290 BEV
frames. All three cameras had healthy active BEV publication, no failed camera
and no pending floorplan authority. Earlier absence of BEV frames was the normal
post-restart floorplan-request dependency, not a new position fallback.

Matched Family Room samples (all distances in metres):

| Source frame / stable ID | Producer canonical XYZ | BEV camera-local X/Z | Evidence |
| --- | --- | --- | --- |
| 2645 / 1013 | `[15.760192134, 0, 9.930904201]` | `[0.187816315, 2.614609925]` | Fresh exact torso depth; accepted canonical measurement; tensor age 0, uncached |
| 2651 / 1013 | `[15.699072112, 0, 10.101270723]` | `[0.136779448, 2.440956111]` | Fresh exact torso depth; accepted canonical measurement; tensor age 0, uncached |
| 2648 / 1013 | `[15.687393176, 0, 10.140276582]` | `[0.127405103, 2.401333185]` | Image-motion prediction, not fresh depth; BEV `worldAdmission=predicted` |

Fresh rows use `world_source=person_anchor_depth_only`, contact basis
`depth:pose_torso_support`, and `displaySource=world_to_camera_local`. Their
world frame revision is
`sceneprior_family-room_20260811T015847Z_8b80dc69a7c4`; transform identity is
`5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217`.
This is actual producer/consumer output, not fitted coordinates or synthetic
screenshots. The lower-level body sampler also retained rejected ankle-spread
attempts alongside accepted torso attempts in occupied live output.

### Final loaded-runtime check

After the one configured-service restart, PID 3309569 loaded the final resolver
and sampling changes with the same Watch-26m baseline selection. A bounded
12-second capture received 231 tracking cohorts, 231 world snapshots and 143
BEV frames after the ordinary PCF requests. All three BEV cameras were active
and healthy, with no runtime errors.

Using the current published calibration bundle to independently reapply the
world-to-horizontal-camera-ground transform, 93 matched canonical/BEV points
agreed to a maximum residual of `8.881784197001252e-16 m`. BEV admission labelled
85 points accepted and eight predicted; accepted here can include current floor
evidence and does not imply fresh depth. This particular startup window contained
no matched fresh body-depth acceptance; the fresh body examples above come from
the preceding loaded sampling run, not this final window.

All three sources progressed at the observed 13.6 FPS with no recovery attempts.
The canonical publication queue had zero pending items and high-water mark
three. Compact body statistic calls averaged 3.04 ms over 93 calls, with maximum
40.30 ms; four ankle calls averaged 6.84 ms. These are short occupied startup
observations under the concurrent detector selection, not a same-model
before/after throughput qualification or a long-run latency guarantee.

## Limits

- The original blanket screenshot is before evidence, not a matched-frame
  ablation: concurrent detector changes prevent attributing every live difference
  to this patch alone.
- These captures demonstrate current torso-depth and BEV delivery. They do not
  independently prove couch placement or qualify every shoulder-only pose.
- The browser dashboard was unavailable. Menon's configured gateway rejected
  the concurrently changed `DS9/scripts/run_canonical_runtime_host.py` checkout
  hash. We did not change that binding, fabricate a browser screenshot, or use
  diagnostic points around the canonical consumer.
- A person with insufficient observed anatomy, low detector confidence,
  contaminated or non-current depth, or rejected registration still cannot
  acquire a truthful fresh coordinate. Existing admitted continuity may hold or
  predict, with its original provenance.
- Shoulders are observed body anatomy, not measured covered feet or a proven
  full-body centre. A stationary anatomical-basis change can remain held; no
  generic shoulder-to-hip offset is invented.
- Native anatomy sampling uses at most three compact statistic calls per
  selected person, two on an ordinary ankle-plus-torso row. This is bounded,
  not a claim of zero added cost or unchanged throughput.

No commit or push was performed.
