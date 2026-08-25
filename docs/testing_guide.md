# Noesis testing guide

Status: direct native-host validation policy, 2026-08-25.

## Principle

Test the capability that changed and stop when its producer, contract, direct
consumer, and practical application path are proven. Ordinary development does
not use release candidates, selectors, state clones, bundle publication,
rollback rehearsal, complete-suite repetition, or long soak tests.

## Escalation ladder

1. Syntax or static check for changed files.
2. Focused unit tests.
3. Focused component/contract tests.
4. One direct live or recorded smoke.
5. Broader regression only when the observed dependency surface demands it.

A docs-only change uses the docs consistency check and `git diff --check` only.
Unchanged model builds, native binaries, registrations, and prior test results
should be reused.

## Select tests by capability

| Change | Minimum meaningful evidence |
| --- | --- |
| Pipeline/config | Config parse/build plus a short source-progress smoke |
| Detector/parser/engine | Engine deserialize, output/parser contract, affected detections on one representative input |
| Tracker/ReID/pose | Metadata bridge test plus tracking/identity/pose consumer smoke |
| Depth | Profile/shape/registration checks plus the affected depth or floorplan request |
| World/BEV | Observation→world commit→BEV consumer test; inspect dots/trails when visualization changed |
| WebSocket/REST | Exact schema/auth test and one direct client request |
| WebRTC/media | One authenticated offer/answer and decoded frames |
| Native extension/plugin | Rebuild affected binary once, load/factory inspection, one direct metadata/path test |
| Frontend | Focused component tests, one production build, and visual check of the changed view |
| Systemd lifecycle | Unit verification plus one bounded start/readiness/stop or restart when the unit changed |

Do not use a liveness response as evidence for perception quality or
performance. Do not use unrelated frontend tests to validate a model change.

## Direct physical world/BEV check

Software self-consistency is not absolute placement evidence. For a tracking
or camera-to-PCF change, use this small independent test:

1. Pick five to eight unambiguous floor locations per affected room and record
   their PCF X/Z coordinates from fixed physical landmarks. Put removable tape
   at those locations.
2. Have one person place both heels on each mark for two seconds, then walk a
   marked path. Record the camera, exact media/frame cohort, posture, and the
   expected X/Z; do not derive expected coordinates from tracker output.
3. Join tracking and BEV only on their complete exact cohort. Require every
   visible, world-valid track to produce one accepted, predicted, or explicitly
   held/dropped BEV result. Verify the backend world-to-BEV transform
   independently within one PCF cell (2.5 cm for current priors).
4. Compare an accepted head with the measured mark. A predicted head is a
   bounded continuity output, not fresh absolute-position evidence; a held head
   must remain at the last-good mark and must not extend the trail. The
   practical initial target is at most 20 cm p95 and 30 cm maximum for clear
   standing/walking contacts, stationary jitter at most 8 cm p95, and no
   unexplained jump over 25 cm. For every continuous trail segment, divide each
   world step by exact cohort time and require it to remain at or below the
   configured human speed. A larger relocation is valid only when the same
   cohort declares `trail_break_required=true` and increments
   `trail_segment_id`. Test seated contacts separately at known chair/seat
   support locations.
5. Reproject the same canonical world point (accepted or predicted) into OSD.
   Its endpoint must remain on the same-frame contact anchor within 25 pixels;
   a missing or out-of-frame anchor must break the trail, never draw to a frame
   edge.
6. Include a brief disappearance and a seated/occluded interval. The first
   exact absent row must remove world state from active authority and emit its
   tombstone. A return within 350 ms may restore state and reuse the lifecycle
   generation only for the same camera/tracker key with matching bbox position
   and scale. Move the returning box elsewhere, make its scale incompatible,
   or delay it beyond 350 ms and verify it starts cold with a new generation.
   No BEV/OSD trail may connect across either absence. Reject-driven prediction must
   stop at 0.40 seconds. Generic holds must stop at 0.40 seconds, while a
   seated/lying hold may reach 2.0 seconds only with stationary same-frame bbox
   evidence and must never append a trail.
7. For a sub-second tracker metadata gap, verify the same settled StableID is
   retained only when the same-camera tracker ID and strict bbox-continuity
   gates agree. A moved box, different tracker ID, duplicate same-frame claim,
   or gap over 0.75 seconds must not use the continuity shortcut.

A recorded motion clip is useful for determinism, continuity, lifecycle, and
performance checks, but it does not replace the measured floor marks for
absolute real-world accuracy.

For the universal resolver, keep the goals separate and direct:

1. At one exact frame, assert every independently valid floor/depth source
   survives into the bounded hypothesis set with the same camera, source,
   tracker, lifecycle generation, track key, frame, observation time,
   calibration revision, world revision, and transform SHA-256.
2. Feed a close floor/depth pair and prove both contributor IDs are present and
   the result differs numerically from both inputs. Feed a pair more than the
   generic compatibility distance apart and prove it is primary/alternate,
   never fused or clamped. Feed three candidates where each secondary agrees
   with the primary but the secondaries disagree with each other; prove only a
   mutually compatible subset can contribute.
3. Perturb a floor anchor by its pixel sigma and a depth anchor by its residual
   sigma; verify the reported anisotropic covariance remains finite/PSD and
   grows for shallow incidence, weak support, or occlusion.
4. Put one candidate outside an authored PCF wall and one inside. Verify PCF
   changes preference/quality while leaving every candidate coordinate
   unchanged. Put two compatible candidates well beyond the same boundary or
   measured extent and prove their agreement cannot fuse away that strong
   conflict: the result remains diagnostic and weak. Do not use unlabeled
   obstacle cells to reposition seated people.
5. After PersonGroundState, verify an accepted filtered point carries
   displacement-inflated covariance. A rejected measurement followed by
   prediction/hold must not inherit the current measurement covariance.
6. Toggle `Localization details` on the normal dashboard. The overlay must use
   the exact current cohort and revision, show no last-seen candidates, and
   leave the canonical dot/trail byte-for-byte unaffected. Where the retired
   room policy has a usable current-frame candidate, its dashed legacy point
   must match that historical selection rule; toggling details off must remove
   the comparison payload without changing the canonical point.
7. Give canonical global fusion a held/predicted row, a missing transform hash,
   and two different target-frame revisions. Prove none becomes fresh fused
   evidence. Then fuse two same-target observations with correlated covariance
   and prove the off-diagonal terms survive conservatively.

## Native runtime checks

Load the installed environment without printing secrets:

```bash
NATIVE_ENV_FILE="$(
  systemctl --user show noesis-appliance.service \
    -p EnvironmentFiles --value --no-pager |
    awk '$1 ~ /\/native\.env$/ { print $1; exit }'
)"
test -n "$NATIVE_ENV_FILE" && test -f "$NATIVE_ENV_FILE"
set -a
. "$NATIVE_ENV_FILE"
set +a
```

Static native preflight (does not open cameras):

```bash
"${NOESIS_DS91_NATIVE_ROOT}/venv/bin/python" \
  DS9/scripts/run_canonical_runtime_host.py check
```

Installed readiness (REST capability/deployment identity and WebSocket health):

```bash
"${NOESIS_DS91_NATIVE_ROOT}/venv/bin/python" \
  DS9/scripts/native_noesis_wait_ready.py --timeout-ms 15000
```

Service status:

```bash
systemctl --user show noesis-appliance.service \
  -p ActiveState -p SubState -p Result -p ExecMainStatus --no-pager
```

## Direct media and telemetry smoke

Use these only when the changed capability can affect media or runtime
telemetry:

```bash
"${NOESIS_DS91_NATIVE_ROOT}/venv/bin/python" \
  scripts/webrtc_gateway_smoke_test.py \
  --ws ws://127.0.0.1:6008 \
  --auth-token-file "$NOESIS_INTERNAL_AUTH_TOKEN_FILE" \
  --duration 8 --pt 103 --min-rtp 10 --min-decoded 1
```

```bash
"${NOESIS_DS91_NATIVE_ROOT}/venv/bin/python" \
  DS9/scripts/zero_copy_stats_smoke_test.py \
  --no-spawn --stats-ws ws://127.0.0.1:6008 \
  --auth-token-file "$NOESIS_INTERNAL_AUTH_TOKEN_FILE" \
  --duration-s 8 --startup-timeout 5 \
  --max-p99-ms 3 --max-violations 0 --log-path /dev/null
```

RTSP is disabled in the canonical graph; an RTSP probe is not a valid baseline
test. Use WebRTC for media validation.

For a recorded-file replay where analytics must observe every frame at the
clip's media rate, add this explicit top-level policy to the replay YAML:

```yaml
recorded_replay:
  realtime: true
  preserve_frames: true
```

This inserts a clock-synchronizing `identity` only for local `file:` MP4/MKV
sources and makes their decode queues non-leaky. It is intentionally absent
from the canonical live configuration; do not infer it from a file URI or use
it as a throughput benchmark.

## Performance checks

Performance work starts with the `deepstream-profile-pipeline` skill and uses
matched cameras/files, configs, model realization, warm-up, and duration. For a
localized regression, collect only enough time to distinguish the change from
normal variance.

Do not improve a performance number by silently reducing the model, input
resolution, inference interval, tracker quality, depth/pose cadence, or enabled
outputs. Hold those capabilities fixed unless the user explicitly requests a
quality/throughput comparison.

Measure the affected layers independently:

1. per-camera decoded/dewarped FPS, progress age, stalls, and recoveries;
2. affected callback/stage latency, including occupied-scene tail values;
3. encoded H.264 access-unit FPS, gap percentiles/maxima, and feeder/queue drops;
4. authenticated WebRTC decoded frames; and
5. the affected perception outputs and canonical world/BEV behavior.

For the Identity-v2/publication isolation specifically, rerun the same paced
three-room files for at least 100 seconds (past the prior source-1 frame 2283
failure). The direct pass conditions are: the process is still running; every
received tracking cohort has its exact paired BEV cohort; source frame IDs keep
advancing; `tracking.publication_worker.overflow_total` and `.failures_total`
remain zero; and the canonical pending gauge does not trend upward. Shadow
`coalesced_total` or shadow-only drop totals are allowed because they describe
diagnostic freshness, but `identity_v2.shadow.worker_failures_total` must be
zero in a healthy store. Compare `tracking.publication_worker_item`,
`tracking.publication_worker_queue_wait`, `tracking.publish`,
`bev.render_and_publish`, `identity_v2.shadow_process_source_frame`, and
`identity_v2.shadow_queue_wait` timings to distinguish work from backlog. Then
confirm one moving person remains continuous on OSD and BEV;
that visual check confirms behavior but does not replace the counters.

The dashboard's combined receiver FPS, an open port, or a single utilization
snapshot is not proof of source or encoded cadence. Use a motion/occupancy-heavy
recorded sample and, when practical, a bounded live multi-person check. Visual
review confirms macroblocking/corruption; numeric source, encoded, and decoded
measurements establish stalls and drops.

A single resource snapshot remains adequate for changes that do not claim an
optimization. The full constraints and accepted reference are in
[`performance_invariants.md`](performance_invariants.md) and
[`runtime_baseline.md`](runtime_baseline.md).

Current practical references are recorded in `runtime_baseline.md`.

## Documentation validation

```bash
./scripts/check_agents_docs_consistency.py
git diff --check
```

No application or package build is required solely because documentation
changed.
