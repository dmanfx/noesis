# Noesis testing guide

Status: direct native-host validation policy, 2026-08-15.

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
