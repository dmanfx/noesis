# DS9.1 direct validation runbook

Use only the section affected by the change. This runbook validates the native
application directly; it does not stage releases, build selectors, clone state,
publish candidates, or rehearse rollback.

## Environment

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

PYTHON="${NOESIS_DS91_NATIVE_ROOT}/venv/bin/python"
```

Never print the loaded environment. It contains paths to private credentials.

## Native preflight

Use when platform, Python dependencies, native artifacts, engine realization,
or canonical config changed:

```bash
"$PYTHON" DS9/scripts/run_canonical_runtime_host.py check
```

This is read-only and does not open cameras. If the managed service already
owns 6008/8080, occupied ports are reported as state rather than a reason to
start another runtime.

## Focused tests

Select the smallest files that directly cover the change. Examples:

```bash
"$PYTHON" -m pytest -q DS9/tests/test_runtime_container_boundary.py
```

The historical filename above contains both boundary contracts; choose a more
specific test when available. Do not infer container authority from the name.

```bash
"$PYTHON" -m pytest -q \
  DS9/tests/test_native_extension_origin.py \
  DS9/tests/test_native_artifact_provenance.py
```

```bash
"$PYTHON" -m pytest -q \
  DS9/tests/test_world_snapshot_runtime.py \
  tests/test_active_floorplan_registry.py
```

Replace examples with the directly affected contract tests; do not run all of
them by default.

## Installed readiness

Use after a runtime or lifecycle change, or when confirming the app is running:

```bash
systemctl --user show noesis-appliance.service \
  -p ActiveState -p SubState -p Result -p ExecMainStatus --no-pager

"$PYTHON" DS9/scripts/native_noesis_wait_ready.py --timeout-ms 15000
```

Readiness proves authenticated REST capability/deployment identity and
WebSocket health. It is not a perception-quality or performance result.

## WebRTC media

Use only when pipeline output, encoder, SHM, gateway, or media dependencies
changed:

```bash
"$PYTHON" scripts/webrtc_gateway_smoke_test.py \
  --ws ws://127.0.0.1:6008 \
  --auth-token-file "$NOESIS_INTERNAL_AUTH_TOKEN_FILE" \
  --duration 8 --pt 103 --min-rtp 10 --min-decoded 1
```

The canonical graph has RTSP disabled. Do not probe 8554 as a baseline check.

## Source progress and zero-copy boundary

Use for graph, source, telemetry, or boundary-performance changes:

```bash
"$PYTHON" DS9/scripts/zero_copy_stats_smoke_test.py \
  --no-spawn \
  --stats-ws ws://127.0.0.1:6008 \
  --auth-token-file "$NOESIS_INTERNAL_AUTH_TOKEN_FILE" \
  --duration-s 8 --startup-timeout 5 \
  --max-p99-ms 3 --max-violations 0 --log-path /dev/null
```

Require advancing source samples, no pipeline error, and zero core violations.
The 3 ms boundary is serialization/dispatch, not end-to-end camera latency.

## BEV/dashboard

For BEV or tracking presentation changes:

1. Run the focused backend and frontend tests for the changed payload/parser.
2. Open the oai2-fe dashboard through Menon.
3. Verify the affected camera's PCF floorplan loads.
4. When people are present, confirm dots/trails match the associated committed
   tracking cohort. Empty occupancy is a valid zero-dot result.
5. Check browser console/network only for the affected message path.

Do not substitute a static PCF point cloud for tracking authority.

## Depth and floorplan

Use the exact request path changed:

- DAv2 changes: profile/registration test plus observed object/world depth.
- MapAnything changes: profile/engine test plus one admitted manual request.
- PCF/Scene Prior changes: catalog/schema/loader test plus one
  `scene_prior_only` floorplan response and dashboard render.
- Calibration changes: exact camera fingerprint/registration and its direct
  projection consumer.

Do not rerun every depth backend, reconstruct every room, or run AMC unless the
task explicitly changes those capabilities.

## Model or native artifact

1. Build the affected engine/binary once with the native host helper.
2. Run `trtexec --loadEngine` or the relevant module/factory load check.
3. Run the parser/tensor/metadata contract test.
4. Exercise one representative direct consumer.

Use `../DS9_REBUILD_AND_SMOKE_GATES.md` for exact artifact families.

## Performance

Use matched inputs and the `deepstream-profile-pipeline` skill. For an observed
regression, one bounded live or non-July sample-MP4 pressure run should report
per-camera/aggregate FPS, CPU, GPU, VRAM, and errors. Do not turn it into a soak
or broad release suite unless the measurement is ambiguous.

## Documentation-only changes

```bash
./scripts/check_agents_docs_consistency.py
git diff --check
```
