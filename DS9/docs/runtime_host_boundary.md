# Native DS9.1 runtime boundary

Status: canonical host execution contract, 2026-08-15.

## Ownership

The systemd user service runs the repository checkout directly through
`DS9/scripts/run_canonical_runtime_host.py`. The supervisor performs a read-only
`check` before `run`, then `exec`s `DS9/noesis/ds9_runtime.py` as the long-lived
process. There is no container process, container filesystem, Docker socket, or
deployment-selector dependency.

## Required environment authorities

The installed environment supplies paths; repository code does not hardcode a
home or storage location.

| Variable | Purpose |
| --- | --- |
| `NOESIS_DS91_NATIVE_ROOT` | Python venv, GStreamer registry, native host support root |
| `NOESIS_DS9_ARTIFACT_ROOT` | Engines and `asset_realization.json` |
| `NOESIS_DS9_RUNTIME_ROOT` | Per-run state, build/cache, depth, and evidence roots |
| `NOESIS_WORLD_JOURNAL_PATH` | Canonical world journal |
| `NOESIS_IDENTITY_V2_STORE` | Identity database |
| `NOESIS_ANALYTICS_CONFIG` | Runtime analytics YAML |
| `NOESIS_ANALYTICS_EXCLUDE_CONFIG` | Runtime exclusion config |
| `NOESIS_SCENE_STORE_PATH` | Scene release store |
| `NOESIS_VIRTUAL_TWIN_ROOT` | Virtual-twin authority root |
| `NOESIS_CAMERA_SECRETS_FILE` | Private camera URI registry |
| `NOESIS_MAPANYTHING_API_KEY_FILE` | Private MapAnything service key |
| `NOESIS_INTERNAL_AUTH_TOKEN_FILE` | Private Noesis↔Menon bearer token |

The `DS9` spelling in environment and wire identifiers is the application
family name. It does not authorize SDK 9.0.

## `check` mode

`check` validates, without binding ports or opening cameras:

1. DeepStream 9.1, CUDA 13.2, TensorRT 10.16.0.72, GStreamer 1.24.2, and driver
   floor.
2. Python 3.12 venv origin and exact dependency constraints.
3. Private file ownership/mode and parseability.
4. DS9.1 artifact realization and the five canonical engines.
5. Pipeline/camera configuration and native extension/plugin availability.
6. Current occupancy of 6008 and 8080.

Run it with the installed venv after loading the environment:

```bash
"${NOESIS_DS91_NATIVE_ROOT}/venv/bin/python" \
  DS9/scripts/run_canonical_runtime_host.py check
```

## `run` mode and process environment

`run` refuses occupied canonical ports, creates one private run root, strips
container/selector variables, pins DS9.1/CUDA paths and the venv, establishes
health identity, disables RTSP, enables WebRTC, runs focused preflight, and
executes the DS9.1 entrypoint.

The canonical arguments select:

```text
DS9/config/infer.yaml
config/cameras.yaml
--pgie-profile yolo26 --size m --tracking-mode baseline
--ws-host 127.0.0.1 --ws-port 6008
--rest-host 127.0.0.1 --rest-port 8080 --enable-rest
```

Normal operation is through systemd. Direct `run` is for an explicitly isolated
development session after the managed service is stopped; never run both.

## Lifecycle and readiness

- Systemd owns restart-on-failure and bounded stop timing.
- Runtime startup is ready only after source progress and required capability,
  deployment-identity, and WebSocket health contracts agree.
- Shutdown stops control admission, closes and drains the runtime publication
  gate, then closes WebSocket egress and drains the remaining pipeline/workers.
- `DS9/scripts/native_noesis_wait_ready.py` is the direct readiness client.

## Security boundary

Noesis listens only on loopback. Menon owns LAN TLS, sessions, browser
authorization, dashboard delivery, and WebSocket ticketing. Secret values stay
in private files and are never passed to the browser or written into docs.

## Explicit non-authorities

- `DS9/scripts/run_canonical_runtime_container.py`
- Docker build/staging scripts and image IDs
- appliance releases, selectors, bundle manifests, and state-clone mechanics
- DS8/DS9.0 Python, native libraries, parsers, plugins, or TensorRT engines

Those files may remain temporarily for history/removal work, but the native
runtime must not read or invoke them.
