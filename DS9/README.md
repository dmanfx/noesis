# DeepStream 9.1 application

`DS9/` is the canonical and only supported DeepStream stack for Noesis. It runs
directly on the host; Docker, DeepStream 8, and DeepStream 9.0 are not supported
runtime or build alternatives. Inert compatibility scripts remain only until
the legacy-removal checklist is completed.

The directory name and compatibility fields use `DS9`, while the required SDK
version is exactly **9.1**.

## Runtime authority

| Concern | Authority |
| --- | --- |
| Host supervisor | `scripts/run_canonical_runtime_host.py` |
| Entrypoint/import gate | `noesis/ds9_runtime.py` |
| Runtime assembly | `noesis/ds9_runtime_core.py` |
| Pipeline implementation | `noesis/pipelines/` |
| Canonical config | `config/infer.yaml` |
| Model/parser configs | `pipelines/` |
| Native bridge sources | `native/` |
| GStreamer plugin sources | `gst-plugins/` and `csrc/` |
| Expected assets | `asset_manifest.yaml` |
| Runtime ownership matrix | `docs/runtime_ownership.yaml` |

## Required host stack

- DeepStream 9.1.0
- CUDA 13.2
- TensorRT 10.16.0.72
- GStreamer 1.24.2
- Python 3.12 native virtual environment
- NVIDIA driver 595.58.03 or newer

`scripts/run_canonical_runtime_host.py check` verifies those versions, exact
Python package origins, private inputs, the selected asset realization, and
port ownership without opening the cameras.

## Canonical lane

- YOLO26-m FP16 detector.
- Native ROI exclusion and NvDCF baseline tracking.
- Swin Tiny ReID and YOLO26 pose.
- Always-on DAv2 metric object/tracking depth.
- Request-gated MapAnything FP32 full-frame depth.
- Canonical observation/world/identity publication.
- Tiled OSD followed by one H.264 encode and private SHM/WebRTC delivery.
- JSON BEV in `camera_local_ground_m`; no BEV JPEG stream.
- RTSP output disabled.

See [`PIPELINE_GRAPH.md`](PIPELINE_GRAPH.md) for the exact topology.

## Native operation

The installed systemd user service is the normal owner. Do not launch a second
copy on ports 6008/8080.

```bash
systemctl --user status noesis-appliance.service
```

For a non-mutating native check:

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

"${NOESIS_DS91_NATIVE_ROOT}/venv/bin/python" \
  DS9/scripts/run_canonical_runtime_host.py check
```

Use [`docs/runtime_host_boundary.md`](docs/runtime_host_boundary.md) for the
environment and lifecycle contract and [`docs/validation_runbook.md`](docs/validation_runbook.md)
for focused application checks.

## Models and native artifacts

The host supervisor selects an external artifact root through
`NOESIS_DS9_ARTIFACT_ROOT`. `asset_realization.json` binds the installed engine
files to `asset_manifest.yaml`. Runtime binaries must originate from native
DS9.1/CUDA 13.2 builds; archived DS8/9.0 or container-layer artifacts fail
closed.

Build only what changed:

- Native extensions: `scripts/build_all_native_ds9.sh` or the specific build
  helper.
- Engine maintenance: `scripts/run_canonical_engine_maintenance_host.sh`.
- Focused rebuild/load guidance: `DS9_REBUILD_AND_SMOKE_GATES.md`.

## Capability status

Baseline tracking is active. MV3DT and AMC are disabled until Kitchen geometry
and synchronized Kitchen/Family Room overlap evidence are accepted. Presence of
V3DT/MV3DT configs or engines does not enable them.

## Documentation

- [`docs/README.md`](docs/README.md): DS9.1 documentation index.
- [`docs/deepstream_9_1_agent_skills.md`](docs/deepstream_9_1_agent_skills.md):
  required skill routing and repo pin overrides.
- [`docs/canonical_world.md`](docs/canonical_world.md): observation/world
  authority.
- [`docs/bev_capture_event_integration.md`](docs/bev_capture_event_integration.md):
  BEV and capture publication.
- [`../docs/runtime_baseline.md`](../docs/runtime_baseline.md): accepted runtime
  and performance baseline.
- [`docs/history/README.md`](docs/history/README.md): migration, container, and
  experiment archive.
