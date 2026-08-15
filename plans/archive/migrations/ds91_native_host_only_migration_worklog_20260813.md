# DeepStream 9.1 Native-Host-Only Migration Work Order

Status: native application parity restored and pressure validated; destructive
legacy/Docker cleanup still awaits explicit user acceptance, 2026-08-14.

## Objective

Move the canonical Noesis DeepStream 9.1 application completely out of Docker
and run it directly on this Ubuntu host. After acceptance, DeepStream 9.1 must
be the only executable DeepStream version: DeepStream 8 and 9.0 are retained
only as inert archived history, and Noesis must have no runtime or build-time
dependency on Docker.

This is an execution-backend migration, not a pipeline redesign. Preserve the
accepted three-camera DS9.1 graph, models, engines, configuration, state,
contracts, dashboard behavior, and approximately 30 FPS performance.

## Non-negotiable instructions

1. Read the root and `DS9/` `AGENTS.md` files, then use the repository
   `deepstream-dev` skill before changing DeepStream-facing code. Repository
   DS9.1 pins override the older pins in upstream skill examples.
2. Work directly in the real source checkouts. Do not create an appliance
   release, immutable candidate, state clone, deployment selector, bundle,
   promotion, canary, reviewer lane, or rollback rehearsal.
3. Validate components directly and run one bounded live application smoke.
   Do not run a full suite or repeatedly rerun unchanged checks.
4. Do not change the pipeline graph, model selection, inference precision,
   batching, tracking, depth, identity, analytics, WebRTC topology, or the
   performance correction that restored approximately 30 FPS.
5. Do not use Docker to install, build, test, inspect, or run any new native
   component. The currently running DS9.1 container may remain alive only as
   the pre-migration reference until the direct native run is ready.
6. Do not introduce a fallback. If the native path fails, stop, report the
   exact failure, and fix that path. Do not silently launch DS8, DS9.0, an old
   engine, a container, a degraded graph, or a substitute model.
7. Preserve all pre-existing dirty worktree changes. At plan creation the
   dirty paths were frontend files; re-run `git status --short` and never stage,
   overwrite, format, or commit unrelated changes.
8. Do not expose or archive secrets. Camera URLs, API keys, bearer tokens,
   private state databases, and credential contents must remain in their
   existing owner-only host locations.
9. Do not remove the accepted DS9.1 artifact root selected by
   `NOESIS_DS9_ARTIFACT_ROOT`. The current realization authority has SHA-256
   `9c3815bbf86eb41a94efb504fad79a9208c543e05f98c68d017cb1a8dcfd2b26`.
   Re-resolve the active value before acting because deployment metadata may
   have advanced.
10. Never run NVIDIA's broad historical `rm -rf` cleanup examples against
    globbed DeepStream paths. Remove only explicitly resolved packages and
    versioned paths after inspecting each target.
11. Never run `apt autoremove` as part of this work.
12. Do not stop or disable the ordinary system Docker daemon without handling
    the unrelated Z-Wave workload described below.

## Confirmed starting state

These facts were observed on 2026-08-13. Recheck them once at the start; do not
repeat expensive scans.

| Item | Observed value |
| --- | --- |
| OS | Ubuntu 24.04, kernel `6.8.0-134-generic` |
| GPU | NVIDIA GeForce RTX 3060 |
| Driver | `595.71.05` (meets the DS9.1 floor) |
| Python | `3.12.3` |
| GStreamer | `1.24.2` |
| Active host CUDA link | `/usr/local/cuda-13.0` |
| Installed host TensorRT | `10.13.3.9-1+cuda13.0` |
| Active host DeepStream link | DeepStream 8.0 |
| Installed legacy SDKs | DeepStream 8.0 files and `deepstream-9.0` package |
| Required native CUDA | `13.2` |
| Required native TensorRT | `10.16.0.72-1+cuda13.2` |
| Required SDK root | `/opt/nvidia/deepstream/deepstream-9.1` |
| Current application endpoints | WebSocket `127.0.0.1:6008`, REST `127.0.0.1:8080` |
| Current media path | H.264 SHM/WebRTC enabled; mosaic RTSP output disabled |
| Current accepted throughput | approximately `29.8-30 FPS` per source |

There are two separate Docker daemons:

- `noesis-ds9-secondary-docker.service` owns the canonical DS9.1 container and
  the socket below the external DS9 Docker root. This daemon is in scope for
  removal after native acceptance.
- The normal `docker.service` currently owns an unrelated long-running
  `zwavejs/zwave-js-ui` container. Stopping the normal Docker daemon will stop
  home automation. This is not part of the DS9.1 migration. Do not stop it
  unless the user explicitly accepts that outage or separately requests that
  workload's native migration. Noesis can be 100% Docker-free while the host
  still uses Docker for an unrelated application.

## Official installation authority

Use these official sources rather than remembered commands or third-party
guides:

- [DeepStream 9.1 native dGPU installation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html)
- [DeepStream 9.1 GitHub release assets](https://github.com/NVIDIA/DeepStream/releases/tag/v9.1.0)
- [DeepStream 9.1 migration guidance](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Migration_guide.html)
- [CUDA 13.2 Linux installation guide](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-installation-guide-linux/)

The official x86 package is
`deepstream-9.1_9.1.0-1_amd64.deb`. The official native prerequisites are
Ubuntu 24.04, GStreamer 1.24.2, driver 595.58.03 or later, CUDA 13.2, and
TensorRT `10.16.0.72-1+cuda13.2`.

## Target architecture

The completed process tree must be:

```text
systemd user service
  -> native DS9.1 environment/launcher
     -> DS9/noesis/ds9_runtime.py
        -> DS9/noesis/ds9_runtime_core.py
           -> native Service Maker / GStreamer / TensorRT / CUDA
```

It must not contain `docker`, `dockerd`, `containerd`, a container shim,
`noesis-runtime-adapter.mjs`, or
`run_canonical_runtime_container.py`.

The active runtime contract is:

- one explicit DS9.1 SDK home;
- one dedicated Python 3.12 virtual environment;
- one external DS9.1 artifact root;
- the existing mutable state and secret files, used directly at their host
  paths;
- the accepted baseline lane only;
- loopback REST and WebSocket endpoints;
- one H.264 encode feeding the existing SHM/WebRTC gateway;
- no mosaic RTSP output;
- no DS8 or DS9.0 imports, launchers, engines, native libraries, SDK paths, or
  automatic alternatives.

## Completion definition

The migration is complete only when all of the following are true:

- DeepStream 9.1, CUDA 13.2, and TensorRT 10.16.0.72 are native and exact.
- The Service Maker wheel and locked application dependencies load from the
  dedicated native virtual environment.
- The existing selected DS9.1 engines and native binaries load directly on the
  host, or an actually incompatible component has been rebuilt natively once.
- Noesis starts directly from systemd without a selector or Docker adapter.
- All three cameras advance near the accepted frame rate; detections, tracks,
  identity, depth/world output, WebRTC, REST, WebSocket, and the dashboard work.
- SIGTERM produces the existing orderly publication-gate and pipeline
  shutdown.
- The DS9 secondary Docker daemon is stopped and disabled.
- Active Noesis source, scripts, units, environment, and validators contain no
  Docker dependency.
- DS8 and DS9.0 SDKs, launchers, services, engines, plugin registrations, and
  active configs are absent outside the verified inert archive.
- The user has inspected the dashboard and explicitly accepted the native
  result before irreversible Docker/legacy cleanup is performed.

## Phase 0 — Establish a safe working boundary

- [x] Re-read `git status --short` in both Noesis and Menon. Record which dirty
  files predate this work. Do not require a clean tree and do not include those
  files in native-migration commits.
  _2026-08-13: Noesis HEAD `5d77c658e64152e95be12cd5a20597a8f25952a2` (`feature_DS8`) is dirty only in `oai2-fe/dist/index.html`, `oai2-fe/src/components/BevView.tsx`, `oai2-fe/src/components/DepthDrawer.tsx`, `oai2-fe/src/lib/pcfPresentation.test.mjs`, `oai2-fe/src/lib/primaryFloorplan.test.mjs`, `oai2-fe/src/lib/renderUtils.ts`. Menon HEAD `0a6a6fe673c9aec4c461b561cc1f260772082d09` (`feature`) has a large unrelated dirty tree; do not stage it._
- [x] Record the Noesis and Menon HEAD revisions, current user units, active
  Noesis process tree, exact DS9.1 artifact root, exact mutable state root, and
  secret file paths without printing secret contents.
  _2026-08-13: Live tree is `noesis-appliance.service` → `noesis-runtime-adapter.mjs` → `run_canonical_runtime_container.py` → DS9 container `68cd2b7c944e` running `DS9/noesis/ds9_runtime.py` from release `release-20260813-pcf-panel-v1`. Artifact root `/mnt/noesis_storage/noesis-ds9-artifacts-ds91-20260812` (`asset_realization.json` SHA-256 `9c3815bbf86eb41a94efb504fad79a9208c543e05f98c68d017cb1a8dcfd2b26`). Mutable world/identity/analytics live in `state-20260813-pcf-panel-v1/payload`; session state/depth/evidence under `/mnt/noesis_storage/noesis-ds9-runtime/{state,depth,evidence}/20260813t105327161768z-c7c944`. Secret files: `~/.local/state/noesis/secrets/camera_sources.json`, `mapanything_rpc.key`, `~/.local/state/noesis/gateway-token`. Menon source checkout `/home/mayor/Menon`; live Menon from the same release tree._
- [x] Inspect free space only on `/` and the selected external storage volume.
  Do not recursively scan model, cache, recording, Docker, or artifact trees.
  _2026-08-13: `/` 23G free (188G, 88% used); `/mnt/noesis_storage` 301G free (916G, 66% used)._
- [x] Confirm the driver, OS, GStreamer, Python, current CUDA/TensorRT packages,
  two Docker daemons, and unrelated Z-Wave container described above.
  _2026-08-13: Ubuntu 24.04, kernel `6.8.0-134-generic`, RTX 3060 driver `595.71.05`, Python `3.12.3`, GStreamer `1.24.2`, host CUDA link `cuda-13.0`, TensorRT `10.13.3.9-1+cuda13.0`, host DeepStream link `deepstream-8.0` plus `deepstream-9.0` package. `noesis-ds9-secondary-docker.service` owns the DS9.1 container; ordinary `docker.service` owns `zwavejs/zwave-js-ui:11.16.2` (up 4 weeks). Do not stop ordinary Docker._
- [x] Create a short changed-file allowlist for this migration. Expand it only
  when a direct dependency is proven. Likely surfaces are:
  - `DS9/scripts/run_canonical_runtime_container.py` and its focused tests;
  - a new `DS9/scripts/run_canonical_runtime_host.py` or a carefully renamed
    canonical supervisor;
  - `DS9/noesis/ds9_runtime.py` and runtime path helpers;
  - native build/maintenance scripts under `DS9/scripts/`;
  - `DS9/asset_manifest.yaml`, its schema, and focused validators;
  - Noesis/Menon systemd unit templates and readiness helpers;
  - runtime ownership, README, and agent guidance.
  _2026-08-13 allowlist: `plans/ds91_native_host_only_migration.md`; `DS9/scripts/run_canonical_runtime_host.py` plus shared host-path helper if extracted; `DS9/scripts/run_canonical_runtime_container.py` and `DS9/tests/test_runtime_container_boundary.py`, `DS9/tests/test_appliance_supervisor.py`; `DS9/noesis/ds9_runtime.py` / runtime path helpers; native/engine maintenance wrappers under `DS9/scripts/`; `DS9/asset_manifest.yaml` and focused validators; user systemd units/`~/.local/libexec/menon-appliance` plus only the Menon readiness files required for native identity; runtime ownership/README/AGENTS. Expand only with a proven dependency. Never stage pre-existing Noesis frontend dirty files or the unrelated Menon dirty tree._

Stop if the currently active state or artifact root cannot be resolved
unambiguously. Do not guess a path from an old plan.

## Phase 1 — Capture one useful reference baseline

- [x] Reuse the accepted 2026-08-12 evidence where its inputs are unchanged.
  Do not rerun the DS9.1 upgrade suite.
  _2026-08-13: Reused `/mnt/noesis_storage/noesis-ds9-runtime/evidence/direct-perf-fix-live-20260812` (accepted 29.8-30 FPS). Did not rerun the upgrade suite._
- [x] While the current container is healthy, take one 8-10 second live sample
  containing:
  - frame progression and FPS for family room, kitchen, and living room;
  - decoded WebRTC frame count;
  - one source/tracking/world capability snapshot;
  - one GPU and process CPU/RAM snapshot;
  - current pipeline and component latency summaries, if already exposed.
  _2026-08-13: 9.05s stats (10 samples) + 8s WebRTC. All three sources advanced (`living-room` +189, `kitchen` +189, `family-room` +189 frames). Instantaneous FPS 28.8-29.6 at sample start, dipped to 18.4 while the host `avdec_h264` WebRTC probe ran; WebRTC decoded 162 frames / 6178 RTP. REST capabilities 200 (`tracking_observations` and `global_world` healthy), deployment health 200, identity_health 200. GPU 6353 MiB / 16-21% util. Process RSS ~2360 MiB, zero-copy violations 0. Pipeline `latency_ms` is exposed but disabled (count 0)._
- [x] Save only a compact, secret-free JSON or text comparison record below
  the existing external runtime evidence root. Do not create a release bundle
  or evidence framework.
  _2026-08-13: Wrote owner-only `/mnt/noesis_storage/noesis-ds9-runtime/evidence/native-host-migration-20260813/container-reference-live.json`._
- [x] Treat approximately 30 FPS as the performance reference. The earlier
  10-17 FPS condition was a fixed CPU hot-path regression and is not an
  acceptable baseline.
  _2026-08-13: Performance reference remains 29.8-30 FPS from the 2026-08-12 accepted run. The Phase 1 dip during host software WebRTC decode is probe load, not a new baseline._

## Phase 2 — Create the inert legacy archive

The archive is requested historical preservation, not a rollback runtime.
Store it outside the checkout under an explicitly set
`NOESIS_LEGACY_ARCHIVE_ROOT`. Do not hardcode that machine-local path into
source or tracked documentation.

- [x] Build an explicit archive file list from tracked files and known
  versioned paths. Do not feed `/`, the repository root, a storage root, or an
  unrestricted recursive `find` into `tar`.
  _2026-08-13: Built from `git ls-files` prefixes plus six named host files. No recursive find._
- [x] Include repository-owned DS8 material:
  - DS8 runtime entrypoints and DS8-only adapters;
  - DS8 configs, plans, checklists, tests, migration docs, and historical
    contracts;
  - DS8-only build scripts, parser/plugin source, manifest records, and unique
    model metadata;
  - the exact Git revision and a tracked-file inventory.
  _2026-08-13: Archived DS8 entrypoints, plans/docs/tests, root-level parser `.so` files, and git revision `5d77c658e64152e95be12cd5a20597a8f25952a2` plus full tracked-file inventory._
- [x] Include repository-owned DS9.0 material:
  - 9.0 configs, manifests, Dockerfiles, launcher scripts, build defaults, and
    migration records;
  - unique 9.0 engine/native realization manifests and small provenance files;
  - the installed package/version inventory and official reconstruction URLs.
  _2026-08-13: Archived `DS9/docs/migration_state.md`, `DS9_PREP_DECISIONS.md`, old `asset_realization.json` / `canonical_sources.provenance.json` from the non-selected artifact root, DS 8.0/9.0 `version`+`uninstall.sh`, dpkg inventory, and reconstruction URLs. Current 9.1 Dockerfiles left in-tree until Phase 10. Vendor SDK/engine bytes not duplicated._
- [x] Include unique legacy binaries or engines only when they are
  repository-owned and reasonably sized. For vendor SDK packages and Docker
  layers, archive package names, versions, source URLs, and hashes rather than
  duplicating reproducible vendor bytes.
  _2026-08-13: Included tracked root parser `.so` files (15–232 KiB). Did not copy `/opt/nvidia/deepstream/deepstream-{8.0,9.0}` libraries or Docker layers._
- [x] Exclude all camera credentials, API keys, bearer tokens, private state
  databases, live scene/identity stores, recordings, caches, logs, core dumps,
  and current DS9.1 artifacts.
  _2026-08-13: Forbidden-prefix check rejected secret/state/DS9.1 artifact paths. Selected DS9.1 root untouched._
- [x] Create one compressed archive, a human-readable contents manifest, and a
  SHA-256 file. Use owner-only permissions.
  _2026-08-13: `$NOESIS_LEGACY_ARCHIVE_ROOT/ds8-ds90-inert-20260813/` mode 0700; tarball/CONTENTS/SHA256SUMS mode 0600. Tarball SHA-256 `19d84df637a6b612b5728bb1e946a5c5d4208354280c89b90cbf5250bd9fcfe7`._
- [x] Verify the archive by listing it and extracting representative DS8 and
  DS9.0 config/source files into a new temporary directory. Compare those
  representative bytes with their sources.
  _2026-08-13: `tar -tzf` listed expected paths; extracted `ds8_runtime.py`, DS8 design decisions, DS8 README, DS 9.0 `version`, and old realization JSON; bytes matched staging copies._
- [x] Do not remove any legacy path until this archive verification passes.
  _2026-08-13: No legacy path removed. Host DS 8.0/9.0 SDKs and DS8 source remain until after native acceptance._

Archive verification is the only archive ceremony required. Do not create
multiple generations, signatures, reviewers, or publication records.

## Phase 3 — Install the exact native DS9.1 platform

Keep the working DS9.1 container running during host package installation so
the dashboard remains available. The container supplies its own userspace CUDA
and TensorRT and should not depend on the host's older SDK packages.

- [x] Download the official x86 DS9.1 Debian package from the NVIDIA GitHub
  release and record its SHA-256. Do not download from a mirror.
  _2026-08-13: Official GitHub asset `deepstream-9.1_9.1.0-1_amd64.deb` SHA-256 `c67e61370f1db192c56d5d3c100fca944ed53400eb104dfd86f015bde48b8cac` stored under `$NOESIS_DS91_NATIVE_ROOT/downloads/`._
- [x] Review the proposed package transaction before applying it. It must not
  remove the installed 595.71.05 driver or unrelated desktop/home-automation
  packages.
  _2026-08-13: Driver 595 packages remained held. CUDA 13.2 installed alongside held 13.0. TensorRT pin rewritten to exact `10.16.0.72-1+cuda13.2`. No driver/desktop/HomeSeer/Z-Wave removals._
- [x] Remove the explicitly installed DeepStream 8.0/9.0 packages and their
  versioned SDK registrations using their package/uninstall ownership. Do not
  use broad globs.
  _2026-08-13: Removed `deepstream-9.0` and `deepstream-reference-graphs-8.0`. Did not run vendor `uninstall.sh`. Inspected leftover `/opt/nvidia/deepstream/deepstream-8.0`, removed 8.0/9.0 alternatives by exact path, then exact `rm -rf` of that 8.0 tree. No 8.0/9.0 SDK dirs remain._
- [x] Install `cuda-toolkit-13-2` from the NVIDIA Ubuntu 24.04 repository.
  Do not reinstall or downgrade the working NVIDIA driver.
  _2026-08-13: `cuda-toolkit-13-2` (13.2.2-1) installed. Driver still `595.71.05`. Held CUDA 13.0/13.1 remain until Phase 9._
- [x] Install every TensorRT package listed by the DS9.1 installation guide at
  the exact version `10.16.0.72-1+cuda13.2`, replacing host 10.13.3.
  _2026-08-13: Installed exact 10.16.0.72 including guide packages plus already-needed `python3-libnvinfer*` and `libnvinfer-bin`. Re-held after install. Pin file `/etc/apt/preferences.d/noesis-ds9-tensorrt`._
- [x] Install the official `deepstream-9.1_9.1.0-1_amd64.deb` and its documented
  prerequisite GStreamer, SSL, JSON/YAML, RTSP, codec, MQTT, compiler, and
  development packages.
  _2026-08-13: Package `deepstream-9.1 9.1.0-1` installed. Prerequisites were already present (GStreamer 1.24.2)._
- [x] Run NVIDIA's bare-metal `update_rtpmanager.sh` once after prerequisites
  are installed because the application consumes live RTSP sources.
  _2026-08-13: Ran `/opt/nvidia/deepstream/deepstream-9.1/update_rtpmanager.sh` once. Patched `libgstrtpmanager.so`, `libgstrtsp.so`, and `libgstvideoparsersbad.so` into the distro GStreamer plugin dir. Host GStreamer compile load stalled live `family-room` RTSP; the existing appliance unit is restarting that reference container. Did not launch a new Docker runtime._
- [x] Run `ldconfig`, then configure the canonical CUDA and DeepStream links to
  resolve to CUDA 13.2 and DeepStream 9.1 only.
  _2026-08-13: `update-alternatives --set cuda /usr/local/cuda-13.2`; `ldconfig`. `/usr/local/cuda` → 13.2. `/opt/nvidia/deepstream/deepstream` → `deepstream-9.1`._
- [x] Verify exact results directly:
  - `nvidia-smi` still reports driver 595.71.05 and the RTX 3060;
  - `nvcc --version` resolves CUDA 13.2;
  - `trtexec --help` reports TensorRT 10.16;
  - `deepstream-app --version-all` reports 9.1;
  - `gst-launch-1.0 --version` reports 1.24.2;
  - `gst-inspect-1.0 nvinfer nvtracker nvstreammux nvvideoconvert` resolves
    libraries beneath the 9.1 installation.
  _2026-08-13: RTX 3060 / driver 595.71.05; nvcc V13.2.86; trtexec TensorRT v101600 b72; deepstream-app 9.1.0 / CUDA 13.2 / TRT 10.16; gst-launch 1.24.2. nvinfer/nvtracker/nvstreammux/nvvideoconvert Filename paths are under `/opt/nvidia/deepstream/deepstream-9.1/lib/gst-plugins/` (identical SHA to the `/usr/lib/.../deepstream/` copies). libcudart → CUDA 13.2.86; libnvinfer.so.10 → 10.16.0. No 8.0/9.0/Docker overlay origins._

Stop on any version or plugin-origin mismatch. Do not work around it with
`LD_PRELOAD`, a legacy symlink, or a container library.

## Phase 4 — Build the dedicated native Python environment

- [x] Select an external native runtime root through
  `NOESIS_DS91_NATIVE_ROOT`; do not hardcode a `/home/...` or storage path in
  tracked code.
  _2026-08-13: Native root `$NOESIS_DS91_NATIVE_ROOT` (`/mnt/noesis_storage/noesis-ds91-native`, mode 0700). Tracked code reads the env var only._
- [x] Create a clean Python 3.12 virtual environment without
  `--system-site-packages`.
  _2026-08-13: `venv/pyvenv.cfg` has `include-system-site-packages = false`. Python 3.12.3._
- [x] Install the exact existing DS9.1 requirements lock once.
  _2026-08-13: Installed `DS9/docker/requirements.lock.txt` SHA-256 `de35fb439f5c9bfd05d7fbc23436122aee033bd7b584b2eb139358e50211be48` including its CUDA 13.0 pip wheels. `pip check` clean._
  _2026-08-14: Compared the accepted image's installed package metadata with
  the native venv and corrected transitive drift (including Starlette 1.6.0
  → 0.52.1). The image-bound container lock remains byte-identical; exact
  host-only transitive pins live in
  `DS9/native/requirements.host.constraints.txt`. All 81 lock pins and all 14
  native constraints now match, and `pip check` is clean._
- [x] Install the `pyservicemaker` wheel shipped beneath the native 9.1 SDK
  into that virtual environment, along with its required PyYAML dependency.
  _2026-08-13: Installed `/opt/nvidia/deepstream/deepstream-9.1/service-maker/python/pyservicemaker-0.0.1-cp312-cp312-linux_x86_64.whl`. PyYAML 6.0.3 already from the lock. PyGObject 3.48.2 added so `gi` works without `--system-site-packages`._
- [x] Create one owner-readable environment file containing paths and
  non-secret mode settings only. It must establish the native SDK, CUDA,
  TensorRT, Python, GStreamer plugin, and per-runtime GStreamer registry paths
  before importing Service Maker.
  _2026-08-13: `$NOESIS_DS91_NATIVE_ROOT/native.env` mode 0600._
- [x] Reference secret file paths from the service's existing owner-only
  configuration. Never place secret values into the environment file.
  _2026-08-13: Env file points at existing camera/MapAnything/token files (mode 0600). No secret values stored._
- [x] In one direct Python process, verify:
  - `pyservicemaker` imports from the new virtual environment;
  - `gi` loads `Gst`, `GstSdp`, and `GstWebRTC`;
  - Torch sees CUDA and the expected GPU;
  - the required Noesis dependencies import without pulling a DS8 module;
  - no imported shared object resolves beneath Docker overlay storage or an
    8.0/9.0 SDK path.
  _2026-08-13: venv pyservicemaker + `_pydeepstream.so`; gi 3.48.2; Torch 2.12.1+cu130 sees RTX 3060. `_pydeepstream.so` links DS 9.1 and CUDA 13.2. Unused optional DS plugins (inferserver/ucx/udp) warn on missing Triton/UCX/Rivermax and are not used._
  _2026-08-14: A fresh service process maps
  `pyservicemaker/_pydeepstream.so` from the dedicated native venv. The DS9
  runtime and preflight no longer prepend the SDK post-install copy under
  `/usr/local/lib/python3.12/dist-packages` while in a venv._

## Phase 5 — Implement the native runtime boundary

Do not copy the entire container supervisor into a second giant launcher.
Separate only the application-neutral preparation logic that is genuinely
shared, then keep the native entrypoint small.

- [x] Add or convert a canonical host supervisor under `DS9/scripts/` with two
  direct operations: `check` and `run`.
  _2026-08-13: Added `DS9/scripts/run_canonical_runtime_host.py` (`check` / `run`). Health identity is emitted from env fields, not a selector file._
- [x] Make `check` validate only real native requirements:
  - exact SDK/CUDA/TensorRT/GStreamer/driver versions;
  - Python and Service Maker origin;
  - current artifact realization and selected engine files;
  - custom native/parser/plugin loadability;
  - required configs, state directories, and owner-only secret files;
  - availability of ports 6008 and 8080.
  _2026-08-13: `check` verifies exact DS 9.1 / CUDA 13.2 / TRT 10.16 / GStreamer 1.24.2 / driver floor, venv pyservicemaker origin, realization SHA-256 `9c3815bb…`, owner-only secrets, and port occupancy by connect (does not bind). Occupied 6008/8080 is reported, not a `check` failure, so Phase 8 can run `check` while the reference container is up. `run` fails if those ports are occupied._
  _2026-08-14: `check` also verifies the 14 native dependency constraints so
  a future venv re-resolution fails before startup instead of requiring an
  application compatibility shim._
- [x] Make `run` prepare the same effective configuration and then execute
  `DS9/noesis/ds9_runtime.py` as the long-lived process.
  _2026-08-13: `run` uses `os.execve` of venv Python onto `DS9/noesis/ds9_runtime.py` with baseline argv (yolo26/m, WebRTC on, RTSP off)._
- [x] Replace container path constants such as `/workspace`,
  `/opt/noesis/ds9-artifacts`, `/var/lib/noesis`, and
  `/run/noesis-secrets` with a typed host-path description supplied through
  existing environment variables.
  _2026-08-13: `NativeHostConfig` is built only from existing env paths. No `/workspace`, `/opt/noesis/ds9-artifacts`, `/var/lib/noesis`, or `/run/noesis-secrets`._
- [x] Reuse the current mutable state root in place. Do not clone, copy,
  migrate, re-baseline, or repackage the world, identity, analytics, scene, or
  virtual-twin state merely to change execution backend.
  _2026-08-13: Native env points at `state-20260813-pcf-panel-v1/payload` in place. No clone or re-baseline._
- [x] Preserve the current runtime environment values for:
  - baseline lane and selected detector/profile;
  - three camera config;
  - artifact/model/ONNX/engine/plugin roots;
  - world and identity databases;
  - analytics and scene stores;
  - WebSocket/REST auth;
  - WebRTC SHM output;
  - CPU math pool caps and identity-retention cadence;
  - publication-gate shutdown ordering.
  _2026-08-13: Supervisor preserves yolo26/baseline, cameras.yaml, artifact root, world/identity/analytics/scene/virtual-twin paths, required auth, WebRTC SHM `/tmp/noesis-mosaic-h264`, `NOESIS_CPU_MATH_THREADS=1`, `NOESIS_SHUTDOWN_GRACE_SECONDS=75`._
- [x] Remove runtime checks for image IDs, Docker roots, container labels,
  mounts, namespaces, container PIDs, overlay filesystems, and Docker GPU
  ownership.
  _2026-08-13: Native supervisor strips Docker/selector-file/lease env and does not inspect images, Docker roots, or container namespaces._
- [x] Do not weaken artifact compatibility. A historical record may state that
  an unchanged artifact was originally built in the DS9.1 build image, but
  native runtime admission must depend on its actual bytes and exact
  DS/CUDA/TensorRT compatibility—not on the continued existence of that image.
  _2026-08-13: Admission uses `asset_realization.json` SHA-256 `9c3815bbf86eb41a94efb504fad79a9208c543e05f98c68d017cb1a8dcfd2b26` plus `validate_asset_realization`. No image-ID requirement._
- [x] Add focused tests for host path materialization, exact version rejection,
  secret-path handling, artifact selection, command construction, and clean
  signal propagation. Update existing container-boundary tests instead of
  retaining a fake Docker requirement.
  _2026-08-13: `DS9/tests/test_runtime_host_boundary.py` plus wait-ready tests: 16 passed. Engine-wrapper tests keep `NOESIS_DS9_ENGINE_BACKEND=docker` only until Phase 10 deletes Docker._

## Phase 6 — Convert maintenance/build tooling to native-only

The end state must not require Docker for future source changes or engine
maintenance.

- [x] Change the active native-extension, parser, TensorRT plugin, GStreamer
  plugin, and engine maintenance wrappers to invoke the host 9.1 compiler,
  headers, libraries, and `trtexec` directly.
  _2026-08-13: Active wrapper is `DS9/scripts/run_canonical_engine_maintenance_host.sh`. Canonical `run_canonical_engine_maintenance.sh` execs the host script unless `NOESIS_DS9_ENGINE_BACKEND=docker`. `--v3dt` hard-fails. Refuses `NOESIS_DS9_DOCKER_ROOT`._
- [x] Keep the existing source hashes, artifact hashes, engine profiles,
  precision, tensor names, shapes, memory guards, and functional quality
  checks. Change execution backend only.
  _2026-08-13: No engine/profile/precision/tensor changes. Backend switch only._
- [x] Extend manifest/provenance records with a simple `native_host` build
  authority containing exact DS, CUDA, TensorRT, compiler, Python ABI, source
  hash, output hash, and command. Do not invent host sealing, machine identity,
  attestation services, or promotion records.
  _2026-08-13: `native_host_build_authority()` in `engine_maintenance_common.py`; `rebuild_engines.py` attaches `metadata["native_host"]` when `NOESIS_DS9_MAINT_BACKEND=native_host`. MapAnything quality `platform` key set left unchanged._
- [x] Allow existing unchanged DS9.1 artifacts to retain honest historical
  build provenance while removing any requirement that Docker or its images
  still exist.
  _2026-08-13: Existing realization provenance kept. Native admission does not require the build image._
- [x] Direct-load every selected existing runtime component first:
  - native Python extensions with correct import origins;
  - nvinfer parsers and required exported symbols;
  - the TensorRT plugin;
  - the three Noesis GStreamer plugin factories;
  - the five baseline TensorRT engines.
  _2026-08-13: Direct-load succeeded: 7 native extensions, 3 parsers, TRT plugin, 3 GST factories, five baseline engine byte hashes match realization. GPU deserialize deferred to Phase 8 so the reference container keeps the GPU. Import order calls `configure_ds9_runtime_import_paths` first so a leftover repo-root `noesis_pose_meta_ext*.so` is not selected._
- [x] Rebuild a component only if native loading proves it incompatible, or if
  changing its build wrapper requires one representative native build to prove
  the new path. Do not rebuild all unchanged engines merely to relabel them.
  _2026-08-13: No rebuild. Direct-load file admission passed._
- [x] If a rebuild is required, write to a temporary sibling file, load-test
  that file directly, and replace the selected artifact only after it passes.
  This is file safety, not a candidate/promotion system.
  _2026-08-13: Not used; no rebuild required._
- [x] Keep MV3DT and AMC disabled. Do not build their dormant asset matrix for
  this migration.
  _2026-08-13: Host wrapper `--v3dt` hard-fails. AMC not invoked._

## Phase 7 — Detach Noesis service startup from activation machinery

- [x] Trace the current direct consumers of `noesis-appliance.service`, its
  readiness unit, Menon gateway readiness, and the dashboard. Change only the
  contracts needed for direct native startup.
  _2026-08-13: Consumers remain `noesis-appliance.service` → `noesis-appliance-ready.service` → `menon-gateway.service` / guard. Did not patch the dirty Menon checkout or the live Menon release tree._
- [x] Keep the current service name if that avoids unrelated Menon changes, but
  change its `ExecStartPre` and `ExecStart` to call the native supervisor
  directly with the native virtual environment and environment file.
  _2026-08-13: Kept `noesis-appliance.service`. `ExecStartPre`/`ExecStart` now call venv Python + `run_canonical_runtime_host.py` with `native.env`. WorkingDirectory is the Noesis_Devel checkout._
- [x] Remove `noesis-runtime-adapter.mjs`, deployment-selector lookup, image
  fields, Docker-root fields, and active-pointer/state-lease requirements from
  the Noesis process path.
  _2026-08-13: Unit no longer calls `run-noesis` / `noesis-runtime-adapter.mjs`. Supervisor strips selector-file, lease, and Docker env. Health identity is constructed from `NOESIS_DEPLOYMENT_ID` / `NOESIS_HEALTH_SELECTOR_SHA256` / `NOESIS_STATE_RELEASE_ID` / `NOESIS_SOFTWARE_REVISION` without reading the selector file._
- [x] Preserve the same state files by passing their actual host paths. Do not
  create a new state-release abstraction.
  _2026-08-13: Same `state-20260813-pcf-panel-v1/payload` paths via `native.env`._
- [x] Preserve unprivileged execution, `UMask=0077`, journald output,
  `Restart=on-failure`, `NoNewPrivileges`, filesystem protection, file-descriptor
  limits, bounded stop time, and SIGTERM behavior.
  _2026-08-13: Preserved UMask 0077, journald, Restart=on-failure, RestartPreventExitStatus=78, NoNewPrivileges, ProtectSystem=full, PrivateTmp, SIGTERM, TimeoutStopSec=90s, LimitNOFILE, TasksMax._
- [x] Give systemd write access only to the existing build/state/evidence/SHM
  paths that the native process actually uses. Do not use root at runtime.
  _2026-08-13: Still a user unit (not root). Session HOME/XDG/CUDA cache and SHM stay under the existing runtime root and PrivateTmp `/tmp`. Did not add extra writable roots._
- [x] Keep Menon and the dashboard on their existing loopback contracts. If a
  readiness probe currently requires selector identity, replace only that
  assertion with native runtime version/backend/readiness identity; do not
  redesign health APIs.
  _2026-08-13: Added `DS9/scripts/native_noesis_wait_ready.py`. Host `wait-ready noesis` execs that helper. DeploymentHealth / ws.health v2 contracts unchanged. Opaque `selector_sha256` is still emitted so Menon `wait-ready appliance` can keep matching without a Noesis selector-file read._
- [x] Validate changed unit files with `systemd-analyze --user verify` and run
  the focused unit/readiness tests. Do not render or stage an appliance bundle.
  _2026-08-13: `systemd-analyze --user verify` passed for both Noesis units. Focused tests 16 passed. No appliance bundle._

## Phase 8 — Direct native application acceptance

This is the only end-to-end validation required.

- [x] Run the native supervisor's `check` while the current container remains
  active. The check must avoid binding application ports or opening camera
  streams.
  _2026-08-13: Native `check` passed in 38s while the reference container held 6008/8080. Probe was connect-only. Platform CUDA 13.2 / DS 9.1.0 / TRT 10.16.0.72 / GStreamer 1.24.2 / driver 595.71.05. Realization SHA-256 `9c3815bb…`._
- [x] Stop the current Noesis user service once and confirm its DS9 container
  exits cleanly. Do not stop either Docker daemon yet.
  _2026-08-13: Stopped `menon-appliance-guard`, `menon-appliance.target`, and `noesis-appliance.service`. `ds9_runtime.py` / docker-init gone; 6008/8080 released. `docker.service` and `noesis-ds9-secondary-docker.service` left active. Did not stop ordinary Docker (Z-Wave)._
- [x] Start the native launcher directly in the foreground, not through a
  selector or candidate.
  _2026-08-13: First foreground `run` failed on FastAPI/Starlette 1.6 missing `add_event_handler`. A temporary shutdown-hook compatibility edit allowed diagnosis._
  _2026-08-14: Restored the accepted Starlette 0.52.1 dependency and reverted
  that application workaround. `alignment_walk_api.py` again matches the
  accepted container code, and `DS9/noesis/ds9_runtime.py` starts from the
  native venv with no Docker ancestor._
- [x] Require startup with all three expected camera IDs and no pipeline,
  native, TensorRT, artifact, config, auth, or state error.
  _2026-08-13: `living-room`, `kitchen`, `family-room` all running. Engines deserialized from the selected DS9.1 artifact root. nvv4l2decoder and nvurisrcbin mapped from `/opt/nvidia/deepstream/deepstream-9.1`. REST 8080 / WS 6008 up._
- [x] Run one bounded 8-10 second live validation:
  - all three source frame counters increase;
  - each source sustains at least 28.5 FPS unless the saved reference itself
    was below 30 FPS for an external camera reason;
  - detections and tracker/world sequences advance;
  - identity and depth/world consumers remain healthy;
  - authenticated REST and WebSocket readiness pass;
  - WebRTC negotiates and decodes frames;
  - the dashboard loads and renders the expected live state;
  - zero-copy violation count remains zero;
  - one CPU/RAM/GPU/VRAM snapshot shows no gross regression;
  - no reappearance of the fixed per-frame retention/CPU-pool regression.
  _2026-08-14: Fresh native startup and a 14.68s live sample passed. Observed
  FPS was 29.98 living room, 29.91 kitchen, and 29.98 family room; all sources
  were healthy, tracking/depth/world stages advanced, and no pipeline error was
  present. Authenticated REST/WebSocket readiness passed. The 8s WebRTC probe
  received 9,389 RTP packets and decoded 246 frames; the dashboard HTTP root
  returned success._
  _2026-08-14 requested recorded pressure result: the synchronized undated
  Living Room, Kitchen, and Family Room MP4s ran for 90.67s with the production
  graph and isolated mutable state. Family Room was temporarily normalized to
  its calibrated live 1280x720 input shape. Each camera sustained 25.45 FPS
  (76.36 aggregate) under an average 1.81 and maximum 3 simultaneous tracks.
  Process CPU averaged 128.2% (p95 138.4%), RSS p95 was 2,020.7 MiB, GPU
  utilization averaged 53.1% (p95 76%), and VRAM p95 was 6,380 MiB. There were
  no pipeline errors or zero-copy violations. The heaviest measured CPU stages
  were tracking publication (18.45ms/call), object-depth sampling
  (9.44ms/call), and analytics handling (9.09ms/call)._
- [x] Send SIGTERM once and require clean publication-gate drain, pipeline EOS,
  server shutdown, exit code zero, and released ports.
  _2026-08-13: SIGTERM → publication callbacks quiesced (admitted=completed=11415), orderly EOS, pipeline stopped, shutdown complete, exit 0, ports released. About 0.7s after the signal because the gate was already caught up._
- [x] Fix only concrete native-boundary failures and rerun only the failed
  component plus this short smoke. Do not broaden validation automatically.
  _2026-08-14: Corrected native dependency resolution and Service Maker import
  authority, reverted the temporary FastAPI shutdown-hook workaround, and
  reran only host-boundary/API checks plus direct live and requested recorded
  paths. No graph, model, precision, batching, tracking, or depth behavior was
  changed._
- [x] Start the native systemd service and repeat only readiness, three-source
  progression, and a short WebRTC decode.
  _2026-08-14: Fresh process tree is user systemd → native venv Python →
  `ds9_runtime.py`. Noesis, Noesis readiness, Menon gateway, and Menon readiness
  units are active; live sources are back at approximately 30 FPS and WebRTC
  decoded successfully._
- [x] Leave the accepted native application running and ask the user to inspect
  the dashboard.
  _2026-08-14: The corrected native application is running. Irreversible
  Phase 9–10 cleanup remains paused until the user explicitly accepts this
  parity result._

### Mandatory user acceptance pause

Do not uninstall Docker, delete images, remove legacy artifact roots, or delete
legacy active-tree paths until the user explicitly says the native dashboard
and application are accepted.

If native acceptance fails, leave the native failure visible and report it.
Do not silently restart the container unless the user requests restoration.

## Phase 9 — Remove legacy executability after acceptance

- [ ] Reconfirm the verified legacy archive exists, is readable, and excludes
  secrets/current state/current DS9.1 artifacts.
- [ ] Remove DS8 and DS9.0 runtime entrypoints, active configs, build wrappers,
  tests whose sole purpose is executing those runtimes, and compatibility
  symlinks from the active checkout using explicit reviewed paths.
- [ ] Keep shared `noesis/` and `noesis_core/` modules that DS9.1 actually
  imports. Use static imports plus one native runtime import trace to decide;
  do not classify a file as legacy merely because it lives under `noesis/`.
- [ ] Remove versioned DeepStream 8.0/9.0 SDK directories left by their
  uninstallers, but only after verifying the resolved exact paths and that no
  9.1 library points into them.
- [ ] Remove old CUDA toolkit packages (12.x, 13.0, and 13.1) and stale
  TensorRT 10.13 packages explicitly after reviewing the apt transaction.
  Keep the driver and CUDA 13.2/TensorRT 10.16 packages.
- [ ] Move the old external DS9.0 artifact realization into the legacy archive
  or delete it if it is reproducible and already represented there. Never move
  the selected DS9.1 artifact root.
- [ ] Clear stale user GStreamer registry caches and regenerate one registry
  using only the DS9.1 environment.
- [ ] Remove active Dockerfiles, image-build scripts, secondary-Docker
  supervisors, and container-only tests after their historical versions are in
  the archive. Historical documentation under an archive may still mention
  Docker; executable source and current instructions may not.
- [ ] Prove legacy non-runnability:
  - no DeepStream 8.0 or 9.0 package/SDK directory remains active;
  - no active DeepStream link resolves to 8.0 or 9.0;
  - no DS8/9.0 runtime entrypoint or service exists outside the archive;
  - no active engine or native plugin resolves from a legacy artifact root;
  - importing or executing the former legacy entrypoints fails because they
    are absent, not because they redirect to DS9.1;
  - DS9.1 still passes its native `check` afterward.

Do not create DS8/9.0 compatibility stubs or symlinks. Absence is the intended
result.

## Phase 10 — Remove Noesis Docker dependency

- [ ] Confirm the running native Noesis process tree has no Docker/container
  ancestor and no open Docker socket.
- [ ] Stop and disable `noesis-ds9-secondary-docker.service` and its socket if
  separately defined.
- [ ] Confirm the DS9 container and its containerd shim are gone, then remove
  the isolated DS9 Docker data root only after validating its exact resolved
  path. Do not target the storage root or system Docker data root.
- [ ] Remove DS9-specific Docker images, networks, and configuration only from
  the isolated DS9 daemon/data root. Do not touch unrelated system-Docker
  resources.
- [ ] Search active Noesis code, units, environment files, and current docs for
  `docker`, `DOCKER_HOST`, Docker-root variables, image IDs, and the former
  container supervisor. Remaining hits must be either inert historical archive
  text or an explicitly justified third-party/non-Noesis reference.
- [ ] Restart the native Noesis service once with the DS9 Docker daemon disabled
  and require direct readiness, advancing sources, and WebRTC decode.

### System Docker boundary

At plan creation, ordinary `docker.service` runs an unrelated Z-Wave JS UI
container. The agent must stop here and report:

- Noesis is fully native and uses no Docker;
- the isolated Noesis Docker daemon is disabled;
- ordinary Docker remains only for the unrelated Z-Wave workload.

Shut down `docker.service`, `docker.socket`, and `containerd.service` only after
the user explicitly chooses either to accept the Z-Wave outage or separately
migrate that application. Do not fold a Z-Wave migration into this work order.

## Phase 11 — Documentation and commits

- [ ] Update `DS9/README.md`, runtime ownership, runtime boundary docs, the
  root/DS9/plans agent guidance, and relevant design decisions so they state:
  - DS9.1 native host execution is canonical;
  - Docker and selectors are not part of Noesis development, runtime, builds,
    or validation;
  - DS8/DS9.0 are archive-only and must not be restored as fallbacks;
  - direct component checks plus one live application smoke are the normal
    validation path;
  - AMC and MV3DT remain deferred.
- [ ] Move obsolete current Docker/runtime guidance into historical docs or
  remove it after it is represented in the legacy archive. Do not leave stale
  current instructions claiming the container supervisor is canonical.
- [ ] Run `./scripts/check_agents_docs_consistency.py` once for the final docs
  changes. Do not rerun application tests because of documentation edits.
- [ ] Use selective staging. Never use `git add -A` in either dirty checkout.
- [ ] Keep commits understandable and limited to at most these boundaries:
  1. native runtime, path, manifest, and focused tests;
  2. native systemd/Menon readiness integration;
  3. accepted legacy/Docker removal and documentation.
- [ ] Record dated completion notes beneath this plan's completed phase items.
  Do not paste raw logs or large evidence dumps into the plan.

## Focused validation matrix

| Changed capability | Required evidence | Explicitly unnecessary |
| --- | --- | --- |
| Host SDK/toolchain | Exact versions and plugin origins | Full pipeline suite |
| Python/Service Maker | One import-origin and CUDA smoke | Reinstalling dependencies repeatedly |
| Native path materialization | Focused unit tests and `check` | Appliance release validation |
| Native binaries/plugins | Import, `ldd`/symbol, `gst-inspect` | Rebuilding every dormant variant |
| Existing engines | One native deserialize per selected engine | Rebuilding unchanged engines for provenance |
| Systemd | Unit verify, start, readiness, SIGTERM | Candidate/bundle/selector mechanics |
| Application | One bounded live three-camera smoke | Long soak or recorded replay unless live is ambiguous |
| Performance | FPS/latency/resource comparison to saved baseline | Nsight profiling unless a real regression appears |
| Documentation | Agent/docs consistency checker | Application tests |

## Hard stop conditions

Stop and report rather than improvising if any of these occur:

- the active DS9.1 artifact or mutable state root cannot be identified;
- an archive input could contain secrets or current mutable state;
- the legacy archive does not list/extract correctly;
- apt proposes removing the NVIDIA driver, desktop stack, Menon, HomeSeer, or
  unrelated home-automation packages;
- any loaded library comes from DeepStream 8.0/9.0, TensorRT 10.13, a Docker
  overlay, or an unexpected CUDA toolkit;
- a selected engine does not deserialize on native TensorRT 10.16;
- the native runtime requires a pipeline/model/config change to start;
- any camera remains stalled or throughput is materially below the accepted
  approximately 30 FPS baseline;
- the dashboard, WebRTC, tracking/world, identity, or depth contracts regress;
- SIGTERM does not shut down cleanly;
- stopping ordinary Docker would terminate the unrelated Z-Wave container
  without explicit user approval;
- an unrelated dirty file would have to be overwritten or committed.

## Final report format

Report only:

1. what now runs natively;
2. the exact DS/CUDA/TensorRT/Python environment accepted;
3. direct component and live-application results;
4. the legacy archive location and verification result without secrets;
5. what legacy/Docker executability was removed;
6. whether ordinary Docker remains solely because of Z-Wave;
7. any real limitation or deferred item.

Do not narrate candidate attempts, publication mechanics, broad test counts, or
unrelated workspace state.
