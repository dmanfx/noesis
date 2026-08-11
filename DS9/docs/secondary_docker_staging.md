# Isolated DS9 Docker Staging

Last updated: 2026-07-10

## Purpose

Use a second Docker daemon to stage the official DS9 image on a large local
filesystem without moving, restarting, or reconfiguring the primary daemon.
This is primarily a staging and inspection boundary. Its only live-runtime use
is the explicitly authorized, exclusive-GPU canary supervised by
`run_canonical_runtime_container.py`; it is not permission to run DS8 and DS9
as parallel GPU owners.

The helper refuses to run without an explicit absolute
`NOESIS_DS9_DOCKER_ROOT`. Nothing in the repository selects a machine-local
mount point. The operator must choose a suitable filesystem for the current
host.

## Isolation Contract

`DS9/scripts/secondary_docker.sh` gives the secondary daemon its own:

- Unix socket;
- Docker data root;
- execution root and PID file;
- Docker daemon ID; and
- containerd runtime and plugin namespaces.

It also starts with `bridge=none`, disables IPv4 and IPv6 Docker rule
management, and disables IP forwarding, masquerade, and the userland proxy.
Only the `host` and `none` built-in networks should exist. The daemon still
uses the host containerd service, but its `noesis-ds9` namespaces are separate
from the primary daemon's `moby` namespace.

## Start And Pull

Choose the staging root explicitly, then keep every Docker command scoped to
the secondary socket:

```bash
export NOESIS_DS9_DOCKER_ROOT=<absolute-large-filesystem-staging-root>
DS9/scripts/secondary_docker.sh start
DS9/scripts/secondary_docker.sh status
DS9/scripts/secondary_docker.sh pull
```

Do not set a global Docker context to this socket. For an ad hoc command, use
the environment emitted by the helper in the current shell only:

```bash
eval "$(DS9/scripts/secondary_docker.sh env)"
docker info
```

## Read-Only, No-GPU Inspection

The inspection container must not receive `--gpus`, host networking, or a
writable root filesystem while the live DS8 runtime owns the GPU. A safe
capability probe uses the secondary `DOCKER_HOST` and starts with:

```bash
docker run --rm \
  --runtime=runc \
  --network=none \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint /bin/bash \
  nvcr.io/nvidia/deepstream:9.0-triton-multiarch \
  -lc 'deepstream-app --version-all; python3 --version'
```

GPU-dependent plugins will report missing driver libraries in this probe.
That is positive evidence that no host GPU was mounted, not evidence that the
image lacks the plugin files.

Do not use the NVIDIA runtime for a CPU-only check. On this host, an experiment
with `--runtime=nvidia` and `NVIDIA_VISIBLE_DEVICES=none` still injected NVIDIA
device nodes. The accepted CPU validation boundary is `--runtime=runc`, no
`--gpus` option, a read-only root filesystem, and an assertion that `/dev`
contains no `nvidia*` entries. When Service Maker loader validation needs host
driver symbols, bind only the host `libcuda.so.1` and `libnvidia-ml.so.1` files
read-only; those library mounts do not add device nodes or make CUDA usable.

### Runtime-secret mount validation

The staged derived image can validate the exact future runtime mounts without
receiving a GPU or starting DS9:

```bash
export NOESIS_DS9_DOCKER_ROOT=<absolute-large-filesystem-staging-root>
DS9/scripts/secondary_docker.sh start
DS9/scripts/validate_runtime_secrets_container.sh
DS9/scripts/secondary_docker.sh stop
```

The check uses runc, `network=none`, a read-only root filesystem, no
capabilities, no NVIDIA device nodes, a read-only repository, and three
individual read-only secret-file binds beneath a uid-owned mode-`0700` tmpfs.
It loads and validates the camera registry, MapAnything key, and internal
bearer without printing their values. Engine maintenance plan/build mode
intentionally has no such mounts and no dependency on runtime-secret
environment variables.

The live boundary is stricter than this no-GPU mount probe. Run the write-free
canonical plan first:

```bash
export NOESIS_DS9_RUNTIME_ROOT=<absolute-private-runtime-root>
python3 DS9/scripts/run_canonical_runtime_container.py plan
```

The plan pins the derived image, gates canonical artifacts and profile/ports,
rejects existing GPU/runtime owners and artifact-transaction lock contention,
and proves the requested mount and 26 GiB memory-ceiling contract without
invoking `docker run`. See `runtime_container_boundary.md`; do not reconstruct
its live command by hand.

## Observed Image Provenance

The official image pulled on 2026-07-10 was:

- reference: `nvcr.io/nvidia/deepstream:9.0-triton-multiarch`;
- registry digest:
  `sha256:2e45070ad134b9ab2caa4a97ba4d52fa8744a4f0db30900bd92828d51425a69a`;
- image ID:
  `sha256:8a671b8988f2d0477543c921082634ddff8c76391b6971d07816d45d5402895b`;
- created: `2026-03-08T23:02:33.997777864Z`;
- platform: `linux/amd64`; and
- unpacked image size: `16,428,012,925` bytes.

The read-only probe reported DeepStream 9.0.0, CUDA runtime 13.1, TensorRT
10.14, cuDNN 9.17, Python 3.12.3, CUDA compiler 13.1.115, CMake 3.28.3, and
GCC 13.3. No NVIDIA device nodes or driver libraries were visible.

## Engine-Build Image Provenance

`DS9/docker/Dockerfile` builds the digest-pinned Noesis engine-build image
through `DS9/scripts/build_secondary_dev_image.sh`. The observed 2026-07-10
authority is:

- tag: `noesis-ds9-dev:9.0-20260710`;
- image ID:
  `sha256:7476b1021376cd67793c95d949cdc7d46eef7704ab98a5a76feed461e4f907a4`;
- created: `2026-07-10T14:37:50.483188799Z`;
- logical image size: `21,920,527,866` bytes;
- base digest: the official digest above;
- Dockerfile SHA-256:
  `c1df566ec73759a5bef0275dc5e9edcdc41be3f6b3f3a102a19d89a5f9860c9e`; and
- Python lock SHA-256:
  `de35fb439f5c9bfd05d7fbc23436122aee033bd7b584b2eb139358e50211be48`.

The build image has exact-pinned build/runtime Python dependencies, NICE,
libav, required codec shared objects, pybind11 2.12.0, and Torch
2.12.1+cu130. A read-only runc validation passed `pip check`, runtime imports,
NICE and H.264 GStreamer inspection, and reported zero Torch CUDA devices.

## Runtime Image Provenance

Live runtime dependencies are layered separately so an unrelated service
package change cannot falsify TensorRT engine evidence. Build and inspect the
runtime layer with:

```bash
export NOESIS_DS9_DOCKER_ROOT=<absolute-large-filesystem-staging-root>
DS9/scripts/build_secondary_runtime_image.sh
```

`DS9/docker/Dockerfile.runtime` uses the exact engine-build image ID in its
`FROM` instruction and adds only the ABI-aligned Ubuntu packages
`python3-gi=3.48.2-1` and
`gir1.2-gst-plugins-bad-1.0=1.24.2-1ubuntu4`. The observed authority is:

- tag: `noesis-ds9-runtime:9.0-20260710`;
- image ID:
  `sha256:ca33b4c6a84fc56b86b71feee2a444299cb2ce7f33ab018cae43ac730aaef5fc`;
- parent image ID:
  `sha256:7476b1021376cd67793c95d949cdc7d46eef7704ab98a5a76feed461e4f907a4`;
- Dockerfile SHA-256:
  `f3345a4c87483a70dc48a7185330880805d38dbc97d18d57c4b08ab1e415d51e`;
- logical image size: `21,927,446,056` bytes; and
- RootFS relationship: the build image's exact 70 layers followed by exactly
  two runtime-only layers.

The Dockerfile and DS9 preflight both require `GLib`, `Gst`, `GstSdp`, and
`GstWebRTC` through PyGObject plus `rtspsrc`, `rtph264depay`, `h264parse`,
`rtph264pay`, `webrtcbin`, `queue`, and `fakesink`. The canonical supervisor
launches by immutable runtime image ID and revalidates the parent-layer prefix;
the engine-maintenance runner continues to use the immutable build image.

## Dependency And Artifact Gates

The base image contains the DS9 SDK headers, Service Maker binding, CUDA
compiler, TensorRT development libraries, GStreamer development headers, and
the C/C++ build toolchain. The derived image supplies the previously missing:

- `pybind11==2.12.0` for ABI-compatible native extensions;
- `DS9/requirements-runtime.txt`, including PyYAML, uvicorn, websockets,
  NumPy, Torch, and application libraries;
- `gstreamer1.0-nice` and `gstreamer1.0-libav` for WebRTC and H.264 decode;
- verification or reinstall of codec libraries when package metadata exists
  but the shared object was removed from the image layer; and
- DS9-owned ONNX sources and outputs declared by `DS9/asset_manifest.yaml`.

The 2026-07-10 CPU-only builds produced and provenance-hashed 16 DS9-owned
non-engine artifacts: six native extensions, three GStreamer plugins, the YOLO
segmentation TensorRT plugin, and six nvinfer parsers. At that checkpoint the
declarative manifest recorded 16 `staged_unverified` artifacts and 24 missing
TensorRT engines. The current external realization records ten selected engines
and passes canonical, V3DT, and Wholebody49 file/provenance validation. Read-only
DS9 imports resolved all six native modules from
`DS9/native_extensions`, not the root DS8 copies.

The fixed-batch-3 pose input was exported on CPU with Ultralytics 8.4.0:

- `yolo26n-pose_b3.onnx`: `12,075,358` bytes;
- SHA-256:
  `be40dd6e2e3bedf3fd5eed289881a607486a4ce2f0f1d72f57bd806a8908ce61`;
- input/output contract: `images[3,3,640,640] -> output0[3,300,57]`; and
- ONNX Runtime CPU repeated-batch maximum delta: `0.0`.

`DS9/scripts/stage_canonical_sources.py` stages the canonical baseline,
explicit alternate, promoted Wholebody49, and V3DT engine inputs into an
explicit artifact root and handles MapAnything external tensor data. The
earlier 2026-07-10 source gate contained 604 files, totaled
`4,359,636,294` bytes, and had aggregate SHA-256
`e1c153c38bb419cf2cf3068e6fe9c42572fad654dbf219329f0d67a2024261e4`.
That digest predates promotion of YOLO26-m to the canonical baseline and
YOLO26-seg-s to the V3DT lane; it is historical capacity evidence, not current
runtime readiness. Re-run staging/verification and record the new aggregate
before live use.
The bundle includes 587 MapAnything sidecars totaling `3,680,837,088` bytes.
These are source inputs only; DS8 TensorRT engines and native binaries remain
forbidden.

The repository checkout's filesystem had insufficient headroom to duplicate
all ONNX sources plus DS9 engines. Keep large machine-local build and daemon
storage on the operator-selected large filesystem, then expose paths through
`NOESIS_DS9_DOCKER_ROOT`, `NOESIS_DS9_ARTIFACT_ROOT`, and deliberate mounts.
Do not add the selected host path to portable configuration.

## Canonical And Promoted Engine Maintenance Commands

Stage or re-verify sources while DS8 is still running. This is CPU and storage
I/O only:

```bash
export NOESIS_DS9_DOCKER_ROOT=<absolute-large-filesystem-docker-root>
export NOESIS_DS9_ARTIFACT_ROOT=<absolute-large-filesystem-artifact-root>

DS9/scripts/secondary_docker.sh start
python3 DS9/scripts/stage_canonical_sources.py
DS9/scripts/run_canonical_engine_maintenance.sh --all --plan
```

The plan mode verifies only the selected pinned staged-source contracts, then
executes the exact rebuild dry runs in runc containers without reading mutable
root model sources or exposing GPU devices. `--all` and `--v3dt` use their
reviewed profile-scoped source gates; `--only` gates exactly that engine.
It also audits unfinished host publication transactions without changing them.
Plan mode is metadata-read-only as well as content-read-only: the engine,
maintenance-evidence, and log directories plus the owner-only transaction lock
must already exist with their reviewed ownership and modes. A missing or unsafe
path is a blocker; `--plan` never creates it, changes its mode, or repairs it.
An actual invocation removes only same-artifact-root orphan build containers,
proves they are absent, and applies prepared-transaction recovery before any
new GPU work. A role-matched container with a missing or different artifact-
root identity is a blocker and is never removed.
During an announced maintenance window, stop DS8 and
confirm `nvidia-smi --id=0 --query-compute-apps=pid,process_name --format=csv,noheader`
is empty. Then run these five commands separately, in order:

```bash
DS9/scripts/run_canonical_engine_maintenance.sh --only yolo26_m
DS9/scripts/run_canonical_engine_maintenance.sh --only reid_swin
DS9/scripts/run_canonical_engine_maintenance.sh --only yolo26_pose_n
DS9/scripts/run_canonical_engine_maintenance.sh --only depth_anything_v2_tracking
DS9/scripts/run_canonical_engine_maintenance.sh --only mapanything
```

Wholebody49 is a promoted alternate profile, not part of canonical `--all`.
Build its engines separately only when that profile is in the maintenance scope:

```bash
DS9/scripts/run_canonical_engine_maintenance.sh --only wholebody49_s_masks
DS9/scripts/run_canonical_engine_maintenance.sh --only wholebody49_x_boxes
```

V3DT is also outside canonical `--all`. Build its YOLO26-seg-s PGIE first, then
its internal engines in order; BodyPose3DNet must exist before the NvMOT helper
initializes the tracker graph:

```bash
DS9/scripts/run_canonical_engine_maintenance.sh --only yolo26_seg_s
DS9/scripts/run_canonical_engine_maintenance.sh --only bodypose3dnet
DS9/scripts/run_canonical_engine_maintenance.sh --only v3dt_tracker_reid
```

YOLO11-seg is an explicit alternate-only engine. It may be built with
`--only yolo11_seg` for a named comparison, but it is never a substitute for
canonical baseline or V3DT readiness.

Each command uses only the secondary daemon. The checkout and external artifact
model root are read-only; only the nested engine-output and private maintenance-
evidence directories are writable. There is no mutable root source-model
mount. The container runs as the invoking non-root uid with a read-only root
filesystem, no network, no capabilities, no-new-privileges, init, GPU device 0,
bounded pids/logs, and an exact 24 GiB memory-and-swap ceiling. Equal memory
and memory-plus-swap limits disable container swap on this swapless host; the
unsupported swappiness option is intentionally omitted and may inspect as
`null` (or `0` on a supporting daemon).
The 24 GiB maintenance ceiling deliberately leaves about 7 GiB outside the
cgroup on the current 31 GiB swapless host because TensorRT compilation is
bursty; the runtime container's separate 26 GiB steady-state ceiling is not a
maintenance-build target. A future per-engine increase requires observed cgroup
OOM evidence.

The runner builds one engine into a temporary output and deserializes both the
candidate and installed path with separate `trtexec --loadEngine ...
--skipInference` invocations. It persists owner-private container inspect and
bounded log evidence in the artifact root. Before launch, the host finalizer
snapshots the exact prior engine and realization, and the builder receives a
read-only bind of that prepared transaction plus its digest. The builder refuses
real work unless its independently preserved prior engine agrees with the host
snapshot. Its evidence also binds the host transaction ID, digest, and mounted
manifest path. A unique hidden `.building-<run-id>` output is never blindly
deleted by the builder: successful installation moves it into place, while an
interrupted output remains under host-transaction cleanup authority. Positive
build, candidate-load, and final-load evidence is necessary
but not sufficient to publish: the host then reconciles and independently
validates the selected realized artifact before committing the engine and
realization as one unit.

Every uncommitted EXIT, SIGINT, or SIGTERM path first force-removes the
maintenance container and proves absence through the secondary daemon, then
rolls back the host engine and realization. The wrapper creates the container
without starting it, obtains its immutable ID, establishes the NVML sampler and
its first sample, and only then starts the container. That ID is used for
inspect, monitoring, logs, and removal; cleanup additionally requires
exact role, transaction, artifact-root, and expected-name labels. A foreign
same-name container is never removed. If containment cannot be proven, artifact
rollback is deliberately skipped and the prepared transaction is left for
recovery; this prevents a still-running builder from racing restored bytes. The
runner checks GPU device 0 before snapshot and again immediately before launch.
GPU name, UUID, compute capability, memory, driver, memory guard, and process
owners are all scoped to that same device. An identity-bound NVML-v2 sampler
reads raw used/reserved bytes every 25 ms from before container start through
exit. A raw-byte breach, NVML error, sampler death, malformed row, or
inter-sample gap above 250 ms signals the wrapper and fails closed.
Continuous process ownership is proven from the immutable inspected container
init PID and its pinned `/proc` start time. Each NVML PID must have a stable
start-time-bound ancestry chain to that init process; the runner requeries a
changing or vanished owner set and rejects foreign, unreadable, recycled, or
persistently ambiguous PIDs. It persists container state and logs before forced
removal on a monitor failure.

The sealed sampler JSONL is not a disposable operator log. It lives at
`models/engine_finalize/<transaction>-<engine>/gpu-memory.jsonl` with mode 0600
under the retained mode-0700 transaction cohort. Its header binds engine,
transaction ID/digest, artifact-root ID, container ID, exact GPU UUID/index,
wrapper PID/start ticks, guard, and cadence. The host reconstructs every sample,
maximum observed, gap, and raw evidence digest instead of trusting the footer.
The finalizer, reconciler, and later realized-artifact validator reopen the same
evidence; realization provenance stores its exact path, digest, identity, and
reconstructed summary. Direct reconciliation requires the same proof. Only the
eight exact artifact/output/maintenance-path/maintenance-digest tuples already
realized at the 2026-07-11T13:18:00Z cutoff may omit it; Wholebody49 may not.
This is robust same-user operational evidence, not cryptographic attestation;
cross-checking the later committed transaction remains a hardening opportunity.

Rollback inventories hidden candidates before launch and removes only a new,
regular, single-link, caller-owned candidate derived from the sole new
transaction-bound maintenance manifest. SHA-256, size, mode, inode, timestamps,
and removal are recorded before and after the unlink. Preexisting, symlinked,
nonregular, hardlinked, unknown, or changed residue is preserved and becomes a
manual-recovery blocker. Prepared cleanup is resumable after a crash, including
the boundary after unlink but before completion evidence. A historical
pre-binding residue may be handled only with the finalizer's explicit
`cleanup-residue` operation under the inherited artifact lock, with exact
transaction, manifest, candidate-hash, and candidate-size CAS arguments. It is
a one-shot repair; subsequent idempotent audits use `recover`, not another
explicit cleanup.
`build_mapanything_guarded.sh` delegates to this same runner so the historical
entrypoint cannot silently use primary Docker.

Engine maintenance and the canonical runtime share the owner-only
`${NOESIS_DS9_ARTIFACT_ROOT}/.noesis-ds9-artifact-transaction.lock`.
Maintenance holds it through startup recovery, the prelaunch snapshot, engine
build, host finalization, and authoritative realized validation. Runtime holds it through immediate
artifact validation, launch/readiness, and confirmation of container GPU
ownership. Lock contention is fail-closed, preventing a runtime from observing
a partially reconciled engine set.

## Capacity And Contention Budget

The source sizes and conservative output estimates, using same-model DS8
artifacts only as sizing evidence, are:

| Engine | Staged source bytes | Estimated engine bytes |
| --- | ---: | ---: |
| YOLO26-m detection (canonical baseline) | 81,910,900 | 44,478,228 |
| TAO Swin-Tiny ReID | 113,893,579 | 61,972,156 |
| YOLO26 pose batch 3 | 12,075,358 | 10,950,372 |
| DAv2 metric depth batch 3 | 98,954,388 | 54,662,948 |
| MapAnything plus sidecars | 3,681,715,613 | 1,850,148,452 |
| YOLO26-seg-s (V3DT) | 41,863,572 | 26,341,932 |
| YOLO11 segmentation (alternate only) | 40,617,409 | 27,558,188 |
| Wholebody49 S masks | 42,539,309 | 33,429,356 |
| Wholebody49 X boxes | 202,881,311 | 113,269,708 |
| V3DT BodyPose3DNet | 70,575,749 | 38,856,068 |
| V3DT tracker ReID ETLT | 96,377,716 | 49,689,004 |
| **Current promoted aggregate** | **Re-verify after restaging** | **Re-verify after builds** |

The secondary base plus derived images consume about 26 GiB of physical daemon
storage because their layers are shared. Reserve at least 50 GiB free on the
selected large filesystem for daemon layers, the promoted source bundle,
engines, logs, and temporary growth. With the explicit
mounts, checkout-root growth remains small and no engine is written to the root
model store. Staging and per-engine gates also preserve 10 GiB of residual free
space by default. A rebuild additionally budgets two full prior-engine copies:
one for the host transaction and one for the in-container builder, plus the
candidate temporary. The checkout filesystem currently fails that gate while
the large artifact filesystem passes.

TensorRT builds are exclusive-GPU maintenance. The non-MapAnything guard is
11,000 MiB; MapAnything is limited to 9,000 MiB, zero auxiliary streams, a 1 GiB
workspace, and optimization level zero. Rebuild-lane host watchdogs are at
least 2,400 seconds: longer than the builder's 30-second probe, 1,800-second
build, two 120-second load bounds, and setup/cleanup margin. The V3DT tracker
watchdog is at least 1,800 seconds, exceeding its 1,290 seconds of bounded
probe, helper compilation, build, and load commands. This prevents the host
from killing a builder after engine installation but before final-load or
rollback evidence. MapAnything is the
dominant risk on a 12 GiB GPU and a swapless host. Historical on-host evidence
shows a related MapAnything plan built in about 348 seconds, but reserve a
45-90 minute window for all five DS9 10.14 builds, deserialization checks, and
a short runtime smoke rather than treating that historical time as a promise.
The installed DeepStream 9 stack requires driver 590+; the runner permits
no-GPU plans on older drivers but rejects actual engine builds before GPU
access.
The current `595.71.05` host driver satisfies this version floor; DS8 product
acceptance and explicit exclusive-GPU ownership remain separate prerequisites.
Wholebody49 S uses a 10,000 MiB guard; X uses an 11,000 MiB guard. Early S
builds observed 9,436-9,441 MiB and rolled back cleanly while the builder policy
isolated TACTIC_DRAM, WORKSPACE, and optimization-level effects. Optimization
level 0 exposed the final mask node's exact tactic-admission requirement:
half-format tactics requested 4,571,136,000 bytes while the float tactic
requested 9,142,272,000 bytes. S therefore retains level 0 and 2 GiB
TACTIC_DRAM; X retains level 3, 4 GiB WORKSPACE, and 2 GiB TACTIC_DRAM.

The identity-bound 25 ms NVML-v2 sampler settled the outer-guard question.
Transaction `20260711T150649395390Z` captured 2,368 samples, a 25.457673 ms
maximum gap, and 9,481 MiB maximum observed with the then-current 8 GiB S
WORKSPACE; it rolled back against the 9,000 MiB guard. Reducing S WORKSPACE to
6 GiB still admitted the half tactic, but transaction `20260711T151452845569Z`
reached 9,476 MiB and rolled back at the same guard. TensorRT's separate
4,603 MiB allocator report and the installed header definitions confirm that
WORKSPACE is an intermediate-operation admission limit, not a reliable bound on
full-device peak. The reviewed S guard is therefore 10,000 MiB, preserving more
than 2 GiB free on the 12 GiB target GPU while remaining below X's guard.

Committed transaction `20260711T152338477896Z` realized the 25,327,772-byte S
engine at SHA-256
`1fb95225e8258af13ac96de5136b85dadb60419a8c49058e70536eef06dd6bdf`.
Its sealed trace has 2,738 samples, a 27.742316 ms maximum gap, and 9,481 MiB
maximum observed. Transaction `20260711T152552934749Z` realized the
108,579,580-byte X engine at SHA-256
`a5f4322d7e123461a1bbc64388b6f0e30c9359091e76ac7a95144648841138c3`;
its trace has 12,750 samples, a 30.858598 ms maximum gap, and 1,017 MiB maximum
observed. Candidate and installed-path deserialization passed for both, and the
final realization is
`6fab7d456c031490f640ee2c3ce5a38922a96ed86a965020ca3051820306dce4`.
Runtime mask/box quality and occupied-scene behavior remain separate gates.
V3DT BodyPose3DNet and tracker ReID each use a 9,000 MiB guard and must run in
that order. TAO Swin-Tiny ReID
uses an 11,000 MiB guard and needs at least
237,837,891 bytes for the ONNX plus old/new atomic engine copies. That guard is
conservative; it is not an observed DS9 peak.

## Safe Maintenance Sequence

1. While DS8 remains live, stage and hash sources, build/validate the derived
   image, run the CPU-only static/tests, and execute `--all --plan`.
2. Record the DS8 process command/start identity, primary Docker daemon ID,
   Z-Wave container start/PID/image/restart count, ports, iptables hashes, and
   GPU process baseline.
3. Stop only DS8 during the announced window and require an empty GPU compute
   process list. Do not restart Docker or alter the host SDK.
4. Start the isolated daemon and verify that only `host` and `none` networks
   exist, the primary daemon ID is unchanged, and the Z-Wave container has the
   same start time, PID, image, and restart count.
5. Run the five canonical engine commands above, plus the two Wholebody49
   commands only when that promoted profile is in scope, and the ordered three
   V3DT commands only when V3DT is in scope. Stop immediately on the first
   failure; do not substitute a DS8 engine or keep a partial engine.
6. Reconcile each engine source/output hash, image ID, command, and timestamp
   into `${NOESIS_DS9_ARTIFACT_ROOT}/asset_realization.json`; never mutate the
   tracked declarative manifest. Require the realization to pin exact SHA-256
   values for `DS9/asset_manifest.yaml` and
   `DS9/config/engine_source_contracts.json`, then run realization/provenance
   validation and full preflight.
7. Run a short DS9 launch/shutdown smoke first. If it is clean, run the focused
   bridge, identity, depth, and RTSP/WebRTC checks from `validation_runbook.md`.
8. If any acceptance gate fails, remove only the failed DS9 output from the
   explicit artifact root, preserve its log, stop DS9, and restart the exact
   recorded DS8 command. The primary daemon, Z-Wave container, DS8 assets, and
   host SDK were never mutated, so rollback is process-level rather than a
   package or filesystem restore.
9. Stop the secondary daemon when maintenance ends. Its image data remains in
   the explicit staging root:

```bash
DS9/scripts/secondary_docker.sh stop
```

Never relocate the primary Docker root or restart the primary daemon as part
of this workflow.
