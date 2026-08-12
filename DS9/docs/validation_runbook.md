# DS9 Validation Runbook

Last updated: 2026-08-12

Commands assume the repo is mounted at `/workspace` inside the DS9 container.
Do not run these against a DS8 install.

The active target is the digest-pinned DeepStream 9.1 image documented in
`deepstream_9_1_direct_upgrade_plan.md`: SDK 9.1, CUDA `13.2.0.046`, TensorRT
`10.16.0.72`, SDK root `/opt/nvidia/deepstream/deepstream-9.1`, and driver
floor `595.58.03`. The current host driver `595.71.05` passes that floor.

The exact official base is pulled and its SDK/toolchain inspection passed. No
application-owned 9.1 binary, engine, or runtime result has been accepted yet.
Dated 9.0 results later in this runbook are retained historical baselines and
must not be reported as 9.1 validation.

For host validation, run the same gates from the repo checkout. First verify the
host stack:

```bash
deepstream-app --version-all
python3 - <<'PY'
import tensorrt as trt
import torch
print("tensorrt", trt.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
```

Run the DS9 CPU/static suite from the repository root with both DS9 adapters
and shared product modules on the import path:

```bash
PYTHONPATH="$PWD/DS9:$PWD" pytest -q DS9/tests
```

Running `pytest DS9/tests` without that explicit path contract is not an
equivalent gate: it cannot resolve both DS9-owned adapters and shared root
modules and fails during collection.

The owner-RTSP orderly-EOS characterization requires an NVIDIA device because
its `nvurisrcbin` graph allocates CUDA-backed buffers. It reports a hardware
skip in an isolated CPU-only gate and runs normally in the exclusive-GPU DS9
validation tier; do not treat the CPU skip as runtime acceptance evidence.

DS9 mounts the shared health, identity-v2, scene, and virtual-twin routers in
addition to its DS9-owned depth, analytics, and ReID-v1 adapters. After changing
any REST surface or `DS9/noesis/server/boundary_metrics.py`, also run the shared
41-JSON-route inventory and single-render gate:

```bash
PYTHONPATH="$PWD/DS9:$PWD" python3 -m pytest -q \
  tests/test_rest_boundary_metrics.py \
  tests/test_rest_product_boundary_coverage.py \
  tests/test_reid_ds9_parity.py
```

The five non-JSON artifact/file routes must remain explicit exemptions. Any
other successful route without response-model timing is a hard failure, not an
unmeasured success.

The 2026-06-16 host cutover snapshot was:

- Driver `595.71.05` on `NVIDIA GeForce RTX 3060`.
- DeepStream `9.0.0`.
- CUDA runtime `13.1`.
- TensorRT `10.14.1.48`.
- Torch `2.12.0+cu130` with CUDA available.

The current 2026-07-10 recovery host is intentionally different: driver
`595.71.05` also satisfies the current DS9.1 minimum, but the host remains on
the DS8 CUDA 13.0 / TensorRT 10.13.3 stack. Use the pinned isolated 9.1 image
for CUDA 13.2 / TensorRT 10.16 work. A host `ds9_preflight.py --env-only` run is
a split diagnostic: it must accept the driver and fail the independent TensorRT
check.

## Start A DS9 Development Container

The following official-base shell is for image investigation and dependency
work, not a live runtime launch. It intentionally has no appliance secrets.

```bash
cd <repo>
docker run --rm -it --gpus all --network host --ipc host \
  -v "$PWD:/workspace" \
  -w /workspace \
  nvcr.io/nvidia/deepstream:9.1-triton-multiarch@sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994 \
  bash
```

Inside the container:

```bash
apt-get update
apt-get install -y \
  python3-pip \
  gstreamer1.0-nice \
  gstreamer1.0-libav \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  libvpx9 \
  libmp3lame0 \
  libx264-164 \
  libx265-199 \
  libmpg123-0t64
python3 -m pip install --break-system-packages \
  /opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker*.whl \
  pyyaml
python3 -m pip install --break-system-packages -r DS9/requirements-runtime.txt
```

If `gst-inspect-1.0 avdec_h264` still fails after the install, the container may
have package metadata without the shared objects. Reinstall the codec packages
and clear the GStreamer registry:

```bash
apt-get install -y --reinstall --no-install-recommends \
  libvpx9 libmp3lame0 libx264-164 libx265-199 libmpg123-0t64
rm -f ~/.cache/gstreamer-1.0/registry*.bin
gst-inspect-1.0 avdec_h264
```

Do not promote that development shell into a runtime command. The canonical
live canary is owned by `run_canonical_runtime_container.py`, which validates
the exact runtime image, its immutable engine-build parent and RootFS prefix,
the official base digest, secondary-daemon isolation, canonical
realized artifacts/profile/ports, exclusive GPU/runtime ownership, immutable source and
artifact mounts, three individual secret files, private writable paths, normal
shutdown, cleanup, and preserved evidence as one transaction.

The two reviewed derived images have distinct authorities. Build the
engine-maintenance image first, then the runtime-only layer:

```bash
export NOESIS_DS9_DOCKER_ROOT=<absolute-secondary-docker-root>
DS9/scripts/build_secondary_dev_image.sh
DS9/scripts/build_secondary_runtime_image.sh
```

The second Dockerfile starts from the exact first image ID and adds the Ubuntu
PyGObject/GstWebRTC bindings. It does not become the recorded builder of
existing TensorRT engines. See `runtime_container_boundary.md` for the exact
IDs, hashes, and realization-rebase contract.

Choose explicit disjoint machine-local roots, then run the write-free plan:

```bash
export NOESIS_DS9_DOCKER_ROOT=<absolute-secondary-docker-root>
export NOESIS_DS9_ARTIFACT_ROOT=<absolute-ds9-artifact-root>
export NOESIS_DS9_RUNTIME_ROOT=<absolute-private-runtime-root>

python3 DS9/scripts/run_canonical_runtime_container.py plan --lane baseline
python3 DS9/scripts/run_canonical_runtime_container.py plan --lane v3dt
python3 DS9/scripts/run_canonical_runtime_container.py plan --lane wholebody49-s
python3 DS9/scripts/run_canonical_runtime_container.py plan --lane wholebody49-x
```

The plan requires `${NOESIS_DS9_ARTIFACT_ROOT}/asset_realization.json` as an
owner-only mode-`0600` overlay. It must link to the exact current SHA-256 of
tracked `DS9/asset_manifest.yaml` and
`DS9/config/engine_source_contracts.json`, may overlay only engine
state/provenance, and must realize every canonical engine. Engine reconciliation
must update that external file, never the tracked manifest.

The runtime and engine-maintenance paths serialize on the owner-only
`${NOESIS_DS9_ARTIFACT_ROOT}/.noesis-ds9-artifact-transaction.lock`. A plan
reports contention as a blocker. A live run holds the lock from immediate
artifact revalidation through launch/readiness and confirmed container GPU
ownership; maintenance holds it through engine publication and realization
reconciliation.

An interrupted maintenance transaction must pass startup `recover` before any
new GPU build. Recovery verifies the exact prelaunch engine, realization,
maintenance-manifest inventory, and hidden-candidate inventory. It resumes a
sealed prepared cleanup from its recorded CAS plan and fails closed on changed
preexisting bytes, reappeared candidates, or ambiguous residue. Do not remove a
`.building-*` file manually. For a pre-binding historical transaction, use the
reviewed `finalize_engine_realization.py cleanup-residue` operation exactly once
under the same inherited artifact-lock descriptor, supplying the current
rolled-back transaction SHA-256, maintenance-manifest SHA-256, candidate
SHA-256, candidate size, and an operator reason. Capture empty maintenance-
container and GPU-owner inventories immediately before and after that command;
then use ordinary `recover` for the idempotence check.

The supervisor uses the standard camera-source, MapAnything-key, and internal
bearer paths unless their existing file-path environment variables select
other owner-only files. It mounts exactly those three files read-only beneath
a uid-owned mode-`0700` tmpfs. It never mounts a secrets directory or passes
secret bytes through an environment variable, argument, image layer, or
evidence file.

The readiness profiles are enumerated and immutable: baseline is YOLO26 detect
`m`; V3DT is YOLO26-seg `s` with `infer_v3dt.yaml` and
`cameras_v3dt.yaml`; Wholebody49 has separate `s` mask and `x` bbox-only lanes.
YOLO11-seg is explicit alternate-only and cannot satisfy readiness. After a
clean plan and explicit release of the exclusive GPU owner, the authorized
baseline 30-second canary is:

```bash
python3 DS9/scripts/run_canonical_runtime_container.py run \
  --lane baseline \
  --authorize-gpu-runtime \
  --duration-seconds 30
```

Use the same transaction for a bounded V3DT canary:

```bash
python3 DS9/scripts/run_canonical_runtime_container.py run \
  --lane v3dt \
  --authorize-gpu-runtime \
  --duration-seconds 30
```

The 30-second result proves exact launch, readiness, GPU ownership, bounded
resources, ordered EOS, cleanup, and sealed evidence. It does not by itself
prove an occupied scene. For V3DT bbox/world/identity evidence, keep one
same-lane supervisor run alive long enough for the authenticated behavior
bundle, then attach from a second shell:

```bash
# Shell 1 (choose and retain one safe session ID):
SESSION=v3dt-live-acceptance
python3 DS9/scripts/run_canonical_runtime_container.py run \
  --session-id "$SESSION" \
  --lane v3dt \
  --authorize-gpu-runtime \
  --resource-soak \
  --duration-seconds 900

# Shell 2, after authenticated readiness is visible:
python3 DS9/scripts/ds9_live_validation_runner.py \
  --session-id "$SESSION" \
  --lane v3dt \
  --no-spawn \
  --skip-preflight \
  --ownership-evidence-dir \
    "$NOESIS_DS9_RUNTIME_ROOT/evidence/$SESSION/launcher" \
  --output-dir \
    "$NOESIS_DS9_RUNTIME_ROOT/evidence/$SESSION/behavior"
```

The attached bundle requires RTSP decode, decoded WebRTC, fresh ReID evidence
with tracker-local shadow-subject continuity and an observed open-set non-force
decision, an exact private identity-evidence linkage for the semantic gate,
authenticated bbox3d observation, and the global-world v2 contract. The latter
binds its minimal privacy-safe transcript to the exact sealed launcher session
and effective V3DT pipeline/tracker/camera/calibration/alignment/camInfo cohort;
v1 and camera-local evidence are rejected.
The supervisor records authenticated capability-health `instance_id` and
`run_id` values after readiness. The runner re-observes those values, rejects a
different runtime, and accepts only the exact plan-declared loopback endpoints
and launcher directory. Every canonical behavior report carries that same
session/lane/runtime identity. The runner creates the behavior output leaf as
an exact owner-owned mode `0700` directory
and rejects an existing insecure or linked directory instead of changing it.
This is required because semantic source/snapshot sealing is valid only beneath
an owner-private parent.
All six behavior producers are create-once. They preflight their complete set
of canonical output names, publish source/snapshot evidence before the report,
and never replace or permission-repair an existing file. If a gate stops after
publishing only part of its bundle, retain that directory for diagnosis, choose
a new `SESSION`, and rerun the supervisor/attachment. Do not delete, relabel, or
fill missing files in the old cohort.
The two zero-copy gates retain the exact 3 ms p99 threshold and zero core-copy
violation allowance. Their JSON results now include separate maximum WS and
REST p99 values plus bounded `noesis.zero_copy.boundary_diagnostics` v1
attribution for budgeted `total` stages. Payload content is never retained.
Large-response serialization uses a dedicated prewarmed worker. The 3 ms total
still includes its dispatch wait, producer-to-event-loop enqueue, conversion,
single admission-freeze encoding, and local send dispatch; each stage remains
visible independently. Sync sender receipts bind finite, bounded, immutable
pre-encoded bytes, and tests require one encoder pass. Explicitly configured
non-canonical coalescing dwell begins only after loop acceptance and is
excluded; canonical authority-gate journal/commit dwell is also excluded from
WebSocket CPU time. Canonical tracking/world/event/BEV is lossless and
non-coalesced. Any
delayed dispatch or send therefore remains a gate failure.

Boundary regression also requires a 256 MiB global frozen-byte limit in
addition to the 256-submission count: saturation rejects before scheduling,
success/failure/explicit gate abort/cancellation release exactly once, an
unresolved gate prevents quiescence, and the shutdown receipt has zero
in-flight bytes. Authenticated telemetry fanout defaults to 8 clients and
is hard-clamped to 16. The 17th client must close with `1013` /
`telemetry_capacity_reached` before registration or initial snapshots; a freed
slot must reconnect, and `/healthz` must still complete while telemetry is full.

Shutdown also requires a WebSocket provider-quiescence receipt before depth or
storage teardown. The receipt proves admission closed, every admitted blocking
depth/floorplan/auto-calibration callback completed, no active lease remains,
and the owned provider executor joined. After the listener is stopped and depth
admission closes, pipeline EOS/wait must quiesce probes before MapAnything
joins; storage closes last. A timeout is fail-closed and cannot be promoted.
The runner derives the identity JSONL from the
same supervisor session when the ownership evidence directory is supplied. It
does not infer identity accuracy from unlabeled household telemetry. Wholebody,
floorplan, world, identity, and semantic claims also carry checksum-covered,
bounded timestamped source transcripts; the first three persist only minimal
replay fields and explicitly exclude images, embeddings, secrets, and full raw
payloads. In
parallel, `--resource-soak` emits the same-session generic v2 cgroup/GPU report
plus checksum-bound raw samples required by the ownership registry. Validation
recomputes the report from those samples, binds the exact container, checkout,
realization, primary lane engine, and configs, and requires all report/sample
times to lie inside the inspected container lifetime. The retired V3DT-only
soak contract is rejected rather than silently reinterpreted.

After the supervisor has sealed a successful final session, record the relevant
dynamic acceptance in the external owner-private registry. This command is
offline: it validates and records existing evidence but never starts Docker or
uses the GPU.

Only a bounded canonical `run` session is promotable as `runtime_session`
evidence. Do not pass an `appliance-run` session to the recorder: appliance
deployment has persistent selector-bound state and an indefinite lifecycle, so
the ownership validator rejects it explicitly as a different evidence
contract.

```bash
python3 DS9/scripts/promote_runtime_ownership_evidence.py \
  --runtime-root "$NOESIS_DS9_RUNTIME_ROOT" \
  --artifact-root "$NOESIS_DS9_ARTIFACT_ROOT" \
  --docker-root "$NOESIS_DS9_DOCKER_ROOT" \
  --capability-id tracking.v3dt \
  --evidence-type runtime_session \
  --subject "v3dt-$SESSION" \
  --lane v3dt \
  --session-id "$SESSION"

python3 DS9/scripts/validate_runtime_ownership.py \
  --artifact-root "$NOESIS_DS9_ARTIFACT_ROOT" \
  --runtime-root "$NOESIS_DS9_RUNTIME_ROOT" \
  --docker-root "$NOESIS_DS9_DOCKER_ROOT" \
  --require-parity
```

The tracked matrix contains no realized/session selectors. A replacement or
revoke must supply the prior event digest through
`--supersedes-event-digest`; never edit or truncate the JSONL registry by hand.
Reserve at least 15 minutes for the same-session supervisor; the complete
serial behavior bundle is intentionally much longer than a 30-second canary.
It cannot certify shutdown by itself; acceptance requires the Shell 1
supervisor summary and sealed ordered-EOS evidence from that same session.

Wholebody49 uses the same pattern with `--lane wholebody49-s` and then
`--lane wholebody49-x`; both supervisor commands must include
`--resource-soak --duration-seconds 300` or longer. The attached runner's
`wholebody-occupied` gate requires people in scene, exact configured source
coverage, at least 30 seconds of strictly advancing source frames, provisional
floors of 2 frame IDs/second per source and no per-source observation gap above
2.5 seconds, no pipeline errors or core CPU-copy violations, and mode-specific
parser consumption. S must exercise instance-mask sampling; X must exercise
bbox sampling and must not exercise the mask path. A separate checksum-bound
report proves at least one directly decoded RTSP H264 frame and one decoded
WebRTC frame from the same session. The 300-second resource gate requires zero
OOM/OOM-kill growth, cgroup memory below 20 GiB, post-warmup growth at most
512 MiB and slope at most 1 MiB/s, PIDs strictly below 4096, S GPU process
memory below 10,000 MiB, and X below 11,000 MiB. These are provisional policy
bounds, not empirical performance claims, and are never relaxed automatically.
The ownership validator derives the complete source-ID set independently from
the reviewed lane pipeline and camera authorities, then requires exact typed,
ordered equality in the launch plan, occupied report, observed set, and source
transcript. Subsets, supersets, reorderings, and booleans are rejected for both
S and X even when a self-consistent producer report is resealed.
An empty house is reported as `scene_status=empty` and cannot satisfy occupied
acceptance. Visual overlay quality is reviewed from that mosaic rather than
inferred from engine serialization.

Do not run that command while engine maintenance or another runtime owns the
GPU. Plan mode never creates a container or requests GPU/network/IPC access.
The live mode alone adds GPU device `0` plus host network/IPC and requires an
exact 26 GiB memory plus equal memory-and-swap ceiling (no container swap
allowance), normal
exit-zero, ordered EOS acknowledgement, container removal, closed ports, and
an unchanged checkout. See `runtime_container_boundary.md` for the full mount,
lock, evidence, and failure contract.

The engine build/plan container remains deliberately different: it mounts no
runtime secrets because model artifacts and fingerprints must remain
secret-independent.

## Preflight And Rebuild Gates

Run preflight first inside the intended DS9 runtime environment:

```bash
python3 DS9/scripts/ds9_preflight.py
```

Confirm the output reports `pyservicemaker` from
`/usr/local/lib/python3.12/dist-packages`. A stale user-site
`pyservicemaker` wheel can shadow the DeepStream 9 binding and cause native
heap corruption during construction/destruction and shutdown; the DS9 preflight
and launcher pin the system binding before importing Service Maker. Preflight
requires driver `595.58.03` or newer independently from TensorRT 10.16.0.72.
The current host therefore reports `595.71.05` as compatible but must still
exit nonzero when run against the host TensorRT 10.13 stack; do not weaken or
bypass that failure. Preflight
also requires `nvdsroiexclude`, `noesisforceidr`, and `noesiseos` to resolve to
the exact DS9-owned binaries under `DS9/gst-plugins/`; finding a same-named DS8
or system plugin is a failure. It also imports the exact GI namespaces used by
the WebRTC gateway and inspects every gateway factory; missing PyGObject,
GstWebRTC introspection, NICE, or an RTP/RTSP element is a startup failure.

If native/plugin/parser/engine artifacts need rebuilding:

```bash
bash DS9/scripts/build_gst_plugins.sh
bash DS9/scripts/build_trt_plugins.sh
bash DS9/scripts/build_custom_parsers.sh
bash DS9/scripts/build_native_extensions.sh
# Build engines through run_canonical_engine_maintenance.sh with explicit
# NOESIS_DS9_DOCKER_ROOT and NOESIS_DS9_ARTIFACT_ROOT.
python3 DS9/scripts/ds9_preflight.py
```

Preflight attests every DS9 native extension before importing any native code.
It requires the sole active-CPython-ABI output under
`DS9/native_extensions/`, exact declared source membership and aggregate source
hashing, and exact output SHA-256 from `DS9/asset_manifest.yaml`. A source being
newer than an output is not stale evidence: release checkouts legitimately
refresh tracked-source mtimes, and CUDA compilation may not be byte-
deterministic. A provenance mismatch must be reconciled through an explicit
DS9 rebuild and reviewed manifest update; never touch timestamps or select a
nearby binary to satisfy startup.

If MapAnything needs a fresh ONNX/plan:

```bash
python3 utils/onnx2trt/export_ma_onnx/export_to_onnx.py \
  --repo external/map-anything \
  --outdir DS9/models/onnx \
  --h 294 --w 518 \
  --skip-ort --skip-simplify --skip-shape-inference

# Ensure the images-input export is named:
# DS9/models/onnx/mapanything_images_294x518_b3.onnx

NOESIS_MAPANYTHING_PYTHON=/path/to/export-venv/bin/python \
NOESIS_MAPANYTHING_GPU_GUARD_MB=9000 \
NOESIS_MAPANYTHING_GUARD_POLL_SECONDS=1 \
DS9/scripts/build_mapanything_guarded.sh

python3 DS9/scripts/ds9_preflight.py
```

## Launch DS9 Runtime

Use one terminal for the runtime:

```bash
cd /workspace
export PYTHONUNBUFFERED=1
export NOESIS_MOSAIC_RTSP_ENABLED=1
export NOESIS_MOSAIC_WEBRTC_ENABLED=1
export NOESIS_POSE_FEATURE_DEBUG=1
python3 DS9/noesis/ds9_runtime.py \
  --cameras-config config/cameras.yaml \
  --enable-rest \
  2>&1 | tee /tmp/noesis-ds9-runtime.log
```

Expected startup evidence:

- `DS9/scripts/ds9_preflight.py` passes.
- Engines load for `mapanything_fullframe`, `depth_tracking_fullframe`,
  `yolo26_pose`, `reid_sgie`, and the active PGIE such as `yolo11_pgie`.
- RTSP port `8554` opens.
- WebSocket/WebRTC signaling port `6008` opens.

Check ports from another shell in the same container:

```bash
ss -ltnp | rg ':(8554|6008)\b'
```

Check runtime log:

```bash
rg 'mapanything_fullframe|depth_tracking_fullframe|yolo26_pose|reid_sgie|yolo11_pgie|Pose features debug|world_quality_reason|pose_present' /tmp/noesis-ds9-runtime.log
```

On the host, `DS9/noesis/ds9_runtime.py` defaults `--storage-base` to
`DS9/data/depth` to avoid root-owned `data/depth` directories left by Docker
validation. If you override `--storage-base`, verify the target is writable by
the host user before running MapAnything or floorplan gates.

## Live Validation Runner

For future occupied-camera live-RTSP evidence passes, prefer the DS9 runner so
all gate logs and the shutdown result land in one evidence bundle:

```bash
python3 DS9/scripts/ds9_live_validation_runner.py
python3 DS9/scripts/ds9_live_validation_runner.py --lane v3dt
python3 DS9/scripts/ds9_live_validation_runner.py --lane wholebody49-s
python3 DS9/scripts/ds9_live_validation_runner.py --lane wholebody49-x
```

The shared host-side clients import authentication helpers through the explicit
repository `scripts` package. Keep `scripts/__init__.py` in release snapshots;
without it, an installed package named `scripts` can win over the repository's
former implicit namespace and make every Python gate fail before connecting.
`tests/test_internal_auth_smoke_clients.py` verifies the concrete package and
helper origins in a fresh subprocess.

By default the runner:

- runs `DS9/scripts/ds9_preflight.py`;
- launches the owned `DS9/noesis/ds9_runtime.py` as the canonical YOLO26-m
  baseline profile, with required internal bearer authentication, REST, RTSP,
  WebRTC, and ReID enabled;
- requires advancing authenticated capability health plus the WebSocket health
  contract and an RTSP DESCRIBE response before and after the behavior gates;
- runs RTSP decode, WebRTC, identity-v2 shadow health/continuity, BEV/track
  parity, strict DS9 bridge, all-active-camera floorplan, MapAnything depth,
  zero-copy stats, and REST-backed zero-copy gates;
- sends SIGTERM and requires exact exit zero; and
- requires the owned runtime log to prove accepted orderly EOS, the
  `reason=shutdown_requested` EOS callback, Service Maker `wait()` return, and
  `Shutdown complete` in that order, while rejecting generic `ERROR`/`CRITICAL`
  lines plus wait-timeout, GStreamer teardown, and native-failure signatures.

The non-baseline lanes are equally fail-closed: they hard-pin their pipeline,
camera config, PGIE, size, and tracking mode and reject arbitrary config
overrides. V3DT substitutes authenticated `v3dt-bbox3d` and `v3dt-world` gates
for baseline BEV/depth gates. Its public frame is Y-up `backend_world_m`, while
raw `bbox3d`/`velocity3d` remain tracker-tuple diagnostics. Wholebody49 substitutes the mode-specific
`wholebody-occupied` gate. When these are attached to the canonical container
supervisor, pass `--no-spawn --skip-preflight`; the runtime already performed
DS9 preflight inside the pinned image, while the host deliberately remains on
the DS8 TensorRT SDK.

The identity gate requires real, fresh server-produced embedding observations,
one non-null shadow subject that remains stable on a tracker across frames,
coherent observation keys, and authenticated `/api/v2/reid/health` before and
after collection. The runner hard-pins `NOESIS_IDENTITY_V2_MODE=shadow`; it
rejects public authority and records `semantic_accuracy=not_evaluated` because
live telemetry has no truth labels. Cross-camera assignment continuity and a
fresh open-set non-force observation are reported as `observed` or
`not_observed`, never silently promoted to pass. For a staged or naturally
occurring session, make either observation mandatory:

For one continuous camera-local tracker, an accepted subject may not change.
The coordinator hard-masks alternatives until a real fresh-evidence gap expires
the tracker state, while the accepted subject must continue to pass every
open-set gate. During enabled evidence capture, every public
`embedding_present=true` row must join its exact observation key, complete
persisted sequence/model/dimension triad, matching strict observation, and
private hash-chain row. A subject flip, stale legacy embedding bit, or DS9
household StableID admission with automatic merge enabled is a runtime defect;
do not weaken the gate or edit the sealed transcript.
Private score rows must also preserve the exact post-hard-mask resolver
candidate surface, including rejected raw similarities and continuity or
copresence constraint reasons; a separately sampled gallery is not replay
evidence.

The semantic observation gate is version 3. Its exact persisted embedding row
is the identity anchor; pose, registered depth, and `backend_world_m` may come
from separate inference frames only within one inclusive 1.5-second capture-
and-observation window. Registered depth means
`depth_registration_status=ok`, a finite positive `depth_registered_m`, and a
matching `depth_used_m` when that alias is present. Every joined frame must
retain the same runtime run, source, camera, tracker lifecycle generation,
non-null identity-v2 subject, compatibility SID, resident UUID or visitor
generation, and canonical camera-calibration/tracking-model/pipeline-config
fingerprints. The publisher owns a contiguous per-source publication sequence;
the per-processed-frame lifecycle registry forces tracker-set transitions,
emits exact tombstones, and assigns a new generation after any disappearance.
Capture status and media PTS must remain coherent. An intervening empty frame,
missing publication, null/conflicting identity, tracker-ID reuse, fingerprint
change, or replay breaks the cohort.

The sealed source is a strict semantic projection, not a copy of the public
tracking payload. Sensitive-name variants and every 256-item numeric vector are
rejected before projection, including non-finite vectors, so projection can
never silently hide a public biometric leak. The live runner and ownership
validator both replay the complete owner-private report, source transcript, and
identity snapshot; an incomplete nested report or coordinated report-only
relabel is not accepted.

```bash
python3 DS9/scripts/ds9_live_validation_runner.py \
  --identity-require-cross-camera \
  --identity-require-open-set
```

These flags prove that the corresponding live mechanism was exercised. They do
not establish resident/unknown classification accuracy and do not clear the
separate licensed-truth, coordinator-replay, occupied-scene, or authority
cutover gates. Do not create fake resident labels to satisfy them.

The version-4 floorplan gate derives the active camera inventory from the
reviewed pipeline and camera configs and serializes fresh requests because all
cameras share one process-owned MapAnything valve. Every camera must return a
non-cache v7 `camera_local_ground_m` payload bound to one immutable snapshot
reference, write ID, content digest, and timestamp plus its checksum-valid
capture-event receipt. The receipt must prove clean pre/post idle and storage
barriers and explicit depth-only `rgb.status=not_requested` evidence.

After all fresh captures, the gate requires active-floorplan and local-BEV N/N
health for the reviewed cameras. It then issues one cache-only request per
camera and requires the identical snapshot identity and floorplan/layer
digests, no fresh-capture fields, and byte-equivalent active-registry and
capture-controller health before/after. A successful first camera, any pre-v4
gate report, a stale/substituted snapshot, controller contention, cache fill on
read, partial BEV health, or the 1x1 zero-density sentinel cannot pass or enter
the ownership registry.

The reconciled BEV health is service activity, not occupancy. A successfully
published exact frame with `footpoints: []` is `active_ready`;
`inactive_ready` means no exact BEV success yet. Before the first valid active
floorplan, the renderer may report `startup_pending` and emit no local BEV, but
active-floorplan N/N is still incomplete. Any invalid authority, or any loss
after readiness, is fatal. The registered behavior selector is
`mapanything_depth_quality_v4`; no pre-v4 report is promotion evidence.

Standalone evidence is written under `DS9/build/live_validation/<timestamp>/`
with a generated session ID. Baseline canonical reports are
`identity-open-set-occupied.json`, `semantic-observation.json`, and
`mapanything-depth-quality.json`; V3DT uses
`v3dt-identity-open-set-occupied.json`, `semantic-observation.json`, and
`v3dt-world-contract.json`; both Wholebody lanes use
`wholebody49-occupied-scene.json`, `wholebody49-media-decode.json`, and
`runtime-resource-soak-report.json` with their checksum-covered source/sample
documents. Reports are owner-private and are revalidated
by the runner before their claims enter the summary. ReID, semantic, floorplan,
V3DT world, Wholebody occupied, and Wholebody decoded-media gates each invoke
their producer-owned strict validator, reread owner-only single-link bounded
files, verify canonical filenames and source digests, replay the analyzer from
the source, and require exact canonical report equality. The runner records
`exact_source_replay=pass` only after that succeeds. It never opens or validates
a behavior report when the producing subprocess failed, timed out, or returned
nonzero. A plain `--no-spawn`
attachment requires an explicit `--session-id`. To create ownership evidence,
also supply the exact supervisor launcher directory as shown above; the runner
rejects a missing or mismatched private launch plan, runtime identity, launcher
path, or canonical endpoint URL. In an ownership-attached baseline run, both
`reid_open_set_occupied_v1` and lane-neutral `semantic_gate_v3` are required and
the semantic report/source/identity-snapshot trio is written directly into the
launcher directory for final checksum coverage.

Semantic v3 replay compares the exact encoded report, source, and every JSONL
snapshot line against their canonical encoders before semantic equality or hash
replay. Whitespace-only rewrites, reordered keys, alternate finite-number
spellings, duplicate keys, and non-finite constants all fail.

Identity source transcripts are schema/contract v2. Version 2 adds the exact
gate policy and projected before/after identity-health snapshots needed to
recompute runtime-health and claim evidence. Version-1 identity transcripts do
not contain enough evidence and are intentionally non-promotable; capture a
fresh session instead of translating or relabeling them. V3DT world v2,
Wholebody occupied v2, and Wholebody media v1 formats remain unchanged, but
their sealed bytes must now be unique-key, finite, canonical JSON.

```bash
python3 DS9/scripts/ds9_live_validation_runner.py \
  --no-spawn \
  --session-id <attached-session-id>
```

An attached run can validate behavior but cannot certify the owned process
lifecycle; the shutdown step is explicitly reported as skipped.

Use `--skip <gate>` only to isolate a known external blocker, and document the
skip in the resulting summary. Lane-specific gate names additionally include
`v3dt-bbox3d`, `v3dt-world`, and `wholebody-occupied`.

If live RTSP ingress starts logging repeated source reset/reconnect warnings,
do not treat late WebRTC, fresh MapAnything, BEV, ReID, or timing failures as
clean DS9 migration evidence until the source state is recovered. The observed
host failure signature was:

- `No data from source since last 10 sec. Trying reconnection`.
- RTSP `Received end-of-file` or `System error`.
- WebRTC warning `No RTSP frames yet; refusing to answer after 15s`.
- fresh `ma_depth_response` timeout with cached floorplan still available.
- zero-copy p99 timing above 3 ms during the degraded period, while fresh-start
  zero-copy runs passed.

## Pose Activation Gate

Run this with the runtime still running. Do not force the DS8 native pose
metadata bridge; under DS9 it segfaulted at the first analytics batch. The
supported DS9 path decodes YOLO26 tensor metadata through Service Maker and
attaches `NOESIS.POSE_FEATURES` with the DS9-built native extension.

```bash
python3 - <<'PY'
import asyncio
import json
import time
import websockets

async def main():
    uri = "ws://127.0.0.1:6008"
    deadline = time.time() + 45
    tracks = 0
    pose_true = 0
    world_valid = 0
    backend_world_m = 0
    reasons = {}
    async with websockets.connect(uri, max_size=None) as ws:
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            if isinstance(raw, (bytes, bytearray)):
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if msg.get("type") != "tracking":
                continue
            for track in msg.get("tracks") or []:
                tracks += 1
                if track.get("pose_present") is True:
                    pose_true += 1
                if track.get("world_valid") is True:
                    world_valid += 1
                if track.get("world_frame") == "backend_world_m":
                    backend_world_m += 1
                reason = track.get("world_quality_reason")
                if reason:
                    reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    print(json.dumps({
        "tracks": tracks,
        "pose_present_true": pose_true,
        "world_valid_true": world_valid,
        "backend_world_m": backend_world_m,
        "world_quality_reasons": reasons,
    }, indent=2))
    raise SystemExit(0 if pose_true > 0 and world_valid > 0 and backend_world_m > 0 else 1)

asyncio.run(main())
PY
```

If the implementation still uses the existing debug counter, require this log
line to show increasing `attached` counts:

```bash
rg 'Pose features debug:.*attached=[1-9]' /tmp/noesis-ds9-runtime.log
```

Do not proceed to readiness claims until this gate passes with
`pose_present_true > 0`, `world_valid_true > 0`, and
`world_frame=backend_world_m` in live telemetry.

## BEV And Track Parity Smoke

Run only after pose activation is proven:

All HTTP and WebSocket smoke clients below require the existing owner-only
internal bearer. They default to `NOESIS_INTERNAL_AUTH_TOKEN_FILE`, then the
standard appliance state path. Use `--auth-token-file <path>` when validating
against a separately provisioned runtime. The option always carries a path,
never token bytes; anonymous or query-token validation is not accepted.

```bash
python3 scripts/menon_bev_track_parity_smoke_test.py \
  --no-spawn \
  --ws ws://127.0.0.1:6008 \
  --pipeline-config DS9/config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --duration 45
```

Pass criteria:

- every world-valid tracking row is `backend_world_m`;
- exact BEV/tracking source-frame-time pairs cover configured cameras N/N;
- every BEV frame matches the sealed active-floorplan authority and configured
  `camera_local_ground_m` frame;
- occupied evidence includes footpoints, exact tracker associations, and one
  independent calibrated geometry comparison per footpoint;
- duplicate, stale, reused, out-of-bounds, display-source, trail, authority,
  and geometry-unavailable violation counters are all zero;
- p95 camera-local geometry error is at or below the script threshold.

The following is a historical pre-camera-local reference only; it is not
current v4 promotion evidence:

```json
{
  "track_total": 2769,
  "track_world_valid": 2769,
  "track_world_frame_backend_world_m": 2769,
  "bev_total": 1632,
  "bev_world_frame_backend_world_m": 1632,
  "comparisons": 1887,
  "p95_err_m": 0.0
}
```

## WebRTC Smoke

Run against the already running DS9 runtime:

```bash
python3 scripts/webrtc_gateway_smoke_test.py \
  --ws ws://127.0.0.1:6008 \
  --duration 6 \
  --pt 103 \
  --min-rtp 10 \
  --min-decoded 1
```

Recent known-good DS9 result after dependency installation:
`rtp_packets=2287`, `decoded_frames=184`.

## RTSP Mosaic Gate

```bash
gst-launch-1.0 -e rtspsrc location=rtsp://127.0.0.1:8554/mosaic latency=100 ! \
  rtph264depay ! h264parse ! avdec_h264 ! fakesink sync=false
```

This should preroll/play without connection or decode errors.

## MapAnything, Depth, Floorplan, And ReID Gates

Run these with DS9 already running. Pass `--no-spawn` to the shared clients so
they use that same instance; the DS9-specific identity and floorplan gates are
attach-only and never spawn a runtime. These are the authenticated clients
selected by the canonical DS9 live-validation runner.

The DS9-specific behavior producers require the runtime identity they are
describing. For a supervised run, copy these values from that session's
owner-private `runtime-identity.json`; otherwise obtain them from the
authenticated `/api/v1/health/capabilities` response before collection:

Before a live rerun after changing the depth tensor native source, rebuild the
DS9-owned `noesis_depth_tracking_tensor_ext`, refresh its manifest provenance,
and rerun static preparation. Startup rejects both an older binary without
`capture_mapanything_tensor_layers_exact` and a binary that still co-exposes
the retired generic `capture_tensor_layers` export. The runtime must report
advancing `mapanything_exact_native_capture_frames_total`, nonzero depth FPS,
and no `mapanything_exact_native_capture_failures_total`,
`mapanything_async_postprocess_failures_total`, or
`mapanything_async_queue_full_total` before either RPC gate can qualify. Clean
shutdown additionally requires `mapanything_async_shutdown_total` when the
worker started, zero `mapanything_async_shutdown_failures_total`, and the
runtime's joined MapAnything-worker lifecycle marker before storage closes.

The CPU-edge copy increments the public
`zero_copy_core.counters.tensor_host_copies_total.mapanything` and
`zero_copy_core.counters.tensor_boundary_copy_bytes_total.mapanything` plus
`zero_copy_core.counters.boundary_serialization_prep.mapanything.exact_native_tensor_to_host`
counters; the in-process instrumentation snapshot also retains duration and
payload bytes for that named preparation stage. It copies each already
frame-local tensor once, for an exact `1,827,504` float32 host bytes per frame,
and is allowed because dense MapAnything snapshots are a serialization/storage
product. The copy remains inside the metadata-lifetime lease because nvinfer
owns the source pointers and exposes no retained device-buffer contract to this
hook; the bridge releases the GIL during copy and records its elapsed time.
Alignment, calibrated masking, storage, and publication remain on the bounded
asynchronous worker rather than the Service Maker operator. The native contract
must not apply `batch_id` as a pointer offset: DS9 nvinfer has already offset
the frame metadata pointer before the hook receives it. Shutdown must close
capture admission, drain accepted jobs, and join that non-daemon worker before
depth storage or WebSocket providers are released; final-job poison fails the
run at teardown.

```bash
SESSION=manual-session
RUNTIME_INSTANCE_ID=copy-authenticated-instance-id-here
RUNTIME_RUN_ID=copy-authenticated-run-id-here
```

```bash
python3 scripts/ma_depth_rpc_smoke_test.py \
  --no-spawn \
  --ws ws://127.0.0.1:6008 \
  --pipeline-config DS9/config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --camera family-room

python3 scripts/zero_copy_stats_smoke_test.py \
  --no-spawn \
  --stats-ws ws://127.0.0.1:6008 \
  --pipeline-config DS9/config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --duration-s 60

python3 scripts/zero_copy_smoke_test.py \
  --no-spawn \
  --stats-ws ws://127.0.0.1:6008 \
  --rest-url http://127.0.0.1:8080/api/v1/depth/refresh \
  --pipeline-config DS9/config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --duration-s 90

python3 DS9/scripts/ds9_floorplan_live_gate.py \
  --ws ws://127.0.0.1:6008 \
  --pipeline-config DS9/config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --max-age-sec 120 \
  --session-id "$SESSION" \
  --runtime-lane baseline \
  --runtime-instance-id "$RUNTIME_INSTANCE_ID" \
  --runtime-run-id "$RUNTIME_RUN_ID" \
  --out DS9/build/live_validation/manual/mapanything-depth-quality.json \
  --source-out \
    DS9/build/live_validation/manual/mapanything-depth-quality-source.json

python3 DS9/scripts/ds9_identity_shadow_live_gate.py \
  --ws ws://127.0.0.1:6008 \
  --rest http://127.0.0.1:8080 \
  --pipeline-config DS9/config/infer.yaml \
  --duration 35 \
  --session-id "$SESSION" \
  --runtime-lane baseline \
  --runtime-instance-id "$RUNTIME_INSTANCE_ID" \
  --runtime-run-id "$RUNTIME_RUN_ID" \
  --out DS9/build/live_validation/manual/identity-shadow.json
```

Both clients use the existing owner-only bearer selected by
`--auth-token-file` or `NOESIS_INTERNAL_AUTH_TOKEN_FILE`. Add
`--require-cross-camera` and/or `--require-open-set` to the identity command
only when the real scene is expected to exercise those events.

`zero_copy_smoke_test.py` requires the DS9 runtime to be launched with
`--enable-rest`; it is not covered by runtime runs that use `--disable-rest`.

If live RTSP ingest is degraded and the goal is to isolate ReID rather than
prove the full production graph, use a validation-only MP4 config under
`DS9/build/` with local file URIs, `streammux.live-source=0`,
`models.reid.enable=true`, pose/depth/MapAnything disabled, and
`validation.reid_smoke_depthless=true`. Launch it with:

```bash
NOESIS_ALLOW_DEPTHLESS_REID_SMOKE=1 \
NOESIS_REID_ENABLED=1 \
NOESIS_MOSAIC_RTSP_ENABLED=0 \
NOESIS_MOSAIC_WEBRTC_ENABLED=0 \
python3 DS9/noesis/ds9_runtime.py \
  --pipeline-config DS9/build/infer_reid_mp4.yaml \
  --cameras-config config/cameras.yaml \
  --disable-rest
```

Then run:

```bash
python3 scripts/reid_stable_id_smoke_test.py \
  --no-spawn \
  --ws ws://127.0.0.1:6008 \
  --pipeline-config DS9/build/infer_reid_mp4.yaml \
  --cameras-config config/cameras.yaml \
  --duration 35
```

This focused gate passed on 2026-06-16 for `family-room:18`. Treat it only as
ReID-path evidence; it does not replace the live-RTSP production ReID, BEV, or
world parity gates.

## Broader Migration Plan Gates

The core production path gates above are necessary but are not the same as full
DS8 option-surface parity. Keep these broader gates separate in reports:

- ROI pruning: prove `nvdsroiexclude` removes excluded objects before tracker,
  ReID, and pose work. Plugin presence from preflight is not enough.
- V3DT: if DS9 is expected to run a V3DT profile, prove
  `noesis_v3dt_meta_ext` extracts visibility, image-foot, and 3D bbox metadata
  in a DS9 runtime smoke. Before the exclusive-GPU window, require the external
  source gate and all three isolated plans:

  ```bash
  NOESIS_DS9_ARTIFACT_ROOT=<absolute-artifact-root> \
    python3 DS9/scripts/stage_canonical_sources.py --verify-only
  NOESIS_DS9_DOCKER_ROOT=<absolute-secondary-docker-root> \
  NOESIS_DS9_ARTIFACT_ROOT=<absolute-artifact-root> \
    DS9/scripts/run_canonical_engine_maintenance.sh --only yolo26_seg_s --plan
  NOESIS_DS9_DOCKER_ROOT=<absolute-secondary-docker-root> \
  NOESIS_DS9_ARTIFACT_ROOT=<absolute-artifact-root> \
    DS9/scripts/run_canonical_engine_maintenance.sh --only bodypose3dnet --plan
  NOESIS_DS9_DOCKER_ROOT=<absolute-secondary-docker-root> \
  NOESIS_DS9_ARTIFACT_ROOT=<absolute-artifact-root> \
    DS9/scripts/run_canonical_engine_maintenance.sh --only v3dt_tracker_reid --plan
  ```

  Build without `--plan` only after every GPU compute owner has stopped and the
  host driver satisfies the DeepStream 9.1 requirement of `595.58.03` or newer,
  in the same order. Then run the canonical `--lane v3dt` plan/canary and attached
  V3DT validation bundle above. Both V3DT clients require the owner-only bearer;
  anonymous WebSocket success is not accepted. The world gate requires
  `world_frame=backend_world_m` and `world_source=bbox3d`, all configured cameras
  with advancing per-camera continuity, native tracker metadata, old raw-tuple
  rejection, floor consistency, native image-foot reprojection at or below
  40 px p95, and exact independent `image_base` replay. It does not claim room
  membership, camera overlap, MV3DT peer identity, time synchronization, or
  fused positions.
- Alternate model profiles: smoke enabled YOLO26-seg, RF-DETR, and Wholebody49
  parser/profile combinations rather than relying only on build/preflight
  success. Wholebody49 `s` must prove mask/tensor/OSD output; `x` must prove
  bbox-only output and no mask rendering. Use the enumerated
  `wholebody49-s`/`wholebody49-x` supervisor and runner lanes; the older direct
  `timeout ... ds9_runtime.py` snippets are diagnostic startup commands, not
  lifecycle or provenance acceptance.
- Native bridge behavior: when object-depth, depth tensor, ReID, or V3DT
  native paths are touched, run focused smokes for those contracts in addition
  to the end-to-end core runtime gates.

Run the focused DS9 bridge smoke against an already-running DS9 runtime:

```bash
python3 DS9/scripts/ds9_bridge_contract_smoke_test.py \
  --ws ws://127.0.0.1:6008 \
  --duration 75 \
  --require-embedding-track
```

For focused ROI hot-restore validation, attach to an authenticated canonical
session whose supervisor validated and mounted the appliance-persistent
analytics pair. Only `persistent/analytics` survives a new supervisor session;
the rest of `/var/lib/noesis/state` remains session-local. The launcher records
private before/after copies and hashes in that session's launcher evidence.
Choose a camera with a real person currently visible; an unoccupied camera is
reported as `blocked` with exit code `2`, never as a synthetic pass. The canonical
implementation is the root validation client
`scripts/roi_reload_smoke_test.py`. The DS9-local path is only an argv/exit-code
compatible launcher for that same file and has no alternate gate behavior.

```bash
SESSION=<same-supervisor-session-id>
python3 scripts/roi_reload_smoke_test.py \
  --no-spawn \
  --hot-restore \
  --camera kitchen \
  --auth-token-file ~/.local/state/noesis/gateway-token \
  --ws ws://127.0.0.1:6008 \
  --rest http://127.0.0.1:8080 \
  --pipeline-config DS9/config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --baseline-timeout 45 \
  --excluded-timeout 45 \
  --restore-timeout 60 \
  --evidence "$NOESIS_DS9_RUNTIME_ROOT/evidence/$SESSION/behavior/roi-hot-restore.json"
```

The gate snapshots the exact original REST stage and active native receipt,
requires advancing target-camera frames and advancing real tracker frames,
then posts a temporary full-frame exclusion only for that stream while sending
all other streams unchanged. HTTP success alone cannot pass: the response and
WebSocket stats must expose the same native request/accepted/failed sequence,
active config SHA-256, reload error count, and objects-removed count. Live
target frames must continue, person tracks must disappear, and the native
objects-removed counter must rise.

Restoration runs from `finally` after any attempted mutation. It must receive a
second exact native acknowledgement whose active hash equals the original,
read back a semantically identical REST stage, and observe the real person
return on continuing frames. A restore failure is terminal even when exclusion
worked. Evidence is bounded owner-only JSON and contains no bearer, endpoint,
source URI, ROI geometry, or image. The 2026-06-16 `baseline_total=2`,
`excluded_total=0`, `restored_total=4` result came from the older counter-based
client and is historical characterization, not evidence for this stricter gate.

Separate native bridge pass criteria (not substitutes for the ROI gate above):

- `depth_tracking_device_frames_total > 0`
- `object_depth_gpu_roi_copies_total > 0`
- `object_depth_attach_total > 0`
- `object_depth_status_total.ok > 0`
- `tensor_host_copies_total.reid > 0`
- `core_path.cpu_copy_violation.total == 0`

Latest broader-gate evidence from 2026-06-15 and 2026-06-16:

- Zero-copy stats passed:
  `samples=43`, `max_zero_copy_violations=0`,
  `max_boundary_p99_ms=1.034443`, `ws_depth_requests=43`,
  `ws_depth_responses=5`.
- Zero-copy REST depth passed with DS9 launched using `--enable-rest`:
  `samples=62`, `max_zero_copy_violations=0`,
  `max_boundary_p99_ms=1.146929`, `rest_refresh_attempts=8`,
  `rest_refresh_success=8`, `rest_refresh_last_status=ok`.
- Current ROI gate/API/builder/native coverage is:
  `pytest -q tests/test_roi_reload_smoke_test.py tests/test_analytics_api.py
  tests/test_pipeline_build.py DS9/tests/test_nvdsroiexclude_plugin.py`.
  Historical live full-frame exclusion pruned pose/debug object counts to zero
  after the `nvdsroiexclude` reload.
- The historical focused MP4 ROI hot-restore observed:
  `baseline_total=2`, `excluded_total=0`, `restored_total=4`.
- YOLO26 segmentation profile smoke passed:
  `python3 DS9/noesis/ds9_runtime.py --cameras-config config/cameras.yaml --pgie-profile yolo26_seg --size s --disable-rest`.
- RF-DETR segmentation profile smoke passed after generated parser/label path
  fixes:
  `python3 DS9/noesis/ds9_runtime.py --cameras-config config/cameras.yaml --pgie-profile rfdetr_seg --size m --disable-rest`.
- Bridge-specific object-depth, depth tensor, and ReID native extraction smoke
  passed with
  `python3 DS9/scripts/ds9_bridge_contract_smoke_test.py --ws ws://127.0.0.1:6008 --duration 75 --require-embedding-track`:
  `stats_samples=76`, `tracking_messages=1472`, `tracks_seen=2614`,
  `depth_ok_tracks=2248`, `embedding_tracks=2609`,
  `depth_tracking_device_frames_total=1533`,
  `object_depth_gpu_roi_copies_total=2246`, `object_depth_attach_total=2612`,
  `object_depth_status_total.ok=2246`, `tensor_host_copies_total.reid=354`,
  `core_path.cpu_copy_violation.total=0`, and `pipeline_errors=[]`.
- RF-DETR detect-only profile smoke passed after staging
  `DS9/models/onnx/rfdetr_{n,s,m}_*.onnx` and building
  `DS9/models/engines/rfdetr_{n,s,m}_*_b3_fp16.engine`:
  `python3 DS9/noesis/ds9_runtime.py --cameras-config config/cameras.yaml --pgie-profile rfdetr --size s --disable-rest`.
  The runtime loaded `DS9/models/engines/rfdetr_s_512_b3_fp16.engine` and
  stayed alive until the bounded smoke timed out with `RUNTIME_RC=124`.
- Occupied-camera live-RTSP host validation passed:
  - BEV/track parity:
    `track_total=11435`, `track_world_valid=11426`,
    `bev_world_frame_backend_world_m=6980`, `comparisons=11065`,
    `p95_err_m=0.0`, and lagged p95 `0.11610091475856413`.
  - ReID stable-ID smoke: `family-room:16`.
  - Bridge smoke with `--require-embedding-track`:
    `tracking_messages=5373`, `tracks_seen=11436`,
    `depth_ok_tracks=11066`, `embedding_tracks=11398`,
    `depth_tracking_device_frames_total=4833`,
    `object_depth_gpu_roi_copies_total=10980`,
    `object_depth_attach_total=11350`,
    `object_depth_status_total.ok=10980`,
    `tensor_host_copies_total.reid=971`,
    `core_path.cpu_copy_violation.total=0`, and `pipeline_errors=[]`.
  - WebRTC smoke: `rtp_packets=1438`, `decoded_frames=92`.
  - RTSP mosaic decode held for 25 seconds and sustained decode held for
    60 seconds.
  - Zero-copy stats:
    `samples=7`, `max_zero_copy_violations=0`,
    `max_boundary_p99_ms=1.123324`, `ws_depth_requests=38`, and
    `ws_depth_responses=4`.
  - MapAnything depth RPC passed cache-first plus fresh for `family-room`.
  - Floorplan RPC passed for `kitchen` and then `family-room`; the first
    `family-room` attempt timed out before succeeding on retry.
  - Live RTSP SIGINT shutdown exited `0`, closed ports `6008`, `8080`, and
    `8554`, and left no DS9 runtime process. The shutdown tail logged
    `Wait thread did not terminate cleanly` and one `source_2` reconnect warning
    after EOS, with no native heap/fatal markers.
- V3DT source/config/build staging and its selected engine realization now exist
  under DS9 ownership, with large bytes in the explicit external artifact root.
  Fresh runtime/world/identity/resource/shutdown evidence remains required. Do
  not use root DS8 pipeline/tracker artifacts as DS9 evidence.
- Host cutover evidence from 2026-06-16 UTC:
  - preflight passed with host DeepStream 9.0.0 / TensorRT 10.14.1.48.
  - fresh host runtime loaded all five engines and opened `:8554`, `:6008`, and
    `:8080`.
  - WebRTC passed with `rtp_packets=6945`, `decoded_frames=602`.
  - RTSP mosaic decode held until bounded timeout with no decode error.
  - zero-copy stats passed with `samples=41`,
    `max_boundary_p99_ms=1.002647`, `max_zero_copy_violations=0`.
  - REST-backed zero-copy passed with `samples=56`,
    `max_boundary_p99_ms=2.549196`, `rest_refresh_success=9`.
  - floorplan RPC passed.
  - native bridge counters moved for object-depth/depth/ReID:
    `depth_tracking_device_frames_total=4389`,
    `object_depth_gpu_roi_copies_total=31`, `object_depth_attach_total=46`,
    `object_depth_status_total.ok=31`, `tensor_host_copies_total.reid=5`,
    `core_path.cpu_copy_violation.total=0`.
  - focused host MP4 ReID stable-ID smoke passed with the validation-only
    depthless ReID config: `family-room:18`.
  - MP4 shutdown/native cleanup smoke passed after pinning the DS9 system
    `pyservicemaker` and removing the stale user-site shadow install: SIGINT
    posted EOS and exited `0` with no fatal Python, segfault, malloc,
    double-free, or heap-corruption markers. Finite MP4 EOS
    (`streammux.live-source=0`, `NOESIS_DS8_LOOP_LOCAL_MP4=0`) also exits `0`.
    Looping MP4 sources could log `Wait thread did not terminate cleanly` under
    the historical launcher workaround. The current gate must reject that line;
    this evidence is regression provenance, not shutdown acceptance.
  - occupied-camera live-RTSP host validation later proved ReID stable IDs,
    BEV/track parity, strict embedding-track bridge, MapAnything depth, floorplan,
    RTSP/WebRTC media output, zero-copy stats, and live RTSP SIGINT shutdown.
    See the broader-gate evidence above for the exact counters.

Suggested bounded alternate-profile commands:

```bash
timeout 45s python3 DS9/noesis/ds9_runtime.py \
  --cameras-config config/cameras.yaml \
  --pgie-profile yolo26_seg \
  --size s \
  --disable-rest

timeout 75s python3 DS9/noesis/ds9_runtime.py \
  --cameras-config config/cameras.yaml \
  --pgie-profile rfdetr_seg \
  --size m \
  --disable-rest

timeout 75s python3 DS9/noesis/ds9_runtime.py \
  --cameras-config config/cameras.yaml \
  --pgie-profile rfdetr \
  --size s \
  --disable-rest

timeout 75s python3 DS9/noesis/ds9_runtime.py \
  --cameras-config config/cameras.yaml \
  --pgie-profile wholebody49 \
  --size s \
  --disable-rest

timeout 75s python3 DS9/noesis/ds9_runtime.py \
  --cameras-config config/cameras.yaml \
  --pgie-profile wholebody49 \
  --size x \
  --disable-rest
```

Do not run the Wholebody49 launch commands until both DS9.1 TensorRT 10.16
engines exist. Before then, use the CPU/source/parser and `--plan` gates in
`../DS9_REBUILD_AND_SMOKE_GATES.md`; expected preflight failure is not runtime
evidence.

## DS8-vs-DS9 Config/Artifact Review

This is a review gate, not an equality check. DS9 paths should differ where
expected, but they must not point at DS8-only engines, parser outputs, or label
paths.

```bash
python3 - <<'PY'
from pathlib import Path

required = [
    "DS9/noesis/ds9_runtime.py",
    "DS9/config/infer.yaml",
    "DS9/config/depth_registration.json",
    "DS9/models/coco_labels.txt",
    "DS9/pipelines/config_infer_primary_yolo11_seg.ini",
    "DS9/pipelines/config_infer_primary_yolo11.ini",
    "DS9/pipelines/config_infer_primary_rfdetr_seg.ini",
    "DS9/pipelines/config_infer_primary_rfdetr.template.ini",
    "DS9/pipelines/config_infer_primary_rfdetr_seg.template.ini",
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit("missing required DS9 paths:\n" + "\n".join(missing))

bad_tokens = [
    "../../models/coco_labels.txt",
    "deepstream-8",
    "deepstream-8.0",
    "DS8/models/engines",
]
bad = []
scan_roots = [
    Path("DS9/noesis"),
    Path("DS9/config"),
    Path("DS9/pipelines"),
    Path("DS9/scripts"),
    Path("DS9/csrc"),
]
for root in scan_roots:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml", ".ini", ".txt", ".sh", ".cpp", ".hpp", ".h", ".cmake"}:
            continue
        text = path.read_text(errors="ignore")
        for token in bad_tokens:
            if token in text:
                bad.append(f"{path}: {token}")
if bad:
    raise SystemExit("unexpected DS8/stale references:\n" + "\n".join(bad[:50]))
print("[OK] Required DS9 paths exist and obvious stale references were not found.")
PY
```

For manual diff review:

```bash
diff -u config/infer.yaml DS9/config/infer.yaml || true
diff -u config/depth_registration.json DS9/config/depth_registration.json || true
```

Review differences intentionally; do not force equality.

## Readiness Criteria

Production readiness requires:

- Preflight passes in the DS9 container.
- Runtime starts through `DS9/noesis/ds9_runtime.py`.
- All expected DS9 engines load.
- Pose activation gate passes.
- BEV/track parity smoke passes.
- WebRTC and RTSP mosaic gates pass.
- MapAnything, depth, floorplan, and ReID gates pass.
- No DS8 fallback/shim/degraded path is required.
- `DS9/README.md`, `DS9/docs/migration_state.md`, and
  `DS9/docs/known_blockers.md` are updated with final validation results.

Latest DS9 validation satisfied the startup, engine-load, pose, BEV, WebRTC,
RTSP, MapAnything, floorplan, ReID, config-review, zero-copy, REST depth, ROI
prune, YOLO26-seg, RF-DETR-seg, object-depth bridge, depth tensor bridge, and
ReID native extraction bridge gates listed above, plus RF-DETR detect-only
startup for `s`, and occupied-camera live-RTSP host tracking/BEV/shutdown
evidence. Full DS8 option-surface parity is still blocked by fresh same-session
V3DT behavior evidence and separately by MV3DT overlap/time-sync/fusion proof;
the selected V3DT engines and shared SV3DT calibration are already realized.
