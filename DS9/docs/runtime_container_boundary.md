# Canonical DS9 Runtime Container Boundary

Status: implemented, adversarially tested, and live-baseline accepted on
2026-07-11. Session `baseline-runtime-image-canary2-20260711t0422z` ran the
canonical baseline for 30 seconds, proved GPU ownership and WebRTC RTP flow,
then exited `0` through acknowledged EOS with no forced removal or error-level
log signatures.

## Decision

`DS9/scripts/run_canonical_runtime_container.py` is the only supported
secondary-Docker entrypoint for the canonical DS9 live canary. A hand-written
`docker run` command is not equivalent: the supervisor owns provenance,
exclusive-runtime checks, immutable mounts, lifecycle acceptance, cleanup, and
evidence sealing as one fail-closed transaction.

The readiness profiles are intentionally narrow:

| Runtime lane | Required PGIE | Required size | Readiness meaning |
| --- | --- | --- | --- |
| Baseline | YOLO26 detect | `m` | Canonical DS8/DS9 baseline parity |
| V3DT | YOLO26 segmentation | `s` | Separate V3DT parity lane |
| Wholebody49 S | DINOv3-S Wholebody49 | `s` | Occupied-scene mask/parser lane |
| Wholebody49 X | DINOv3-X Wholebody49 | `x` | Occupied-scene bbox-only lane |
| YOLO11 segmentation | Explicit alternate only | N/A | Never satisfies canonical readiness |

## Separate Build And Runtime Image Authorities

TensorRT build provenance and runtime service dependencies have different
change cadences and therefore use different immutable image identities:

- `noesis-ds9-dev:9.0-20260710` at
  `sha256:7476b1021376cd67793c95d949cdc7d46eef7704ab98a5a76feed461e4f907a4`
  is the engine-build authority. Existing engine maintenance evidence remains
  bound to this image and is never rewritten merely because a runtime package
  changes.
- `noesis-ds9-runtime:9.0-20260710` at
  `sha256:ca33b4c6a84fc56b86b71feee2a444299cb2ce7f33ab018cae43ac730aaef5fc`
  is the live-runtime authority. `DS9/docker/Dockerfile.runtime` starts from
  the exact build-image ID, then adds the ABI-aligned Ubuntu PyGObject and
  GstWebRTC introspection packages needed by the canonical gateway.

The tracked manifest records the build authority under `target.build_image`
and the runtime authority under `runtime.image`. The external realization
continues to prove which image actually built every engine. Adopting a new
runtime image may rebase only the realization's tracked-manifest anchor after
the rebase tool proves the build target, source contracts, and every engine
record are unchanged. Relabeling engine evidence as though the runtime image
built it is forbidden.

The runtime Dockerfile verifies all four GI namespaces and every gateway
factory at build time. DS9 preflight repeats that check inside the launched
environment. Before any plan or run, the supervisor validates both exact image
IDs, the official DeepStream base digest, parent labels, repository Dockerfile
hashes, and the actual RootFS relationship: all 70 parent layers must be the
exact prefix of the runtime's 72 layers. The live command uses the immutable
runtime image ID, not its tag.

The supervisor exposes those four reviewed lanes through `--lane`; it accepts
no arbitrary pipeline, camera config, detector profile, size, or tracking
combination. `baseline` hard-pins `baseline/yolo26/m`, `v3dt` hard-pins
`v3dt/yolo26_seg/s` plus the DS9 V3DT pipeline and camera inventory, and the two
Wholebody49 lanes hard-pin the shared baseline graph plus `wholebody49/s` or
`wholebody49/x`. Unknown or misspelled lanes fail before Docker launch.

## Host Inputs

Choose three disjoint machine-local roots outside the checkout. Do not encode
their values in repository configuration:

```bash
export NOESIS_DS9_DOCKER_ROOT=<absolute-secondary-docker-root>
export NOESIS_DS9_ARTIFACT_ROOT=<absolute-ds9-artifact-root>
export NOESIS_DS9_RUNTIME_ROOT=<absolute-private-runtime-root>
```

The existing owner-only runtime files are selected by these variables or their
standard appliance paths:

```bash
export NOESIS_CAMERA_SECRETS_FILE=<absolute-camera-source-json>
export NOESIS_MAPANYTHING_API_KEY_FILE=<absolute-mapanything-key-file>
export NOESIS_INTERNAL_AUTH_TOKEN_FILE=<absolute-internal-bearer-file>
```

The supervisor validates each file as a single-link regular file owned by the
calling user, validates its content without printing it, and rejects a file
inside any broader checkout, artifact, runtime, or Docker mount. Only those
three files are mounted read-only beneath `/run/noesis-secrets`, whose parent
is a uid-owned mode-`0700` tmpfs. Secret values never enter arguments,
environment values, image layers, or evidence.

The artifact root must also contain the owner-only mode-`0600` realization
overlay `asset_realization.json`. The tracked `DS9/asset_manifest.yaml` remains
an immutable declaration and is never rewritten with machine-local build
results. The realization may change only `state` and complete `provenance` for
an existing base `tensorrt_engine`; artifact identity, output path, sources,
and compatibility remain base-owned.

The realization pins both tracked trust anchors:

```text
base_manifest.path = DS9/asset_manifest.yaml
source_contracts.path = DS9/config/engine_source_contracts.json
```

Each record carries the exact current SHA-256. A missing realization, stale
digest, unknown artifact, non-engine overlay, immutable-field override,
incomplete lane-specific engine set, unsafe mode/link, or incomplete provenance
is a hard plan blocker. This keeps a locally realized engine graph auditable
without making the source checkout a mutable build database.

## Write-Free Plan

Run the plan after the secondary daemon and staged artifact root exist:

```bash
python3 DS9/scripts/run_canonical_runtime_container.py plan
python3 DS9/scripts/run_canonical_runtime_container.py plan --lane v3dt
python3 DS9/scripts/run_canonical_runtime_container.py plan --lane wholebody49-s
python3 DS9/scripts/run_canonical_runtime_container.py plan --lane wholebody49-x
```

Plan mode does not create directories or containers and never requests a GPU,
host network, or host IPC. It verifies and reports:

- the exact runtime and engine-build image IDs, official base digest, both
  Dockerfile digests, Python lock digest, and exact parent/runtime RootFS
  prefix plus two-layer delta;
- the secondary daemon's exact data root, `runc` default, NVIDIA runtime
  availability, and exact `host` plus `none` network inventory;
- exact lane artifact bytes and provenance through the required external
  realization overlay and the corresponding authoritative manifest profiles;
- an exact compatibility match between GPU device `0` and every selected
  engine's private build evidence: NVIDIA driver version, physical GPU UUID,
  model name, compute capability, and total memory must all agree;
- the selected lane's hard-pinned pipeline, cameras, PGIE, size, tracking,
  batch/GIE identity, RTSP/WebRTC contract, and fixed ports `6008`, `8080`, and
  `8554`;
- the exact source-ID set derived from that lane's pipeline and camera
  inventory, which execution requires the persistent exclusion stage to cover
  exactly (disabled exclusion entries remain required coverage);
- absence of another DS9 runtime container/process, GPU compute owner, or
  canonical-port listener;
- availability of the shared DS9 artifact transaction lock; and
- a content digest over every tracked and non-ignored untracked checkout file.

Exit `0` means the report is ready for an explicitly authorized run. Exit `3`
means the plan completed but has runtime blockers. Exit `1` means the contract
itself is invalid. A plan is evidence about current state, not authorization to
launch.

## Explicit Live Canary

Only after the plan is clean and the GPU maintenance owner has released the
device may an operator run:

```bash
python3 DS9/scripts/run_canonical_runtime_container.py run \
  --authorize-gpu-runtime \
  --duration-seconds 30

python3 DS9/scripts/run_canonical_runtime_container.py run \
  --lane v3dt \
  --authorize-gpu-runtime \
  --duration-seconds 30
```

The short V3DT canary is lifecycle evidence only. V3DT and both reviewed
Wholebody49 ownership lanes add `--resource-soak` and run for at least 300
seconds while the same-session behavior runner is attached; the validation
runbook may use a longer duration so serial occupied-scene gates have enough
time to finish. The generic v2 soak binds raw cgroup/GPU samples to the exact
container, checkout, realization, primary lane engine, and configs. The old
V3DT-only evidence contract is rejected rather than reinterpreted.

`--duration-seconds 0` runs until the supervisor receives SIGINT or SIGTERM.
The authorization flag is mandatory; there is no environment-variable bypass.
Only this mode adds NVIDIA device `0`, `--network=host`, and `--ipc=host`.
Those namespaces are required for the intentional local camera/media surface;
the WebSocket and REST control listeners remain fixed to loopback and require
the internal bearer.

The container runs as the calling non-root uid/gid with a read-only root
filesystem, all capabilities dropped, the exact ordered security options
`no-new-privileges` plus `label=disable`, bounded pids and logs, an exact 26 GiB
memory plus memory-and-swap ceiling with zero container swap allowance, and the
immutable image ID rather than a mutable tag. Inspection also requires no
legacy `Devices` or `Binds`, empty PID and user-namespace modes, and the exact
Docker read-only and masked path policies accepted on this appliance. The
appliance's sensitive Docker defaults are part of the identity: AppArmor must
be `docker-default`, `CgroupnsMode` must be `private`, `OomKillDisable` and
`Dns` must be present as null, while `Sysctls` and `StorageOpt` must be absent.
An omitted `Dns` or an explicitly present `Sysctls`/`StorageOpt` is a different,
rejected representation even when its decoded value would otherwise be null.
Port bindings, links, extra hosts/groups, auto-remove, UTS mode, volume
inheritance, and publish-all-ports must match the reviewed inspect values
exactly. `seccomp=unconfined` is never accepted. Docker on
this swapless cgroup-v2 host reports `MemorySwappiness=null` even when given a
zero request, so the launcher omits that unsupported flag and accepts only
inspection values `null` or `0`; the enforceable contract is exact
`MemorySwap == Memory`. On the
current 31 GiB swapless host, that leaves roughly 5 GiB outside the container
for the supervisor, Docker, and host services; it is a containment ceiling,
not a promise that DS9 may safely consume all 26 GiB. The checkout and
external DS9 artifact root are read-only. Four distinct per-session host paths
and one narrowly scoped appliance-persistent analytics path are mounted
writable:

| Host path under `NOESIS_DS9_RUNTIME_ROOT` | Container path | Purpose |
| --- | --- | --- |
| `build/<session>` | `/var/lib/noesis/build` | Effective configs and caches |
| `state/<session>` | `/var/lib/noesis/state` | Journals, identity, aliases, scenes |
| `depth/<session>` | `/var/lib/noesis/depth` | MapAnything snapshots |
| `evidence/<session>/runtime` | `/var/lib/noesis/evidence` | Runtime diagnostics and identity evidence |
| `persistent/analytics` | `/var/lib/noesis/state/analytics` | Durable analytics YAML plus its derived exclusion INI |

The analytics bind is added after the parent session-state bind, so only that
nested directory persists across sessions. It is seeded once from the reviewed
analytics YAML, then preserved; every launch rejects a partial pair, derived
INI mismatch, symlink, non-regular file, unexpected entry, unsafe owner/mode,
source-coverage drift, YAML over 4 MiB, or exclusion INI over 1 MiB. All other
state remains deliberately per-session for canary isolation. This launcher
does not silently import or mutate a production household identity, journal,
alias, scene, depth, or virtual-twin store. Promoting any of those stores still
requires its own reviewed migration and is not a reason to broaden the
persistent mount.

### Selector-driven appliance mode

The `plan`/`run` table above remains the isolated engineering-canary boundary.
`appliance-check` and `appliance-run` are a stricter deployment adapter. They
accept only `--deployment-selector` and `--selector-sha256`; the selector fixes
the lane, images, roots, realization, ownership matrix, endpoints, checkout
snapshot, and state release. There is no DS8 retry or alternate DS9 lane.

For `appliance-run`, the selected release replaces every state-bearing canary
mount that would otherwise create split product truth:

| Selected host state | Container path |
| --- | --- |
| `runtime/noesis-build` | `/var/lib/noesis/build` |
| `payload/world.db` | `/var/lib/noesis/state/world_ds9.sqlite3` |
| `payload/identity.db` | `/var/lib/noesis/state/household/identity_v2.sqlite3` |
| `payload/analytics` | `/var/lib/noesis/state/analytics` |

The analytics directory is validated in place and is never seeded, copied, or
replaced from checkout content in appliance mode. The world and identity files
are individual writable bind mounts. All four payload files must be real,
single-link, owner-owned mode-`0600` files beneath an owner-only release; the
build root must already be a real owner-only mode-`0700` directory. Session
state remains mounted only for non-selected ancillary state and mount parents.

Menon owns the one full-lifetime shared lease for this state release and wraps
the Noesis command with it. The Noesis supervisor deliberately accepts no lease
path, lease environment variable, or nested lock. It verifies the full
`noesis-runtime-v1` checkout/runtime identity at admission and again
immediately before Docker exec. The pre-exec gate also reopens the canonical
selector, manifest, and baseline and requires the originally admitted inode for
every mutable payload, parent directory, release root, and build directory.
The exact selected supervisor is the owner-owned, single-link executable at
`DS9/scripts/run_canonical_runtime_container.py`; alternate checkout scripts
and inherited nested-lease variables fail closed. Deployment health exposes
only the retained selector digest. Mutable release payloads may advance while
the outer lease is held; the immutable activation baseline remains the
activation evidence and is not misreported as a per-health rehash.

The supervisor and engine-maintenance runner share
`${NOESIS_DS9_ARTIFACT_ROOT}/.noesis-ds9-artifact-transaction.lock`, an
owner-only mode-`0600` advisory lock. An authorized runtime acquires it before
the immediate realization/file/provenance revalidation. While holding the
lock, it queries GPU `0` again and requires the identity and driver to equal
both the plan and every engine record selected by that lane. It then holds the lock
through container launch, inspection, port readiness, and confirmation that
the only GPU compute owner descends from the container init process. Engine
maintenance holds the same exclusive lock through build and realization
reconciliation. Contention or hardware/driver drift is a hard blocker; neither
side observes or launches against a partially replaced or device-incompatible
engine graph.

## Acceptance And Cleanup

After Docker returns a container ID, the supervisor validates the actual
container inspection rather than trusting its requested command. It requires
the exact image, user, lane label, command, profile, environment, bind sources,
read/write flags, tmpfs destinations, security settings, memory ceiling,
namespace modes, and GPU device request. It then waits for all three canonical
ports, performs an authenticated read of `/api/v1/health/capabilities`, and
records that runtime's exact `instance_id` and `run_id` in
`runtime-identity.json`. It then proves that the resulting GPU process belongs
to the inspected container before releasing the artifact transaction lock.
Any attached behavior runner must use the exact loopback WebSocket, REST, and
RTSP URLs declared by the launch plan and must observe those same runtime IDs.

Shutdown is accepted only when all of the following hold:

1. the supervisor delivered SIGTERM;
2. Docker reported exit code `0` without a forced removal;
3. the log contains orderly-EOS acceptance, the
   `reason=shutdown_requested` EOS callback, Service Maker `wait()` return, and
   `Shutdown complete` in that order;
4. no error, critical, native-crash, wait-timeout, or GStreamer teardown
   signature is present;
5. the stopped container was removed and all canonical ports closed; and
6. the checkout content digest is unchanged; and
7. the appliance analytics pair still satisfies the exact private-file,
   source-coverage, size, and YAML-to-INI derivation contract.

Checkout bytes are hashed from no-follow file descriptors while the named
entry, every opened parent, and repository root are identity-checked. The
ownership validator also repeats the complete artifact-realization binding and
current checkout snapshot immediately before returning success, so a rename
swap or late mutation cannot survive on an earlier read.

If startup or shutdown fails, forced removal is containment only and the gate
remains failed. It is never reported as an accepted fallback.

Launcher-owned evidence is kept outside the container-writable evidence mount
at `evidence/<session>/launcher/`. It contains the launch plan, pre-run checkout
summary, immutable session-local copies and SHA-256 metadata for the analytics
pair both before launch and after shutdown, authenticated runtime identity,
runtime log, container inspection, final summary, and `SHA256SUMS`, all under
owner-only directories/files. The evidence directory survives container
removal; the next session receives a new evidence directory while reusing only
the validated persistent analytics pair.

An attached behavior runner may add only the canonical, session/lane-bound
reports named by the ownership registry directly to this launcher directory.
Its logs and summary live in the sibling session `behavior/` directory so the
launcher directory remains flat and completely checksum-covered. The
supervisor's final `SHA256SUMS` covers the canonical report files alongside its
own launch, inspection, shutdown, and summary evidence; a report in an
unrelated behavior directory cannot prove runtime ownership. V3DT additionally
stores checksum-bound raw cgroup/GPU samples. Wholebody, floorplan, and V3DT
world gates store bounded timestamped minimal-field source transcripts;
identity and semantic gates store redacted transcripts, and the semantic gate
seals the exact identity JSONL snapshot it consumed. Its source also seals the
exact acquisition start/end: observed samples remain inside that interval,
capture timestamps receive only the fixed two-second pre-window latency
allowance, and the whole interval must remain inside the inspected container
lifetime. Bounds apply to every tracking frame, including empty frames, while
canonical publication timestamps must fall between their observation time and
acquisition end. These paths retain no images, raw embeddings, full unredacted
payloads, or secrets. Ownership validation replays each source and recomputes
its report. Semantic publication/lifecycle continuity is deliberately scoped
to that partial acquisition window with an externally unanchored first value;
subsequent values are contiguous. Only a source's first received frame may
carry a counted tombstone whose predecessor predates attachment; all later
tombstones bind exact in-window published presence. Attached baseline and V3DT
runs use the same lane-neutral `semantic_gate_v3` authority and write its
report, source, and identity snapshot into this checksum-covered launcher
directory. Duplicate-key or non-finite JSON is rejected before replay.

Validated realized/session selectors are promoted only through
`promote_runtime_ownership_evidence.py` into the fixed owner-private
`runtime-ownership/` registry below the runtime root. The tracked ownership YAML
remains selector-free static policy. Recording is offline and does not start a
container or touch the GPU.

## Validation Without Live Use

The adversarial suite uses a fake Docker executable and never exposes a GPU:

```bash
python3 -m pytest -q DS9/tests/test_runtime_container_boundary.py
python3 -m ruff check \
  DS9/scripts/run_canonical_runtime_container.py \
  DS9/tests/test_runtime_container_boundary.py
```

It covers daemon/image/network drift, existing runtime ownership, secret-byte
non-disclosure, read-only mount enforcement, profile/GPU/memory drift, artifact
lock contention and launch-span ownership, exact engine/build-host GPU and
driver compatibility, TERM/exit-zero removal, failed-shutdown containment,
private evidence, checkout mutation, fail-closed lane selection, V3DT
camera/config/profile isolation, and the two exact Wholebody49 materialization
lanes. Analytics boundary cases additionally cover cross-session preservation,
partial/mismatched/linked/non-regular/unexpected state, exact source coverage,
nested mount inspection, and restart acceptance of an API-valid YAML payload
larger than the former 1 MiB limit while keeping the derived INI within 1 MiB.

V3DT behavior acceptance pairs the same-lane supervisor run with the
authenticated `sv3dt_meta_smoke_test.py` and
`v3dt_world_contract_smoke_test.py`. The latter accepts only a privacy-safe v2
source/report pair bound to the exact same sealed launcher session and effective
config cohort. It requires `world_frame=backend_world_m`,
`world_source=bbox3d`, the locked `xzy` conversion, all-camera continuity,
native image-foot reprojection, and independent image-base replay; v1 and old
camera-local/raw-tuple evidence are rejected. This per-camera SV3DT contract
does not claim MV3DT overlap, synchronization, peer association, or fused
positions. Wholebody49 behavior acceptance pairs the
same-lane RTSP decode with `wholebody49_occupied_scene_smoke_test.py`: the S lane
must consume instance-mask depth samples; the X lane must consume bbox samples
and must not activate the mask path. Subjective overlay aesthetics remain a
separate review of the captured mosaic rather than a synthetic pass condition.
