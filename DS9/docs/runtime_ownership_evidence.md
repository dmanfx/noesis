# DS9 runtime ownership evidence

`runtime_ownership.yaml` schema v2 is selector-free normative policy. It records
static implementation/contract truth and repository-source assertions only.
Dynamic `asset_realization` and `runtime_session` selectors live in the
explicit owner-private registry below the runtime root, never in the checkout.

- `repository_source` proves reviewed implementation/configuration exists in
  the current checkout; it does not prove live behavior. Reads are descriptor
  anchored and YAML assertions reject duplicate explicit keys.
- `asset_realization` binds exact manifest profiles, tracked manifest and
  source-contract digests, private realization, selected engine IDs, runtime
  image (`runtime_image_id`), and every selected output digest. The runtime
  image ID is part of the closed selector and artifact binding, not validation-
  only detail.
- `runtime_session` contains that artifact binding plus one exact canonical
  lane/session, complete launcher `SHA256SUMS`, successful container identity,
  lifecycle and removal, canonical endpoints, authenticated runtime instance
  and run IDs, unchanged checkout, and registered behavior documents.

Static status and dynamic acceptance are independent. `parity`, `shared`, and
`adapter_specific` describe implemented contract truth even when required live
evidence is absent; the validator reports that absence as an effective blocker.
`known_gap` and `blocked` are reserved for real static implementation or
contract gaps and can never be overridden by registry evidence. Fixing one
intentionally changes the matrix digest and requires fresh promotions.

The validator owns the exact 27 capability IDs and minimum evidence tiers.
`artifacts.canonical_graph` requires realized proof; V3DT, Wholebody49, ReID,
and MapAnything require fresh runtime sessions. Editing a status cannot clear a
missing-evidence blocker, and `--require-parity` fails on both static and
dynamic blockers.

## Registered behavior

Every behavior report binds an exact filename, closed producer schema,
contract/version, session, lane, supervisor-observed runtime IDs, JSON types,
and required values. Wholebody requires separate S-mask and X-box occupied
sessions. Each Wholebody session also requires direct RTSP plus WebRTC decoded-
frame evidence and the generic 300-second resource soak. V3DT requires its
world, semantic, occupied identity, and that same versioned generic resource
soak in one session. The retired V3DT-only soak contract is rejected rather
than reinterpreted. ReID remains open-set shadow evidence with semantic
accuracy `not_evaluated` and public authority `blocked`. Baseline ReID
promotion requires both that identity evidence and the lane-neutral
`semantic_gate_v3`; identity-only evidence cannot claim multimodal semantic
completeness. MapAnything requires non-empty output for every reviewed active
camera.

Wholebody, floorplan, and V3DT world reports checksum-bind bounded timestamped
source transcripts. They retain only the minimal counters and geometric facts
needed for replay; their closed privacy policy forbids raw embeddings, images,
unredacted payloads, and secrets. The validator independently replays each
transcript, recomputes its report exactly, and requires the observation window
inside Docker's inspected lifetime. Identity and semantic gates use the same
pattern with redacted transcripts and a sealed identity JSONL snapshot. The
identity source contract is v2: it seals the exact gate policy plus projected
before/after health responses, allowing runtime-health and identity claims to
be recomputed rather than trusted from the report. V1 identity sources are
insufficient and rejected; all other behavior source versions are unchanged.
The
semantic source additionally seals its exact wall-clock acquisition interval.
Every received tracking frame, including an empty frame, must be inside the
observed/capture bounds; capture time may
precede its start by at most the named two-second receive/processing-latency
allowance and may never follow its end. Ownership binds the acquisition
interval itself inside the inspected container lifetime. Every canonical
observation also proves `observed_at_us <= published_at_us <=` acquisition end.
Publisher sequence
and tracker lifecycle generation are contiguous within the captured interval,
while their first observed values are explicitly externally unanchored because
an attached live capture may begin mid-run. A missing-predecessor tombstone is
accepted and counted only on a source's first received frame; every later
disappearance tombstone binds its last-seen frame and timestamp to the exact
preceding in-window published presence. Raw reports, transcripts, identity
snapshot rows, and ownership inputs reject duplicate JSON keys and non-finite
numbers before projection or model validation without recording offending keys
or values.
Behavior producers publish through one shared create-once private-file
primitive. Each bounded payload is fsynced into a same-directory mode-`0600`
temporary inode and then hard-linked into its absent canonical name; an
existing regular file, symlink, hard link, or interrupted temporary residue is
never repaired, chmodded, or replaced. Source evidence is published before its
report. Every producer preflights the complete output-name cohort under one
mode-`0700` directory, so any partial cohort requires a new session ID and
remains intact for diagnosis. A crash before publication completes leaves
either temporary residue or a multi-link inode, both of which fail validation.

Every registered report and replay source must match the exact byte encoder
owned by its producer. Equivalent JSON with different whitespace, key order, or
number spelling is not promotion evidence even when hashes and report metadata
are resealed consistently. Semantic identity-snapshot JSONL rows and resource-
soak raw cgroup/GPU samples follow the same rule. Resource acceptance binds the
inspected container, checkout, realization, primary lane engine, pipeline
config, and camera config; requires every sampled GPU owner to belong to that
container; and rejects OOM, leak, cgroup-memory, GPU-memory, PID, cadence, or
binding drift. Current Wholebody liveness floors and resource ceilings are
provisional policy bounds, not empirical performance claims, and are never
relaxed automatically.

The supervisor emits authenticated `runtime-identity.json`; the attached runner
requires that identity and the exact `launch-plan.json`, then writes reports and
source transcripts directly into the flat launcher directory for every lane,
including the baseline semantic report/source/snapshot trio. The supervisor's
final `SHA256SUMS` seals every byte. A generic JSON file, a nearby behavior
directory, a root-relabeled report, or a synthetic selector cannot prove the
session.

The live runner uses those same producer validators for ReID, semantic,
floorplan, V3DT world, Wholebody occupied, and Wholebody decoded media. Each
input is an owner-private, single-link, size-bounded file with its canonical
name; source digest, session/lane/runtime identity, canonical serialization,
and exact replayed report must all agree. Failed, timed-out, or nonzero producer
steps are never followed by report validation. Authenticated capability health,
WebSocket health, `launch-plan.json`, `runtime-identity.json`, and V3DT-bound
JSON are decoded with the same duplicate-key/non-finite rejection before their
existing endpoint, lane, and session checks.

## External promotion registry

With an explicit `--runtime-root`, validation reads only these fixed paths:

- `runtime-ownership/promotions.jsonl`
- `runtime-ownership/promotions.head.json`
- `runtime-ownership/.promotions.lock`

The runtime root and registry directory are owned mode `0700`; registry files
are owned, single-link mode `0600`. Reads and appends use no-follow descriptor
traversal, root/parent/name/inode CAS, and an advisory lock. Every JSONL event is
canonical unique-key JSON and binds matrix ID/digest, capability/evidence key,
complete selector, session/lane/runtime IDs when applicable, checkout digest,
artifact binding, launcher `SHA256SUMS` digest, previous event digest, explicit
per-key supersession digest, and timestamp. File and directory data plus the
persisted head are fsynced before success.

Only active events for the exact current matrix digest participate in
validation. Old-matrix events remain chained for audit but are inactive.
Replacing or revoking a key must name its current head with
`--supersedes-event-digest`; branches, implicit replacement, chain splicing,
registry-only rollback, and head-only rollback fail closed.

Record a realized canonical graph:

```bash
python3 DS9/scripts/promote_runtime_ownership_evidence.py \
  --runtime-root "$NOESIS_DS9_RUNTIME_ROOT" \
  --artifact-root "$NOESIS_DS9_ARTIFACT_ROOT" \
  --capability-id artifacts.canonical_graph \
  --evidence-type asset_realization \
  --subject ds9-canonical
```

Record a validated runtime session:

```bash
python3 DS9/scripts/promote_runtime_ownership_evidence.py \
  --runtime-root "$NOESIS_DS9_RUNTIME_ROOT" \
  --artifact-root "$NOESIS_DS9_ARTIFACT_ROOT" \
  --docker-root "$NOESIS_DS9_DOCKER_ROOT" \
  --capability-id model.reid_profile \
  --evidence-type runtime_session \
  --subject "baseline-$SESSION" \
  --lane baseline \
  --session-id "$SESSION"
```

The recorder constructs selectors from current authorities rather than
accepting user-authored hashes. It independently validates the complete
candidate, then performs terminal matrix, artifact, checkout, and launcher CAS
checks while the registry transaction is locked. Terminal artifact CAS includes
the exact validated runtime image ID, so an image-authority change cannot be
dropped between candidate validation and registry append. It never starts a
runtime, container, build, or GPU job.

Validate all current promotions:

```bash
python3 DS9/scripts/validate_runtime_ownership.py \
  --artifact-root "$NOESIS_DS9_ARTIFACT_ROOT" \
  --runtime-root "$NOESIS_DS9_RUNTIME_ROOT" \
  --docker-root "$NOESIS_DS9_DOCKER_ROOT"
```

External roots are never inferred from environment variables. The artifact
root is owned mode `0700` or `0750`; every runtime/evidence directory is `0700`;
private realization, session, checksum, log, and JSON files are owned,
single-link mode `0600`. Launcher checksums cover every flat launcher file.

## Runtime and transaction invariants

Runtime proof replays the exact Docker environment, mounts, tmpfs,
init/restart/ulimit/log policy, image, GPU, driver, daemon, lane, ports, and
state/auth paths. Security options are exactly
`["no-new-privileges", "label=disable"]`, AppArmor is `docker-default`, the
cgroup namespace is private, OOM kill remains enabled, sysctls are null, memory
swappiness is null/zero, and memory+swap equals the memory ceiling. Extra
devices, binds, namespaces, DNS/host/group/link/volume overrides, or read-only
and masked-path drift fail closed.

Plan/summary/container timestamps are exact RFC3339 UTC. Plan-to-summary slack
is at most 120 seconds; summary-to-container-start and container-finish-to-
summary slack are each at most 10 seconds. Freshness comes from Docker's
inspected finish time. Sessions must satisfy lane duration, remain within 24
hours, and match current checkout/native, GPU, driver, daemon, and image state.

`runtime_session` promotion intentionally accepts only the isolated bounded
`run --duration-seconds ...` canary lifecycle. `appliance-run` has an
indefinite supervisor lifecycle plus selector-bound persistent build, world,
identity, and analytics state, so it is rejected explicitly rather than being
misread as the ephemeral canary contract. Appliance health remains separate
deployment evidence. For promotable V3DT canaries, every launch-plan session
path and the inspected `/var/lib/noesis/build` mount must resolve independently
to `<runtime-root>/build/<session-id>`; launch-plan text cannot redirect replay
to an alternate tree.

Authority and private paths use held descriptors and `openat`/`O_NOFOLLOW`.
Immediately before success, validation holds the artifact transaction lock,
rereads the launcher cohort, revalidates artifacts, recomputes checkout, then
repeats launcher and checkout CAS after terminal artifact validation. In-flight
changes in any cohort fail closed.

The registry is tamper-evident inside the appliance's same-UID private-file
trust boundary, not cryptographic remote attestation. A hostile same-UID process
can rewrite both JSONL and persisted head together; preventing that requires a
future privileged or remote monotonic anchor. Rewriting only one, truncating,
splicing, or conflicting through the supported command is detected.

Never promote a live capability from repository-source evidence. Record the
appropriate realized/session proof and keep failed or stale selectors visible.
