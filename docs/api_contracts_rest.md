# DS9.1 REST API contracts
_Status: canonical native-host contract, updated 2026-08-15._

This document describes the REST endpoints used by the DS9.1 runtime. Implementations must preserve these contracts unless all consumers are updated in lockstep.

## Wire-render and boundary-metric contract

All 41 successful FastAPI-rendered product JSON routes shared by DS9.1 baseline and the disabled V3DT adapter keep
FastAPI's declared `response_model`, status, headers, aliases, and exclusion
settings as the wire authority. This includes depth, analytics, ReID v1/v2,
capability/deployment health, scene metadata, and virtual-twin metadata.
Handlers mark only their response-model construction time on the current
request. `BoundaryMetricsRoute` then observes the completed FastAPI response
body after the framework's one real validation/render path. It records the
exact returned byte length and never performs a second `model_dump` or JSON
render for measurement.

REST boundary samples use a monotonic clock and true rolling 10-second and
60-second windows. The metrics expose pooled p99 and max-per-path p99 for both
windows. Admission uses the worst budgeted `stage=total` path/window, because a
low-volume slow endpoint can disappear inside a pooled percentile. Provider
work before response assembly is excluded; endpoint model assembly, sync-worker
handoff, FastAPI response-model filtering/aliasing, the actual JSON render, and
response creation are included in the 3 ms boundary total. Metric detail is
bounded and additive; none of these fields changes a REST response body.
Aggregate/path sample buffers cap at 4096 and fail the affected path/window
closed while saturation evidence is live. Tagged
`boundary_serialization_errors_total` counts only response-model, validation,
render, and local-response failures—not provider or network failures. Runtime
stats use the compact getter with at most eight ranked budget-path diagnostics;
the full getter remains available for explicit diagnostics.

A successful route using `BoundaryMetricsRoute` fails closed if it reaches the
wire without either a measurement mark or an explicit exemption. The only
current exemptions are four integrity-checked, already rendered scene artifact
responses, the virtual-twin `FileResponse`, and the lease-backed dense-depth
component stream; those six pre-rendered/file/stream routes do not claim
framework JSON CPU-budget evidence. Provider failures and FastAPI error
responses are not successful product payloads and do not fabricate samples.

## Authentication boundary

The runtime REST application is internal to the native host. Every request
requires `Authorization: Bearer <internal gateway token>`, loaded from the
owner-only `NOESIS_INTERNAL_AUTH_TOKEN_FILE`. Menon terminates browser sessions,
CSRF/origin policy, roles, confirmations, and audit, then calls Noesis with this
internal credential. Wildcard and regex CORS are forbidden; exact origins are
available only for explicit loopback development.

`NOESIS_INTERNAL_AUTH_MODE=disabled` is explicit development configuration and
is executable only on `localhost` or a literal loopback address. REST and
WebSocket startup reject wildcard, LAN, or arbitrary hostname binds in disabled
mode. Token absence or corruption fails startup; it never selects an
unauthenticated path.

## Capability health

`GET /api/v1/health/capabilities` returns strict
`noesis.capability.health` v1. A capability becomes healthy only after
monotonic, contract-compatible producer progress. It ages to degraded and then
failed when publication stops; an open listener alone never counts as health.
The canonical runtime reports `tracking_observations` and `global_world` from
actual observation/world publication cycles.

## Native deployment health

`GET /api/v1/health/deployment` is the native runtime identity/readiness
surface. It retains the strict `noesis.appliance.deployment_health` v1 schema
for consumer compatibility and returns it
only after the runtime has bound one immutable producer identity and both
`tracking_observations` and `global_world` are currently healthy:

```json
{
  "contract": "noesis.appliance.deployment_health",
  "contract_version": 1,
  "deployment_id": "<deployment-id>",
  "selector_sha256": "<64 lowercase hex characters>",
  "state_release_id": "<state-release-id>",
  "runtime_family": "ds9",
  "runtime_variant": "<family-prefixed exact variant>",
  "instance_id": "<producer instance>",
  "run_id": "<producer run>",
  "boot_id": "<kernel boot identity>",
  "software_revision": "<exact Git revision>",
  "generated_at_us": 1,
  "ready": true
}
```

The route returns `503` when native health identity is missing, the producer is
not bound, the capability monitor is unavailable, either required capability
has not advanced into healthy state, or producer identities disagree. It never
turns listener liveness into readiness.

`selector_sha256` is a legacy field name retained on the wire. The native
supervisor supplies a fixed health-authority digest through
`NOESIS_HEALTH_SELECTOR_SHA256`; it does not read a deployment-selector file.
`state_release_id` names the installed state baseline. The response deliberately
does not rehash the checkout or state on every health read.

## Scene releases

The `/api/v1/scenes` surface owns immutable coherent reconstruction releases:

- `POST /api/v1/scenes/releases` registers strict `noesis.scene.release` v1.
- `GET /api/v1/scenes/releases` and `GET /api/v1/scenes/releases/{id}` inspect candidates.
- `GET /api/v1/scenes/current` returns the one promoted release.
- `GET /api/v1/scenes/current/payload` returns that release plus exact camera manifests, camera artifact URLs, and an `authored_scene_dependency_urls` map keyed by dependency role.
- `GET /api/v1/scenes/current/authored-scene` serves the exact release-owned authored home bytes.
- `GET /api/v1/scenes/current/authored-dependencies/{role}` serves one exact release-owned OBJ dependency such as its MTL or texture.
- `GET /api/v1/scenes/current/validation-report` serves the exact release validation evidence.
- `GET /api/v1/scenes/current/cameras/{camera_id}/artifacts/{role}` serves only an artifact selected by the current release.
- `POST /api/v1/scenes/releases/{id}/promote` atomically promotes with `actor_id` and `expected_current_release_id`.
- `POST /api/v1/scenes/releases/{id}/rollback` performs an explicit compare-and-swap rollback.
- `GET /api/v1/scenes/history` returns append-only promotion history.

Registration never implies promotion. Every camera revision includes a closed
artifact inventory with relative path, byte length, and SHA-256. Promotion and
the `/current` metadata and `/current/payload` reads revalidate safe paths,
exact manifest and artifact bytes, camera/revision identity, the release-owned
authored scene, every declared MTL/texture dependency, and the validation
report. Individual binary routes load the current integrity-checked release row
and verify only the selected file's exact length and SHA-256, avoiding an
O(N-squared) full-cohort rehash while still failing closed on mutation. They
return the already verified byte snapshot, not a path that is reopened later,
with `ETag: "sha256-<digest>"` and `X-Noesis-Scene-Release` headers.

All reads reject symlinks, hardlinks, non-regular files, unsafe path
components, replacement during a read, and linked or extra content in a
release-owned bundle. Existing state is never chmod-repaired. Scene v1 allows
at most 16 cameras, 128 artifacts per camera, 64 authored dependencies, 512
declared files, and 1 GiB of declared content. Files are non-empty; authored
scenes are capped at 64 MiB, manifests and validation evidence at 5 MiB each,
and other artifacts at 256 MiB each. All authored bundle paths must be unique
and owned by `releases/<release_id>/`. Payload artifact URLs stay under the
current-release namespace; they do not point back to the unrestricted revision
catalog. Menon renders `current/payload`; legacy virtual-twin `latest` is an
operator/diagnostic catalog, not coherent multi-camera product truth. The
native DS9.1 baseline and disabled V3DT adapter use this same shared router and
store implementation.

### Whole-home PCF review assembly

The Room Walk service may expose one explicitly selected, review-only multi-room
PCF sidecar for the current authored scene:

- `GET /api/v1/scenes/current/review-assemblies/whole-home` returns the strict
  descriptor, artifact URL, byte length, SHA-256, registration status and
  uncertainty, source-room counts, and its exact scene binding.
- `GET /api/v1/scenes/current/review-assemblies/whole-home/artifacts/multiroom_points_glb`
  serves the descriptor-selected point-cloud GLB only after revalidating its
  safe regular path, exact byte length, SHA-256, and current scene-release
  binding. Contract version 3 uses the `multiroom_points_glb` role.
- The same descriptor-selected artifact route may serve
  `.../artifacts/multiroom_surface_mesh_glb` for contract version 4. The v4
  surface mesh is a bounded, derived review presentation (maximum 64 MiB),
  never canonical scene geometry; it carries its source-point digest and the
  same scene, calibration, runtime, registration, and camera-marker binding.

The binding includes the current scene release, authored-model digest,
calibration digest, runtime configuration digest, runtime world-alignment
digest, and the exact `world_to_scene_col_major` similarity used by Menon. It
also includes a strict `camera_anchor` block. The anchor records the calibrated
Family Room camera pose used by the authored scene and the admitted Scene Prior
reference-camera pose inside the final PCF assembly. Contract version 3 uses a
floor-locked planar anchor: camera X/Z and heading may map to the authored
device reference, while Y remains unchanged. The prior's 1.63 m reference
camera height, metric scale, and gravity remain authoritative. Sparse
static-to-phone depth-backed PnP is retained only as an uncertainty diagnostic;
its translation cannot move the anchor, and the static image contributes no
geometry to the phone reconstruction.

The descriptor additionally carries `camera_markers`: one explicit
assembly-frame position for each Family Room, Kitchen, and Living Room static
camera. The Family marker is the solved anchor pose. Kitchen and Living Room
markers are their Scene Prior reference-camera positions composed through the
same recorded room-registration transforms that produced the joined geometry.
Menon renders them as yellow review spheres inside the same transform group as
the PCF points. They are diagnostics, not alignment inputs or geometry.

Menon must compare the release binding with its already loaded scene and live
Noesis runtime configuration. It then derives one proper planar correction
that maps the admitted Family camera X/Z and heading to the calibrated device
reference while preserving assembly Y, and composes it after
`world_to_scene`. The one resulting transform applies to the complete
multi-room artifact. Bounding-box or corner anchoring, vertical camera-forced
translation, whole-cloud
ICP, reflections, per-room transforms, and visual nudges are not part of the
contract. A mismatch, mutation, missing field, improper camera-pose matrix,
incomplete or displaced camera-marker set, rejected registration represented
as canonical, or unavailable current release fails closed. The endpoint never
promotes the PCF, changes tracking/world
authority, or makes a rejected join canonical. Menon exposes this route only to
the single-owner review capability and presents it as a toggleable sidecar over
the unchanged authored model.

## 1. Depth API

**Module:** `noesis/server/depth_api.py`

### Manual refresh endpoint

- `POST /api/v1/depth/refresh?seconds=<int>`

This endpoint changes GPU/runtime state and is intentionally not available via
GET. Product calls must pass through Menon's authenticated action boundary;
loopback engineering calls remain POST-only.

### Query Parameters

- `seconds` (int, 1–300):
  - Requested duration in seconds for which depth should be enabled.

### Response (200 OK)

```json
{
  "started_at": <int epoch-seconds>,
  "will_disable_at": <int epoch-seconds>,
  "enabled": <bool>,
  "seconds": <int>
}
```

This matches the `DepthRefreshResponse` Pydantic model.

### Error Responses

- `503 Service Unavailable`
  - Depth pipeline not ready (e.g., `infer.yaml` missing or DS9.1 pipeline failed to initialize).
- `500 Internal Server Error`
  - Depth control failed or returned a malformed payload.

Environment/config notes:
- Pipeline config path: `NOESIS_DS9_PIPELINE_CONFIG` (default
  `DS9/config/infer.yaml`).
- Explicit stub mode (tests only): set `NOESIS_DEPTH_API_FORCE_STUB=1` to bypass DS9.1 bindings while preserving the contract. Import failures never select the stub implicitly.

Implementation note: the underlying `enable_depth(seconds)` function in the
DS9-owned pipeline controls the one MapAnything valve; no alternate legacy path
may answer the request.

### Exact dense snapshot component stream

`ma_depth_response.payload.components[*].url` names the only dense-depth bulk
read route:

```text
GET /api/v1/depth/snapshots/{camera_id}/{snapshot_id}/components/{component}
    ?snapshot_ref={portable-store-relative-reference}
    &content_sha256={64-lowercase-hex-exact-snapshot-digest}
```

`component` is exactly `depth`, `conf`, `mask`, or `rgb`; unknown names and
path-like alternates are rejected. Both query parameters are required exactly
once. The path camera/write ID, portable snapshot reference, and snapshot
content digest must resolve to one immutable committed snapshot. There is no
path lookup, `latest` lookup, redirect, range response, legacy-inline response,
or substitute snapshot on mismatch.
Only the public `capture_event_fused` / `intra_capture` role is admitted. Raw
capture commits are storage-internal and return `404 bulk_snapshot_role_invalid`
even if all supplied identity fields are otherwise exact.

The component response is an uncompressed complete row-major byte stream with:

- `Content-Type: application/octet-stream`
- exact `Content-Length`
- `Cache-Control: no-store`
- `X-Noesis-Component-Sha256: <descriptor component digest>`
- `X-Noesis-Snapshot-Id: <descriptor snapshot_id>`

The payload dtype and shape come only from the compact descriptor: `depth` and
`conf` are little-endian float32 (`<f4`) at `[H,W]`; `mask` is uint8 (`|u1`) at
`[H,W]`; optional `rgb` is uint8 at `[H,W,3]`. A client must validate the
descriptor, same-origin URL identity, response status and headers, exact byte
length, and the body SHA-256 before constructing typed arrays.

Every direct Noesis request still requires the internal bearer. In the product
path, an operator/owner browser makes the generated relative request to Menon's
same-origin gateway. Menon validates the exact path/query allowlist, strips
browser cookies, range/conditional headers, and other ambient credentials,
adds the owner-only Noesis bearer server-side, validates the upstream stream
headers, and relays with backpressure. The browser never sees that bearer.

Descriptor admission and component streaming own separate bounded leases.
Producing the compact descriptor opens an identity-keyed cohort grace lease so
retention cannot remove the snapshot between the WebSocket response and the
first HTTP GET. The default grace is 30 seconds idle with a non-renewable
180-second absolute lifetime; repeated descriptor reads or component activity
may move only the idle deadline and never the absolute deadline. A
manager-owned monotonic reaper expires that lease even when optional retention
pruning is disabled and no later request arrives.

Each component GET then owns a second stream lease from exact identity
validation through its last byte. Completion, stream failure, or downstream
cancellation closes that stream and immediately releases the stream lease. It
does not pretend that the separate descriptor grace has ended; that bounded
lease remains available for the rest of the cohort and is released by its idle
or absolute deadline. Thus retention cannot prune a component mid-transfer,
while an abandoned descriptor or browser request cannot pin storage
indefinitely. The shared resource ceiling is 8,388,608 pixels, 64 MiB per
component, and 128 MiB per descriptor cohort (enough for an ordinary 3840x2160
depth/confidence/mask/RGB snapshot).

New commits write manifest payload version 2 with exact raw-component shape,
dtype, byte-count, and digest records. Version 1 commits remain valid for the
existing read-only snapshot loader, but are intentionally non-bulk: descriptor
or component access returns `409 bulk_component_manifest_missing`. Other exact
bulk failures map to `404` for missing/mismatched snapshot identity, `409` for
manifest/integrity failure, `413` for resource-limit failure, and `500` for an
unexpected stream-open failure.

This route is a declared streaming exemption from the 3 ms framework JSON
serialization budget. The small `ma_depth_response` descriptor remains subject
to the WebSocket assembled-JSON SLO; component transfer latency, digest work,
and client-side decoding/normals are reported as separate bulk diagnostics.

## 2. Analytics ROI API

**Module:** `noesis/server/analytics_api.py`

The analytics API exposes the writable `exclude` stage shared by the DS9.1
baseline and disabled V3DT adapter. `NOESIS_ANALYTICS_CONFIG` identifies the durable YAML
source of truth and `NOESIS_ANALYTICS_EXCLUDE_CONFIG` identifies its derived
native exclusion INI. The repo-owned pre-tracker `nvdsroiexclude` element is the
only canonical object-pruning path. There is no post-tracker Python pruning
hook or degraded fallback.

### Common Data Structures (Pydantic Models)

- `ROI`:

```json
{
  "id": "<string>",
  "description": "<string|null>",
  "points_px": [[<float x>, <float y>], ...]
}
```

- `ROIStreamState`:

```json
{
  "stream_id": "<string>",
  "label": "<string|null>",
  "enable": <bool>,
  "rois": [ROI, ...]
}
```

- `ROIListResponse`:

```json
{
  "stage": "<stage-name>",
  "config_width": <int|null>,
  "config_height": <int|null>,
  "defaults": { ... },
  "streams": [ROIStreamState, ...],
  "config_path": "<string|null>"
}
```

- `ROIUpdateRequest`:

```json
{
  "stage": "<stage-name>",
  "streams": [
    {
      "stream_id": "<string>",
      "label": "<string|null>",
      "enable": <bool|null>,
      "rois": [ROI, ...]
    },
    ...
  ]
}
```

- `ROIUpdateResponse` extends `ROIListResponse` with:

```json
{
  "reloaded": true,
  "reload_receipt": {
    "request_sequence": <int>,
    "accepted_sequence": <int>,
    "failed_sequence": <int>,
    "active_config_sha256": "<64 lowercase hex characters>",
    "reload_error_count": <int>,
    "objects_removed_count": <int>
  }
}
```

`reloaded=true` means the native element synchronously accepted the exact
derived INI bytes. It is not inferred from an HTTP status, a file write, or an
incremented Python counter. The request and accepted sequences must equal the
new monotonic request, the failed sequence must not equal it, the active hash
must equal the derived INI SHA-256, and the error count must remain unchanged.

### Endpoints

Implementation provides the following endpoints:

- **GET** `/api/v1/analytics/rois?stage=<stage-name>`
  - Returns `ROIListResponse`.
- **POST** `/api/v1/analytics/rois`
  - Body: `ROIUpdateRequest`
  - Returns: `ROIUpdateResponse` with the exact native reload receipt.

### Behavior

- Only `stage="exclude"` is writable. Stream IDs are canonical non-negative
  decimal strings and an update may name only existing streams. Stream and ROI
  IDs must be unique within the request.
- ROI IDs use 1–64 letters, digits, dots, underscores, or hyphens. A polygon has
  3–128 finite points and must remain within the stage dimensions. The integer-
  rounded native polygon must contain at least three distinct points and have
  non-zero area.
- An enabled stream requires at least one ROI. A disabled stream may have an
  empty ROI list; deleting the final ROI therefore requires `enable=false` in
  the same request. The complete stored stage retains exact coverage for every
  configured source. Missing coverage is fatal on the native streaming path.
- The durable YAML is limited to 4 MiB. The derived native INI is separately
  limited to 1 MiB. Both limits are enforced before a commit and by the DS9
  supervisor on the next session, so an accepted API state remains restartable.
- Before graph construction, each runtime loads the durable YAML, requires the
  `exclude` stage, derives the INI, and points the pre-tracker element at that
  exact writable path. After activation it verifies the native initial active
  hash and zero-error receipt before exposing the runtime as healthy.
- A mutation is one fail-closed transaction. The API validates both serialized
  forms, snapshots both files, writes atomically with fsync, requests one
  monotonic native reload with the expected INI SHA-256, and publishes the
  cache/runtime receipt only after full acknowledgement. YAML parsing,
  transaction snapshots/replacements, and native INI reads reject duplicate
  keys, linked/non-regular targets, replacement races, and over-limit content
  at their respective boundaries; existing permissions are not repaired
  implicitly.
- A definite pre-commit or native rejection restores the YAML, INI, and cache
  to their exact snapshots. If rollback fails, or native activation may have
  succeeded but its complete receipt/publication cannot be proven, the
  analytics state is permanently poisoned for that process and the runtime
  begins fatal shutdown. A file-only rollback is not reported as recovery from
  an ambiguous native commit.

### Shutdown quiescence

Synchronous FastAPI handlers may outlive the Uvicorn listener thread. Runtime
shutdown therefore stops the REST server, joins its thread, acquires the
analytics transaction lock, and retains that lock through native teardown. The
resulting shutdown receipt must prove the server/worker pair was consistent,
stop was requested, the server thread stopped, and the transaction lock was
retained. Failure to obtain this lease is fatal; callback-owned native state is
not torn down while a late ROI transaction could still enter.

### Native analytics persistence boundary

The native host service receives explicit writable
`NOESIS_ANALYTICS_CONFIG` and `NOESIS_ANALYTICS_EXCLUDE_CONFIG` paths from its
environment. It does not mount or seed a container path. The runtime validates
that exact pair before graph construction, and accepted API edits persist
across host-service restarts because the same files remain configured. Runtime
scratch state, build outputs, depth output, and diagnostic evidence stay under
unique session directories beneath `NOESIS_DS9_RUNTIME_ROOT`.

### Error Handling

- `404 Not Found`: requested stage does not exist.
- `422 Unprocessable Entity`: unsupported writable stage, unknown/duplicate
  stream, unsafe or duplicate ROI ID, non-finite/out-of-bounds/degenerate
  polygon, enabled stream with no ROI, or serialized size violation.
- `500 Internal Server Error`: transaction preparation or durable persistence
  failed before native commit; exact rollback is attempted.
- `503 Service Unavailable`: config is missing/poisoned, the required native
  pipeline or reload bridge is unavailable, native acknowledgement fails, or a
  post-dispatch commit is ambiguous. A later request never clears poisoned
  state.

Environment/config notes:
- Analytics config path: `NOESIS_ANALYTICS_CONFIG` (default `config/nvdsanalytics.yaml`).
- Exclusion INI override: `NOESIS_ANALYTICS_EXCLUDE_CONFIG`; otherwise pulled from the `analytics_exclude` component config in the live pipeline.

## 3. ReID Alias API

**Module:** `noesis/server/reid_api.py`

The ReID alias API manages StableID “soft merges” (aliasing), suggestions, and
audit history. These endpoints are DS9.1-only and operate on the live
`StableIDManager` in memory.

### Common Data Structures (Pydantic Models)

- `AliasListResponse`:

```json
{
  "enabled": <bool>,
  "aliases": { "<src>": <dst>, ... },
  "alias_file": "<string|null>",
  "copresence_window_s": <float>,
  "history_count": <int>
}
```

- `MergeRequest`:

```json
{
  "a": <int>,
  "b": <int>,
  "canonical": <int|null>,
  "append_embeddings": <bool>,
  "force": <bool>
}
```

- `MergeResponse`:

```json
{
  "applied": <bool>,
  "src": <int>,
  "dst": <int>,
  "canonical": <int>,
  "reason": "<string|null>",
  "aliases": { "<src>": <dst>, ... }
}
```

- `MergeBatchRequest`:

```json
{
  "pairs": [MergeRequest, ...],
  "force": <bool>
}
```

- `MergeBatchResponse`:

```json
{
  "results": [MergeResponse, ...],
  "applied_count": <int>,
  "failed_count": <int>,
  "aliases": { "<src>": <dst>, ... }
}
```

- `UnsetRequest`:

```json
{
  "src": <int>
}
```

- `UnsetResponse`:

```json
{
  "src": <int>,
  "removed": <bool>,
  "reason": "<string|null>",
  "aliases": { "<src>": <dst>, ... }
}
```

- `SuggestRequest`:

```json
{
  "min_sim": <float|null>,
  "limit": <int>,
  "require_inactive": <bool>
}
```

- `SuggestCandidate`:

```json
{
  "a": <int>,
  "b": <int>,
  "sim": <float>,
  "pose_sim": <float|null>,
  "canonical": <int>,
  "preferred_canonical": <int>,
  "a_embedding_count": <int>,
  "b_embedding_count": <int>,
  "blocked": <bool>,
  "block_reason": "<string|null>"
}
```

- `SuggestResponse`:

```json
{
  "candidates": [SuggestCandidate, ...],
  "default_min_sim": <float>
}
```

- `HistoryResponse`:

```json
{
  "history": [
    { "ts": <float>, "action": "<merge|unset|clear_all>", "...": "..." }
  ]
}
```

Notes:
- `aliases` is a canonicalized map view (alias sources point directly to the
  current canonical).
- JSON map keys are strings; treat them as ints in clients.

### Endpoints

- **GET** `/api/v1/reid/aliases`
  - Returns `AliasListResponse`.
- **POST** `/api/v1/reid/aliases/merge`
  - Body: `MergeRequest`
  - Returns: `MergeResponse`.
- **POST** `/api/v1/reid/aliases/merge-batch`
  - Body: `MergeBatchRequest`
  - Returns: `MergeBatchResponse`.
- **POST** `/api/v1/reid/aliases/unset`
  - Body: `UnsetRequest`
  - Returns: `UnsetResponse`.
- **POST** `/api/v1/reid/aliases/suggest`
  - Body: `SuggestRequest`
  - Returns: `SuggestResponse`.
- **POST** `/api/v1/reid/aliases/clear`
  - Returns: `{ "cleared": <int>, "aliases": {} }`.
- **GET** `/api/v1/reid/aliases/history?limit=<int>`
  - Returns: `HistoryResponse` with newest-first entries (default limit = 100).

### Behavior and Guardrails

- **Canonical choice:** For manual merges, canonical defaults to the lower ID,
  but the API accepts an explicit `canonical` to apply a preferred canonical
  from suggestions.
- **Active guardrail:** If both IDs are currently active, merges are rejected
  unless `force=true`.
- **Co-presence guardrail:** If the pair was co-present within
  `copresence_window_s` (default 600s), merges are rejected unless `force=true`.
  Suggestions still surface such pairs with `blocked=true`.
- **`min_sim` tuning:** `SuggestRequest.min_sim` is model-dependent. Re-tune
  after ReID model changes.
- **Mutual-nearest-neighbor filter:** Suggestions require mutual nearest
  neighbors with a minimum margin to reduce false positives; blocked candidates
  include a `block_reason`.
- **Batch semantics:** `merge-batch` prevalidates the full set, treats connected
  components as a single merge, rejects conflicting canonicals, and returns
  per-pair results. Applied merges are committed; failed pairs remain untouched.

### History, Persistence, and Limits

- Alias history is capped at `alias_history_max` entries (default 1000). Oldest
  entries are pruned on save.

### Known Limitations

- Zone-state dwell timers are not remapped on alias; dwell timing effectively
  resets when an ID canonicalizes.

## 3.1 Household Identity REST (planned)

**Module:** `noesis/server/reid_api.py` (extensions; Phase 1–3)

Household mode keeps the alias endpoints above but defaults suggest-only merge
behavior. Resident list/enroll endpoints are available when household mode is on
(default; unset `NOESIS_HOUSEHOLD_IDENTITY` or set `=1`):

Environment:
- `NOESIS_HOUSEHOLD_IDENTITY` defaults to on (`1` when unset). Opt out with `=0`
  for legacy open-world StableID debugging.
- `NOESIS_HOUSEHOLD_ARCHIVE_STATE=1` (default when household on) archives legacy
  `~/.noesis/reid_gallery.npz`, `reid_aliases.json`, and `sid_pool.json` into
  `~/.noesis/household/backups/<UTC>/` on first cutover.
- `NOESIS_CAMERA_TOPOLOGY_FILE` (default `config/camera_topology.yaml` via repo root).

### Resident endpoints

- **GET** `/api/v1/reid/residents`
  - Returns enrolled household residents (UUID, display name, gallery stats).
  - Requires household mode; returns 400 otherwise.
- **POST** `/api/v1/reid/residents/enroll`
  - Body: `{ "display_name": string, "stable_id"?: int, "visitor_id"?: int }`
  - Binds a visitor or stable ID into the resident enrollment table (sticky ID 1..N).
  - Remaps live tracks/zones/ghosts/gallery when the visitor SID differs from the new resident SID.
- **PATCH** `/api/v1/reid/residents/{uuid}`
  - Update `display_name` (propagates to live tracks).
- **DELETE** `/api/v1/reid/residents/{uuid}`
  - Removes enrollment; remaps any live/ghost references to a fresh visitor SID.
- **GET** `/api/v1/reid/identity_health`
  - Mint / false-share / overlap / gallery counters plus resident list.

These household resident and health routes are exact DS9.1 product parity;
their source, OpenAPI paths/schemas, response bytes, and boundary instrumentation
are regression-tested together.

Frontend: oai2-fe topbar **People** drawer consumes these endpoints (plus suggest-only alias merge).

Persistence (household mode):
- `~/.noesis/household/residents.json`
- `~/.noesis/household/resident_gallery.npz`
- `~/.noesis/household/visitor_gallery.npz`
- `~/.noesis/household/sid_pool.json`
- `~/.noesis/household/backups/` (archived pre-cutover state)

## 3.2 Identity v2 API

**Module:** `noesis/server/reid_v2_api.py`

The versioned identity surface is mounted in the authenticated native DS9.1
baseline and disabled V3DT adapter at `/api/v2/reid`. It is backed by one
process-owned SQLite store, open-set runtime, and whole-frame coordinator. The browser never
submits an embedding: enrollment intent names an exact server observation key,
and the server consumes the short-lived immutable evidence cached for that key.

Runtime mode is `NOESIS_IDENTITY_V2_MODE=disabled|shadow|authoritative` and
defaults to `shadow`. `authoritative` startup is rejected unless
`NOESIS_IDENTITY_V2_SCORING_ARTIFACT` names a valid version-2 JSON calibration
artifact; heuristic default scores are never allowed to become public identity
authority. Because that artifact has scorer-only scope, it is necessary but not
sufficient for public cutover. `NOESIS_IDENTITY_V2_SCORING_ARTIFACT_SHA256` independently pins its
exact bytes, while `NOESIS_IDENTITY_V2_MODEL_SEMANTIC_PROFILE_SHA256` must match
the digest newly derived from the active engine, model YAML, nvinfer crop and
preprocessing, tensor, the actually loaded ReID metadata-extension binary, and
Python normalization/gallery-scoring implementations. The artifact contract is
`noesis.identity.open_set_calibration` v2 and must bind the exact active
`model_sha256`, `model_layer`, `embedding_dim`, semantic profile, deterministic
evidence-unit policy, conservative gallery envelope, and separately hashed
benchmark and household datasets. Its authority scope is the open-set scorer
policy only. Version 1
artifacts reject because their frame-level independence semantics are
ambiguous. The provenance-locked subject-disjoint benchmark holdout supplies
the generic safety claim: at least 300 challenge-covered resident people and
300 challenge-covered unknown people, scored worst-case across all encounters
for each person, with exact one-sided 95% FAR and misidentification upper bounds
at or below 1%. Benchmark train requires 50 resident and 50 unknown
challenge-covered people. Benchmark provenance must attest licensed real-human
truth; household provenance must attest owner-consented real-human truth.

The separately captured household stratum requires 10 known/10 unknown train
and 20 known/20 unknown holdout encounters. Both household partitions require
zero false accepts, zero misidentifications, FRR at or below 35%, and zero
harmful or rejection-rescuing resident-prior changes. Household policy search
may only retain or raise benchmark rejection gates and cannot change its score
mapping or bounded prior. Metrics aggregate all frames at physical-encounter
worst case; deterministic five-second tracklet units and bounded fit thinning
never become extra confidence trials. An arbitrary policy object, a weaker
local policy, or a calibration for another engine/profile is rejected. Every
challenge has non-empty resident+visitor competition and an impostor;
zero-exemplar or blocked candidates cannot support fitting or gallery capacity.
The runtime envelope uses the minimum coverage across both strata and rejects
startup or prospective gallery growth outside it before durable mutation.

Public authority also requires
`NOESIS_IDENTITY_V2_AUTHORITY_CUTOVER_ARTIFACT` and its independent exact-byte
`NOESIS_IDENTITY_V2_AUTHORITY_CUTOVER_ARTIFACT_SHA256` pin. The strict
`noesis.identity.authority_cutover` v1 contract binds the scorer artifact,
active model semantic profile, DS9.1 executable authority profile, camera
topology, and exact camera set. It contains two distinct literal-pass evidence
records—whole-frame coordinator replay and occupied-scene runtime—whose owner-
private report paths, sizes, SHA-256 values, revisions, and completion times are
re-read and verified before the identity store opens. DS8, DS9.0, or mismatched
DS9.1 evidence cannot authorize the current native runtime; changed code,
topology, cameras, scorer bytes, native extraction, or report bytes fail
startup. The generated schema is
`contracts/schema/identity_authority_cutover.schema.json`.

`disabled` leaves the routes mounted but returns 503 because no v2
runtime is registered. The offline workflow and operator commands are in
`plans/household_identity/calibration_and_enrollment.md`.

Read endpoints:

- `GET /api/v2/reid/health` returns store counts, scoring calibration status,
  scorer-only authority scope, active engine and semantic-profile SHA-256,
  dimension/layer/mode, authority runtime-profile SHA-256, public cutover status
  and cutover artifact ID, calibrated resident/visitor/total/exemplar gallery
  maxima, and bounded observation-cache health. It never exposes cutover report
  paths or contents.
- `GET /api/v2/reid/residents` lists durable residents.
- `GET /api/v2/reid/enrollment/proposals` and
  `GET /api/v2/reid/enrollment/proposals/{proposal_uuid}` inspect pending or
  confirmed enrollment evidence.
- `GET /api/v2/reid/evidence/status` reports whether private score-only capture
  is enabled, its contract/source/session/runtime, retained event/byte counts,
  configured record/byte/age bounds, prune count, and last event time. It never
  returns a path, candidate scores, embeddings, or vectors. Evidence contract
  v2 is sequence/hash chained and requires a private retained-head/tail
  checkpoint; unexplained deletion or truncation fails before calibration.
  `recorded_event_count` and `last_observed_at_us` are durable-only. Async
  writers additionally report `pending_event_count`, `pending_frame_count`,
  `dropped_event_count`, `failed`, and `last_error`; pending rows are not yet
  calibration evidence. Every row also binds the runtime-derived model
  semantic profile, which label provenance must match exactly.
- `GET /api/v2/reid/migration/review` returns an optional sanitized,
  biometric-free legacy review containing duplicate normalized-name groups,
  blocked records, and residents without importable anchors. The response
  always declares `apply_available_from_api=false`.

Mutation endpoints:

- `POST /api/v2/reid/enrollment/proposals` accepts only
  `{key, display_name, compatibility_sid?, update_resident_uuid?, ttl_s?}`.
  `key` is the exact `{run_id,camera_id,tracker_id,frame_id,observation_id}`
  published by Noesis. Extra fields, including `embedding`, are rejected.
- `POST /api/v2/reid/enrollment/proposals/{proposal_uuid}/confirm` requires the
  same exact key plus the proposal evidence digest. Expired, replayed, stale, or
  mismatched evidence does not mutate enrollment state.
- `PATCH /api/v2/reid/residents/{resident_uuid}` changes the display name.
- `DELETE /api/v2/reid/residents/{resident_uuid}` deletes the resident and all
  cascaded biometric evidence.

The v2 store defaults to `~/.noesis/household/identity_v2.sqlite3` with owner-
only directory/file modes. It is independent of legacy NPZ/JSON state; runtime
startup never performs or implies a partial legacy migration. No migration
apply endpoint exists.

## 4. Virtual Twin API

Read-only APIs expose Noesis-owned offline virtual-twin revisions built under
`data/virtual_twin/revisions/<revision_id>/`. These endpoints do not trigger
reconstruction work and do not fall back to browser-side depth projection.
Revisions are built by `scripts/build_virtual_twin_reconstruction.py`, which
triggers the DS9.1 `/api/v1/depth/refresh` path and consumes the persisted
MapAnything Zarr snapshots under `data/depth/<camera>/.../*.zarr`. The builder
also copies RGB keyframes and MapAnything depth/confidence/mask arrays into the
revision bundle so the reconstruction evidence survives depth-retention pruning.

### Endpoints

- **GET** `/api/v1/virtual-twin/revisions`
  - Returns revision summaries sorted newest-first.
- **GET** `/api/v1/virtual-twin/latest`
  - Returns the latest revision manifest, metrics, tracking alignment, and
    artifact URLs.
- **GET** `/api/v1/virtual-twin/revisions/{id}/manifest`
  - Returns one revision manifest plus artifact URLs.
- **GET** `/api/v1/virtual-twin/revisions/{id}/metrics`
  - Returns reconstruction metrics for one revision.
- **GET** `/api/v1/virtual-twin/revisions/{id}/tracking-alignment`
  - Returns tracking-alignment readback for one revision.
- **GET** `/api/v1/virtual-twin/calibration/scene-extrinsics`
  - Returns the Menon-scene camera calibration poses from
    `config/camera_calibration_menon_obj.json`.
  - Response fields:
    - `source`: calibration source identifier.
    - `path`: repo-relative calibration path.
    - `frame`: calibration frame; currently `menon_scene`.
    - `cameras`: mapping of camera IDs to calibration payloads.
    - `preview_meta`: optional preview metadata from the calibration file.
  - Missing or malformed calibration data returns a JSON error rather than an
    empty calibration fallback.
- **GET** `/api/v1/virtual-twin/revisions/{id}/artifacts/{path}`
  - Serves revision-relative GLB, JSON, PLY, or NPZ artifacts.

Dashboard browsers consume these artifacts through Menon's authenticated
same-origin gateway; the internal bearer is never exposed to browser code.
Explicit loopback development may declare exact origins with
`NOESIS_REST_CORS_ORIGINS`. Regex, wildcard, RFC1918-default, and
`NOESIS_REST_CORS_ALLOW_ALL` behavior is forbidden by the runtime boundary.

### Artifact Contract

Each revision is expected to contain:

- `manifest.json`
- `planes.json`
- `surfaces.glb`
- `points.ply`
- `points.npz`
- `tracking_alignment.json`
- `metrics.json`

`surfaces.glb` is the browser-facing Menon artifact. It is a textured triangle
mesh in Menon scene units constrained to supported structural model surfaces,
not a free-floating dense depth point cloud. The embedded texture atlas is baked
from the saved pipeline RGB keyframes after projecting model-surface texels back
through the revision registration and applying the MapAnything depth/confidence
gate. The texture bake also reprojects the sampled MapAnything pixel into Menon
scene space and requires it to land on the same structural surface that the
texel is coloring. The bake also uses a camera-view visibility map of the Menon
structural OBJ, so a pixel can color only the front-most model surface assigned
to that image location; pixels that belong to another wall/floor block, or to an
occluded surface behind it, are rejected.
Dense reconstructed points remain in `points.ply` and `points.npz` for analysis
and diagnostics. The atlas may also apply an explicit exposure/gamma/contrast
tone map recorded in `metrics.browser_render_budget.texture`.
By default the builder rejects effectively grayscale keyframes for RGB texture
bakes; `--allow-grayscale-texture` is reserved for explicit diagnostic builds.

The manifest also exposes `evidence_dirs.keyframes` and
`evidence_dirs.mapanything`; per-frame rows list revision-relative RGB PNG and
MapAnything NPZ evidence paths plus calibration metadata. `metrics.json`
includes `room_model_leakage_ratio`, `room_model_leakage`, browser render budget,
model-surface projection counts/distances, texture-atlas coverage metrics, plane
residuals, registration normal errors, dense surface-refinement diagnostics, and
gate booleans. Artifact paths are revision-relative and path traversal is
rejected.

`tracking_alignment.json` remains an explicit artifact input. Menon can keep it
in readback mode or opt into applying the revision transform through its
`reprojectionApplyVirtualTwinAlignment` setting after
`scripts/validate_virtual_twin_tracking.py` passes for the revision camera.

## 5. Guided alignment-walk API

DS9.1 mount the same owner-authenticated, one-active-session controller
under `/api/v1/alignment-walk`. The controller consumes the runtime's own
authenticated WebSocket stream, persists owner-private evidence, and exposes
only run-local tracklet keys and image geometry to the operator.

- **POST** `/api/v1/alignment-walk/sessions`
  - Requires `duration_s`, a complete
    `noesis.alignment.walk_waypoints` v1 manifest, and
    `scene_binding={release_id, authored_scene_sha256,
    world_to_scene_sha256}`.
  - Waypoints declare `expected_scene_xyz`; Noesis derives the legacy XZ view.
  - Returns `session_id`, `state` (with a `status` compatibility alias),
    `deadline_at_us`, waypoint state, and bounded active tracks.
- **GET** `/api/v1/alignment-walk/sessions/{session_id}`
  - Returns capture state, reflected markers, waypoint marker state, counts,
    scene-binding verification, and active tracks.
  - Active tracks contain only the run-local tracklet key, camera, frame,
    bbox/image foot/image size, confidence, and age. World/depth/identity data
    is not an operator selection input.
- **POST** `/api/v1/alignment-walk/sessions/{session_id}/markers`
  - Accepts `waypoint_id`, `phase=arrived|leave`, optional actor, and a run-local
    `tracklet_key`. `arrived` requires an explicit tracklet; `leave` may omit it.
- **POST** `/api/v1/alignment-walk/sessions/{session_id}/finish`
  - Stops capture within a fixed bound and returns a terminal
    `complete|failed` state. Runtime shutdown marks an active capture failed.
- **POST** `/api/v1/alignment-walk/sessions/{session_id}/analysis`
  - Verifies the sealed capture and writes FIT/HOLDOUT camera-rotation and
    physical-depth candidate artifacts. Concurrent analyses are serialized.
- **GET** `/api/v1/alignment-walk/sessions/{session_id}/results`
  - Returns `state=results_ready` with bounded report/check summaries,
    aggregate metrics, advisory similarity diagnostics, and per-camera
    candidate summaries. `candidate_summary` is the authoritative final
    HOLDOUT-gated admission readback and includes `fit_solver_status`,
    `calibration_candidate_status`, `advisory_only=true`, and
    `active_config_modified=false`; it is distinct from advisory similarity.
    Full frame evidence remains private on disk.

The scene transform digest is always backend verified. If the calibration
bundle cannot authoritatively expose the requested release or authored-scene
digest, `scene_binding_verification.status=partial` names the verified and
operator-asserted fields; it never claims a full scene-release match.

## 6. Manual semantic-capture API

DS9.1 expose the same owner-private batch capture contract for the OAI2
Sem-seg diagnostic. A browser starts capture only through Menon's coordinated
`POST /api/diagnostics/semantic-seg/captures` action; Menon supplies internal
authentication, idempotency, audit, and exact latest-result readback.

- **POST** `/api/v1/semantic-seg/captures`
  - Accepts exactly `{ "model": "s" | "l" }`.
  - Starts one serialized TensorRT/DeepStream run over the three canonical room
    streams. Model or camera selection in the browser never calls this route.
  - Returns `noesis.semantic_seg.capture` v1 only after aligned RGB, class-map,
    and masked JPEG/PNG artifacts exist for Living Room, Kitchen, and Family
    Room. A busy capture returns `409`; timeout or unavailable runtime assets
    return an explicit error without changing the last completed result.
- **GET** `/api/v1/semantic-seg/captures/latest/{model}`
  - Returns the most recently completed manifest for the exact model. This is
    the readback boundary for Menon's coordinated action, not an inference
    trigger.
- **GET** `/api/v1/semantic-seg/captures/{capture_id}/{camera_id}/{artifact}`
  - Serves exact `raw`, `class-map`, or `masked` image bytes. The same-origin
    gateway permits these reads to operator and owner roles; capture creation
    remains owner-only through the coordinated route.

Runtime capture files live in the writable Noesis state boundary. The Small and
Large fixed batch-3 engines, ADE20K labels, and semantic parser are immutable release
assets. Partial capture directories are removed and are never published as
latest.

## 7. Validation Toolbox API Status

The Noesis/Menon validation toolbox currently exposes CLI/report contracts, not
DS9.1 REST endpoints. Its public artifacts are JSON files under
`diagnostics/validation/<run_id>/`, plus optional visual evidence indexed by
`visual/index.json`.

If a future change publishes validation reports, Menon camera reprojection
evidence, or regression summaries over REST, add the endpoint, request, response,
and failure/status semantics here before treating that surface as public.

## 8. Future DS9.1 REST Endpoints

If additional DS9.1 REST endpoints are introduced (e.g., for calibration, debug, or pipeline control), they should:

- Reuse the same pattern of Pydantic models for clear JSON schemas.
- Be documented here with:
  - Path and method.
  - Request parameters/body.
  - Response model.
  - Error responses.
