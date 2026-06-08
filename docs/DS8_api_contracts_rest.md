# DS8 REST API Contracts
_Status: validation-toolbox addendum current as of 2026-05-27._

This document describes the REST endpoints used by the DS8 runtime. Implementations must preserve these contracts unless all consumers are updated in lockstep.

## 1. Depth API – `/api/v1/depth/refresh`

**Module:** `noesis/server/depth_api.py`

### Endpoint

- `GET /api/v1/depth/refresh?seconds=<int>`

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
  - Depth pipeline not ready (e.g., `infer.yaml` missing or DS8 pipeline failed to initialize).
- `500 Internal Server Error`
  - Depth control failed or returned a malformed payload.

Environment/config notes:
- Pipeline config path: `NOESIS_DS8_PIPELINE_CONFIG` (default `config/infer.yaml`).
- Stub mode (tests/offline): set `NOESIS_DEPTH_API_FORCE_STUB=1` to bypass DS8 bindings while preserving the contract.

Implementation note: the underlying `enable_depth(seconds)` function in `noesis.pipelines.ds8_pipeline` must control DS8 gating (e.g., a `BufferOperator` gate) and not deprecated valve paths.

## 2. Analytics ROI API

**Module:** `noesis/server/analytics_api.py`

The analytics API exposes ROI configuration for DS8 analytics stages, typically mapped to a DS8 analytics YAML file via `NOESIS_ANALYTICS_CONFIG`.

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

### Endpoints

Implementation provides the following endpoints:

- **GET** `/api/v1/analytics/rois?stage=<stage-name>`
  - Returns `ROIListResponse`.
- **POST** `/api/v1/analytics/rois`
  - Body: `ROIUpdateRequest`
  - Returns: `ROIUpdateResponse` (includes `reloaded` flag).

### Behavior

- On update:
  - The in-memory analytics config cache is updated.
  - The YAML file at the analytics config path is written or maintained consistently.
  - A reload hook (registered via `attach_analytics_reload_bridge`) is invoked, or the DS8 analytics component’s runtime config is updated directly.

### Error Handling

- If the analytics config file is missing or invalid, the API should respond with suitable 4xx/5xx codes and log the issue.
- If a stage name is missing, `404 Not Found` or a structured error should be returned.

Environment/config notes:
- Analytics config path: `NOESIS_ANALYTICS_CONFIG` (default `config/nvdsanalytics.yaml`).
- Exclusion INI override: `NOESIS_ANALYTICS_EXCLUDE_CONFIG`; otherwise pulled from the `analytics_exclude` component config in the live pipeline.

## 3. ReID Alias API

**Module:** `noesis/server/reid_api.py`

The ReID alias API manages StableID “soft merges” (aliasing), suggestions, and
audit history. These endpoints are DS8-only and operate on the live
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

## 4. Virtual Twin API

Read-only APIs expose Noesis-owned offline virtual-twin revisions built under
`data/virtual_twin/revisions/<revision_id>/`. These endpoints do not trigger
reconstruction work and do not fall back to browser-side depth projection.
Revisions are built by `scripts/build_virtual_twin_reconstruction.py`, which
triggers the DS8 `/api/v1/depth/refresh` path and consumes the persisted
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

The DS8 REST app enables CORS for localhost and RFC1918 private-network browser
origins by default so Menon dev/probe pages can fetch these artifacts from the
Noesis REST port without special browser flags. Operators can override this with
`NOESIS_REST_CORS_ORIGINS`, `NOESIS_REST_CORS_ORIGIN_REGEX`, or
`NOESIS_REST_CORS_ALLOW_ALL`.

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

## 5. Validation Toolbox API Status

The Noesis/Menon validation toolbox currently exposes CLI/report contracts, not
DS8 REST endpoints. Its public artifacts are JSON files under
`diagnostics/validation/<run_id>/`, plus optional visual evidence indexed by
`visual/index.json`.

If a future change publishes validation reports, Menon camera reprojection
evidence, or regression summaries over REST, add the endpoint, request, response,
and failure/status semantics here before treating that surface as public.

## 6. Future DS8 REST Endpoints

If additional DS8 REST endpoints are introduced (e.g., for calibration, debug, or pipeline control), they should:

- Reuse the same pattern of Pydantic models for clear JSON schemas.
- Be documented here with:
  - Path and method.
  - Request parameters/body.
  - Response model.
  - Error responses.
