# DS8 Tensor Path Contracts (MapAnything/ReID/Pose/StableID)
_Status: pose + ReID native + StableID GPU-similarity implemented; MapAnything strict-ID/async hardening implemented. Updated: 2026-02-08._

## 2026-02-08 closure update
- Added missing tensor-path validation modules:
  - `tests/test_reid_tensor_contracts.py`
  - `tests/test_pose_tensor_contracts.py`
  - `tests/test_pose_meta_native_contracts.py`
- Completed runtime counter coverage:
  - `tensor_host_copies_total.reid`
  - `tensor_host_copies_total.pose`
  - `tensor_boundary_copy_bytes_total.pose_meta`
  - `tensor_boundary_copy_bytes_total.depth_store`
- Added payload-size guards for pose meta attach in both Python (`hooks.py`) and native bridge (`native/noesis_pose_meta_ext.cpp`).

## Scope
- Primary implementation file: `noesis/pipelines/hooks.py`.
- Related native bridge in scope: `native/noesis_pose_meta_ext.cpp`.
- Related boundary sink for MapAnything depth persistence: `geometry/depth_source.py`.

## Core-Path vs Boundary-Path
- Core-path: tensor metadata selection and identity-critical compute that must not stall or regress tracking/StableID correctness.
- Boundary-path: serialization/persistence/visualization work where host copies are allowed with explicit budget and instrumentation.

## Current Tensor Paths and Host Copies

### MapAnything (full-frame SGIE)
- Path: `_MapAnythingOperator.handle_metadata` -> `MapAnythingProcessor.handle_nvds_tensor_ds8`.
- Core-path segment:
  - `handle_metadata`: iterates `frame_meta.tensor_items`, converts via `as_tensor_output()`, filters by `unique_id`.
  - `handle_nvds_tensor_ds8` async mode: selects depth/conf/mask tensor handles and `clone()`s them before queueing `_MapAnythingJob`.
- Boundary-path segment:
  - `_start_async_worker` loop -> `MapAnythingProcessor._to_numpy` (`torch_dlpack.from_dlpack(...).detach().cpu().numpy()` host materialization).
  - `_emit_from_tensors` (`np.asarray`, `cv2.resize`, mask creation) CPU post-process.
  - `DepthStorageManager.store` -> `_create_job` (`np.ascontiguousarray(...).copy()`) for async disk writer.
- Extra compatibility path (not DS8 core): `handle_nvds_tensor` -> `_extract_tensor_layers` -> `_numpy_from_layer` (`np.array(..., copy=True)` and fallback buffer copies).

### ReID (object SGIE)
- Path: `_AnalyticsTelemetryProcessor.handle_frame_ds8` -> `_extract_reid_embedding_ds8` -> `_extract_reid_embedding_native`.
- Core-path segment:
  - `handle_frame_ds8` gates embedding extraction via `stable_id_mgr.needs_embedding(...)`.
  - `_extract_reid_embedding_ds8` uses native extraction and normalizes returned vectors before `_maybe_assign_stable_id(...)`.
- Host copy sites in core-path:
  - Native extension returns host `float32` embedding data for StableID matching.

### Pose (object SGIE)
- Feature path: `_PoseFeatureOperator.handle_metadata` -> `PoseFeatureProcessor.handle_frame_ds8` -> `_extract_pose_native`.
- Overlay path: `_PoseKeypointOverlayOperator.handle_metadata` -> `PoseKeypointOverlayProcessor.handle_batch_ds8` -> `_extract_pose_payload`.
- Core-path segment:
  - Native extension decode (`extract_pose_keypoints`) enforces `unique_id` and layer contract.
- Host copy sites:
  - Native extension returns ROI/absolute keypoints payload on host for boundary serialization.
- Boundary-path segment:
  - Pose feature JSON serialization: `json.dumps(payload, ...)` in `PoseFeatureProcessor.handle_frame_ds8`.
  - Native meta attach: `noesis_pose_meta_ext.attach_pose_features(...)`.
  - Native copies in `native/noesis_pose_meta_ext.cpp`:
    - `attach_pose_features*`: `g_strdup(payload_json.c_str())`.
    - `pose_meta_copy`: `g_strdup(...)` on metadata copy callbacks.
    - `extract_pose_features`: returns `py::str(payload)` to Python.

### Pose (implemented native path)
- Native extractor added: `noesis_pose_meta_ext.extract_pose_keypoints(...)`.
  - Decodes `NVDSINFER_TENSOR_OUTPUT_META` in C++ from `output0`.
  - Uses strict `unique_id == pose gie_id`.
  - Handles live tensor layout observed as `shape=[300,57]`.
- Python pose feature hook (`PoseFeatureProcessor`):
  - Uses native keypoint payload, not Python tensor decode.
  - Publishes `keypoints_roi` and `keypoints_abs`.
- Overlay alignment patch:
  - Overlay now prefers `keypoints_roi` and remaps via source-bbox -> current-bbox scaling.
  - Fixes per-camera/tile offsets caused by stage-to-stage bbox transform differences.

## Strict Invariants

### Global invariants
1. `unique_id` must match configured GIE id before any host materialization.
2. Approved host-materialization points in production:
   - MapAnything: `MapAnythingProcessor._to_numpy` (boundary worker path).
   - ReID/Pose: native DS8 metadata extensions.
3. Legacy `NvDsInferTensorMeta` decoding (`_extract_tensor_layers` / `_numpy_from_layer`) is compatibility-only and must not run in DS8 production path.
4. Any fallback that processes "first available tensor" on GIE mismatch is forbidden in core-path.

### MapAnything invariants
1. Production mode must keep async postprocess enabled (`NOESIS_MAPANYTHING_POSTPROCESS_ASYNC=1`) so probe thread does not do D2H copies.
2. `handle_nvds_tensor_ds8` may perform GPU `clone()` for ownership transfer, but must not perform host copies in async mode.
3. Boundary persistence copies in `DepthStorageManager._create_job` are allowed and treated as boundary budget.

### ReID invariants
1. ReID embedding extraction occurs only when `needs_embedding(...)` returns true.
2. Decoded embedding dtype is `float32`, flattened 1-D, L2-normalized (norm ~1.0, finite).
3. Production path uses native tensor-meta decode; CPU extraction path is removed in hard-cutover.

### StableID invariants (hybrid CPU/GPU)
1. StableID public contract is unchanged: only `stable_id` is user-visible; backend choice is internal.
2. GPU scope is vector math only: embedding normalization, cosine similarity, top-k candidate search.
3. CPU scope remains stateful identity logic: aliases, hysteresis, TTL, merge policy, zone/camera rules.
4. CPU and GPU backends must produce equivalent assignment outcomes on parity fixtures.
5. Backend init/compute failures must fail fast in hard-cutover runtime.

### Pose invariants
1. Pose tensor extraction must use `unique_id == pose gie_id` and prefer named `output0`.
2. Pose feature and overlay paths must not independently materialize the same tensor twice per object/frame after refactor.
3. Native payload attachment is boundary-path and must remain bounded in payload size and copy count.
4. Overlay must render from ROI-local keypoints remapped to the current bbox (`sx = dst_w/src_w`, `sy = dst_h/src_h`) to avoid tiler/tracker drift.

## Refactor Directives (Exact Files/Functions)

1. `noesis/pipelines/hooks.py::_MapAnythingOperator.handle_metadata`
- Remove fallback that processes `converted_items[0]` when no `unique_id` match.
- On mismatch, drop tensor meta and increment a strict-mismatch counter.

2. `noesis/pipelines/hooks.py::MapAnythingProcessor.handle_nvds_tensor_ds8`
- Make async path the only production path.
- Keep only tensor-handle ownership transfer in probe thread; move all D2H conversion to worker.

3. `noesis/pipelines/hooks.py::MapAnythingProcessor._to_numpy`
- Convert this into shared helper usage (same DLPack policy as Pose/ReID), with explicit stream-aware `__dlpack__` call.

4. `noesis/pipelines/hooks.py::_AnalyticsTelemetryProcessor._extract_reid_embedding_ds8`
- Keep `unique_id` + layer-name selection strict; if fallback layer heuristic is used, emit one-time warning and metric.

5. `noesis/pipelines/hooks.py::PoseFeatureProcessor.handle_frame_ds8`
- Split into core tensor/feature compute vs boundary serialization/attach section and instrument both timings separately.

9. `native/noesis_pose_meta_ext.cpp::attach_pose_features`, `attach_pose_features_frame`, `pose_meta_copy`
- Add payload-size guard and reject oversized JSON payloads before `g_strdup`.
- Keep copy/release semantics unchanged (DeepStream requires deep copy for user meta).

10. `geometry/depth_source.py::_create_job`
- Keep defensive copies (boundary-path) but expose copy-size counters to tie boundary bytes to emitted depth frames.

11. `reid/stable_id_manager.py` (new backend abstraction)
- Add `StableIDComputeBackend` contract with methods:
  - `normalize(emb_batch)`
  - `cosine_similarity(query, gallery)`
  - `topk(similarity, k)`
- Implement `CpuNumpyBackend` as parity reference and fallback.
- Implement `TorchCudaBackend` for production vector math path.

8. `reid/stable_id_manager.py` (hybrid integration)
- Route similarity/top-k path to backend adapter.
- Keep lifecycle and state transitions CPU-side.
- Add runtime counters for backend mode and fallback count.

9. `noesis/ds8_runtime.py` + stats surface
- Expose stableid backend observability keys:
  - `stableid_backend_mode`
  - `stableid_gpu_match_p50_ms`
  - `stableid_gpu_match_p95_ms`
  - `stableid_gallery_size`

## Acceptance Checks

### Static checks
1. Allowed copy sites audit:
- Command:
  - `rg -n "cpu\\(\\)\\.numpy|cudaMemcpy|memmove|np\\.array\\(.*copy=True|\\.copy\\(" noesis/pipelines/hooks.py native/noesis_pose_meta_ext.cpp geometry/depth_source.py`
- Pass criteria:
  - Copy callsites exist only in approved functions listed above.

2. No GIE mismatch fallback:
- Command:
  - `rg -n "converted_items\\[0\\]|no matching gie_id.*fallback|Fallback: process the first tensor_meta" noesis/pipelines/hooks.py`
- Pass criteria:
  - No fallback-to-first-tensor behavior remains in core path.

### Unit/integration checks
1. MapAnything strict-ID behavior
- Add/extend test: `tests/test_mapanything_postprocess.py`.
- Pass criteria:
  - When `tensor_items` IDs do not match MapAnything `gie_id`, no depth result is emitted/stored.

2. ReID extraction contract
- Add test module: `tests/test_reid_tensor_contracts.py`.
- Pass criteria:
  - Embedding extracted only when `needs_embedding` true.
  - Returned embedding is finite `float32`, 1-D, and `abs(norm - 1.0) < 1e-3`.

3. Pose single-materialization contract
- Add test module: `tests/test_pose_tensor_contracts.py`.
- Pass criteria:
  - With both feature and overlay hooks enabled, one object/frame tensor is materialized once (cache hit for second consumer).

4. Native pose payload guard
- Add test module: `tests/test_pose_meta_native_contracts.py`.
- Pass criteria:
  - Oversized payloads are rejected; valid payloads still round-trip via `extract_pose_features`.

5. StableID CPU/GPU parity
- Add test module: `tests/test_stable_id_gpu_parity.py`.
- Pass criteria:
  - Same replayed embedding stream yields identical stable-id assignment timeline on CPU and GPU backends.

6. StableID GPU backend live check
- Add/extend runtime test to assert `stableid_backend_mode == "gpu"` when CUDA is available and enabled.
- Pass criteria:
  - No silent fallback to CPU in normal runtime.

### Runtime counters (required for cutover)
1. Add counters in `hooks.py`:
- `tensor_host_copies_total{model=mapanything|reid|pose}`
- `tensor_gie_mismatch_drops_total{model=...}`
- `tensor_boundary_copy_bytes_total{path=depth_store|pose_meta}`
2. Pass criteria in DS8 smoke/soak:
- `tensor_gie_mismatch_drops_total` may increase, but no fallback processing occurs.
- `mapanything` host copies occur only in async worker context.
- `reid` host copies approximately equal extracted embeddings.
- Boundary copy bytes grow only with depth store / pose meta attach activity.

### StableID rollout env defaults
1. `NOESIS_STABLEID_GPU_ENABLED=1` when CUDA is available.
2. `NOESIS_STABLEID_GPU_DEVICE=cuda:0` by default.
3. `NOESIS_STABLEID_GPU_MIN_GALLERY=32` default crossover threshold.
4. `NOESIS_STABLEID_GPU_BACKEND=torch` initial backend.
