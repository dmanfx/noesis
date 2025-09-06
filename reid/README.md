Re-Identification (ReID) and StableID

Overview
- Backbone: OSNet (`osnet_ibn_x1_0`) via torchreid, MSMT17 weights (classifier head dropped).
- Extractor: `reid/embedding_extractor.py` initializes torchreid either via FeatureExtractor or build_model fallback with safe device handling.
- Manager: `reid/stable_id_manager.py` assigns a stable, global ID using embeddings, ghosts, and a gallery.

Key Features
- Crop expansion and quality gating to reduce drift on tiny/blurred crops.
- Flip test-time augmentation.
- Partial feature fusion (PCB-like): multi-stripe and multi-scale crop embeddings fused and L2-normalized.
- Adaptive penalties for large scale/brightness shifts.
- Additional disambiguation: spatial distance penalty (when a candidate ID is already active on the same sensor) and color histogram mismatch penalty.
- EMA centroid per identity for robust comparisons over time.
- Anti-merge guard: when a stable ID is already active on the same sensor, require a small extra similarity margin before allowing a new track to adopt that ID.
- Ghost registry per camera to reattach after short occlusions.
  - Age-adaptive ghost matching: for older ghosts, a slightly higher similarity threshold is required to reduce accidental reuse by new people.

Important Config Flags (config.py -> models)
- `REID_ENABLED`: master toggle.
- Model: `REID_MODEL_PATH`, `REID_MODEL_NAME`, `REID_IMAGE_SIZE`.
- Thresholds: `REID_COS_SIM_THRESHOLD` (same-cam), `REID_COS_SIM_HIGH_THRESHOLD` (gallery).
- Update cadence: `REID_EMBED_INTERVAL_S`.
- Robustness: `REID_CROP_EXPAND`, `REID_TTA_FLIP`, `REID_MIN_CROP_H`, `REID_MIN_LAPLACIAN`.
- Adaptive: `REID_ADAPTIVE_PENALTY`, `REID_SIZE_PENALTY_ALPHA`, `REID_BRIGHTNESS_PENALTY_BETA`.
- Disambiguation: `REID_SPATIAL_PENALTY`, `REID_SPATIAL_PENALTY_DELTA`, `REID_COLOR_PENALTY_GAMMA`.
- Partial fusion: `REID_STRIPE_FUSION`, `REID_STRIPE_COUNT`, `REID_MULTI_SCALE_CROPS`.
- Smoothing: `REID_EMA_ALPHA`.
- Anti-merge: `REID_ACTIVE_ID_GUARD_STRICT`, `REID_ACTIVE_ID_GUARD_MARGIN`.
- Ghost strictness: `REID_GHOST_STRICT_AGE_S`, `REID_GHOST_EXTRA_MARGIN`.

Pipeline Wiring
- DeepStream pipeline constructs `StableIDManager` with config flags (see `deepstream_video_pipeline.py`).
- Embeddings are computed in the analytics probe using the latest GPU JPEG per stream.

Defaults tuned for partial occlusion
- Lower same-cam sim threshold (`0.65`) to reduce misses.
- Crop expand `0.12`, flip TTA enabled, stripe fusion (3 stripes) and half-crops enabled.
- EMA centroid `alpha=0.20` for quick adaptation through chandelier occlusions.
- NvDCF tracker tuned: reduced search region padding, higher visibility threshold (see `pipelines/config_tracker_nvdcf_batch.yml`).

Notes
- Classifier weights are intentionally dropped; ReID uses 512-D features pre-classifier.
- If torchreid FeatureExtractor module is missing, we fall back to `build_model` path automatically.
