# Household Identity — Performance & Zero-Copy

Latency and memory rules for all phases. If a change violates these, stop and
record a decision — do not silently add CPU paths.

## Pipeline authority

```text
NVMM frames
  → PGIE / tracker / ReID SGIE (GPU)
  → NvDsInferTensorMeta on object (device/host per DS config)
  → native/noesis_reid_meta_ext.cpp unwrap + L2 normalize
  → hooks pass float embedding into StableIDManager
  → GPU batch matmul for gallery similarity when candidate count ≥ gpu_min_gallery
```

**Forbidden on DS8 path**
- Re-enabling `use_extractor=True` / torchreid crop path in runtime
- New appsink branches for ReID crops
- Per-frame full-frame CPU copies for identity
- Unbounded embedding extracts per frame

## Budgets (defaults; env-tunable)

| Knob | Default | Notes |
|------|---------|-------|
| ReID SGIE `secondary-reinfer-interval` | 12 (Swin) | Keep; new objects still infer immediately |
| `NOESIS_REID_EMBED_INTERVAL_S` | 1.0 | Manager-side refresh cadence |
| `NOESIS_REID_EMBEDS_PER_FRAME_MAX` | `max(4, visible_persons)` capped at 8 | Phase 0: raise from 2 carefully |
| Gallery GPU matmul threshold | 32 → lower to 8 in household mode | Small resident galleries still benefit from vectorized path |
| Pose into StableID | only when pose quality gates pass | Avoid extra work on junk pose |
| World overlap check | O(active tracks) using cached footpoints | No new projection |

## Zero-copy checklist for implementers

1. Native reid ext returns a contiguous float32 vector; avoid Python list materialization.
2. Prefer `np.asarray(..., dtype=np.float32)` without copy when already float32/C-contiguous.
3. Gallery centroids stored L2-normalized once; do not renormalize every compare if invariant holds.
4. Batch similarity: one matmul `[T,D] @ [D,N]` for T tracklets vs N identities per resolve tick.
5. Do not deepcopy galleries in hot path; lock scope must be minimal.
6. Autosave gallery off hot path (existing interval OK); never save per frame.

## Detection-wake interaction

Identity work runs when people are present. Coordinate with
`plans/DS8/ds8_yolo26_performance_optimization_plan.md`:

- Do not undo pose/depth cadence budgets while raising ReID extract cap.
- Measure before/after with existing live health / detection-wake counters when
  available.
- Target: no more than ~+10% detection-wake CPU vs pre-change baseline at 3
  cameras / 2 people, after Phase 0.

## Backbone cost

| Model | Dim | Engine present | Role |
|-------|-----|----------------|------|
| Swin-Tiny TAO | 256 | yes | **Default** |
| OSNet-IBN | 512 | yes | Lean fallback profile |
| SOLIDER-Swin | TBD | no | Optional upgrade if Swin closed-world still confuses residents |

Prefer Swin default: already selected in `config/infer.yaml`, smaller embedding
dim (faster gallery matmul) than OSNet-512.

## Latency targets (home, 3 cams, ≤4 people)

| Stage | Target |
|-------|--------|
| Native emb extract / object | < 0.3 ms host |
| Resolve tick (all active tracklets) | < 2 ms typical |
| Extra overlap geometry checks | < 0.2 ms |
| End-to-end identity assign in hook | stay within existing tracking publish budget |
