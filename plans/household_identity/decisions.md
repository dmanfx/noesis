# Household Identity — Design Decisions

Locked decisions for implementers. Update this file when a decision changes;
also add a short entry to `plans/DS8/ds8_design_decisions.md`.

## D1 — Closed-world residents + ephemeral visitors

- **Decision:** Split identity into `resident` (enrolled, stable) and `visitor`
  (ephemeral TTL). Public IDs for residents are small and sticky. Visitors use a
  separate numeric range or typed ID and recycle aggressively.
- **Rationale:** User goal is 4 main inhabitants + ≤15 people short-term. Open-world
  monotonic SIDs cannot meet that validation bar.
- **Date:** 2026-07-08

## D2 — Global exclusivity with FoV-overlap permit

- **Decision:** Default: one SID active on at most one camera. Exception: if the
  camera pair is marked overlapping in topology **and** world/BEV positions are
  consistent (distance ≤ threshold, optional velocity gate), the same SID may be
  active on both cameras (true co-visibility).
- **Rationale:** Kitchen and family-room share FoV. Blind `allow_multi_zone_active=True`
  caused false shares; blind exclusivity would break legitimate dual-view.
- **Date:** 2026-07-08

## D3 — Disable pressure auto-merge

- **Decision:** Turn off auto-merge-as-cleanup (`auto_merge_enabled` default off for
  household mode). Keep suggest-merge API for human confirmation. Manual enrollment
  merges remain.
- **Rationale:** Live aliases showed 504 mappings / 222→358 — auto-merge glued
  fragmentation instead of fixing matching.
- **Date:** 2026-07-08

## D4 — Provisional IDs are not public permanent IDs

- **Decision:** Tracks without strong embedding evidence emit `identity_state=provisional`
  (or equivalent) and must not allocate a permanent resident/visitor slot until
  confirmation gates pass.
- **Rationale:** Weak first crops and missing embeddings are the primary mint drivers.
- **Date:** 2026-07-08

## D5 — ReID backbone: Swin-Tiny now, SOLIDER optional

- **Decision:** Canonical DS8 ReID SGIE remains / hardens
  `pipelines/config_infer_secondary_reid_swin.ini` (TAO Swin-Tiny, 256-d, FP16 engine
  already present). OSNet remains available as a lean profile, not default.
  Optional Phase 2 stretch: export SOLIDER-Swin if quality gaps remain after
  closed-world + geometry land.
- **Rationale:** Swin engine is already built (`models/engines/reid_swin_tiny_...`)
  and selected in `config/infer.yaml`. Avoid download/optimize churn unless metrics
  demand it. SOLIDER is the documented SOTA-adjacent upgrade path.
- **Date:** 2026-07-08

## D6 — Pose is secondary evidence, must be wired

- **Decision:** Pass pose features/quality from analytics hooks into
  `StableIDManager.update()`. Pose boosts / pose-only match only when quality gates
  pass; never overrides a hard exclusivity reject.
- **Rationale:** Manager already implements pose fusion; hooks currently omit it.
- **Date:** 2026-07-08

## D7 — Geometry from existing world/BEV, not a new CPU branch

- **Decision:** Overlap permits consume already-computed track world/footpoint
  fields from the analytics processor. Do not add new appsink/CPU decode paths.
- **Rationale:** GPU-first policy; world points already exist for BEV.
- **Date:** 2026-07-08

## D8 — Archive poisoned identity state on cutover

- **Decision:** On household-mode enable, archive
  `~/.noesis/reid_gallery.npz`, `reid_aliases.json`, `sid_pool.json` (and any
  configured overrides) to a timestamped backup dir, then start clean. Provide an
  explicit env/flag to opt out for debugging only.
- **Rationale:** Current gallery/aliases encode thousands of false fragments.
- **Date:** 2026-07-08

## D9 — Assignment is global per tick, not greedy per track

- **Decision:** Phase 2 replaces pure greedy gallery hits with mutual-nearest /
  Hungarian assignment across active unresolved tracklets vs candidate identities,
  subject to exclusivity + overlap constraints.
- **Rationale:** Greedy best-hit causes ID theft under similar clothing.
- **Date:** 2026-07-08

## D10 — Names bind to resident UUID, not raw SID integers

- **Decision:** Human aliases (`Mayor`, etc.) bind to a stable resident UUID /
  enrollment record. Numeric `stable_id` remains the wire ID but may be remapped
  only through the enrollment table.
- **Rationale:** Integer SIDs will still churn for visitors; names must not.
- **Date:** 2026-07-08

## D11 — Performance budgets are first-class

- **Decision:** See `performance.md`. Default embedding extract budget scales with
  visible person count up to a hard cap; gallery match stays GPU matmul when
  candidate count warrants; no torchreid extractor on DS8 path.
- **Date:** 2026-07-08
