# Phase 2 — SOTA Matching

**Goal:** Improve discrimination and association quality under home conditions
(clothing similarity, lighting, overlap) while staying zero-copy / low-latency.

**Depends on:** Phase 1 gate G1

## Scope

### In
- Confirm Swin-Tiny as default ReID SGIE; portable paths; engine rebuild docs
- Quality-gated gallery updates (size, blur, occlusion proxy, aspect)
- Multi-exemplar galleries (outfit / appearance modes) per identity
- Global assignment (MNN or Hungarian) per resolve tick
- Spatiotemporal handoff using topology + BEV continuity
- Optional SOLIDER export/optimize stretch if Swin still confuses residents

### Out
- Enrollment product UX (Phase 3)

## Backbone plan

### Default (implement now)
1. Ensure `config/infer.yaml` and household profiles use
   `pipelines/config_infer_secondary_reid_swin.ini`.
2. Remove hardcoded `/home/mayor/...` from OSNet ini (portable relative paths).
3. Document engine rebuild in `models/REID_ENGINE_REBUILD_COMMANDS.txt`.
4. Tune thresholds for 256-d Swin space (do not assume OSNet-512 thresholds).

### Optional upgrade (only if G1 metrics show appearance confusion)
1. Download/export SOLIDER-Swin ONNX (script under `scripts/`).
2. Build TensorRT FP16 engine, batch 16, 256x128.
3. Add `pipelines/config_infer_secondary_reid_solider.ini`.
4. A/B on recorded clips before flipping default.

Do **not** block Phase 2 core on SOLIDER if Swin is already default.

## Quality-gated gallery

Accept embedding into gallery only if:
- crop_h ≥ min_crop_h
- laplacian/blur proxy ≥ min (if available) OR SGIE object size gates
- not flagged heavy occlusion if mask/pose available
- identity_state already visitor/resident (not provisional junk)

Maintain per-identity exemplar clusters (max K, e.g. 8) via simple online
clustering on cosine distance; compare query to best exemplar + centroid.

## Global assignment

Each analytics tick (or coalesced resolve):
1. Collect unresolved tracklets with embeddings
2. Score vs candidate identities (residents + active/recent visitors)
3. Apply exclusivity + overlap constraints as hard masks
4. Solve MNN or Hungarian on score matrix
5. Unassigned tracklets stay provisional / mint visitor per Phase 1 rules

Keep hot path vectorized (see `performance.md`).

## Spatiotemporal handoff

- If identity released on cam A and appears on cam B within handoff window:
  prefer that identity with relaxed appearance threshold **only if** camera
  topology adjacency exists OR historical transition prior.
- Overlap dual-active remains Phase 0 permit logic.

## Files to touch

- `pipelines/config_infer_secondary_reid_*.ini`
- `reid/stable_id_manager.py` — quality gates, exemplars, assignment
- `scripts/export_reid_*.py` / download scripts as needed
- `tests/test_stable_id_assignment.py`
- `performance.md` evidence notes

## Action checklist

- [ ] Harden Swin default + portable OSNet paths
- [ ] Quality-gated gallery updates + multi-exemplar storage
- [ ] Global MNN/Hungarian assignment with constraint masks
- [ ] Topology-aware handoff relaxation
- [ ] (Optional) SOLIDER export + engine + A/B
- [x] Unit tests for assignment conflicts and quality rejects
  _2026-07-08: `test_stable_id_assignment.py`, `test_stable_id_gallery_quality.py` (16 passed with gallery regression)._
- [ ] Perf spot-check vs Phase 0 baseline

## Risks

- Hungarian every frame at high person count — keep N small (household) and
  early-out when T=1.
- Over-strict quality gates → empty galleries → more visitors. Tune with metrics.
