**Context Summary**

- Goal: Persist “RealIDs” (named people) across days and cameras while keeping current in-session StableID tracking stability. You want to label people by name and have the system recognize them later, with a lightweight, high‑leverage fine‑tuning UI that surfaces only the most impactful corrections (not a full annotation workflow).
- Current engine: Strong single‑camera stability via OSNet embeddings, stripe/multi‑scale fusion, EMA centroids, adaptive penalties (scale, brightness), ghost strictness, and spatial/color disambiguation; soft caps on active IDs and global reuse to prevent unbounded SID growth; cross‑camera handoff window and margin to preserve IDs when moving between rooms.
- Pain points to solve: Day‑to‑day identity continuity, occasional adult/adult merges, cross‑camera continuity still inconsistent, and avoiding an explosion of ephemeral IDs (keep total identities realistic for homes).
- UX intent: A “Fine‑Tune” pane that:
  - Shows a curated Uncertainty Queue (borderline cases near thresholds, cross‑camera handoff candidates, low‑margin assignments, outliers).
  - Allows batch merge/split/reassign with undo/audit.
  - Lets users curate “exemplars” per person to improve centroids while guarding against drift.
  - Surfaces “almost matches” so edge cases don’t languish unassigned.
- Design philosophy: Treat Person as the durable entity (registry), while SIDs are ephemeral session labels mapping to Persons when confidence suffices. Use a registry threshold/hysteresis stronger than same‑session matching; accept only high‑quality, diverse samples into galleries; record negative pairs to prevent repeat mistakes; maintain per‑camera normalization to aid cross‑camera handoffs.

---

**Architecture Overview**

- Person Registry
  - Stores long‑lived Person records with name, exemplars (embeddings + thumbnails), centroid, negatives, and metadata.
  - Backed by SQLite for reliability and indexing; JSON backup/export for portability.
- Session Tracking
  - Continues to use StableIDs for real‑time assignment; maps SIDs → Person IDs when registry match confidence is met over a short hysteresis.
- Matching Layers
  - Ghosts (same camera, recent disappearance).
  - Session gallery (EMA centroid, penalties).
  - Registry (global, higher threshold + hysteresis, negative pairs, per‑camera normalization, cross‑camera handoff bias).
- Controls and Caps
  - Soft caps on active SIDs per sensor + global ID reuse to keep identity count in a reasonable range.
  - No hard cap on Person IDs; recycle only SIDs, never Persons.
- Frontend Fine‑Tune Pane
  - Slide‑out panel with Uncertainty Queue, Low‑Margin Assignments, and People (exemplar curation) tabs.
  - Batch actions: merge, split, reassign, negative pair, pin/unpin exemplars.
- APIs
  - REST endpoints to drive the pane: list persons, queue items, thumbnails; merge/split/reassign; pin/unpin; undo/audit.

---

**Data Model**

- Person (persistent)
  - id: UUID
  - name: string (optional at creation)
  - created_at, updated_at, last_seen_ts
  - centroid: float[512] (EMA, persisted)
  - exemplars: list of EmbeddingRef (embedding_id, ts, sensor_id, quality, pinned: bool, path to thumbnail)
  - negatives: list of person_id or embedding_id pairs to penalize
  - per_camera: stats (brightness/white balance offsets, count) to normalize matches
  - status: enum (active, disabled)
- EmbeddingRef (persistent)
  - id: UUID
  - person_id: UUID (nullable for unknowns)
  - embedding: float[512] (optionally compressed/quantized or referenced by blob file)
  - ts, sensor_id, bbox, quality metrics (blur, size), color hist
  - thumbnail_path: string (small JPEG)
- SID Assignment (ephemeral)
  - sensor_id, ds_obj_id, sid, mapped_person_id (nullable), last_emb_ts
  - live centroid (session), last bbox/brightness/color
- NegativePair (persistent)
  - left_person_id/right_person_id or person_id vs. embedding_id, ts, reason

---

**Persistence & Storage**

- Backend: SQLite `~/.noesis/registry.db` with WAL mode; indices on person_id, ts, sensor_id.
- Blobs: Thumbnails under `~/.noesis/person_thumbs/{person_id}/...jpg`; embeddings stored as BLOBs or float arrays compressed (FP16) depending on footprint.
- Backup/Export: JSON (schema versioned) `~/.noesis/exports/registry-YYYYMMDD.json` with separate tar of thumbnails.
- Migration: Migrate from existing in‑memory galleries; seed Persons from current stable adults (user‑guided in pane).

---

**Matching Pipeline**

- Preprocessing
  - Stripe/multi‑scale fusion (already implemented).
  - EMA centroid maintained per SID and per Person.
  - Per‑camera normalization: store camera offsets (mean brightness/color shift) from confirmed cross‑camera pairs; adjust candidate embeddings or penalties at match time.
- Session Matching (unchanged core)
  - Ghost → session gallery (EMA centroid + adaptive penalties) → provisional or new SID with soft caps.
- Registry Matching (new layer)
  - Trigger: when SID has a fresh high‑quality embedding (quality gate already in place).
  - Score: cosine(embedding, person_centroid), with penalties for:
    - Large scale change (optional when mapping across cameras).
    - Color histogram distance.
    - Negative pairs (strong penalty).
  - Thresholds:
    - Registry threshold T_reg ≥ session high threshold.
    - Hysteresis: require K consistent frames (e.g., K=2) above T_reg to attach a Person ID; else remain Unknown.
  - Post‑match:
    - Attach person_id to SID; use name in OSD.
    - Candidate embedding eligible to enter exemplars only if high quality AND margin ≥ m_min OR user‑pinned later.

---

**Cross‑Camera Handoff**

- Handoff Bias
  - If a Person was seen on Cam A at time t and a near‑match appears on Cam B at t+Δ, reduce T_reg by Δ_margin if Δ within window (configurable, currently 6s/0.04).
- Active‑Here Guard (unchanged)
  - If a candidate Person is already active on this sensor, require small extra margin to avoid merging two co‑present people.
- Normalization Learning
  - From confirmed cross‑camera assignments (auto or user), update per‑camera normalization stats:
    - Maintain rolling deltas for brightness/hue between camera centroids for that Person (or globally).
    - Apply small compensation to reduce sensor bias.

---

**Soft Caps & ID Reuse**

- Per‑Sensor Soft Cap
  - Confirm new SID creation over N frames when cap reached; evict stale active locals (>grace) before creating.
  - Return provisional negative IDs for UI stability while pending.
- Global Soft Cap for SIDs (not Persons)
  - When total SIDs ≥ cap, attempt to map into an existing identity with slightly relaxed threshold; else recycle an old, fully inactive SID number (clear its gallery/state first).
- Person IDs
  - Never recycled; renameable and durable across days.
  - New Person created only on explicit enrollment/reassignment in pane or automatic creation if strongly Unknown beyond a session-defined policy.

---

**Quality & Drift Control**

- Quality Gating (in place)
  - Skip tiny/blurred crops; fuse stripes/halves; flip TTA.
- Exemplar Admission Policy
  - Auto-add only high-quality, high-margin samples up to K with diversity across camera/pose/lighting.
  - Pinning in UI overrides; pinned exemplars anchor centroid.
- Negative Pairs
  - When user rejects an assignment or reassigns, record negative pair to penalize future matches and prevent repeat confusion.
- Hysteresis
  - Registry attach requires K consistent frames above threshold, reducing flicker.

---

**API Endpoints**

- Persons
  - GET ` /api/persons` (list with stats, last_seen, pinned count, outliers)
  - POST `/api/persons` (create/rename/delete)
  - GET  `/api/persons/:id` (details, exemplars)
  - POST `/api/persons/:id/exemplars/pin|unpin` (pin/unpin embedding_ids)
- Matching & Corrections
  - GET  `/api/queue/uncertainty` (borderline, handoff candidates, low‑margin assigned, outliers)
  - POST `/api/queue/act` (merge/split/reassign/negative, with list of embedding_ids; returns updated centroids)
  - POST `/api/match/assign` (assign embedding_id → person_id)
  - POST `/api/match/negative` (record negative pair)
- Media
  - GET `/api/thumbnails/:embedding_id` (jpeg)
- Audit/Undo
  - GET `/api/audit` (recent actions)
  - POST `/api/audit/undo` (undo last N actions if safe)

---

**Frontend UX**

- Fine‑Tune Pane (slide‑out)
  - Tabs:
    - Uncertainty: curated list (borderline ±0.02 around T_reg, x‑cam handoff near miss, low‑margin assigned, outliers).
    - Recent Low‑Margin: confirm/reject quick actions for last N low‑margin assignments.
    - People: per‑person exemplars (grid of 6–9), pin/unpin, add/remove; quick merge/split controls.
  - Thumbnails:
    - Small crop (with camera/time/score margin badges).
    - Hover: show alt fusions (stripe/half) and histogram sparkline.
  - Actions:
    - Merge to Person A (multi‑select), Split selected → New Person, Reassign selected → Person B.
    - Mark Negative: “not Person X” sets negative pair.
    - Undo button (last action), history link.
  - Empty State:
    - “No items need review” with tips; show People tab to curate exemplars.

- Camera Handoff Helpers
  - “Likely same as” suggestions when a person disappears on Cam A and a candidate appears on Cam B within window; one‑tap confirm.

- Accessibility & Performance
  - Lazy load thumbnails; keyboard shortcuts for batch select/apply; clear feedback on applied changes.

---

**Metrics & Telemetry**

- Matching
  - attach_rate, reassign_rate, low_margin_count, xcam_handoff_success
- Quality
  - exemplars_added_auto/pinned, drift_prevented_by_negatives
- Queue
  - backlog size, time_to_resolution, user_actions/day
- Caps
  - active_ids_per_sensor, total_sid_count, recycled_sid_count
- Errors
  - registry_load/save, db_ops_latency

---

**Privacy & Security**

- On‑device storage; opt‑in enrollment; easy delete/export per Person.
- Audit trail of corrections; undo.
- Optional encryption at rest for registry DB and thumbnails.

---

**Implementation Phases**

- Phase 1: Registry & Matching Backbone
  - Add SQLite schema and DAO; load/save on startup/shutdown (with autosave interval).
  - Maintain in‑memory centroids; implement registry matching + hysteresis; integrate with current pipeline.
  - Add per‑camera normalization scaffold; record confirmed cross‑camera pairs.
  - Expose basic REST endpoints for persons and thumbnails.
- Phase 2: Uncertainty Queue & Corrections
  - Curate queue (borderline band, x‑cam near misses, low‑margin assigned, outliers via Mahalanobis distance).
  - Endpoints for queue list + actions (merge/split/reassign/negative, pin/unpin).
  - Apply actions → update registry, recompute centroids, record audit; broadcast UI updates.
- Phase 3: Fine‑Tune Pane (UI)
  - Implement slide‑out pane with the three tabs, thumbnails, multi‑select, batch actions, undo, loading states.
  - Wire quick confirm/reject for low‑margin assignments.
- Phase 4: Cross‑Camera Improvements
  - Implement handoff suggestions; learn per‑camera normalization deltas from confirmed cross‑cam pairs and apply small compensation in scoring.
  - Tune handoff window/margin with field data.
- Phase 5: Stability & Hygiene
  - Exemplar admission policy (auto vs. pinned); cap K with diversity selector.
  - Negative pair integration and penalties in scoring.
  - Soft caps: verify SID pool recycling, telemetry, and safeguards.
- Phase 6: Ops & Rollout
  - Metrics dashboards and logs; backup/export tools.
  - Privacy controls (delete/export per Person).
  - Documentation and migration guide.

---

**Testing Plan**

- Unit tests:
  - DAO, centroid updates, negative pair penalties, hysteresis.
- Scenario tests:
  - Single‑camera occlusion; cross‑camera handoff within/outside window; merges and splits via pane; negative pair preventing repeat confusion.
- Load tests:
  - Registry with thousands of embeddings; query latency; thumbnail serving.
- UX tests:
  - Queue sizing, selection/undo flows, accessibility.

---

**Edge Cases**

- Similar adults (uniforms): negative pairs + color penalty + spatial guard; require higher margins for merging.
- Identical clothing across days: rely on exemplars + user confirmation; avoid auto‑adding low‑margin samples.
- Kids growth/clothing change: EMA centroids with diverse exemplars mitigate drift.
- Party bursts: soft caps prevent ID explosion; queue surfaces the most valuable corrections.

---

**Future Work**

- Vector index (Faiss/Annoy) for larger registries.
- Online calibration per camera (learned normalization).
- Device‑level encryption, multi‑user accounts.
- Semi‑supervised selection of informative samples for the queue.

---

**Why We’re Doing This (Intent & Rationale)**

- Persisted Identity: You want consistent, nameable identities across days and cameras. Treating Person as a durable entity with a registry separates “who” from ephemeral tracking SIDs and gives you a proper place to store knowledge about individuals.
- Efficient Human Input: The fine‑tune pane focuses on high‑leverage actions (uncertainty, low‑margin, cross‑camera near misses) and avoids turning users into annotators. Pinning exemplars and recording negative pairs provide strong guardrails against drift with minimal effort.
- Robust Matching: By layering ghost/session/registry matching with EMA centroids, quality gating, disambiguation (scale/brightness, spatial/color), and hysteresis, we reduce flips and merges while enabling confident auto‑attachment and smoother cross‑camera handoffs.
- Practical Constraints: Soft caps on SIDs (per sensor and globally) prevent identity explosion in typical homes (1–4 people standard, 5–7 upper bound), while global reuse keeps the numeric ID pool tidy without affecting durable Person IDs.
- Cross‑Camera Continuity: A recency‑aware handoff window with small threshold relaxation, plus per‑camera normalization learned from confirmed pairs, brings consistency when walking from room to room.
- Safety & Control: On‑device storage, undo/audit, export/delete, and negative pairs ensure you can correct mistakes and maintain data hygiene without risking privacy or drift.
