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

## D12 — Resident prior is bounded and cannot create evidence

- **Decision:** Resident candidates may receive a bounded ranking prior only
  after quality, model, topology, absolute appearance, and ambiguity eligibility
  gates. The prior can select among already-admissible candidates but cannot
  convert unknown/rejected evidence into a resident decision.
- **Rationale:** Residents are the common case in this home, but forcing a known
  name onto a visitor is a more damaging failure than temporarily reporting an
  enrolled resident as unknown.
- **Date:** 2026-07-09

## D13 — Enrollment state is durable; visitor slots are generational

- **Decision:** Maintenance and generic SID reuse may never purge enrolled
  resident galleries. Reused visitor numeric slots carry a generation now and
  migrate to durable session UUIDs in identity schema v2. Gallery contents, not
  cached enrollment counters, are authoritative for identity health.
- **Rationale:** Numeric compatibility IDs are not durable subject identity, and
  stale counts or recycled slots must not present false continuity.
- **Date:** 2026-07-09

## D14 — Identity v2 is one process-owned whole-frame service

- **Decision:** DS8, V3DT, and DS9 construct one shared identity-v2 service per
  runtime run. It owns one SQLite store, open-set runtime, and frame coordinator;
  hooks detach all server primitives from a source frame and invoke the
  coordinator exactly once rather than resolving per object.
- **Rationale:** Per-object greedy calls are order-sensitive and cannot enforce
  simultaneous one-to-one assignment. Shared product logic prevents DS8/DS9
  behavior drift while keeping SDK wrappers adapter-local.
- **Date:** 2026-07-10

## D15 — Shadow by default; calibrated evidence gates authority

- **Decision:** `NOESIS_IDENTITY_V2_MODE` is
  `disabled|shadow|authoritative` and defaults to `shadow`. Authoritative startup
  requires a valid versioned scoring artifact whose actual bytes are hashed;
  heuristic calibration is never public identity authority.
- **Rationale:** A new resolver needs household replay/live evidence before it
  may replace working labels, and a mode flag must not turn uncalibrated scores
  into truth.
- **Date:** 2026-07-10

## D16 — Enrollment consumes exact server evidence only

- **Decision:** The enrollment proposal API accepts a server observation key and
  naming intent, never an embedding. The key binds runtime run, camera, tracker,
  frame, content-addressed active model profile, and exact float32 evidence in a
  bounded one-use cache.
- **Rationale:** Client-supplied biometrics are forgeable and stale track/SID
  selectors can enroll the wrong person. Exact immutable evidence makes proposal
  and confirmation replay/staleness safe.
- **Date:** 2026-07-10

## D17 — Overlap sharing requires fresh geometry and appearance proof

- **Decision:** Loading topology creates coordinator edges, not blanket sharing
  permission. Each identity-specific permit requires the declared camera pair,
  contemporaneous timestamps, valid world points within the configured distance,
  and appearance similarity above the configured threshold. Permits are not
  sticky and camera-local proof disappears with the track.
- **Rationale:** Topology alone says two cameras can overlap, not that two current
  detections are the same person.
- **Date:** 2026-07-10

## D18 — Authoritative v2 never mutates or displays legacy identity

- **Decision:** In authoritative mode hooks skip legacy StableID assignment,
  copresence, maintenance, dwell, and transition work until the complete v2
  frame resolves. SID-dependent dwell/transition state then uses only the v2
  compatibility SID. An evidence-backed rejection remains `unknown`; missing
  embedding evidence without a continuity hold is `provisional`. The first
  one-shot metadata walk stamps neutral `#XX` rather than reusing legacy state;
  D21 defines the separate safe downstream restamp once resolution is complete.
- **Rationale:** Running both identity owners mutates incompatible state and can
  attribute analytics to the wrong person. Retaining object wrappers for a
  second OSD pass is unsafe, so video may show a resolved identity only through
  a fresh downstream metadata callback and an exact decision join.
- **Date:** 2026-07-10

## D19 — Scoring authority requires disjoint held-out evidence

- **Decision:** Superseded by D23. The original v1 artifact counted frame-level
  outcomes and allowed only three independence groups, so it could not support
  its claimed 1% safety rate when adjacent frames were correlated. Version 1
  labels, datasets, and authority artifacts now fail closed.
- **Rationale:** A model-bound policy blob proves configuration compatibility,
  not open-set quality. Keeping this superseded entry records why frame counts
  and a three-group minimum were removed rather than silently reinterpreted.
- **Date:** 2026-07-10

## D20 — Browser enrollment and migration review are biometric-free

- **Decision:** Menon owner enrollment consumes only current server observation
  keys through propose/confirm and an action-bound confirmation. Browser APIs
  expose score-capture status and a sanitized legacy migration review, never
  evidence files, embeddings, gallery vectors, source paths, or an apply route.
- **Rationale:** Biometric material belongs inside the Noesis appliance process;
  enrollment intent and migration adjudication do not require it in JavaScript.
- **Date:** 2026-07-10

## D21 — Authoritative OSD uses an exact downstream metadata join

- **Decision:** DS8, V3DT, and DS9 attach the installed Service Maker
  `BatchMetadataOperator` as an explicit tiler-sink probe. It joins a bounded
  `(camera, frame, tracker)` cache populated after whole-frame resolution,
  touches each fresh object wrapper once, and stamps `#XX` on every miss or
  mismatch.
- **Rationale:** The analytics iterator's wrappers are transient and unsafe to
  retain, while a downstream tiler-sink callback is ordered after resolution
  and still has source metadata. This produces same-frame authoritative video
  labels without a second pass over dangling wrappers or stale legacy state.
- **Date:** 2026-07-10

## D22 — Identity evidence is bounded private rolling state

- **Decision:** Score-only evidence uses create-private-or-validate-private
  paths, complete newline-terminated records, a sequence/previous-event hash
  chain, monotonic timestamps, and deterministic contiguous-prefix retention
  bounded by record count, bytes, and age. A private checkpoint binds the
  retained predecessor, head, tail, count, and bytes. Calibration artifacts,
  datasets, labels, and migration review inputs are owner-only single-link
  files; existing permissions are never changed automatically.
- **Rationale:** Score-only data is still household identity evidence. Unlimited
  capture can exhaust the appliance, while chmod/follow-link behavior can cross
  trust boundaries. Deterministic pruning preserves current-record tamper
  verification and makes retention observable without silently rotating to an
  alternate path.
- **Date:** 2026-07-10

## D23 — Authority separates generic benchmark evidence from local verification

- **Decision:** Calibration authority is `noesis.identity.open_set_calibration`
  v2 and requires two model-profile-identical strata. A provenance-locked,
  subject-disjoint benchmark holdout carries the generic claim using at least
  300 challenge-covered resident people and 300 challenge-covered unknown
  people. Outcomes are worst-case across every encounter for each person, and
  exact one-sided 95% Clopper-Pearson upper bounds must be no greater than 1%
  for false acceptance and misidentification. Benchmark train requires 50/50
  challenge-covered resident/unknown people but carries no holdout confidence
  claim. A smaller household stratum (10/10 train and 20/20 holdout
  known/unknown encounters) verifies local domain behavior with zero false
  accepts, zero misidentifications, FRR at most 35%, and zero harmful or
  rejection-rescuing prior changes. Household tuning may only keep or raise
  benchmark rejection gates and may not alter its calibration or resident
  prior.
- **Decision:** Observations are deterministically clustered by capture
  session/run/source/camera/tracklet/five-second bucket, capped at 300 per unit,
  and thinned to one center representative for at most eight fit units per
  encounter. Fit weights are encounter-balanced. Authority metrics aggregate
  all observations at encounter worst case, then benchmark outcomes at truth-
  person worst case. Multiple frames, windows, or encounters for one person
  never increase the benchmark confidence denominator. Session, run, and
  encounter IDs may not cross train/holdout; subject-disjoint additionally
  separates private person keys, including unknown people. Household residents
  may repeat across visits, but household unknown people remain split-disjoint.
- **Decision:** Every qualifying challenge has non-empty hard-allowed resident
  and visitor competition plus an impostor; training also requires the genuine
  candidate. Zero-exemplar/blocked candidates are excluded. Evidence rows bind
  a runtime-derived semantic-profile digest. The scorer-only artifact carries
  the minimum supported gallery envelope and requires independent artifact-byte
  and active-semantic pins; runtime gallery growth fails before mutation.
- **Decision:** Known truth in both calibration strata is resident truth.
  Visitor candidates must provide competition, but visitor continuity cannot
  substitute for the household resident-recall gate.
- **Decision:** Benchmark identities must be licensed real people and household
  identities must be owner-consented real household people. Synthetic or
  generated identity truth cannot support the real-person safety claim.
- **Rationale:** Large-N public/benchmark evidence can make the generic 1%
  statement statistically meaningful without asking one homeowner to label
  hundreds of owner-labeled home encounters. Local evidence then detects camera/home domain shift
  while being mathematically unable to relax or rescue the globally supported
  rejection policy. Reporting the local wide bounds prevents a small home
  sample from masquerading as generic safety evidence.
- **Date:** 2026-07-10

## D24 — Scorer calibration cannot authorize the public runtime

- **Decision:** The model semantic profile hashes the actually loaded
  `noesis_reid_meta_ext` binary plus the Python normalization, gallery-similarity,
  and open-set scoring implementations. Public `authoritative` mode additionally
  requires a separately byte-pinned `noesis.identity.authority_cutover` v1
  artifact. That artifact binds the exact scorer artifact, model and semantic
  profile, DS8/DS9 executable authority profile, topology, camera set, and two
  distinct owner-private reports: whole-frame coordinator replay and an
  occupied-scene runtime gate. Both report bytes are verified at startup.
- **Rationale:** A scorer-only calibration cannot prove joint assignment,
  overlap, OSD, adapter, or occupied-scene behavior. Static semantic labels also
  cannot detect a rebuilt native extractor. Separating these scopes prevents an
  otherwise valid calibration artifact from silently promoting unreviewed
  runtime behavior.
- **Date:** 2026-07-10

## D25 — Tracker continuity is a hard open-set constraint

- **Decision:** After a resident or visitor subject is accepted for one
  camera-local tracker, every different gallery subject is hard-masked for the
  lifetime of that tracker state. The accepted subject still has to pass the
  ordinary quality, absolute appearance, calibrated-confidence, ambiguity,
  topology, and exclusivity gates; otherwise the fresh row is `unknown`.
  Alternative evidence never causes a direct subject switch. A new subject may
  be acquired only after the configured fresh-evidence gap expires the complete
  tracker state.
- **Decision:** `embedding_present` means that the process-owned identity-v2
  adapter extracted a valid embedding for that exact public frame. Legacy
  gallery/cache diagnostics cannot set it. When private shadow capture is
  enabled, a true value is associated with the exact observation key and the
  complete persisted sequence/model/dimension provenance triad.
- **Decision:** Private score evidence records the exact post-hard-mask
  candidate rows consumed by the resolver. It may not take a second gallery
  snapshot that loses tracker-continuity/copresence constraints or races
  retention; rejected candidates retain their raw similarity plus the exact
  hard-constraint reason for deterministic replay.
- **Decision:** DS9 applies the same default-on household StableID admission as
  DS8. Household mode requires `auto_merge_enabled=false` and
  `allow_multi_zone_active=false` even when old deployment environment knobs
  request otherwise; failure to construct that policy aborts StableID
  admission rather than silently running the legacy policy.
- **Rationale:** A sealed occupied DS9 session showed Hungarian contention
  assign a tracker its second-choice visitor and then replace it with the local
  winner 0.683 seconds later. The same run inherited stale
  `embedding_present=true` from legacy diagnostics without exact persisted
  provenance and exposed a DS9-only legacy auto-merge. These are truth and
  parity failures, not thresholds to weaken.
- **Date:** 2026-07-11
