# DS8 Design Decisions Log

Use this file to record non-trivial design choices made during the DS8 migration. Each entry should be short and reference any relevant docs or code.

## Template

- **Date:** YYYY-MM-DD
- **Author:** (Codex / human)
- **Area:** (e.g., Pipeline, Telemetry, REST, Metadata)
- **Decision:**
- **Rationale:**
- **References:** (docs URLs, file paths, etc.)

## Entries

- **Date:** 2026-08-11
- **Author:** Codex
- **Area:** Room-walk fusion, static-world alignment, and Scene Prior promotion
- **Decision:** Use the consistency-gated, prior-conditioned MapAnything + DA3
  reconstruction with the DA3 trajectory as pose carrier for approved room
  priors. Keep reconstruction inputs phone-only; use the calibrated static
  reconstruction afterward as world-frame alignment authority and independent
  validation evidence. Do not add the static RGB/depth frame to the joint model
  batch because the controlled Living Room variant did not improve quality.
- **Decision:** When forward-visibility evidence proves that an imported static
  target cloud has the local-X/Z half-turn convention but the calibrated camera
  pose is valid, rotate only the alignment working copy of the target cloud 180
  degrees about the calibrated camera center and world-up axis. Preserve the
  deployed camera calibration and record the correction in the alignment
  report; fail ambiguous cases.
- **Decision:** Publish approved geometry as an immutable 2.5 cm Scene Prior in
  `shadow` mode. The phone reconstruction supplies static room evidence; live
  producer-owned tracking and its calibrated projection remain authoritative.
- **Decision:** Reuse the Depth drawer's established Obstacles, Heightfield,
  Point cloud, and Visible floor modes for promoted room evidence instead of
  adding fusion/prior-specific duplicate subtabs. Select the dedicated
  fixed+phone obstacle, surface, point, and floor-support products per view;
  fall back to the conditioned prior and then the live cache only when the
  stronger per-view product is unavailable. All four views use camera-right
  toward screen-right and camera-forward toward screen-up.
- **Rationale:** The conditioned fusion produced the cleanest furniture and
  floorplan representation while retaining the best phone-walk internal and
  held-out consistency. Separating phone reconstruction from static alignment
  prevents circular validation, and correcting reversed target evidence at its
  own boundary avoids corrupting a valid global camera pose. Reusing stable UI
  representations keeps comparisons meaningful while allowing better evidence
  to improve their geometry without multiplying controls by data source.
- **Validation:** Family Room scan `20260810-215847-571c6efe` passed alignment
  after target forward visibility improved from 0% to 100% with the recorded
  half-turn correction. The selected fusion measured 3.35 cm internal and 4.40
  cm held-out median reprojection error, 12.25 cm/9.37 cm static source/target
  medians, and 12.94 cm fixed-camera depth delta. Scene Prior
  `sceneprior_family-room_20260811T015847Z_b2e023e59271` was promoted as
  deployment `deploy-20260811-family-room-conditioned-prior-v2-ds9`; live REST
  readback returned its exact 1,358,760-byte GLB with SHA-256
  `df980fa615db93f1c9f60dcae52f7acedb70646f42375e1786fb024aca4b325d`.
- **References:** `tools/mapanything_phone_scan/README.md`,
  `docs/scene_prior_v1.md`, `tools/mapanything_phone_scan/alignment.py`,
  `tools/mapanything_phone_scan/build_conditioned_scene_prior_bundle.py`.

- **Date:** 2026-08-08
- **Author:** Codex
- **Area:** Mosaic H.264 transport, WebRTC lifecycle, and quality defaults
- **Decision:** The canonical dashboard path is one GPU mosaic encode followed
  by byte-stream, AU-aligned SHM fanout: `sink_tee →` bounded leaky raw queue
  `→ nvvideoconvert → noesisforceidr → nvv4l2h264enc → h264parse →` bounded
  non-leaky AU queue `→ shmsink`. One process-wide feeder reads
  `shmsrc → h264parse → appsink` and copies complete AUs to bounded per-peer
  `appsrc → rtph264pay →` bounded non-leaky RTP queue `→ webrtcbin` pipelines.
  RTSP is disabled by default and remains optional tooling only. The active
  3840x720 profile uses 12,000 kbps CBR and `iframeinterval=10`,
  `idrinterval=10`; the former RTSP-prefixed H.264 keys and component names are
  retired.
- **Rationale:** This removes localhost UDP backlog, RTSP re-consumption, and
  double RTP packetization from browser delivery. Only the raw pre-encode queue
  is deliberately leaky, protecting analytics without dropping compressed
  frames mid-GOP. Slow-peer admission rejects a newly arriving whole AU only
  for that peer, requests a fresh IDR, and never drops individual RTP packets.
  Twelve Mbps, rather than the proposed eight-Mbps starting point, gives the
  3840x720/30 composite additional headroom for three moving scenes, OSD edges,
  and the short GOP; it remains a measurable starting floor, not a substitute
  for occupied-scene appliance acceptance.
  The feeder proves the SHM socket, PLAYING state, and first AU before startup
  succeeds; unexpected feeder error/EOS is runtime-fatal. Peer rebuild and
  shutdown serialize against AU push, gateways do not mutate the WebSocket
  registry while rebuilding, and the signaling server is the sole strong owner
  of live/retiring slots. A gateway bus error or unexpected EOS retires and
  stops only that slot; failed factory registration also stops the unregistered
  pipeline. RTP output moves from the drain to `webrtcbin` only under the
  blocking-pad probe; failure does not fall back to an unsafe direct relink.
  Timestamps are regenerated in each peer clock domain, and all feeder/gateway
  callbacks share one ref-counted default GLib context driver.
- **Validation:** Focused bridge/gateway, topology, runtime-wiring, canary, and
  plugin tests cover queue bounds, path ownership, startup failure, peer reset,
  shutdown, config parity, and decoded-media gating. A local NVIDIA encoder
  probe produced IDR NALs at frames 0/10/20/30/40 with `10/10`; the rejected
  `10/1` setting produced an IDR for all 45 frames. With the natural GOP set to
  `100/100`, a `noesisforceidr` request after output frame 14 was acknowledged
  by the trigger's monotonic property contract and NVENC emitted frame 15 as the only
  additional non-delta frame, proving that late-peer recovery does not depend
  on RTSP. Local end-to-end signaling tests carried synthetic AU-aligned H.264
  through `shmsink`, the shared feeder, the actual `WebSocketServer`, and one
  and then two simultaneous gateways; the single peer decoded 158/159 received
  RTP packets and both concurrent peers decoded 159/160. Live-camera dashboard
  transport acceptance on 2026-08-09 deployed
  `deploy-20260809-mosaic-shm-v2-ds9` on top of the retained semseg Small/Large
  v12 work. The live SHM stream delivered 240 complete AUs in 9.822 seconds
  (24.43 fps, 9.82 Mbit/s, 24 keyframes), one peer decoded 307 frames in 10
  seconds, and two simultaneous peers decoded 221 and 220 frames. All six
  appliance units remained active, with no listener on UDP 5400 or RTSP 8554
  and no severe media signature in the startup/peer lifecycle log. The
  occupied-person visual motion check described in `docs/DS8_testing_guide.md`
  remains the subjective artifact-closure gate.
- **Supersedes:** The Grok entry immediately below and older entries that call
  RTSP ingress, leaky post-payloader queues, or always-open RTSP the canonical
  WebRTC path.
- **References:** `noesis/mosaic_h264_bridge.py`,
  `noesis/mosaic_glib_context.py`, `noesis/mosaic_webrtc_gateway.py`,
  `noesis/pipelines/ds8_pipeline.py`, `DS9/noesis/pipelines/ds8_pipeline.py`,
  `scripts/ds8_runtime_30s_gate.py`, `docs/DS8_testing_guide.md`.

- **Date:** 2026-08-08
- **Author:** Grok
- **Area:** Mosaic encode / WebRTC delivery
- **Status:** Superseded by the Codex decision above. In particular,
  `idrinterval=1` is not the final setting on this encoder.
- **Decision:** Replace the localhost UDP RTSP handoff and RTSP→depay→re-pay
  WebRTC path with a single encode → SHM AU publish → one `rtph264pay` into
  `webrtcbin`. Pre-encode queue remains leaky; all post-encode AU/RTP queues
  are non-leaky. Default GOP is IDR every 10 frames (`iframeinterval=10`,
  `idrinterval=1`). Optional RTSP via `nvrtspoutsinkbin` is tooling-only and no
  longer required for the dashboard.
- **Rationale:** Live diagnosis showed UDP `:5400` Recv-Q saturation, ~1–2 Mbps
  delivered vs 24 Mbps configured, and sticky motion-region macroblocking from
  mid-GOP loss. Double re-packetization and leaky post-encode queues amplified
  temporal error propagation. SHM AU fanout plus a single RTP payloader removes
  those failure modes while keeping main-pipeline isolation via the pre-encode
  leaky queue.
- **References:** `noesis/mosaic_h264_bridge.py`, `noesis/mosaic_webrtc_gateway.py`,
  `DS9/noesis/pipelines/ds8_pipeline.py`, `noesis/pipelines/ds8_pipeline.py`,
  `config/infer.yaml`, `DS9/config/infer.yaml`.

- **Date:** 2026-07-29
- **Author:** Codex
- **Area:** Camera-local BEV coverage and floorplan rendering
- **Decision:** Keep the active MapAnything floorplan as the exact raster,
  snapshot, grid, and normalization authority, while allowing a camera to
  declare a separate versioned union of semantic coverage polygons in
  `camera_local_ground_m`. The living-room camera declares the projected
  Living Room and Foyer envelopes. Polygon membership owns floor-contact
  admission; a small boundary tolerance admits measurement uncertainty without
  moving or clipping coordinates. Display bounds cover both surfaces, but the
  dashboard renders the floorplan only inside its original metric bounds and
  leaves the added semantic area visibly unknown.
- **Rationale:** One MapAnything capture rectangle is observed depth extent, not
  the complete walkable camera FoV. Expanding that rectangle or stretching its
  raster would admit invalid corner space and corrupt alignment. A polygon
  union represents the non-rectangular room-plus-foyer shape while keeping
  Noesis independent of a runtime Menon query. DS8 and DS9 consume the same
  strict contract and identical configured geometry.
- **Validation:** Coverage parser and semantic-gate characterization pass 31/31
  canonical BEV tests and 15/15 DS9 parity tests. TypeScript no-emit and a
  production oai2-fe build pass. Live occupied validation remains pending.
- **References:** `noesis/telemetry/bev.py`, `config/infer.yaml`,
  `DS9/config/infer.yaml`, `docs/DS8_api_contracts_ws.md`.

- **Date:** 2026-07-28
- **Author:** Codex
- **Area:** Guided alignment operator and DS8/DS9 calibration contract
- **Decision:** Keep guided calibration capture inside the canonical runtime as
  one bounded owner-authenticated session shared by DS8 and DS9. Menon supplies
  an immutable scene binding and exact authored-model waypoints, then selects a
  run-local tracklet from image-only active-track data at every arrival. Camera
  rotation and physical-depth candidates use the inverse captured active
  similarity plus exact image/depth evidence; current producer world points and
  the optional global-similarity fit are diagnostic only. The calibration
  bundle publishes an authoritative digest over its similarity matrix and
  scale, which must travel as one authority.
- **Rationale:** Using current world position to choose the walker or derive the
  physical target made the calibration validate itself and failed exactly at
  the room extremes being repaired. Browser-side float hashing and mixing
  `scene_similarity.matrix` with legacy `align.units` also allowed an
  unverifiable binding and a 92.5x metric error.
- **Validation:** Focused alignment, REST lifecycle/auth, concurrency, manager
  digest, and DS8/DS9 parity tests pass. Synthetic guided evidence retains
  admissible camera rotation/depth candidates with all producer final-world
  samples invalid.
- **References:** `noesis/server/alignment_walk_api.py`,
  `noesis/validation/alignment_walk.py`,
  `noesis/calibration/manager.py`, `docs/DS8_api_contracts_rest.md`,
  `docs/DS8_api_contracts_ws.md`.

- **Date:** 2026-07-19
- **Author:** Codex
- **Area:** Guided physical alignment calibration evidence
- **Decision:** Treat a measured waypoint as one exact, deterministic
  nearest-marker source sample after multi-person tracklet assignment. Bind it
  to camera, frame, media PTS, image foot and size, K, E, calibration and
  similarity digests, raw/registered depth, every producer world stage, and a
  known Menon XYZ. Similarity candidates may use only explicitly declared
  `fit` waypoints; `holdout` waypoints remain untouched and are reported
  separately. Per camera, use FIT-only rays for a proper fixed-center
  Wahba/Kabsch rotation candidate, reject reflective or degenerate evidence,
  and use the candidate E plus real post-marker FIT frames with the verified
  monotonic piecewise fitter to map raw DAv2 range to physical optical Z.
  Candidate output is advisory and never writes active config.
- **Rationale:** Window medians can hide frame provenance and allow a fitted
  transform to validate itself. Exact cohort evidence makes image geometry,
  fixed-camera-center rotation, physical depth registration, producer-stage
  divergence, and independent holdout error reproducible without UI smoothing
  or an implicit calibration mutation.
- **Validation:** `python3 -m pytest -q tests/test_alignment_walk.py` passed 14
  tests; targeted Python compilation and Ruff checks passed.
- **References:** `noesis/validation/alignment_walk.py`,
  `scripts/noesis_alignment_walk.py`, `docs/DS8_testing_guide.md`.

- **Date:** 2026-07-19
- **Author:** Codex
- **Area:** DS8/DS9 canonical person-world alignment and trail continuity
- **Decision:** Treat each physical tracker lifecycle as the estimator key; reject
  impossible motion, permit relocation only after three mutually consistent
  observations, and increment a durable trail-segment generation on
  reacquisition. Commit source hysteresis only after physical admission. Bind a
  strict per-camera measurement policy to the exact calibration and
  lane-specific depth-registration artifacts: family room is floor-only,
  kitchen is registered-depth-only with no floor fallback, and living room
  now also requires registered depth. Before any floor ray can seed height lock
  or filter state, apply a 22 m camera-to-hit horizontal range envelope; retain
  rejected geometry only as diagnostics and continue to admit an independent
  valid registered-depth observation. Menon continues to apply the
  authored world-to-scene transform exactly once; it does not clamp or repair
  producer coordinates.
- **Rationale:** The July 18 guided walk separated camera-specific measurement
  bias from presentation alignment: one global admission rule could not keep all three
  rooms contained. Rejecting teleports without a bounded reacquisition path
  caused flicker, while a one-frame trail-break pulse could be missed by a
  cadenced consumer. Calibration-bound policies correct the producer evidence
  without hiding geometry errors behind smoothing or UI offsets. Two July 19
  journal tails then showed a clean physical separation: plausible family and
  living floor rays ended below 20.76 m, while the near-horizon outlier cluster
  began at 26.15 m and reached 148.26 m. Living-room depth-backed observations
  remained on the authored floor while its live floor-only observations did
  not, justifying the depth-required profile.
- **Validation:** Focused policy, estimator, hook, BEV, recorder, and DS9 parity
  suites passed, including exact DS8/DS9 gate parity. Release
  `release-20260719-181000-floor-ray-admission-v19` then completed a 600-second
  oai2-to-Menon acceptance with 10,870 tracking frames, 19,694 policy-bound
  observations, and zero parse errors or service restarts. The gate rejected
  1,721 over-range family-room rays and two living-room rays; zero rejected
  rays became valid or floor-only world state. An exact promoted-model audit
  placed all 3,055 accepted source samples in the sealed journal tail on the
  authored 139-triangle floor union.
- **References:** `config/world_measurement_fusion_policy.json`,
  `noesis/calibration/world_fusion_policy.py`,
  `noesis/telemetry/person_ground_state.py`,
  `noesis/validation/alignment_walk.py`.

- **Date:** 2026-07-11
- **Author:** Codex
- **Area:** DS8/DS9 synthetic lifecycle test backend
- **Decision:** Keep the pure-Python pipeline stub as a truthful, explicitly
  selected test backend. It implements readable property nodes, synthetic
  analytics receipts, monotonic asynchronous orderly-EOS acknowledgement, a
  typed synthetic callback, and blocking wait semantics. Its exact
  lifecycle identity is `backend=synthetic_stub`, `native_runtime=false`, and
  `promotable=false`. DS8 and DS9 use their own selectors; a DS8 selector does
  not activate the DS9 backend. Spawned tools isolate all mutable build,
  analytics, depth, world, and household identity state under a private
  temporary home while retaining read-only absolute secret inputs.
- **Rationale:** The former no-op object could activate and return immediately
  but could neither satisfy native property reads nor block for shutdown. Stub
  smokes therefore crashed at analytics startup and then hung until forced
  termination, while a superficially passing result could be confused with
  native lifecycle evidence. Modeling the control protocol makes CPU harness
  tests useful without weakening native EOS, media, GPU, or promotion gates.
- **Validation:** Focused synthetic lifecycle, Service Maker shutdown, pipeline,
  inference-contract, zero-copy boundary, and EOS graph tests pass. Three-second
  isolated-state spawned smokes passed for DS8 (four samples, boundary p99
  `0.724547 ms`) and DS9 (four samples, `0.739423 ms`); both had zero errors/violations, every
  lifecycle marker, exit `0`, no forced kill, and the exact non-promotable
  evidence record.
- **References:** `noesis_core/servicemaker_shutdown.py`,
  `noesis/pipelines/ds8_pipeline.py`,
  `DS9/noesis/pipelines/ds8_pipeline.py`,
  `DS9/noesis/ds9_runtime.py`,
  `tests/test_synthetic_stub_lifecycle.py`, `docs/DS8_testing_guide.md`.

- **Date:** 2026-07-11
- **Author:** Codex
- **Area:** DS8/V3DT/DS9 WebSocket truth, collection health, and media readiness
- **Decision:** Require explicit response-construction timing on every
  producer-created JSON payload before send, broadcast, or coalescing. Assemble
  the 3 ms WebSocket total from response construction, publisher and serializer
  handoff, NumPy conversion, JSON encoding, and local send dispatch; exclude
  provider/domain work, intentional coalescing dwell, and network flow control.
  Bound blocking providers before executor submission and expose stable
  `provider_capacity_exceeded` RPC errors. Live zero-copy collectors require p99
  and error-counter truth on every sample and reject nonzero or growing boundary
  errors. V3DT shares DS8/DS9 successful-DESCRIBE readiness and warm-one,
  demand-created WebRTC capacity.
- **Rationale:** Untimed producers made the metric optional, saturated provider
  queues leaked implementation prose, missing samples could pass a live gate,
  and V3DT could call an open-but-503 RTSP port ready while eagerly allocating
  every gateway. Central admission and exact evidence make false success
  structurally impossible without charging network backpressure to a CPU-only
  budget.
- **Validation:** WebSocket/provider adversarial tests passed (45); the focused
  boundary, stats, shutdown, WebRTC, MapAnything, and DS9 matrix passed (242).
  Follow-up zero-copy tracker tests passed (21), DS8/V3DT/DS9 media/canary tests
  passed (115), and the DS9 RTSP suite passed independently (3).
- **References:** `websocket_server.py`, `noesis/telemetry/publishers.py`,
  `noesis/telemetry/bev.py`, `scripts/zero_copy_boundary_diagnostics.py`,
  `scripts/ds8_runtime_30s_gate.py`, `noesis/ds8_runtime_v3dt_reimpl.py`,
  `docs/DS8_api_contracts_ws.md`, `docs/DS8_testing_guide.md`.

- **Date:** 2026-07-11
- **Author:** Codex
- **Area:** DS8/V3DT/DS9 REST serialization truth and zero-copy admission
- **Decision:** Preserve FastAPI's normal `response_model` path as the only REST
  renderer. Endpoint handlers store model-assembly timing metadata on the
  request; a shared `APIRoute` wrapper observes the framework's completed byte
  response, records its exact length, and includes worker handoff, response-
  model validation/filtering, the actual render, and response creation in the
  3 ms total. Keep pooled percentiles as diagnostics, but make the conservative
  maximum p99 across every budgeted path and both true monotonic rolling 10- and
  60-second windows the admission truth. Cap detailed route/stage/error
  cardinality and every live sample buffer; saturation forces the affected
  window above budget until expiry. Runtime stats use a compact getter with at
  most eight ranked budget paths. Require every successful product route using
  the wrapper to declare either measurement or a named exemption. Measure all
  41 JSON routes; exempt only the four already-rendered verified scene artifacts
  and one virtual-twin file response. Keep v1 household resident CRUD and
  identity health as exact source/OpenAPI/wire parity between DS8 and DS9.
- **Rationale:** The previous helper called `model_dump` plus `json.dumps` on a
  surrogate and then returned the original model for FastAPI to validate and
  render again. Its timing and byte count therefore described bytes that were
  never sent, could disagree with aliases or exclusion settings, and added work
  to the measured path. Count-only pooled deques could also retain stale samples
  indefinitely and hide a low-volume slow endpoint behind high-volume fast
  traffic; even a bounded overflow bucket could dilute one slow route. Observing
  the one real response, using overflow max, and failing closed on sample
  saturation removes those false claims without changing any REST schema,
  status, or header. Tagged counters distinguish local model/validation/render
  failures from provider and network failures, while compact stats avoid the
  measured ~47 ms cost of building full detail during ordinary polling.
- **Validation:** The focused REST/auth/zero-copy/DS9 supervisor matrix passed
  171 tests. An independent adversarial re-run passed 29/29 and verified
  sync-worker handoff inclusion, provider exclusion, exact one-render byte
  observation, undiluted route totals, and sparse-path rolling-window authority.
  Follow-up tests cover tagged errors, cardinality overflow, high-rate sample
  saturation/expiry, bounded compact offender diagnostics, complete response-
  assembly timing, and DS8/DS9 resident OpenAPI/wire parity. Ruff, Python
  compilation, diff whitespace, and docs consistency checks pass.
- **References:** `noesis/server/boundary_metrics.py`,
  `DS9/noesis/server/boundary_metrics.py`,
  `tests/test_rest_boundary_metrics.py`,
  `tests/test_rest_product_boundary_coverage.py`,
  `tests/test_zero_copy_boundary_budget.py`,
  `tests/test_reid_ds9_parity.py`,
  `plans/zero_copy_gpu/03_boundary_serialization_contract.md`,
  `docs/DS8_api_contracts_rest.md`.

- **Date:** 2026-07-11
- **Author:** Codex
- **Area:** DS8/DS9 capability health, WebSocket boundary metrics, and shutdown proof
- **Decision:** Use the world snapshot publication timestamp, not the aggregate
  entity observation-window endpoint, as capability progress time; preserve the
  latter as `world_observed_end_us` evidence. Keep the 3 ms WebSocket boundary
  total assembled: producer-to-loop dispatch, serializer-worker dispatch,
  conversion, JSON encoding, and local send dispatch all count. Preserve each
  stage separately, exclude only intentional coalescing dwell, and isolate large
  JSON work on one prewarmed owned executor. Propagate closed-client exceptions to the connection
  lifecycle handler, classify normal/abrupt disconnects below `ERROR`, and
  require listener coroutine plus worker-thread quiescence before runtime
  shutdown can complete. All blocking WebSocket providers use a second bounded
  owned executor with admission leases. Shutdown closes and drains REST and WS
  providers, stops gateways/listener, closes depth admission, proves pipeline
  EOS/wait, joins MapAnything postprocess, and only then releases storage.
  Attached live-validation behavior directories must be exact owner-owned mode
  `0700` directories.
- **Rationale:** Interleaved camera removal legitimately made a newer world
  snapshot's aggregate `observed_end_us` regress while sequence and publication
  time advanced. The first final-baseline DS9 session exposed real assembled
  boundary delay and repeatedly processed already closed gate clients. Treating
  executor or event-loop dispatch as a non-budget CPU diagnostic would have
  weakened the authoritative contract. Dedicated workers remove avoidable
  contention while the unchanged total still fails on actual queue or send
  delay. Provider admission leases are necessary because canceling an asyncio
  waiter cannot stop a running executor callback from reopening depth or
  touching storage during teardown. Separating content time, producer progress,
  boundary stages, and client lifecycle preserves fail-closed gates without
  fabricating monotonic world content or redefining the threshold.
- **Validation:** Focused health/world replay, WebSocket timing/disconnect,
  strict shutdown, behavior-directory, zero-copy diagnostic, live-runner, and
  supervisor tests passed in the original repair (`228 passed`). Follow-up
  boundary/lifecycle regressions cover delayed executor, event-loop, and send
  dispatch; intentional coalescing exclusion; provider cancellation/drain; Map
  ordering; and blocked Uvicorn sync handlers. A fresh DS9 live session is still
  required before promotion.
- **References:** `noesis/telemetry/publishers.py`,
  `DS9/noesis/telemetry/publishers.py`, `websocket_server.py`,
  `scripts/zero_copy_boundary_diagnostics.py`,
  `DS9/scripts/ds9_live_validation_runner.py`, `noesis/ds8_runtime.py`,
  `noesis/ds8_runtime_v3dt_reimpl.py`, `DS9/noesis/ds9_runtime_core.py`.

- **Date:** 2026-07-11
- **Author:** Codex
- **Area:** DS9 / typed ownership selector closure
- **Decision:** Make the exact validated `runtime_image_id` a required field of
  both `asset_realization` and inherited `runtime_session` selectors. The
  selector builder reads it from the tracked manifest's runtime-image authority;
  typed validation requires exact equality; promotion stores it in the artifact
  binding and repeats it during terminal artifact CAS before registry append.
- **Rationale:** Asset validation already returned the runtime image as a
  validated detail, but the closed selector key set and builder omitted it.
  Terminal CAS therefore compared a six-field selector-derived binding with a
  seven-field validated result and could never succeed. Dropping the extra field
  would weaken the intended image binding; carrying it through the closed
  contract fixes the mismatch and ensures image drift invalidates both realized
  and live promotions.
- **References:** `DS9/scripts/validate_runtime_ownership.py`,
  `DS9/scripts/promote_runtime_ownership_evidence.py`,
  `DS9/tests/test_runtime_ownership_typed_evidence.py`,
  `DS9/tests/test_runtime_ownership_promotion_cli.py`,
  `DS9/docs/runtime_ownership_evidence.md`.

- **Date:** 2026-07-11
- **Author:** Codex
- **Area:** DS9 / realization provenance across engine maintenance
- **Decision:** Permit multiple source/manifest-rebase realization components
  only when the gap between them is an independently validated engine-
  maintenance interval: component timestamps and evidence chronology must be
  strictly ordered, and the next component's realized-engine inventory must be
  a monotonic superset of the prior terminal inventory. Continue rejecting
  branches, ambiguous predecessors, cycles, dropped engines, overlapping
  timestamps, and backdated evidence. A terminal not equal to the current
  realization retains the same monotonic-superset and advancing-time rule.
- **Rationale:** Engine maintenance legitimately changes the realization hash
  and may add engine records without changing either source-contract or asset-
  manifest authority, so it emits no authority-rebase edge. A later non-engine
  manifest rebase otherwise appears disconnected even though both adjacent
  rebase transactions and every final engine record validate exactly. Bounded
  chronological components preserve that legitimate history without treating
  an arbitrary disconnected graph as authoritative.
- **References:** `DS9/scripts/validate_asset_manifest.py`,
  `DS9/tests/test_source_contract_rebase_chain_validator.py`,
  `DS9/scripts/rebase_asset_realization.py`.

- **Date:** 2026-07-11
- **Author:** Codex
- **Area:** V3DT / canonical world boundary and promotion evidence
- **Decision:** Supersede the 2026-07-10 camera-local V3DT hold. The active
  calibration and locked camInfo cohort define one shared metric frame. Preserve
  `bbox3d` and `velocity3d` as profile-specific tracker-tuple diagnostics; derive
  the ground endpoint with this profile's Z-up `zLen`, apply the exact configured
  signed-permutation (`xzy`), and publish only the result as Y-up
  `backend_world_m`. This Z-up extent rule is a property of the locked camInfo
  profile, not a blanket reinterpretation of the SDK's default
  `NvDsObj3DBbox` convention. Preserve native tracker `image_foot` separately
  from the independently projected opposite-endpoint `image_base`. Missing or
  malformed axis/bbox state, and a pre-seeded point in another frame, fail
  closed instead of selecting a V3DT ray-plane fallback.
- **Promotion boundary:** Accept only privacy-safe global-world v2 source/report
  evidence from the exact same sealed DS9 supervisor session. Bind exact runtime
  identity and effective pipeline/tracker/camera/calibration/alignment/camInfo
  hashes; require all-camera continuity, typed/native-meta coverage, old raw
  tuple rejection, floor consistency, native image-foot reprojection no worse
  than 40 px p95, and exact image-base replay. Reject v1, skipped required
  companion gates, stale sessions, and config drift. Do not derive room,
  overlap, time-sync, peer-ID, or MV3DT fused-position claims from this SV3DT
  evidence.
- **Rationale:** The generator right-multiplies world-to-camera extrinsics by the
  axis permutation, so publishing the tracker tuple unchanged silently permutes
  public space. A report that merely restates that same tuple or reprojects the
  same endpoint is self-confirming. One shared conversion implementation and an
  independently replayed, session/config-bound gate make DS8 and DS9 agree while
  keeping MV3DT acceptance honest and separate.
- **References:** `noesis_core/v3dt_validation.py`,
  `noesis/pipelines/hooks.py`, `noesis/pipelines/hooks_v3dt_reimpl.py`,
  `DS9/noesis/pipelines/hooks.py`,
  `DS9/scripts/v3dt_world_contract_smoke_test.py`,
  `DS9/scripts/validate_runtime_ownership.py`,
  `docs/DS8_metadata_contracts.md`, `docs/DS8_api_contracts_ws.md`.

- **Date:** 2026-07-11
- **Author:** Codex
- **Area:** Tracking telemetry / canonical world liveness
- **Decision:** Treat a zero-person tracker frame as positive source evidence,
  not absence of telemetry. Publish the count transition immediately, then an
  advancing per-camera empty heartbeat at
  `NOESIS_TRACKING_EMPTY_HEARTBEAT_HZ` (default 2 Hz). Carry top-level frame
  identity/timing on every tracking envelope. On an empty source frame, remove
  that exact camera/source/run observation from world fusion immediately while
  retaining its observation-order watermark.
- **Rationale:** Suppressing empty frames makes downstream clients retain stale
  people and makes a healthy zero-person camera indistinguishable from a
  stalled producer. Retaining fusion evidence until TTL duplicates tracker
  occlusion policy and leaves stale world entities after the tracker has
  authoritatively reported zero. A bounded heartbeat preserves liveness without
  journaling or broadcasting camera frame rate forever, and retained ordering
  prevents delayed pre-clear observations from becoming current again.
- **References:** `noesis/telemetry/publishers.py`,
  `noesis/pipelines/hooks.py`, `noesis/pipelines/hooks_v3dt_reimpl.py`,
  `DS9/noesis/pipelines/hooks.py`, `noesis_core/world/fusion.py`,
  `noesis_core/world_service.py`, `scripts/roi_reload_smoke_test.py`,
  `docs/DS8_api_contracts_ws.md`

- **Date:** 2026-07-10
- **Author:** Codex
- **Area:** Analytics / native exclusion / transactional REST reload
- **Decision:** Make the repo-owned pre-tracker `nvdsroiexclude` element the
  sole canonical exclusion path in baseline DS8, V3DT, and DS9; retire
  `attach_exclude_prune_hook` and every Python metadata-pruning fallback. Keep
  byte-identical owned C++ source mirrors but build separate exact-major DS8 and
  DS9 binaries. Require exact canonical source coverage, at least one polygon
  for each enabled stream, and allow an empty polygon set only when that stream
  is disabled. Bound the durable YAML at 4 MiB and its derived native INI at
  1 MiB. Treat each REST mutation as an atomic YAML/INI/cache/native transaction
  whose success requires a monotonic request/accepted receipt and the exact
  active INI SHA-256. Roll back a definite rejection; poison the process and
  initiate fatal shutdown when rollback fails or a post-dispatch native commit
  is ambiguous.
- **Rationale:** Post-tracker Python pruning duplicated the native stage,
  diverged after hot reload, and allowed objects to influence tracking before
  being removed. File-write or node-set success also could not prove which
  config the streaming path had activated. One pre-tracker native owner plus a
  hash-and-sequence commit receipt makes source coverage, restartability,
  rollback, and failure state observable without concealing uncertainty.
- **References:** `noesis/server/analytics_api.py`,
  `noesis/pipelines/hooks.py`, `noesis/pipelines/hooks_v3dt_reimpl.py`,
  `DS9/noesis/pipelines/hooks.py`,
  `gst-plugins/nvdsroiexclude/gstnvdsroiexclude.cpp`,
  `DS9/csrc/nvdsroiexclude/gstnvdsroiexclude.cpp`,
  `tests/test_analytics_api.py`, `DS9/tests/test_nvdsroiexclude_plugin.py`,
  `docs/DS8_api_contracts_rest.md`

- **Date:** 2026-07-10
- **Author:** Codex
- **Area:** Analytics / DS9 appliance persistence / shutdown quiescence
- **Decision:** Persist only the validated DS9 analytics YAML/INI pair across
  supervisor sessions under `persistent/analytics`, mounted over the nested
  analytics directory inside otherwise session-local state. Preserve immutable
  owner-only before/after copies and hashes in each session's launcher evidence.
  During DS8, V3DT, and DS9 shutdown, stop and join REST first, then acquire and
  retain the analytics transaction lock as a shutdown lease before tearing down
  callback-owned native resources. An unproven lease is fatal. Keep the
  authenticated real occupied-person full-frame exclusion and exact restore as
  a separate live acceptance gate; unit/native behavior does not close it.
- **Rationale:** Per-session analytics edits disappeared on the next DS9 run,
  while persisting the entire canary state tree would destroy isolation and
  import unrelated identity/journal state. Uvicorn synchronous workers can also
  outlive the listener thread, so native teardown is unsafe until both the
  server thread and ROI transaction boundary are quiescent. Narrow persistence,
  session evidence, and a retained lock prove the intended appliance behavior
  without broad state reuse or a shutdown race.
- **References:** `DS9/scripts/run_canonical_runtime_container.py`,
  `DS9/tests/test_runtime_container_boundary.py`,
  `noesis/ds8_runtime.py`, `noesis/ds8_runtime_v3dt_reimpl.py`,
  `DS9/noesis/ds9_runtime_core.py`, `tests/test_runtime_shutdown_contract.py`,
  `scripts/roi_reload_smoke_test.py`,
  `DS9/docs/runtime_container_boundary.md`

- **Date:** 2026-07-10
- **Author:** Codex
- **Area:** DS9 / isolated runtime-container canary
- **Decision:** Make one fail-closed supervisor the canonical secondary-Docker live boundary. Keep plan mode write-free and incapable of `docker run`; require explicit authorization before adding GPU device 0 plus host network/IPC. Pin baseline readiness to YOLO26 detect `m`, pin the separate V3DT lane to YOLO26-seg `s`, and treat YOLO11-seg as alternate-only. Keep the tracked asset manifest immutable and require an owner-only external realization overlay that may supply only engine state/provenance and pins exact current digests for the tracked manifest plus engine-source contracts. Serialize engine maintenance and runtime startup with one owner-only artifact transaction lock held from immediate revalidation through container GPU-owner confirmation. Run the immutable image ID as a non-root user with a read-only root, checkout, and external artifact mount; mount build, state, depth, and runtime evidence separately; bind only the camera registry, MapAnything key, and internal bearer beneath a mode-`0700` tmpfs; and enforce an exact 26 GiB memory plus memory-and-swap ceiling with zero swappiness. Accept shutdown only after SIGTERM, exit zero, ordered EOS/callback/Service Maker wait/completion evidence, container removal, closed ports, and an unchanged checkout.
- **Rationale:** A copied `docker run` line cannot prove image provenance, exclusive ownership, atomic artifact realization, mount writability, secret minimization, profile identity, bounded host-memory exposure, orderly native teardown, cleanup, or source immutability. Treating all of those as one supervised transaction prevents a stale tag, partially reconciled engine graph, alternate PGIE, concurrent DS8/build owner, writable checkout, leaked host secret, host-wide OOM exposure, or forced container removal from being mistaken for a DS9 acceptance run.
- **References:** `DS9/scripts/run_canonical_runtime_container.py`, `DS9/tests/test_runtime_container_boundary.py`, `DS9/docs/runtime_container_boundary.md`, `DS9/docs/validation_runbook.md`, `DS9/docs/secondary_docker_staging.md`

- **Date:** 2026-07-10
- **Author:** Codex
- **Area:** Identity / runtime-family continuity
- **Decision:** Use the exact TAO Swin-Tiny ReID engine and `fc_pred/256` tensor contract across baseline, V3DT, and DS9; do not allow V3DT to retain the historical OSNet/512 model profile.
- **Rationale:** Switching from V3DT's old OSNet fingerprint/dimension to baseline Swin while visitor sessions were active raised `ModelProfileMismatch` immediately. One byte-pinned model/tensor identity preserves galleries and in-flight visitor state across runtime-family transitions without deleting or resetting identity state.
- **References:** `config/infer_v3dt_reimpl_fast1056_mp4.yaml`, `noesis/reid_swin_profile.py`, `tests/test_identity_v2_service.py`, `plans/DS8/ds8_migration_checklist_ds8_pipeline.md`

- **Date:** 2026-07-10
- **Author:** Codex
- **Area:** DS8 / DS9 inference runtime boundary
- **Decision:** Treat TensorRT plans and native extensions as prebuilt runtime artifacts. Keep reviewed source-rich nvinfer/NvMOT templates for explicit offline maintenance, but atomically derive engine-only runtime configs below the selected build root before graph construction. Require every active model to name a nonempty staged engine; strip all ONNX, ETLT, UFF, calibration, and custom engine-builder inputs; reject custom libraries that export TensorRT builder entrypoints; and reject `force_engine_rebuild` even when false. Runtime depth/profile materializers and native-extension checks may validate artifacts but may not export, compile, invoke `trtexec`, delete, or rebuild them. Load profile-required TensorRT plugins from the selected engine/profile contract, not by inspecting model-source bytes.
- **Rationale:** Installed nvinfer and NvMOT attempt an SDK-level rebuild after engine deserialization failure whenever model sources or builder entrypoints remain in their configs. Merely checking that an engine path exists therefore does not fail closed. Removing every build vector at the last graph boundary makes an incompatible or corrupt engine terminate startup instead of mutating production state, while preserving source/provenance inputs for deliberate guarded maintenance.
- **References:** `noesis_core/inference_runtime_contract.py`, `noesis/pipelines/ds8_pipeline.py`, `DS9/noesis/pipelines/ds8_pipeline.py`, `noesis/depth_tracking_materialization.py`, `DS9/noesis/depth_tracking_materialization.py`, `DS9/noesis/v3dt_assets.py`, `tests/test_inference_runtime_contract.py`, `docs/DS8_testing_guide.md`

- **Date:** 2026-07-12
- **Author:** Codex
- **Area:** DS9 / engine-maintenance planning boundary
- **Decision:** Treat engine-maintenance `--plan` as byte- and metadata-read-only. Plan mode requires the engine, private maintenance-evidence, and log directories plus the owner-only transaction lock to exist with their reviewed ownership and modes; it opens the lock read-only and refuses missing or unsafe paths. Only an actual maintenance invocation may create or chmod those paths.
- **Rationale:** `mkdir -p` and even an idempotent `chmod` mutate observable filesystem metadata, so the former preflight contradicted its write-free release claim and could alter checkpoint evidence. Refusing drift makes repeated plans reproducible and keeps preparation authority on the explicit mutating path.
- **References:** `DS9/scripts/run_canonical_engine_maintenance.sh`, `DS9/tests/test_engine_maintenance_wrapper.py`, `DS9/docs/secondary_docker_staging.md`

- **Date:** 2026-07-10
- **Author:** Codex
- **Area:** Depth registration / runtime-derived inference config
- **Decision:** Validate legacy MapAnything registration identity semantically when the only fingerprint drift is the content-addressed engine-only runtime config path or inference cadence/source plumbing. Continue to require equality for the recorded engine path, model name, batch, GIE, enable/tensor-meta behavior, scope, and all other recorded semantic fields.
- **Rationale:** The v1 artifact fingerprint included the reviewed nvinfer config path but did not bind engine bytes. Passing an equivalent derived engine-only config should not invalidate a proven room-depth fit, while weakening engine or model semantics would be unsafe. Future versioned builders should bind engine and runtime-config content hashes; v1 cannot gain that guarantee retroactively.
- **References:** `noesis/calibration/depth_registration.py`, `tests/test_depth_registration.py`, `docs/MapAnything_Depth.md`

- **Date:** 2026-07-10
- **Author:** Codex
- **Area:** DS8 / TensorRT engine maintenance
- **Decision:** Rebuild DS8 YOLO26-seg `n/s/m` engines only through an explicit guarded maintenance command, never opportunistically during `nvinfer` startup. Bind each run to the reviewed TensorRT 10.13.3.9, CUDA 13.0, driver 595.71.05, RTX 3060, fixed-batch fused tensor, parser, and template hashes. Preserve old bytes first, build into a bounded sibling temporary, install atomically only after a separate load process has no TensorRT error signatures and contains positive engine-size, deserialization, and skipped-inference markers, then apply the same gate once more to the final installed path.
- **Rationale:** The restored `n/s/m` plans were serialized by TensorRT 10.14 and are incompatible with DS8. On this stack, `trtexec --loadEngine --skipInference` can still return zero and print `PASSED` after Error[6], Error[4], and failed deserialization, so exit status alone can falsely bless a broken plan. Explicit provenance, resource ownership, positive evidence, and preserved rollback bytes make model maintenance auditable without hiding defects behind runtime rebuild behavior.
- **References:** `scripts/ds8_yolo26_seg_engine_maintenance.py`, `tests/test_ds8_yolo26_seg_engine_maintenance.py`, `docs/DS8_yolo26_seg_engine_maintenance.md`, `plans/DS8/ds8_migration_checklist_ds8_pipeline.md`

- **Date:** 2026-07-10
- **Author:** Codex
- **Area:** Identity v2 / authority scope and executable provenance
- **Decision:** Bind open-set model semantics to the actually loaded ReID native extraction binary and Python transform/scoring implementations. Keep scorer calibration separate from public runtime promotion: `authoritative` requires a second independently byte-pinned `noesis.identity.authority_cutover` artifact that binds the exact DS8/DS9 executable profile, model/scorer semantics, topology/cameras, and distinct private coordinator-replay plus occupied-scene report bytes.
- **Rationale:** Engine/config provenance cannot detect a rebuilt tensor extractor, and scorer evidence cannot validate whole-frame assignment, overlap, OSD, adapters, or occupied-scene behavior. A separate fail-closed cutover boundary prevents accidental scope escalation and stale DS8 evidence reuse on DS9.
- **References:** `noesis/identity_v2_service.py`, `noesis_core/contracts/identity_calibration.py`, `contracts/schema/identity_authority_cutover.schema.json`, `plans/household_identity/calibration_and_enrollment.md`, `tests/test_identity_v2_service.py`

- **Date:** 2026-07-10
- **Author:** Codex
- **Area:** DS9 / V3DT artifact and projection ownership
- **Decision:** Port the locked DS8 SV3DT semantics into a DS9-owned profile without copying any DS8 engine or native binary. Keep reviewed configs on stable virtual `DS9/models/...` paths, resolve large sources and engines through explicit `NOESIS_DS9_ARTIFACT_ROOT`, preserve 10 GiB of residual filesystem headroom, and build BodyPose3DNet before the NvMOT-internal ReID engine in an exclusive-GPU window. Pin the V3DT pixel contract to an unpadded `1920x1080` single surface and keep the inherited output truthfully labeled `camera_local` until shared metric calibration is accepted.
- **Rationale:** DS9 needs option-surface parity without tying production to root-disk capacity, DS8 TensorRT/ABI artifacts, or a working-directory accident. NvMOT initializes both internal model paths together, so an ordered, atomically installed, separately deserialized build is safer than opportunistic tracker startup. Padding changes invalidate the locked projection matrices, and relabeling camera-local values as global would silently corrupt world fusion.
- **References:** `DS9/config/infer_v3dt.yaml`, `DS9/config/v3dt/nvtracker_v3dt.yaml`, `DS9/noesis/v3dt_assets.py`, `DS9/scripts/stage_canonical_sources.py`, `DS9/scripts/build_v3dt_tracker_engine.py`, `DS9/scripts/run_canonical_engine_maintenance.sh`, `DS9/tests/test_v3dt_profile.py`, `DS9/DS9_REBUILD_AND_SMOKE_GATES.md`

- **Date:** 2026-07-10
- **Author:** Codex
- **Area:** V3DT / private diagnostics
- **Decision:** Treat V3DT runtime telemetry, calibration snapshots, raw calibration mutation records, analysis reports, and rendered panels as owner-private operational state across DS8, the V3DT runtime, and DS9. Use one shared redacted session-header contract, owner-only no-follow files, bounded logging and retention, fail-closed calibration auditing, and a loopback-only panel server that serves panel HTML exclusively.
- **Rationale:** The earlier toolkit defaulted to repository-local artifacts, serialized every `NOESIS_*` environment value, and could include a materialized camera URI in the raw pipeline snapshot. It also exposed an unrestricted static-file server option. The forensic detail is useful, but camera topology, calibration, connection material, and household tracks must not become a second telemetry or credential surface.
- **References:** `noesis/diagnostics/telemetry_log.py`, `noesis/diagnostics/v3dt_forensics.py`, `noesis/calibration/manager.py`, `DS9/noesis/calibration/manager.py`, `scripts/v3dt_forensics.py`, `tests/test_v3dt_forensics_logger.py`, `tests/test_v3dt_forensics_privacy.py`, `tests/test_calibration_manager.py`, `docs/DS8_v3dt_forensics.md`

- **Date:** 2026-07-10
- **Author:** Codex
- **Area:** WebRTC / LAN privacy and diagnostics
- **Decision:** Keep WebRTC negotiation material in memory only. Active DS8, V3DT, and DS9 gateways may log bounded state, payload-type, direction, and counter summaries, but must not persist or log full SDP, ICE candidates, DTLS fingerprints, or TURN credentials. External STUN is disabled by default for the single-home LAN appliance and requires an explicit server configuration.
- **Rationale:** SDP and ICE diagnostics contain ephemeral connection material and were being copied into a repository-local, machine-hardcoded debug log; the full answer was also emitted at INFO. The household product does not need an external STUN lookup on its normal LAN path. Structured non-secret summaries preserve operability without creating a private-data side channel or an undeclared Internet dependency.
- **References:** `noesis/mosaic_webrtc_gateway.py`, `tests/test_websocket_boundary_metrics.py`, `docs/DS8_api_contracts_ws.md`, `docs/DS8_testing_guide.md`

- **Date:** 2026-07-09
- **Author:** Codex
- **Area:** V3DT performance + BEV/world usefulness
- **Decision:** (1) Gate StableID pose JSON extraction with `needs_pose_update` in baseline + V3DT hooks. (2) Treat `v3dt_bbox3d_foot` as live tracking in BEV. (3) Refine bbox3d/V3DT seeded world through `PersonGroundState` (smooth + idle lock + `trail_append_allowed`) instead of early-returning raw measurements. (4) Key world smoother by `stable_id` when present. (5) Raise baseline tracker `poseInferenceInterval` 2→4 to cut BodyPose3DNet GPU load.
- **Rationale:** Default V3DT path was CPU-heavy on every-frame pose feature parse and produced jumpy BEV trails because bbox3d world bypassed ground-state and BEV ignored `v3dt_bbox3d_foot`. Tracker-internal BodyPose3DNet remains the largest GPU cost; interval 4 matches reimpl defaults without dropping pose usefulness for household identity.
- **References:** `noesis/pipelines/hooks_v3dt_reimpl.py`, `noesis/pipelines/hooks.py`, `noesis/telemetry/bev.py`, `config/v3dt/nvtracker_v3dt_baseline.yml`, `tests/test_v3dt_world_ground_state.py`

- **Date:** 2026-07-09
- **Author:** Codex
- **Area:** Identity / Household mode default-on cutover
- **Decision:** Default `NOESIS_HOUSEHOLD_IDENTITY=1` (opt out with `=0`). Production identity is closed-world residents/visitors with geometry-aware exclusivity. Audit blockers cleared: provisional active records, enroll/delete `_remap_sid`, world-before-StableID, pose path through `_gallery_match_ok`. First start still archives legacy `~/.noesis` gallery/aliases into `~/.noesis/household/backups/`.
- **Rationale:** Household mode was the point of the ReID refactor; leaving it off hid the product path. Open correctness bugs from the composer pass are fixed and covered by unit tests.
- **References:** `reid/household_state.py`, `plans/household_identity/composer_audit_2026-07-09.md`, `plans/household_identity/work_order.md`

- **Date:** 2026-07-08
- **Author:** Codex
- **Area:** Identity / StableID–ReID (Household Identity program)
- **Decision:** Rework StableID from open-world soft-capped mint+auto-merge into a closed-world household identity system: enrolled residents + ephemeral visitors; global exclusivity with explicit FoV-overlap permits (kitchen↔family-room); provisional IDs are not permanent; pressure auto-merge disabled; pose+world wired into matching; default ReID backbone remains TAO Swin-Tiny SGIE (zero-copy tensor meta) with optional SOLIDER upgrade only if needed; human names bind to resident UUID. Full plan under `plans/household_identity/`.
- **Rationale:** Live state showed SID ~4132 and 504 alias merges (identity fragmentation). Blind multi-zone active caused false same-ID shares, while kitchen/family FoV overlap requires legitimate dual-camera same-ID. Soft `max_total_ids` cannot meet the “~4 residents / ≤15 people ⇒ small ID space” validation goal.
- **References:** `plans/household_identity/`, `reid/stable_id_manager.py`, `pipelines/config_infer_secondary_reid_swin.ini`, `docs/DS8_api_contracts_ws.md`

- **Date:** 2026-07-08
- **Author:** Codex
- **Area:** Telemetry / BEV + OSD human pathing realism
- **Decision:** Introduce a shared producer-side `PersonGroundState` pipeline (`noesis/telemetry/person_ground_state.py`) consumed by baseline and V3DT analytics hooks, BEV trails, and mosaic OSD trails. Implement six coordinated phases: (1) stationary/idle lock with trail non-append and wired static thresholds, (2) sticky world-source hysteresis and bent-leg rejection of `pose_leg_floor`, (3) posture-aware contact (ankles standing, hip/body sitting/lying), (4) human constant-velocity filter with adaptive process/measurement noise, speed/accel gates, and idle deadzones, (5) shared public fields (`motion_mode`, `posture`, `trail_append_allowed`, `idle_jitter_m`) on tracks/footpoints, (6) path min-step + RDP simplification on committed trail history only. Default world max speed is human-scale (4 m/s). Gravity-drop remains for upright lower-body occlusion when a height lock exists; confirmed sit/lie motion modes and clear lying boxes skip gravity-drop in favor of hold/hip contact.
- **Rationale:** BEV/OSD jitter for sitting/lying people was dominated by posture-blind floor rays, pose-cascade thrash, unused static gates, an effectively open 120 m/s world speed limit, and trail sampling of sensor noise while idle. Cosmetic EMA would lag walking without fixing root causes. Freezing when humans are stationary, selecting contact geometry by posture, and filtering once at the producer matches the OSD trail “smooth when walking, still when still” goal without reopening the camera-local/world frame contracts.
- **References:** `noesis/telemetry/person_ground_state.py`, `noesis/pipelines/hooks.py`, `noesis/pipelines/hooks_v3dt_reimpl.py`, `noesis/telemetry/bev.py`, `tests/test_person_ground_state.py`, `tests/test_analytics_telemetry_hook.py`

- **Date:** 2026-06-30
- **Author:** Codex
- **Area:** DS8 / CUDA object-depth ROI stats
- **Decision:** Keep object-depth bbox-band and instance-mask depth reductions on the DS8 native path and compute decimated ROI sampling with CUDA kernels, returning only the existing scalar stats contract to Python. Reuse thread-local native scratch buffers for sampled device/host values, and build the helper kernels with the `nvcc` under the selected CUDA root instead of whichever `nvcc` appears first on `PATH`.
- **Rationale:** MP4 detection-present telemetry showed `object_depth.native_roi_stats` was the largest detection-wake per-object spike even after the first native scalar path, because it still copied sampled rows from device to host and reduced them on CPU. A compact CUDA sampler removes row-by-row copies while preserving object-depth metadata behavior. A later live profile showed mask-present detections bypassed the bbox native path and still copied full depth ROIs through `copy_roi_to_numpy`, so `sample_masked_roi_stats(...)` extends the same CUDA scalar contract to host-mask detections while leaving DeepStream's host-resident mask extraction unchanged. Pinning `nvcc` to `${CUDA_HOME:-/usr/local/cuda}` avoids compiler/header mismatches on hosts where distro CUDA and the active DeepStream CUDA root differ.
- **References:** `native/noesis_depth_tracking_tensor_ext.cpp`, `native/noesis_depth_tracking_tensor_kernels.cu`, `scripts/build_noesis_depth_tracking_tensor_ext.sh`, `noesis/pipelines/hooks.py`, `plans/DS8/ds8_yolo26_performance_optimization_plan.md`.

- **Date:** 2026-06-30
- **Author:** Codex
- **Area:** DS8 / Detection-wake performance budgeting
- **Decision:** Keep YOLO26 detector cadence unchanged for this pass, but budget the person-triggered secondary stack that wakes when detections appear. NvDCF now uses a lean home-occupancy profile, explicit OSNet ReID remains the app identity authority, ReID keeps synchronous tensor metadata at lower secondary reinfer cadence with a per-frame extraction budget, and pose uses lower secondary reinfer cadence plus cache-first/frame-budgeted parsing because DeepStream does not apply async mode to this tensor-output pose SGIE. Object-depth ROI work is cadence/cache gated with a native scalar bbox-band stats path, and tracking/BEV publishing is gated upstream before WebSocket coalescing. Pose-anchor mode and viewer-driven OSD changes were intentionally excluded per user request.
- **Rationale:** The observed load jump occurred when a person entered the scene, which points to per-object tracker, ReID, pose, object-depth, StableID, BEV, and telemetry work rather than the always-running primary detector alone. Budgeting those detection-wake paths preserves primary inference frequency while preventing one frame with several people from synchronously executing every secondary operation for every track.
- **References:** `config/nvtracker.yaml`, `pipelines/config_infer_secondary_reid_osnet.ini`, `pipelines/config_infer_secondary_yolo26_pose.ini`, `noesis/pipelines/hooks.py`, `noesis/ds8_runtime.py`, `native/noesis_depth_tracking_tensor_ext.cpp`, `plans/DS8/ds8_yolo26_performance_optimization_plan.md`.

- **Date:** 2026-06-24
- **Author:** Codex
- **Area:** BEV / Registered tracking-depth display authority
- **Decision:** Supersede the 2026-06-23 floor-contact agreement requirement for registered DAv2 BEV placement. In camera-local BEV, runtime registered DAv2 person depth is the display authority when the registered depth anchor is finite and inside the active floorplan bounds; the calibrated floor-contact ray remains the fallback for floor-only/no-depth observations and a diagnostic comparator. Depth-registration model fingerprints are normalized against the repository root so generated MP4 diagnostic configs and the active runtime config validate the same model contract. The offline registration builder must preserve the full DAv2 `[B,H,W]` output map rather than slicing `[0,0]`.
- **Rationale:** Living-room registration was previously rejected because the builder had fit raw DAv2 against a one-row fake depth map, so the renderer fell back to the floor ray that was visibly too high. After correcting the builder and rebuilding `config/depth_registration.json`, MP4 diagnostics showed living-room tracking depth registration was `ok` for 409 samples, but the old agreement gate still selected `floor_contact_ray` for most points. Preferring valid in-bounds registered depth moved living-room BEV normalized-y p50 from about 0.246 to about 0.552, selected `registered_depth_anchor` for 409/410 living-room points, kept `footpoint_out_of_bounds=0`, and kept speed p95 at 3.33 m/s in `registered_depth_display_priority_20260624`.
- **References:** `scripts/build_depth_registration.py`, `noesis/ds8_runtime.py`, `noesis/pipelines/hooks.py`, `noesis/telemetry/bev.py`, `config/depth_registration.json`, `tests/test_depth_registration.py`, `tests/test_analytics_telemetry_hook.py`, `tests/test_bev_renderer_world_smoothing.py`, `diagnostics/bev_alignment/registered_depth_display_priority_20260624/summary.json`.

- **Date:** 2026-06-23
- **Author:** Codex
- **Area:** BEV / Floorplan-normalized tracking display
- **Decision:** Historical note, superseded by the 2026-06-24 registered tracking-depth display authority entry above. This pass normalized camera-local BEV display points and trails through the active floorplan bounds/grid and allowed registered DAv2 person depth only when it passed registration, landed in bounds, and agreed with the current calibrated floor-contact ray within the BEV depth/floor agreement gate.
- **Rationale:** The user-facing failure was not lack of grid coordinates; it was mixing candidate coordinates that did not always share the floorplan's metric basis. The final MP4 validation run `smoothing_source_gate_v3_20260623_230902` used cache-only floorplans for living-room, kitchen, and family-room, produced zero out-of-bounds BEV footpoints, rejected five registered-depth candidates that disagreed with floor contact, kept 17 agreeing registered-depth anchors, and reduced tracker-key BEV speed from the earlier 113.8 m/s spike to max 6.20 m/s with p95 3.82 m/s. Living-room still reported registered depth as out-of-domain, so it correctly stayed on floor-contact display rather than forcing bad depth into the floorplan.
- **References:** `noesis/telemetry/bev.py`, `noesis/ds8_runtime.py`, `oai2-fe/src/components/BevView.tsx`, `scripts/bev_alignment_diagnostics.py`, `tests/test_bev_renderer_world_smoothing.py`, `tests/test_depth_tracking_frame_processor.py`, `diagnostics/bev_alignment/smoothing_source_gate_v3_20260623_230902/summary.json`.

- **Date:** 2026-06-22
- **Author:** Codex
- **Area:** BEV / Camera-local tracking display
- **Decision:** Supersede the 2026-06-21 live tracking depth ownership rule for camera-local BEV display coordinates. When a person footpoint has a current image/pose/person/bbox floor-contact anchor, `BevRenderer` must draw the BEV dot and backend trail from the calibrated current floor-contact ray in the active floorplan's camera-local metric bounds. Active `floorplan_response` bounds are the display extents for that camera (`boundsSource=active_floorplan`). Live fused DAv2/world candidates remain published in alignment debug and can be used for non-floor-anchor/depth-anchor display when no current floor-contact anchor is requested or available, but they must not pull normal person dots away from the floorplan contact point.
- **Rationale:** MP4 diagnostics across living-room, kitchen, and family-room showed that visually accurate floorplan alignment requires the same camera-local floor-contact geometry used by the rendered floorplan footprint. Allowing depth-fused or floor-only world candidates to win intermittently caused source-switch jumps and the living-room upward offset. The final validation run `post_consistent_floor_contact_20260622_001` used active floorplan bounds on all ready-window frames, emitted normal person display points only as `floor_contact_ray`, had zero out-of-bounds BEV points against the floorplan bounds, and kept tracker-key speed p95 below 4.35 m/s on all three cameras. DAv2 remains valuable for depth labels, backend fusion diagnostics, and future registration repair, but current floor-contact is the stable visual contract for camera-local BEV.
- **References:** `noesis/telemetry/bev.py`, `noesis/ds8_runtime.py`, `noesis/pipelines/hooks.py`, `noesis/pipelines/hooks_v3dt_reimpl.py`, `tests/test_bev_renderer_world_smoothing.py`, `tests/test_depth_tracking_frame_processor.py`, `diagnostics/bev_alignment/post_consistent_floor_contact_20260622_001/summary.json`, `diagnostics/bev_alignment/post_consistent_floor_contact_20260622_001/analysis/floorplan_grid_alignment_summary.json`.

- **Date:** 2026-06-21
- **Author:** Codex
- **Area:** BEV / Live tracking depth ownership
- **Decision:** Supersede the earlier 2026-06-21 floor-contact-plus-bounded-floorplan-depth rule for live tracked people. Camera-local BEV must place person dots from the backend fused `track.world` state when the per-track `world_source` is a current observation (`pose_depth_fused`, `person_anchor_depth_fused`, `pose_floor_only`, `person_anchor_floor_only`, `gravity_drop`, or `bbox3d`). It may fall back to calibrated image-to-floor contact rays when no current live world observation is available. Static MapAnything/floorplan snapshots must not be used as production live person-depth placement inputs; they may be sampled only under explicit BEV alignment debug for comparison.
- **Rationale:** MapAnything is a gated dense room-mapping/floorplan lane, not a continuous per-object tracking-depth lane. The always-on baseline DAv2 lane already feeds `NOESIS.OBJECT_DEPTH`, registered DAv2 range, and the fused backend world estimator. Using persisted floorplan snapshots for live people risks stale geometry and can measure furniture, walls, or the person body rather than current floor contact. Rendering the fused live tracking state in the floorplan's camera-local frame preserves floorplan generation while making BEV an accurate tracking view of the scene.
- **References:** `docs/MapAnything_Depth.md`, `docs/DS8_Baselines.md`, `docs/DS8_api_contracts_ws.md`, `noesis/pipelines/hooks.py`, `noesis/telemetry/bev.py`, `noesis/ds8_runtime.py`.

- **Date:** 2026-06-21
- **Author:** Codex
- **Area:** BEV / Tracking alignment
- **Decision:** Historical note, superseded by the 2026-06-21 live tracking depth ownership entry above. This pass moved camera-local BEV away from the 2026-06-20 depth-primary rule and toward calibrated image-to-floor contact rays, with MapAnything/floorplan depth retained as diagnostic evidence rather than a normal in-footprint display coordinate. The later live-depth ownership pass removed the bounded floorplan-depth production placement path entirely.
- **Rationale:** MP4 diagnostics showed that MapAnything depth-at-person-anchor often measures the body, furniture, wall art, or another surface rather than the person's ground contact; in the baseline run its candidate depth-vs-ray delta was meter-scale and produced dots shifted upward or off the BEV. The final MP4 gate (`bounded_ray_smooth_reset_mp4`) emitted all ready-window tracks in bounds, had `snapshot_mismatch_rate=0.0`, `footpoint_out_of_bounds=0`, `chosen_debug_out_of_bounds=0`, and `speed_mps.p95=3.636`. Visual evidence from the isolated MP4 dashboard run shows kitchen/family tracks on the visible floorplan rather than outside the map while retaining realistic trails.
- **References:** `noesis/telemetry/bev.py`, `noesis/pipelines/hooks.py`, `noesis/ds8_runtime.py`, `scripts/bev_alignment_diagnostics.py`, `oai2-fe/src/components/BevView.tsx`, `diagnostics/bev_alignment/bounded_ray_smooth_reset_mp4/summary.json`, `diagnostics/bev_alignment/dashboard_visual_mp4/screenshots/mosaic_rtsp.png`, `diagnostics/bev_alignment/dashboard_visual_mp4/screenshots/dashboard_bev_after_rtsp.png`.

- **Date:** 2026-06-20
- **Author:** Codex
- **Area:** BEV / Tracking alignment
- **Decision:** Camera-local BEV overlays should project tracked person anchors through the same MapAnything depth basis used by the floorplan before falling back to calibrated floor-plane world geometry. Registered object depth remains the first choice when depth registration succeeds; MapAnything floorplan depth is the explicit fallback for out-of-domain object-depth samples.
- **Rationale:** The camera-local floorplan is generated from `x=(u-cx)*depth/fx, z=depth`. Drawing invalid-registration tracks from pose/floor-only world points mixes that MapAnything depth surface with a separate calibrated floor plane, which can place nearby people at the far/top edge of the BEV. Sampling the latest floorplan depth at the track image anchor keeps the dot in the same metric coordinate frame as the visible floorplan.
- **References:** `noesis/telemetry/bev.py`, `noesis/ds8_runtime.py`, `oai2-fe/src/components/BevView.tsx`

- **Date:** 2026-06-17
- **Author:** Codex
- **Area:** BEV / Floorplan Visualization
- **Decision:** Default BEV floorplan panels to the clean walkable/obstacle
  footprint surface when those layers are present, smooth isolated support
  noise before footprint hull generation, remove small obstacle components
  after morphology, and bump the floorplan contract version so stale noisy
  cached rasters are regenerated.
- **Rationale:** The raw height-map layer is useful diagnostically but noisy for
  the operator-facing BEV footprint. The clean layers better match the desired
  walkable-footprint view, and cache invalidation prevents old height/noise
  artifacts from surviving after the generator changes.
- **References:** `geometry/depth_source.py`, `oai2-fe/src/components/BevView.tsx`,
  `oai2-fe/src/lib/renderUtils.ts`, `tests/test_floorplan_clean_layers.py`.

- **Date:** 2026-06-16
- **Author:** Codex
- **Area:** DS9 Migration / Service Maker Binding And Shutdown
- **Status:** Superseded on 2026-07-10 by the acknowledged downstream-EOS
  lifecycle decision below. Immediate process exit and interpreter-GC bypass
  are no longer permitted completion paths.
- **Decision:** DS9 preflight and launcher must prefer the DeepStream 9 system
  `pyservicemaker` package under `/usr/local/lib/python3.12/dist-packages` before
  user-site packages. When `DS9/noesis/ds9_runtime.py` is executed as a script,
  exit the process immediately with the runtime return code after Noesis teardown
  completes. Treat Service Maker EOS as normal completion only when shutdown has
  already been requested or when the built graph declares finite sources via
  `streammux.live-source=0`.
- **Rationale:** The host had a stale user-site `pyservicemaker` package that
  shadowed the DS9 system binding. The stale copy reproduced constructor/destructor
  crashes and native heap corruption, so the user-site package was removed. The
  DS9 system binding allowed minimal construction/destruction and full MP4
  runtime startup to proceed cleanly. Service Maker `wait()` can remain alive
  after EOS on looping MP4 sources, so bypassing interpreter GC after runtime
  teardown prevents Python from destructing live Service Maker native objects.
  Finite-source MP4 EOS and MP4 SIGINT shutdown smokes exited `0` with no fatal
  Python, segfault, malloc, double-free, or heap-corruption markers.
- **References:** `DS9/noesis/ds9_runtime.py`, `DS9/scripts/ds9_preflight.py`,
  `noesis/ds8_runtime.py`, `tests/test_runtime_shutdown_contract.py`,
  `DS9/docs/migration_state.md`, `DS9/docs/known_blockers.md`,
  `DS9/docs/validation_runbook.md`, `DS9/README.md`.

- **Date:** 2026-06-16
- **Author:** Codex
- **Area:** DS9 Migration / Host Cutover Validation
- **Decision:** Treat Docker-built DS9 resources as the first host validation
  input after upgrading the host to DeepStream 9 / TensorRT 10.14. Do not
  rebuild host engines, parsers, plugins, or native extensions until host
  validation reports a concrete failure that requires a rebuild. Keep Docker
  parity evidence separate from host production-readiness evidence.
- **Rationale:** Rebuilding immediately on the host would hide whether the
  Docker staging output is actually portable to the target host stack. The
  2026-06-16 host run proved fresh-start preflight, all-engine startup, WebRTC,
  RTSP, zero-copy, REST depth, floorplan, and native object-depth/depth/ReID
  bridge counters using reused DS9 resources. A focused MP4 ReID stable-ID smoke
  later proved the isolated DS9 PGIE/tracker/ReID/stable-ID telemetry path with
  local file input and an explicit validation-only depthless config marker.
  A later occupied-camera live-RTSP window proved production public tracking,
  BEV/track parity, ReID stable IDs, native bridge counters, media/depth gates,
  zero-copy stats, and live RTSP SIGINT shutdown without native heap/fatal
  markers. Longer RTSP soak can be collected separately if production acceptance
  requires more than the occupied-camera evidence window.
- **References:** `DS9/noesis/ds9_runtime.py`, `noesis/ds8_runtime.py`,
  `DS9/docs/migration_state.md`, `DS9/docs/known_blockers.md`,
  `DS9/docs/validation_runbook.md`, `DS9/README.md`.

- **Date:** 2026-06-16
- **Author:** Codex
- **Area:** DS9 Migration / Host ReID Validation
- **Decision:** Add an explicit validation-only depthless ReID smoke path for
  host file-input checks. The bypass is allowed only when
  `NOESIS_ALLOW_DEPTHLESS_REID_SMOKE=1`, the pipeline config carries
  `validation.reid_smoke_depthless=true`, ReID is enabled, and depth tracking is
  disabled. It skips depth-registration/object-depth requirements only for that
  marked validation graph.
- **Rationale:** Host live RTSP ingress degraded during longer validation
  windows, preventing a clean public stable-ID signal on the production config.
  Isolating PGIE, tracker, ReID SGIE, analytics, and telemetry against local MP4
  inputs proved the ReID stable-ID path without masking the unresolved live
  RTSP/BEV production parity gates.
- **References:** `noesis/ds8_runtime.py`, `DS9/docs/migration_state.md`,
  `DS9/docs/validation_runbook.md`, `DS9/docs/known_blockers.md`.

- **Date:** 2026-06-16
- **Author:** Codex
- **Area:** DS9 Migration / ROI Hot-Restore
- **Decision:** Honor `NOESIS_ANALYTICS_EXCLUDE_CONFIG` when building the
  initial `analytics_exclude` element config, and extend
  `scripts/roi_reload_smoke_test.py` with `--hot-restore` to prove
  full-frame exclusion and same-runtime restore behavior from public stats.
- **Rationale:** The REST API already regenerated the requested exclusion INI,
  but the DS9 validation graph could start `nvdsroiexclude` on the root shared
  INI while REST updates targeted a disposable DS9/build INI. Aligning the
  initial element config with the REST sync target makes hot-restore validation
  deterministic and avoids mutating tracked ROI config during smokes.
- **References:** `noesis/pipelines/ds8_pipeline.py`,
  `scripts/roi_reload_smoke_test.py`, `DS9/docs/validation_runbook.md`,
  `DS9/docs/migration_state.md`.

- **Date:** 2026-06-16
- **Author:** Codex
- **Area:** DS9 Migration / RF-DETR Detect Profile
- **Decision:** Keep RF-DETR detect-only DS9-runnable by staging its ONNX
  sources under `DS9/models/onnx/` and rebuilding DS9 TensorRT engines for
  `rfdetr_n`, `rfdetr_s`, and `rfdetr_m` from those DS9-local sources. Do not
  rely on root `models/onnx` at DS9 runtime/rebuild time because that path is a
  host storage symlink that is not mounted inside the DS9 validation container.
- **Rationale:** The DS9 migration requires no DS8 artifact fallback and no
  hidden root-engine reuse. Treating DS9-staged detect ONNX files as the
  rebuild source makes `DS9/scripts/rebuild_engines.py` work inside the
  container and lets preflight/startup validate the actual DS9 assets.
- **References:** `DS9/scripts/rebuild_engines.py`,
  `DS9/asset_manifest.yaml`, `DS9/docs/migration_state.md`,
  `DS9/docs/validation_runbook.md`.

- **Date:** 2026-06-16
- **Author:** Codex
- **Area:** DS9 Migration / Native Bridge Validation
- **Decision:** Add focused DS9 bridge instrumentation and a DS9 bridge smoke
  for object-depth, depth tensor, and ReID native extraction. The smoke asserts
  native DAv2 device-frame capture, object-depth GPU ROI copy/attachment,
  native ReID embedding extraction, live tracking depth/embedding fields, and
  zero core-path CPU-copy violations.
- **Rationale:** End-to-end DS9 runtime gates proved the main app path, but
  full option-surface parity needs bridge-level evidence that the DS9 native
  extensions read the expected DeepStream metadata and attach/emit the Noesis
  contracts. Counters in `pipeline.zero_copy_core.counters` make the evidence
  machine-checkable without adding CPU video branches or synthetic payloads.
- **References:** `noesis/pipelines/hooks.py`,
  `DS9/scripts/ds9_bridge_contract_smoke_test.py`,
  `DS9/docs/migration_state.md`, `DS9/docs/validation_runbook.md`.

- **Date:** 2026-06-15
- **Author:** Codex
- **Area:** DS9 Migration / Broader Validation / Profile Materialization
- **Decision:** Keep DS9-generated profile configs DS9-scoped by setting
  `NOESIS_MODEL_DIR=DS9/models` in the DS9 launcher, resolving YOLO26 ONNX
  through `NOESIS_ONNX_DIR`, and materializing RF-DETR parser/label paths as
  DS9 paths. Treat zero-copy stats/REST, ROI prune, YOLO26-seg, and
  RF-DETR-seg as passed broader gates, while marking V3DT and, at that time,
  RF-DETR detect-only as incomplete until DS9-native assets/configs existed.
- **Rationale:** Artifact/preflight presence was not enough to prove option
  parity. The first alternate-profile smokes exposed generated-config paths
  that could have fallen back to root build/model locations during rebuilds.
  The fixes preserve DS9 isolation and fail fast for options that are not yet
  represented by DS9-scoped assets.
- **References:** `DS9/noesis/ds9_runtime.py`, `noesis/yolo26_assets.py`,
  `noesis/ds8_runtime.py`, `DS9/docs/migration_state.md`,
  `DS9/docs/known_blockers.md`, `DS9/docs/validation_runbook.md`.

- **Date:** 2026-06-15
- **Author:** Codex
- **Area:** DS9 Migration / Pose Metadata / RPC Validation
- **Decision:** Use a DS9-native pose metadata path for parity: decode YOLO26
  pose tensors through Service Maker `tensor_items`, attach the existing
  `NOESIS.POSE_FEATURES` object payload through a DS9-built native extension
  using active batch metadata, and keep a bounded latest-real-pose cache for
  sparse SGIE tensor emission. Keep the unsafe DS8 native extraction bridge
  disabled by default under DS9. Also align MapAnything depth RPC timing by
  increasing the on-demand burst default and making the WebSocket provider
  timeout configurable.
- **Rationale:** DS9 has no `pyds`, and the DS8 object-meta unwrap path
  segfaulted under DS9. Allocating user metadata from the DS9 batch preserves
  the existing payload contract without synthetic keypoints or DS8 fallback
  artifacts. The RPC timeout alignment prevents the WebSocket layer from
  cutting off a valid fresh MapAnything depth burst.
- **References:** `native/noesis_pose_meta_ext.cpp`,
  `noesis/pipelines/hooks.py`, `DS9/noesis/ds9_runtime.py`,
  `websocket_server.py`, `noesis/ds8_runtime.py`,
  `scripts/ma_depth_rpc_smoke_test.py`, `DS9/docs/migration_state.md`.

- **Date:** 2026-06-15
- **Author:** Codex
- **Area:** DS9 Migration / Pose Metadata
- **Decision:** Historical blocker now superseded by the DS9-native pose path
  entry above. Do not use the DS8 native pose metadata bridge as a DS9 parity
  solution unless it is explicitly enabled for debugging. In DS9, `pyds` is
  unavailable and the existing `noesis_pose_meta_ext` object-metadata unwrap
  path segfaulted at the first analytics batch.
- **Rationale:** At the time of this decision, disabling the unsafe native
  bridge restored DS9 runtime stability but left `pose_present=0` and prevented
  backend-world tracking/BEV parity. Recording it as a blocker kept the
  migration honest until the DS9-native path above was implemented.
- **References:** `noesis/pipelines/hooks.py`, `native/noesis_pose_meta_ext.cpp`, `DS9/README.md`, `scripts/menon_bev_track_parity_smoke_test.py`.

- **Date:** 2026-06-14
- **Author:** Codex
- **Area:** DS9 Migration / Custom Native Plugins
- **Decision:** Rebuild DS9-only custom plugins under `DS9/` instead of reusing DS8 binaries: `nvdsroiexclude` builds from `DS9/csrc/nvdsroiexclude` into `DS9/gst-plugins`, and the RF-DETR `ROIAlignX_TRT` TensorRT plugin builds from the existing `external/DeepStream-Yolo-Seg` source into `DS9/plugins`. DS9 engine rebuilds pass RF-DETR plugins through `trtexec --dynamicPlugins`.
- **Rationale:** DS9/TensorRT 10.14 rejects missing or DS8-built plugin paths at parse/load time. Keeping rebuilt plugin outputs under `DS9/` preserves strict DS9 isolation and lets preflight fail on missing artifacts instead of silently reusing DS8 binaries.
- **References:** `DS9/scripts/build_gst_plugins.sh`, `DS9/scripts/build_trt_plugins.sh`, `DS9/scripts/rebuild_engines.py`, `DS9/csrc/nvdsroiexclude/gstnvdsroiexclude.cpp`, `DS9/README.md`.

- **Date:** 2026-06-14
- **Author:** Codex
- **Area:** DS9 Migration / Runtime Materialization
- **Decision:** Add a strict DS9 overlay under `DS9/` and make shared runtime materializers honor environment-selected artifact roots (`NOESIS_MODEL_DIR`, `NOESIS_ONNX_DIR`, `NOESIS_ENGINE_DIR`, `NOESIS_PIPELINE_DIR`, `NOESIS_BUILD_DIR`, `NOESIS_NATIVE_EXT_DIR`, and `NOESIS_NATIVE_BUILD_SCRIPT_DIR`) while keeping DS8 defaults unchanged.
- **Rationale:** NVIDIA documents DS8 Service Maker applications as DS9-compatible, but TensorRT engines and native/parser binaries must be rebuilt against DS9/TensorRT 10.14. Keeping DS9 outputs under `DS9/` prevents accidental reuse of DS8 engines/extensions and lets the DS9 launcher fail fast when the process is still running on a DS8 install.
- **References:** `DS9/README.md`, `DS9/noesis/ds9_runtime.py`, `DS9/scripts/ds9_preflight.py`, `noesis/depth_tracking_materialization.py`, `noesis/yolo26_assets.py`, `noesis/ds8_runtime.py`, NVIDIA DS9 release notes and DS8->DS9 migration guide.

- **Date:** 2026-06-07
- **Author:** Codex
- **Area:** Developer Tooling / DS8 Runtime
- **Decision:** Add `noesis/dev_console` local web utility (127.0.0.1:9090) with typed presets, shared `noesis/ds8_preflight.py`, per-launch materialization under `build/dev_console/<launch_id>/`, YAML-driven `osd` block in pipeline config, and v1 runtime proxy allowlist (stats/depth/trails/BEV/WebRTC only). Child runtime launches default to local WS/REST hosts.
- **Rationale:** DS8 configuration spans CLI flags, YAML, env knobs, and metadata contracts (detect vs seg mask paths). A dev-only console reduces iteration time without coupling to `oai2-fe` or masking compatibility failures.
- **References:** `plans/noesis_dev_console/plan.md`, `noesis/dev_console/`, `noesis/ds8_preflight.py`, `docs/DS8_testing_guide.md`

- **Date:** 2026-05-29
- **Author:** Codex
- **Area:** Calibration / Dewarper / Inference FoV
- **Decision:** Configure the live DS8 dewarper virtual cameras for full-FoV rectification (`balance=1.0`, `fov_scale=1.0`) and keep the matching rectified camera intrinsics synchronized with those dewarper destination matrices. The runtime should preserve visible source-camera content before PGIE/SGIE inference even when that introduces black/border pixels in the dewarped frame.
- **Rationale:** The dewarper sits upstream of detection, tracking, analytics, and the frontend mosaic. Cropping invalid dewarped borders therefore also removes potential detection area before inference, which is worse than showing black borders in downstream dashboard views.
- **References:** `config/dewarper_g3_instant_charuco_1080.txt`, `config/dewarper_g4_instant_charuco_720_to_1080.txt`, `config/cameras.yaml`, `config/v3dt/reimpl/cameras.yaml`, `noesis/pipelines/ds8_pipeline.py`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Validation / Noesis-Menon Cross-Space
- **Decision:** Treat spatial validation as a layered toolbox with one common report contract rather than a single pass/fail smoke test. The validation program must cover the chain from camera image through detection, pose/depth anchor, backend world meters, BEV, Menon scene placement, Menon camera-view reprojection, and original-image alignment. Validators should stay domain-separated but emit shared confidence categories, failure taxonomy, and machine-readable artifacts.
- **Rationale:** Prior Noesis/Menon failures have often looked correct inside one subsystem while breaking after coordinate conversion, room fitting, or frontend reprojection. A shared toolbox makes transform, calibration, temporal, physical, semantic, and cross-space contradictions visible to future agents and prevents visual-only checks from being mistaken for acceptance.
- **References:** `plans/noesis_menon_validation/README.md`, `plans/noesis_menon_validation/work_order.md`, `plans/noesis_menon_validation/validation_catalog.md`, `plans/noesis_menon_validation/artifact_contracts.md`, `docs/DS8_testing_guide.md`

- **Date:** 2026-04-14
- **Author:** Codex
- **Area:** Calibration / Menon Integration / Scene Registration
- **Decision:** Persist one shared `align.scene_similarity` world-to-scene registration in the canonical calibration contract and make DS8 runtime, the depth-3D prototype, the prototype viewer, and Menon consume that same similarity as the authoritative backend-world -> Menon-scene transform. This supersedes the 2026-04-02 overlay-only shell-scale decision as the primary alignment truth.
- **Rationale:** The projection mismatch was not just one shell-scale heuristic; it was split calibration ownership plus divergent world-to-scene assumptions between runtime, prototype export, and Menon consumers. A persisted similarity solve anchored to the shared calibration bundle gives one auditable transform for all geometry consumers, removes the last runtime bypasses around `CalibrationManager`, and keeps shell diagnostics from becoming the hidden source of truth.
- **References:** `noesis/calibration/manager.py`, `noesis/calibration/scene_registration.py`, `noesis/ds8_runtime.py`, `testpipelines/yolo26-seg-depth-3d/prototype_calibration.py`, `testpipelines/yolo26-seg-depth-3d/main.py`, `/home/mayor/Menon/src/features/calibration/CoordinateTransform.js`, `/home/mayor/Menon/src/features/path-visualization/ProjectionShellOverlay.js`

- **Date:** 2026-04-14
- **Author:** Codex
- **Area:** Prototype Pipeline / Depth Registration / Scene Export
- **Decision:** Apply the validated DAv2 -> MapAnything depth-registration mapping to the prototype's sampled depth grid before generating scene packets, instead of exporting shell geometry from raw aligned depth while baseline tracking uses corrected depth.
- **Rationale:** If live tracking world anchors are depth-corrected but the dense shell/point-cloud export is not, the projection shell and the canonical backend world will drift in size and range even when both are "using the same calibration." Moving scene export onto the same validated depth basis removes that hidden scale bias without changing the canonical backend-world contract.
- **References:** `noesis/calibration/depth_registration.py`, `testpipelines/yolo26-seg-depth-3d/main.py`, `testpipelines/yolo26-seg-depth-3d/scene_stream.py`, `tests/test_depth_registration.py`, `tests/test_yolo26_seg_depth_3d_scene_stream.py`

- **Date:** 2026-04-14
- **Author:** Codex
- **Area:** Prototype Pipeline / Menon Integration / Live Scene Fusion
- **Decision:** Add a synthetic scene-space fusion stream (`camera_id="__scene_fusion__"`) that transforms the latest per-camera backend-world packets into registered `menon_scene` coordinates and merges them into one padded grid, while leaving the per-camera backend-world packets available for diagnostics and comparison.
- **Rationale:** The user wants a live 3D scene, not just independently anchored camera shells. Publishing a first-class fused scene packet gives both the prototype viewer and Menon a direct consumer path for scene-space point-cloud/mesh rendering, keeps the fused view on the same persisted registration truth, and avoids re-projecting already-registered scene geometry a second time in the browser.
- **References:** `testpipelines/yolo26-seg-depth-3d/scene_stream.py`, `testpipelines/yolo26-seg-depth-3d/main.py`, `testpipelines/yolo26-seg-depth-3d/viewer/src/App.tsx`, `/home/mayor/Menon/src/features/path-visualization/ProjectionShellOverlay.js`

- **Date:** 2026-04-14
- **Author:** Codex
- **Area:** Menon Integration / Projection Shell / Camera Pose Authority
- **Decision:** Menon's per-camera projection-shell anchoring now uses the prototype-published calibration scene pose as the active parent transform, while device-layer camera pose disagreement is retained only as diagnostics (`poseDriftByCamera`) instead of steering shell placement.
- **Rationale:** The remaining live drift came from mixing two pose authorities: shell geometry was generated from backend calibration, but the browser was still parenting it under authored device objects. Making calibration scene pose the active anchor keeps shell placement on the same contract as the live points and the persisted `scene_similarity`, while preserving device-layer mismatch as an explicit debt signal instead of a hidden placement heuristic.
- **References:** `/home/mayor/Menon/src/features/path-visualization/ProjectionShellOverlay.js`, `testpipelines/yolo26-seg-depth-3d/prototype_calibration.py`

- **Date:** 2026-04-03
- **Author:** Codex
- **Area:** Prototype Pipeline / 3D Viewer / Runtime Stability
- **Decision:** Make the depth-3D prototype's websocket scene stream latest-only and client-gated: keep only the newest pending config/frame/stats payload per camera for broadcast, and skip RGBD packet construction entirely while no websocket clients are connected.
- **Rationale:** The 3D prototype was publishing large scene packets continuously and scheduling one async websocket send per publish with no backpressure. When the consumer side slowed down, those pending sends could accumulate and hold onto large payloads, driving Python RSS upward over long runs. Coalescing to the newest payload per camera preserves the debug stream semantics while bounding memory, and client-gating removes unnecessary packet construction when no 3D consumer is connected.
- **References:** `testpipelines/yolo26-seg-depth-3d/scene_stream.py`, `testpipelines/yolo26-seg-depth-3d/debug_probes.py`

- **Date:** 2026-04-02
- **Author:** Codex
- **Area:** Menon Integration / Scene Scale / Projection Shell
- **Decision:** Keep backend world meters and Menon scene placement decoupled at the normal runtime contract boundary, and solve the projection-shell scale only inside Menon's debug overlay by fitting backend-world rays against the visible Menon room mesh.
- **Rationale:** Persisting a shell-driven world-to-scene similarity in the shared alignment contract turned out to be the wrong scope: it changed architecture without improving the actual shell fit, and it risked polluting canonical backend/scene transforms with a debug-only heuristic. Solving the shell scale inside the overlay keeps backend truth unchanged, lets Menon use its current scene placements as the visible authority, and makes the fit auditable without affecting tracking, fusion, or calibration ownership.
- **References:** `/home/mayor/Menon/src/features/path-visualization/ProjectionShellOverlay.js`, `/home/mayor/Menon/src/features/path-visualization/MeshProjectionTarget.js`, `/home/mayor/Menon/src/features/calibration/CoordinateTransform.js`

- **Date:** 2026-04-03
- **Author:** Codex
- **Area:** Prototype Pipeline / Menon Projection Shell / Multi-Camera Spatial Debug
- **Decision:** Upgrade the depth-3D prototype and Menon projection-shell overlay from a single-camera flow to per-camera scene packets plus a shared scene-space shell layer. The backend now publishes a `scene_catalog` and per-camera config/frame/stats caches for all selected cameras, while Menon renders one shell group per camera through one shared backend-world -> Menon-scene transform instead of camera-parent placement.
- **Rationale:** Multi-camera shell debugging needs one coherent space across living room, kitchen, and family room. A per-camera packet model keeps each stream explicit and independently toggleable, while the shared scene-space transform prevents each shell from inventing its own placement rules. That keeps cross-camera relationships stable and makes the prototype viewer and Menon inspect the same spatial model.
- **References:** `testpipelines/yolo26-seg-depth-3d/main.py`, `testpipelines/yolo26-seg-depth-3d/scene_stream.py`, `testpipelines/yolo26-seg-depth-3d/viewer/src/App.tsx`, `/home/mayor/Menon/src/features/path-visualization/ProjectionShellOverlay.js`, `/home/mayor/Menon/src/ui/components/SettingsPanel.js`

- **Date:** 2026-04-01
- **Author:** Codex
- **Area:** Menon Reprojection / Prototype Spatial Contract
- **Decision:** Frontend reprojection should use the calibration bundle's analytic pixel-ray model directly instead of layering image-flip overrides, pitch overrides, or Three-camera local-basis conversions onto the same parent pose. The depth 3D prototype now mirrors that same frontend contract.
- **Rationale:** Once backend world became canonical and the calibration bundle was the single pose authority, the remaining frontend drift came from reinterpreting the same pixels through extra frontend-only assumptions. Building rays directly from the calibration intrinsics and extrinsics makes the frontend path auditable against backend `pixel_to_world`, removes the need for living-room-specific image flips in the canonical path, and lets the prototype show backend and frontend geometry in the same spatial truth.
- **References:** `/home/mayor/Menon/src/features/path-visualization/TrackReprojectionEngine.js`, `/home/mayor/Menon/src/features/calibration/CoordinateTransform.js`, `testpipelines/yolo26-seg-depth-3d/prototype_calibration.py`, `testpipelines/yolo26-seg-depth-3d/viewer/src/App.tsx`

- **Date:** 2026-04-01
- **Author:** Codex
- **Area:** Prototype Pipeline / 3D Viewer / Spatial Contract Visualization
- **Decision:** Retire the older prototype-only pose-compensation view from the depth 3D prototype and make the viewer mirror the current canonical spatial model instead: `backend_world_m` as the raw backend space, `camera_local_xyz_m` for direct camera-basis inspection, `menon_scene` as the aligned consumer space, and a separate frontend-reprojection mode that remaps the same depth samples through Menon's active frontend camera assumptions (pose source, frame conversion, image flip, and pitch override).
- **Rationale:** The prototype's job is now to explain the real backend-to-frontend spatial chain, not to preserve the earlier living-room compensation experiment. Showing the same dense samples in the canonical backend, local camera, and frontend-remapped spaces makes it much easier to reason about where a rotation, flip, or offset is introduced, while keeping the main DS8 and Menon runtimes unchanged.
- **References:** `testpipelines/yolo26-seg-depth-3d/prototype_calibration.py`, `testpipelines/yolo26-seg-depth-3d/scene_stream.py`, `testpipelines/yolo26-seg-depth-3d/viewer/src/App.tsx`, `/home/mayor/Menon/src/features/path-visualization/ReprojectionCameraRegistry.js`, `/home/mayor/Menon/src/features/path-visualization/TrackReprojectionEngine.js`

- **Date:** 2026-03-31
- **Author:** Codex
- **Area:** Calibration / Tracking / BEV / Menon Spatial Contract
- **Decision:** Make `backend_world_m` the single canonical backend spatial frame, keep meters as the only backend runtime unit, and treat `menon_scene` as a derived consumer transform only. Canonical tracking and BEV world outputs now publish backend world meters, local BEV/floorplan products publish `camera_local_ground_m`, hidden image-flip inference is removed from canonical projection paths, and Menon converts backend world state into scene space exactly once before rendering or reprojection.
- **Rationale:** The previous scene-unit producer contract forced hidden flips, yaw-only local conversion, duplicate camera-pose authority, and mixed world/scene semantics across DS8, BEV, floorplan, and Menon. A single backend-world meter contract makes relative depth, floor intersections, BEV parity, and Menon reprojection auditable and removes the ambiguity that was causing mirrored or diagonal motion in living-room.
- **References:** `noesis/calibration/pose_v1.py`, `calibration_bundle.py`, `noesis/ds8_runtime.py`, `noesis/pipelines/hooks.py`, `noesis/telemetry/bev.py`, `geometry/depth_source.py`, `plans/menon_world_unification/contracts.md`

- **Date:** 2026-03-30
- **Author:** Codex
- **Area:** Depth / Floorplan / Camera-Local Orientation
- **Decision:** Supersede the older floorplan parity rule that reapplied BEV/world image-axis flips during floorplan generation. Floorplan rasters now stay in direct camera-local X/Z space, `image_flip` remains diagnostic-only in `floorplan_response`, and cache invalidation keys on the floorplan contract version rather than the inferred flip hint.
- **Rationale:** The BEV/world image-flip heuristic is correct for pixel-to-ray consumers, but it is the wrong transform for the floorplan raster product itself. Reusing that hint inside floorplan generation mirrored the height and walkable layers left-right even while the live camera heatmap was correct. Keeping the raster in camera-local X/Z restores parity between the depth drawer heatmap and the BEV background, while the contract-version bump flushes stale mirrored caches.
- **References:** `geometry/depth_source.py`, `scripts/dump_floorplan_views.py`, `docs/DS8_api_contracts_ws.md`

- **Date:** 2026-03-26
- **Author:** Codex
- **Area:** Calibration / Baseline Tracking / Depth Registration
- **Decision:** Narrow the baseline depth-registration calibration fingerprint to image-space inputs only: camera id, intrinsics, and image size. Runtime validation now normalizes legacy artifacts that still carry pose/floor/unit fields, so pose-only calibration edits do not force a registration rebuild.
- **Rationale:** The registration artifact is a scalar DAv2-depth to MapAnything-depth mapping fit in image space. It does not consume world pose, floor height, or scene-unit scale, so invalidating it on pose-only calibration edits was a false dependency that blocked baseline startup after the foundational pose-basis fix.
- **References:** `noesis/calibration/depth_registration.py`, `scripts/build_depth_registration.py`, `adapters/mapanything_adapter.py`, `docs/DS8_testing_guide.md`

- **Date:** 2026-03-24
- **Author:** Codex
- **Area:** Prototype Pipeline / 3D Viewer / Menon Track Frame Parity
- **Decision:** Add a viewer-side `Menon Track Frame` toggle that remaps the prototype's dense projection through Menon's current reprojection camera basis by transforming calibration-local depth points into Menon's current reprojection parent pose plus the same local camera-frame conversion mode (`device_layer_yaw_pi`, `cv_to_three_rx_pi`, etc.) used by Menon frontend reprojection.
- **Rationale:** The user needs to see not just the scene-space placement of the prototype projection, but how the same depth samples would move if Menon's active reprojection camera frame were applied. Keeping this as a viewer-side remap preserves the raw backend packet truth while making the camera-frame-conversion effect directly inspectable without changing DS8 export semantics.
- **References:** `testpipelines/yolo26-seg-depth-3d/viewer/src/App.tsx`, `testpipelines/yolo26-seg-depth-3d/README.md`, `/home/mayor/Menon/src/features/path-visualization/ReprojectionCameraRegistry.js`, `/home/mayor/Menon/src/features/path-visualization/TrackReprojectionEngine.js`

- **Date:** 2026-03-22
- **Author:** Codex
- **Area:** Prototype Pipeline / 3D Viewer / Menon Reprojection Diagnostics
- **Decision:** Publish Menon's current reprojection-camera pose as an explicit `menon_reprojection` block in `scene_config`, with both scene-space and inverse-mapped raw-world variants, and render it as a second labeled camera arrow in the prototype viewer instead of re-implementing Menon's device-registry math inside the browser.
- **Rationale:** The user needs a visual, auditable comparison between the prototype's calibration-world projection and the pose Menon tracks are actually using for reprojection. Keeping the Menon-derived pose in the backend config makes `/api/config` the source of truth, keeps the viewer simple, and avoids silently diverging from Menon's current device-layer camera offsets for living-room.
- **References:** `testpipelines/yolo26-seg-depth-3d/prototype_calibration.py`, `testpipelines/yolo26-seg-depth-3d/scene_stream.py`, `testpipelines/yolo26-seg-depth-3d/viewer/src/App.tsx`, `/home/mayor/Menon/src/features/path-visualization/ReprojectionCameraRegistry.js`, `/home/mayor/Menon/public/config/virtual-devices.json`

- **Date:** 2026-03-22
- **Author:** Codex
- **Area:** Prototype Pipeline / 3D Viewer / Coordinate Space Parity
- **Decision:** Keep the scene-frame payload exported by `testpipelines/yolo26-seg-depth-3d/` in raw calibrated world space, but include explicit alignment metadata (`align.matrix`, `scene_per_m`, `floor_y`) in `scene_config` and let the viewer toggle between `Raw World` and `Menon Scene` at render time.
- **Rationale:** The raw packet is the auditable backend truth and should not silently change shape or units. Menon parity requires the client-side scene-unit scale and alignment transform, not a different backend export. Carrying both in the config lets operators compare the prototype's direct world projection against the Menon-aligned scene without duplicating geometry payloads or obscuring which layer introduced a rotation/scale mismatch.
- **References:** `testpipelines/yolo26-seg-depth-3d/scene_stream.py`, `testpipelines/yolo26-seg-depth-3d/prototype_calibration.py`, `testpipelines/yolo26-seg-depth-3d/viewer/src/App.tsx`, `noesis/calibration/geometry.py`, `/home/mayor/Menon/src/features/calibration/CoordinateTransform.js`

- **Date:** 2026-03-21
- **Author:** Codex
- **Area:** Prototype Pipeline / 3D Viewer / Scene Export
- **Decision:** Keep the new live 3D viewer as a standalone DS8 testpipeline under `testpipelines/yolo26-seg-depth-3d/` instead of extending `oai2-fe`. The prototype exports one throttled calibrated-world structured RGBD grid (`5 Hz`, `160x90` default) plus person-centric projected detections over a prototype-local same-origin WebSocket/HTTP server, and the browser derives both point-cloud and mesh modes from that single packet.
- **Rationale:** The request was explicitly for a new testpipeline, not a main-app feature. Reusing the existing `yolo26-seg-depth` prototype preserves the agreed DS8 depth/object-depth path, while keeping the viewer local avoids coupling the main frontend to an experimental transport. Sending one structured RGBD grid keeps the backend contract simple, supports both render modes without duplicate payloads, and keeps failure behavior explicit when calibration or depth is unavailable.
- **References:** `testpipelines/yolo26-seg-depth-3d/main.py`, `testpipelines/yolo26-seg-depth-3d/pipeline.py`, `testpipelines/yolo26-seg-depth-3d/scene_stream.py`, `testpipelines/yolo26-seg-depth-3d/viewer/src/App.tsx`, `testpipelines/yolo26-seg-depth-3d/README.md`

- **Date:** 2026-03-16
- **Author:** Codex
- **Area:** Calibration / Offline Depth Registration Operations
- **Decision:** The canonical way to build `config/depth_registration.json` is an offline RTSP-backed fit that uses the live MapAnything service as the dense reference source and a temporal-stability mask to suppress moving foreground pixels. The builder may run its DAv2 side on CPU to avoid VRAM contention with the MapAnything service, but the runtime path remains unchanged and GPU-first.
- **Rationale:** Registration generation is an offline calibration step, not part of the live DS8 runtime. Using the live MapAnything service keeps the artifact aligned with the actual reference model, while temporal stability filtering lets the build tolerate minor/static occupancy without inventing per-camera hand edits. Moving only the builder-side DAv2 pass to CPU avoids unnecessary GPU OOM during artifact generation without creating a second runtime depth path.
- **References:** `scripts/build_depth_registration.py`, `services/mapanything_svc/server.py`, `docs/MapAnything_Depth.md`, `config/depth_registration.json`

- **Date:** 2026-03-16
- **Author:** Codex
- **Area:** Runtime / Baseline Tracking / Depth Registration
- **Decision:** Baseline non-`v3dt` DS8 tracking uses a prebuilt, read-only DAv2→MapAnything room-registration artifact keyed by camera. The runtime loads it before activation, validates calibration/model fingerprints for every enabled camera, and applies the resulting monotonic piecewise range mapping only inside the pose-ray depth observation path. Raw `NOESIS.OBJECT_DEPTH` remains unchanged; `track.world` remains the only authoritative world output.
- **Rationale:** The remaining room-relative error was dominated by forward/back range bias, not left/right geometry. Registering raw DAv2 anchor range into MapAnything-aligned room range preserves the agreed pose+depth estimator architecture, improves room-relative placement without inventing per-camera manual fudge factors, and keeps MapAnything as an offline reference rather than a runtime fallback.
- **References:** `noesis/calibration/depth_registration.py`, `noesis/calibration/depth_registration_builder.py`, `scripts/build_depth_registration.py`, `noesis/ds8_runtime.py`, `noesis/pipelines/hooks.py`, `docs/DS8_metadata_contracts.md`

- **Date:** 2026-03-12
- **Author:** Codex
- **Area:** Runtime / Baseline Tracking / Pose+Depth World Estimation
- **Decision:** In baseline (non-`v3dt`) DS8 mode, use one canonical person-anchor estimator in `hooks.py`: pose-derived image anchor when available, otherwise the person mask/depth image anchor from `NOESIS.OBJECT_DEPTH.anchor_uv`. Keep an always-on DAv2 depth-tracking lane (`vits`, `518x294`, batch 3, every 2 frames) as a concurrent range observation on that same current-anchor ray, and fuse floor + depth observations inside the canonical per-track world-state update. The runtime must fail fast if the depth-tracking branch or `noesis_depth_meta_ext` is unavailable, and downstream BEV/trail consumers must use the resulting canonical `track.world` instead of running a second estimator.
- **Rationale:** The user’s actual problem was far-camera drift in room-relative tracking plus missing world tracks whenever pose failed despite strong person detections. Treating DAv2 as extra annotations or keeping pose as the only admissible current anchor would not materially change that behavior. Accepting `anchor_uv` as another first-class current-frame person anchor preserves pose as the strongest source when it exists, keeps everything inside one estimator, avoids BEV-side workarounds, and removes the unacceptable “seg person but no world track” gap without reintroducing bbox-bottom synthesis.
- **References:** `noesis/depth_tracking_materialization.py`, `noesis/ds8_runtime.py`, `noesis/pipelines/ds8_pipeline.py`, `noesis/pipelines/hooks.py`, `native/noesis_depth_meta_ext.cpp`, `docs/DS8_api_contracts_ws.md`, `docs/DS8_metadata_contracts.md`

- **Date:** 2026-03-15
- **Author:** Codex
- **Area:** Runtime / Baseline Tracking / DAv2 Tensor Extraction
- **Decision:** Baseline DAv2 extraction in the main DS8 runtime must bypass both the Service Maker Python tensor wrapper path and the old full-frame host-copy path. The canonical implementation probes `depth_tracking_fullframe` directly, configures the depth SGIE as a raw tensor-meta branch (`network-type=100`, `output-tensor-meta=1`, `disable-output-host-copy=1`), and uses a native bridge (`noesis_depth_tracking_tensor_ext`) to walk `NvDsInferTensorMeta` from `FrameMetadata`, read `out_buf_ptrs_dev`, align the depth map to canonical frame size on the GPU, and retain the aligned result in a device-owned frame object. The only remaining host reads in the baseline DAv2 path are per-object ROI copies during object-depth fusion because DeepStream still exposes segmentation masks to us as host-side `NvOSD_MaskParams`.
- **Rationale:** Sustained baseline reads through `TensorOutputUserMetadata.as_tensor_output().get_layers()` were reproducibly segfaulting inside `libnvds_service_maker.so`, and leaving `network-type=1` on a device-only DAv2 branch reintroduced a second crash inside DeepStream's classifier postprocessor (`ClassifyPostprocessor::parseAttributesFromSoftmaxLayers`). Matching NVIDIA's own raw-tensor sample configs with `network-type=100` removes that unsupported postprocess path, while on-device alignment eliminates the largest new DAv2 host copy without changing the agreed pose+depth estimator or reintroducing an alternate runtime route.
- **References:** `native/noesis_depth_tracking_tensor_ext.cpp`, `scripts/build_noesis_depth_tracking_tensor_ext.sh`, `noesis/pipelines/hooks.py`, `noesis/pipelines/ds8_pipeline.py`, `noesis/ds8_runtime.py`, `tests/test_depth_tracking_frame_processor.py`

- **Date:** 2026-05-30
- **Author:** Codex
- **Area:** Test Harnesses / Model Validation / Pose
- **Decision:** Added a dedicated quick-validation harness `testpipelines/deimv2-wholebody49/` (following the exact yolo26-pose/ structure and DS8 Service Maker patterns) for the PINTO DEIMv2-Wholebody49 (49-keypoint whole-body) model. It consumes all three current sources via nvmultiurisrcbin, produces a configurable 2x2 (or 1x3) nvmultistreamtiler mosaic, runs the model via nvinfer (engine preferred, tensor-meta output), and drives rich nvosd diagnostics (bboxes + 49-pt skeleton lines + circles + per-tile counts/labels) from a python BatchMetadataOperator overlay. The harness was live-tested end-to-end (graph construction, linking, 3-source playback, tiler mosaic, probe firing, drawing, timed clean shutdown) using existing batch-3 engines as structural stand-ins; real model artifacts drop in with only output-blob-names adjustment after one --debug run.
- **Rationale:** User explicitly requested a test pipeline in testpipelines/ using the other harnesses in that folder as the "how to quickly build" reference, with the user's current 3 sources, ONNX/TensorRT, GPU-only, and proper nvosd-style diagnostics for the 49 kpts. This location and pattern are the sanctioned way to do rapid model bring-up in the repo (see multiple design entries referencing testpipelines/* prototypes). It stays completely separate from the canonical noesis/ DS8 runtime until/unless the model is promoted.
- **References:** `testpipelines/deimv2-wholebody49/`, `testpipelines/yolo26-pose/` (primary pattern), PINTO_model_zoo 488_DEIMv2-Wholebody49 (EDGES + CLASSES), `plans/DS8/ds8_master_work_orders.md` (test harnesses used for experimental models)

- **Date:** 2026-03-15
- **Author:** Codex
- **Area:** Runtime / Baseline Tracking / Depth Fusion Tuning
- **Decision:** Baseline pose+DAv2 fusion now weights depth using the support of the chosen lower-body / torso anchor band (`anchor_sample_count`, `anchor_valid_fraction`) instead of whole-mask support alone, lowers the acceptable support floor for far-camera anchors, gives trusted fused updates a modest alpha bump, and shortens `anchor_hold` to `0.40s`.
- **Rationale:** The remaining issue was weak depth influence at distance plus visible trail drift from stale held world points. Whole-mask support penalized far-camera people even when the actual anchor band was clean, and a longer hold window let stale positions linger in BEV. Using anchor-band support keeps depth active where it is geometrically useful, while the shorter hold TTL preserves brief-occlusion resilience without letting old positions drift across the floorplan.
- **References:** `noesis/pipelines/hooks.py`, `noesis/metadata/object_depth.py`, `tests/test_analytics_telemetry_hook.py`, `docs/DS8_metadata_contracts.md`

- **Date:** 2026-03-15
- **Author:** Codex
- **Area:** Telemetry / BEV World Mode / Trail Ownership
- **Decision:** In world BEV mode, the canonical `track.world` head point from the backend is authoritative and must not be smoothed again inside `BevRenderer`. The backend still owns trail history (`trail_smoothing_owner=backend`), but `bev_world_points_smoothed=false` explicitly means the BEV payload head points are direct canonical track-world samples. World-mode BEV also skips `anchor_hold` points so stale held positions do not render as drifting trail heads.
- **Rationale:** The measured BEV drift came from a second world-space smoother and trail-stage EMA operating after the fused track world filter, not from the fused estimator itself. Removing that extra stage reduced head-point drift against canonical `track.world` from large scene-scale offsets to zero in settled windows while keeping producer-owned trail history for the frontend.
- **References:** `noesis/telemetry/bev.py`, `oai2-fe/src/components/BevView.tsx`, `tests/test_bev_renderer_world_smoothing.py`, `docs/DS8_api_contracts_ws.md`

- **Date:** 2026-03-16
- **Author:** Codex
- **Area:** Runtime / BEV / Occlusion Motion Prediction
- **Decision:** Disable inferred motion across occlusion gaps for both the trail overlay path and the baseline world estimator while room-scale BEV alignment is being tuned. Concretely: default `gap_predict_ttl_s=0` in trail config and stop constant-velocity extrapolation in `_predict_world_state()`.
- **Rationale:** The explicit occlusion-gap trail predictor and the backend constant-velocity extrapolation were both capable of producing rebound/oscillation artifacts that confused BEV validation. The user’s current priority is trustworthy room-relative placement, not graceful motion hallucination across dropouts, so the canonical path should use direct measurements plus short `anchor_hold` only until a redesigned predictor is justified.
- **References:** `config/infer.yaml`, `noesis/pipelines/hooks.py`, `tests/test_analytics_telemetry_hook.py`

- **Date:** 2026-03-11
- **Author:** Codex
- **Area:** Prototype Pipeline / Semantic Segmentation / Utility Outputs
- **Decision:** Build the room-layout utility around `SegFormer-B5` fine-tuned on ADE20K, but export a DS8-specific ONNX/TensorRT wrapper that collapses the full 150-class output into compact layout groups (`other`, `wall`, `floor`, `ceiling`, `window`, `door`, `stairs`) at `1024x576`, then write averaged probability bundles plus binary masks per source.
- **Rationale:** Official ADE20K leaderboards show stronger semantic models such as Mask2Former, but the current workspace has a clean `transformers -> ONNX -> TensorRT -> nvinfer network-type=2` path for SegFormer and an official mmdeploy TensorRT support lane for mmseg SegFormer, while Mask2Former would require a different deployment stack and custom-op handling. Exporting only the structural groups keeps the DS8 segmentation output small enough for a periodic utility, preserves probability maps for downstream thresholding, and avoids carrying a full 150-channel tensor through DeepStream when Noesis only needs room-layout constraints.
- **References:** `testpipelines/room_layout_seg/model_setup.py`, `testpipelines/room_layout_seg/pipeline.py`, `testpipelines/room_layout_seg/probes.py`, `https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640`, `https://raw.githubusercontent.com/open-mmlab/mmsegmentation/main/configs/segformer/metafile.yaml`, `https://raw.githubusercontent.com/open-mmlab/mmsegmentation/main/configs/mask2former/metafile.yaml`, `https://mmdeploy.readthedocs.io/en/latest/04-supported-codebases/mmseg.html`

- **Date:** 2026-03-06
- **Author:** Codex
- **Area:** Metadata / Prototype Pipeline / Depth Fusion
- **Decision:** Canonical DS8 object-level depth attachment uses a new `NOESIS.OBJECT_DEPTH` JSON user-meta payload attached through a native Service Maker bridge (`noesis_depth_meta_ext`) instead of a Python-only side channel.
- **Rationale:** Service Maker Python wrappers still do not expose a supported object user-meta append path for arbitrary payloads. A native bridge keeps the prototype aligned with the existing DS8 pose-meta pattern, preserves portability into the main app, and lets downstream consumers distinguish metric-vs-relative depth via explicit payload fields.
- **References:** `docs/DS8_metadata_contracts.md`, `native/noesis_depth_meta_ext.cpp`, `noesis/metadata/object_depth.py`

- **Date:** 2026-03-07
- **Author:** Codex
- **Area:** Prototype Pipeline / Depth Fusion / Geometry
- **Decision:** The seg+depth prototype uses a deterministic linear DS8 graph (`depth_infer -> seg_preproc -> seg_infer -> fusion -> overlay`) with a bounded internal aligned-depth cache between depth capture and object fusion. All object-depth sampling happens in canonical post-mux frame space and only from decoded instance masks; bbox fallback is not permitted.
- **Rationale:** The earlier prototype mixed source-native and DS8 frame dimensions and let rendering concerns blur fusion failures, which made left/right FoV depth bugs hard to localize. Capturing aligned full-frame depth once, passing it forward through an internal cache, and enforcing segment-only sampling in one canonical coordinate space keeps geometry semantics explicit and makes transform/mask failures observable instead of silently changing sampling mode.
- **References:** `testpipelines/yolo26-seg-depth/debug_probes.py`, `testpipelines/yolo26-seg-depth/pipeline.py`, `native/noesis_depth_meta_ext.cpp`, `docs/DS8_metadata_contracts.md`

- **Date:** 2026-03-10
- **Author:** Codex
- **Area:** Prototype Pipeline / Performance / Benchmarking
- **Decision:** The single-stream seg+depth prototype drops `nvmultistreamtiler`, defaults to mask+text overlay without bbox drawing, and exposes explicit benchmark controls for seg-only mode, depth cadence, and depth input shape (`--disable-depth`, `--depth-every-n-frames`, `--depth-input-shape`, `--show-bbox`) while keeping the same canonical object-depth metadata contract.
- **Rationale:** The user needed to isolate why the prototype looked much heavier than the main runtime without changing object-depth semantics first. Removing the unused single-stream tiler and default bbox drawing cuts avoidable display overhead, while the new CLI/runtime stats make it possible to compare seg-only versus seg+depth and to quantify what depth reuse changes before shrinking model fidelity.
- **References:** `testpipelines/yolo26-seg-depth/main.py`, `testpipelines/yolo26-seg-depth/pipeline.py`, `testpipelines/yolo26-seg-depth/debug_probes.py`, `testpipelines/yolo26-seg-depth/README.md`

- **Date:** 2026-03-10
- **Author:** Codex
- **Area:** Prototype Pipeline / Depth Fusion / Spatial Geometry
- **Decision:** Keep `Depth Anything V2 metric vits` as the runtime depth model, but split raw object-depth statistics from a new person-only spatial projection path that derives a mask-based foot anchor plus calibration-aware world points using the existing DS8 calibration stack. The prototype uses the lower-body depth sample when it agrees with the floor-plane projection and otherwise guards the final world point with the floor intersection. Its standalone calibration resolver also mirrors the current DS8 baseline runtime behavior by normalizing live projection snapshots to `unit_scale=1.0` when the selected `camera_calibration.json` extrinsics are already meter-scale.
- **Rationale:** The user needs spatially credible person positioning for later 3D tracking integration more than perfect monocular meters. Reusing the DS8 calibration stack avoids ad-hoc per-camera scale tweaks, while the floor-guard policy makes the output more useful for room-relative positioning when monocular metric depth drifts. Raw object-depth fields stay intact so downstream consumers can inspect the original DAv2 signal separately from the guarded spatial point, and matching the runtime’s meter-scale snapshot semantics avoids shrinking world coordinates by `align.units.s_obj_to_m` a second time.
- **References:** `testpipelines/yolo26-seg-depth/debug_probes.py`, `testpipelines/yolo26-seg-depth/prototype_calibration.py`, `noesis/calibration/geometry.py`, `noesis/metadata/object_depth.py`, `docs/DS8_metadata_contracts.md`

- **Date:** 2026-03-07
- **Author:** Codex
- **Area:** Telemetry / BEV Trails / World-Path Contract
- **Decision:** In world BEV mode (`frame_mode=world`), producer-side BEV telemetry owns trail history, but this 2026-03-07 point-smoothing decision is superseded by the 2026-03-15 canonical-world update below. Current contract truth is `trail_smoothing_owner=backend` with direct canonical `track.world` head points and `bev_world_points_smoothed=false`.
- **Rationale:** The original fix correctly moved smoothing ownership away from the frontend, but later live tuning showed the remaining drift was coming from a second backend BEV world-space smoother layered on top of the already-filtered `track.world`. The durable rule is now: backend owns trail history, while head points in world mode are the canonical track-world samples with no extra BEV low-pass.
- **References:** `noesis/telemetry/bev.py`, `oai2-fe/src/components/BevView.tsx`, `config/infer.yaml`, `docs/DS8_api_contracts_ws.md`

- **Date:** 2026-03-09
- **Author:** Codex
- **Area:** Telemetry / BEV World Anchors / Degraded Fallbacks
- **Decision:** When the current-frame canonical person-anchor observations degrade, reuse the last reliable anchor briefly (`anchor_hold`) and then fail closed instead of projecting bbox-bottom onto the floor. The admissible current-frame anchors are pose-derived image anchors and the person mask/depth image anchor from `NOESIS.OBJECT_DEPTH.anchor_uv`.
- **Rationale:** The worst BEV placement errors were coming from degraded bbox-bottom floor intersections for far or partially occluded people, especially seated/support-surface cases where a short-lived loss of current anchor evidence should not relocate the person behind furniture or off the floorplan. Preserving a recent reliable anchor reduces teleports, and refusing to synthesize a new floor point keeps the canonical world path honest when the backend no longer has enough geometric evidence.
- **References:** `noesis/pipelines/hooks.py`, `tests/test_analytics_telemetry_hook.py`, `docs/DS8_api_contracts_ws.md`

- **Date:** 2026-03-10
- **Author:** Codex
- **Area:** Telemetry / UI / BEV Trails
- **Decision:** When BEV world-mode smoothing is backend-owned, publish the backend trail polylines in `bev-frame.trails`, have `oai2-fe` render those directly instead of reconstructing trail history from `footpoints`, and bypass the BEV trail-stage EMA/speed clamp so the producer does not smooth the same motion twice.
- **Rationale:** The remaining drift/arcing regression came from split trail-history ownership plus double smoothing: backend stabilized the head points, then BEV trail history applied a second EMA/speed clamp, and the frontend sampled its own tail. That stack produced visible post-stop drift and loops. Publishing the producer trail state and sampling already-smoothed points directly removes the extra lag and keeps the dashboard trail identical to the backend BEV path.
- **References:** `noesis/telemetry/bev.py`, `oai2-fe/src/components/BevView.tsx`, `docs/DS8_api_contracts_ws.md`, `tests/test_bev_renderer_world_smoothing.py`

- **Date:** 2026-03-11
- **Author:** Codex
- **Area:** Telemetry / BEV Trails / Identity Ownership
- **Decision:** BEV world-path smoothing and trail history are keyed by tracker-local identity (`trackerId` when present, otherwise `stableId`) instead of stable-first identity. `stableId` remains metadata for labels/colors and for cross-camera semantics, but it does not own per-camera trail history.
- **Rationale:** `nvOSD` trail state is tracker-local (`sensor_id + track_id`). BEV was previously collapsing history and smoother state onto `stableId`, which allowed StableID remaps/reuse to splice an older trajectory onto the current head point and produced detached arcs even when current positions were correct. Matching BEV state ownership to the nvOSD path removes that hidden extra logic and keeps history tied to one concrete tracker trajectory.
- **References:** `noesis/telemetry/bev.py`, `noesis/pipelines/hooks.py`, `docs/DS8_api_contracts_ws.md`, `tests/test_bev_renderer_world_smoothing.py`

- **Date:** 2026-02-15
- **Author:** Codex
- **Area:** Depth / Floorplan
- **Decision:** For kitchen camera floorplans, replace legacy NaN→global-min fill with a floor-plane reconstruction pass that (1) estimates the floor plane from low-percentile observed floor cells, (2) fills occluded/empty (and sparse floor-like) cells from that plane, and (3) preserves observed obstacle heights. Keep non-kitchen cameras on legacy behavior for rollout safety.
- **Rationale:** Kitchen height maps showed occlusion-related floor artifacts (wavy/amorphous surfaces) around island blind spots. A kitchen-scoped global floor-plane fill removes those artifacts while preserving current contracts (`floorplan_response.height`) and minimizing regression risk on other cameras.
- **References:** `geometry/depth_source.py`, `tests/test_floorplan_height_fill.py`, `plans/DS8/ds8_migration_checklist_ds8_runtime.md`

- **Date:** 2026-02-09
- **Author:** Codex
- **Area:** Menon Integration / Producer Canonicalization / Units
- **Decision:** Decouple producer projection units from scene units by adding `align.scene_units.s_obj_to_m` and applying `meters -> scene units` conversion before `align.matrix` when stamping `menon_scene`.
- **Rationale:** Live trails were valid but compressed near origin because producer outputs were meter-scale while Menon scene geometry used larger OBJ units; this fixes scale without moving normalization ownership back to Menon.
- **References:** `calibration_bundle.py`, `noesis/ds8_runtime.py`, `noesis/pipelines/hooks.py`, `noesis/telemetry/bev.py`, `plans/DS8/menon_world_unification/m5_4_scene_unit_scaling_alignment_2026-02-09.md`

- **Date:** 2026-02-09
- **Author:** Codex
- **Area:** Menon Integration / Visual Stability
- **Decision:** Add Menon render-layer path stabilization (speed-gated jump clamp + smoothing) in `PathVisualizer` while keeping producer telemetry unchanged.
- **Rationale:** Remaining visual jitter after producer fixes is dominated by transient outlier updates; render-layer stabilization improves operator usability without contract churn.
- **References:** `Menon/src/features/path-visualization/PathVisualizer.js`, `plans/DS8/menon_world_unification/m5_5_menon_path_motion_stabilization_2026-02-09.md`

- **Date:** 2026-02-09
- **Author:** Codex
- **Area:** Menon Integration / Producer Diagnostics / Projection Triage
- **Decision:** Add producer-owned world projection reason telemetry (`world_reason` per track and `world_reason_counts` per camera in stats) so geometry failures are explicit and profile-agnostic.
- **Rationale:** Live symptoms (origin clustering / sparse geometry) need deterministic payload-level reason codes to separate rendering issues from upstream projection failures.
- **References:** `noesis/pipelines/hooks.py`, `noesis/ds8_runtime.py`, `plans/DS8/menon_world_unification/m5_2_world_reason_diagnostics_2026-02-09.md`

- **Date:** 2026-02-09
- **Author:** Codex
- **Area:** Menon Integration / Producer Projection Recovery
- **Decision:** Add producer-side lower-row ray recovery scan and max-distance gating to the floor projection path when bbox bottom-center ray misses.
- **Rationale:** Dominant `ray_no_floor_intersection` failures under live baseline runtime reduced usable scene-world track output; recovery scan improves projection yield without relaxing consumer contract strictness.
- **References:** `noesis/pipelines/hooks.py`, `plans/DS8/menon_world_unification/m5_3_ray_recovery_projection_2026-02-09.md`

- **Date:** 2026-02-09
- **Author:** Codex
- **Area:** Menon Integration / Coordinate Ownership / Producer-Consumer Contract
- **Decision:** Move canonical scene-frame normalization ownership from Menon to Noesis producer outputs, and enforce strict `menon_scene` geometry acceptance in Menon consumers (drop non-scene tracks with diagnostics while preserving occupancy updates).
- **Rationale:** Cross-camera 3D association and ReID in Noesis require shared coordinates upstream. Producer-owned canonicalization removes duplicate transforms and prevents downstream double-transform ambiguity.
- **References:** `noesis/pipelines/hooks.py`, `noesis/ds8_runtime.py`, `Menon/src/services/WebSocketClient.js`, `plans/DS8/menon_world_unification/m5_1_noesis_scene_producer_cutover_2026-02-09.md`

- **Date:** 2026-02-09
- **Author:** Codex
- **Area:** Menon Integration / Tracking Normalization
- **Decision:** Keep phase-1 ingestion on `stats` and implement Menon-side normalization in `WebSocketClient` with `stable_id` precedence, deterministic cross-camera dedup, explicit `menon_scene` stamping after alignment, timestamp derivation from `payload.timestamp`, and reasoned drop/RPC diagnostics.
- **Rationale:** This closes contract-critical phase-1 gaps with low transport churn and preserves existing occupancy/path interfaces while making failures diagnosable.
- **References:** `plans/DS8/menon_world_unification/m1_1_normalization_spec_complete_2026-02-09.md`, `plans/DS8/menon_world_unification/contracts.md`, `/home/mayor/Menon/src/services/WebSocketClient.js`

- **Date:** 2026-02-09
- **Author:** Codex
- **Area:** Planning / Program Governance
- **Decision:** Create a dedicated DS8 project workspace at `plans/DS8/menon_world_unification/` as the authority for this workstream's roadmap, checklists, contracts, validation evidence, and handoffs.
- **Rationale:** The world-unification effort spans multiple sessions and agents. A single local planning subtree with explicit rules and append-only timeline artifacts reduces context loss and prevents undocumented scope drift.
- **References:** `plans/DS8/menon_world_unification/AGENTS.md`, `plans/DS8/menon_world_unification/program_roadmap.md`, `plans/DS8/menon_world_unification/work_order.md`, `plans/DS8/menon_world_unification/handoff.md`

- **Date:** 2026-02-09
- **Author:** Codex
- **Area:** Menon Integration / Contracts / Telemetry Compatibility
- **Decision:** Close phase-2 contract parity with additive DS8 metadata updates (`track_id`, `timestamp_us`, and `contract_version(s)`) while keeping Menon enforcement warn-only for version markers during the `stats`-primary rollout.
- **Rationale:** This provides deterministic identity/timing normalization and safer upgrade diagnostics without forcing an immediate `tracking` transport migration or breaking existing consumers.
- **References:** `noesis/pipelines/hooks.py`, `noesis/telemetry/publishers.py`, `noesis/ds8_runtime.py`, `noesis/calibration/manager.py`, `plans/DS8/menon_world_unification/m2_1_cross_system_consistency_pass_2026-02-09.md`

- **Date:** 2026-02-09
- **Author:** Codex
- **Area:** Menon Integration / Tracking Transport Migration
- **Decision:** Execute Option-1 migration with dual-lane shadow ingestion: normalize both `stats` and `tracking`, publish a runtime-selectable active lane, emit lane parity diagnostics, and keep tracking contract-version checks warn-only through M4.1.
- **Rationale:** This supports safe transport migration with measurable parity before hard cutover, while preserving existing UI contracts and minimizing regression blast radius.
- **References:** `Menon/src/services/WebSocketClient.js`, `plans/DS8/menon_world_unification/m4_1_tracking_primary_migration_2026-02-09.md`, `plans/DS8/menon_world_unification/contracts.md`

- **Date:** 2026-02-04
- **Author:** Codex
- **Area:** ReID / REST
- **Decision:** Introduce a StableID alias map for soft merges. Canonical defaults to the lower ID, but the REST API accepts an explicit canonical (to apply `preferred_canonical` from suggestions). Gallery entries now store `(ts, emb)` so merges keep the newest N; `pose_gallery` already uses timestamped tuples. Co-presence guardrail (10 min default) uses `(low, high)` keys because co-presence is tracked per-camera at the call site. **DS8 defaults aliases on; DS7 unchanged** because DS8 explicitly passes `aliases_enabled=True`. Use a single `_lock` for update + alias operations to avoid deadlocks. Suggestions apply a mutual-nearest-neighbor filter. `sid_global_first_seen` is cleaned up in `_purge_sid_state()` and `_alloc_sid()` to avoid stale timestamps on reused IDs. `unset_alias()` is an undo and does not purge state. Alias history is capped at `alias_history_max` (default 1000). Zone-state dwell timers are not remapped on alias (v1 limitation).
- **Rationale:** Provide reversible, operator-controlled merges with auditability and bounded memory while keeping concurrency safe and suggestions conservative.
- **References:** `reid/stable_id_manager.py`, `noesis/server/reid_api.py`, `noesis/ds8_runtime.py`, `docs/DS8_api_contracts_rest.md`

- **Date:** 2026-02-01
- **Author:** Codex
- **Area:** ReID / Pose
- **Decision:** Feed YOLO26 pose feature meta into StableIDManager as a gated secondary similarity signal (pose-only fallback when embeddings are missing) and store pose vectors in bounded RAM (deque + TTL + global cap) with no disk persistence; add a native extractor to read pose meta from `NvDsObjectMeta`.
- **Rationale:** Pose ratios can reinforce identity when appearance embeddings are weak or missing, but pose features are high‑volume; bounded in‑memory storage prevents resource growth. Service Maker bindings do not expose object user meta lists, so a native bridge is required to read the pose payload safely.
- **References:** `reid/stable_id_manager.py`, `noesis/pipelines/hooks.py`, `native/noesis_pose_meta_ext.cpp`, `docs/DS8_metadata_contracts.md`

- **Date:** 2026-02-01
- **Author:** Codex
- **Area:** Telemetry / Stats
- **Decision:** Add an always-on DS8 latency heartbeat using DeepStream’s built-in latency measurement, sampling at `osd:src` via a Service Maker `Probe` and exporting rolling p50/p95/max to the WebSocket `stats` payload.
- **Rationale:** Keeps a low-overhead, continuously available end-to-end latency signal without introducing DS7 pad probes or CPU branches. Service Maker buffer probes require attaching to an output pad, so `osd:src` is the last stable tap point before the sink tee/RTSP branches. Uses `ctypes` to call `libnvds_meta.so:nvds_measure_buffer_latency` to avoid the `pyds.nvds_measure_buffer_latency` wrapper’s unconditional stdout printing.
- **References:** `noesis/telemetry/latency_metrics.py`, `noesis/pipelines/ds8_pipeline.py`, `noesis/ds8_runtime.py`, `docs/DS8_api_contracts_ws.md`, `/opt/nvidia/deepstream/deepstream/sources/includes/nvds_latency_meta.h`

- **Date:** 2026-02-01
- **Author:** Codex
- **Area:** Metadata / Pose
- **Decision:** Enable host tensor copy for the YOLO26 pose SGIE, default pose tensor decoding to the manual DLPack path (torch off), and share a per-frame pose tensor cache between feature extraction and keypoint overlay so each tensor is consumed once.
- **Rationale:** Pose hooks run twice per object (features + overlay). Consuming the same tensor via torch DLPack was the highest-risk path for DeepStream user-meta corruption and segfaults. Host copy plus a single decode per frame eliminates ownership hazards while preserving pose features and overlays.
- **References:** `pipelines/config_infer_secondary_yolo26_pose.ini`, `noesis/pipelines/hooks.py`, `plans/DS8/ds8_migration_checklist_hooks.md`

- **Date:** 2026-01-31
- **Author:** Codex
- **Area:** Depth / WebSocket / UI
- **Decision:** Compute and attach optional MapAnything normals to `ma_depth_response` payloads on demand, using camera-space normals encoded as float16.
- **Rationale:** Normals are needed for the depth drawer without introducing a streaming telemetry channel; computing on demand keeps bandwidth and compute bounded while reusing existing depth RPC flow.
- **References:** `geometry/depth_source.py`, `noesis/ds8_runtime.py`, `oai2-fe/src/components/DepthDrawer.tsx`, `docs/DS8_api_contracts_ws.md`, `docs/DS8_metadata_contracts.md`

- **Date:** 2026-01-30
- **Author:** Codex
- **Area:** Metadata / Pose
- **Decision:** Attach YOLO26 pose features as object‑level user meta (`NOESIS.POSE_FEATURES`) via a small pybind11 helper that uses `nvds_add_user_meta_to_obj`, storing a compact JSON payload.
- **Rationale:** Service Maker Python wrappers do not expose `obj_user_meta_list`/append APIs, so a native shim is the most reliable way to attach per‑object metadata without introducing DS7 pad probes or CPU branches.
- **References:** `noesis/pipelines/hooks.py`, `native/noesis_pose_meta_ext.cpp`, `docs/DS8_metadata_contracts.md`

- **Date:** 2026-01-30
- **Author:** Codex
- **Area:** Telemetry / Floorplan
- **Decision:** Apply the same extrinsics‑driven image‑axis flips used by BEV homography to floorplan generation, and tag cached floorplans with `image_flip` for invalidation.
- **Rationale:** BEV flips image axes when the camera frame is mirrored relative to world axes; without applying the same flip in floorplan generation, the top‑down background appears left↔right mirrored while tracks move correctly.
- **References:** `geometry/depth_source.py`, `noesis/telemetry/bev.py`

- **Date:** 2026-01-29
- **Author:** Codex
- **Area:** Pipeline / PGIE / Segmentation / GPU-first
- **Decision:** Fuse YOLO26-seg models to output a single tensor (`output0`) that already contains per-detection masks (fixed 64x64 flattened) for the top 30 detections, remove the prototype tensor from graph outputs, and enable `disable-output-host-copy=1` so DeepStream keeps outputs in device memory.
- **Rationale:** The unfused graph (`output0` + `output1` proto) triggers a large device→host copy for `output1` and forces CPU-side mask composition, which violates the DS8 GPU-first policy and increases overhead. Fusing mask compose into the TensorRT engine keeps the heavy work on-GPU and allows the custom parser to copy only the minimal slices required to populate DeepStream metadata. Capping to 30 detections bounds ROIAlign workload and yielded a large utilization drop on YOLO26n (reported ~10% GPU in DS8 runs vs ~80–90% when composing 300 masks).
- **References:** `utils/fuse_yolo26_seg_onnx.py`, `pipelines/config_infer_primary_yolo26_seg.template.ini`, `pipelines/nvdsinfer_yolo26_seg/nvdsinfer_yolo26_seg.cpp`, `noesis/ds8_runtime.py`, `models/yolo26{n,s,m}-seg_fused.onnx`, `models/engines/yolo26{n,s,m}-seg_fused_b3_fp16.engine`

- **Date:** 2026-01-28
- **Author:** Codex
- **Area:** Pipeline / PGIE Profiles
- **Decision:** Add a `yolo26_seg` PGIE profile with a CLI-only `--size {n,s,m}` selector (default `m`) that materializes a size-specific PGIE INI under `build/` from a generic template and points the overlay at the chosen b3 engine.
- **Rationale:** Keeps the DS8 config surface clean (single template), avoids maintaining multiple static INIs, and guarantees the engine/parser paths are explicit and validated at runtime without DS7 fallback.
- **References:** `noesis/ds8_runtime.py`, `pipelines/config_infer_primary_yolo26_seg.template.ini`, `pipelines/nvdsinfer_yolo26_seg/`, `docs/DS8_testing_guide.md`

- **Date:** 2026-01-23
- **Author:** Codex
- **Area:** V3DT / Baseline / Naming
- **Decision:** Normalize the SV3DT baseline filenames to stable, short paths:
  `config/infer_v3dt_baseline.yaml`,
  `config/v3dt/nvtracker_v3dt_baseline.yml`,
  `config/v3dt/caminfo_baseline/`,
  `config/cameras_v3dt_baseline.yaml`,
  `config/archive/calibration_v3dt_baseline.json`,
  `config/dewarper_v3dt_baseline.txt`,
  `config/analytics_exclude_baseline.ini`.
- **Rationale:** The shortened baseline names avoid long, brittle filenames and make
  the locked baseline easier to reference across scripts, docs, and checklists.
- **References:** `plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md`,
  `plans/DS8/v3dt/README.md`, `plans/DS8/v3dt/AGENTS.md`

- **Date:** 2026-01-22
- **Author:** Codex
- **Area:** V3DT / Baseline / Calibration
- **Decision:** Lock the SV3DT baseline on the preview calibration + camInfo set:
  `config/archive/calibration_v3dt_baseline.json`,
  `config/v3dt/caminfo_baseline/`,
  tracker `config/v3dt/nvtracker_v3dt_baseline.yml`,
  and pipeline `config/infer_v3dt_baseline.yaml`.
  CamInfo uses `w2p`, `INVERT_E=0`, `Y_FLIP=1`, `WORLD_AXES=xzy`, `WORLD_SCALE=1`, model height 2.2m.
- **Rationale:** This configuration yields the lowest reprojection errors observed to date across
  family-room and kitchen and materially improves living-room stability; it is the first
  end-to-end set that reproduces tracks without SV3DT collapse.
- **References:** `plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md`, `plans/DS8/v3dt/iteration_notes_2026-01-22.md`,
  `diagnostics/trackdump_runs/rect774_pitch_m16_h220_k21_l15_live_20260122_225218/v3dt_report_20260122_225453.md`

- **Date:** 2026-01-22
- **Author:** Codex
- **Area:** V3DT / PGIE / Projection
- **Decision:** Keep PGIE aspect-ratio handling disabled for SV3DT runs:
  `maintain-aspect-ratio=0` and `symmetric-padding=0`.
- **Rationale:** When PGIE letterboxing is enabled, SV3DT projection math sees a
  double aspect correction (streammux + PGIE) which produces large reprojection
  errors and unstable 3D fits.
- **References:** `pipelines/config_infer_primary_yolo11_seg.ini`,
  `diagnostics/trackdump_runs/rect774_pitch_m15_h200_20260122_195943/v3dt_report_20260122_200148.md`

- **Date:** 2026-01-22
- **Author:** Codex
- **Area:** V3DT / Alignment / BEV
- **Decision:** Keep alignment scale in meters (`config/ply_alignment.json` `units.s_obj_to_m=1.0`).
- **Rationale:** Non-unit scaling (e.g., 0.010849) shrinks world translations by ~100×,
  corrupting BEV and pixel→world outputs when extrinsics are already in meters.
- **References:** `config/ply_alignment.json`, `noesis/metadata/mapanything_pose.py`,
  `plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md`

- **Date:** 2026-01-21
- **Author:** Codex
- **Area:** Bidnetpipe / Segmentation / Model Prep
- **Decision:** Bake ADE20K mean/std normalization and softmax into the exported BiSeNetV2 ONNX so nvinfer receives probability maps without requiring per-channel std in `nvinfer` config.
- **Rationale:** `nvinfer` supports scalar `net-scale-factor` and per-channel offsets but not per-channel std. Exporting an ONNX wrapper that applies `/255`, mean/std normalization, and softmax keeps preprocessing faithful to the BiSeNet training pipeline and allows `segmentation-threshold` to operate on probabilities.
- **References:** `Bidnetpipe/model_setup.py`, `Bidnetpipe/DOCS.md`, https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html, https://github.com/CoinCheung/BiSeNet

- **Date:** 2026-01-18
- **Author:** Codex
- **Area:** V3DT / Telemetry / BEV
- **Decision:** Prefer `image_base` (bbox3d base-center projection) for BEV/footpoint anchoring, then fall back to `image_foot`, then bbox bottom-center.
- **Rationale:** `image_foot` can land on a cuboid edge when SV3DT foot location is offset; using `image_base` aligns the projected 3D cuboid base with the person center while preserving safe fallbacks when the projection is missing or invalid.
- **References:** `noesis/pipelines/hooks.py`, `docs/DS8_metadata_contracts.md`, `docs/DS8_api_contracts_ws.md`

- **Date:** 2026-01-15
- **Author:** Codex
- **Area:** V3DT / Calibration / Dewarper
- **Decision:** Keep `nvdewarper` output at the rectified ChArUco K, but apply a **family-room-only intrinsics scale (2.7x)** in SV3DT camInfo to align projected cuboids.
- **Rationale:** Scaling both the dewarper output K and SV3DT camInfo made cuboid projection worse. Scaling camInfo only (with dewarper fixed) yields a near-ideal family-room projection (H ratio ~1.01, W ratio ~1.03). This isolates the correction to SV3DT projection math while preserving the undistorted image geometry.
- **References:** `config/cameras_preview_charuco_fr_rtsp_dewarp.yaml`, `config/dewarper_family_room_charuco_rtsp.txt`, `config/v3dt_preview_charuco_fr_rtsp_dewarp/camInfo_family-room.yml`, `diagnostics/v3dt_report_20260115_192145.md`, `plans/DS8/v3dt/iteration_notes_2026-01-15.md`
  _2026-01-22 (Codex): Superseded — current baseline removes the 2.7x camInfo scale hack and uses per-camera pitch preview extrinsics instead._

- **Date:** 2026-01-14
- **Author:** Codex
- **Area:** V3DT / Calibration / camInfo
- **Decision:** Allow `scripts/generate_v3dt_caminfo.py` to use an explicit `resolution` field in `config/cameras.yaml` intrinsics models when scaling K to streammux size, instead of assuming principal point is centered.
- **Rationale:** ChArUco intrinsics can yield a principal point offset from image center. Inferring base resolution from `2*cx/2*cy` mis-scales K when `cx/cy` are off-center, leading to projection drift. Explicit `resolution` keeps scaling consistent with the true calibration frame size.
- **References:** `scripts/generate_v3dt_caminfo.py`, `config/cameras_preview_charuco_fr.yaml`

- **Date:** 2026-01-14
- **Author:** Codex
- **Area:** V3DT / Calibration / Auto Tilt
- **Decision:** Add an optional image-space Y flip to the depth-plane fit used by the tilt preview tool (`--flip-image-y`), to align MapAnything depth (image Y-down) with Y-up world coordinates.
- **Rationale:** Auto-tilt runs were producing correct downward pitch but ~180° roll; flipping the image-space Y axis yields upright roll while preserving negative pitch, indicating the depth plane normals were computed in image Y-down coordinates.
- **References:** `noesis/calibration/tilt_preview.py`, `scripts/auto_tilt_from_depth.py`, `docs/DS8_v3dt_forensics.md`

- **Date:** 2026-01-13
- **Author:** Codex
- **Area:** V3DT / Calibration
- **Decision:** Add a depth-plane **tilt-only** preview tool that aligns the ground-plane normal to +Y while preserving yaw and camera center, writing to a separate preview calibration file.
- **Rationale:** We need a safe way to correct pitch/roll drift (tilt) without overwriting Menon’s calibration or perturbing yaw/translation. A preview file enables side-by-side camInfo comparisons before committing.
- **References:** `noesis/calibration/tilt_preview.py`, `scripts/auto_tilt_from_depth.py`, `docs/DS8_v3dt_forensics.md`, `plans/DS8/v3dt/README.md`

- **Date:** 2026-01-13
- **Author:** Codex
- **Area:** V3DT / Calibration / Telemetry
- **Decision:** Treat SV3DT world as **Z-up** (height = `bbox3d.zLen`) and add a camInfo axis-remap escape hatch (`NOESIS_V3DT_CAMINFO_WORLD_AXES`, e.g., `xzy` to swap Y/Z) to align Y-up calibration with DeepStream’s SV3DT frame.
- **Rationale:** Live telemetry showed `bbox3d.zLen ≈ 1.7m` while `bbox3d.yLen ≈ 0.7m`, matching model height and indicating Z is the vertical axis. DeepStream documentation describes world foot positions as 2D (X,Y) on the ground plane, which implies Z-up. The remap keeps calibration files intact while letting SV3DT receive the expected axis convention.
- **References:** `/tmp/ds_sv3dt_docs/gst-nvtracker.html`, `scripts/generate_v3dt_caminfo.py`, `noesis/pipelines/hooks.py`, `noesis/diagnostics/v3dt_forensics.py`, `plans/DS8/v3dt/status_summary_2026-01-12_meters_caminfo.md`

- **Date:** 2026-01-13
- **Author:** Codex
- **Area:** V3DT / Runtime Defaults
- **Decision:** Set V3DT camInfo defaults to the working baseline: `NOESIS_V3DT_AUTOGEN_CAMINFO=1`, `NOESIS_V3DT_CAMINFO_MATRIX_TYPE=w2p`, `NOESIS_V3DT_CAMINFO_INVERT_E=1`, `NOESIS_V3DT_CAMINFO_Y_FLIP=1`, `NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy`.
- **Rationale:** This baseline produces bbox3d on all cameras with correct height scale; making it the default reduces setup errors and ensures new agents reproduce the best-known state without manual env tweaking.
- **References:** `noesis/ds8_runtime.py`, `scripts/generate_v3dt_caminfo.py`, `docs/DS8_v3dt_forensics.md`, `plans/DS8/v3dt/README.md`
  _2026-01-22 (Codex): Superseded — locked baseline now uses `INVERT_E=0` with pre-generated camInfo (`NOESIS_V3DT_AUTOGEN_CAMINFO=0`)._

- **Date:** 2026-01-12
- **Author:** Codex
- **Area:** Diagnostics / V3DT
- **Decision:** Add a DS8 V3DT forensics toolkit that captures a calibration snapshot (inputs + derived projection math), logs per-frame tracking telemetry to NDJSON, generates a report (with concrete checks like bbox3d coverage, height sanity, and reprojection error), and renders an offline HTML panel. Persist raw `set_extrinsics` payloads to `logs/calibration_raw/` for audit.
- **Rationale:** SV3DT failures were being debugged by implication; a deterministic snapshot + runtime log + report makes root causes explicit (units, projection mismatch, Y-axis convention, bbox3d coverage, track fragmentation) and keeps raw Menon payloads intact for comparison.
- **References:** `noesis/diagnostics/v3dt_forensics.py`, `noesis/diagnostics/telemetry_log.py`, `scripts/v3dt_forensics.py`, `docs/DS8_v3dt_forensics.md`, `noesis/calibration/manager.py`

- **Date:** 2026-01-11
- **Author:** Codex
- **Area:** Pipeline / PGIE Profiles
- **Decision:** Implement a runtime `--pgie-profile {yolo11_seg,rfdetr_seg}` “profile overlay” that deep-merges only `preprocess.config-file`, `models.pgie.config-file-path`, and `models.pgie.engine` into the selected base YAML, writes `build/effective_pipeline_<profile>.yaml`, and builds DS8 from that effective config.
- **Rationale:** Avoids maintaining duplicate near-identical pipeline YAMLs and guarantees the PGIE swap does not alter DS8 topology; keeps default behavior YOLO while enabling an explicit RF-DETR startup switch with clear, auditable effective config logging.
- **References:** `noesis/ds8_runtime.py`, `config/infer.yaml`, `config/infer_v3dt_medium.yaml`, `config/infer_v3dt_sv3dt.yaml`, `plans/DS8/ds8_rfdetr_seg_swap_checklist_validated.md`

- **Date:** 2026-01-11
- **Author:** Codex
- **Area:** Pipeline / RF-DETR Segmentation
- **Decision:** Start DS8 RF-DETR integration with `RFDETRSegPreview` defaults (`resolution=432`, `num_queries=200`, `num_classes=90`) and a people-only DeepStream mapping (`DS classId=0`), cropping query masks to the detected bbox (DeepStream per-object instance mask expectation).
- **Rationale:** Matches upstream `rfdetr==1.3.0` SegPreview defaults (minimizes “guessing”/overrides), preserves existing DS8 person-only + ReID-on-classId=0 behavior, and ensures masks render correctly via nvdsosd by supplying bbox-relative instance masks rather than full-frame masks.
- **References:** `pipelines/config_preproc_rfdetr_432.ini`, `pipelines/config_infer_primary_rfdetr_seg.ini`, `pipelines/nvdsinfer_rfdetr_seg/nvdsinfer_rfdetr_seg.cpp`, `plans/DS8/ds8_rfdetr_seg_swap_checklist_validated.md`

- **Date:** 2026-01-11
- **Author:** Codex
- **Area:** Pipeline / RF-DETR Parser Scoring + Class IDs
- **Decision:** Score RF-DETR detections using sigmoid on the “person” logit only (`score = sigmoid(logits[person_idx])`) and default `NOESIS_RFDETR_PERSON_CLASS_IDX=1` (COCO category id 1), mapping those detections to DeepStream `classId=0` for DS8 parity.
- **Rationale:** Upstream RF-DETR postprocess uses per-class sigmoid (not softmax) and COCO class indices correspond to COCO category IDs (1..90). Scoring/mapping this way preserves the existing DS8 “person == classId 0” contract required by ReID (`operate-on-class-ids=0`) while avoiding brittle best-class filtering under sigmoid multi-label logits.
- **References:** `pipelines/nvdsinfer_rfdetr_seg/nvdsinfer_rfdetr_seg.cpp`, `/home/mayor/.local/lib/python3.12/site-packages/rfdetr/models/lwdetr.py` (`PostProcess`), `/home/mayor/.local/lib/python3.12/site-packages/rfdetr/util/coco_classes.py`, `plans/DS8/ds8_rfdetr_seg_swap_checklist_validated.md`

- **Date:** 2026-01-11
- **Author:** Codex
- **Area:** Pipeline / Mosaic RTSP / WebRTC
- **Decision:** Treat `sinks.mosaic_sink` and `sinks.bev_sink` as **semantic placeholders** (for future Flow retrievers) and do not build/link them as real `fakesink` tee branches when RTSP mosaic output is enabled.
- **Rationale:** Building extra fakesink branches off the post-OSD tee can interfere with RTSP mosaic delivery on some DS8 builds, resulting in an RTSP port that accepts TCP but times out on RTSP OPTIONS/SETUP/PLAY and prevents the WebRTC gateway from answering offers. Keeping the RTSP branch as the only terminal output (when enabled) restores RTSP responsiveness and WebRTC video delivery.
- **References:** `noesis/pipelines/ds8_pipeline.py`, `noesis/ds8_runtime.py`, `config/infer_v3dt_medium.yaml`, `scripts/webrtc_gateway_smoke_test.py`

- **Date:** 2026-01-09
- **Author:** Codex
- **Area:** Calibration / WebSocket RPC / SV3DT camInfo / Units
- **Decision:** Persist DS8 camera extrinsics (`config/camera_calibration.json` `E`) in **meters** and auto-coerce cm-like translations to meters in `CalibrationManager.set_extrinsics()` to prevent SV3DT camInfo scaling drift (camInfo remains cm via `NOESIS_V3DT_CAMINFO_WORLD_SCALE=100`).
- **Rationale:** A real-world failure mode was observed where `set_extrinsics` inputs effectively used centimeters (camera center Y ≈ 147–230), but camInfo generation assumes meters and multiplies translation by `WORLD_SCALE=100`. This produced projection translations 100× too large and caused SV3DT track termination/mask flicker under motion. Coercing cm→m at the persistence boundary keeps DS8 calibration conventions stable and prevents accidental reintroduction of this class of bug.
- **References:** `noesis/calibration/manager.py`, `config/camera_calibration.json`, `config/v3dt/camInfo_living-room.yml`, `config/v3dt/camInfo_kitchen.yml`, `config/v3dt/camInfo_family-room.yml`, `noesis/ds8_runtime.py` (`_ds8_set_extrinsics_handler`, `_maybe_sync_v3dt_caminfo`), `scripts/generate_v3dt_caminfo.py`, `plans/DS8/v3dt/ds8_v3dt_hardening_work_order.md` (V3DT-H01)

- **Date:** 2025-12-30
- **Author:** Codex
- **Area:** Tracking / SV3DT / MV3DT / Calibration / Units
- **Decision:** Standardize all V3DT world coordinates on **METERS** (not centimeters). Changed `NOESIS_V3DT_CAMINFO_WORLD_SCALE` default from `100` to `1.0` in both `scripts/generate_v3dt_caminfo.py` and `noesis/ds8_runtime.py`. Regenerated all camInfo files with `modelInfo.height: 1.7`, `radius: 0.35` (meters).
- **Rationale:** Meters is the standard/canonical convention for world coordinates. The documentation already specified meters, but the implementation was using centimeters. Aligning to meters simplifies reasoning, matches documentation, and follows best practices.
- **References:** `scripts/generate_v3dt_caminfo.py`, `noesis/ds8_runtime.py`, `noesis/pipelines/hooks.py`, `config/v3dt/camInfo_*.yml`, `plans/DS8/v3dt/ds8_v3dt_hardening_work_order.md` (V3DT-H01)
  _2026-01-09 (Codex): Superseded — current SV3DT path keeps calibration/telemetry in meters, but generates camInfo in centimeters (`NOESIS_V3DT_CAMINFO_WORLD_SCALE=100`) for SV3DT stability; see 2026-01-09 entry above._

- **Date:** 2025-12-30
- **Author:** Codex
- **Area:** Tracking / SV3DT / Metadata Bridge / Native Extension
- **Decision:** Add try/catch protection around all meta extractions in `noesis_v3dt_meta_ext` native bridge; confirm `NVDS_OBJ_WORLD_FOOT_LOCATION` cannot be extracted because Service Maker does not expose `ObjectWorldFootLocationUserMetadata` class (only `ObjectImageFootLocationUserMetadata` for image-space foot); world footpoints must be derived from `bbox3d` (yCentre − 0.5·yLen).
- **Rationale:** Grep of `/opt/nvidia/deepstream/deepstream/service-maker/includes/metadata.hpp` shows only: `ObjectVisibilityUserMetadata`, `ObjectImageFootLocationUserMetadata`, `Object3DBBoxUserMetadata`. There is no class for world foot location. The try/catch wrappers ensure individual extraction failures don't crash the pipeline and are logged once. This supersedes the 2025-12-29 entry which incorrectly implied the class existed but crashed.
- **References:** `native/noesis_v3dt_meta_ext.cpp`, `/opt/nvidia/deepstream/deepstream/service-maker/includes/metadata.hpp`, `plans/DS8/v3dt/ds8_v3dt_hardening_work_order.md` (V3DT-H05)

- **Date:** 2025-12-29
- **Author:** Codex
- **Area:** Tracking / SV3DT / Metadata Bridge
- **Decision:** Disable extracting `NVDS_OBJ_WORLD_FOOT_LOCATION` via Service Maker `ObjectImageFootLocationUserMetadata` in the DS8 native bridge (`noesis_v3dt_meta_ext`) because it segfaults in `deepstream::ObjectImageFootLocationUserMetadata::getImageFootLocation()`; rely on `bbox3d`-derived world footpoints instead.
- **Rationale:** Under SV3DT, iterating `NVDS_OBJ_WORLD_FOOT_LOCATION` inside a Service Maker `BatchMetadataOperator` triggered a reproducible SIGSEGV in NVIDIA's Service Maker library. Dropping the field keeps DS8 SV3DT stable while preserving the core 3D outputs (`bbox3d`/`velocity3d`); BEV/world are still computed from the 3D bbox (ground footpoint = yCentre − 0.5·yLen).
- **References:** `native/noesis_v3dt_meta_ext.cpp`, `noesis/pipelines/hooks.py`, `/opt/nvidia/deepstream/deepstream/lib/libnvds_service_maker.so` (crash site), `scripts/sv3dt_meta_smoke_test.py`
  _2025-12-30 (Codex): Superseded by the 2025-12-30 entry above which clarifies that the class doesn't exist in Service Maker._

- **Date:** 2025-12-29
- **Author:** Codex
- **Area:** Tracking / SV3DT / Calibration / camInfo
- **Decision:** Treat `config/camera_calibration.json` `E` as **world→camera** when generating V3DT camInfo projection matrices (`projectionMatrix_3x4_w2p`) and expose an override via `NOESIS_V3DT_CAMINFO_INVERT_E` (default `0`).
- **Rationale:** Menon composes `Twc` (camera→world) then sends `E = inv(Twc)` (world→camera) to Noesis. SV3DT expects `projectionMatrix_3x4_w2p = K @ E[:3,:]` in world→camera form; inverting again produces massive projection errors (degenerate 3D cuboids and SV3DT track termination). The env flag keeps the behavior auditable if a different calibration source provides `Twc` instead.
- **References:** `noesis/ds8_runtime.py`, `scripts/generate_v3dt_caminfo.py`, `plans/DS8/v3dt/README.md`, `plans/DS8/v3dt/oom_killed_infer_v3dt_debug.md`, https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html
  _2026-01-12 (Codex): Updated default to `NOESIS_V3DT_CAMINFO_INVERT_E=0` after Menon confirmed it sends `E` as world→camera for all cameras._

- **Date:** 2025-12-28
- **Author:** Codex
- **Area:** Tracking / SV3DT
- **Decision:** Tune SV3DT for robustness/accuracy by (1) enabling `TrajectoryManagement.enableReAssoc` and `ReID` (tracker-internal ReID-based re-association), (2) switching to cascaded association matching, (3) raising `minDetectorConfidence` to suppress low-confidence jitter/ghosts, and (4) running BodyPose3DNet periodically (`poseInferenceInterval=4`) instead of first-frame-only height initialization.
- **Rationale:** The observed failure modes (jitter, fragmentation, ghosts) are classic symptoms of low-confidence detections creating tentative targets, overly strict frame-to-frame association, and insufficient re-association after occlusion. NVIDIA’s SV3DT reference config includes both trajectory re-association and a dedicated ReID module; enabling these and tightening lifecycle thresholds is the most direct “heavy but stable” configuration to explore the tracker’s ceiling before optimizing cost.
- **References:** `config/v3dt/nvtracker_sv3dt.yml`, `config/v3dt/nvtracker_sv3dt_sample.yml`, `/tmp/deepstream_reference_apps/deepstream-tracker-3d/configs/config_tracker_NvDCF_accuracy_3D.yml`, `/tmp/deepstream_reference_apps/deepstream-tracker-3d/README.md`

- **Date:** 2025-12-27
- **Author:** Codex
- **Area:** Telemetry / BEV / UI
- **Decision:** Add a BEV dual-trail mode that publishes `footpoints_alt` alongside primary `footpoints` (typically SV3DT vs bbox) and renders both as light/dark variants of the same hue without duplicating the head dot; expose a `dual_trail_visualization_enabled` toggle via WebSocket and the settings panel.
- **Rationale:** Comparing SV3DT and bbox paths side-by-side helps validate 3D tracking quality without changing external telemetry contracts; using the same hue with light/dark variants keeps identity consistent while avoiding visual clutter. Keeping the dot to a single primary path prevents confusing “double heads” while still showing both trails.
- **References:** `noesis/telemetry/bev.py`, `noesis/pipelines/hooks.py`, `websocket_server.py`, `oai2-fe/src/components/BevView.tsx`, `docs/DS8_api_contracts_ws.md`

- **Date:** 2025-12-28
- **Author:** Codex
- **Area:** Tracking / Telemetry / V3DT
- **Decision:** Prefer `NVDS_OBJ_WORLD_FOOT_LOCATION` for BEV/world footpoints when SV3DT/MV3DT is enabled; extend the V3DT meta bridge to extract `world_foot` (float[2]) and `visibility` and plumb them through telemetry. Add a DS8 runtime preflight that regenerates `config/v3dt/camInfo_*.yml` from `config/cameras.yaml` + `config/camera_calibration.json` when a V3DT tracker config is selected.
- **Rationale:** DeepStream already computes an explicit ground-plane footpoint (`ptWorldFeet`), which is a better BEV input than inferring footpoint from the 3D bbox centroid (less jitter/out-of-room drift). CamInfo files are static inputs to SV3DT/MV3DT; if calibration changes (e.g., auto-calibrate updates `camera_calibration.json`) but camInfo files are not regenerated, 3D outputs can silently become inconsistent—especially harmful for MV3DT fusion. Auto-sync keeps runs reproducible and avoids “it worked yesterday” drift.
- **References:** `native/noesis_v3dt_meta_ext.cpp`, `noesis/pipelines/hooks.py`, `noesis/ds8_runtime.py`, `/opt/nvidia/deepstream/deepstream-8.0/sources/gst-plugins/gst-nvtracker/nvtracker_proc.cpp` (meta payload types)

- **Date:** 2025-12-27
- **Author:** Codex
- **Area:** Tracking / Telemetry / BEV
- **Decision:** Expose `NVDS_OBJ_3D_META` (`NvDsObj3DBbox`) to DS8 telemetry hooks via a small native bridge module (`noesis_v3dt_meta_ext`) that uses Service Maker C++ metadata iterators (Python `ObjectMetadata` does not expose `obj_user_meta_list`). Emit additive `bbox3d`/`velocity3d` fields and derive a best-effort `world` footpoint from the 3D bbox; when present, BEV uses SV3DT world coordinates instead of 2D bbox ray-plane intersections. Add an optional `world_frame` label from config (`camera_local` until global calibration is available).
- **Rationale:** SV3DT provides higher-quality 3D estimates and robust footpoints without introducing CPU branches. Because Service Maker Python bindings do not expose arbitrary per-object user meta, a minimal pybind11 extension is the cleanest in-process way to read `NVDS_OBJ_3D_META` without introducing `nvmsgconv` or repurposing preprocess metadata. The world-frame label keeps cross-camera comparisons guarded until shared global extrinsics are provided.
- **References:** `noesis/pipelines/hooks.py`, `noesis/telemetry/bev.py`, `native/noesis_v3dt_meta_ext.cpp`, `scripts/build_noesis_v3dt_meta_ext.sh`, `plans/DS8/v3dt/integration_plan.md`, `docs/DS8_api_contracts_ws.md`, `docs/DS8_metadata_contracts.md`

- **Date:** 2025-12-27
- **Author:** Codex
- **Area:** Tracking / Calibration / IDs
- **Decision:** Integrate SV3DT on all DS8 cameras for 3D state estimation and occlusion-robust tracking; enable MV3DT only for true-overlap “vision neighbor” cameras (kitchen ↔ family-room); keep `stable_id` as the only user-visible identity and treat MV3DT’s global tracker ID as an internal `mv3dt_id` hint/constraint; require a shared, metric, Y-up global calibration before enabling MV3DT.
- **Rationale:** SV3DT improves single-camera robustness (priority A). MV3DT is designed for overlapping FoVs and can mis-associate across non-overlap cameras, so restrict its neighbor graph initially. Preserving the stable-id-only external contract avoids UI/WS breakage while still leveraging MV3DT’s strongest capability (cross-camera ID propagation + multi-view fusion) as an internal signal. MV3DT fundamentally depends on a shared global world frame, so calibration must be solved early.
- **References:** `plans/ds8/v3dt/README.md`, `plans/ds8/v3dt/research_notes.md`, `plans/ds8/v3dt/integration_plan.md`, `plans/ds8/v3dt/codex_agent_prompt.md`, `plans/DS8/ds8_id_contract_v2.md`, https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html, https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_MV3DT.html

- **Date:** 2025-12-27
- **Author:** Codex
- **Area:** Metadata / Calibration
- **Decision:** In DS8, distribute intrinsics via the runtime `calibration-bundle` (and `_CalibrationProvider.snapshot()` for BEV) instead of attaching intrinsics as per-frame user meta; disable `attach_intrinsics_hook` by default in `noesis/ds8_runtime.py`.
- **Rationale:** Service Maker `FrameMetadata` in our current bindings does not expose a supported writable user-meta surface for custom payloads, which caused repeated debug logs and no effective attachment. DS7 semantics already treat intrinsics as calibration data rather than per-frame metadata; DS8 already broadcasts `calibration-bundle` and uses it for BEV/depth consumers.
- **References:** `noesis/ds8_runtime.py`, `noesis/pipelines/hooks.py`, `noesis/metadata/intrinsics.py`, `docs/DS8_metadata_contracts.md`, `plans/DS8/ds8_migration_checklist_hooks.md`

- **Date:** 2025-12-26
- **Author:** Codex
- **Area:** Depth / BEV / UI
- **Decision:** Render floorplan/BEV canvases with letterboxed “contain” scaling (no stretch) and map overlays/footpoints to the content rect; scale intrinsics to the depth raster with aspect-preserving resize + padding before generating floorplan grids.
- **Rationale:** Fixed-aspect canvases were stretching the floorplan footprint (wide/obtuse look). Depth-derived X/Z needs intrinsics aligned to the actual depth raster resolution (including streammux padding) to avoid horizontal inflation.
- **References:** `oai2-fe/src/lib/renderUtils.ts`, `oai2-fe/src/components/BevView.tsx`, `oai2-fe/src/components/DepthDrawer.tsx`, `geometry/depth_source.py`

- **Date:** 2025-12-24
- **Author:** Codex
- **Area:** Calibration / WebSocket RPC
- **Decision:** Auto-calibrate-all in DS8 derives the camera list from `config/cameras.yaml` labels (source-id order) instead of relying on `auto_calibrate_from_latest_depth` defaults.
- **Rationale:** The calibration script defaults are static (`DEFAULT_CAMERAS`) and can drift from the DS8 runtime configuration. Using the DS8 camera labels ensures the calibrate-all button targets the configured sources without DS7 fallbacks.
- **References:** `noesis/ds8_runtime.py`, `scripts/auto_calibrate_from_depth.py`, `config/cameras.yaml`, `docs/DS8_api_contracts_ws.md`

- **Date:** 2025-12-25
- **Author:** Codex
- **Area:** Telemetry / Occupancy / UI
- **Decision:** When DS8 tracking telemetry cannot infer a per-object zone from nvdsanalytics `roiStatus`, fall back to using the camera/stream name as the track `zone` (and therefore the occupancy “room” key and `dwell_time` basis). Preserve `roiStatus`-derived zones when they exist.
- **Rationale:** The current DS8 analytics config commonly enables `overcrowding`/`direction_detection` but not `roi-filtering`, so `roiStatus` is often absent and the UI showed `Zone: -` and `0.0s` dwell for all tracks. The fallback keeps the dashboard’s Active Tracks + Occupancy sections useful without requiring an immediate analytics config refactor.
- **References:** `noesis/pipelines/hooks.py`, `config/config_nvdsanalytics_post.ini`, `oai2-fe/src/App.tsx`, `docs/DS8_api_contracts_ws.md`

- **Date:** 2025-12-25
- **Author:** Codex
- **Area:** Telemetry / Occupancy
- **Decision:** Add an occupancy grace window so zones are not vacated immediately when object tracks drop out briefly; implement as stable-id last-seen tracking with `NOESIS_OCCUPANCY_GRACE_S` (default 1.0s, clamp 0–10s).
- **Rationale:** Temporary detector/tracker occlusions caused occupancy to flicker (drop to 0 then rebound). A small grace window provides a smoother, more “human” occupancy signal without changing the DeepStream analytics configuration.
- **References:** `noesis/pipelines/hooks.py`, `plans/DS8/ds8_migration_checklist_hooks.md`, `oai2-fe/src/App.tsx`

- **Date:** 2025-12-24
- **Author:** Codex
- **Area:** IDs / Telemetry / OSD
- **Decision:** Make `stable_id` the only user-visible identity across DS8 (mosaic OSD labels, WS/REST tracking payloads, and `bev-frame` footpoints). Keep `track_id` internal-only. Ensure StableIDManager never emits negative/provisional IDs by returning a positive pending stable ID during new-track hysteresis, and add a telemetry-level fallback stable-ID allocator if StableIDManager is unavailable/unhealthy. Stamp mosaic OSD labels inside the analytics telemetry hook (pre-tiler) and format as `"label sid <stable_id> <confidence>"` to keep mosaic/BEV/Active Tracks IDs consistent across all cameras.
- **Rationale:** User-facing tracker IDs caused confusing mismatches and collisions across cameras; tiler-stage metadata can collapse camera context, so stamping per-source before tiler keeps IDs correct. Positive-only stable IDs eliminate `XX` placeholders and prevent leaking raw tracker IDs.
- **References:** `noesis/pipelines/hooks.py`, `reid/stable_id_manager.py`, `oai2-fe/src/App.tsx`, `docs/DS8_api_contracts_ws.md`, `docs/DS8_metadata_contracts.md`

- **Date:** 2025-12-20
- **Author:** Codex
- **Area:** Analytics / ROI Editor
- **Decision:** Implement the ROI editor as a header chip that opens a right-side drawer with a solo-tile zoom view derived from the live WebRTC mosaic, editing exclusion polygons in the exclude stage pixel space (`config_width/config_height`). The UI relies on additive `stats.payload.pipeline.mosaic_layout` metadata for tile mapping and uses REST `/api/v1/analytics/rois` for apply/reload; v1 focuses on polygon editing only (no inverse/class-id controls).
- **Rationale:** Keeps ROI editing aligned with DS8’s canonical exclusion path (`nvdsroiexclude`) and avoids guessing tiler layout or GStreamer internals in the UI. Using the mosaic video for preview ensures the editor stays in sync with the running pipeline without introducing new appsinks or CPU branches.
- **References:** `noesis/ds8_runtime.py` (stats payload + CORS), `noesis/server/analytics_api.py` (ROI REST + INI sync), `oai2-fe/src/components/RoiEditorDrawer.tsx`, `docs/DS8_api_contracts_ws.md`, `docs/DS8_api_contracts_rest.md`

- **Date:** 2025-12-21
- **Author:** Codex
- **Area:** Pipeline / ReID / StableID
- **Decision:** Drive StableID from the dedicated OSNet ReID SGIE (`reid_osnet`, `gie_id=3`, batch=16) and consume embeddings in the telemetry hook (probe attached to the ReID element) using a manual DLPack→cudaMemcpy decode (avoid torch dlpack consumer). Keep StableID people-only and allow the same StableID to be active on multiple cameras (overlapping FoVs).
- **Rationale:** Torch `from_dlpack` consumption of Service Maker tensor outputs triggered a native double-free/segfault in `TensorOutputUserMetadata::getLayers()` during DS8 runs. A manual DLPack decode that copies the embedding to host memory (without transferring ownership) avoids the crash while keeping the ReID path GPU-first. Attaching telemetry at the ReID element ensures tensor meta is still valid at the point of access.
- **References:** `noesis/pipelines/ds8_pipeline.py`, `noesis/pipelines/hooks.py`, `reid/stable_id_manager.py`, `pipelines/config_infer_secondary_reid_osnet.ini`, `config/infer.yaml`, `scripts/reid_stable_id_smoke_test.py`

- **Date:** 2025-12-20
- **Author:** Codex
- **Area:** ReID / StableID
- **Decision:** Implement appearance-based StableID in DS8 by adding a dedicated per-object ReID SGIE (OSNet) after the tracker (`tracker → reid_sgie → analytics`) with `output-tensor-meta=1`, then consuming the embedding tensor (`features`) inside `_AnalyticsTelemetryProcessor` to drive `StableIDManager.update(..., embedding=...)`. Gate StableID creation on embeddings (`require_embeddings=true`) to avoid allocator-only IDs when ReID outputs are missing.
- **Rationale:** DS7’s StableID intention was OSNet-based global IDs, but it lacked a robust per-object embedding source in the pipeline. DS8 can keep the entire path GPU-first (crop/resize/infer on GPU), update galleries with multiple embeddings over time, and preserve the “live feel” by running ReID at a controlled cadence (`secondary-reinfer-interval`) with async SGIE output.
- **References:** scripts/export_reid_osnet_to_onnx.py (exports OSNet-IBN MSMT17 with ImageNet normalization in-graph + dynamic batch), pipelines/config_infer_secondary_reid_osnet.ini (SGIE settings: batch-size=16, output-tensor-meta=1), config/infer.yaml (models.reid + reid.* knobs), noesis/pipelines/ds8_pipeline.py (reid SGIE insertion), noesis/pipelines/hooks.py (embedding extraction + OSD label append), reid/stable_id_manager.py (gallery/ghost matching).
  _2025-12-20 (Codex): Superseded by the 2025-12-21 SGIE-driven StableID approach above (manual DLPack decode; no nvtracker ReID meta dependency)._

- **Date:** 2025-12-20
- **Author:** Codex
- **Area:** ReID / StableID
- **Decision:** Use NvDeepSORT’s built-in ReID path inside `nvtracker` with an OSNet ONNX+TensorRT engine (normalization/L2 inside ONNX) and `outputReidTensor: 1`, then feed StableIDManager from `NVDS_TRACKER_OBJ_REID_META` via pyds traversal in `_AnalyticsTelemetryProcessor` (`reid.source=tracker`). Keep the standalone ReID SGIE disabled by default.
- **Rationale:** Avoids Service Maker tensor-meta dereference instability and SGIE tensor-meta pool tuning, keeps ReID tightly coupled to tracking/association, and provides a single authoritative embedding source for cross-camera StableID while remaining GPU-first.
- **References:** config/nvtracker.yaml (NvDeepSORT + OSNet ReID settings), config/infer.yaml (`reid.source=tracker`, `models.reid.enable=false`), noesis/pipelines/hooks.py (`_extract_tracker_reid_embedding_pyds`), /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDeepSORT.yml (reference), /opt/nvidia/deepstream/deepstream/sources/tracker_ReID/README (tracker ReID model config).
  _2025-12-21 (Codex): Superseded by the SGIE-driven StableID + tee-after-analytics approach due to Flow Buffer/pyds bridge instability (SIGSEGV) and difficulty reliably accessing `NVDS_TRACKER_OBJ_REID_META` from Service Maker metadata._

- **Date:** 2025-11-29
- **Author:** Codex
- **Area:** Telemetry / Occupancy
- **Decision:** Derive DS8 occupancy counts inside `_AnalyticsTelemetryProcessor` from analytics ROI zone labels and publish via `pipeline.occupancy_publisher`, emitting vacate events when a zone disappears; track telemetry carries `stable_id` (nullable) and per-zone `dwell_time` derived from per-track entry timestamps to maintain DS7-compatible schema.
- **Rationale:** Keeps occupancy and dwell semantics aligned with DS7 without adding DS7 pad probes; uses DS8 batch metadata already available in the analytics operator and preserves WS payload shape stability even when REID is disabled.
- **References:** docs/DS8_metadata_contracts.md, noesis/pipelines/hooks.py

- **Date:** 2025-11-29
- **Author:** Codex
- **Area:** Pipeline / Gating
- **Decision:** Defer DS8 BufferOperator-based depth gating because `pyservicemaker.flow.BufferOperator` is unavailable in the installed DS8 package; keep depth-enabled stats and MapAnything processor-side gating, and surface a warning when gating cannot be attached.
- **Rationale:** Avoid guessing undocumented Flow APIs; gating requires a documented BufferOperator attach point. Current DS8 build lacks the class, so upstream drop control cannot be implemented safely.
- **References:** noesis/pipelines/ds8_pipeline.py (depth_gate_attach/depth_gate_supported), plans/DS8/ds8_migration_checklist_ds8_pipeline.md
  _2025-12-16 (Codex): Superseded by the 2025-12-14 decision to use an upstream GStreamer `valve` gate for physical SGIE compute savings + startup preroll stability._

- **Date:** 2025-11-29
- **Author:** Codex
- **Area:** Pipeline / Gating
- **Decision:** Keep depth gating logical-only until Service Maker exposes a supported attach path for `BufferOperator`—the class exists, but `Pipeline.attach`/`Node.attach` only accept Probe/Receiver/Feeder, so gating cannot be wired without undocumented calls.
- **Rationale:** API inspection shows BufferOperator cannot be attached via documented Pipeline/Flow interfaces; forcing attachment would require private `_instance` hooks. Depth enable continues to gate MapAnything processing logically and via REST timers until NVIDIA exposes an operator attach interface.
- **References:** /home/mayor/.local/lib/python3.12/site-packages/pyservicemaker/pipeline.py, noesis/pipelines/ds8_pipeline.py, plans/DS8/ds8_migration_checklist_ds8_pipeline.md
  _2025-12-16 (Codex): Superseded by the 2025-12-14 `mapanything_valve` gating approach (canonical for GPU baseline reduction)._

- **Date:** 2025-11-29
- **Author:** Codex
- **Area:** Pipeline / MapAnything branch
- **Decision:** Terminate the MapAnything SGIE branch at a dedicated `fakesink` instead of feeding it into the tiler, keeping the branch metadata-only and avoiding multiple consumers on a single OSD pad.
- **Rationale:** MapAnything outputs are consumed via BatchMetadataOperator (tensor meta only); linking to tiler introduced invalid multi-sink linking without a tee and doubled frames in mosaic. A drop sink preserves tensor processing while keeping the mosaic path clean.
- **References:** noesis/pipelines/ds8_pipeline.py (mapanything_fullframe_sink, sink_tee), plans/DS8/ds8_master_work_orders.md

- **Date:** 2025-11-29
- **Author:** Codex
- **Area:** Pipeline / Mosaic output
- **Decision:** Provide mosaic output as an RTSP stream (`nvrtspoutsinkbin` `rtsp_out`) consumed by the RTSP→WebRTC gateway; WebRTC is the canonical mosaic delivery path.
- **Rationale:** Keeps mosaic delivery off the WebSocket binary path and avoids brittle in-pipeline retrieval mechanisms.
- **References:** noesis/pipelines/ds8_pipeline.py (`rtsp_out`), noesis/mosaic_webrtc_gateway.py, noesis/ds8_runtime.py

- **Date:** 2025-11-29
- **Author:** Codex
- **Area:** Pipeline / Depth gating
- **Decision:** Depth gating will be implemented via a `DepthGateOperator(BufferOperator)` wrapped in a `Probe("depth_gate", DepthGateOperator(pipeline))` attached at `pipeline.depth_gate_attach` (MapAnything SGIE). `Pipeline.attach` takes the Probe, not the operator directly.
- **Rationale:** Matches DS8 docs (FrameSkipper pattern) and avoids unsupported attachments; current runtime remains logical-only gating until this operator is integrated.
- **References:** pyservicemaker Probe/BufferOperator docs, plans/DS8/ds8_migration_checklist_flow_retrievers.md, plans/DS8/ds8_master_work_orders.md

- **Date:** 2025-11-29
- **Author:** Codex-patcher
- **Area:** Pipeline / MapAnything
- **Decision:** Added a dedicated MapAnything `nvinfer` config (`pipelines/config_infer_secondary_mapanything.ini`) that expects fused 12-channel tensor meta, enables `input-tensor-from-meta`, and pins the engine path; `models.mapanything` now references this config via `config-file-path` with an absolute engine to avoid mis-resolving under `config/`.
- **Rationale:** DS8 `nvinfer` refused to start without a config file; aligning with the fused-input design from `mapanything_preprocess_fused` avoids CPU pre-processing guesses and keeps engine resolution deterministic, even though the current plan file targets an older TensorRT build.
- **References:** pipelines/config_infer_secondary_mapanything.ini, config/infer.yaml, pipelines/mapanything_preprocess_fused/README.md

- **Date:** 2025-11-29
- **Author:** Codex-build
- **Area:** Pipeline / MapAnything Engine
- **Decision:** Rebuilt MapAnything TensorRT engine (`ma_model_fp16_b3_fused.plan`) from ONNX model (`ma_onnx_out_fused/model_fused_sim.onnx`) using TensorRT 10.13.0 to resolve version mismatch with DS8 nvinfer. Engine configured with batch size 3 (min=1, opt=3, max=3), FP16 precision, and input shape `mapanything_fused:3x12x518x518`.
- **Rationale:** The existing engine was built with an older TensorRT version incompatible with TensorRT 10.13.0 used by DS8. Rebuilding ensures compatibility and allows DS8 to load the engine without version errors. Used fixed batch size 3 to match DS8 pipeline configuration (`batch_size: 3` in `config/infer.yaml`).
- **References:** export_ma_onnx/export_to_onnx.py, pipelines/config_infer_secondary_mapanything.ini, config/infer.yaml, /tmp/trtexec_build.log

- **Date:** 2025-11-29
- **Author:** Opus
- **Area:** Hooks / DS8 API Compatibility
- **Decision:** Added DS8 pyservicemaker API compatibility layer to hooks.py operators. DS8 uses `batch_meta.frame_items` (iterable) instead of `batch_meta.frame_meta_list` (linked list), `frame_meta.object_items` instead of `obj_meta_list`, and `tensor_meta.get_layers()` → `np.from_dlpack()` for tensor extraction.
- **Rationale:** DS8 pyservicemaker BatchMetadataOperator receives pyservicemaker.BatchMetadata objects with different API than pyds. Each operator now checks for `frame_items` attribute (DS8 path) before falling back to pyds linked-list iteration (DS7 fallback). MapAnythingProcessor.handle_nvds_tensor_ds8 uses get_layers()+DLPack for tensor conversion.
- **References:** noesis/pipelines/hooks.py (_IntrinsicsOperator, _MapAnythingOperator, _AnalyticsTelemetryOperator, _ExcludePruneOperator)

- **Date:** 2025-11-29
- **Author:** Opus
- **Area:** Pipeline / Validation Results
- **Decision:** DS8 pipeline validation completed with partial success. Pipeline builds/prepares/activates correctly; PGIE and MapAnything SGIE engines load from TensorRT 10.13 plans; tracker and analytics initialize; 3 RTSP sources connect; REST API responds correctly. Remaining issues at the time: (1) MapAnything depth FPS=0 because SGIE expects 12-channel fused tensor from preprocess but current preprocess produces 3x640x640 for PGIE; (2) DS8 object_items is read-only so exclusion prune cannot remove objects.
- **Rationale:** Captures validation state for Phase 7 progress. MapAnything required full-frame SGIE mode and proper gating; exclusion should leverage analytics ROI filtering instead of metadata modification.
- **References:** plans/DS8/ds8_migration_checklist_ds8_pipeline.md, plans/DS8/ds8_migration_checklist_hooks.md, plans/DS8/ds8_migration_checklist_ds8_runtime.md

- **Date:** 2025-11-30
- **Author:** Codex
- **Area:** Pipeline / MapAnything
- **Decision:** Use the existing DS7 full-frame MapAnything TensorRT engine (RGB image input) for the DS8 MapAnything branch instead of the fused 12-channel tensor-from-meta engine. DS8 `nvinfer` for MapAnything should consume images directly (no `input-tensor-from-meta` in its config) and load the same engine file used by the legacy DS7 pipeline, keeping semantics aligned with the external MapAnything service path.
- **Rationale:** The fused 12-channel engine required a dedicated tensor-producing preprocess stage and diverged from the legacy behavior, where MapAnything consumes full-frame BGR images and performs its own fusion internally. Reusing the DS7 full-frame engine simplifies the DS8 pipeline, avoids tensor-from-meta complexity, and keeps the depth path consistent with the existing deployment, while still benefiting from DS8 Service Maker integration.
- **References:** config/infer.yaml (models.mapanything), pipelines/config_infer_secondary_mapanything.ini, deepstream_video_pipeline.py (MapAnything branch), docs/reference/DEPTH_STACK_FLOW_V2.md, docs/reference/Pipeline_Architecture_Diagram.md

- **Date:** 2025-11-30
- **Author:** Codex
- **Area:** Pipeline / MapAnything Engine
- **Decision:** Pin the DS8 MapAnything SGIE to `/home/mayor/Noesis_Devel/models/mapanything_depth/1/model.plan` (max batch 3, FP16, image input 3x518x518 RGB) with `infer-dims=3;518;518`, `model-color-format=0`, and `input-tensor-from-meta=0`, keeping `output-tensor-meta=1` so depth tensors propagate via metadata.
- **Rationale:** Aligns DS8 execution with the DS7 full-frame engine artifact and removes the fused 12-channel tensor-from-meta dependency; preserves tensor meta emission for `MapAnythingProcessor` while ensuring nvinfer consumes GPU surfaces directly.
- **References:** config/infer.yaml, pipelines/config_infer_secondary_mapanything.ini, models/mapanything_depth/config.pbtxt

- **Date:** 2025-12-01
- **Author:** Codex
- **Area:** Pipeline / MapAnything Engine
- **Decision:** Re-exported MapAnything as a single-input ONNX (images only; intrinsics baked with fx=fy=1000, cx=W/2, cy=H/2) and rebuilt the FP16 TensorRT engine with TensorRT 10.13 (`models/mapanything_depth/1/model.plan`, batch 1–3, input `images`, output `depth`). DS8 configs now point at this engine with `input-tensor-from-meta=0` and `infer-dims=3;518;518`.
- **Rationale:** DS8 nvinfer documentation only guarantees a single input tensor; the two-input/fused paths depended on undocumented tensor-from-meta plumbing and mismatched preprocessing, leading to depth FPS=0 and TRT version mismatches. The rebuilt engine is compatible with the installed TensorRT and keeps MapAnything full-frame on-GPU.
- **References:** export_ma_onnx/export_to_onnx.py, ma_onnx_out_clean/model.onnx, models/mapanything_depth/1/model.plan, pipelines/config_infer_secondary_mapanything.ini, config/infer.yaml

- **Date:** 2025-12-01
- **Author:** Codex
- **Area:** Hooks / MapAnything tensor processing
- **Decision:** Convert DS8 `TensorOutputUserMetadata` to numpy via torch DLPack fallback (explicit `__dlpack__(0)`), enforce finite-mask filtering, and fall back to wall-clock timestamps when DS8 FrameMetadata lacks epoch-based `buffer_pts`. Also relax gie_id filtering when necessary by converting tensor_items via `as_tensor_output()`.
- **Rationale:** Numpy cannot consume DS8 GPU tensors directly; torch DLPack conversion produces CPU arrays without guessing pyservicemaker internals. Finite masks avoid NaN min/max warnings, and wall-clock PTS prevents retention logic from pruning snapshots dated at 1970. Ensures depth_result telemetry and snapshot storage run in DS8 runtime.
- **References:** noesis/pipelines/hooks.py (MapAnythingProcessor, _MapAnythingOperator), plans/DS8/ds8_migration_checklist_hooks.md, plans/DS8/ds8_migration_checklist_depth_api.md

- **Date:** 2025-12-02
- **Author:** Codex
- **Area:** Telemetry / BEV
- **Decision:** Generate BEV status + JPEG directly from the analytics telemetry hook using calibration data (cameras.yaml intrinsics + camera_calibration.json extrinsics) and `BevRenderer`, instead of adding a new Flow/appsink retriever. Overlay drawing respects BEV config/overlay toggles, and BEV WS framing stays `len + b"bev:<cameraId>" + jpeg`.
- **Rationale:** Avoids introducing another appsink in DS8 (GPU-first guardrail) while still providing BEV frames to oai2-fe. Reuses existing analytics metadata traversal and calibration bundle to compute footpoints and publish BEV JSON + binary frames without CPU pad probes.
- **References:** noesis/pipelines/hooks.py (_AnalyticsTelemetryProcessor._publish_bev), noesis/telemetry/bev.py, noesis/ds8_runtime.py (_CalibrationProvider wiring, ws_server calibration_getter), plans/DS8/ds8_migration_checklist_flow_retrievers.md

- **Date:** 2025-12-13
- **Author:** Codex
- **Area:** Mosaic / WebRTC
- **Decision:** Keep the RTSP→WebRTC bridge as a separate GI/GStreamer pipeline (`noesis/mosaic_webrtc_gateway.py`) and answer browser offers by (1) extracting the offered H.264 PT and setting `rtph264pay.pt`, (2) after `set-remote-description` completes, requesting/linking `webrtcbin`’s `sink_%u` pad and forcing its transceiver to `SENDONLY` with H.264 RTP caps, and (3) creating the SDP answer only after the sender is linked and at least one encoded frame has been observed. For reconnects, rebuild the gateway pipeline (new `webrtcbin`) per offer, and use a pad-blocking probe when relinking to avoid `not-linked` stream errors.
- **Rationale:** On this environment’s GStreamer (`webrtcbin` 1.24.2), generating an answer before a sender stream is bound can yield “ICE connected but black” (no RTP/decoded frames). Reusing a single webrtcbin across peer reconnections also leads to connected ICE/DTLS but no media. Rebuilding the gateway per offer plus safe relinking produces consistent `a=sendonly` answers and actual video delivery; validated with `scripts/webrtc_gateway_smoke_test.py` across sequential sessions and PT=96/PT=103.
- **References:** noesis/mosaic_webrtc_gateway.py, websocket_server.py, gst-inspect-1.0 webrtcbin

- **Date:** 2025-12-03
- **Author:** Planner
- **Area:** Depth RPCs (MapAnything / Floorplan)
- **Decision:** Align DS8 depth RPC behavior with DS7’s WebSocket contracts: DS8 runtime exposes `ma_depth_provider` and `floorplan_provider` on `WebSocketServer` that delegate to `geometry.depth_source.DepthStorageManager` to serve `ma_depth_response` and `floorplan_response` payloads identical to DS7’s, but backed by DS8’s on‑GPU MapAnything depth snapshots.
- **Rationale:** The oai2-fe depth drawer is driven via `get_ma_depth`/`ma_depth_response` and `get_floorplan`/`floorplan_response`. Providing DS7-compatible RPC payloads in DS8 removes the last DS7-only dependencies for depth/floorplan visualization.
- **References:** main.py (_ma_depth_provider), geometry/depth_source.py (DepthStorageManager.load_latest_depth, generate_topdown_floorplan), websocket_server.py (get_ma_depth/get_floorplan handlers), noesis/ds8_runtime.py (providers), plans/DS8/ds8_master_work_orders.md, plans/DS8/ds8_migration_checklist_ds8_runtime.md

- **Date:** 2025-12-06
- **Author:** Codex
- **Area:** Mosaic
- **Decision:** Standardize on RTSP→WebRTC gateway for mosaic delivery and remove non-WebRTC mosaic delivery approaches from DS8-facing documentation.
- **Rationale:** A single canonical mosaic path prevents doc drift and matches the deployed UI (WebRTC).
- **References:** noesis/pipelines/ds8_pipeline.py (`rtsp_out`), noesis/ds8_runtime.py (gateway startup), noesis/mosaic_webrtc_gateway.py, websocket_server.py (WebRTC signaling)

- **Date:** 2025-12-07
- **Author:** Codex
- **Area:** Mosaic Output / WebRTC (CANONICAL PATH)
- **Decision:** Implement a decoupled **RTSP-to-WebRTC Media Gateway** (`noesis/mosaic_webrtc_gateway.py`) as the **canonical mosaic output path**. The gateway runs a separate GStreamer pipeline (`rtspsrc → rtph264depay → h264parse → rtph264pay → webrtcbin`) that consumes the DS8 RTSP mosaic output (`rtsp://127.0.0.1:8554/mosaic`) and serves it via WebRTC to browser clients. Signaling (offer/answer/ICE) flows through the existing `WebSocketServer`.
- **Rationale:** Direct `webrtcbin` integration into the DS8 pyservicemaker pipeline proved unreliable due to undocumented pad/transceiver behaviors. The decoupled gateway keeps the DS8 pipeline unchanged (focused on inference/analytics/RTSP output) while providing a clean, standard WebRTC endpoint. This approach:
  - Uses standard GStreamer elements with documented behavior
  - Decouples WebRTC complexity from the DS8 pipeline lifecycle
  - Allows independent debugging of media vs. signaling issues
  - Keeps RTSP output as the gateway ingress and diagnostic endpoint
  - Successfully negotiates H.264 sendonly with proper SDP (codec and direction)
- **References:** noesis/mosaic_webrtc_gateway.py, noesis/ds8_runtime.py (webrtc_gateway startup), websocket_server.py (register_webrtc_gateway), oai2-fe/src/hooks/useWebRTCClient.ts, plans/ds8_mosaic_webrtc_plan.md

- **Date:** 2025-12-07
- **Author:** Codex
- **Area:** WebRTC / SDP Negotiation
- **Decision:** Configure `webrtcbin` transceivers with explicit **SENDONLY** direction and **H.264** caps after setting remote description (before creating answer). Frontend sets H.264 as preferred codec via `setCodecPreferences()` to ensure codec match. Payload type 103 used to match Chrome's H.264 offer.
- **Rationale:** Initial SDP answers showed `a=inactive` and VP8 codec despite H.264 being sent. The fix required: (1) browser sends H.264-first codec preferences, (2) gateway uses `get-transceiver` signal to access and modify transceiver direction after offer is processed, (3) PT 103 matched between `rtph264pay` and browser offer. This resolved the connection failure (ICE connected but peer connection stayed at "connecting").
- **References:** noesis/mosaic_webrtc_gateway.py (_on_set_remote_description_done), oai2-fe/src/hooks/useWebRTCClient.ts (setCodecPreferences), config/infer.yaml (mosaic_output.mosaic_webrtc_enabled)

- **Date:** 2025-12-09
- **Author:** Codex
- **Area:** Pipeline / Sources & Lifecycle
- **Decision:** For the DS8 path, configure `nvmultiurisrcbin` directly from `infer.yaml` (URI list + streammux settings) but disable its embedded Civetweb REST server by setting `port="0"` and align key behavior flags (`drop-pipeline-eos`, cache/sort/align) with the DS7 `noesis_multiurisrcbin.ini`. In `ds8_runtime`, call `ds8_pipeline.prepare(on_message=_psm_message_cb)` and treat non-success prepare return codes from `pyservicemaker.Pipeline.prepare` as hard failures.
- **Rationale:** DS7 relied on `noesis_multiurisrcbin.ini` and a dynamic REST port allocator to avoid port-9000 conflicts; DS8 initially used nvmultiurisrcbin defaults, which started Civetweb on port 9000 and could abort the process when that port was in use. DS8 does not need the REST management API because URIs are driven by YAML, so disabling it avoids clashes while keeping ingest semantics. Wiring the documented `on_message` callback into `prepare()` and checking its integer return code ensures Service Maker errors and EOS/state transitions are logged and prevent partially initialized graphs from running silently.
- **References:** noesis/pipelines/ds8_pipeline.py (multi_cfg/prepare), noesis/ds8_runtime.py (`_on_pyservicemaker_message`, `main`), pipelines/noesis_multiurisrcbin.ini, gst-inspect-1.0 nvmultiurisrcbin

- **Date:** 2025-12-12
- **Author:** Codex
- **Area:** Mosaic Output / WebRTC Gateway
- **Decision:** Fix the RTSP→WebRTC gateway to (1) insert a leaky `queue` between `rtph264pay` and `webrtcbin` to avoid pre-negotiation backpressure stalls, (2) derive the H.264 RTP payload type from the browser’s SDP offer and set `rtph264pay.pt` accordingly (no hard-coded PT), and (3) move the gateway FPS probe to the `h264parse` output (encoded frames) instead of the RTP payloader output (packet-rate), avoiding Python/GIL starvation.
- **Rationale:** The previous gateway could stall after a single buffer and/or starve other Python threads (WS server, signal handling) because `webrtcbin` can block until negotiation and the RTP-level probe fires at ~kHz packet rates. Matching the offer’s PT is required by WebRTC SDP semantics, and queue isolation keeps media flowing even when peers are absent.
- **References:** noesis/mosaic_webrtc_gateway.py, websocket_server.py (webrtc_offer routing), plans/ds8_mosaic_webrtc_plan.md

- **Date:** 2025-12-12
- **Author:** Codex
- **Area:** Runtime / Shutdown
- **Status:** Superseded on 2026-07-10 by the acknowledged downstream-EOS
  lifecycle decision below. SIGTERM now initiates verified orderly quiescence.
- **Decision:** Treat WebSocket startup failure as fatal in DS8 runtime, and re-assert `SIGTERM` default handling (terminate) to make `timeout(1)`-based debug runs reliable (avoid orphaned DS8 processes holding ports).
- **Rationale:** A running DS8 pipeline without a WS server breaks WebRTC signaling and UI telemetry, and orphan processes cause misleading “websocket doesn’t connect” regressions via port conflicts. Default SIGTERM termination avoids reliance on Python-level shutdown when GI/GStreamer callbacks can delay signal handling.
- **References:** noesis/ds8_runtime.py

- **Date:** 2025-12-12
- **Author:** Codex
- **Area:** Hooks / MapAnything Postprocess
- **Decision:** Make MapAnything postprocess non-blocking by cloning DS8 `Tensor` outputs and processing conversion/storage/telemetry on a background worker thread with a bounded queue (drop-on-full). Also treat non-epoch PTS values as invalid and fall back to `time.time_ns()` to avoid 1970-dated depth snapshots.
- **Rationale:** Inline Python tensor conversion/storage inside the Service Maker metadata operator caused a repeatable pipeline stall (“~1–20 frames then GPU 0% / no frames to UI”). A bounded async worker prevents backpressure from stalling the pipeline, and stronger PTS epoch detection keeps depth timestamps and retention behavior sane.
- **References:** noesis/pipelines/hooks.py (MapAnythingProcessor async queue + PTS fallback), noesis/ds8_runtime.py (NOESIS_MAPANYTHING_POSTPROCESS_ENABLED), .cursor/debug.log (gateway frame counts)

- **Date:** 2025-12-12
- **Author:** Codex
- **Area:** WebRTC / Codec Interop
- **Decision:** For the RTSP→WebRTC gateway, enforce H.264 `packetization-mode=1` in webrtcbin transceiver caps and re-inject SPS/PPS periodically via `h264parse config-interval=1` and `rtph264pay config-interval=1`.
- **Rationale:** Browsers often prefer/require `packetization-mode=1` for H.264 WebRTC, and late-join peers can end up “connected but black” if SPS/PPS were only present at RTSP startup. Periodic config insertion plus explicit packetization mode stabilizes playback.
- **References:** noesis/mosaic_webrtc_gateway.py, oai2-fe/src/hooks/useWebRTCClient.ts

- **Date:** 2025-12-13
- **Author:** Codex
- **Area:** Mosaic Output / WebRTC Gateway

- **Date:** 2025-12-17
- **Author:** Codex
- **Area:** Hooks / Mosaic Overlay
- **Decision:** Implement DS8 mosaic motion trails as a `BatchMetadataOperator` probe attached upstream of `nvdsosd` (at `tiler`), rendering trails via Service Maker display metadata (`BatchMetadata.acquire_display_meta()` + `FrameMetadata.append()` + `pyservicemaker.osd.Line/Text`) with a time-window history PLUS a per-track point cap (`max_points_per_track`) and newest-first rendering to prevent “frozen heads” when display-meta capacity is reached; apply stride sampling + decimation, footpoint EMA smoothing, and age-based alpha fade; restrict to people class (`class_id=0`) and color by StableIDManager `stable_id` when available; expose multi-trail budgets via `max_segments_per_track`, `max_tracks`, and `max_display_metas`; also render the same trail history into per-camera BEV JPEGs in `BevRenderer` so trails are visible in the dashboard’s BEV views.
- **Rationale:** DS8’s canonical mosaic delivery is in-pipeline (tiler → osd → RTSP/WebRTC), so overlays must be rendered before `rtsp_out` without introducing CPU appsink branches or Flow retrievers. Time-window trails behave consistently across FPS; stable-id coloring gives visual continuity across re-tracks/cameras; locking ensures safe runtime toggling via WebSocket control messages.
- **References:** config/infer.yaml, noesis/pipelines/hooks.py, noesis/ds8_runtime.py, docs/reference/Deepstream 8 Structure
- **Decision:** Treat the RTSP→WebRTC bridge as a two-phase link: keep RTSP ingest running into a drain sink until a WebRTC offer arrives; after `set-remote-description` completes, request/link `webrtcbin`’s `sink_%u` pad and only then create an SDP answer.
- **Rationale:** Requesting/linking the sender pad before the offer caused “ICE connected but no video” (no RTP/decoded frames) because `webrtcbin` can answer `a=sendonly` without actually mapping a sender stream to the negotiated session. Deferring the link until after remote description ensures the sender stream is bound to the peer connection and media actually flows. Verified with `scripts/webrtc_gateway_smoke_test.py` for both PT=96 and PT=103 (browser-like PT).
- **References:** noesis/mosaic_webrtc_gateway.py, scripts/webrtc_gateway_smoke_test.py

- **Date:** 2025-12-23
- **Author:** Codex
- **Area:** Telemetry / UI Trails
- **Decision:** When `stable_id` is missing, use a camera-namespaced fallback color key (`(camera_index + 1) * 2^32 + track_id`) for frontend trails/legend and the DS8 mosaic overlay; keep stable_id-based colors global. Disable BEV JPEG binaries by default (meta-only mode), while preserving the existing BEV JPEG framing when enabled.
- **Rationale:** Namespaced fallback colors prevent cross-camera color collisions before stable IDs are available. Disabling BEV JPEGs reduces unnecessary CPU/bandwidth because the frontend uses cached floorplans + canvas overlays; opt-in preserves legacy behavior when needed.
- **References:** plans/DS8/ds8_trails_id_refactor_plan.md, plans/DS8/ds8_trails_id_refactor_work_order.md, oai2-fe/src/lib/camera.ts, oai2-fe/src/lib/trails.ts, noesis/pipelines/hooks.py, noesis/telemetry/bev.py, config/infer.yaml, docs/DS8_api_contracts_ws.md

- **Date:** 2026 (current implementation session)
- **Author:** Codex (per approved BEV data feeds plan)
- **Area:** Telemetry / BEV
- **Decision (superseding 2025-12-23):** BEV JPEG binary delivery fully retired from the active meta-only baseline. Removed conditional cv2 rendering + framed binary send from BevRenderer, the dead `onBevImage` path and binary parsing in the dashboard WS client, and the associated flag plumbing in the DS8 runtime. The general binary coalescer in the WS server is retained (commented) for potential future binary depth. Contract and runtime logs updated. DS8 changes validated first, then minimal parity applied to the active v3dt path per explicit requirement.
- **Rationale:** The path was dead in the canonical dashboard (handler never wired), produced unnecessary per-frame CPU + bandwidth when the flag was flipped during debug, and increased maintenance surface. Meta-only + canvas + floorplan is the supported, lower-cost path aligned with all recent design decisions.
- **References:** noesis/telemetry/bev.py, noesis/ds8_runtime.py, oai2-fe/src/hooks/useWebSocketClient.ts, websocket_server.py, docs/DS8_api_contracts_ws.md, plans/DS8/bev_dashboard_and_data_feeds_analysis_outline.md (Item 1)

- **Date:** 2026 (current session, Items 4+5)
- **Author:** Codex (per approved plan + user direction: "Minimal for #2 + config+discovery for #4 + continue all 5 now")
- **Area:** Dashboard / BEV
- **Decision:** Hardcoded 3-camera topology (Item 4) addressed via config + discovery (seed the original three + grow knownCameras from calibration-bundle, mosaic_layout, floorplan, and bev meta payloads). Added `discoverCamerasFromPayloads` helper. Frame mode / projection / calib parity (Item 5) strengthened with explicit vestigial flip comment + guard that surfaces missing calibration during world projection.
- **Rationale:** The original three rooms remain the default for perfect backward compat. New cameras now participate in BEV without code changes. The calib guard directly attacks a historical source of silent bad BEV output.
- **References:** oai2-fe/src/lib/camera.ts (discoverCamerasFromPayloads), oai2-fe/src/App.tsx, oai2-fe/src/lib/coordTransforms.ts, noesis/telemetry/bev.py, plans/DS8/bev_dashboard_and_data_feeds_analysis_outline.md

- **Date:** 2026-05-27
- **Author:** Codex (following external gpt-5.5 xhigh review + explicit user direction)
- **Area:** Dashboard / BEV / Camera Topology
- **Decision:** Completed the staged Item 4 goal by making the primary user-visible BEV panel rendering (the two .bev-row grids in App.tsx that host <BevView> components) dynamic, driven directly from the grown `knownCameras` list (original three seeded + discovered at runtime from calibration-bundle, mosaic_layout, floorplan, bev-frame, etc.). The legacy `cameraOrder` constant and `buildMosaicCameraIdToSlotKey` logic are retained exclusively for stable tiler source_id → slot mapping (a separate concern from which panels the user sees). Per-camera state maps/refs (bevMeta*, floorplanData, statuses, poses, latency, fps, occ, vacancy, prevActive, frameMode, etc.) are now lazily extended for newly discovered keys. CSS .bev-row already has responsive auto-fit support in key modes; no breaking layout changes for the common 3-cam case.
- **Rationale:** The "config+discovery" work was explicitly the first stage (user confirmation: "my only reasoning for choosing the config+discovery approach was to do this in stages. The goal should still be dynamic panels."). Switching the actual rendering lists from the static `cameraOrder` to the live `knownCameras` directly eliminates the hardcoded topology for the BEV experience the user watches, while preserving all contracts, mosaic stability, per-cam calibration/floorplan/trail integrity, and backward compatibility. Other secondary consumers (TelemetryPanel, LatencyCard, certain timer loops) left on the legacy list for this pass to bound scope and risk; they can follow in a later increment.
- **References:** Codex (gpt-5.5 xhigh) review 2026-05-27, plans/DS8/bev_dashboard_and_data_feeds_analysis_outline.md (hardcoded camera topology issue), prior 5-item plan.md (Item 4 staging note), oai2-fe/src/App.tsx (knownCameras + updateKnownCamerasFromPayload + the two cameraOrder.map sites + per-cam initializers), oai2-fe/src/lib/camera.ts (discoverCamerasFromPayloads + cameraOrder), oai2-fe/src/styles/app.css (.bev-row grid rules), docs/DS8_api_contracts_ws.md (BEV payload has no topology assumption).

- **Date:** 2026-05-27 (follow-up to dynamic panels)
- **Author:** Codex (after second external gpt-5.5 xhigh review)
- **Area:** Dashboard / BEV / Camera Topology
- **Decision (fix for resolver gap):** Strengthened `resolveDisplayCameraKey` (App.tsx) and `buildMosaicCameraIdToSlotKey` to support truly dynamic cameras. The resolver now falls back to accepting any normalized id that is already present in `knownCameras` (or treats unknown normalized strings as first-class dynamic `CameraKey`s). The mosaic mapper now populates entries using the actual `camera_id` values from layout sources (not only the legacy fixed 0/1/2 slots from `cameraOrder`). `cameraIndex` was made safe for dynamic keys via stable hash. This closes the last data-flow gap so that 4th/5th+ cameras discovered at runtime actually populate their `<BevView>` panels instead of producing empty slots.
- **Rationale:** The previous "dynamic panels" change made the JSX rows use `knownCameras`, but `handleBevMeta` still routed through the old resolver, silently dropping payloads for real extra cameras (exactly as the second Codex review diagnosed). The resolver + mapper fixes make end-to-end dynamic camera support work while preserving every legacy alias and the original three rooms.
- **References:** Second Codex (gpt-5.5 xhigh) review 2026-05-27, oai2-fe/src/App.tsx:resolveDisplayCameraKey + buildMosaicCameraIdToSlotKey + handleBevMeta, oai2-fe/src/lib/camera.ts:detectCameraKey + cameraIndex + discoverCamerasFromPayloads, prior design decision on staged Item 4.

- **Date:** 2026-05-27 (final hardening pass)
- **Author:** Autonomous follow-up after third Codex review
- **Area:** Dashboard / BEV / Camera Topology
- **Decision:** Hardened `detectCameraKey` (camera.ts) to use an explicit allow-list of strong aliases for the original three rooms only. Removed the previous broad `.includes('kitchen')` / `.includes('family')` / `.includes('living')` patterns that could silently collapse new dynamic camera names (e.g. `kitchen-2`, `family-room-side`) into the legacy rooms. Added conservative short prefixes only for the classic old forms. Dynamic names now reliably stay distinct.
- **Rationale:** The third Codex review (after the resolver fix) called this out as the remaining medium risk. The change eliminates a plausible source of future silent data mis-routing for extra cameras while preserving all documented legacy aliases for the original three rooms.
- **References:** Third Codex review (resolver fix) 2026-05-27, oai2-fe/src/lib/camera.ts:detectCameraKey.

- **Date:** 2026-05-27 (final closure)
- **Author:** Autonomous Codex-driven loop (per user authorization)
- **Area:** Dashboard / BEV / Camera Topology
- **Decision:** Closed the autonomous review/fix loop. Multiple sequential gpt-5.5 + xhigh Codex reviews (initial analysis → dynamic panels implementation → resolver/mapping fixes → alias hardening → final closure) resulted in a "solid" verdict. The only remaining items are low-risk polish (short-prefix alias collision surface + desire for a focused test). These were accepted as non-blocking.
- **Rationale:** The original high-priority "hardcoded 3-camera topology" problem from the BEV analysis outline has been resolved for the live dashboard experience. All critical data-flow paths for dynamically discovered cameras now work. The user authorized proceeding without further confirmation on each Codex response.
- **References:** Full sequence of Codex reports under plans/DS8/ (2026-05-27), the staged Item 4 work, resolver + alias hardening changes, final "solid" verdict.

- **Date:** 2025-12-13
- **Author:** Codex
- **Area:** WebRTC / Browser SDP Interop
- **Decision:** When parsing browser SDP offers that advertise multiple H.264 payload types, prefer `packetization-mode=1` and a constrained‑baseline `profile-level-id` (`42e0…`) when selecting the H.264 PT to mirror in the answer and configure `rtph264pay.pt`.
- **Rationale:** Some browser offers include multiple H.264 PTs for different profiles/packetization-modes; selecting the first H.264 PT (often `profile-level-id=42001f`) could trigger `webrtcbin` to generate an `a=inactive` answer for m=video (“ICE connected but no video”). Preferring constrained‑baseline improves interop and eliminates the inactive-answer failure mode for captured Chrome/Chromium-style offers.
- **References:** noesis/mosaic_webrtc_gateway.py (`_extract_h264_payload_type`), scripts/webrtc_gateway_browser_offer_replay_test.py, `.cursor/webrtc_offers/*.sdp`

- **Date:** 2025-12-13
- **Author:** Codex
- **Area:** Mosaic Output / Encoded Mosaic Sizing
- **Decision:** Keep the encoded mosaic output fixed at 1920×1080 (for RTSP/WebRTC encoder compatibility) and enable `nvmultistreamtiler square-seq-grid=true` by default so the tiler selects a square layout (e.g., 3 sources → 2×2) and avoids stretching streams into tall/skinny tiles. Provide an escape hatch via `NOESIS_MOSAIC_TILER_SQUARE_SEQ_GRID=0` plus `NOESIS_MOSAIC_TILER_COLUMNS` / `NOESIS_MOSAIC_TILER_ROWS` for explicit layouts.
- **Rationale:** DS7 achieved correct per-stream aspect in a 3×1 layout by increasing mosaic width (e.g., 5760×1080), but that can exceed NVENC/V4L2 limits for the DS8 RTSP/WebRTC encoding path. If the output stays 1920×1080, a 1×N tiler layout distorts 16:9 streams (e.g., 1×3 → 640×1080 tiles). Square tiling preserves per-tile 16:9 within the fixed output at the cost of empty tiles.
- **References:** noesis/pipelines/ds8_pipeline.py (tiler config), deepstream_video_pipeline.py (DS7 tiler sizing), DEBUG_PROGRESS.md

- **Date:** 2025-12-14
- **Author:** Codex
- **Area:** WebRTC Gateway / Backpressure
- **Decision:** Move the gateway’s leaky queue to the H.264 access‑unit stage (`h264parse → queue(leaky) → rtph264pay`) and keep the RTP queue non‑leaky (`rtph264pay → queue → webrtcbin`), and set `h264parse/rtph264pay config-interval=-1` (SPS/PPS with every IDR).
- **Rationale:** A leaky, tiny queue placed *after* `rtph264pay` can drop individual RTP packets under negotiation/backpressure, corrupting keyframes. This produces the failure mode “ICE connected, bytesReceived increasing, framesDecoded=0 until much later” because the browser never receives an intact IDR to start decoding. Dropping whole frames before packetization avoids partial-keyframe corruption while still preventing RTSP ingest from stalling.
- **References:** noesis/mosaic_webrtc_gateway.py, docs/DS8_testing_guide.md, DEBUG_PROGRESS.md

- **Date:** 2025-12-14
- **Author:** Codex
- **Area:** Mosaic Output / Runtime Toggles
- **Decision:** Historical transitional state (superseded on 2026-01-10): NVJPEG/appsink mosaic output was temporarily kept opt-in while RTSP/WebRTC became canonical.
- **Rationale:** This was an intermediate migration step; current DS8 state removes mosaic NVJPEG/appsink support entirely to avoid conflicting mosaic delivery paths.
- **References:** noesis/pipelines/ds8_pipeline.py (historical branch-gating removal), noesis/ds8_runtime.py (historical startup log).
  _2026-01-10 (Codex): Superseded — DS8 no longer supports mosaic NVJPEG/appsink output; `mosaic_output.jpeg_enabled` is removed and mosaic video is RTSP→WebRTC only._

- **Date:** 2025-12-14
- **Author:** Codex
- **Area:** Tracking / Stable IDs
- **Decision:** Feed `StableIDManager` with lightweight per-track synthetic patches (bbox-derived colors, bounded 256×256) instead of adding CPU appsink branches; fall back to `stable_id=track_id` when the manager returns `None`, and expose `NOESIS_REID_TEST_MODE=1` to inject a synthetic tracking sample for smoke tests.
- **Rationale:** Stable IDs must stay enabled (NOESIS_REID_ENABLED default on) without violating the GPU-first policy. Synthetic patches provide deterministic input for the manager and allow smoke tests to assert non-null/persistent IDs without adding full-frame CPU crops or backpressure.
- **References:** noesis/pipelines/hooks.py (_reid_crop_from_track, _maybe_assign_stable_id, NOESIS_REID_TEST_MODE), noesis/ds8_runtime.py (_build_stable_id_manager), scripts/reid_stable_id_smoke_test.py.
  _2025-12-20 (Codex): Superseded by ID Contract v2 (`plans/DS8/ds8_id_contract_v2.md`): DS8 must use real person crops/embeddings for global stable IDs; synthetic patches are removed/disabled and stable_id may be null until embeddings are available._

- **Date:** 2025-12-20
- **Author:** Codex
- **Area:** Identity / Stable IDs
- **Decision:** Adopt ID Contract v2: treat `stable_id` as the master, global person identity (cross-camera) used for all user-facing features; allow a single `stable_id` to be concurrently active on multiple cameras (overlapping FoVs); compute stable IDs from real person crops/embeddings (no synthetic patches), producing crops in-pipeline via Service Maker `BufferOperator` + DLPack (GPU crop, CPU ROI upload) and emitting `stable_id=null` when ReID is disabled/unavailable.
- **Rationale:** DS8’s current synthetic-patch feed prevents appearance-based matching, so stable IDs cannot be meaningfully global across cameras and cannot reliably survive tracker ID churn. Contracting stable IDs as primary makes UI joins/replay predictable and enables overlap deployments. Cropping on GPU and uploading only the ROI preserves the GPU-first policy without introducing new appsink/CPU branches.
- **References:** plans/DS8/ds8_id_contract_v2.md, reid/stable_id_manager.py, noesis/pipelines/hooks.py, noesis/pipelines/ds8_pipeline.py, docs/DS8_api_contracts_ws.md, docs/DS8_metadata_contracts.md

- **Date:** 2025-12-14
- **Author:** Codex
- **Area:** Depth / Floorplan RPC
- **Decision:** `DepthStorageManager.generate_topdown_floorplan` now returns a minimal zero-density payload (cached + persisted) when no valid points exist and honours `max_age_sec`/cache-only paths; calibration bundles are seeded onto the manager in `ds8_runtime` and reused for floorplan RPCs.
- **Rationale:** UI/WS clients expect a floorplan payload even when depth snapshots have no valid points; previously this returned errors like `no_points` or `missing_calibration`. A zero-grid response keeps RPCs successful, improves cache hit rate, and surfaces stale/missing data via bounds/point_count while avoiding restarts.
- **References:** geometry/depth_source.py (DepthStorageManager caches/fallback), noesis/ds8_runtime.py (_ds8_floorplan_provider, calibration seeding), scripts/floorplan_rpc_smoke_test.py.
- **Date:** 2025-12-13
- **Author:** Codex
- **Area:** Analytics / Exclusion
- **Decision:** Treat `config/nvdsanalytics.yaml` as the exclusion source of truth: ROI REST updates rewrite the YAML, regenerate `config/config_nvdsanalytics_exclude.ini`, refresh `pipeline.config` stages, and node.set both `analytics` and `analytics_exclude`; hooks re-read the live analytics YAML when extracting exclusion polygons.
- **Rationale:** The DS8 pipeline had been reading stale INI configs, so ROI REST edits never reached the running nvdsroiexclude element. Syncing YAML→INI and pushing runtime updates keeps exclusion behavior aligned with live ROI edits without a restart.
- **References:** noesis/server/analytics_api.py, noesis/pipelines/hooks.py, noesis/pipelines/ds8_pipeline.py

- **Date:** 2025-12-18
- **Author:** Codex
- **Area:** UI / BEV Visualization
- **Decision:** Draw BEV motion trails client-side in the dashboard `BevView` canvas using `bev-frame.footpoints` (and optional `stableId`) rather than baking trails into a frequently-updated BEV JPEG.
- **Rationale:** The dashboard BEV panels already render the floorplan height map client-side and should remain static/low-refresh; a canvas overlay provides smooth, multi-track trails without increasing JPEG encoding/bandwidth.
- **References:** oai2-fe/src/components/BevView.tsx, noesis/telemetry/bev.py, docs/DS8_api_contracts_ws.md

- **Date:** 2026-03-05
- **Author:** Codex
- **Area:** UI / BEV Visualization
- **Decision:** Retain the last renderable floorplan in `oai2-fe` and ignore transient `floorplan_response` errors or payloads with no drawable grid layers when painting dashboard BEV views.
- **Rationale:** The backend can legitimately return stale/missing floorplan responses after the initial cache warmup while BEV footpoint telemetry continues. Preserving the last good floorplan keeps the dashboard background stable instead of dropping to a blank canvas until a manual refresh.
- **References:** oai2-fe/src/App.tsx, oai2-fe/src/components/BevView.tsx, plans/DS8/ds8_master_work_orders.md

- **Date:** 2025-12-13
- **Author:** Codex
- **Area:** Depth / MapAnything telemetry
- **Decision:** Store MapAnything depth snapshots under FE camera labels (with numeric fallback) and seed `DepthStorageManager.calibration_bundle` from `cameras.yaml`/calibration assets; WS depth/floorplan providers map numeric/name camera keys to return DS7-compatible payloads.
- **Rationale:** Depth drawer/floorplan requests are keyed by camera name, but DS8 was storing snapshots under numeric source ids without calibration, producing “no depth”/`missing_calibration` responses. Canonical camera ids plus a populated calibration bundle align DS8 responses with FE expectations.
- **References:** noesis/pipelines/hooks.py (MapAnythingProcessor camera labels), noesis/ds8_runtime.py (_ds8_ma_depth_provider/_ds8_floorplan_provider, calibration bundle wiring), geometry/depth_source.py

- **Date:** 2025-12-13
- **Author:** Codex
- **Area:** Tracking / Stable IDs
- **Decision:** Enable `StableIDManager` in DS8 runtime (gated by `NOESIS_REID_ENABLED`) using bbox-only updates (no new CPU/appsink crops) and maintain/prune tracks each frame; tracking telemetry carries stable_id/dwell fields matching DS7 schema.
- **Rationale:** DS8 previously omitted stable ids, diverging from DS7 occupancy/telemetry semantics. Instantiating StableIDManager preserves schema parity while keeping the GPU-first guardrail; embeddings remain optional via the existing manager.
- **References:** noesis/ds8_runtime.py (_build_stable_id_manager, pipeline.stable_id_mgr), noesis/pipelines/hooks.py (_maybe_assign_stable_id/_maintain_stable_ids), plans/DS8/ds8_migration_checklist_ds8_pipeline.md

- **Date:** 2025-12-14
- **Author:** Codex
- **Area:** MapAnything / SGIE Gating
- **Decision:** Gate MapAnything SGIE compute with an explicit GStreamer `valve` in the DS8 graph (`main_tee → mapanything_queue → mapanything_valve → mapanything_fullframe`) and “prime” the branch at startup: briefly open the valve during activation, then close it when depth remains disabled (default). The prime window is configurable via `NOESIS_MAPANYTHING_GATE_PRIME_SECONDS` (default 1.0s).
- **Rationale:** The DS7 stack uses valve-based gating so MapAnything only runs on-demand. In DS8, BufferOperator-only gating was not reliably preventing SGIE compute and starting with the valve closed could stall the pipeline at ~1 frame (no WebRTC video; GPU ~0%) because the SGIE branch hadn’t prerolled. Priming allows SGIE pads/events to preroll, then the valve closes to keep GPU utilization near the DS7 baseline until depth is explicitly enabled.
- **References:** noesis/pipelines/ds8_pipeline.py (mapanything_valve, activate prime), noesis/ds8_runtime.py (activation via ds8_pipeline.activate), plans/DS8/ds8_migration_checklist_ds8_pipeline.md
  _2025-12-16 (Codex): Validated end-to-end: DS8 runtime starts with depth OFF by default (`NOESIS_DEPTH_ENABLE_SECONDS=0`), `/api/v1/depth/refresh?seconds=N` toggles the valve deterministically (logs `Depth gate toggled: enabled=… valve.drop=…`), GPU spikes only during the window, and oai2-fe shows depth heatmap + views while WebRTC mosaic stays connected._

- **Date:** 2025-12-16
- **Author:** Codex
- **Area:** Mosaic Output / WebRTC vs NVJPEG
- **Decision:** Treat RTSP→WebRTC gateway as the canonical mosaic output and remove NVJPEG/appsink mosaic output from supported DS8 paths.
- **Rationale:** A single mosaic delivery contract (RTSP ingress to WebRTC playback) prevents drift and removes backpressure/branching risks from deprecated JPEG mosaic plumbing.
- **References:** noesis/pipelines/ds8_pipeline.py (`rtsp_out`), noesis/ds8_runtime.py (gateway startup), noesis/mosaic_webrtc_gateway.py, scripts/webrtc_gateway_smoke_test.py
  _2026-01-10 (Codex): Implemented backend removal of the mosaic NVJPEG/appsink path; docs/checklists updated to RTSP→WebRTC-only mosaic._

- **Date:** 2025-12-15
- **Author:** Codex
- **Area:** MapAnything / ONNX export + TensorRT engine
- **Decision:** Export MapAnything ONNX via `MapAnything.forward()` (torch-only) and derive `depth/conf/mask` tensors for SGIE output; configure the SGIE to run at `infer-dims=3;294;518` with aspect-preserving padding, and build the TensorRT engine with fixed batch=3 (min=opt=max=3) to match the DS8 deployment’s `batch_size=3`.
- **Rationale:** Exporting via `MapAnything.infer()` included numpy-based postprocess and produced mostly-zero/sparse depth outputs in DS8 (black heatmap). On TensorRT 10.13, building the engine with dynamic batch 1–3 hit a Myelin internal error; fixed batch 3 is stable and matches the streammux batch size used by this repo’s DS8 config. 294×518 preserves 16:9 structure for 1080p sources while aligning with the model’s patch constraints.
- **References:** export_ma_onnx/export_to_onnx.py, pipelines/config_infer_secondary_mapanything.ini, config/infer.yaml, scripts/ma_depth_rpc_smoke_test.py

- **Date:** 2025-12-17
- **Author:** Codex
- **Area:** Pipeline / Mosaic layout
- **Decision:** Default the DS8 tiler to an explicit 3-column by 1-row mosaic so the primary RTSP/WebRTC output matches the one-row view expected by the UI, while keeping the previous square-seq-grid behavior and the `NOESIS_MOSAIC_TILER_COLUMNS/ROWS` overrides available.
- **Rationale:** The UI now consumes one composite stream, so forcing a single row of three tiles avoids the tall skinny tiles from square-auto layouts and keeps the player’s `StreamPanel` from introducing extra padding or distortion; rows still auto-scale (via ceil(source_count/columns)) if more cameras are added, and the telemetry log `ds8_mosaic_tiler_config` now reports the fixed 3×1 grid for debugging.
- **References:** noesis/pipelines/ds8_pipeline.py (tiler config), oai2-fe/src/styles/app.css (stream overlay), plans/DS8/ds8_migration_checklist_ds8_pipeline.md

- **Date:** 2025-12-16
- **Author:** Codex
- **Area:** Telemetry / BEV calibration
- **Decision:** Auto-fit BEV extents now use camera-local ray hits (translated/rotated by camera pose) and intrinsics are scaled to the streammux resolution using camera specs/intrinsics models (with a principal-point fallback) to mirror DS7 behavior.
- **Rationale:** DS8 was mixing camera-local footpoints with world-frame extents and unscaled intrinsics (e.g., 720p models), pushing BEV points into the top-left corner and dropping most of the room coverage. Local extents plus resolution-scaled K keep BEV overlays centered and consistent across all cameras.
- **References:** noesis/telemetry/bev.py, noesis/ds8_runtime.py, config/camera_calibration.json, config/cameras.yaml, intrinsics.json
  _2026-01-27 (Codex): Superseded — BEV now stays in the calibration world frame to align with legacy tracker coordinates when global extrinsics are in use._

- **Date:** 2025-12-16
- **Author:** Codex
- **Area:** Depth / MapAnything alignment
- **Decision:** Undo MapAnything SGIE letterboxing by cropping symmetric padding and resizing depth/conf/mask to the streammux (camera) resolution before persistence, so stored depth snapshots line up with intrinsics and downstream floorplan/BEV consumers.
- **Rationale:** DS8 had been storing 294×518 model-space tensors while floorplan generation assumed full-res camera coordinates, producing an up-left shifted “wedge” and missing coverage. Aligning tensors back to the camera frame restores DS7-style geometry and keeps DepthStorageManager outputs in the same pixel space as intrinsics.
- **References:** noesis/pipelines/hooks.py (`MapAnythingProcessor._align_to_frame`), pipelines/config_infer_secondary_mapanything.ini (maintain-aspect-ratio/symmetric-padding), geometry/depth_source.py (floorplan projection uses intrinsics).

- **Date:** 2025-12-31
- **Author:** Codex
- **Area:** Calibration / Menon WebSocket Compatibility
- **Decision:** Accept Menon calibration identifiers sent as objects (e.g., `{"id","name"}`) by coercing them into a string camera id, and treat Menon legacy `{cmd:"calibrate"}` as a compatibility alias that runs depth-based auto-calibration only when not immediately preceded by `set_extrinsics` (otherwise it acts as an ACK to avoid overwriting Menon-provided extrinsics).
- **Rationale:** Menon frequently sends `{cmd:"calibrate"}` right after publishing `set_extrinsics`; without a guard this would trigger depth auto-calibration and overwrite the extrinsics that Menon is intended to own (home 3D model source-of-truth). Coercing structured camera ids avoids requiring immediate frontend changes while keeping DS8’s calibration storage keyed to `config/cameras.yaml` labels.
- **References:** websocket_server.py (`cmd:"calibrate"` handler + `set_extrinsics` timestamp guard), noesis/ds8_runtime.py (camera id resolution + `unknown_camera` guard), plans/DS8/ds8_migration_checklist_ds8_runtime.md, plans/DS8/ds8_migration_checklist_websocket_server.md

- **Date:** 2026-01-01
- **Author:** Codex
- **Area:** MapAnything / Pose Inputs
- **Decision:** When supplying camera poses to MapAnything, derive MapAnything `camera_poses` as `T_wc = inv(E)` from `config/camera_calibration.json` where `E` is stored as world→camera (column-major), then apply `config/ply_alignment.json` as a post-multiply `T' = M @ T_wc` (row-major), and apply `units.s_obj_to_m` to translation only (unit conversion without scaling rotation).
- **Rationale:** MapAnything expects OpenCV cam→world 4×4 poses; Noesis stores world→camera extrinsics as `E`. Alignment and unit scaling are needed for consistent "world" coordinates across consumers, but scaling must not be baked into the rotation block of a pose matrix. Keeping the conversion in one shared utility prevents convention drift when re-exporting/rebuilding pose-conditioned MapAnything models/engines.
- **References:** noesis/metadata/mapanything_pose.py, config/camera_calibration.json, config/ply_alignment.json, docs/DS8_MIGRATION_KNOWLEDGE_BASE.md

- **Date:** 2026-01-01
- **Author:** Codex
- **Area:** Calibration / Unified Conventions
- **Decision:** Standardize all DS8 calibration on the following conventions:
  - **E (extrinsics):** 4×4 world→camera, **column-major** (Fortran order), Y-up world, +Z forward camera, meters.
  - **K (intrinsics):** Loaded from `config/cameras.yaml` ONLY; `intrinsics.json` and `config.py` are deprecated.
  - **align.matrix:** 4×4 **row-major** (C order); Menon applies it client-side; DS8 `pixel_to_world` does NOT apply it.
  - **`set_extrinsics` inputs:** Accept both `E` (world→camera) and `Twc` (camera→world, inverted before storage).
  - **Streammux scaling:** K is scaled when source intrinsics resolution differs from streammux output.
- **Rationale:** Multiple calibration sources and inconsistent conventions caused K divergence between `calibration-bundle` and runtime snapshots. A single `CalibrationManager` owning all loading/validation/broadcasting eliminates this drift.
- **References:** plans/DS8/ds8_calibration_workflow_unification_work_order.md, docs/DS8_api_contracts_ws.md §8, noesis/calibration/manager.py

- **Date:** 2026-01-02
- **Author:** Codex
- **Area:** Calibration / Frontend Name Resolution
- **Decision:** Implement robust "fuzzy matching" resolution in Menon's `CalibrationManager` to map user-facing device names to canonical backend stream keys. Logic includes:
  1. Exact match.
  2. Case-insensitive match.
  3. Normalization (spaces to underscores/hyphens).
  4. Substring containment (stripping explicit "Camera" suffix/noise).
- **Rationale:** Users tend to name devices descriptively (e.g., "Family Room Camera") while the backend requires strict keys from `camera_calibration.json` (e.g., "family-room"). Enforcing strict naming in the UI is hostile; strict backend keys are necessary for config stability. The resolver bridges this gap transparently, logging the mapping for verification.
- **References:** Menon/src/features/calibration/CalibrationManager.js (`resolveCameraId`), noesis/config/camera_calibration.json

- **Date:** 2026-01-14
- **Author:** Codex
- **Area:** SV3DT / Distortion Handling
- **Decision:** For wide-angle cameras, undistort the video **before** PGIE/tracker using `nvdewarper`, and treat the dewarped output as pinhole (fx/fy/cx/cy preserved, distortion zeroed) for SV3DT camInfo generation and tilt fitting.
- **Rationale:** SV3DT camInfo is pinhole-only; leaving distortion unmodeled causes region-dependent tracking and undersized cuboids. `nvdewarper` is the DeepStream-supported GPU path for correcting lens distortion.
- **References:** `/opt/nvidia/deepstream/deepstream-8.0/sources/apps/sample_apps/deepstream-dewarper-test/README`, `config/dewarper_family_room_charuco_rtsp.txt`, `config/infer_v3dt_medium_preview_charuco_fr_rtsp_dewarp.yaml`

- **Date:** 2026-01-27
- **Author:** Codex
- **Area:** Telemetry / BEV / Coordinate frame
- **Decision:** Keep BEV footpoints and auto-fit extents in the calibration world frame (homography X/Z) and remove camera-local yaw/translation rotation. Continue using camera position only for max-distance gating.
- **Rationale:** The camera-local rotation introduced for V3DT caused BEV points to drift relative to legacy tracker world coordinates when global extrinsics are in use. World-frame BEV keeps tracking telemetry and BEV views aligned for non-V3DT runs.
- **References:** noesis/telemetry/bev.py, docs/DS8_metadata_contracts.md, plans/DS8/v3dt/work_order.md
## 2026-01-21: Bidnetpipe uses a custom segmentation parser for floor-only masks

- **Decision:** For the Bidnetpipe DS8 segmentation test pipeline, use a custom
  semantic segmentation parser (`NvDsInferParseCustomBiSeNetFloor`) so the
  `NvDsInferSegmentationMeta` class_map is generated as a binary floor vs
  background mask.
- **Rationale:** `nvinfer`/`nvsegvisual` do not provide a config-driven per-class
  filter for semantic segmentation masks, and host-side `class_map` rewrites were
  not reflected in `nvsegvisual`. A custom parser ensures the metadata itself is
  already filtered before visualization and downstream stats.
- **Docs/refs:** `nvdsinfer_custom_impl.h` for the parser signature; `gst-nvinfer`
  config keys `custom-lib-path` and `parse-segmentation-func-name` (see
  `Bidnetpipe/DOCS.md`).

- **Date:** 2026-01-29
- **Author:** Codex
- **Area:** Telemetry / BEV / Image-axis alignment
- **Decision:** Auto-infer image-axis flips from extrinsics (right/up vs expected) so BEV footpoints are always camera-local with +X to image-right and +Z forward; update the FE to render BEV X without a hard-coded mirror.
- **Rationale:** V3DT-era calibrations introduced a 180° roll (X/Y sign change) relative to legacy extrinsics, which inverted BEV trails vs the stream. Auto-detecting axis flips keeps BEV aligned with the image/floorplan without requiring manual calibration edits.
- **References:** `noesis/telemetry/bev.py`, `geometry/homography.py`, `oai2-fe/src/components/BevView.tsx`, `docs/DS8_api_contracts_ws.md`

- **Date:** 2026-01-29
- **Author:** Codex
- **Area:** Depth / Floorplan / Coordinate frame
- **Decision:** Generate floorplan bounds/grids in the same camera-local ground-plane frame as BEV (translate by camera position, rotate by yaw) and invalidate cached floorplans that lack the new `frame` marker.
- **Rationale:** V3DT-era calibrations shifted extrinsics to a shared world frame; floorplans were still generated in raw camera coordinates, causing BEV trails to appear scaled against the background. Aligning floorplan coords to BEV restores consistent overlays for default tracking.
- **References:** `geometry/depth_source.py`, `noesis/telemetry/bev.py`

- **Date:** 2026-01-29
- **Author:** Codex
- **Area:** Calibration / BEV / Legacy extrinsics
- **Decision:** Use `config/camera_calibration_legacy.json` for default (non‑V3DT) tracking BEV/floorplan calibration, with `NOESIS_CALIBRATION_EXTRINSICS` as an explicit override and `camera_calibration.json` reserved for V3DT runs.
- **Rationale:** The V3DT calibration file introduces pose differences that exaggerate BEV depth for kitchen/living; default tracking is validated against the pre‑V3DT calibration set.
- **References:** `noesis/ds8_runtime.py`, `config/camera_calibration_legacy.json`

- **Date:** 2026-02-10
- **Author:** Codex
- **Area:** Calibration / Menon world unification
- **Decision:** For Menon world unification runs, make PoseV1 (`position`, `yaw_pitch_roll_deg`, `rotation_order=YXZ`, `frame=menon_scene`) the strict calibration authority for baseline DS8: strict pose-only mode is enabled by default, startup aborts when any configured camera lacks valid pose/floor geometry, runtime `set_extrinsics` rejects legacy `E`/`Twc`-only payloads in strict mode, and baseline BEV/tracking world outputs default to `menon_scene`.
- **Rationale:** The feature is easier to validate and debug when one calibration authority and one world frame are enforced; mixed extrinsics sources and meter/coercion branches caused ambiguous behavior and harder defect isolation.
- **References:** `noesis/ds8_runtime.py`, `noesis/pipelines/hooks.py`, `noesis/telemetry/bev.py`, `calibration_bundle.py`, `scripts/calibration_rpc_smoke_test.py`, `scripts/menon_pose_calibration_smoke_test.py`, `plans/menon_world_unification/contracts.md`

- **Date:** 2026-02-10
- **Author:** Codex
- **Area:** Calibration / Menon world unification / baseline source lock
- **Decision:** For baseline (non-`v3dt`) DS8 runtime, default calibration source is `config/camera_calibration_menon_obj.json` with fail-loud behavior when missing; keep `v3dt` default on `config/camera_calibration.json`. Preserve `NOESIS_CALIBRATION_EXTRINSICS` as explicit operator override for both modes.
- **Rationale:** The previous implicit fallback made root-cause analysis ambiguous when trails/points appeared compressed; one mode-specific default artifact removes hidden calibration source switching while preserving V3DT compatibility.
- **References:** `noesis/ds8_runtime.py`, `config/camera_calibration_menon_obj.json`, `config/camera_calibration.json`, `plans/menon_world_unification/contracts.md`

- **Date:** 2026-03-24
- **Author:** Codex
- **Area:** DS8 utility prototype / living-room pose compensation
- **Decision:** Apply a testpipeline-only local `roll z = 180` compensation to the `testpipelines/yolo26-seg-depth-3d` living-room source pose and publish it as `calibration.pose_compensation`, while leaving Menon's live reprojection pose and runtime logic unchanged.
- **Rationale:** Playwright + live frame analysis showed the dense RGBD shell was upside down because image-down projected toward world-up. Applying the correction only inside the prototype lets us validate the right pose fix visually before changing the main app.
- **References:** `testpipelines/yolo26-seg-depth-3d/prototype_calibration.py`, `testpipelines/yolo26-seg-depth-3d/viewer/src/App.tsx`, `testpipelines/yolo26-seg-depth-3d/scene_stream.py`

- **Date:** 2026-03-25
- **Author:** Codex
- **Area:** Calibration / PoseV1 camera basis
- **Decision:** Treat PoseV1 `menon_scene` camera poses as scene-camera summaries that require a fixed local 180 degree roll when converting to DS8/OpenCV image extrinsics, and keep stored `pose` and raw `E` synchronized at persistence time.
- **Rationale:** The previous PoseV1->`E` conversion used the scene pose basis directly as an image-camera basis. That made lower image rows project toward world-up, which inverted dense depth/world projections while leaving forward direction superficially plausible. Applying the fixed local basis rotation at the canonical pose<->extrinsics boundary fixes the foundational roll error once, and synchronizing raw `E` prevents downstream consumers like V3DT camInfo generation from reading stale pre-fix matrices.
- **References:** `noesis/calibration/pose_v1.py`, `calibration_bundle.py`, `noesis/calibration/manager.py`, `config/camera_calibration.json`, `config/camera_calibration_menon_obj.json`, `scripts/generate_v3dt_caminfo.py`, `tests/test_menon_pose_extrinsics.py`, `tests/test_calibration_manager.py`

- **Date:** 2026-04-03
- **Author:** Codex
- **Area:** DS8 utility prototype / Menon projection shell contract
- **Decision:** Publish explicit prototype shell geometry in canonical backend camera-local meters and Menon-anchor-local meters, and make Menon’s debug projection shell inherit full pose from each camera’s `DeviceVisual` orientation while solving one shared `scene_units_per_meter` from environment ray hits with pairwise camera scale only as a bootstrap fallback.
- **Rationale:** The earlier Menon shell kept reconstructing local geometry from world points and parenting it under position-only camera groups, which reintroduced flipped/backward shells and let orientation error leak into scale. Publishing local shell coordinates once at the prototype boundary and anchoring them to the authored Menon camera pose keeps the calibration-to-projection pipeline explicit and leaves scale as a single shared room-fit solve instead of a hidden per-camera reinterpretation.
- **References:** `testpipelines/yolo26-seg-depth-3d/scene_stream.py`, `testpipelines/yolo26-seg-depth-3d/main.py`, `testpipelines/yolo26-seg-depth-3d/viewer/src/App.tsx`, `../Menon/src/features/path-visualization/ProjectionShellOverlay.js`, `../Menon/src/three/DeviceLayer.ts`

- **Date:** 2026-04-03
- **Author:** Codex
- **Area:** Menon projection shell / anchored multi-camera placement
- **Decision:** Menon’s primary multi-camera shell path uses prototype-published `anchor_local_xyz` as the only shell geometry input, parents each shell under a neutral child that copies the corresponding `DeviceVisual` pose, and derives one shared `scene_units_per_meter` from camera-anchor pairwise distances. The older scene-space re-scaling path and environment-ray-fit scalar are not the primary placement authority.
- **Rationale:** The previous scene-space shell path kept reinterpreting already-correct geometry after the prototype boundary, so orientation regressions and size inflation reappeared whenever shells were switched back to scene-space scaling. Using anchor-local geometry plus the authored camera-object pose keeps shell orientation faithful to the calibration authoring path, while a shared pairwise camera scale gives one stable cross-camera scalar without contaminating it with per-frame shell content.
- **References:** `testpipelines/yolo26-seg-depth-3d/scene_stream.py`, `testpipelines/yolo26-seg-depth-3d/prototype_calibration.py`, `../Menon/src/features/path-visualization/ProjectionShellOverlay.js`, `../Menon/src/three/DeviceLayer.ts`

- **Date:** 2026-04-05
- **Author:** Codex
- **Area:** Menon projection shell / anchored basis and shared scale constraints
- **Decision:** The anchored projection shell now uses one explicit fixed camera-local to Menon-anchor-local basis (`camera_roll_pi`) published by the prototype, and Menon no longer derives a per-camera residual basis from calibration pose deltas. Shared shell scale is solved from environment ray hits, then capped by a cross-camera non-overlap constraint before rendering under the authored camera anchors.
- **Rationale:** The previous residual-basis path effectively canceled the camera anchors’ authored rotations, so shells kept drifting back toward calibration-scene orientation and family-room could look wrong even when its Menon camera object was correct. Separating orientation from scale keeps camera anchors authoritative for pose, while the overlap cap prevents the environment-fit scalar from inflating until non-overlapping camera shells collide in scene space.
- **References:** `testpipelines/yolo26-seg-depth-3d/prototype_calibration.py`, `testpipelines/yolo26-seg-depth-3d/scene_stream.py`, `testpipelines/yolo26-seg-depth-3d/main.py`, `../Menon/src/features/path-visualization/ProjectionShellOverlay.js`

- **Date:** 2026-04-14
- **Author:** Codex
- **Area:** Menon projection shell / inspection visibility
- **Decision:** When Menon’s projection-shell overlay is enabled, it temporarily switches the environment to ghost mode for inspection, keeps shell points visible in both mesh and point-cloud modes, and renders mesh mode with brighter fill plus explicit wireframe. The shell-only prototype’s multi-camera export defaults to a denser `scene_grid_width=96` so room patches are readable at whole-house zoom.
- **Rationale:** The prior combination of dense house walls, low shell sampling, and mesh-only visibility made aligned shells look like “nothing” or tiny slivers even when the geometry was correct and in-frame. The inspection preset makes the projection layer legible without requiring manual ghost toggles, and the denser export reduces the blotchy under-sampled look that broke visual validation.
- **References:** `testpipelines/yolo26-seg-depth-3d/main.py`, `../Menon/src/features/path-visualization/ProjectionShellOverlay.js`, `../Menon/src/ui/components/SettingsPanel.js`, `../Menon/src/main.ts`

- **Date:** 2026-04-14
- **Author:** Codex
- **Area:** Menon projection shell / prototype reconstruction authority
- **Decision:** Menon’s primary live point-cloud and mesh overlay now mirrors the prototype viewer contract: per-camera and `__scene_fusion__` geometry is rendered from packet world points transformed directly into `menon_scene`, using packet RGB colors as vertex colors. Anchor-local shell scaling, FoV shell sizing, and authored device-camera pose parenting are diagnostic-only paths and are not the active reconstruction renderer.
- **Rationale:** The user goal is to align the prototype’s live 3D reconstruction with the house model, not to visualize camera frusta. Re-basing reconstruction under camera anchors and recoloring it with debug colors made the result look like drifting slivers and obscured whether the actual room surfaces were aligned. Rendering the exact prototype reconstruction basis in Menon keeps the consumer faithful to the producer and makes visual validation meaningful.
- **References:** `testpipelines/yolo26-seg-depth-3d/viewer/src/App.tsx`, `testpipelines/yolo26-seg-depth-3d/scene_stream.py`, `../Menon/src/features/path-visualization/ProjectionShellOverlay.js`

- **Date:** 2026-04-14
- **Author:** Codex
- **Area:** Menon-scene registration / per-camera correction
- **Decision:** Keep backend-world reconstruction packets canonical, but publish an explicit per-camera `scene_registration_correction` payload and apply it only in Menon-scene rendering as a translation-only scene delta derived from calibration-scene camera pose versus the authored Menon camera position. Menon’s default reconstruction websocket also returns to the full prototype feed on `8773`; the shell-only `8774` stream stays diagnostic.
- **Rationale:** The persisted world-to-scene similarity from camera positions gets the reconstruction into the right part of the house, but the camera centers still disagree with the authored Menon camera placements by tens of scene units, which leaves slabs hanging outside room envelopes. A translation-only correction materially improves visual alignment from multiple viewing angles without corrupting backend-world truth or trusting the authored camera rotations, which are still not reliable enough to drive a full rigid per-camera correction.
- **References:** `testpipelines/yolo26-seg-depth-3d/prototype_calibration.py`, `testpipelines/yolo26-seg-depth-3d/main.py`, `testpipelines/yolo26-seg-depth-3d/scene_stream.py`, `testpipelines/yolo26-seg-depth-3d/viewer/src/App.tsx`, `../Menon/src/features/path-visualization/ProjectionShellOverlay.js`

- **Date:** 2026-05-02
- **Author:** Codex
- **Area:** Virtual twin reconstruction / Noesis-owned artifact boundary
- **Decision:** Add the living-room virtual twin as an offline Noesis revision bundle built from DS8-persisted MapAnything depth snapshots plus ZeroPlane plane inference, with registration against Menon structural OBJ surfaces and Menon consuming only the served GLB/JSON artifacts. Each revision persists its RGB keyframes and MapAnything depth/confidence/mask arrays, then filters the browser GLB against the Menon structural envelope and records a room/model leakage gate.
- **Rationale:** Dense browser-side projection had too many calibration and performance authorities. A versioned backend artifact keeps the DS8 `/api/v1/depth/refresh` trigger, persisted MapAnything evidence, ZeroPlane, plane fusion, structural registration, tracking readback, leakage metrics, and render budget together so Menon can render the result without solving geometry per frame or depending on depth-retention snapshots.
- **References:** `noesis/virtual_twin/`, `scripts/build_virtual_twin_reconstruction.py`, `noesis/server/virtual_twin_api.py`, `../Menon/src/features/path-visualization/VirtualTwinSurfaceOverlay.js`

- **Date:** 2026-05-02
- **Author:** Codex
- **Area:** DS8 REST / Menon virtual twin consumption
- **Decision:** Enable DS8 REST CORS by default for localhost and RFC1918 private-network browser origins, while keeping explicit `NOESIS_REST_CORS_ORIGINS`, `NOESIS_REST_CORS_ORIGIN_REGEX`, and `NOESIS_REST_CORS_ALLOW_ALL` operator overrides.
- **Rationale:** Menon is a browser app that normally runs from a different dev or LAN origin than Noesis REST. The virtual twin GLB/JSON endpoints must be consumable by ordinary browsers without disabling web security, and this keeps the allowance scoped to local/private control surfaces by default.
- **References:** `noesis/ds8_runtime.py`, `docs/DS8_api_contracts_rest.md`, `../Menon/src/features/path-visualization/VirtualTwinSurfaceOverlay.js`

- **Date:** 2026-05-02
- **Author:** Codex
- **Area:** Virtual twin reconstruction / Menon render artifact
- **Decision:** Serve `surfaces.glb` as a model-surface triangle mesh in Menon scene units, using MapAnything+ZeroPlane support to select/color structural OBJ surfaces, while retaining dense fused depth only in `points.ply` and `points.npz`.
- **Rationale:** A decimated registered point cloud still looked like a floating warped depth projection and could be mistaken for solved room geometry. Constraining the browser artifact to Menon's structural mesh makes the served virtual-twin layer a revisioned model-surface artifact and leaves raw dense geometry as diagnostics.
- **References:** `noesis/virtual_twin/builder.py`, `noesis/virtual_twin/artifacts.py`, `noesis/virtual_twin/menon_obj.py`, `docs/DS8_api_contracts_rest.md`, `../Menon/src/features/path-visualization/VirtualTwinSurfaceOverlay.js`

- **Date:** 2026-05-03
- **Author:** Codex
- **Area:** Virtual twin reconstruction / RGB surface texture bake
- **Decision:** Bake `surfaces.glb` with an embedded RGB texture atlas generated from the revision's saved pipeline keyframes. Each supported model-surface triangle receives UVs into the atlas; atlas texels are projected through the explicit world-to-Menon registration and accepted only when the corresponding MapAnything depth/confidence gate agrees.
- **Rationale:** Surface support tint proved the right floors and walls were selected, but it did not expose the camera RGB evidence the user expected from the Noesis pipeline. Baking textures in Noesis keeps the browser artifact versioned and explicit while avoiding a return to Menon's browser-side dense projection solver.
- **References:** `noesis/virtual_twin/builder.py`, `noesis/virtual_twin/artifacts.py`, `scripts/build_virtual_twin_reconstruction.py`, `docs/DS8_api_contracts_rest.md`

- **Date:** 2026-05-03
- **Author:** Codex
- **Area:** Virtual twin reconstruction / texture readability
- **Decision:** Apply an explicit exposure/gamma/contrast tone map to filled virtual-twin atlas texels, and persist the tone-map values in metrics.
- **Rationale:** Living-room keyframes are grayscale and underexposed, so the first RGB-textured surface artifact was accurate but visually dull. Tone mapping keeps the artifact grounded in the pipeline image evidence while making it readable in Menon.
- **References:** `noesis/virtual_twin/builder.py`, `scripts/build_virtual_twin_reconstruction.py`, `docs/DS8_api_contracts_rest.md`

- **Date:** 2026-05-03
- **Author:** Codex
- **Area:** Virtual twin reconstruction / RGB evidence gate
- **Decision:** Reject RGB-texture virtual-twin builds when captured keyframes are effectively grayscale, unless the operator explicitly passes `--allow-grayscale-texture`.
- **Rationale:** The living-room source can be in IR/night grayscale mode while still arriving as three-channel frames. Failing closed prevents a grayscale diagnostic artifact from being mistaken for an RGB reconstruction.
- **References:** `noesis/virtual_twin/builder.py`, `scripts/build_virtual_twin_reconstruction.py`, `docs/DS8_api_contracts_rest.md`

- **Date:** 2026-05-03
- **Author:** Codex
- **Area:** Virtual twin reconstruction / surface texture alignment
- **Decision:** Refine each revision's initial plane-to-OBJ registration with a dense point-to-structural-surface yaw/scale/translation solve, then require every RGB atlas texel to be colored only by source pixels that back-project onto the same Menon structural surface.
- **Rationale:** The first RGB surface bundle selected the right floor/wall surfaces, but the texture behaved like a broad projector when the global transform was slightly rotated, oversized, or shifted. A dense surface-support correction tightens the explicit artifact transform, and per-surface pixel gating prevents RGB from one wall/floor block from bleeding onto a different model object while keeping Menon as a renderer rather than a geometry solver.
- **References:** `noesis/virtual_twin/builder.py`, `docs/Virtual_Twin_Reconstruction.md`, `docs/DS8_api_contracts_rest.md`

- **Date:** 2026-05-03
- **Author:** Codex
- **Area:** Virtual twin reconstruction / front-most surface assignment
- **Decision:** Add a camera-view structural visibility pass for texture fitting: rasterize Menon's structural OBJ into a front-most visible-surface map, solve image-pixel-to-surface scale/translation with tightly capped yaw steps, and require RGB atlas texels to use only source pixels assigned to the same visible surface.
- **Rationale:** Per-surface depth agreement alone still allowed texture to land on a wrong coplanar or behind-wall surface when the artifact scale was too large. The visibility map makes pixel ownership explicit by structural surface, while the yaw cap preserves the already-good orientation and lets the correction primarily shrink and shift the bundle.
- **References:** `noesis/virtual_twin/builder.py`, `docs/Virtual_Twin_Reconstruction.md`, `docs/DS8_api_contracts_rest.md`

- **Date:** 2026-05-11
- **Author:** Codex
- **Area:** DS8 calibration / snapshot scaling
- **Decision:** Treat the active `config/cameras.yaml` camera model resolution as authoritative for runtime `CalibrationSnapshot` scaling, and consult legacy `config.py` camera specs only when the active camera model has no resolution. The family-room G4 camera now uses a rectified 1920x1080 model derived from the latest raw Charuco fisheye calibration, paired with a matching DS8 dewarper config.
- **Rationale:** Legacy camera specs can describe the raw capture size while DS8 consumers need the post-dewarper/streammux model. Letting the legacy spec override the active model double-scaled the family-room K matrix and made BEV/depth-registration fingerprints disagree with the current camera contract.
- **References:** `config/cameras.yaml`, `config/infer.yaml`, `config/dewarper_g4_instant_charuco_720_to_1080.txt`, `noesis/ds8_runtime.py`

- **Date:** 2026-05-11
- **Author:** Codex
- **Area:** BEV floorplan / calibration cache authority
- **Decision:** Floorplan cache entries include a calibration fingerprint derived from the active K, extrinsics, floor alignment, and frame metadata, and cached floorplans are rejected when that fingerprint differs from the current calibration bundle. The frontend accepts producer-declared metric floorplan units and keeps BEV points in the floorplan frame after backend projection.
- **Rationale:** A visually plausible floorplan can still be wrong after intrinsics or extrinsics change. Failing closed on calibration mismatch prevents stale, pre-calibration floorplans from making tracking overlays look aligned by accident or drift out of view because the raster and tracks were produced under different coordinate contracts.
- **References:** `geometry/depth_source.py`, `oai2-fe/src/App.tsx`, `oai2-fe/src/components/BevView.tsx`, `docs/DS8_api_contracts_ws.md`

- **Date:** 2026-05-11
- **Author:** Codex
- **Area:** BEV floorplan / display-frame contract
- **Decision:** The main inline BEV/floorplan visualization publishes camera-local display coordinates (`frame=camera_local_ground_m`) directly from the backend. Backend-world tracking remains the canonical tracking telemetry frame, while `BevRenderer` converts or image-anchor-projects only for the BEV display payload and declares the selected path with `footpoints[].displaySource`.
- **Rationale:** The floorplan raster is generated from MapAnything depth in direct camera-local X/Z meters, so drawing backend-world points over that raster and relying on browser-side reinterpretation can place otherwise valid tracks outside the floorplan. Keeping display points, trails, and floorplan bounds in one declared frame makes out-of-bounds validation meaningful and prevents empty request-extent margins from hiding coordinate mistakes.
- **References:** `config/infer.yaml`, `noesis/telemetry/bev.py`, `noesis/pipelines/hooks.py`, `geometry/depth_source.py`, `docs/DS8_api_contracts_ws.md`, `docs/Telemetry_Schema.md`

- **Date:** 2026-05-11
- **Author:** Codex
- **Area:** BEV floorplan / surface-coordinate rendering
- **Decision:** Inline BEV overlays resolve displayed track points through the floorplan raster surface, not a rectangular envelope. The backend first projects local BEV points with the same camera-local depth-unprojection basis as MapAnything when fused anchor depth is available (`displaySource=image_depth_anchor`); the frontend then converts metric X/Z to floorplan grid coordinates and only snaps to nearby rendered/walkable cells, leaving truly unmappable points out of the overlay.
- **Rationale:** Floorplan bounds describe the raster extent, not the actual visible room surface. A rectangular containment check can still draw tracks over black/no-map cells or hide coordinate errors. The surface resolver ties the overlay to the cells the floorplan actually shows while preventing far-off coordinates from being silently clamped into the room.
- **References:** `noesis/telemetry/bev.py`, `noesis/pipelines/hooks.py`, `oai2-fe/src/components/BevView.tsx`, `docs/DS8_api_contracts_ws.md`, `docs/Telemetry_Schema.md`

- **Date:** 2026-05-11
- **Author:** Codex
- **Area:** BEV floorplan / registered-depth authority
- **Decision:** BEV floorplan display may use image-anchor depth unprojection only when the object-depth sample has a valid MapAnything registration status (`depth_registration_status=ok`). If registration rejects the depth sample, the displayed BEV point falls back to the common image-floor/world path. Clean walkable/obstacle floorplan layers are generated for every room, and the frontend uses those surface layers as the snap/drop authority without room-specific exceptions while rendering inline panels with the height map for readability.
- **Rationale:** Raw object depth and MapAnything floorplan depth can be in different local range bases. Family-room looked correct because its visible tracks had usable registered depth, while living-room and kitchen were drawing raw rejected ranges into a MapAnything floorplan. Failing closed on unregistered depth and sharing the same clean-layer floorplan contract across rooms prevents one room from succeeding through a special path while others drift outside the visible footprint.
- **References:** `noesis/pipelines/hooks.py`, `geometry/depth_source.py`, `oai2-fe/src/components/BevView.tsx`, `docs/DS8_api_contracts_ws.md`, `docs/Telemetry_Schema.md`

- **Date:** 2026-05-11
- **Author:** Codex
- **Area:** DS8 mosaic stream / trail overlay coordinates
- **Decision:** Mosaic stream trails convert analytics image anchors through the configured tiler tile rectangle when DeepStream frame metadata does not expose a compositor rectangle, and they use the calibration/source image size for that conversion.
- **Rationale:** The video stream overlay is drawn after sources have been tiled into one mosaic. Source-local foot anchors from tracking telemetry must be offset and scaled into the correct tile; otherwise a valid track from kitchen or family-room can be drawn detached from the person, often over the wrong part of the mosaic.
- **References:** `noesis/pipelines/hooks.py`, `tests/test_trail_overlay_floor_plane_anchor.py`

- **Date:** 2026-05-22
- **Author:** Codex
- **Area:** DS8 V3DT / protected experiment lane
- **Decision:** Keep the V3DT reimplementation isolated from the baseline DS8 runtime by using copied runtime/hooks/config artifacts, run the fast SV3DT tracker at `tracker-width=1920` and `tracker-height=1056`, and publish `image_base` from the projected opposite vertical bbox3d endpoint while keeping the public world footpoint on the floor-plane endpoint.
- **Rationale:** The baseline non-V3DT command must remain untouched. Live MP4 diagnostics showed the no-Pose fast tracker is accurate only when its operating resolution is pinned near the calibrated stream size with a 32-aligned height; default tracker sizing was fast but projected outside the room. The native V3DT image-foot point aligns with the center-side endpoint, while the opposite endpoint lands on the visible detector base and gives the frontend a useful image anchor.
- **References:** `noesis/ds8_runtime_v3dt_reimpl.py`, `noesis/pipelines/hooks_v3dt_reimpl.py`, `config/infer_v3dt_reimpl_fast1056_mp4.yaml`, `config/v3dt/reimpl/nvtracker_sv3dt_yolo26s_fast.yml`

- **Date:** 2026-05-11
- **Author:** Codex
- **Area:** BEV floorplan / live MapAnything source authority
- **Decision:** A non-cache-only `get_floorplan` request must drive the canonical DS8 MapAnything branch itself by opening the MapAnything depth valve, waiting for a fresh stored depth snapshot, and then generating the floorplan from that live snapshot. Cache-only requests may still return retained floorplans, but live validation must not silently rely on a retained raster when no fresh MapAnything snapshot exists.
- **Rationale:** Live RTSP tracking can continue while the MapAnything branch is gated off. If floorplan RPC only reads the existing cache, the frontend can compare live tracks against an old floorplan and make coordinate mapping look worse or better by accident. Coupling fresh floorplan generation to the main DS8 depth gate keeps validation on the same pipeline path as the tracks and avoids standalone or stale data sources.
- **References:** `noesis/ds8_runtime.py`, `noesis/pipelines/ds8_pipeline.py`, `websocket_server.py`, `docs/DS8_testing_guide.md`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Noesis/Menon validation / cross-space agreement
- **Decision:** Menon-facing validation traces can include `bev_trails`, `avatars`, and `collision` evidence, and the shared toolbox validates BEV-to-Menon trail agreement, rendered avatar height, collision proxy clearance, and movement-heading agreement as separate checks instead of folding them into one opaque Menon confidence.
- **Rationale:** Noesis and Menon can each look plausible while disagreeing about path shape, rendered person scale, or whether an avatar is passing through scene geometry. Keeping these checks separate preserves the failure taxonomy and points future agents toward transform, semantic, or collision/debug evidence instead of hiding contradictions inside one score.
- **References:** `noesis/validation/menon.py`, `noesis/validation/menon_trace.py`, `plans/noesis_menon_validation/artifact_contracts.md`, `plans/noesis_menon_validation/work_order.md`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Noesis/Menon validation / generated scene geometry
- **Decision:** Generated-scene validation fixtures can declare `scene.coordinate_systems` and `scene.room_geometry_constraints`, and the shared toolbox validates coordinate units, axes, pose convention, handedness, origin stability, scale anchors, wall/opening constraints, and surface continuity as explicit report rows.
- **Rationale:** Scene reconstruction artifacts are easy to make visually plausible while hiding wrong units, mirrored transforms, drifting origins, malformed openings, or broken wall/floor continuity. Encoding these measurements as fixture data gives future agents objective failure categories before those defects propagate into BEV and Menon placement.
- **References:** `noesis/validation/scene.py`, `scripts/noesis_validation_runner.py`, `plans/noesis_menon_validation/artifact_contracts.md`, `plans/noesis_menon_validation/minimal_fixture.json`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Noesis/Menon validation / temporal tracking
- **Decision:** Tracking fixtures and telemetry can carry `occluded` and `occlusion_uncertainty_m`, and the shared toolbox validates occlusion bridges and doorway transitions as explicit temporal checks.
- **Rationale:** A world point can pass frame-level geometry checks while the temporal story remains wrong. Occlusion spans need plausible visible endpoints and widened uncertainty, and room changes should pass through known openings instead of appearing through walls.
- **References:** `noesis/validation/tracking.py`, `noesis/validation/telemetry.py`, `plans/noesis_menon_validation/artifact_contracts.md`, `plans/noesis_menon_validation/minimal_telemetry.ndjson`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Noesis/Menon validation / identity continuity
- **Decision:** Tracking validation treats `stable_id`, `tracker_id`, optional `reid_identity` or `appearance_id`, and `reid_confidence` as separate identity evidence instead of collapsing them into one trust score. The toolbox reports StableID continuity and ReID/geometry agreement as distinct temporal checks.
- **Rationale:** A track can move plausibly while carrying the wrong identity, or ReID can contradict an otherwise continuous geometric path. Separating tracker continuity, StableID assignment, appearance evidence, and confidence lets future agents distinguish projection defects from identity churn, split/merge events, and weak ReID evidence.
- **References:** `noesis/validation/tracking.py`, `noesis/validation/telemetry.py`, `plans/noesis_menon_validation/artifact_contracts.md`, `plans/noesis_menon_validation/work_order.md`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Noesis/Menon validation / camera reprojection
- **Decision:** Menon camera-view validation traces can declare `camera_reprojections` with source-frame, Menon-render, projected-avatar, detected-bbox, room-edge, floor-grid, anchor, pixel-error, and overlap evidence. The shared toolbox treats missing source/render/layer evidence as blocked and numeric misalignment as projection warning or failure.
- **Rationale:** Menon top-down agreement is not enough to prove the room, camera calibration, and live entity placement agree with the original camera image. A selected-camera render compared against the source frame provides an end-to-end gate from Noesis world state back to pixel-space explainability.
- **References:** `noesis/validation/menon.py`, `noesis/validation/menon_trace.py`, `plans/noesis_menon_validation/artifact_contracts.md`, `plans/noesis_menon_validation/minimal_menon_trace.json`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Noesis/Menon validation / depth reconstruction
- **Decision:** Depth validation fixtures can extend `scene.depth_anchors` with relative-order groups, object-depth errors, static-scene variance, edge alignment errors, and confidence/fusion weights, all reported through `SCENE.depth_consistency`.
- **Rationale:** Metric anchor error alone cannot prove monocular depth is useful for scene generation or tracking support. The added fields let the toolbox catch wrong near/far ordering, unstable empty-scene depth, bad depth discontinuities, and low-confidence depth being over-weighted in fusion.
- **References:** `noesis/validation/scene.py`, `scripts/noesis_validation_runner.py`, `plans/noesis_menon_validation/artifact_contracts.md`, `plans/noesis_menon_validation/minimal_fixture.json`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Noesis/Menon validation / BEV path quality
- **Decision:** BEV validation reports include `BEV.path_smoothness`, computed from ordered track X/Z samples using path-length inflation and heading-change thresholds.
- **Rationale:** Speed and acceleration checks catch physically impossible motion, but a projected trail can still be locally plausible while visually zig-zagging in the top-down view. A separate BEV path-shape row points future agents toward smoothing ownership, duplicate transforms, identity switches, or projection jitter without hiding the underlying samples.
- **References:** `noesis/validation/bev.py`, `noesis/validation/telemetry.py`, `scripts/noesis_validation_runner.py`, `plans/noesis_menon_validation/artifact_contracts.md`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Noesis/Menon validation / regression triage
- **Decision:** Registry regression summaries aggregate non-passing check failure types plus expectation/artifact misses into `failure_categories`, `dominant_failure_category`, and `suggested_diagnostic_focus`.
- **Rationale:** A failed validation suite should tell the next agent which class of evidence to inspect first instead of forcing them to scan every check row. Categorizing failures preserves the common taxonomy and makes regression output actionable for calibration, transform, projection, scene, temporal, semantic, sync, model, data quality, infrastructure, and recent-code-change investigations.
- **References:** `scripts/noesis_validation_regression_runner.py`, `plans/noesis_menon_validation/artifact_contracts.md`, `plans/noesis_menon_validation/work_order.md`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Noesis/Menon validation / semantic scene constraints
- **Decision:** Scene fixtures can extend `scene.semantic_objects` with `allowed_rooms`, `doorway_clearance_m`, `free_space_clearance_m`, `walkable_area_blocked_ratio`, and `known_anchor_error_m`, and the shared toolbox reports those as separate semantic counters instead of collapsing them into a generic object-plausibility result.
- **Rationale:** A generated room can have plausible object boxes while still blocking an opening, assigning furniture to the wrong room, eroding walkable space, or drifting from known static anchors. Keeping these counters separate makes scene-generation failures actionable before they contaminate BEV containment or Menon collision evidence.
- **References:** `noesis/validation/scene.py`, `scripts/noesis_validation_runner.py`, `plans/noesis_menon_validation/artifact_contracts.md`, `plans/noesis_menon_validation/minimal_fixture.json`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Noesis/Menon validation / detection reprojection evidence
- **Decision:** Detection projection fixtures can carry source image `image_bbox_xyxy` plus reprojected `projected_bbox_xyxy`, `projected_bbox_iou`, and `bbox_center_error_px` evidence. `TRACK.detection_world_projection` validates image-space person aspect and 3D bbox reprojection overlap, and projection confidence uses those components when present.
- **Rationale:** A detection can map to a plausible floor point while the projected person volume misses the camera image or has an impossible bbox shape. Keeping source bbox and projected 3D bbox evidence in the same sample catches calibration, footpoint, and transform errors before they appear as Menon or BEV drift.
- **References:** `noesis/validation/tracking.py`, `scripts/noesis_validation_runner.py`, `plans/noesis_menon_validation/artifact_contracts.md`, `plans/noesis_menon_validation/minimal_fixture.json`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Noesis/Menon validation / Menon asset sanity
- **Decision:** Scene fixtures can include `scene.menon_assets` rows for room mesh, floor mesh, wall mesh, camera markers, collision parity, and debug overlay layer evidence, all reported through `SCENE.menon_assets`.
- **Rationale:** Menon placement can pass world-to-scene transform checks while the rendered assets are scaled, shifted, non-walkable, blocking a doorway, missing a camera marker, or unable to display the diagnostic overlays needed for review. Validating those asset properties in the shared fixture report keeps scene asset regressions separate from live tracking or calibration failures.
- **References:** `noesis/validation/scene.py`, `scripts/noesis_validation_runner.py`, `plans/noesis_menon_validation/artifact_contracts.md`, `plans/noesis_menon_validation/minimal_fixture.json`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Noesis/Menon validation / mesh quality
- **Decision:** Mesh quality fixtures can include LOD preservation, texture atlas alignment/stretch, BVH validity, and collision/BVH parity measurements in `scene.meshes`, reported through `SCENE.mesh_quality`.
- **Rationale:** A GLB/OBJ/PLY room shell can satisfy basic topology while decimation changes the geometry, textures drift away from the mesh, or raycast/collision BVHs disagree with the visible surface. Keeping these checks in the mesh-quality row gives asset regressions a scene failure category before they surface as Menon placement or camera reprojection defects.
- **References:** `noesis/validation/scene.py`, `scripts/noesis_validation_runner.py`, `plans/noesis_menon_validation/artifact_contracts.md`, `plans/noesis_menon_validation/minimal_fixture.json`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Noesis/Menon validation / projection confidence components
- **Decision:** Projection samples can declare explicit `reprojection_score`, `temporal_smoothness_score`, and `semantic_validity_score` components, and `TRACK.projection_confidence` reports component coverage counts alongside the aggregate score.
- **Rationale:** Projection confidence needs to show which evidence contributed to the trust decision. Keeping reprojection, temporal, and semantic components visible prevents a good footpoint or floor contact score from masking jitter, semantic contradictions, or weak camera reprojection evidence.
- **References:** `noesis/validation/tracking.py`, `scripts/noesis_validation_runner.py`, `plans/noesis_menon_validation/artifact_contracts.md`, `plans/noesis_menon_validation/minimal_fixture.json`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Noesis/Menon validation / world-BEV agreement
- **Decision:** Fixture and Menon trace reports can include `world_to_bev_col_major` plus `world_bev_points`; the shared toolbox emits `BEV.world_round_trip` to validate world-to-BEV agreement and BEV-to-world round-trip error.
- **Rationale:** BEV and Menon trails can appear self-consistent while BEV is quietly using a shifted or rescaled world frame. An explicit world/BEV transform check catches scale, origin, and double-conversion errors before path agreement is interpreted as true cross-space validation.
- **References:** `noesis/validation/menon.py`, `noesis/validation/menon_trace.py`, `scripts/noesis_validation_runner.py`, `plans/noesis_menon_validation/artifact_contracts.md`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Noesis/Menon validation / room opening geometry
- **Decision:** Room-geometry fixtures distinguish window cutout evidence from doorway/opening evidence with `window_width_m`, `window_height_m`, and `window_sill_height_m`, reported as `impossible_window_count` under `SCENE.room_geometry_constraints`.
- **Rationale:** Doorways and windows fail for different reasons and point agents at different diagnostics. Separating window width, height, and sill plausibility keeps generated-scene checks from hiding bad window segmentation inside generic opening counters.
- **References:** `noesis/validation/scene.py`, `scripts/noesis_validation_runner.py`, `plans/noesis_menon_validation/artifact_contracts.md`, `plans/noesis_menon_validation/minimal_fixture.json`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Noesis/Menon validation / camera reprojection overlays
- **Decision:** Fixture camera reprojection overlays can draw detected image bboxes, projected bboxes/avatar extents, and mask polygons in addition to floor grids, anchors, room outlines, mesh edges, and footpoints.
- **Rationale:** Numeric reprojection checks are easier to debug when the source-frame evidence is visible in one artifact. Showing detected and projected extents together helps future agents distinguish footpoint selection errors from camera calibration, avatar scale, segmentation, or transform-chain errors.
- **References:** `noesis/validation/visuals.py`, `scripts/noesis_validation_runner.py`, `plans/noesis_menon_validation/artifact_contracts.md`, `plans/noesis_menon_validation/minimal_fixture.json`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Noesis/Menon validation / static object placement
- **Decision:** Menon trace and fixture reports can include `objects` evidence, and the shared toolbox emits `MENON.object_placement` to validate static/generated object world-to-scene placement, rendered dimensions, support contact, room compatibility, and wall/furniture collision proxies.
- **Rationale:** Avatar placement alone does not prove the generated Menon scene is semantically usable. Furniture and other static objects can be shifted, scaled, floating, assigned to the wrong room, or intersecting walls while live tracks still look plausible. A separate object-placement row preserves that failure class and keeps scene asset errors distinct from tracking errors.
- **References:** `noesis/validation/menon.py`, `noesis/validation/menon_trace.py`, `scripts/noesis_validation_runner.py`, `plans/noesis_menon_validation/artifact_contracts.md`

- **Date:** 2026-05-27
- **Author:** Codex
- **Area:** Noesis/Menon validation / latency alignment
- **Decision:** Menon trace and fixture reports can include `latency_samples`, and the shared toolbox emits `MENON.latency_alignment` to compare Noesis frame timestamps with telemetry, Menon update, Menon render, and Menon display timestamps.
- **Rationale:** A spatial placement can be correct but stale. Separating total display lag and update/render/display queue lag gives future agents sync evidence without confusing timestamp defects with projection or transform defects.
- **References:** `noesis/validation/menon.py`, `noesis/validation/menon_trace.py`, `scripts/noesis_validation_runner.py`, `plans/noesis_menon_validation/artifact_contracts.md`

- **Date:** 2026-06-06
- **Author:** Codex
- **Area:** Virtual twin / MapAnything plane reconstruction
- **Decision:** The virtual-twin builder computes camera-space normals from DS8 MapAnything depth snapshots and calibration, persists them in revision-local `mapanything/*.npz` evidence, and exports per-plane `normal_support`, `depth_support`, and `fusion_score` in `planes.json` schema v2. Menon may use those fields for generated-plane scoring and validation, but the final floorplan geometry remains the Noesis stream-derived plane equation.
- **Rationale:** The current generated-plane floorplan benefits from dense normal evidence to suppress skewed or misplaced fragments, but using Menon model geometry as the final output would defeat the reconstruction goal. Keeping normals as an additive evidence layer improves fidelity while preserving the camera/MapAnything/ZeroPlane evidence chain and older artifact compatibility.
- **References:** `noesis/virtual_twin/geometry.py`, `noesis/virtual_twin/builder.py`, `docs/Virtual_Twin_Reconstruction.md`, `tests/test_mapanything_normals_fusion.py`

- **Date:** 2026-06-15
- **Author:** Codex
- **Area:** DS9 migration / MapAnything TensorRT build safety
- **Decision:** DS9 MapAnything TensorRT plans are built through `DS9/scripts/build_mapanything_guarded.sh`, which validates ONNX external tensor sidecars, constrains TensorRT build pressure (`builderOptimizationLevel=0`, `maxAuxStreams=0`, `workspace:1024`), and stops the Docker build if GPU memory exceeds the configured guard.
- **Rationale:** An unguarded TensorRT build of the full MapAnything export can put enough pressure on a 12 GB GPU to destabilize the workstation. The guarded build keeps DS9 validation reproducible without falling back to DS8 engines or alternate runtime paths.
- **References:** `DS9/scripts/build_mapanything_guarded.sh`, `DS9/scripts/rebuild_engines.py`, `utils/onnx2trt/export_ma_onnx/export_to_onnx.py`, `DS9/README.md`

- **Date:** 2026-06-20
- **Author:** Codex
- **Area:** DS8 detector PGIE / YOLO26 dynamic batching
- **Decision:** The DS8 YOLO26 detector profile uses explicit dynamic-batch-safe assets for every supported size (`n/s/m/l/x`): `models/yolo26{size}_dynamic_b1-3.onnx` and `models/engines/yolo26{size}_dynamic_b1-3_fp16.engine`. The runtime fails fast if those assets are missing instead of selecting the newest matching static engine, and detector preprocess configs materialize active source IDs from `config/infer.yaml`.
- **Rationale:** Live RTSP batching can produce partial batches during startup or jitter. Static-batch YOLO26 TensorRT engines can deserialize but fail at enqueue time on partial batches, cascading into misleading stream/preprocess errors. Naming and requiring dynamic assets keeps the runtime deterministic and makes missing assets an explicit operator error.
- **References:** `noesis/ds8_runtime.py`, `scripts/download_export_build_yolo26_detect.py`, `config/infer.yaml`

- **Date:** 2026-06-21
- **Author:** Codex
- **Area:** DS8 detector PGIE / DEIMv2 Wholebody49 promotion
- **Decision:** The DS8 runtime exposes the Wholebody49 prototype as an optional `--pgie-profile wholebody49` profile with promoted sizes `s` and `x`. Size `s` uses the validated DINOv3-S instance-mask engine, while size `x` uses the existing DINOv3-X label-only FP16 engine with a bbox parser. The X mask ONNX is not promoted until a matching engine is built and runtime-validated.
- **Rationale:** The main DS8 PGIE slot must produce DeepStream object metadata for tracker, analytics, ReID, pose, and telemetry. The DINOv3-S mask variant already does that through an instance-mask parser; the DINOv3-X label-only variant needs a bbox parser because the prototype overlay consumed raw tensors directly. Promoting only copied, named assets keeps the prototype isolated and avoids hidden fallbacks to testpipeline paths.
- **References:** `noesis/ds8_runtime.py`, `noesis/deimv2_wholebody49_assets.py`, `pipelines/nvdsinfer_deimv2_wholebody49/`, `testpipelines/deimv2-wholebody49/README.md`

- **Date:** 2026-06-28
- **Author:** Codex
- **Area:** DS8 YOLO26 runtime / performance gating
- **Decision:** YOLO26 runtime optimization keeps detector/tracker cadence intact while gating or coalescing downstream work: DAv2 depth tracking defaults to a lower cadence with cached object-depth fusion, pose SGIE uses a short reinfer interval plus per-track payload reuse, WebRTC gateway slots are demand-created, RTSP mosaic output remains always-open by default with explicit `NOESIS_MOSAIC_RTSP_DEMAND_GATED=1` owner-driven gating only for experiments, and tracking/BEV websocket payloads are latest-only coalesced.
- **Rationale:** The review found large avoidable GPU and CPU costs outside the primary detector: depth, pose, idle media fanout, JSON broadcast churn, per-frame debug I/O, and host ROI copies. RTSP output stays open by default because gating `nvrtspoutsinkbin` at startup can make the local RTSP server return 503 and leave warm WebRTC gateways frame-starved. These controls reduce load without adding alternate DS8 paths or lowering PGIE inference cadence, and each knob is explicit through DS8 runtime configuration or environment variables.
- **References:** `noesis/ds8_runtime.py`, `noesis/pipelines/ds8_pipeline.py`, `noesis/pipelines/hooks.py`, `websocket_server.py`, `noesis/mosaic_webrtc_gateway.py`, `plans/DS8/ds8_yolo26_performance_optimization_plan.md`

- **Date:** 2026-06-30
- **Author:** Codex
- **Area:** DS8 YOLO26 runtime / detection-wake pose secondary load
- **Decision:** For the canonical YOLO26 live profile, pose feature attachment first reuses cached per-track pose payloads before native tensor extraction, and the pose SGIE uses the existing batch-3 YOLO26n pose TensorRT engine instead of the batch-16 engine.
- **Rationale:** Live profiling showed that pose tensor extraction pressure could be greatly reduced by honoring DeepStream's cached secondary-inference payloads, but GPU load during detections remained high when the one-to-three-person home scene used a batch-16 pose SGIE. The batch-3 engine keeps detector/tracker cadence and pose functionality intact while matching the common active-person count more closely; the cache-first feature path keeps Python/native tensor extraction proportional to actual secondary refreshes rather than every metadata frame.
- **References:** `noesis/pipelines/hooks.py`, `config/infer.yaml`, `pipelines/config_infer_secondary_yolo26_pose.ini`, `tests/test_analytics_telemetry_hook.py`, `plans/DS8/ds8_yolo26_performance_optimization_plan.md`

- **Date:** 2026-07-02
- **Author:** Codex
- **Area:** BEV floorplan / trail source stability
- **Decision:** Camera-local BEV resets per-camera smoother and trail history whenever the active floorplan coordinate-space signature changes, and keeps valid registered-depth anchors as the stable active-floorplan display source. Live fused-world candidates are computed early and exposed in alignment debug, but they do not override an in-bounds registered-depth display point.
- **Rationale:** MP4 diagnostics showed the worst visible trail jump came from splicing pre-floorplan samples into the active floorplan trail, not from a lack of cosmetic trail smoothing. A trial policy that globally preferred live fused-world points over registered depth introduced world/registered source flips and worse trail p95, so the retained fix clears stale coordinate-space history while preserving the source that remains stable once the active floorplan is ready.
- **References:** `noesis/telemetry/bev.py`, `scripts/bev_alignment_diagnostics.py`, `tests/test_bev_renderer_world_smoothing.py`, `docs/DS8_api_contracts_ws.md`, `diagnostics/bev_alignment/source_consistency_living_resetonly_20260702_003422/summary.json`

- **Date:** 2026-07-03
- **Author:** Codex
- **Area:** DS8 tracking / ReID overhaul (occlusions + cross-camera + long-term memory)
- **Decision:** The baseline NvDCF tracker (`config/nvtracker.yaml`) now uses the DS8 accuracy profile: cascaded data association, ReID-based target re-association (`reidType 2`, TAO ReIdentificationNet ResNet50 inside nvtracker), `maxShadowTrackingAge 300`, and a full-resolution (1920x1080) tracker surface. The cross-camera StableID SGIE was upgraded from OSNet-IBN MSMT17 to the NVIDIA TAO ReIdentificationNet Transformer (Swin-Tiny, 256-dim, `fc_pred` output) with a dynamic-batch (1..16) FP16 TensorRT engine and TAO ImageNet preprocessing in the nvinfer config. `StableIDManager` gained (a) exemplar-max similarity (match against all stored gallery exemplars plus the EMA centroid), (b) diversity-preserving gallery retention (a full gallery replaces the most-similar exemplar instead of evicting the oldest), and (c) persistent long-term identity memory (`~/.noesis/reid_gallery.npz` by default, autosaved and restored across restarts with an age cap). Runtime env knobs: `NOESIS_REID_COS_SIM_THRESHOLD`, `NOESIS_REID_COS_SIM_HIGH_THRESHOLD`, `NOESIS_REID_GHOST_MAX_AGE_S`, `NOESIS_REID_GALLERY_SIZE`, `NOESIS_REID_GALLERY_PERSIST`, `NOESIS_REID_GALLERY_FILE`, `NOESIS_REID_GALLERY_MAX_AGE_S`, `NOESIS_REID_GALLERY_AUTOSAVE_S`; `NOESIS_REID_TOTAL_ID_REUSE_MIN_AGE_S` default raised 60→900 s. The `models.reid.layer` pipeline key selects the SGIE tensor-meta output layer for hooks.
- **Rationale:** The previous baseline tracker had tracker-internal ReID disabled, greedy matching, and no re-association, so occlusions fragmented per-camera track IDs before StableID could help. The Swin-Tiny TAO model is a human-centric foundation model (~3M crops) with substantially better cross-domain ReID than 2019-era OSNet, and its 15 ms FP16 batch-16 cost on the RTX 3060 fits the existing `secondary-reinfer-interval 6` cadence. Centroid-only EMA matching forgot old appearances; exemplar-max plus diverse retention plus on-disk persistence is what makes re-identification work after long absences and across restarts. Tracker-internal ReID (ResNet50, per-camera re-assoc) and SGIE ReID (Swin, cross-camera identity) are intentionally separate models with separate jobs.
- **References:** `config/nvtracker.yaml`, `pipelines/config_infer_secondary_reid_swin.ini`, `config/infer.yaml`, `reid/stable_id_manager.py`, `noesis/ds8_runtime.py`, `noesis/pipelines/hooks.py`, `models/REID_ENGINE_REBUILD_COMMANDS.txt`, `tests/test_stable_id_manager_gallery.py`

- **Date:** 2026-07-03
- **Author:** Codex
- **Area:** DS8 cross-camera StableID handoff + tracking pipeline performance
- **Decision:** Cross-camera gallery matching in `StableIDManager` no longer applies bbox-scale, brightness, or spatial penalties when the candidate identity's reference appearance came from a different camera (`sid_last_bbox_sensor` tracks the source sensor); those penalties remain for same-camera matches. The cross-camera handoff relaxation widened from 4 s/0.02 to 12 s/0.06 (env: `NOESIS_REID_XCAM_WINDOW_S`, `NOESIS_REID_XCAM_MARGIN`) and the default gallery high threshold dropped 0.72→0.70 (`NOESIS_REID_COS_SIM_HIGH_THRESHOLD`). For GPU headroom (visible tearing under motion): baseline NvDCF drops HOG features (ColorNames-only), tracker-ReID extraction interval 8→16 frames, the Swin ReID SGIE reinfer interval 6→12 frames, and the tracker runs at the plugin-default 960x544 surface instead of 1080p; the pose SGIE stays on the batch-3 engine per the 2026-06-30 decision.
- **Rationale:** Live use showed same-camera re-identification working but different stable IDs across cameras. The dominant cause was the adaptive scale penalty: person bbox area routinely differs 2-16x between rooms, subtracting 0.08-1.2 from cosine similarity and pushing legitimate handoffs below threshold — cross-camera geometry differences are expected and are not identity evidence. The relaxed handoff window matches real between-room walking times. The perf trims each roughly halve a per-frame GPU cost without touching detector cadence or identity quality gates: StableID consumes ~1 embedding/s per track, so a 12-frame SGIE reinfer interval still oversupplies embeddings, and re-association needs only periodic ReID features.
- **References:** `reid/stable_id_manager.py`, `noesis/ds8_runtime.py`, `config/nvtracker.yaml`, `pipelines/config_infer_secondary_reid_swin.ini`, `config/infer.yaml`, `tests/test_stable_id_manager_gallery.py`

- **Date:** 2026-07-03
- **Author:** Codex
- **Area:** DS8 StableID convergence for weak-imaging cameras (camera-agnostic)
- **Decision:** Two global (per-deployment-consistent, not per-camera) StableID mechanisms: (1) the early-reconcile window for fresh low-support tracks widened from 2.5 s/2 attempts/support<=2 to 6 s/4 attempts/support<=4 (env: `NOESIS_REID_EARLY_RECONCILE_WINDOW_S`, `NOESIS_REID_EARLY_RECONCILE_MAX_ATTEMPTS`, `NOESIS_REID_EARLY_RECONCILE_MAX_SUPPORT`), so a track that minted a fresh SID from a poor first crop keeps re-testing ghost/gallery matches while better embeddings arrive and remaps to the person's real SID; (2) duplicate-identity detection (`suggest_aliases`, used by auto-merge) now scores SID pairs as the max of centroid similarity and a robust top-3 mean of cross-exemplar similarities, so same-person IDs split across different camera viewpoints can still consolidate. A per-sensor threshold-offset option was prototyped and deliberately removed at the user's request to keep behavior identical across cameras/homes.
- **Rationale:** Live testing showed kitchen/living-room handoffs working but the family-room camera (720p dewarped/upscaled, soft crops) minting separate IDs. The first clean embedding there often arrives seconds after track creation, outside the old reconcile window; centroid-only merge scoring also blurred multi-view identities below the merge bar precisely when views differ, which is the cross-camera duplicate case. Both fixes are appearance-quality-driven and deployment-portable rather than camera-specific tuning.
- **References:** `reid/stable_id_manager.py`, `noesis/ds8_runtime.py`, `tests/test_stable_id_manager_gallery.py`

- **Date:** 2026-07-07
- **Author:** Codex
- **Area:** DS8 runtime / CPU utilization / tracking telemetry
- **Decision:** Cap CPU math-library worker pools before NumPy/OpenCV/Torch-backed imports in the canonical DS8 runtime and hooks (`NOESIS_CPU_MATH_THREADS`, default `1`, preserving explicit pool env vars), and trim the `tracking_telemetry_stage` Python hot path by lazy-materializing diagnostics-only track copies and skipping recursive NumPy conversion for JSON-native tracking payloads.
- **Rationale:** Live YOLO26-L profiling showed the largest CPU spike came from uncapped BLAS/OpenMP-style worker pools, not from a DeepStream copy path. Import-time caps remove that runaway host scheduling without changing detector/tracker cadence, adding appsinks, or introducing CPU video branches. The telemetry changes remove per-object allocations and recursive payload walks that do not affect published contracts, while retaining conversion for true NumPy payloads and preserving diagnostics when explicitly enabled.
- **References:** `noesis/ds8_runtime.py`, `noesis/pipelines/hooks.py`, `noesis/telemetry/publishers.py`, `tests/test_zero_copy_invariants.py`, `plans/DS8/ds8_yolo26_performance_optimization_plan.md`

- **Date:** 2026-07-08
- **Author:** Codex
- **Area:** DS8 object-depth fusion / zero-copy CPU reduction
- **Decision:** The canonical object-depth fusion path now treats host ROI depth copies as an explicit debug-only path (`NOESIS_OBJECT_DEPTH_ALLOW_HOST_ROI_COPY=1`). Production object-depth relies on native scalar/stat samplers; when those are unavailable it attaches `native_stats_unavailable` instead of copying a depth crop to NumPy. Object-depth metadata attachment caches compact JSON strings, and masked object-depth passes pre-thresholded `uint8` masks to the native extension. The combined native person-mask stats call now also returns `foot_u`/`foot_v`, so the common path no longer rescans masks in Python to compute the person foot anchor.
- **Rationale:** This keeps the always-on DAv2 tracking lane aligned with the zero-copy goal: full-frame depth stays device-owned, object-depth uses compact native stats, and the remaining host-resident mask metadata is represented as a byte mask rather than a float mask. Failing closed when native stats are missing is preferable to silently reintroducing per-object depth ROI D2H copies.
- **References:** `noesis/pipelines/hooks.py`, `native/noesis_depth_tracking_tensor_ext.cpp`, `native/noesis_depth_tracking_tensor_kernels.cu`, `tests/test_depth_tracking_frame_processor.py`

- **Date:** 2026-07-09
- **Author:** Codex
- **Area:** DS8/DS9 product contracts and world ownership
- **Decision:** Runtime-neutral product truth lives in the regular `noesis_core` package. DS8 and DS9 normalize SDK-owned metadata into immutable, versioned observation, identity, world, scene, health, and action contracts. The backend owns global entity fusion; compatible observations are uncertainty-weighted, while contradictory observations remain explicit conflicts rather than being averaged.
- **Rationale:** Copied runtime trees and browser-side fusion made behavior drift and hid source disagreement. A shared product core gives both SDK generations the same semantics without pretending their graph, metadata, ABI, asset, or lifecycle code is interchangeable.
- **References:** `noesis_core/contracts/`, `noesis_core/world/fusion.py`, `plans/spatial_os/implementation_plan.md`, `tests/test_noesis_core_contracts.py`, `tests/test_noesis_core_world_fusion.py`

- **Date:** 2026-07-09
- **Author:** Codex
- **Area:** DS8 MapAnything / calibrated dewarper validity
- **Decision:** Canonical and V3DT MapAnything processors derive their valid output pixels from the declared nvdewarper calibration and explicit source geometry. Invalid pixels are represented as NaN depth and zero confidence; configured invalid calibration fails visibly instead of substituting a full-frame-valid mask.
- **Rationale:** Dewarped output rectangles include pixels without valid source support. Treating those pixels as real depth corrupts registration and downstream world geometry, while silently filling them masks the calibration defect.
- **References:** `geometry/dewarper_validity.py`, `config/infer.yaml`, `noesis/pipelines/hooks.py`, `noesis/pipelines/hooks_v3dt_reimpl.py`, `tests/test_dewarper_validity_mask.py`

- **Date:** 2026-07-10
- **Author:** Codex
- **Area:** DS8/DS9 contract compatibility and replay evidence
- **Decision:** Public product payloads require an explicit contract name/version and coordinate frame/units. JSON Schema and TypeScript are generated from the runtime-neutral Pydantic source, and deterministic characterization evidence uses owner-only, contract-validated, SHA-256-chained `noesis.replay` archives.
- **Rationale:** Filling missing versions or coordinate semantics with defaults makes incompatible producers appear healthy. One generated contract source and tamper-evident replay let DS8, DS9, Menon, and offline validation compare the same evidence without retaining SDK metadata wrappers.
- **References:** `noesis_core/contracts/`, `noesis_core/replay.py`, `scripts/export_noesis_core_schemas.py`, `contracts/`, `tests/test_noesis_core_contracts.py`, `tests/test_noesis_core_replay.py`

- **Date:** 2026-07-10
- **Author:** Codex
- **Area:** Household identity v2 open-set assignment
- **Decision:** Identity-v2 applies quality, absolute appearance, hard-constraint, calibrated-confidence, and pre-prior ambiguity gates before a bounded resident assignment prior. Each simultaneous tracklet has its own unknown option; assignments are joint and one-to-one unless an identity-specific overlap permit explicitly authorizes sharing.
- **Rationale:** Residents are the common case in this home, so a small prior improves close admissible ranking, but visitors must never be forced onto a known name. Joint deterministic assignment prevents iteration order and greedy identity theft from deciding similar-person outcomes.
- **References:** `reid/identity_v2/`, `tests/test_identity_v2_scoring.py`, `tests/test_identity_v2_resolver.py`, `plans/household_identity/decisions.md`

- **Date:** 2026-07-10
- **Author:** Codex
- **Area:** Virtual twin / coherent scene release state
- **Decision:** Scene release content is immutable and content-addressed. Current scene selection is a separate atomic compare-and-swap pointer with append-only promotion/rollback history; camera artifacts use normalized relative paths under a configured root.
- **Rationale:** Mutating candidate records into promoted records breaks reproducibility, while choosing independently newest camera revisions creates mixed calibration/model cohorts. Immutable content plus one promotion pointer gives Menon an auditable coherent scene and safe rollback.
- **References:** `noesis_core/contracts/scene.py`, `noesis_core/scene_store.py`, `tests/test_noesis_core_scene_store.py`, `plans/spatial_os/implementation_plan.md`

- **Date:** 2026-07-10
- **Author:** Codex
- **Area:** Household identity-v2 runtime ownership and authority gate
- **Decision:** DS8, V3DT, and DS9 use one shared process-owned identity-v2
  service. Hooks detach the complete source-frame primitive batch and resolve it
  once. Shadow is the default; authoritative mode requires an artifact-backed
  scoring calibration. The runtime hashes actual active ReID engine bytes and
  requires explicit output layer/dimension and exact camera topology.
- **Rationale:** Whole-frame assignment is the only place one-to-one and exact
  overlap constraints can be applied without object iteration order deciding
  identity. Explicit model/topology provenance and a calibrated authority gate
  prevent a mode toggle from silently making guessed scores public truth.
- **References:** `noesis/identity_v2_service.py`, `reid/identity_v2/`,
  `tests/test_identity_v2_service.py`, `tests/test_identity_v2_runtime_integration.py`

- **Date:** 2026-07-10
- **Author:** Codex
- **Area:** Identity-v2 enrollment and overlap evidence
- **Decision:** Enrollment consumes one exact, bounded, server-produced
  observation key; request bodies cannot supply embeddings. Dual-camera sharing
  requires topology, contemporaneous world distance, and appearance evidence on
  every permit and clears camera-local proof when a track disappears.
- **Rationale:** Client biometrics and stale SID selectors are forgeable. A
  topology edge is only possibility, not proof that two detections are one body.
- **References:** `noesis/server/reid_v2_api.py`,
  `noesis/identity_v2_service.py`, `tests/test_reid_v2_api.py`

- **Date:** 2026-07-10
- **Author:** Codex
- **Area:** Integrations / MQTT and Influx credentials
- **Decision:** Active source/configuration contains no MQTT password or Influx
  token defaults. The currently unwired integration sinks default off. A future
  explicit depth-diagnostics activation reads each credential only at publisher
  construction from an existing owner-owned `0400`/`0600` regular file, rejects
  symlinks/hardlinks/ambiguous content, and treats missing credentials,
  dependencies, or client initialization as fatal. Environment support is
  limited to `*_FILE` path overrides; plaintext secret variables are rejected.
  Multi-sink initialization is transactional and the reader never chmods a
  configured file or parent directory.
- **Rationale:** The repository embedded live-form credential defaults even
  though DS8, V3DT, and DS9 bind no occupancy publisher and the optional depth
  publisher has no active constructor call. Loading or silently disabling sinks
  at config import would expand exposure and mask operator intent. File-only,
  fail-closed activation preserves truthful dormant behavior while establishing
  a secure boundary for future wiring. The deployed Influx operator credential
  was revoked and replaced with separate owner and bucket-scoped writer tokens;
  the deployed MQTT user password was also rotated. Replacement values exist
  only in owner-only secret files and both retired credentials were verified
  unusable without printing them.
- **References:** `config.py`, `geometry/depth_publisher.py`,
  `tests/test_depth_diagnostics_credentials.py`,
  `docs/integrations/occupancy_mqtt_influx.md`

# 2026-07-10 — Canonical world contracts, replay, and exact appliance endpoints

- DS8 and DS9 publish the same strict person observation, global world
  snapshot, and semantic event models from `noesis_core`; SDK metadata remains
  adapter-local and is copied into primitives once.
- Provisional world entities are stable for one `(run, camera, tracker)` rather
  than one frame. Residents key by UUID; visitors include runtime run and
  non-repeating generation. Contradictory simultaneous cameras are retained as
  conflicts and never averaged.
- Effective calibration, model/tracker file content, and runtime configuration
  are fingerprinted in every observation. Canonical observations, snapshots,
  and events are also retained in a bounded owner-only integrity-chained SQLite
  journal.
- REST and WebSocket controls default to loopback. Required endpoints fail on
  collision; port hopping and assuming an unrelated existing listener are
  prohibited. Required RTSP/WebRTC/depth startup failures mark the runtime
  failed instead of continuing in an undeclared degraded mode.

# 2026-07-10 — Coherent scene release ownership

- A scene release selects explicit camera revisions as one immutable cohort.
  Shared calibration/model bundle hashes are separate from each camera's
  individual calibration/model/manifest hashes.
- Promotion verifies the exact manifest bytes, revision/camera identity, every
  declared artifact, cohort timing, authored-home content, and validation
  report, then atomically compare-and-swaps one current release. Menon must not
  reconstruct production truth from independently newest revisions.

# 2026-07-10 — Verifiable journal retention

- The runtime contract journal discards only a contiguous oldest sequence
  prefix and anchors the retained chain to the final discarded hash. Timestamp
  age never deletes a middle record when delayed or clock-adjusted evidence is
  appended out of order.
- An asynchronous writer failure remains a fatal runtime condition, but teardown
  still drains the queue, delivers the stop sentinel, joins the writer, and then
  reports the failure. This preserves both fail-closed behavior and deterministic
  shutdown.
- **2026-07-12 supersession:** The asynchronous writer remains a tested generic
  journal utility, but canonical runtime world authority no longer uses it.
  Tracking/world/event release now requires the synchronous journal's exact
  append acknowledgement; accepting a queued async batch was not proof that the
  visible cohort had been retained.

# 2026-07-10 — Scene releases close over bounded descriptor-verified bytes

- Camera manifests alone are not an immutable scene: every declared camera
  artifact carries a safe relative path, byte length, and SHA-256 in the release
  contract. The authored OBJ, all referenced MTL/texture dependencies, and the
  validation report are materialized under one release-owned directory with the
  same proof. Bundle paths are unique, and a release ID cannot be overwritten
  with different bytes.
- Roots and path components are opened without following links. Every file must
  be non-empty, regular, single-link, bounded, stable across the descriptor read,
  and match its declared length and SHA-256. OBJ/MTL discovery is strict UTF-8
  with bounded files, lines, dependencies, and known map options. A release tree
  is staged off-path and atomically published without replacement; existing
  content is validated and never overwritten or chmod-repaired.
- Promotion, `/current`, and `/current/payload` revalidate the complete cohort.
  Each role-addressed binary route verifies only its selected file against the
  integrity-checked current release row and returns that verified byte snapshot,
  avoiding both path-reopen races and O(N-squared) hashing. Mutated or missing
  render content returns an explicit conflict. DS8, V3DT, and DS9 mount the same
  router and store implementation.

# 2026-07-10 — Atomic world identity transitions

- Canonical fusion explicitly owns each `(camera, source, run, tracker)`
  tracklet. A transition from provisional or unknown to a resident/visitor
  removes the prior entity observation before inserting the resolved subject,
  so one physical track does not appear twice during the world TTL window.
- Open-set rejection remains an explicit `unknown` world subject without a
  display name. It is not mislabeled as still-provisional evidence.

# 2026-07-10 — Identity-v2 authority excludes legacy mutation

- Authoritative DS8, V3DT, and DS9 hooks do not call legacy StableID assignment,
  copresence, or maintenance. Dwell and line-crossing transitions are computed
  only after whole-frame v2 resolution and only for its positive compatibility
  SID.
- A fresh-evidence rejection remains `unknown`; absence of fresh evidence and a
  continuity hold is `provisional`. The initial transient metadata walk stamps
  neutral `#XX`; it never retains wrappers or displays stale legacy identity.
  A separate tiler-sink callback may display the resolved decision only through
  the exact bounded join described below.

# 2026-07-10 — Development auth disablement is loopback-only

- `NOESIS_INTERNAL_AUTH_MODE=disabled` may bind only `localhost` or a literal
  loopback address. Both REST and WebSocket startup reject wildcard, LAN, and
  arbitrary hostname listeners in this mode across DS8, V3DT, and DS9.
- This makes the documented development exception an executable network
  boundary instead of relying on the operator to remember a matching host flag.

# 2026-07-10 — Depth telemetry uses opaque artifact identity

- `DepthResult` retains its real storage reference for internal persistence and
  reconstruction code, but `DepthTelemetryPublisher` emits only
  `noesis-depth://artifact/<sha256>`. Filesystem roots, camera directories, and
  backend storage URIs do not cross the WS boundary in DS8 or DS9.

# 2026-07-10 — Canonical world does not replace local diagnostic BEV geometry

- DS8 and DS9 share the strict `backend_world_m` product observation/snapshot
  contract. The DS8 camera-local BEV renderer may keep its separately declared
  display frame because that path has its own registration and trail-stability
  evidence.
- Runtime parity is measured at the product contract boundary; changing proven
  local display geometry merely to equalize a config key would create risk
  without improving global world truth.

# 2026-07-10 — DS8/DS9 late-viewer media parity

- Both adapters require a successful DESCRIBE for the exact normalized RTSP
  mosaic mount before WebRTC startup and require a keyframe requester. A TCP
  listener alone is not treated as usable media.
- One bounded warm gateway is configurable; additional client slots are created
  on demand up to the declared maximum instead of eagerly starting every
  gateway at runtime boot.

# 2026-07-10 — DS9 MapAnything build intent is not artifact evidence

- The guarded DS9 MapAnything builder now requests true FP16 and a regression
  test prevents the prior BF16/FP16-name mismatch. This source correction does
  not clear the cutover blocker.
- A newly built artifact must carry precision/checksum/build provenance and pass
  tensor, NaN, depth, and floorplan parity. The historically loadable BF16 plan
  with an FP16 filename is explicitly non-qualifying.

# 2026-07-10 — Identity-v2 authority requires held-out calibration evidence

- Superseded by the correlation-aware decision below. Artifact v1 counted
  frame-level outcomes while accepting only three declared independence groups;
  it therefore could not support its own 1% safety claim. Version 1 calibration
  labels, datasets, and authority artifacts now fail closed rather than being
  reinterpreted.
- Resident prevalence may influence assignment only through a bounded prior
  after all evidence gates. It cannot turn an unknown/rejected observation into
  a known resident. Shadow remains the default until real household holdout
  evidence passes.
- Browser enrollment uses short-lived exact server observation keys and a
  two-step proposal/confirmation. Evidence capture status and legacy migration
  review are owner-only and biometric-free; migration has no browser apply
  route and no automatic duplicate resolution.
- Evidence/replay state is owner-only, newline-complete, size/count/age bounded,
  and written through create-private-or-validate-private atomic paths. Existing
  files are never chmodded or followed through links. Evidence contract v2 uses
  a sequence/previous-event hash chain plus a private head/tail/count/byte
  checkpoint; deterministic pruning removes only a contiguous prefix and keeps
  the predecessor reference, so unexplained interior, prefix, or suffix loss is
  rejected before labeling or calibration.

# 2026-07-10 — Identity-v2 authority is two-stratum and person-statistical

- Authoritative scoring accepts only
  `noesis.identity.open_set_calibration` v2. It binds the active engine bytes,
  layer, dimension, runtime-derived crop/preprocessing/tensor semantic digest,
  deterministic evidence-unit policy, a provenance-locked benchmark dataset,
  and a separate household domain-verification dataset. Independent exact-byte
  artifact and active-semantic pins are mandatory.
- The subject-disjoint benchmark holdout carries the generic claim with at
  least 300 challenge-covered resident people and 300 challenge-covered unknown
  people. Each person's result is worst-case across encounters. Exact one-sided
  95% Clopper-Pearson upper bounds for false acceptance and misidentification
  must each be at most 1%; FRR remains at most 35%. Benchmark train needs 50
  resident and 50 unknown people for fitting/policy selection but is not
  presented as the large-N confidence result.
- The local dataset needs 10 known/10 unknown train and 20 known/20 unknown
  holdout encounters. Both household partitions require zero false accepts,
  zero misidentifications, FRR at most 35%, and zero harmful or
  rejection-rescuing prior changes. Household tuning may only keep or raise the
  benchmark rejection bars and cannot change its calibrated mapping or bounded
  resident prior.
- Raw events are deterministically grouped by capture
  session/run/source/camera/tracklet/five-second window, capped at 300 per unit,
  thinned to one center fit representative for at most eight temporally spread
  units per encounter, and encounter-balanced before class balancing. Metrics
  retain all observations, aggregate at encounter worst case, and aggregate
  benchmark confidence again at truth-person worst case. Multiple frames,
  windows, tracklets, or encounters for one person never increase benchmark
  confidence denominators. Session/run/encounter leakage always rejects;
  subject-disjoint separates benchmark person keys, and household unknown
  people are split-disjoint even though residents may repeat across visits.
- Every authority encounter contains hard-allowed, non-empty resident and
  visitor gallery competition plus an impostor; training also requires genuine
  coverage. The scorer-only artifact authorizes no coordinator claim and caps
  runtime gallery shape at the minimum coverage observed across both strata.
  Startup and prospective mutations fail closed outside that envelope.
- This division lets large provenance-reviewed evidence carry the generic
  statistical claim without making homeowner labeling infeasible. The smaller
  home corpus detects installed-camera domain shift but is structurally unable
  to loosen or rescue benchmark rejection behavior. No fixture or unit artifact
  clears the live evidence gate.

# 2026-07-10 — Identity-v2 OSD uses a fresh downstream exact join

- Installed Service Maker APIs prove an explicit tiler-sink
  `BatchMetadataOperator` callback is downstream of whole-frame resolution and
  still upstream of source collapse and nvOSD. DS8, V3DT, and DS9 attach the
  same adapter there.
- The callback walks fresh metadata wrappers exactly once and joins a bounded
  `(camera, frame, tracker)` decision cache. Resident and visitor labels render
  only on an exact match; stale, missing, unknown, provisional, and mismatched
  rows remain `#XX`. SDK wrappers are never retained across callbacks.

# 2026-07-10 — Runtime credentials are owner-only authorities, not config

- Active DS8, V3DT, and DS9 pipeline YAML identifies cameras with stable
  `uri_secret` references. Complete locators are read from one owner-only JSON
  authority, materialized only in process memory, and removed before public
  serialization and before source property maps reach Service Maker.
- The MapAnything RPC key has no source default. It is an authority-grade
  URL-safe value in an owner-only single-link file and is compared for every
  inference request. Missing, weak, shared, symlinked, hardlinked, or inline
  credential state is a startup error; there is no unauthenticated fallback.
- Public config, world fingerprints, model/depth-registration fingerprints,
  generated pipeline YAML, dev-console artifacts, and reconstruction
  provenance depend on stable references rather than secret bytes. Credential
  rotation therefore does not stale spatial/model evidence.
- The deprecated pre-DS8 `config.py` locator list is not changed under the
  active-stack policy. `mapanything_config.py` no longer imports that module,
  so the residual cannot act as DS8/DS9 or MapAnything source authority.

# 2026-07-10 — Validation clients use the production internal-auth boundary

- **Area:** DS8/DS9 validation / internal REST and WebSocket authentication
- **Decision:** Every canonical live smoke client reads one existing owner-only
  token file and sends the bearer only in the HTTP or WebSocket Authorization
  header. Orchestrators pass a token-file path, never token bytes; missing or
  invalid state aborts before a runtime is spawned or a network gate starts.
- **Rationale:** An anonymous validation path cannot prove an authenticated
  appliance contract. A shared read-only client helper keeps HTTP, WebSocket,
  standalone-runtime, and DS9-orchestrated behavior identical while preventing
  URL/query, command-line, report, or environment-value disclosure.

# 2026-07-10 — Person-ground behavior is shared; SDK traversal stays adapted

- Posture classification, support-point selection, source hysteresis, human
  motion filtering, idle locking, and path commitment have one owner in
  `noesis/telemetry/person_ground_state.py`. DS8 and DS9 may differ only in
  transient SDK metadata traversal, projection, and attachment mechanics.
- DS9 extends its telemetry package path to consume that neutral owner and no
  longer carries a private `_WorldAnchorState` or `_PoseAnchorCandidate`
  algorithm. A DS8-vs-DS9 characterization test guards behavioral symmetry.

# 2026-07-10 — Wholebody49 semantics are shared; artifacts remain SDK-owned

- Variant selection, tensor names, parser modes, preprocessing, mask/bbox
  semantics, and generated config policy are one product contract in
  `noesis/deimv2_wholebody49_assets.py`, parameterized by runtime artifact-root
  environment variables.
- Parser binaries, ONNX staging, TensorRT engines, build provenance, and ABI
  checks belong to each SDK adapter. DS9 owns its parser source/build output and
  engine paths and must never copy or load a DS8 engine.
- Source/config/parser parity is not runtime-quality parity. DS9 Wholebody49
  remains blocked until both TensorRT 10.14 engines deserialize and occupied
  scenes prove tensor/parser/mask/OSD quality, throughput, and GPU memory. This
  preserves the earlier decision that successful serialization alone cannot
  rehabilitate a poor-performing variant.

# 2026-07-11 — Wholebody49 parser callbacks are exact and fail closed

- The active DS8 and DS9 custom parsers accept only the exact, unique output
  set for the selected model: `label_xyxy_score` alone for boxes, or
  `label_xyxy_score` plus `masks` for instance masks. Layer order is irrelevant;
  positional, renamed, duplicate, missing, and additional outputs are rejected.
- Callback dimensions are the batch-stripped NvDsInfer shapes `[1240,6]` and
  `[1240,80,80]`, with exact element counts and FLOAT storage. Coordinates are
  finite normalized xyxy values, class IDs are finite integral values, and only
  body class `0` may produce an object. Non-finite consumed mask values reject
  the callback and release all results allocated during that invocation.
- Installed DS8 and pinned DS9 `nvdsinfer` `SplitFullDims` implementations both
  remove dimension zero before populating `NvDsInferLayerInfo::inferDims`.
  Accepting batch-bearing shapes or positional output guesses would therefore
  conceal an SDK/model ABI drift instead of preserving compatibility.
- Parser source remains byte-identical, but each SDK must rebuild and validate
  its own binary. Source parity never authorizes copying a DS8 parser into DS9;
  the DS9 manifest/realization must be rebased after its native rebuild and
  before any Wholebody49 runtime canary.

# 2026-07-10 — ReID semantics are shared; DS9 engines and quality remain gated

- DS8, V3DT, and DS9 use one NVIDIA TAO ReIdentificationNet Transformer
  Swin-Tiny product contract: provenance-locked deployable ONNX, RGB ImageNet
  preprocessing, direct `256x128` resize, dynamic batch 1..16, raw `fc_pred`
  tensor metadata, and a 256-dimensional normalized embedding. There is no
  custom parser and no OSNet fallback.
- The SDK-neutral contract lives in `noesis/reid_swin_profile.py`. Runtime
  adapters own their config paths, ONNX staging, TensorRT engines, manifests,
  and ABI/runtime gates. Hook adapters obtain layer and dimension from the
  effective model configuration and pass both explicitly to the native bridge.
- DS9 source/config/build parity is not runtime identity parity. DS9 remains
  blocked until an exclusive-GPU window produces and separately deserializes a
  TensorRT 10.14 engine with complete provenance, followed by occupied-scene
  resident/visitor open-set, cross-camera handoff, false-accept, continuity,
  throughput, and GPU-memory validation. Engine startup alone cannot clear it.

# 2026-07-10 — Appliance WebSocket health is a separate bounded path

- DS8, V3DT, and DS9 reserve authenticated `/healthz` for one exact
  `noesis.ws.health` v1 frame. This proves the WebSocket upgrade and handler
  path without registering a telemetry client or invoking initial UI snapshot,
  calibration, stats, trail, world, or WebRTC work.
- Startup validates the health frame once. The appliance guard checks core
  capability/HTTPS health every ten seconds and this lightweight WebSocket
  path every sixth cycle, with independent consecutive-failure budgets. A
  normal root-path connection is never used as a health probe.

# 2026-07-10 — DS9 V3DT source, engine, platform, and world gates stay separate

- DS9 owns its V3DT pipeline, camera/camInfo/tracker configuration, native
  bridge, provenance, runtime materialization, smoke, and NvMOT build helper.
  Large sources and engines resolve through an explicit external artifact root
  with residual-capacity and atomic-install gates; DS8 binaries are not reused.
- Source/config parity does not claim engine or runtime parity. Actual builds
  require the installed DeepStream 9 driver floor in addition to an exclusive
  GPU window, then separate deserialization, bbox3d, identity, resource, and
  shutdown evidence. The locked profile remains `camera_local` until a shared
  metric calibration is separately accepted for canonical world fusion.

# 2026-07-10 — DS9-capable driver migration preserves the DS8 host stack

- The host driver moves from APT/DKMS-managed open driver 580.167.08 to exact
  Ubuntu 595.71.05 only after complete source/runtime and offline-package
  checkpoints. Noble's 590 package is transitional, and the NVIDIA runfile is
  not mixed into this package-managed installation.
- Host CUDA remains 13.0 and host TensorRT remains 10.13.3 for DS8. TensorRT
  10.14 is confined to the pinned DS9 build/runtime image, so platform enablement
  does not silently become a host inference-stack migration.
- A coherent 595 module after reboot is necessary but insufficient. Every
  configured DS8 engine must deserialize and DS8 runtime, decoded media,
  identity, MapAnything/floorplan, resource, shutdown, and desktop/RDP gates
  must pass before DS9 receives the exclusive GPU. Any regression triggers the
  exact cached 580 rollback and the same DS8 acceptance suite.
- The rollback rehearsal is a separate evidence state, never shorthand for an
  executed rollback. It rehashes the complete private checkpoint and exact 33
  archive/control contracts, verifies current module/DKMS/boot coherence, and
  runs only a local-archive `apt-get --simulate --no-download` transaction. The
  held 595 cohort is removed explicitly. Generic removals are rejected unless
  an exact cached relationship proves solver necessity; `nvidia-prime` is the
  sole such package because `nvidia-driver-580-open` conflicts/replaces it and
  the accepted 580 baseline records it removed. Execution still requires an
  exclusive maintenance window, initramfs rebuild, reboot, and complete DS8
  engine/runtime/media/world/identity/depth/resource/desktop acceptance.

# 2026-07-10 — DS8/DS9 shutdown requires acknowledged downstream EOS

- **Area:** DS8, V3DT, and DS9 Service Maker lifecycle
- **Decision:** Supersede the 2025-12 default-SIGTERM termination, the
  2026-06-16 immediate-process-exit workaround, and source-classification-only
  shutdown behavior. Each SDK-owned graph contains
  its own repo-built `noesiseos` zero-copy transform immediately after
  `streammux`. A monotonic control request atomically enters terminal buffer-drop
  state, returns from the Service Maker property setter, and has a detached GLib
  worker push standard downstream EOS only after setter locks are released.
  Runtime teardown requires the exact bridge acknowledgement, the expected
  pipeline EOS callback, and `Pipeline.wait()` thread completion before closing
  identity, world, storage, diagnostics, WebSocket, or other callback-owned
  resources. `Pipeline.stop()`, forced termination, and interpreter-exit bypass
  are not canonical shutdown paths. A retained idle stdin pipe prevents
  Service Maker's noninteractive stdin watcher from generating false EOF/GLib
  lifecycle faults. Downstream valves use EOS-preserving drop modes. A
  `Pipeline.wait()` exception is recorded independently from thread completion
  and always invalidates quiescence; a returned thread alone is not success.
- **Rationale:** Installed Service Maker `Pipeline.stop()` only sends EOS at the
  pipeline boundary, while NVIDIA `nvurisrcbin` reconnect handling intentionally
  consumes upstream EOS. Minimal live reproductions showed `stop()` return while
  `wait()` remained alive and sources reconnected; allowing Python teardown in
  that state produced a native `sem_wait` crash. Synchronous event injection
  from a property setter deadlocked, and placing the bridge after asynchronous
  inference produced upstream flow errors. The asynchronous post-streammux
  bridge preserves live-source reconnect during operation, avoids Service Maker
  setter/stream locks, reaches every terminal, and makes native quiescence an
  observable fail-closed contract.
- **Validation:** Repo-owned DS8 and DS9 plugin build/origin/manifest tests,
  exact-ack and finite-source unit tests, graph and runtime shutdown contract
  tests (including fail-closed wait-exception coverage), and the canonical
  authenticated DS8 YOLO26m live gate passed. The live
  gate observed advancing sequences `38` through `1017` for 30 seconds, accepted
  EOS, received the expected callback, returned from `wait()`, completed shutdown
  in 1.567 seconds, exited `0`, and required no forced kill. The full root suite
  passed with `785 passed, 15 skipped`; the DS9 suite passed with `90 passed`.
- **References:** `noesis_core/servicemaker_shutdown.py`,
  `gst-plugins/noesiseos/`, `DS9/gst-plugins/noesiseos/`,
  `noesis/pipelines/ds8_pipeline.py`, `DS9/noesis/pipelines/ds8_pipeline.py`,
  `noesis/ds8_runtime.py`, `noesis/ds8_runtime_v3dt_reimpl.py`,
  `DS9/noesis/ds9_runtime_core.py`, `scripts/ds8_runtime_30s_gate.py`.

# 2026-07-10 — DS9 live identity and floorplan claims are evidence-scoped

- **Area:** DS9 behavior acceptance / shared identity and floorplan contracts
- **Decision:** The routine occupied-scene identity gate proves authenticated
  shadow-runtime health, fresh server-produced ReID observations, exact
  observation keys, and tracker-local shadow-subject continuity while requiring
  public authority to remain blocked. Cross-camera assignment continuity and a
  fresh open-set non-force decision are separate `observed`/`not_observed`
  claims that can be made mandatory for a staged or naturally occurring scene.
  Neither state is reported as person-level accuracy without licensed truth.
  The floorplan gate derives the active camera inventory from the reviewed
  configs and requires a fresh, non-empty v7 meter-space payload for every
  active camera; a first-camera pass or 1x1 zero-density sentinel is not
  acceptance evidence.
- **Rationale:** Repeated numeric IDs can prove only frame persistence, and one
  successful RPC can hide stale or missing geometry on other cameras. Separating
  mechanism exercise from labeled accuracy prevents optimistic promotion while
  still allowing ordinary family-only health runs to finish without fabricated
  resident or unknown labels. Identity authority remains gated by the existing
  scorer, licensed-truth, coordinator-replay, and occupied-scene artifacts.
- **Validation:** Focused DS9 gate and runner tests cover shadow/authority/model
  health, coherent fresh observations, subject flips, opt-in absent events,
  truthful report statuses, active-camera inventory, stale/empty/corrupt grids,
  and owner-private report ingestion. The focused behavior set passed 37 tests;
  the complete DS9 suite passed 278 tests, with only pre-existing dependency
  deprecation warnings. Ruff, syntax compilation, diff checks, and the
  AGENTS/docs consistency check also passed.
- **References:** `DS9/scripts/ds9_identity_shadow_live_gate.py`,
  `DS9/scripts/ds9_floorplan_live_gate.py`,
  `DS9/scripts/ds9_live_validation_runner.py`,
  `docs/DS8_api_contracts_ws.md`, `plans/household_identity/validation.md`.

# 2026-07-11 — DS9 Wholebody builds cap optimizer tactic DRAM explicitly

- **Area:** DS9 Wholebody49 TensorRT engine maintenance
- **Decision:** Supersede the workspace-only Wholebody `trtexec` build contract
  with one DS9-owned TensorRT 10.14.1.48 C++ builder. The builder retains the
  existing FP16 semantics, uses TensorRT 10's explicit-batch default, requires
  the exact S-mask or X-box tensor set, adds one fixed batch-three
  `images=3x3x640x640` optimization profile, sets WORKSPACE to exactly 4 GiB,
  and sets TACTIC_DRAM to exactly 2 GiB. Both limits are positive powers of two,
  enforced by C++ compile-time assertions, Python maintenance validation, and
  independent source-contract/provenance validation. It registers a fail-closed
  error recorder and verifies both configured limits through their getters
  before building. Optimization level, tactic sources, and auxiliary streams
  remain unchanged so this is a single-variable memory-ceiling adjustment.
  `trtexec` remains the separately recorded candidate and final-path
  deserializer; the outer 9 GiB S and 11 GiB X host guards remain unchanged.
- **Rationale:** Two guarded S-mask builds reached 9,441 MiB and 9,436 MiB even
  with the 4 GiB workspace flag, and both rolled back cleanly. The first native
  builder trial then limited TACTIC_DRAM to 4 GiB but still reached 9,441 MiB
  across 78 host samples; transaction `20260711T075013061941Z` stopped before
  candidate publication and proved the prior realization unchanged. TensorRT
  10.14's `trtexec` CLI does not expose the `kTACTIC_DRAM` pool, whose dGPU
  default is total device memory. Transaction `20260711T080353252694Z` then
  rejected the 3 GiB value before candidate creation or optimizer allocation,
  with a 51 MiB peak across two host samples and the prior realization retained.
  The exact API failure was `Error Code 3: API Usage Error (Parameter check
  failed, condition: (pool == MemoryPoolType::kDLA_MANAGED_SRAM && poolSize ==
  0) || (static_cast<int64_t>(poolSize) >= kDLA_MIN_MEMORY_POOL_SIZE &&
  hasSingleBit(poolSize)).  In validatePoolSize at
  /_src/optimizer/api/builderConfig.cpp:340)`. Lowering only TACTIC_DRAM to the
  next legal 2 GiB value is therefore the controlled experiment. Raising the
  host guard would hide the optimizer allocation problem; changing optimization
  or tactic selection simultaneously would make the result ambiguous.
- **Provenance boundary:** Maintenance snapshots the reviewed zero-external-
  data ONNX and the hash-pinned C++ source into private container tmpfs before
  parsing or compilation. Evidence binds both exclusive copies, the exact
  compiler command and empty `-Werror` log, compiled executable digest/mode,
  variant-specific positive transcript, exclusive mode-0600 candidate, and
  independent loads. Partial candidates remain exclusively owned by the host
  finalizer's audited rollback/recovery transaction.
- **Validation:** Strict compilation and linkage against the pinned DS9 image,
  C++ boundary sanitizer coverage, focused build/rollback/transcript/provenance
  tests, source-contract chain tests, lint/syntax checks, and CPU-only S/X
  maintenance plans. No GPU engine build or runtime-quality claim is made by
  this decision.
- **References:** `DS9/csrc/wholebody49_engine_builder/`,
  `DS9/scripts/rebuild_engines.py`,
  `DS9/scripts/engine_maintenance_common.py`,
  `DS9/scripts/validate_asset_manifest.py`,
  `DS9/DS9_REBUILD_AND_SMOKE_GATES.md`.

# 2026-07-11 — DS9 Wholebody workspace ceilings are variant-specific

- **Area:** DS9 Wholebody49 TensorRT engine maintenance
- **Decision:** Supersede the shared 4 GiB WORKSPACE setting with a value
  selected only from the already validated builder variant: S masks uses
  exactly 2 GiB under its unchanged 9,000 MiB guard, while X boxes retains
  exactly 4 GiB under its unchanged 11,000 MiB guard. Both variants retain the
  legal 2 GiB TACTIC_DRAM ceiling, FP16, fixed batch-three profile, exact output
  sets, and TensorRT's default optimization-level, tactic-source, and
  auxiliary-stream policies. The compiled source has power-of-two assertions
  for both workspace values and the shared tactic limit. Runtime setter/getter
  checks and exactly one variant-specific transcript marker bind the selected
  value; producer and independent provenance validators reject a swapped,
  missing, or duplicate workspace marker.
- **Rationale:** Transaction `20260711T081443440868Z` proved that lowering only
  TACTIC_DRAM from 4 GiB to the legal 2 GiB value did not reduce the S build's
  9,441 MiB peak across 78 samples. The host finalizer proved no candidate was
  published, rollback completed with the engine already absent, and realization
  `c183f91aaecaca3764081b87c962a87233b2b5c21b2d028e2c12f113a9cb16dd`
  remained authoritative. Installed TensorRT 10.14.1.48 headers define
  WORKSPACE as per-operation intermediate storage and document that a smaller
  limit removes tactics over that threshold. Restricting only S is therefore
  the narrowest next controlled experiment; reducing X without observed need
  would spend performance headroom under its larger guard.
- **Provenance boundary:** The shared builder-source digest changes for both
  contracts, so both S and X must be included in one append-only source-contract
  realization rebase before GPU work. Only S changes its semantic workspace
  value; X retains 4 GiB. The predecessor snapshot is immutable and no engine,
  external realization, or tracked asset manifest is changed by the source
  implementation step.
- **Validation:** The 134-test producer, transcript, independent-maintenance-
  proof, rollback, provenance, source-contract-chain, and adversarial set
  passes. Strict compilation against the pinned image and both isolated S/X
  no-GPU plans also pass without authority mutation. The real guarded builds
  remain separate required gates.
- **References:** `DS9/csrc/wholebody49_engine_builder/`,
  `DS9/config/engine_source_contracts.json`,
  `DS9/DS9_REBUILD_AND_SMOKE_GATES.md`,
  `DS9/tests/test_engine_build_specs.py`,
  `DS9/tests/test_wholebody_builder_maintenance_proof.py`.

# 2026-07-11 — DS9 Wholebody S uses optimization level zero

- **Area:** DS9 Wholebody49 TensorRT engine maintenance
- **Decision:** Retain the reviewed variant workspaces (S 2 GiB, X 4 GiB),
  shared 2 GiB TACTIC_DRAM, FP16, fixed batch-three profile, exact output sets,
  tactic sources, auxiliary-stream policy, and 9,000/11,000 MiB outer guards.
  Select and pin TensorRT builder optimization level from the validated variant:
  S masks uses level 0 while X boxes explicitly retains level 3. The setter is
  checked through the registered error recorder, getter-verified immediately,
  getter-verified again before serialization, and represented by exactly one
  variant-specific transcript marker. Producer and independent proof validators
  reject missing, swapped, duplicate, non-integer, or out-of-range levels.
- **Rationale:** Transaction `20260711T083837156365Z` proved that the legal
  2 GiB S WORKSPACE limit alone still reached 9,436 MiB across 78 samples. The
  finalizer proved no candidate, an already-absent engine, complete rollback,
  and realization
  `cda791ecc5aacd6e043f8714c6d19c8bd72879f5492dbea56fe223e34a59dc99`
  preserved. Installed TensorRT 10.14.1.48 headers define level 0 as disabling
  dynamic kernel generation and selecting the first tactic that executes
  successfully. This is the narrowest verified remaining search-pressure
  control. X keeps level 3 because its larger 11,000 MiB guard has not produced
  contrary evidence.
- **Provenance boundary:** The shared builder-source digest changes both source
  contracts. S semantically changes to optimization level 0; X explicitly pins
  its existing level-3 behavior. Both unrealized contracts must therefore move
  through one append-only source-contract realization rebase before GPU work.
  No engine, external authority, or tracked asset manifest changes during the
  implementation step.
- **Validation:** The 140-test focused producer/transcript/proof/rollback/
  provenance/source-chain/adversarial set, strict pinned-image compilation, and
  both no-GPU plans pass. Lint, syntax, JSON, shell, docs-consistency, and diff
  checks also pass without GPU or authority mutation. The real build and
  occupied-scene quality gate remain separate evidence.
- **References:** `DS9/csrc/wholebody49_engine_builder/`,
  `DS9/config/engine_source_contracts.json`,
  `DS9/DS9_REBUILD_AND_SMOKE_GATES.md`,
  `DS9/tests/test_engine_build_specs.py`,
  `DS9/tests/test_wholebody_builder_maintenance_proof.py`.

# 2026-07-11 — DS9 Wholebody S admits the evidenced half tactic with 8 GiB WORKSPACE

- **Area:** DS9 Wholebody49 TensorRT engine maintenance
- **Decision:** Raise only S-mask WORKSPACE from 2 GiB to 8 GiB. Retain S
  builder optimization level 0, X WORKSPACE at 4 GiB and optimization level 3,
  shared 2 GiB TACTIC_DRAM, FP16, fixed batch-three profiles, exact output sets,
  tactic sources, auxiliary-stream policy, and the 9,000/11,000 MiB outer
  guards. The variant getter checks, compile-time power-of-two assertions,
  source contract, producer transcript proof, and independent validator all pin
  the exact 8,589,934,592-byte S value.
- **Rationale:** Transaction `20260711T130350917674Z` proved optimization level
  0 solved the previous build-peak failure: the run peaked at 506 MiB across 19
  samples. It then failed cleanly at the final mask-producing Myelin node because
  both half-format attempts requested exactly 4,571,136,000 bytes against
  budgets of 1,952,111,042 and 1,954,461,634 bytes; the float-format attempt
  requested 9,142,272,000 bytes against 1,856,672,132 bytes. TensorRT advised
  increasing WORKSPACE, skipped all tactics, and returned Error Code 10 with no
  implementation. An 8 GiB pool is comfortably above the observed half request
  while remaining below the observed float request. This is the narrowest
  power-of-two admission change that preserves the intended FP16 path. The
  unchanged outer guard remains authoritative if actual allocation pressure
  rises.
- **Failure and rollback evidence:** No candidate was published. The engine was
  already absent, candidate cleanup completed, rollback reported the realization
  already prior, and realization
  `3a896c8dc8dc50d424e9f87e69e9396f9d10e8251e44312d2859312596defc8e`
  remained authoritative.
- **Provenance boundary:** The shared builder-source digest changes both
  unrealized Wholebody contracts even though only S changes workspace semantics.
  The immutable predecessor is the artifact-root-relative snapshot
  `source_contract_rebase/inputs/20260711T130600Z-wholebody-s-workspace-8192-opt0/old_engine_source_contracts.json`
  with SHA-256
  `2e9a73a18f2105b547118b43b01a2636fa0b05e26e3be284da4830eb027096a2`.
  No engine, external realization, or tracked asset manifest changes during this
  implementation step.
- **Validation:** Strict compilation against the pinned TensorRT 10.14.1.48
  image passes, as do 141 focused producer/transcript/independent-proof/
  rollback/provenance/source-chain/adversarial tests and isolated no-GPU S/X
  plans. The builder digest is
  `d066b43fb6c1f13d6b28213c5fd05d2d55a504fd4c2167ac5509abe73fa6918a`;
  the new source-contract digest is
  `67597d75664b42841bb5dea232b42e312306622e6496abb3f57689ad5c5e4a85`.
  A deterministic dry-run at `2026-07-11T13:10:00Z` proposes realization
  `493d1cdb168a80751a366b0d12382e0e647e1a2a91b35cc14ee3c09a48a95bb0`
  with only `source_contracts.sha256` and `updated_at_utc` changed. Apply and the
  real GPU build remain separately authorized gates.
- **References:** `DS9/csrc/wholebody49_engine_builder/`,
  `DS9/config/engine_source_contracts.json`,
  `DS9/DS9_REBUILD_AND_SMOKE_GATES.md`,
  `DS9/tests/test_engine_build_specs.py`,
  `DS9/tests/test_wholebody_builder_maintenance_proof.py`.

# 2026-07-11 — DS9 Wholebody logger ignores only VERBOSE before bounded copy

- **Area:** DS9 Wholebody49 TensorRT engine maintenance and proof
- **Decision:** Set the dedicated builder's minimum captured TensorRT severity
  to INFO. Return immediately for `Severity::kVERBOSE`, before bounded copy,
  output, truncation bookkeeping, or error bookkeeping. Continue to emit and
  bound INFO, WARNING, ERROR, and INTERNAL_ERROR; any captured-message
  truncation remains fatal, and ERROR/INTERNAL_ERROR continue to set sticky
  fatal state. Both S and X maintenance contracts pin `minimum_severity=info`,
  `verbose=ignored_before_copy`, `captured_message_truncation=fatal`, and
  `error_state=sticky_fatal`. Producer and independent transcript proof require
  one exact marker for each property.
- **Rationale:** Transaction `20260711T131135058479Z` proved the preceding
  8 GiB S workspace policy successfully generated an engine in 55.7403 seconds.
  The old wrapper observed 4,880 MiB across only 19 coarse samples, which is not
  an authoritative peak; TensorRT's 4,603 MiB result covers only its allocator.
  Publication was nevertheless withheld because
  the previous logger bounded every severity and one overlong message among
  55,191 VERBOSE diagnostics set its sticky truncation flag. The log contained
  25 INFO, three WARNING, zero ERROR, and zero INTERNAL_ERROR messages. Ignoring
  only VERBOSE aligns logging admission with the useful diagnostic threshold;
  weakening captured-message truncation or real-error checks would discard the
  fail-closed proof boundary and is explicitly rejected.
- **Failure and rollback evidence:** The builder failed after serialization and
  before candidate write, so no candidate was published. The host transaction
  completed candidate cleanup, found the engine already absent, reported the
  realization already prior, and preserved
  `493d1cdb168a80751a366b0d12382e0e647e1a2a91b35cc14ee3c09a48a95bb0`.
- **Provenance boundary:** The logger implementation is shared, so its source
  digest changes both unrealized Wholebody contracts; both also gain the exact
  logger-policy object. The immutable predecessor is the artifact-root-relative
  snapshot
  `source_contract_rebase/inputs/20260711T131400Z-wholebody-logger-info/old_engine_source_contracts.json`
  with SHA-256
  `67597d75664b42841bb5dea232b42e312306622e6496abb3f57689ad5c5e4a85`.
  No engine, external realization, or tracked asset manifest changes during this
  implementation step.
- **Validation:** A host-compiled harness extracts and executes the production
  Logger class: overlong VERBOSE causes no output/state, overlong INFO/WARNING
  sets fatal truncation state, overlong ERROR/INTERNAL_ERROR sets truncation and
  sticky error state, and a normal real error remains sticky. Strict pinned
  TensorRT 10.14.1.48 compilation, 148 focused producer/proof/rollback/
  provenance/adversarial tests, and isolated S/X no-GPU plans pass. The builder
  digest is
  `2ef26613e87ff9bafb68e8e400e3e596012d75f7b98ff54276ae180154849780`;
  the source-contract digest is
  `7049fe51e553393a5aa4d646ac792c2233dd38849c7f1e022c74854e787aecca`.
  A deterministic dry-run at `2026-07-11T13:18:00Z` proposes realization
  `e87a620dbcc1f231cc8e2e753e20fe49babb00322ce1184f71fba09a697a501b`
  with only `source_contracts.sha256` and `updated_at_utc` changed. Apply and a
  new real build remain separately authorized gates.
- **References:** `DS9/csrc/wholebody49_engine_builder/`,
  `DS9/config/engine_source_contracts.json`,
  `DS9/DS9_REBUILD_AND_SMOKE_GATES.md`,
  `DS9/tests/test_engine_build_specs.py`,
  `DS9/tests/test_wholebody_builder_maintenance_proof.py`.

# 2026-07-11 — Engine realization requires durable raw NVML guard evidence

- **Area:** DS9 TensorRT engine maintenance, finalization, and realized
  provenance
- **Decision:** Create each maintenance container without starting it, launch an
  NVML-v2 sampler bound to the exact GPU UUID/index, engine, container,
  transaction digest, artifact-root ID, and wrapper PID/start ticks, require an
  initial sample, and only then start the container. Sample raw used/reserved
  bytes every 25 ms and reject any raw-byte guard breach, NVML failure, sampler
  death, malformed evidence, or gap above 250 ms. Store the mode-0600 JSONL at
  `models/engine_finalize/<transaction>-<engine>/gpu-memory.jsonl` inside the
  retained mode-0700 transaction cohort. Persist its exact path, digest,
  identities, policy, and independently reconstructed summary under
  `provenance.maintenance.gpu_memory_guard`. Finalization, reconciliation, and
  later authoritative realization validation each reopen and reconstruct the
  raw evidence; a mutating direct reconcile cannot omit the guard.
- **Compatibility boundary:** Wholebody49 always requires the guard. Existing
  non-Wholebody realization records are not broadly grandfathered by schema or
  timestamp. Only the eight exact artifact ID/output digest/maintenance
  path/secure maintenance digest tuples present at the
  2026-07-11T13:18:00Z cutoff may remain guardless. Changing any tuple consumes
  the exemption. A present null or malformed guard is invalid. Authoritative
  provenance validation also requires file checks; it cannot return success
  with `check_files=False`.
- **Rationale:** Transaction `20260711T132034937854Z` generated and independently
  deserialized a 25,348,956-byte S engine but finalization exposed a missing
  caller-known maintenance-manifest path and rolled the no-prior engine back.
  The old wrapper's 4,846 MiB result came from 20 coarse samples while an
  operator observed roughly 9,472 MiB. The preceding 4,880 MiB result was
  similarly coarse, and TensorRT's 4,603 MiB metric was allocator-scoped. None
  can prove compliance with the 9,000 MiB S ceiling. Starting a high-frequency
  monitor before GPU work and making its raw evidence part of artifact
  acceptance closes both the transient-sampling gap and the direct-reconcile
  omission path.
- **Trust boundary:** This is fail-closed operational evidence under the
  appliance's same-UID private-file model, not cryptographic attestation. The
  evidence and maintenance host-transaction fields could still be fabricated by
  a hostile process with the same UID. Cross-checking the eventually committed
  finalizer transaction is retained as hardening debt because realization
  validation occurs before the transaction transitions from prepared to
  committed.
- **Validation:** The sampler/Wholebody/reconcile/finalizer suite passes 94
  tests; the complete maintenance-wrapper suite passes 33, including transient
  breach, already-exited sampler, sampler death, evidence mutation/absence,
  direct-reconcile omission, strict legacy exemption, and transaction-versus-
  realization equality. Live read-only validation accepts the unchanged eight
  pre-guard records at realization
  `e87a620dbcc1f231cc8e2e753e20fe49babb00322ce1184f71fba09a697a501b`
  with zero errors or blockers. No engine build or GPU run was performed after
  this design change.
- **References:** `DS9/scripts/nvml_gpu_memory_sampler.py`,
  `DS9/scripts/run_canonical_engine_maintenance.sh`,
  `DS9/scripts/finalize_engine_realization.py`,
  `DS9/scripts/reconcile_engine_provenance.py`,
  `DS9/scripts/validate_asset_manifest.py`,
  `DS9/tests/test_nvml_gpu_memory_sampler.py`,
  `DS9/tests/test_engine_maintenance_wrapper.py`.

# 2026-07-11 — Tier-4 browser evidence is a private exact-state admission contract

- **Area:** DS8/DS9 to Menon cross-space validation
- **Decision:** Admit live Tier-4 browser evidence only when a fresh authenticated
  owner/operator session proves the exact final same-origin page, and the page's
  canonical world state, presentation/debug snapshot, active scene cohort,
  current render paths, authored transform, entity identity and lifecycle,
  timestamps, backend positions, and rendered positions agree under one
  run/sequence cursor. DS8 and DS9 are valid producers of that shared contract;
  replay/test producers and self-labeled debug or latest-path arrays are not.
  Browser snapshots, traces, screenshots, and reports are private household
  evidence and therefore require owner-only directories and files.
- **Rationale:** A plausible path array or an authenticated page alone cannot
  prove that Menon rendered current canonical Noesis state. Exact cross-surface
  binding prevents stale, mixed-run, transformed, spoofed, or selectively copied
  debug data from being promoted as integration evidence, while producer-neutral
  admission preserves the DS8-to-DS9 cutover boundary.
- **Validation:** The canonical browser fixture passes the shared trace report.
  Focused tests reject absent or expired session proof, origin/final-URL changes,
  stale entities, cursor and path drift, inactive cohorts, identity/lifecycle
  changes, wrong positions or transform counts, and unsafe evidence paths; IPv4
  and IPv6 LAN origins are normalized canonically.
- **References:** `noesis/validation/menon_browser.py`,
  `scripts/noesis_validation_capture_menon_trace.py`,
  `plans/noesis_menon_validation/validation_tiers.md`,
  `plans/noesis_menon_validation/minimal_menon_browser_snapshot.json`.

# 2026-07-11 — DS9 static ownership truth is separate from dynamic acceptance

- **Area:** DS8-to-DS9 successor governance and runtime evidence promotion
- **Decision:** Keep `DS9/docs/runtime_ownership.yaml` selector-free. Its status
  describes static implementation/contract truth only: implemented surfaces may
  be `parity`, `shared`, or `adapter_specific` while current evidence is absent;
  `known_gap` and `blocked` remain normative and cannot be overridden. Store
  realized/runtime selectors only in a fixed owner-private external registry
  below the explicit DS9 runtime root. Registry events use canonical unique-key
  JSONL, a global hash chain, explicit per-key supersession, current matrix and
  checkout digests, complete artifact/session bindings, a persisted head,
  descriptor CAS, advisory locking, and fsync before success.
- **Rationale:** Editing hashes into the tracked matrix made promotion mutate
  the checkout it was meant to prove and created a matrix-digest deadlock.
  Independent static and dynamic axes preserve honest contract state, let a
  current promotion clear only an evidence requirement, and intentionally
  invalidate prior promotions when real policy changes.
- **Behavior evidence:** Wholebody, floorplan, and V3DT world claims now seal
  bounded timestamped minimal-field source transcripts, matching the existing
  identity/semantic replay model. Reports are exactly recomputed and their
  source windows must fall inside the inspected container lifetime. Images,
  raw embeddings, full payloads, and secrets are excluded.
- **Trust boundary:** This is tamper-evident under the appliance's owner-private
  same-UID model, not remote attestation. A hostile same-UID process can rewrite
  both registry and head; a privileged or remote monotonic anchor is future
  hardening. One-sided rollback, truncation, splice, branch, and conflict are
  rejected now.
- **Validation:** Offline adversarial coverage includes selector-free policy,
  static blocker precedence, missing-evidence blockers, current and stale matrix
  epochs, explicit supersession/revoke, digest/chain/duplicate-key/rollback and
  symlink rejection, session/event binding, source privacy/replay, Docker
  swappiness, and terminal launcher/checkout CAS. No live runtime or GPU action
  is part of registry recording.
- **References:** `DS9/scripts/validate_runtime_ownership.py`,
  `DS9/scripts/runtime_ownership_registry.py`,
  `DS9/scripts/promote_runtime_ownership_evidence.py`,
  `DS9/docs/runtime_ownership_evidence.md`.

# 2026-07-11 — Wholebody49 tactic admission and the full-device build guard are separate contracts

- **Area:** DS9 TensorRT 10.14 Wholebody49 S-mask and X-box engine maintenance
- **Decision:** Keep S at builder optimization level 0, 6 GiB WORKSPACE, and
  2 GiB TACTIC_DRAM, with a 10,000 MiB outer NVML guard. Keep X at optimization
  level 3, 4 GiB WORKSPACE, 2 GiB TACTIC_DRAM, and its existing 11,000 MiB
  outer guard. WORKSPACE must be positive and MiB-aligned; TACTIC_DRAM retains
  TensorRT's positive power-of-two rule. Both variants continue to use the
  DS9-owned C++ builder, exact fixed batch-three profiles, FP16, unchanged tactic
  sources/auxiliary-stream policy, private source/ONNX snapshots, exclusive
  candidate creation, and independent candidate/final deserialization.
- **Why:** The final S mask node requires a 4,571,136,000-byte half-format
  tactic, so 2 GiB WORKSPACE cannot build the graph. The installed TensorRT
  10.14 headers define WORKSPACE as intermediate operation memory and level 0 as
  selecting the first tactic that succeeds. A successful builder transcript
  separately reported a 4,603 MiB TensorRT allocator peak. Full-device NVML
  evidence nevertheless remained near 9.48 GiB: transaction
  `20260711T150649395390Z` reached 9,481 MiB with 8 GiB WORKSPACE, while
  `20260711T151452845569Z` reached 9,476 MiB with 6 GiB. WORKSPACE reduction is
  therefore not a valid process-peak control. A 10,000 MiB S guard accepts the
  repeatable measured envelope while preserving more than 2 GiB free on the
  12 GiB target and remaining stricter than the X guard.
- **Source realization:** Builder SHA-256
  `2d751f52fc095bd90e418e048fad8296653875d43b8a18871d5b0a7c7938dc67`
  and source-contract SHA-256
  `7ddd449c82e80d5c3195c0ccfb4ba4c3d4d542c654234f5d1f31d4ea549a20e0`
  were applied through append-only source-rebase transaction
  `20260711T151317697215Z`, producing realization
  `f1bad48cc1638399da2a08f614801c7120409a8d8a8fc5c730f846146f231224`.
  The first 10,000 MiB S build generated and loaded a candidate, but transaction
  `20260711T151938705545Z` rejected a disconnected source-rebase graph and
  rolled back. The corrected validator requires an exact internal rebase graph
  while allowing only a monotonic successor realization: no realized artifact
  may disappear and the realization timestamp must strictly advance.
- **Realized result:** Committed transaction `20260711T152338477896Z` installed
  the mode-0600, 25,327,772-byte S engine at SHA-256
  `1fb95225e8258af13ac96de5136b85dadb60419a8c49058e70536eef06dd6bdf`.
  Its sealed guard evidence contains 2,738 samples, a 27.742316 ms maximum gap,
  and 9,481 MiB maximum observed. Committed transaction
  `20260711T152552934749Z` installed the mode-0600, 108,579,580-byte X engine at
  SHA-256
  `a5f4322d7e123461a1bbc64388b6f0e30c9359091e76ac7a95144648841138c3`.
  Its evidence contains 12,750 samples, a 30.858598 ms maximum gap, and 1,017
  MiB maximum observed. Candidate and installed-path deserialization passed for
  both. The final ten-engine realization is
  `99dc3aba39f6a177f404a22ac70e02b2109c89ca476eb717540e3577a4ad614b`;
  canonical, V3DT, and Wholebody49 profiles pass file/provenance validation.
- **Boundary:** These results prove deterministic artifact realization and the
  sampled build-memory envelope. They do not prove live S-mask or X-box quality,
  occupied-scene behavior, throughput, runtime memory stability, or cutover
  readiness; those remain separate typed runtime-evidence gates.
- **References:** `DS9/csrc/wholebody49_engine_builder/`,
  `DS9/config/engine_source_contracts.json`,
  `DS9/scripts/run_canonical_engine_maintenance.sh`,
  `DS9/scripts/validate_asset_manifest.py`,
  `DS9/DS9_REBUILD_AND_SMOKE_GATES.md`.

# 2026-07-11 — Wholebody runtime promotion requires three replayable truths

- **Area:** DS9 Wholebody49 live quality, resources, media, and ownership
- **Decision:** A Wholebody49 promotion requires three distinct typed documents
  from one terminally sealed supervisor session: occupied parser-path evidence
  v2, direct-RTSP plus WebRTC decoded-media evidence, and generic resource-soak
  evidence v2. Occupancy is never inferred from resource or media health. The
  occupied transcript records only timestamp, source ID, frame ID, person count,
  three zero-copy counters, and readiness/error booleans; it requires the exact
  configured source inventory, at least 30 seconds, strictly advancing frames,
  a provisional 2 frame-ID/s floor per source, and no observation gap above 2.5
  seconds. An empty house is explicitly `scene_status=empty` and cannot pass.
- **Resource boundary:** V3DT and both Wholebody lanes share the same 300-second,
  five-second-sample, 60-second-warmup evaluator. It rejects OOM/OOM-kill,
  cgroup memory at 20 GiB, post-warmup growth above 512 MiB or slope above 1
  MiB/s, PIDs at 4096, S GPU process memory at 10,000 MiB, and X/V3DT GPU
  process memory at 11,000 MiB. Every sampled GPU owner must descend from the
  exact container. The report and raw samples bind container, image, checkout,
  realization, primary lane engine, pipeline config, and camera config. These
  are provisional fail-closed policy bounds, not empirical throughput claims;
  they are never relaxed automatically.
- **Media/lifecycle boundary:** The media report persists only closed connection
  states and aggregate RTP/decode counters, requires at least one directly
  decoded RTSP H264 frame and one decoded WebRTC frame, and contains no image,
  SDP, ICE, address, or secret payload. CPU `avdec_h264` exists only in this edge
  validation client. The runtime-session validator separately requires exact
  readiness, container inspection, zero OOM, exit zero, acknowledged ordered
  EOS, removal, closed ports, unchanged checkout, current artifacts, and
  terminal launcher/checksum CAS. The retired V3DT-only soak contract is
  rejected rather than silently reinterpreted.
- **Validation:** Focused producer, transcript, decoded-media, resource,
  supervisor, runner, ownership-replay, and adversarial tests pass without
  starting a runtime, Docker container, or GPU workload. Fresh live S and X
  sessions remain required.
- **References:** `DS9/scripts/wholebody49_occupied_scene_smoke_test.py`,
  `DS9/scripts/wholebody49_media_decode_gate.py`,
  `DS9/scripts/run_canonical_runtime_container.py`,
  `DS9/scripts/validate_runtime_ownership.py`,
  `DS9/docs/runtime_ownership_evidence.md`.

# 2026-07-11 — Selector-bound appliance admission and shared state ownership

- **Area:** DS8/DS9 deployment, runtime identity, mutable product state, health
- **Decision:** Appliance deployment uses one closed canonical selector and the
  cross-language `noesis-runtime-v1` identity. A clean Git revision/tree is not
  sufficient: the identity also byte-binds every approved ignored runtime
  directory, native extension, parser, and selected external model artifact.
  Admission rejects duplicate JSON keys, noncanonical bytes, checkout drift,
  symlink/hardlink escapes, and mutation during inventory. DS9 repeats the full
  identity immediately before container exec; health retains the exact selector
  digest instead of claiming a full rehash on each request.
- **State boundary:** The selected release owns exact analytics, identity, and
  world schema versions plus an immutable activation baseline. Its four mutable
  payload files and release-local build directory are bound by exact process
  environment and admitted inode cohort, rechecked immediately before runtime
  state use. DS9 mounts those same selected stores into canonical container
  paths; it does not seed, copy, or fall back to canary state. The selected DS9
  executable is frozen to the canonical owner-owned, single-link supervisor.
  Menon owns the one full-runtime shared release lease. Noesis accepts no nested
  lease interface or inherited lease variable.
- **Health boundary:** Selector-driven REST health v1 and WebSocket health v2
  require one exact producer identity and healthy advancing
  `tracking_observations` plus `global_world`. Non-selector WebSocket health
  remains v1. A selector runtime that is not ready closes/fails; it never emits
  partial selector health or downgrades to v1.
- **Rationale:** A release is a coherent product deployment, not a source commit
  plus whichever ignored engines, writable databases, or cached configs happen
  to be present. Binding code, runtime bytes, state, producer progress, and
  lifecycle ownership prevents split truth while keeping mutable home state
  outside immutable checkouts.
- **Validation:** Generated schema drift passes; focused Noesis appliance,
  health, inference-path, DS9 supervisor/mount, and cross-language Menon
  deployment/release tests pass without launching Docker, a service, or GPU
  work.
- **References:** `contracts/noesis-runtime-v1.md`,
  `noesis_core/appliance.py`, `noesis/ds8_runtime.py`,
  `DS9/scripts/run_canonical_runtime_container.py`,
  `docs/DS8_api_contracts_rest.md`, `docs/DS8_api_contracts_ws.md`.

# 2026-07-11 — Canonical BEV ownership and raw-only capture-event fusion

- **Area:** DS8/DS9 local BEV display, capture-event depth fusion, and failure truth
- **Decision:** Keep one SDK-neutral BEV implementation at
  `noesis/telemetry/bev.py`. DS9 extends the telemetry namespace and may not
  shadow that module. Ownership evidence is an isolated DS9-first import-origin
  check plus the canonical 25-case behavior characterization, not source-string
  resemblance.
- **Decision:** Reset camera-local smoother and tracker trail history when the
  selected measurement source or active floorplan coordinate-space signature
  changes. A failed current homography is not replaced with a cached transform;
  homography and publication failures update structured health and invoke the
  runtime failure callback.
- **Decision:** `noesis_core.capture_event_fusion` owns one deterministic,
  raw-only intra-capture cohort. Cache-only requests reject before any provider
  or write; derived inputs, mixed scopes, duplicates, stale baselines, overwide
  cohorts, and result/source substitution fail closed. Optional RGB must be a
  typed timestamped frame already offered by the active pipeline. No second
  source reader, CPU decode branch, or hidden RGB fallback is permitted.
- **Runtime boundary:** Current DS9 hooks do not expose a legitimate full RGB
  frame. Initial runtime integration therefore uses explicitly requested
  depth-only fusion or returns `rgb_frame_unavailable`; GPU-first RGB enrichment
  is separate. DS8, protected V3DT, and DS9 now share the same owned controller,
  active-floorplan registry, failure callback, and exact response provider;
  real DS9 promotion evidence remains a separate release gate.
- **Validation:** Canonical BEV `25/25`; combined focused BEV, capture-fusion,
  and ownership tests pass without a runtime, GPU, camera open, or network
  source.
- **References:** `noesis/telemetry/bev.py`,
  `noesis_core/capture_event_fusion.py`, `DS9/tests/test_bev_parity.py`,
  `DS9/tests/test_capture_event_fusion.py`,
  `DS9/docs/bev_capture_event_integration.md`.

# 2026-07-11 — Tracker-lifetime identity continuity and exact embedding provenance

- **Area:** Shared DS8/V3DT/DS9 household identity and public observation truth
- **Decision:** Once identity-v2 accepts a subject for a camera-local tracker,
  hard-mask every alternative subject until the tracker state expires after a
  real fresh-evidence gap. The locked subject receives no force-match bonus: it
  must still pass all prior-independent open-set and exclusivity gates, else the
  fresh result is unknown. A post-resolution invariant also downgrades any
  concurrent gallery-race subject change to unknown.
- **Decision:** The process-owned v2 adapter, not legacy StableID cache
  diagnostics, owns exact-frame `embedding_present`. Fresh server evidence
  carries its observation key, and enabled shadow capture adds only the complete
  persisted sequence/model/dimension triad to both public and canonical
  observation surfaces.
- **Decision:** Shadow JSONL consumes the resolver's exact post-constraint
  candidate rows. It does not recompute a second gallery snapshot after or
  alongside resolution, so continuity/copresence masks and their reasons remain
  replayable and cannot drift across a retention mutation.
- **Decision:** DS9 must construct legacy StableID under the same default-on
  household policy as DS8. Household admission forces and verifies
  `auto_merge_enabled=false` and `allow_multi_zone_active=false`; an old
  environment request cannot override those invariants.
- **Rationale:** The first sealed occupied DS9 baseline transcript observed a
  same-tracker visitor switch within 0.683 seconds, stale frame-level embedding
  claims without persisted association, and a DS9-only automatic alias merge.
  These violated tracker continuity, semantic provenance, and the no-pressure-
  merge product rule while identity-v2 remained shadow-only.
- **Validation:** Deterministic replay of the contested two-track assignment,
  exact service provenance tests, and isolated DS9 builder-policy execution all
  pass without launching Docker/GPU work or altering the sealed evidence.
- **References:** `plans/household_identity/decisions.md`,
  `docs/DS8_api_contracts_ws.md`, `docs/DS8_metadata_contracts.md`,
  `DS9/docs/migration_state.md`.

# 2026-07-11 — Depth snapshots are immutable transactional commits

- **Area:** Shared DS8/V3DT/DS9 MapAnything snapshot storage and fusion
- **Decision:** Admit every depth snapshot as a unique camera/timestamp write
  with a process-unique write ID and monotonic sequence through one bounded
  queue. Publish only a validated same-parent staging tree carrying a complete
  file-hash commit manifest, using no-replace rename and parent-directory fsync.
  A write failure permanently poisons the manager with the first failure;
  callers receive exact write, flush, and shutdown receipts instead of inferred
  queue state or synchronous fallback behavior.
- **Decision:** Treat committed snapshot trees as immutable. RGB and fusion
  provenance must be part of the original atomic job; post-commit dataset or
  attribute attachment is rejected. Fusion pins every source against retention
  until its exact derived commit completes. Startup indexes only fully validated
  committed manifests in timestamp order. Pre-contract Zarr trees are reported
  and ignored unless an operator explicitly republishes one through the
  validation-and-commit migration API. A public immutable snapshot descriptor
  binds camera, timestamp, write sequence, path/reference, manifest/content
  digests, and role. Fusion seals that identity for every requested source and
  rejects the entire operation if any member cannot be proven exactly.
- **Decision:** A MapAnything result is not observable until its exact write
  handle returns a commit receipt. DS8, protected V3DT, and DS9 wait within the
  shared finite 0.1–60 second bound (30 seconds default); failure poisons the
  worker and signals runtime shutdown before any frame receipt or telemetry is
  emitted. The former storage-disable `memory://` publication path is retired,
  with no direct-write fallback.
- **Rationale:** A returned pathname previously meant only that work might have
  been queued; queue pressure silently switched to direct writes, worker errors
  were log-only, two workers could disorder the newest index, fusion attributes
  were mutated after a generic flush, and retention could remove active fusion
  inputs. Those behaviors made durability, restart recovery, and shutdown
  evidence unknowable. The transactional boundary makes every accepted or
  rejected state replayable without introducing a second storage path.
- **Validation:** `tests/test_depth_storage_transactions.py` exercises blocked
  writers and deadlines, saturation, disk poison, reordered completion,
  duplicate reservation, interrupted restart and explicit migration, exact
  fusion commit, read-lease pruning, immutable RGB input, parent durability,
  and corruption rejection. `tests/test_depth_capture_event_adapter.py` proves
  the public descriptor/evidence boundary and derived-input exclusion.
- **Validation:** `tests/test_mapanything_worker_lifecycle.py` and
  `DS9/tests/test_mapanything_exact_native_capture.py` prove exact
  store/receipt/record/publication order plus timeout poison, runtime callback,
  and no-publication behavior in all active hook families.
- **References:** `geometry/depth_source.py`,
  `noesis/depth_capture_event.py`,
  `tests/test_depth_storage_transactions.py`,
  `tests/test_depth_capture_event_adapter.py`, `docs/DS8_testing_guide.md`.

# 2026-07-11 — Public controls require an applied-state owner

- **Area:** Noesis WebSocket controls, dashboard, and Menon gateway
- **Decision:** Retire `update_detection_config`, `set_detection_toggle`, and
  `ma_heatmap_ready` across the server and clients. Do not acknowledge or
  rebroadcast an apparent applied state unless an active DS8/V3DT/DS9 runtime
  callback owns the operation and can return a real receipt.
- **Rationale:** These routes had no registered runtime callbacks. Detection
  sliders and toggles only rebroadcast client input, while the heatmap-ready
  notification performed no server action, creating product UI that looked
  operational but could not change or confirm runtime state.
- **Validation:** The WebSocket regression suite asserts the retired routes are
  absent; the dashboard production build passes; Menon's gateway tests pass
  under its declared Node 22 runtime.
- **References:** `websocket_server.py`, `oai2-fe/src/hooks/useWebSocketClient.ts`,
  `oai2-fe/src/components/ControlsPanel.tsx`,
  `oai2-fe/src/components/DepthDrawer.tsx`,
  `Menon/server/gateway/ws-proxy.js`.

# 2026-07-11 — Capture events and active floorplans are exact owned transactions

- **Area:** DS8, protected V3DT, and DS9 depth/floorplan RPCs and inline BEV
- **Decision:** One synchronous capture controller owns the process-wide
  MapAnything valve, per-camera admission, worker-idle barriers, durable storage
  frontiers, and exactly one raw-only fusion. Cache-only requests return before
  admission or capture. The shared valve prevents safe cross-camera overlap, so
  concurrent capture attempts fail explicitly with `capture_event_busy` until a
  source-specific gate exists.
- **Decision:** A fresh response reloads only the fused commit identified by its
  canonical camera, portable storage reference, write ID, manifest/content
  digests, and timestamp. Floorplan generation pins that exact commit and keys
  its cache by snapshot identity. Every non-cache floorplan request is an
  explicit fresh capture; it never asks for “latest” before or after fusion and
  never returns an older cached payload beside a capture error. Exact success
  atomically publishes isolated bounded memory entries under its write-ID key
  and explicit cache-only `latest` alias without overwriting the generic disk
  cache.
- **Decision:** Inline BEV is locked to `camera_local_ground_m` in every active
  DS8/DS9 baseline and V3DT configuration. A bounded registry accepts only
  contract-valid floorplans with exact snapshot and calibration identities;
  calibration changes clear affected records before publication. Canonical
  world telemetry remains independently owned in `backend_world_m`.
- **Decision:** Startup owns Python resources in a reversible transaction until
  native activation begins. Once activation is attempted, only external ingress
  may close until a live Service Maker wait owner is proven. Normal teardown
  releases callback-owned storage and services only after REST, WebSocket,
  WebRTC, MapAnything, EOS, and wait-loop quiescence receipts succeed.
- **Rationale:** The former path could open a second RTSP reader, mutate a
  committed snapshot, fuse the first arriving frame before the cohort drained,
  race two RPCs through one valve, regenerate a floorplan from a later raw
  frame, or tear storage down beneath partially activated native callbacks.
  Those states looked successful but were not replayable or ownership-safe.
- **Validation:** Transactional storage, controller, exact floorplan, active
  registry, lifecycle, boundary, and synthetic non-promotable smoke tests cover
  the CPU-only contract. Promotion still requires fresh real DS9 baseline,
  V3DT, and Wholebody runtime evidence.
- **References:** `noesis/capture_event_controller.py`,
  `noesis/depth_capture_event.py`, `geometry/depth_source.py`,
  `noesis_core/active_floorplan.py`, `noesis_core/startup_lifecycle.py`,
  `DS9/docs/bev_capture_event_integration.md`.

# 2026-07-11 — Runtime ownership is one fail-closed transaction through finalization

- **Area:** DS8, protected V3DT, and DS9 process lifecycle ownership
- **Decision:** Every resource created during reversible startup is registered
  atomically with its cleanup, and every process-global binding registers an
  idempotent cleanup before mutation. Registration or installation failure
  performs and records the exact local rollback; an unproven rollback cannot
  return normally.
- **Decision:** A phase-aware main guard owns all otherwise-unhandled failures
  from the first reversible binding through successful Service Maker wait-owner
  handoff. Before activation, it drains the transaction. Once activation is
  attempted, it arms the independent watchdog before closing external ingress,
  records the exact ingress-cleanup receipt, and preserves callback-owned
  dependencies until process exit whenever native ownership is ambiguous.
- **Decision:** Runtime completion is a positive proof, not the absence of an
  exception. Persistence, owned services, storage, diagnostics, and every
  process-global binding finalize independently; the startup transaction may
  become quiesced and the watchdog may be canceled only after every receipt
  succeeds. No finalizer failure may emit `Shutdown complete`.
- **Rationale:** Construction, registration, and global installation were
  separate operations, so exceptions could leave an unowned resource or a
  partially installed binding. Assembly and wait-handoff exceptions could also
  escape the lifecycle transaction, while normal teardown could report success
  after a close or persistence failure. One explicit ownership protocol makes
  those states fail-closed and keeps DS8/DS9 successor behavior aligned.
- **Validation:** Focused lifecycle, shutdown-contract, synthetic-backend,
  storage, diagnostics, and DS9 profile coverage passed 131 tests. Targeted
  syntax and lint checks passed. Fresh native runtime acceptance remains a
  separate promotion gate.
- **References:** `noesis_core/startup_lifecycle.py`,
  `noesis/ds8_runtime.py`, `noesis/ds8_runtime_v3dt_reimpl.py`,
  `DS9/noesis/ds9_runtime_core.py`.

# 2026-07-11 — DS9 native freshness is content provenance, not filesystem time

- **Area:** DS9 native extension admission and immutable release worktrees
- **Decision:** Before any DS9 extension import, attest all six exact native
  artifacts against the fixed tracked manifest. Admission requires strict
  unique-key YAML, exact artifact/source/builder membership, validator-identical
  aggregate source hashing, one active-CPython-ABI output directly under
  `DS9/native_extensions`, exact output SHA-256, and stable regular owner file
  identities. Runtime materialization repeats the two depth attestations before
  graph construction. Multiple wildcard outputs, symlinks, hardlinks, path
  escape, mutation during read, or any source/output mismatch fail startup.
- **Decision:** Keep DS8's existing native policy unchanged. DS8 has no
  equivalent DS9 artifact-manifest authority, and consuming the successor's
  manifest would collapse the intentional SDK-major ownership boundary.
- **Rationale:** The first immutable-checkpoint launch rejected an unchanged
  depth bridge solely because checkout creation made the tracked source mtime
  newer than its separately hydrated binary. All source aggregates and binary
  hashes matched reviewed provenance. An isolated rebuild then reproduced five
  extensions byte-for-byte while two identical CUDA/NPP tensor builds produced
  different binaries due to NVCC-generated identifier salts. Timestamp order
  and blind rebuilding are therefore both weaker than exact content identity.
- **Validation:** Focused provenance, origin, pre-import ordering, manifest
  ambiguity, inference-runtime, and DS9 profile tests pass. Exact DS9 build and
  runtime images completed no-GPU rebuild, import/API, and ELF linkage smokes
  with no device nodes; fresh live baseline acceptance remains a separate gate.
- **References:** `DS9/noesis/native_artifact_provenance.py`,
  `DS9/noesis/ds9_runtime.py`, `DS9/scripts/ds9_preflight.py`,
  `DS9/scripts/validate_asset_manifest.py`,
  `DS9/tests/test_native_artifact_provenance.py`.

# 2026-07-12 — Validation helpers have explicit repository package authority

- **Area:** Shared authenticated smoke clients and DS9 live evidence runner
- **Decision:** Treat root `scripts/` as an explicit Python package. Every
  command that inserts the repository root and imports
  `scripts.internal_auth_client` must resolve the repository package and helper
  files exactly; a subprocess regression owns that origin contract.
- **Rationale:** A healthy sealed DS9 baseline processed advancing world state
  and RTSP video, but every Python behavior gate exited before connecting. The
  host had an unrelated regular package named `scripts`; under PEP 420 it won
  over the repository's implicit namespace even with the repository path first.
  An explicit package marker removes that environment-dependent ambiguity
  without duplicating authentication logic or adding an alternate client path.
- **Validation:** The internal-auth client suite and all 17 authenticated smoke
  command `--help` imports pass under the same host interpreter that reproduced
  the collision. Fresh same-checkout live behavior evidence remains required.
- **References:** `scripts/__init__.py`,
  `scripts/internal_auth_client.py`,
  `tests/test_internal_auth_smoke_clients.py`,
  `DS9/scripts/ds9_live_validation_runner.py`.

# 2026-07-12 — Semantic completeness is a bounded immutable identity cohort (semantic-v2 draft; superseded below)

- **Area:** DS9 occupied semantic acceptance and V3DT runtime ownership
- **Decision:** An exact public-track/canonical-observation/private-evidence
  embedding row is the semantic identity anchor. Pose, usable depth, and
  `backend_world_m` may come from independently scheduled frames only when the
  selected cohort spans at most 1.5 seconds and every observed frame retains
  the same runtime run, source, camera, tracker, non-null identity-v2 subject,
  compatibility SID, resident UUID or visitor generation, and exact
  calibration/model/config fingerprints. Exact-frame evidence is span zero.
- **Decision:** Any observed tracker absence, null/conflicting identity,
  tracker-ID reuse, fingerprint change, replay, or out-of-bound endpoint breaks
  the cohort. Pose/depth/world availability is reported independently so a
  failed join cannot be misreported as a missing producer component.
- **Rationale:** ReID, pose, DAv2 depth, and canonical world run at different
  cadences. Requiring one lucky all-modal frame rejected healthy pipelines;
  joining by tracker ID alone could instead splice a reused tracker or changed
  identity/config. The bounded immutable segment preserves causality without a
  fallback or temporal smoothing path.
- **Validation:** The semantic v2 unit suite covers both inclusive endpoints,
  one-microsecond overflow, cross-camera/run/fingerprint splices, observed
  disappearance/reuse, null/conflicting identity, resident and visitor
  continuity, source/private replay, and raw-vector privacy. The live runner
  and ownership validator require schema/contract v2 and recompute the report
  from its sealed source transcript plus exact private evidence snapshot.
- **References:** `DS9/scripts/ds9_semantic_observation_smoke_test.py`,
  `DS9/scripts/validate_runtime_ownership.py`,
  `DS9/docs/validation_runbook.md`, `docs/DS8_testing_guide.md`.

# 2026-07-12 — Semantic v3 binds publication, lifecycle, capture, registration, and privacy

- **Area:** DS8/V3DT/DS9 tracking publication and DS9 occupied semantic
  acceptance.
- **Decision:** A camera frame ID is not a publication sequence because tracking
  is intentionally rate limited. The publisher therefore owns one contiguous
  per-source sequence, while a shared registry processes every frame, assigns a
  positive tracker lifecycle generation, forces publication whenever the
  tracker-key set changes, and emits exact disappearance tombstones. A reused
  numeric tracker ID never inherits the prior lifecycle generation. Because an
  attached live gate may begin mid-run, its first observed publication sequence
  and lifecycle generation are explicitly externally unanchored; continuity is
  claimed only within the sealed acquisition window. Tombstone last-seen
  frame/time must equal the registry's actual preceding published presence.
- **Decision:** Semantic cohorts independently bound captured and observed time
  to 1.5 seconds and require coherent media PTS. Depth qualifies only with
  `depth_status=ok`, `depth_registration_status=ok`, positive finite
  `depth_registered_m`, and a finite positive `depth_used_m` matching within
  relative and absolute tolerance `1e-6` when present. Canonical observations
  use the actual `camera_calibration`, `tracking_model_manifest`, and
  `pipeline_config` roles; unavailable calibration is non-qualifying.
- **Decision:** The source checksum covers the exact wall-clock acquisition
  start/end. Observed samples must fall inside that closed interval. Capture
  time may precede its start only by a fixed two-second allowance for bounded
  receive and processing latency, and may never follow its end. Runtime
  ownership independently places the acquisition interval inside the inspected
  container lifetime.
- **Decision:** The sealed transcript is an explicit semantic projection.
  Nested identity mappings are exact-projected; sensitive encoded/mixed
  embedding shapes, nested 256-value tensors, and authentication material are
  rejected before persistence. A failure seals only a bounded redaction marker,
  never the offending bytes. Live-runner acceptance exactly replays the
  owner-private report, projected source, acquisition window, and identity
  snapshot.
- **Rationale:** Semantic-v2 was never promoted live and its synthetic fixtures
  hid incompatible runtime roles, raw-depth admission, ambiguous tracker reuse,
  observed-time-only joins, duplicate associations, denylist privacy gaps, and
  incomplete runner validation. An explicit v3 contract is more honest than
  silently redefining v2.
- **Validation:** Focused semantic, runtime-world, tracking-continuity,
  empty-frame, runner, and ownership producer suites cover the discovered false
  passes and exact v3 replay. Fresh occupied runtime evidence remains required.
- **References:** `noesis_core/tracking_continuity.py`,
  `noesis_core/world_service.py`,
  `DS9/scripts/ds9_semantic_observation_smoke_test.py`,
  `DS9/scripts/ds9_live_validation_runner.py`,
  `DS9/scripts/validate_runtime_ownership.py`.

# 2026-07-12 — Semantic v3 authority is lane-neutral and receipt-bound

- **Area:** DS8/protected-V3DT/DS9 tracking lifecycle publication and DS9
  semantic ownership replay.
- **Decision:** Keep processed tracker continuity separate from committed public
  presence. A tombstone names the last successfully published presence, not a
  later rate-limited frame. Tracking sequence, lifecycle presence, publish-gate
  timestamp, and published count advance only after the canonical tracking
  admission succeeds. A failed transition remains due; each retry retains the
  fixed last-published presence and regenerates its disappearance frame/time.
  Once canonical admission succeeds, a later auxiliary side-effect failure
  cannot revoke that receipt.
- **Decision:** A partial acquisition may accept and count a tombstone without
  an in-window predecessor only on the first received tracking frame for that
  source. Every later tombstone must exactly match an in-window published
  lifecycle presence. Acquisition bounds cover every received tracking frame,
  including empty frames, and every canonical observation publication clock
  must satisfy `observed_at_us <= published_at_us <= acquisition_finished_at_us`.
- **Decision:** All raw WebSocket, report, source-replay, identity-snapshot, and
  ownership JSON uses the shared unique-key, finite-number UTF-8 decoder before
  projection or model validation. Duplicate-key source rejection is
  checksum-covered without retaining the key or payload.
- **Decision:** The behavior authority is the lane-neutral
  `semantic_gate_v3`, valid for baseline and V3DT. Baseline ReID promotion
  requires both `reid_open_set_occupied_v1` and `semantic_gate_v3`; neither
  evidence document can stand in for the other. Any ownership-attached runner
  writes the semantic report, source, and exact identity snapshot directly to
  the checksum-covered launcher directory. Post-gate dispatch supplies only
  identity arguments to ReID validation and supplies the reviewed pipeline plus
  exact identity-snapshot path to semantic validation.
- **Rationale:** Processed frames are not downstream facts; treating them as
  published created unverifiable tombstones and allowed failed transitions to
  consume continuity state. Partial live capture needs one explicit boundary
  exception, not a permanent missing-predecessor escape. Neutral semantic
  ownership prevents baseline ReID promotion from claiming live multimodal
  completeness with identity evidence alone.
- **Validation:** Shared strict-JSON, semantic-gate, ownership-producer,
  live-runner, tracking-continuity, publisher, empty-frame, DS9 parity, and
  runtime-world focused suites pass. Fresh occupied baseline and V3DT runtime
  evidence remains required.
- **References:** `noesis_core/strict_json.py`,
  `noesis_core/tracking_continuity.py`, `noesis/telemetry/publishers.py`,
  `DS9/scripts/ds9_semantic_observation_smoke_test.py`,
  `DS9/scripts/ds9_live_validation_runner.py`,
  `DS9/scripts/validate_runtime_ownership.py`.

# 2026-07-12 — Behavior acceptance replays producer evidence instead of trusting reports

- **Area:** DS9 behavior evidence, live-runner post-validation, and runtime
  ownership promotion.
- **Decision:** Identity shadow, V3DT world, Wholebody occupied, and Wholebody
  decoded-media producers each export one strict sealed-evidence validator.
  Validators read only owner-private single-link bounded files with canonical
  names, reject duplicate-key/non-finite JSON, verify source digest and exact
  session/lane/runtime identity, reconstruct the canonical source, rerun the
  producer analyzer, and require exact canonical report equality. V3DT also
  rebuilds and compares the live config/launch-plan binding.
- **Decision:** Identity source evidence advances from v1 to v2. V2 seals the
  exact gate policy and projected before/after health snapshots needed to
  recompute health and claim rows. V1 cannot prove those claims and receives no
  compatibility route. V3DT world v2, Wholebody occupied v2, and decoded-media
  v1 remain version-stable.
- **Decision:** The live runner invokes producer validators for ReID, semantic,
  floorplan, V3DT world, Wholebody occupied, and Wholebody media (the latter
  behind the `webrtc` step). It validates only after a successful, non-timeout,
  zero-return-code subprocess and appends `exact_source_replay=pass` only after
  exact replay. Authenticated capability/WS health and supervisor launch/runtime
  identity JSON use the shared strict decoder before existing contract checks.
- **Rationale:** A report and matching source hash can still collude if the
  runner trusts report fields or if the source omits inputs used to derive a
  claim. Producer-owned replay closes that gap and prevents stale files from a
  failed subprocess from being mistaken for fresh evidence.
- **Validation:** Adversarial tests cover report mutation, source mutation,
  duplicate keys, non-finite input, schema/version drift, real V3DT
  config/launcher rebinding, all runner dispatch branches, and failed/timeout
  suppression. Focused producer, runner, ownership, registry, and promotion
  suites pass without live runtime or GPU use.
- **References:** `DS9/scripts/ds9_identity_shadow_live_gate.py`,
  `DS9/scripts/v3dt_world_contract_smoke_test.py`,
  `DS9/scripts/wholebody49_occupied_scene_smoke_test.py`,
  `DS9/scripts/wholebody49_media_decode_gate.py`,
  `DS9/scripts/ds9_live_validation_runner.py`,
  `DS9/scripts/validate_runtime_ownership.py`.

# 2026-07-11 (local) — Inline BEV authority and publication are exact pairs

- **Area:** DS8/protected-V3DT/DS9 BEV telemetry, floorplan readiness, and live
  acceptance
- **Decision:** Keep inline floorplan BEV locked to
  `camera_local_ground_m`/meters. Every BEV frame carries exact source ID, frame
  ID, and observation time, and every current footpoint repeats the frame ID.
  Validation joins that exact cohort and independently applies the calibrated
  world-to-camera transform, registered-depth unprojection, or image-ray/floor
  intersection declared by `displaySource`; it never directly subtracts
  canonical `backend_world_m` coordinates from camera-local coordinates.
- **Decision:** A configured active-floorplan provider is the sole bounds
  authority. Before its first valid record, `None` records
  `startup_pending` and emits no camera-local BEV/status or fatal callback; it
  does not select config bounds or auto extents. Invalid provider data fails
  immediately, and any missing/invalid authority after readiness is fatal with
  state `lost`. A successfully emitted exact empty frame is still a renderer
  success and `active_ready`; `inactive_ready` means no exact BEV success yet,
  not empty occupancy. The replayable DS9 floorplan behavior gate is v4 and
  retains N/N exact fresh/cache-only floorplan and active-registry checks.
- **Decision:** While BEV is configured, tracking and BEV use one pair-safe
  interval: `max(selected tracking interval, BEV interval)`. Tracking publishes
  first and BEV runs only after that same-frame publication succeeds. Count
  transitions and tracker-lifecycle/key-set changes force both pair members;
  tracking failure suppresses BEV. Without BEV, the tracking cadence is
  unchanged.
- **Decision:** Registered-depth admission is shared by canonical observations,
  BEV production, and the independent parity oracle: both statuses must be
  `ok`, `depth_registered_m` must be positive and finite, and a present
  `depth_used_m` must match it within relative/absolute tolerance `1e-6`.
  Protected V3DT also scales every source-frame image anchor/candidate/bbox into
  the active calibration image size before BEV projection, matching DS8 and
  DS9.
- **Rationale:** Independent tracking/BEV gates could publish an unjoinable BEV
  frame above 15 fps. Treating the empty startup registry as a fatal floorplan
  loss could terminate the runtime before a capture could establish authority,
  while reusing config bounds would conceal the same defect. Finally, loose
  depth admission and unscaled V3DT pixels could make validation disagree with
  canonical world semantics. One authority lifecycle, one exact publication
  pair, and one depth/image-space contract make the display replayable.
- **Validation:** Focused root BEV/depth/parity suites passed (115), focused DS9
  parity suites passed (112), and the pair-safe root/DS9/V3DT suites passed (81
  root plus 101 isolated DS9). Python compilation and Ruff checks were clean
  apart from repository-preexisting hook import-order suppressions. Fresh native
  runtime promotion remains a separate gate.
- **References:** `noesis/telemetry/bev.py`,
  `noesis_core/depth_contract.py`, `noesis_core/runtime_health.py`,
  `noesis_core/tracking_continuity.py`,
  `noesis/pipelines/hooks_v3dt_reimpl.py`,
  `scripts/menon_bev_track_parity_smoke_test.py`,
  `DS9/scripts/ds9_floorplan_live_gate.py`.

# 2026-07-12 — MapAnything engines require functional admission before install

- **Area:** DS9 MapAnything TensorRT maintenance and realization
- **Decision:** Select correctness-first FP32 for the next DS9 MapAnything
  engine and omit a precision flag entirely. Before `install_candidate`, run
  exactly one real TensorRT inference against a tracked-hash, owner-private,
  identical-batch fixture and seal a receipt binding candidate bytes, fixture,
  exact command, build/runtime platform, float32 IO, finite and positive
  counts, confidence sentinel rejection, mask coverage, distribution envelope,
  batch consistency, and raw output evidence. The host finalizer and direct
  realization reconciler must independently revalidate that receipt and the
  installed bytes before publishing realization.
- **Rationale:** The prior FP16 engine passed build and independent
  deserialization but emitted all-NaN depth, an all-zero mask, and invalid
  confidence sentinels. The isolated FP32 diagnostic was finite, positive,
  dense, and batch-consistent; BF16 was finite but materially farther from the
  DS8-good reference. Deserialization therefore proves compatibility, not
  function. Placing the gate before atomic replacement prevents a broken plan
  from becoming canonical, while revalidation closes forged-receipt and direct-
  reconcile publication paths. FP32 functional admission still does not claim
  DS8 parity or live multi-camera quality.
- **Validation:** Focused source, admission, wrapper, finalizer, reconciliation,
  and adversarial numeric tests cover empty precision argv, skip-inference-only
  proof, NaN/Inf, confidence sentinels, zero masks, envelope drift, batch
  divergence, changed receipt/engine hashes, prior-engine preservation, and a
  valid atomic install. Guarded transaction `20260712T052425739603Z` then
  installed the 3,883,865,652-byte engine at SHA-256
  `eabc1169c7d725ed7cff171ed54c402c4282fdcf23b87a58f4e41a95fe23ecc8`.
  Real batch-three inference produced finite positive depth/confidence and
  99.9954% mask coverage; both loads passed, and 8,038 NVML samples observed a
  4,655 MiB peak with a 29.100129 ms maximum gap. Final realization
  `6fab7d456c031490f640ee2c3ce5a38922a96ed86a965020ca3051820306dce4`
  passes canonical, V3DT, and Wholebody49 file/provenance validation. Live
  multi-camera depth/floorplan quality remains separately gated.
- **References:** `DS9/scripts/rebuild_engines.py`,
  `DS9/scripts/engine_maintenance_common.py`,
  `DS9/scripts/finalize_engine_realization.py`,
  `DS9/scripts/reconcile_engine_provenance.py`,
  `DS9/docs/MapAnything_Engine_Parity_Plan.md`,
  `DS9/tests/test_mapanything_engine_quality_gate.py`.

# 2026-07-12 — Canonical publication is release-gated by committed authority

- **Area:** DS8/protected-V3DT/DS9 tracking, global world, BEV, WebSocket
  lifecycle, and shutdown
- **Decision (tightening the 2026-07-11 exact-pair decision):** Cross-thread
  WebSocket sends return typed sender-admission receipts or raise typed
  lifecycle/capacity failures. Admission is finite by in-flight count,
  batch-message count, and encoded bytes. Public JSON is finite-only and is
  encoded once into immutable bytes before receipt; post-receipt caller mutation
  cannot change delivery. A canonical tracking/world/event batch receives an
  unresolved one-shot authority gate: sender admission reserves ownership but
  no client delivery may begin until `commit_then_release()` succeeds.
  Canonical tracking/world/event/BEV types bypass the generic latest-only
  coalescer even if a generic interval is configured for their type.
- **Decision:** The 256-submission queue also owns one 256 MiB global in-flight
  byte budget, so worst-case 32 MiB batches cannot multiply into multi-gigabyte
  retention. Exact frozen bytes are reserved atomically with scheduling and
  released once on success, failure, explicit gate abort, or cancellation;
  shutdown waits for every unresolved gate and requires zero reserved bytes.
  Aborted submissions/bytes are separately observable. Authenticated telemetry
  fanout is separately bounded to 8
  clients by default and hard-clamped to 1–16. Excess clients close with `1013`
  / `telemetry_capacity_reached` before registration or initial snapshots;
  `/healthz` is excluded from that set and remains available at capacity.
- **Decision:** A tracking frame, its canonical world snapshot, and all world
  events are prepared under one publication lock and admitted as one ordered,
  non-coalesced batch. `CanonicalWorldService.prepare()` forks isolated
  copy-on-write fusion state without changing authoritative fusion, sequences,
  lifecycle events, or journal. The runtime uses the synchronous
  completion-proven `ContractJournal`; `commit()` requires an exact append
  count before installing its owner-bound candidate and rejects retention that
  cannot hold the complete current cohort. `AsyncContractJournal` is rejected
  at this authority boundary. Only then does the outbox release the
  frozen batch. Abandoned candidates are hard-bounded to 16 per service.
  Pre-admission failure discards its candidate and preserves sequence zero for
  an exact retry. Journal/authority failure after admission aborts the gate so
  no client sees the candidate, then poisons the publisher because commit
  outcome may be uncertain. A release failure after successful commit also
  poisons, preventing an unannounced successor.
- **Decision:** Fusion authority is service-private. The service detaches an
  injected fusion instance, exposes no mutable fusion object, and returns only
  its cached immutable last-committed snapshot. Reads cannot consume snapshot
  sequence, expire entities, clear sources, or bypass the service revision and
  journal.
- **Decision:** Canonical routing is exclusive. The gated route accepts only a
  tracking-first batch whose optional separate snapshot/events exactly match
  the embedded cohort; generic sync/batch/targeted/async/coalescer routes reject
  tracking, world snapshot/event, and BEV. BEV uses its own dedicated typed
  receipt route after tracking commit. The gate callback is an in-process trust
  boundary rather than a cryptographic capability, so a static regression
  permits production gated-API references only in the shared WebSocket owner
  and byte-identical DS8/DS9 tracking publishers. Generic routing inspects raw
  text/UTF-8 binary top-level JSON types, while server-frozen envelopes bind an
  owner token, exact encoded byte count, and matching declared/encoded type;
  neither raw JSON nor caller-forged frozen values bypasses the table.
- **Decision:** Tracking lifecycle/tombstone state, cadence gates, and health
  progress advance only after the typed tracking receipt. Paired BEV receives
  that exact source/frame/time/sequence/submission cohort and returns one of
  `admitted`, `startup_pending`, or `failed`; an admitted BEV submission must
  follow the tracking batch. Gate release is not a client-delivery ACK and does
  not make multiple WebSocket frames transport-atomic.
- **Rationale:** Mutating world or lifecycle authority before a best-effort
  cross-thread handoff created irrecoverable phantom state, while generic
  coalescing could create sequence gaps and split tracking/world/BEV cohorts.
  An opaque prepare/commit transaction plus a release-gated, exactly bounded
  outbox prevents both phantom world state and the inverse race where the event
  loop exposes an admitted candidate before its journal commit. Synchronous
  journal acknowledgement makes runtime retention a prerequisite for release;
  the throughput cost is intentionally paid in the publication critical path
  instead of hiding late disk failure behind visible authority.
- **Performance:** The fusion fork copies only mutable container layers and
  shares immutable contract values. On the reference host, a 64-entity,
  four-camera preparation measured about 8 ms median with about 1.4 MiB peak
  traced allocation; a regression gate requires median below 30 ms and peak
  below 8 MiB. At the pair-safe approximately 12 Hz/source cadence, three
  serialized sources have an aggregate 27.8 ms service period. A representative
  six-person-per-camera three-source synchronous
  prepare+journal+authority cycle measured about 24.4 ms median locally and is
  gated below 27 ms. Maximum-capacity 64-entity preparation remains a stress
  contract, not an admitted steady-state journal load. Sync JSON regression
  tests also require exactly one encoder pass.
- **Validation:** Focused world/service, WebSocket boundary, BEV, V3DT, DS9
  empty-frame, and DS9 parity suites cover rollback/retry, owner/identity-bound
  candidates, commit-gated release, commit abort with zero delivery, gate
  cancellation, shutdown waiting, exact byte release/abort accounting,
  synchronous journal acknowledgement, private non-mutating snapshot reads,
  finite-only single/batch/targeted sends, mutation isolation, exact
  order/cohort, startup pending, and failed receipts. Fresh live runtime
  validation remains required.
- **References:** `websocket_server.py`, `noesis_core/world/fusion.py`,
  `noesis_core/world_service.py`, `noesis/telemetry/publishers.py`,
  `noesis/telemetry/bev.py`, `noesis/pipelines/hooks.py`,
  `noesis/pipelines/hooks_v3dt_reimpl.py`,
  `DS9/noesis/pipelines/hooks.py`.

# 2026-07-12 — Behavior evidence is immutable and canary-bound

- **Area:** DS9 behavior producers, runtime ownership, V3DT, Wholebody49, and
  semantic-v3 promotion evidence
- **Decision:** Every behavior source, snapshot, and report is a bounded
  owner-private create-if-absent publication. The shared primitive writes and
  fsyncs a same-directory temporary inode, publishes with a no-replace hard
  link, fsyncs the parent, and validates owner, mode, inode, size, and link
  count. Producers preflight the complete canonical filename cohort and publish
  sources before reports. Existing outputs or interrupted residue are never
  replaced, chmodded, or completed; the operator must start a fresh session.
- **Decision:** Every registered behavior report and replay source, including
  resource-soak raw samples, must equal the exact encoder owned by its producer,
  not merely parse to an equal object. Semantic-v3 identity JSONL rows carry the
  same exact-byte rule.
  Wholebody ownership independently derives the complete lane source inventory
  from reviewed pipeline/camera authorities and requires exact typed ordered
  equality across launch plan, report, observations, and source. V3DT ownership
  derives every session path and the ephemeral build root independently, then
  cross-checks the launch plan and inspected mount before replay.
- **Decision:** `runtime_session` promotion is bounded ephemeral-canary
  evidence only. Selector-bound `appliance-run` has persistent build/state and
  an indefinite lifecycle, so it is rejected explicitly and remains a separate
  deployment-health contract.
- **Rationale:** Replace-on-write let reruns erase evidence, self-declared
  source subsets could satisfy Wholebody, and launch-plan text could redirect
  V3DT replay to an alternate build tree. Object-equal JSON also allowed raw
  encoding drift. Closing all four boundaries makes one checksum-covered
  session an auditable fact rather than a producer assertion while preserving
  the appliance/canary lifecycle distinction.
- **Validation:** Private publication races, existing symlink/hard-link and
  residue, parent mutation, partial-bundle rerun, all six producers, non-finite
  payloads, semantic whitespace/key-order/number spelling, Wholebody
  subset/superset/reorder/boolean for S and X, alternate/resealed V3DT paths,
  source-ID drift, cross-validator whitespace/key-order reseals, and appliance
  rejection are covered by focused tests. No
  runtime, container, GPU job, registry mutation, or protected identity-state
  mutation was performed.
- **References:** `noesis_core/private_paths.py`,
  `DS9/scripts/validate_runtime_ownership.py`,
  `DS9/scripts/ds9_semantic_observation_smoke_test.py`,
  `DS9/scripts/v3dt_world_contract_smoke_test.py`,
  `DS9/scripts/wholebody49_occupied_scene_smoke_test.py`,
  `DS9/docs/runtime_ownership_evidence.md`.

# 2026-07-12 — Native DS8 parity canaries own an external mutable-state cohort

- **Area:** DS8/protected-V3DT lifecycle validation, immutable checkpoints,
  calibration, identity continuity, and runtime secrets
- **Decision:** `scripts/ds8_runtime_30s_gate.py` requires one explicit absolute
  external state root. The root must be an owner-owned mode-0700 real directory,
  separate from the checkout and operator home, outside every sealed checkpoint,
  and either empty or marked with the exact canary-state contract. A marked root
  may be reused for sequential baseline/V3DT runs; initialization validates
  canonical analytics/calibration seed bytes but never clears persisted
  household gallery, resident/visitor, identity-v2, world, or scene state.
  Logs and verdicts are private create-once files with caller-unique names.
- **Decision:** The launcher builds the child environment from a narrow
  OS/GPU/media allowlist instead of inheriting the operator session. It resolves
  the camera registry, MapAnything RPC key, and internal bearer file paths before
  isolating HOME and passes paths only. All three files must already be
  owner-private; bearer loading cannot create or repair authority. Mutable HOME,
  Python user base, XDG/TMP/CUDA/GStreamer caches, generated build artifacts,
  depth/floorplan storage, analytics YAML/INI, calibration/alignment/audit,
  StableID/identity-v2, world journal, scene database, virtual-twin root, and
  diagnostics are rooted in the canary cohort. Runtime argv always includes the
  exact external `--storage-base`.
- **Decision:** Baseline DS8, protected V3DT, and DS9 honor the same explicit
  calibration/alignment path variables. Protected V3DT also routes its generated
  effective pipeline and GStreamer registry through the external build/state
  authorities while retaining the reviewed repo-owned patched tracker binary as
  an immutable input.
- **Rationale:** The prior canary could mutate `build/`, `data/depth`, the native
  exclusion INI, calibration JSON, GStreamer registry, and operator household
  identity merely by starting. That made evidence from a sealed source tree
  non-reproducible and could silently alter the family's protected gallery.
  Explicit state ownership preserves immutable-source provenance and allows the
  V3DT-to-baseline identity continuity proof without borrowing production state.
- **Validation:** Focused state-root, environment, secret-path, load-only token,
  calibration parity, V3DT build-path, lifecycle, and existing canary tests pass
  without starting DeepStream, a container, or a GPU workload. Because the
  runtime source contract changed, final native validation must use a newly
  sealed checkpoint and supersede the canonical graph before promotion.
- **References:** `scripts/ds8_runtime_30s_gate.py`,
  `tests/test_ds8_runtime_30s_gate.py`, `noesis/ds8_runtime.py`,
  `noesis/ds8_runtime_v3dt_reimpl.py`, `DS9/noesis/ds9_runtime_core.py`,
  `docs/DS8_testing_guide.md`.

# 2026-07-12 — The release-gated world journal uses durable WAL storage contract v2

- **Area:** Canonical world authority, synchronous publication release,
  private runtime state, restart admission, and three-camera cadence
- **Decision:** `ContractJournal` owns one persistent SQLite connection for its
  process lifetime. All access remains serialized by its existing reentrant
  lock, and cross-thread connection use is enabled only behind that lock. The
  exact connection profile is WAL, `synchronous=FULL`, and a 1,000-page
  autocheckpoint. Every operation rechecks that profile. Shutdown requires an
  exact `(0, 0, 0)` truncate-checkpoint receipt, closes the connection even on
  failure, and makes a close failure sticky.
- **Decision:** Storage contract v2 binds the main database and private
  WAL/shared-memory cohort by owner, exact mode, link count, parent, and inode.
  Startup performs SQLite integrity checking, exact schema/table/index
  validation, and a complete replay of the retained state, contiguous
  sequences, anchor boundary, timestamps, canonical JSON, contract column,
  model normalization, and hash chain. A validated v1 `DELETE` journal may
  migrate once: WAL is selected, FULL synchronization is confirmed, and v2 is
  stamped transactionally. An interrupted v1/WAL migration is retryable. A v2
  database not already in WAL, a corrupt v1 database, or an unrelated version-0
  database is rejected without repair or adoption.
- **Decision:** An append receipt means every named record remains retained at
  commit. Batches larger than record capacity and batches whose own timestamp
  span would prune their prefix roll back without an acknowledgement. Any
  `BaseException` rolls back; rollback failure closes and poisons the journal
  because transaction outcome is uncertain. Async persistence remains
  forbidden at canonical world authority.
- **Rationale:** The previous open/commit/close rollback-journal transaction
  paid three storage-sensitive commit paths for each serialized three-camera
  aggregate. On the reference host its median moved between roughly 27 and
  37 ms under ordinary storage variance, crossing the 27 ms cadence gate. WAL
  with FULL synchronization preserves completion-proven durability while
  removing rollback-journal churn; it does not defer acknowledgement, weaken
  synchronization, or batch visible authority asynchronously. Explicit v2
  admission prevents that performance change from becoming silent repair on a
  later restart.
- **Performance:** The test warms three aggregate cycles and measures 31. Both
  median and arithmetic mean remain gated below 27 ms. Twenty independent
  local runs passed both gates; the worst mean was about 17.9 ms and the worst
  median about 17.6 ms. Maximum filesystem latency is intentionally a live-soak
  concern rather than a flaky unit assertion.
- **Validation:** Focused journal, private-path, runtime-world, and world-service
  suites cover v1 migration and v2 restart, corrupt/foreign/no-repair
  admission, physical corruption normalization, exact schema/history/state,
  GENESIS/pruned-anchor relations, private sidecars and inode swaps,
  cross-thread serialization, unclean-process recovery, retention ACKs,
  BaseException rollback and rollback-failure poison, exact sticky close, and
  the longer cadence gate. Fresh live DS8/DS9 soak evidence remains required.
- **References:** `noesis_core/journal.py`,
  `noesis_core/world_service.py`, `tests/test_noesis_core_journal.py`,
  `tests/test_noesis_core_world_service.py`, `docs/DS8_testing_guide.md`.

# 2026-07-12 — Reviewed inference sources are portable and mutable console state follows the build authority

- **Area:** DS8/DS9 inference configuration, immutable source checkpoints, and
  dev-console materialization
- **Decision:** Canonical baseline and supported alternate pipeline YAMLs point
  at tracked, reviewed DAv2 source INIs under their owning `pipelines/` trees.
  Those source configs use repository-prefixed relative model paths. Production
  continues to derive an engine-only nvinfer config below `NOESIS_BUILD_DIR`;
  the reviewed source is never replaced by or made dependent on an ignored
  generated build file. Active DS8 model, label, and parser references use the
  same repository-relative resolver contract.
- **Decision:** Dev-console launch artifacts, activity history, and saved
  profiles share the existing `NOESIS_BUILD_DIR` authority, with the historical
  repository `build/` directory only as the default. OSD mask/box policy is a
  total mapping of the selected reviewed profile and size: segmentation and
  Wholebody49 S expose masks, while detection and Wholebody49 X do not. File
  existence in a generated build directory cannot change that policy.
- **Rationale:** A sealed checkout could previously require a generated DAv2
  INI containing another checkout's absolute paths and could still write
  dev-console state into its own source tree. Detect profiles also inherited
  the base segmentation OSD policy whenever their generated INI was absent.
  One reviewed source plus one explicit mutable build authority makes source
  provenance reproducible and removes both hidden environment dependencies.
- **Validation:** Focused pipeline, dev-console, and inference-runtime tests
  passed (153), including a canonical graph build from an empty external build
  root and inspection of derived pose, DAv2, and MapAnything engine-only
  configs. DS9 static preparation passed without the old generated config.
  YAML/INI parsing, active-path scans, Python compilation, and focused lint
  passed; live DeepStream/GPU behavior was not changed or launched.
- **References:** `pipelines/config_infer_secondary_depth_tracking_da2.ini`,
  `DS9/pipelines/config_infer_secondary_depth_tracking_da2.ini`,
  `noesis_core/inference_runtime_contract.py`,
  `noesis/dev_console/paths.py`, `noesis/dev_console/materialize.py`.

# 2026-07-19 — Registration rejection is scoped to metric depth

- **Area:** DS8/DS9 baseline person-ground world placement and Menon trails.
- **Decision:** A rejected registered-depth sample is never projected, clipped,
  or used for height lock. It does not invalidate a separate calibrated
  floor-ray candidate when the startup-bound camera policy has positive floor
  weight and explicitly permits floor-only placement. Depth-required profiles
  remain fail-closed. DS9 owns the same small prediction wrapper as DS8 around
  the shared physical filter.
- **Rationale:** Live successor evidence showed the living-room floor candidate
  was valid but suppressed by an unrelated depth-domain rejection, and then
  exposed the DS9 adapter calling a prediction wrapper it had not defined.
  Scoping rejection to the failed observation preserves the evidence-backed
  camera policy; restoring adapter parity prevents valid world updates from
  disappearing at the callback boundary.
- **Validation:** Focused allowed/required policy tests and seven cross-runtime
  person-ground parity tests pass. Final bounded DS9/Menon acceptance is stored
  with the release evidence.

# 2026-07-19 — Strict world provenance keeps invalid decisions and derives rooms from admitted evidence

- **Area:** Shared DS8/DS9 canonical observations, synchronous world journal,
  global fusion, and privacy-safe alignment capture.
- **Decision:** `noesis.observation.person` v1 gains an optional bounded
  `world_diagnostics` object. The canonical service emits it for invalid world
  observations and whenever producer diagnostics exist, retaining only finite
  floor/depth candidates, filter decisions, policy weights, and depth status.
  An invalid observation always receives an explicit first-divergence reason;
  diagnostics never promote a rejected point into fusion.
- **Decision:** World source evidence preserves the exact strict observation ID,
  source zone label, `zone_source`, and explicit authority. Only an
  `nvdsanalytics_roi` label is spatially authoritative and eligible to derive
  `WorldEntity.room_id`; a camera-name fallback remains diagnostic for
  occupancy/dwell but never votes on room. Rejected positional sources do not
  vote; disagreement among accepted authoritative sources yields a null room
  and a visible conflict. Owner-private alignment captures replace raw
  observation IDs with stable per-run ephemeral keys, preserving exact joins
  and zone provenance without persisting runtime or identity-bearing
  identifiers.
- **Rationale:** A missing Menon marker was previously indistinguishable from a
  transport/render failure once the strict observation discarded the producer's
  invalid-world evidence, and canonical entities lost their available room
  semantics. The additive evidence makes the first failed geometry stage and
  source-to-snapshot lineage deterministic while retaining the existing
  fail-closed fusion and privacy boundaries.
- **Validation:** Focused observation/schema, fusion, world-service/journal,
  sanitizer/privacy, and DS9 world-snapshot tests pass; no runtime or service
  was restarted.

# 2026-07-24 — Existing overcrowding membership is the canonical room designation

- **Area:** Shared DS8/protected-V3DT/DS9 analytics adapters, canonical
  observations, world fusion, and Menon room occupancy.
- **Decision:** The established post-tracker overcrowding ROI is the single
  household room designation. NVIDIA attaches its label per object in
  `ocStatus`; the frame `object-threshold` controls overcrowding state, not
  object membership. One shared DS8/V3DT/DS9 resolver therefore uses exactly
  one unique nonempty `ocStatus` label first and accepts a unique `roiStatus`
  label only when no overcrowding membership exists. Identical duplicates are
  harmless; padded, empty, overlength, malformed, or distinct competing labels
  fail closed without trimming, normalization, truncation, or first-wins
  selection. No duplicate room polygon taxonomy is introduced.
- **Decision:** `PersonObservation`, `WorldSourceEvidence`, and the canonical
  world service preserve exact zone bytes and provenance. `WorldEntity`
  validates its derived room against accepted authoritative source rows:
  rejected, camera-default, and unprovenanced rows do not vote; one exact label
  requires the same non-null `room_id`; no vote requires null; and competing
  labels require null plus `conflict=true`.
- **Rationale:** Existing room occupancy already has the correct per-object
  polygon label. Reusing that evidence joins aggregate occupancy and entity
  membership without a second editable geometry definition, while the strict
  resolver and entity validator prevent ambiguous or normalized identifiers
  from becoming spatial authority.
- **Validation:** Focused analytics resolver/config/adapter, observation/schema,
  direct entity-contract, fusion, world-service, validation-fixture, and DS9
  parity tests pass; no runtime or service was restarted.

# 2026-07-19 — Dewarped resolution and Menon camera anchors are explicit geometry authorities

- **Area:** DS8/DS9 person-ground projection, protected V3DT parity, Menon
  diagnostic rays, and `backend_world_m` to `menon_scene` registration.
- **Decision:** Every producer uses the calibrated dewarped output resolution
  declared by `CameraIntrinsics`; a principal point is never interpreted as an
  image extent. Menon ray and reprojection diagnostics likewise require an
  explicit image size and fail closed when it is absent. The three active
  cameras declare `1920x1080` in both runtime and public scene calibration.
- **Decision:** The global scene similarity is rebuilt from the current Menon
  camera-device anchors and backend camera centers. The fit carries exact
  camera-anchor and camera-calibration SHA-256 bindings plus metric RMSE and
  worst-anchor residuals. The calibration manager rejects a mismatched anchor
  digest or a fit above its declared 5 cm limit; DS8 and DS9 use identical
  parsing and validation. Menon's diagnostic camera centers are synchronized to
  the same bound anchor state.
- **Rationale:** The former `2*cx,2*cy` inference rescaled valid foot pixels and
  moved floor intersections by meter-scale distances. The former similarity
  also retained an obsolete kitchen anchor and missed current camera devices by
  58.46 scene units RMSE. These are geometry-authority defects and must not be
  hidden with renderer clamping, room fitting, or presentation smoothing.
- **Validation:** Focused projection/parity, scene-registration, calibration
  manager, authored-scene, production-renderer, and Menon coordinate tests pass.
  The admitted camera-anchor fit is 2.35 cm RMSE and 2.94 cm worst residual.
  Physical person-position admission remains a separate live/guided-waypoint
  gate.

# 2026-07-26 — RF-DETR detection uses every-other-frame cadence

- **Area:** DS9 RF-DETR 1.8.3 detection PGIE selection and downstream GPU
  headroom.
- **Decision:** Materialized RF-DETR detection configs require `interval=1`.
  The attestation validator rejects cadence drift. This applies to every
  selected RF-DETR detection size and does not alter the active YOLO baseline.
- **Rationale:** The cadence matches the selected YOLO26-M baseline and keeps
  the architecture comparison free of a reinference-rate confound. In the
  corrected three-camera peer gate, RF-DETR Medium at this cadence added only
  2.34 GPU-utilization percentage points and 4.14 W mean power while preserving
  depth-device throughput in an empty scene. In the matched-path guided
  occupied gate it added 1.28 GPU-utilization percentage points and 4.28 W mean
  power, with a ten-point higher utilization maximum. Every-frame RF-DETR
  inference is not selected without separate evidence that its added work
  preserves the headroom needed by tracking, ReID, pose, depth, MapAnything,
  mosaic, and telemetry stages.
- **Validation:** RF-DETR Medium loaded its exact FP16/TF32 engine and current
  attested parser on all three live feeds; bridge, RTSP DESCRIBE, hardware
  decode, strict zero-copy, explicit counter-delta, and runtime-error gates
  passed. The first occupied RF-DETR attempt used a different, phone-interrupted
  route and is invalid for comparison. In the replacement matched-path arm,
  RF-DETR produced 17.73% fewer person rows, 26.92% fewer fresh embedding rows,
  and only two living-room identity-journal rows versus YOLO's 19. The evidence
  is not frame-labeled ground truth, so it supports retaining YOLO26-M rather
  than claiming a detector-accuracy ranking. The generation-221 selector-driven
  YOLO26-M baseline was restored. The earlier Large-gate cumulative counter
  maxima are not treated as throughput deltas.

# 2026-07-28 — Lower-body occlusion makes gravity-drop authoritative

- **Area:** Shared DS8/DS9 baseline person-ground tracking and BEV/OSD trails.
- **Decision:** Treat feet/ankle, knee, and waist/hip occlusion as explicit
  standing-person states rather than waiting for the current anchor to become
  invalid. A recent trusted full-body height lock, current upper-body evidence,
  lost lower-body keypoint groups, and detector-box collapse relative to the
  learned upright body and shoulder scale determine entry. Explicit bent-leg,
  sitting, or lying evidence blocks entry and immediately releases an active
  occlusion state. Otherwise, three consecutive clear lower-body updates are
  required before direct pose/depth authority resumes.
- **Decision:** Learn per-track nose, shoulder, and hip height fractions from
  trusted full-body frames. During occlusion, reconstruct floor contact from the
  calibrated bbox-top ray plus consistent visible-body plane rays, reject
  spatial outliers, and publish that reconstructed point as both the world
  observation and `image_foot`. A syntactically valid bbox-bottom or registered
  depth sample on a counter/table edge is demoted while this state is active.
- **Decision:** The static MapAnything floorplan is not used as a live range
  substitute. Camera calibration, the learned upright body model, and the
  existing floor plane remain the runtime geometry authorities.
- **Rationale:** The former gravity-drop path ran only after all current anchors
  failed. Counter and table edges commonly produce valid but physically wrong
  bbox/depth anchors, so tracks could jump despite every fallback being enabled.
  Making the occlusion state authoritative corrects the source selection before
  filtering instead of hiding the jump with additional smoothing.
- **Validation:** 104 focused shared-state, analytics/world, V3DT ground-state,
  and BEV tests pass; 7 DS8-vs-DS9 parity tests pass. Live occupied
  counter-walk validation remains pending.

# 2026-08-01 — Scene priors are site/space evidence and remain shadow-only

- **Area:** MapAnything room-walk reconstruction, authored-scene semantics,
  tracking diagnostics, and floorplan review.
- **Decision:** Model a prior revision by portable `site_id` and physical
  `space_id`, independently of sensor identity. A mutable site catalog binds
  cameras to exact immutable revisions. Authored room labels and OBJ groups
  remain semantic authority; scan geometry supplies measured occupancy and
  height evidence only.
- **Decision:** Store prior grids in `backend_world_m` and resample them through
  current calibrated camera extrinsics when extending a camera-local floorplan
  response. Preserve every live layer unchanged. Composite cells select live
  observed height first and use static evidence only where live observation is
  unknown.
- **Decision:** V1 is explicit `shadow` mode. Per-track prior diagnostics cannot
  mutate, admit, reject, smooth, or replace `track.world`, canonical
  observations, global world state, zone semantics, or lower-body occlusion
  handling. Named objects, doorways, navigation, and tracking correction need
  separate reviewed contracts before promotion.
- **Rationale:** A room scan is durable evidence about a physical space, not a
  camera artifact or a second world authority. Separating space from sensor
  identity supports multiple cameras, repeated room scans, additional rooms,
  and multi-home beta deployment without creating room-specific schemas or
  silently feeding static geometry into live person placement.
- **Validation:** The verified Living Room bundle deterministically produced
  revision `sceneprior_living-room_20260801T152036Z_109fe40d4e18` from 531,368
  selected points with 77.20% authored-cell observation coverage. Five focused
  contract/integrity/composition tests pass, generated schemas are current,
  Python compile checks pass, and the frontend production build succeeds.

# 2026-08-02 — Scene-prior PNG reviews use the verified reference-camera frame

- **Area:** Generic Scene Prior construction and standalone review artifacts.
- **Decision:** Keep the immutable 2.5D grid in canonical `backend_world_m`, but
  render `preview.png` in the source bundle's verified fixed-camera ground
  frame. Calibrated camera-right maps to screen-right and calibrated
  camera-forward maps toward image-up. Derive both axes from the exact camera
  extrinsic composed with the target reconstruction's floor-world correction;
  do not use a room-specific mirror or heuristic image flip.
- **Decision:** Record the calibration and target-metadata fingerprints, camera
  position, ground axes, and local preview bounds in an additive revision
  field. Include that metadata in immutable identity so orientation changes
  produce a new revision while preserving prior bytes.
- **Rationale:** Backend-world +X points nearly opposite the Living Room
  camera's screen-right direction, so the former world-axis PNG appeared
  backwards even though its runtime geometry was correct. Separating review
  orientation from geometry authority makes the artifact intuitive for every
  calibrated room without perturbing tracking or floorplan composition.
- **Validation:** Seven focused scene-prior tests pass. A deterministic Living
  Room rebuild produces camera-right `(-0.973939, -0.226811)` and
  camera-forward `(-0.226811, 0.973939)`; its canonical `grid.npz` SHA-256 is
  identical to the superseded revision, and the new PNG matches the captured
  fixed-camera layout.

# 2026-08-07 — YOLO26 ADE20K remains a semantic utility lane

- **Area:** DS8 three-room scene parsing and YOLO26 semantic model comparison.
- **Decision:** Keep the official YOLO26 ADE20K Nano, Small, Medium, and Large
  models in a separate `testpipelines/yolo26-sem-ade20k/` utility. Do not treat
  their dense class maps as detector PGIE objects or replace the active PGIE.
- **Decision:** Build TensorRT engines for the exact three-room contract as
  static FP16 `FLOAT [3,3,640,640] -> UINT8 [3,640,640]` graphs. Preserve the
  exported class IDs through a custom DeepStream semantic parser and copy only
  selected RGB evidence frames and masks at the final snapshot edge. Resolve
  canonical camera locators in memory through the existing owner-private
  source registry.
- **Decision:** Do not fabricate confidence or probability tensors. The
  exported graph provides one class ID per pixel, so this lane can produce
  dense scene labels and masks but cannot satisfy consumers that require
  per-class probabilities, instance identity, boxes, tracking, or ReID.
- **Rationale:** A static batch exactly matches the three live rooms and avoids
  carrying unused dynamic profiles. Keeping semantic metadata separate from
  detector objects prevents the category error that made the model look like a
  replacement PGIE while still making its ADE20K scene structure available for
  downstream evaluation.
- **Validation:** All four FP16 engines deserialize with the exact input/output
  binding contract and pass short TensorRT runs on the RTX 3060. Each engine
  loaded through DS8 `nvinfer`, captured Living Room, Kitchen, and Family Room
  masks, handled EOS for all three sources, and exited successfully. The
  focused utility tests pass (3 tests), Python compilation and scoped
  whitespace checks pass, and the four mosaics were visually inspected.
- **Follow-up validation (2026-08-08):** The saved raw Living Room fixed-camera
  anchor from MapAnything phone scan `20260802-162254-8bcc7dd7` was verified by
  its manifest SHA-256 and processed through every exact engine. The same frame
  filled all three static-batch positions, and each engine produced three
  identical class maps. Full-resolution overlays preserve the source aspect
  ratio; no phone-walk frame or derived rectified image entered inference.
- **UI integration (2026-08-08):** Replace the unused `oai2-fe` depth-panel
  Stats tab with a **Sem-seg** tab that reads only a copied, checksummed set of
  the saved well-lit Living Room source and Nano/Small/Medium/Large class maps.
  Model switching, opacity, hover inspection, and legend filtering are browser
  display operations; they must not launch semantic inference, request depth,
  or change a class map. The legend uses a stable unique color for all 150
  ADE20K IDs, lists only IDs present in the selected map, and clears an active
  filter either by selecting the same ID again or by an explicit clear action.
- **Rationale:** This preserves the semantic utility lane's separation from the
  detector PGIE while placing its derived view with the other operator-facing
  inference products. Shipping exact saved evidence makes the tab usable when
  the camera/depth runtime is disconnected and prevents a seemingly harmless
  UI interaction from hiding new inference or capture work.
- **Validation:** The frontend contract suite passes 49 tests, the Vite
  production build includes all five saved assets, and a live Chromium smoke
  activated the tab without camera diagnostics, rendered the 1920x1080 source
  and 22-class Large legend, isolated wall class 0, exercised both clear paths,
  and reported no browser exceptions.

# 2026-08-09 — Sem-seg refresh is an explicit serialized diagnostic action

- **Area:** OAI2 Sem-seg viewer, DS8/DS9 runtime utility inference, and Menon
  same-origin diagnostics.
- **Decision:** Preserve saved semantic evidence as the initial viewer state,
  but add one explicit Refresh capture action as the sole inference trigger.
  The selected model always runs as its fixed batch-3 TensorRT engine across
  Living Room, Kitchen, and Family Room; camera/model selection and tab load
  remain display-only.
- **Decision:** Keep engine, parser, and label bytes immutable in the Noesis
  release. Serialize capture requests process-wide, write results under the
  writable runtime-state boundary, validate all three RGB/class-map/masked
  artifacts, and atomically publish latest only after the batch completes.
- **Decision:** Browser writes go through an owner-authenticated Menon action
  with CSRF, idempotency, audit, and exact latest-manifest readback. Noesis
  serves the resulting images through bounded operator-readable GET routes;
  the internal token and filesystem paths never enter browser payloads.
- **Rationale:** The operator needs a fresh view without making ordinary pane
  interactions unexpectedly expensive or hiding inference behind a reload.
  Batch publication keeps the retained fixed engines honest and prevents a failed
  room stream from mixing new and old camera evidence.
- **Validation:** Focused capture-manager, REST boundary, gateway policy/action,
  manual-only frontend invariant, TypeScript, and production build checks pass.
  Promotion additionally requires one live button-triggered three-camera
  capture and browser inspection of the returned class maps.

# 2026-08-09 — Retain only Small and Large semantic flavors

- **Area:** YOLO26 ADE20K utility assets, manual capture API, and Sem-seg UI.
- **Decision:** Retain only the Small and Large fixed batch-3 engines and saved
  evidence. Nano and Medium are removed from runtime assets, model setup, API
  schemas, viewer choices, and writable capture state.
- **Rationale:** Small and Large are the only useful operating points from the
  four-model comparison; retaining the other two adds storage and operator
  choice without useful project value.
- **Validation:** The focused Python and frontend contracts enforce the two-size
  set, both retained engines deserialize under the DS9 TensorRT runtime, and a
  live Small three-camera refresh is the promotion smoke.
