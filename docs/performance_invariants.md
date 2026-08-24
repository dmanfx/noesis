# Canonical pipeline performance invariants

Status: current for the native DeepStream 9.1 baseline, accepted 2026-08-23.

These constraints preserve the occupied-scene behavior of the canonical
three-camera application. They apply to pipeline construction, native bridges,
tracking and identity hooks, world/BEV publication, persistence, integrations,
and mosaic delivery. Exact measured references live in
[`runtime_baseline.md`](runtime_baseline.md).

## Authority classes

Performance policy follows data authority rather than treating every branch as
interchangeable:

1. **Media spine:** decode, dewarp, preprocess, inference, tracking, analytics,
   tiling, OSD, and encode. It must continue when optional work is late.
2. **Canonical spatial publication:** `tracking`, its exact world
   snapshot/events, and the paired `bev-frame`. This is an ordered committed
   cohort and is never generic latest-only traffic.
3. **Optional or reconstructable work:** secondary depth candidates, score-only
   identity evidence persistence, gallery autosave, diagnostics, manual
   reconstruction, and external integrations. This work must bound its own
   backlog and may lose freshness rather than stall the media spine.

## Steady-state hot path

- Keep full video surfaces in NVMM/GPU memory through GPU OSD and NVENC. A
  display boundary does not authorize mapping the mosaic through host memory.
- Always-on callbacks must not perform filesystem/database/network writes,
  process-wide CUDA synchronization, cold CUDA/model initialization, or
  unbounded waits on another thread or queue.
- Allocate reusable device buffers, CUDA streams/events, pinned host buffers,
  and scratch state outside steady-state per-frame work. Pools and queues have
  explicit finite capacity and observable exhaustion/drop counters.
- Extract or convert each native tensor/surface once per frame. Multiple
  consumers share one compact owned result rather than repeating a device-host
  copy, NumPy conversion, or JSON normalization.
- Bound work that scales with detections, tracks, cameras, clients, or retained
  records. An optimization is incomplete if it performs well only in empty
  rooms and develops unbounded tails in occupied scenes.
- Durable and external I/O runs outside the media callback through bounded
  workers. Failure is reported through health/counters and cannot silently
  switch algorithms or block video.

The gated MapAnything capture is the documented exception to the callback-copy
rule. NVIDIA tensor metadata owns a limited lifetime, so an admitted manual
capture may synchronously copy its exact bounded tensor payload before returning
the buffer. That copy remains isolated on the request-gated branch, is measured,
and does not authorize host staging in an always-on path.

## Queue and failure behavior

- The DAv2 observation branch is finite and latest-frame-only. Device readiness
  is query-only; a pending result is retried or replaced by an admissible prior
  observation without waiting on the frame path.
- Optional evidence/persistence queues discard or coalesce stale work according
  to their explicit contract. They never grow without bound.
- Diagnostic Identity-v2 shadow scoring and visitor persistence use one
  isolated worker with at most one pending scalar snapshot per source and 64
  pending sources. A newer pending cohort replaces the older one for that
  source. Copy, capacity, scorer, or persistence failure drops/degrades only
  shadow evidence; it cannot delay, mutate, reject, or stop canonical
  tracking/world/BEV. No raw frame, surface, SDK object, diagnostic row, or
  public embedding may enter this queue. Authoritative identity is not covered
  by this optional policy because it owns same-frame public identity.
- Canonical tracking/world/BEV cohorts do **not** use that policy. Their
  admission is all-or-none, ordered, revision-bound, and fail-closed. A client
  or transport may disconnect, but the producer cannot manufacture a mixed or
  last-seen cohort to hide the failure.
- Queue isolation is not a substitute for a valid downstream contract. If a
  required canonical consumer cannot accept bounded work, surface the error
  rather than adding an unbounded queue.

## Capability preservation

Optimize data movement, synchronization, algorithmic duplication, batching,
caching, and bounded scheduling before reducing capability. Changes to the
selected model, input resolution, inference interval, tracker quality, depth or
pose cadence, or enabled outputs are explicit product/quality tradeoffs—not
transparent performance fixes. They require user approval and matched quality
evidence.

World and ground estimation remain single-authority computations under
ADR-020. Dashboard BEV, OSD trails, Menon, and other views may apply only the
revision-checked transform appropriate to their view; they may not create a
second estimator or smoother to make presentation cheaper.

## What counts as proof

`disable-output-host-copy=1`, NVMM caps, low GPU utilization, an open port, or a
dashboard FPS label is not sufficient proof by itself. Trace the direct native
consumer and measure the layers the change can affect:

- **Input:** per-camera decoded/dewarped FPS, progress age, stalls, and recovery
  attempts.
- **Processing:** affected callback/stage latency including tail values under
  representative occupancy.
- **Encode:** H.264 access units per second, arrival-gap percentiles/maxima, and
  feeder/queue drops.
- **Delivery:** authenticated WebRTC connection plus decoded frame count.
- **Correctness:** detections, tracks, StableID, pose, depth, canonical world,
  BEV, and floorplan behavior affected by the change.

For paced replay, inspect `tracking.publication_worker.pending_total`, its high
watermark and overflow/failure/completion totals, plus
`identity_v2.shadow.pending_sources`, its high watermark, in-flight state, and
completed/coalesced/drop/failure totals. Stage timings
`tracking.publication_worker_item`, `tracking.publish`,
`tracking.publication_worker_queue_wait`, `bev.render_and_publish`,
`identity_v2.shadow_process_source_frame`, and
`identity_v2.shadow_queue_wait` identify processing time separately from
backlog delay. A returned-to-zero queue gauge alone is not proof; the high
watermark and monotonic totals must agree with source-frame progress.

Use matched inputs/config/model realization and sufficient warm-up. Performance
work must include a motion/occupancy-heavy recorded sample and, when practical,
a bounded live multi-person check. Visual review confirms corruption or
macroblocking; source/encoded/decoded measurements establish cadence and drops.

## Change checklist

Before accepting a hot-path change:

1. State which authority class and memory boundary changed.
2. Show every new queue, wait, allocation, copy, synchronization point, and
   external I/O call on that path.
3. Prove finite capacity and failure behavior for optional work.
4. Prove canonical cohort ordering/identity remains exact.
5. Verify models, resolution, inference cadence, tracker, and outputs were not
   silently reduced.
6. Run focused tests, one representative recorded motion check, and the direct
   live/media consumer checks warranted by the change.
7. Remove temporary recordings, profiles, generated configs, and test services;
   leave the requested canonical runtime state explicit.
