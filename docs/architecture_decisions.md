# Architecture decisions

This is the current decision ledger. Historical DS8 and container-era ledgers
remain in the archives for rationale and chronology.

## ADR-001 — One native DeepStream 9.1 runtime

**Accepted:** 2026-08-15

The only supported DeepStream application is DS9.1 on the host through
`DS9/scripts/run_canonical_runtime_host.py`. DS8, DS9.0, and Docker are not
fallbacks. Inert legacy entrypoints remain only until the recorded destructive
cleanup is accepted. The supervisor verifies exact platform, Python, secrets,
artifact, and port authority before launch.

**Why:** Native parity restored the accepted graph and performance without a
second execution environment. One authority removes container/host drift and
the operational cost of candidate promotion mechanics.

## ADR-002 — Direct application validation is the default

**Accepted:** 2026-08-12; reaffirmed 2026-08-15

Normal work ends after focused tests and one practical direct smoke of the
changed capability. Immutable releases, selectors, state clones, evidence
sealing, broad suites, and promotion ceremony require explicit need.

**Why:** Validation must serve application behavior rather than displace it.

## ADR-003 — Baseline three-camera tracking remains canonical

**Accepted:** 2026-08-12; MV3DT option updated 2026-08-19

YOLO26-m + NvDCF baseline tracking remains the default. Kitchen/Family MV3DT is
an explicit opt-in under ADR-018; AMC remains disabled. Living Room/Family Room
do not overlap and Kitchen/Living Room are adjacency only.

**Why:** Cross-camera 3D tracking requires correct shared geometry and occupied,
synchronized overlap evidence. Enabling it earlier would create false spatial
authority.

## ADR-004 — GPU-first graph with explicit CPU boundaries

**Accepted:** DS8 era; carried forward and revalidated on DS9.1

Decode, preprocessing, inference, tracking, analytics, tiling, and OSD remain in
NVMM/GPU memory. CPU conversion is limited to declared serialization,
persistence, reconstruction, and display boundaries and is measured.

## ADR-005 — DAv2 and MapAnything have different duties

**Accepted:** 2026-08-10; rebound to DS9.1 assets 2026-08-12

DAv2 is always-on object/tracking depth. MapAnything is a gated full-frame
manual capture/reconstruction lane. Hardened registration binds both model
profiles to the exact engine and config bytes; it may not be weakened when an
engine changes.

## ADR-006 — Canonical publication is transactional

**Accepted:** 2026-08-10; lifecycle wiring corrected 2026-08-12

Tracking, its world snapshot/events, and the associated BEV publication form an
ordered committed cohort. A shared publication gate covers depth and analytics
callbacks. Shutdown closes and drains that gate before WebSocket egress.

**Why:** Dashboard state must never outrun committed world authority, and native
callbacks must not race transport teardown.

## ADR-007 — Prior-Conditioned Fusion is the canonical room source

**Accepted:** 2026-08-13; lifecycle and authority clarified 2026-08-15

PCF means **Prior-Conditioned Fusion**: DA3 world poses and sparse reliable
metric depth condition MapAnything, then the two surface estimates are
consistency-gated with DA3 retained as pose carrier. Its selected artifact name
is `prior_conditioned_consensus_da3_carrier`. A calibrated static-camera room
revision aligns and independently validates the phone-only reconstruction; its
RGB/depth is not inserted into the phone inference batch.

An approved PCF candidate is sealed as a room-scan bundle and converted into an
immutable, camera-bound Scene Prior before Noesis may load it. For that admitted
Scene Prior, the depth drawer and BEV floorplan use the PCF reconstruction.
Full measured reconstruction extent is distinct from authored semantic room
membership. Live tracking authority is paired into the PCF BEV payload; PCF
does not create, move, or replace tracks. The canonical lifecycle and gates are
in [`PCF_Workflow.md`](PCF_Workflow.md).

**Why:** Phone coverage produces the clearest room geometry observed so far;
DA3 supplies the stable metric pose carrier, sparse DA3 depth constrains
MapAnything without treating every correlated pixel as truth, and final
consistency gating rejects unsupported conflicts. Keeping the static camera
independent preserves a meaningful registration check and the existing
calibrated tracking authority. Room-layout models, learned point refiners,
meshes, and Gaussian splats remain optional derivatives unless they demonstrate
new measured capability; they are not PCF admission dependencies or replacement
geometry authority.

## ADR-008 — Menon owns the browser boundary

**Accepted:** 2026-08-10; preserved on native host

Noesis REST and WebSocket ports remain loopback-only and require a private
internal bearer. Menon owns browser authentication, one-use WebSocket tickets,
dashboard delivery, and media forwarding. RTSP is disabled; the mosaic uses one
H.264 SHM/WebRTC path.

## ADR-009 — Stable identity is product identity

**Accepted:** DS8 era; active in DS9.1

`stable_id` is the user-visible identity. Tracker IDs are transient diagnostic
identifiers. Resident/visitor, exclusivity, generation, and persistence rules
must remain explicit and fail closed.

## ADR-010 — Useful history is archived, not normative

**Accepted:** 2026-08-15

Superseded DS7/DS8, DS9.0, container, migration, experiment, and validation
diaries are retained under `docs/history/`, `DS9/docs/history/`, and
`plans/archive/`. Active indexes never route implementation work through them.

## ADR-011 — SV3DT metadata and cuboid presentation are separate

**Accepted:** 2026-08-15 for the Living Room optimized profile

SV3DT keeps `outputFootLocation` enabled because image-foot and 3D object user
metadata are tracking evidence. The tracker-generated red foot dot and blue
projected cuboid are debug presentation only; a V3DT-only probe removes those
exact display primitives without removing their user metadata. Noesis draws a
replacement cuboid whose bottom-face centroid is the person's instance-mask
base, with tracked bbox bottom-center as a bounded fallback.

The correction is visual only: it does not change 3D world coordinates,
StableID, tracker association, or the baseline pipeline. Other room profiles
must enable and validate the presentation independently.

**Why:** NVIDIA couples useful projection metadata to a debug cuboid whose
image projection can be visibly displaced from the tracked person. Product
presentation must remain person-anchored without corrupting tracking authority.

## ADR-012 — Multi-room PCF joins are observation-gated planar pose graphs

**Accepted:** 2026-08-15 for the lab workflow; no Kitchen/Family canonical join
accepted yet

Two independently accepted PCF rooms may share a home-world transform only
after original-RGB overlap discovery, bidirectional depth-backed PnP, a fixed
metric scale, an accepted-floor Y lock, a view-balanced planar pose graph, and
complete-view plus temporal-segment holdouts. One accepted room remains the
immutable world gauge. Whole-room ICP, nearest-neighbor initialization,
reflection, scale fitting, and visually nudged transforms are not admission
paths.

If the evidence graph fails, a short doorway connector walk adds overlap; the
system does not repeat full-room capture or promote the best-looking rejected
transform. A rejected transform may drive a clearly marked review-only
reintegrated artifact, with no Scene Prior, tracking, or dashboard authority.
Human review uses one recorded reference-camera ground basis applied to all
registered rooms after reintegration. Backend-world X/Z plots are not an
orientation contract. Camera-display handedness, raster row inversion, and 3D
view transforms must not be fed back into registration or applied to only one
room. In particular, Kitchen has no pre-registration flip; its accepted proper
Sim(3) is the complete local-to-backend transform.
The reproducible workflow and gates are in
[`PCF_Multiroom_Registration.md`](PCF_Multiroom_Registration.md).

**Why:** Cross-room surfaces can produce a convincing but wrong join when a
patterned floor or one textured view dominates. Independent phone-view and
temporal holdouts measure whether the placement generalizes, while the common
accepted floor removes a weak PnP degree of freedom. A short connector is the
minimum household action that supplies missing evidence without rebuilding
accepted rooms.

## ADR-013 — PCF presentation uses one camera-ground coordinate contract

**Accepted:** 2026-08-15

PCF and Scene Prior metric geometry remains in proper local/world frames through
all registration and serialization authority. A calibrated camera-ground
display frame is derived only after registration: positive X is camera-right,
positive Y is height above floor, and positive Z is camera-forward. For the
deployed OpenCV cameras its backend-to-display linear determinant is `-1`
because camera Y-down becomes height Y-up; that matrix is presentation-only and
may transform display positions, never registration geometry or pose
rotations.

Serialized display rasters use row zero at maximum camera-forward Z and columns
from minimum to maximum camera-right X. oai2-fe draws that row order directly.
New Scene Prior revision manifests use the same literal orientation value;
already-deployed immutable manifests with the older numeric-grid label remain
read-compatible but never trigger another display flip.
Three.js preserves model geometry and obtains camera-right screen-right plus
camera-forward screen-up through camera placement, not negative model scale.
Room-specific 180-degree rotations, consumer-applied `image_flip`, and manual
presentation nudges are prohibited. The implementation and complete retained-
operation inventory are in
[`PCF_Coordinate_Orientation_Audit.md`](PCF_Coordinate_Orientation_Audit.md).

**Why:** A display reflection can be visually plausible while reversing
asymmetric landmarks or contaminating metric registration. Separating proper
physical transforms, the explicit camera-basis conversion, raster addressing,
and viewer-camera presentation makes each operation testable and prevents a
Living Room convention from becoming hidden geometry authority.

## ADR-014 — Phone walks use adaptive views and gated provider windows

**Accepted:** 2026-08-15

Phone-walk preparation analyzes a dense candidate stream and selects views from
relative image quality, viewpoint change, temporal coverage, and adjacent-view
feature connectivity. The 256-view configuration is an emergency ceiling, not
a requested count. Selection records candidate count, selected timestamps,
reasons, quality, connectivity, and whether that ceiling constrained the walk.

MapAnything remains joint only within a measured 80-view GPU window. Longer
selected sequences use 24 exact duplicate views between neighboring windows.
Duplicate camera poses initialize a proper Sim(3); pixel-corresponding 3D from
the duplicate images robustly refines scale and translation while preserving
the pose-derived rotation. Both camera and dense-surface gates must pass before
a window enters the reconstruction. Unique views are retained in one base
phone frame and their review points are confidence-weighted in 3.5 cm voxels.
DA3 uses the same exact-overlap, fail-closed registration pattern with a
48-view joint window and 16 duplicate views. The resulting full adaptive DA3
trajectory—not a 48-view truncation—carries the sparse metric priors and final
PCF reconstruction.
This does not mix in a static-camera frame or change static reconstruction and
calibration authority.

**Why:** Uniform 48-view sampling left multi-second gaps and weak or broken
adjacent overlap in all three stored room walks. The RTX 3060 runs 80 views but
OOMs at 96 even with native DS9.1 stopped, so one larger monolithic pass is not
a valid implementation. Exact-overlap windows preserve the full adaptive walk
while making both GPU capacity and cross-window geometric consistency explicit
and fail-closed.

## ADR-015 — The Room Walk browser may build review-only PCF candidates

**Accepted:** 2026-08-16

After a DA3 base reconstruction has passed registration to the explicitly
selected static camera and revision, the Room Walk browser may run the selected
Prior-Conditioned Fusion path as an optional final stage. The application runs
DA3-pose-plus-sparse-depth conditioned MapAnything over the same immutable
prepared views, fuses it against DA3 with the DA3 pose carrier, and evaluates
the result in the independent static-camera world. PCF jobs are persisted
separately on configurable large storage and expose their GLB, raw evidence,
diagnostics, metrics, provenance, and log in the browser.

This browser boundary ends at a review candidate. It does not seal a room-scan
bundle, build or bind a Scene Prior, or change live tracking/world authority.
On GPU-constrained hosts, an explicit deployment setting may give the job a
recorded resource lease that pauses the native appliance only when it was
already active and restores it on success, failure, or interrupted-tool
recovery. The current action also refuses provider-specific
added-video revisions because silently omitting or pretending to absorb those
views would violate the exact shared-view PCF contract.

**Why:** PCF is the selected room-reconstruction method, but requiring an
expert to manually reproduce three canonical commands after every accepted
walk made the routine path error-prone. A persisted, fail-closed browser job
can automate those same commands without widening the publication boundary or
weakening camera/revision identity checks.

## ADR-016 — Whole-home PCF is a release-bound Menon review sidecar

**Accepted:** 2026-08-16 for review; the current three-room registration remains
rejected for canonical use

Noesis owns the metric PCF assembly and publishes its immutable point artifact,
provenance, registration status, uncertainty, and complete binding to the
current authored scene. The binding records the scene release, authored-model,
calibration, runtime-configuration, and runtime world-alignment digests plus the
exact live `world_to_scene_col_major` similarity. It also binds the calibrated
Family Room device-reference camera pose and that camera's admitted Scene Prior
reference pose inside the final PCF. The anchor keeps metric scale, gravity,
floor height, calibrated pitch, and calibrated roll fixed. Sparse mutual RGB
matches and PCF-depth-backed PnP remain an uncertainty diagnostic only; their
free translation is not admitted. A static image supplies registration
evidence only; its points are not inserted into the phone reconstruction.

Menon is a read-only presentation consumer: it verifies those values against
its loaded cohort and applies one floor-locked planar camera correction to the
entire assembly. The correction maps the admitted Family camera X/Z and heading
to the device-reference X/Z and heading while applying zero vertical
translation, then composes with `world_to_scene`. Menon may expose the result
only as an owner-visible toggleable sidecar. It does not anchor a PCF corner or
bounding box, recenter, reflect, ICP-align, transform rooms independently,
visually nudge, or write the PCF into its authored model.

The review descriptor also carries the inferred static-camera center for every
room in the common assembly frame. Menon may render these centers as yellow
diagnostic spheres, but must place them in the same transform group as the PCF
geometry. A marker may never drive an additional per-room adjustment. Its
purpose is to make room-registration disagreement with authored camera devices
visible instead of hiding that disagreement behind the primary Family anchor.
The Menon review renderer may apply a non-authoritative ceiling cutaway above
1.85 m to the loaded point copy so the rooms remain inspectable from above.
That presentation filter never changes the immutable GLB or Noesis geometry.

Any missing or mismatched binding, changed artifact bytes, or unavailable
current release fails closed. A rejected multi-room registration remains
visibly labeled review-only with its measured uncertainty; successful visual
projection does not grant tracking, world, Scene Prior, deployment, or
canonical geometry authority.

**Why:** The authored model and PCF already share the Noesis world only through
an explicit deployed scene transform. Reusing that authority gives a
reproducible overlay without inventing a second alignment in Menon, while exact
release and byte binding prevents a convincing but stale or misregistered
artifact from silently entering the digital home.

## ADR-017 — Kitchen/Family MV3DT remains an isolated evaluation lane (superseded)

**Accepted:** 2026-08-19 for recorded-input evaluation only

The canonical appliance remains non-MV3DT. A separate Kitchen/Family review
profile may run only with `NOESIS_MV3DT_EVALUATION=1`; the runtime rejects that
profile without the explicit flag. It binds the current review-only
Kitchen-to-Family transform, keeps Family Room as the fixed gauge, and gives
only Kitchen and Family Room cross-camera MQTT edges. Living Room retains its
local geometry and uses a self-topic synchronization loop because the ordered
DS9.1 communicator blocks a batched tracker when a stream has an empty peer
entry. The self-loop is not a vision-neighbor edge and exposes no other
camera's measurements or IDs to Living Room.

Recorded cohorts use complete batches, the shared tracker batch counter for
frame IDs, and profile-owned analytics state materialized outside both Git and
the baseline writable state. This prevents appliance state from silently
replacing the V3DT portrait/reflection exclusions while preserving the
non-V3DT runtime unchanged.

The profile is not promotable. Both the July single-person cohort and the
multi-person stress segment advance normally after the communicator fix, and
visual samples place the corrected cuboid base under the tracked feet. The
single-person exclusions reduce coincident Kitchen/Family observations from
81 frames to 12, but neither cohort produces a verified cross-room StableID
handoff. The bound static transform is still rejected by its held-out geometry
gates, and the recordings do not provide an accepted same-person shared-FOV
correspondence set. Canonical MV3DT therefore remains blocked on accepted
Kitchen geometry plus a synchronized occupied Kitchen/Family overlap capture.

**Why:** This preserves a fast, directly testable two-room implementation
without granting tracking authority to rejected geometry or changing the
proven baseline/SV3DT lane.

## ADR-018 — Kitchen/Family MV3DT is a ready explicit runtime option

**Accepted:** 2026-08-19; supersedes ADR-017 for current operation

`--tracking-mode mv3dt` selects the accepted Kitchen/Family Room profile. It is
never the implicit default, so baseline and SV3DT behavior remain unchanged.
Kitchen and Family Room are the only peer edge. Living Room remains local-only;
its MQTT self-topic satisfies the DS9.1 ordered communicator without exposing
peer measurements or IDs.

The native host supervisor exposes the same explicit selector on both `check`
and `run`; omitting it remains baseline. The live cameras do not publish one
shared PTP/NTP timestamp domain, so streammux uses complete current-frame
batches with `sync-inputs: 0`, and `useBatchNumForFrameId: 1` gives every camera
in each batch the common MV3DT frame ID. The recorded lane retains its infinite
complete-batch timeout; the live lane retains the normal finite timeout.

The accepted geometry uses independent Kitchen and Family Room static-camera
anchors in the Family Room gauge. Shared MQTT connection startup prevents a
camera communicator from joining only at end-of-stream. The 1.7 m object model,
two-frame probation and common-frame gate, 0.18 peer score, 4.75 m peer fusion
safety radius, and 0.05 peer-visibility floor are confined to this MV3DT
profile. Product identity uses MV3DT's batch-global tracker ID as one StableID
manager key and unions present IDs across cameras; baseline and SV3DT retain
their camera-scoped lifecycle.

Acceptance evidence is direct application behavior: all three July
Kitchen/Family doorway episodes adopted a shared native ID, including late
reassociation after the peer view disappeared. The multi-person recording
produced shared IDs on visually confirmed same-person pairs without merging
the other person or persistent partial-body duplicate tracks. A captured
rendered mosaic confirmed that the replacement cuboid bottom-face centroid
stays on the person-mask foot/gravity point at near, far, doorway, and full-body
positions.

A native-host live-camera smoke loaded the accepted profile, connected all
three communicators, started WebRTC/REST/WebSocket, completed 412 publication
callbacks without rejection, and delivered 137 encoded mosaic frames without
drops during the bounded active interval.

**Why:** The prior blocker was not missing code; it was incorrect camera
anchors, a communicator startup race, an over-strict match threshold, and
camera-scoped handling of a batch-global MV3DT ID. Those failures are now
corrected and exercised while the normal tracking lanes remain isolated.

## ADR-019 — Live MV3DT waits for complete camera batches

**Accepted:** 2026-08-19; supersedes the live batching and live-smoke claims in
ADR-018

The opt-in live MV3DT profile uses `streammux.batched-push-timeout: -1` while
retaining `live-source: 1` and `sync-inputs: 0`. The ordered MV3DT peer-message
synchronizer and `useBatchNumForFrameId: 1` require every tracker input buffer
to contain all three camera frames. The installed `nvstreammux` documents that
a finite timeout pushes a partial batch; live RTSP jitter reproduced a tracker
wedge after 137-139 complete output batches, followed by graph-wide
backpressure and failed orderly EOS. The earlier 137-frame live smoke therefore
was startup evidence, not sustained-liveness evidence.

The synchronized July file replay completed 7,110 source-frame publications
and normal finite-source EOS with complete batches. Live MV3DT now applies the
same completeness invariant. A probe-free live run then sustained all three
sources at approximately 30 FPS for 148 seconds after first output, delivered
4,480 encoded mosaic frames with zero drops, completed 13,443 publication
callbacks, and accepted orderly EOS. An unavailable camera remains a
fail-closed condition handled by the existing per-source progress watchdog and
supervisor-owned restart; the runtime does not feed partial batches to MV3DT
or silently fall back to baseline/SV3DT. Baseline configuration and graph
construction are unchanged.

**Why:** Queue isolation cannot repair a peer synchronizer that received an
invalid partial cohort. Enforcing complete tracker cohorts removes the trigger
while preserving the accepted batch-global identity and geometry contracts.

## ADR-020 — One revision-bound ground state feeds every spatial view

**Accepted:** 2026-08-23

Baseline person grounding is estimated once in the camera's active
`backend_world_m` revision. Every active room prior has an explicit revision
binding: identity is still an explicit edge, while a leveled prior uses its
recorded rigid `Wc`. Raw `E` remains calibration evidence; live floor rays use
the active `E @ inv(Wc)` view and its leveled floor. Frame, transform,
calibration, alignment, and Scene Prior revisions must agree or the observation
fails closed.

Seated or lying hips and torsos are never projected as floor contacts. Visible
ankles or supported person pixels may contribute; bbox-only, cached-as-current,
and contaminated depth remain diagnostics. BEV, the dashboard, OSD trails, and
later 3D clients consume the filtered world state and may only apply a
revision-checked view transform. A 2D BEV transform uses horizontal
camera-right and camera-forward axes, never the pitched camera frame. They
cannot reselect depth, cast a new floor ray, or smooth the canonical point a
second time.

A current canonical hold stays visible with explicit held provenance and does
not extend its trail. Its normal cap is 0.40 seconds; only a trusted
seated/lying lifecycle with exact-frame stationary bbox continuity may hold for
up to 2.0 seconds. When a current depth sample is unavailable or stale, a
bounded canonical constant-velocity prediction may remain visible with
predicted provenance. Reject-driven prediction remains anchored to the last
accepted state, is capped at 0.40 seconds, does not advance last-good state, and
may extend history only when `trail_append_allowed=true`. Exact absence removes
the state from active authority and places it in a 0.75-second quarantine. It
is restored only for the same camera/tracker key when the returning bbox also
matches the prior image position and scale; otherwise the new lifecycle starts
cold. BEV/OSD trail history remains generation-keyed and is never restored.
OSD joins the canonical track on the exact
source/frame cohort, maps the declared source-image basis into the configured
mosaic tile, and breaks history on tracker lifecycle or coordinate-basis
changes. It never connects a missing/out-of-bounds canonical anchor to bbox
bottom or a clamped mosaic edge.

The physical gate applies to the proposed published posterior as well as the
raw measurement innovation. Measurement-noise slack may admit evidence for
filtering, but it may not create a continuous step faster than
`max_speed_mps`. Such a proposal is quarantined and uses the bounded prediction
path; only evidence-backed reacquisition may relocate the head, and that
relocation always increments the trail segment.

**Why:** A second display estimator and two different floor revisions produced
posture- and range-dependent metre-scale placement errors. One explicit state
and one immutable frame edge make every view numerically comparable and keep
uncertain evidence visible without granting it position authority.

## ADR-021 — Optional depth and persistence cannot stall media

**Accepted:** 2026-08-23

The mosaic path uses GPU `nvdsosd`; mapping the 3840x720 RGBA surface through
the host is forbidden. Full-frame DAv2 remains a secondary observation branch:
its queue is latest-frame-only, device readiness is query-only, and its private
CUDA work cannot back-pressure tracking, OSD, or NVENC. Compact per-person ROI
results use private streams and reusable pinned host buffers.

Periodic identity evidence and gallery persistence run through bounded writers
outside the media callback. StableID prewarms the production-shaped CUDA
similarity operation during startup so lazy Torch/CUDA initialization cannot
land on the first occupied frame. Mux and tiler pools are explicitly sized for
the bounded metadata queues rather than relying on SDK defaults.

Native tensor/surface extraction is single-owner per frame and shared by its
consumers. Work amplified by detections, tracks, cameras, clients, or retained
evidence has explicit bounds. Performance changes preserve the selected models,
resolution, inference cadence, tracker, and outputs unless an explicit matched
quality tradeoff is approved. Config flags alone do not establish zero-copy;
the native consumer and source/encode/delivery layers must be measured. The
operational rules are centralized in
[`performance_invariants.md`](performance_invariants.md).

**Why:** The observed corruption and freezes were buffer starvation and
synchronization effects, not insufficient detector throughput. Optional
evidence and durable I/O must degrade their own freshness without delaying the
authoritative video/tracking path.

## ADR-022 — Image-motion prediction is display-only continuity

**Accepted:** 2026-08-23

When a current person ground measurement is missing or physically rejected, the
canonical producer may transport the last physically accepted image foot by
the current detector-box affine motion and project it through the active
corrected floor plane. This `image_motion_prediction` is bounded by lifecycle,
short-TTL, image-step/speed, ray, and metric-speed gates; it never updates the
world filter or its accepted image origin. Stationary `anchor_hold` remains the
only fallback for a seated/lying box proven stationary, and failed transport
after material bbox motion fails closed.

**Why:** The tracker can move coherently while DAv2/pose contact is temporarily
unavailable. Reusing a fixed world point makes the BEV lie about position;
integrating a rejected CV state compounds error. A one-frame projective bridge
keeps all displays on the same PCF world while preserving honest authority and
bounded failure behavior.

## ADR-023 — Shadow identity is isolated from canonical publication

**Accepted:** 2026-08-23

Identity-v2 authoritative mode remains synchronous after the source metadata
walk because its decision owns the same-frame public identity. Shadow mode is
diagnostic: the media callback copies at most 64 compact primitives with a
bounded embedding dimension only after the exact tracking/world/BEV cohort is
admitted. It gives a separate deep scalar snapshot to one bounded shadow
worker. That worker retains at most the newest pending item per source and at
most 64 pending sources; replacing a stale same-source item, rejecting an
excess source, or becoming unavailable affects shadow freshness only. It does
not retain frame surfaces, SDK objects, diagnostic rows, or public embeddings.

Canonical rows never wait for shadow scoring and are never mutated by its
result. A rate-gated camera frame has no shadow work. An admitted cohort,
including an empty heartbeat, is eligible for shadow work but may be superseded
by a newer pending cohort from the same source. A source reconnect scopes the
coordinator's tracker key by `source_epoch` while leaving the public
process-local numeric tracker ID unchanged. Copy, binding, mode, resolver, or
visitor-persistence failure degrades and stops only the shadow lane; it does
not fail the canonical publication worker or stop media. Authoritative
identity retains its synchronous fail-closed behavior.

The runtime exposes current/high-water pending counts, in-flight state,
completed/coalesced/drop/failure totals, and per-call shadow and canonical
publication timings in the existing core-path stats. The canonical worker
remains exact per-source FIFO with no coalescing or silent drops.

**Why:** A paced replay against the populated visitor store measured shadow
processing at about 22.4 source-frames/s, including roughly 89 ms p95 tails,
while canonical publication required about 29.6 source-frames/s. Visitor
observation/session SQLite mutations dominated that tail. Sharing the exact
publication worker therefore accumulated backlog until its finite queue
failed. Shadow comparison and visitor persistence are reconstructable evidence;
tracking/world/BEV are the product authority and must remain independent.
