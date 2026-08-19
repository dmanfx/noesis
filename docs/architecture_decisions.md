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

## ADR-007 — PCF is the canonical depth-panel floorplan source

**Accepted:** 2026-08-13

For an admitted camera-bound Scene Prior, the depth drawer and BEV floorplan use
the immutable PCF revision. Full measured reconstruction extent is distinct
from authored semantic room membership. Live tracking authority is paired into
the PCF BEV payload; PCF does not create or replace tracks.

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
