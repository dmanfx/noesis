# WO-2E: Pair Room Walk with static video and tracking capture

Status: implemented and directly verified, 2026-09-05.

The user wants to record the room's static RTSP camera during a browser phone
walk and retain tracking evidence for later alignment, calibration, and
validation against reconstructed phone poses.

## Capture contract

- Select the physical room camera before starting. Resolve its source through
  the active native DS9.1 configuration and existing private credential path.
- A paired recording starts only after original encoded static video and
  selected-camera canonical tracking are both arriving. Save phone video/IMU,
  static video, raw tracking/world cohorts, calibration/runtime provenance,
  clock exchanges, packet timing, and optional user markers under one session.
- Preserve original encoded camera quality without adding inference, decoded
  CPU frame branches, or work on the canonical media callback. The recorder and
  telemetry observer bound their own duration, disk use, queues, and waits.
- Preserve tracking messages in receive order with their exact source, epoch,
  frame, media-PTS, publication-sequence, and world-frame identities. Never join
  records using latest camera state. Record partial capture and missing data.
- Start, stop, heartbeat, markers, and phone-upload association are retry-safe.
  Lost tabs expire a finite recording lease; upload failure retains both the
  phone bundle and static-session evidence. Service restart records interruption.

## Authority and timing

Independent RTSP clients do not inherently share a PTS origin. Save packet PTS,
host monotonic/UTC receipt times, segment/clock metadata, and source reference
timestamps where available. Retain raw browser/server clock exchanges and
uncertainty; do not claim synchronized acquisition timestamps from HTTP timing.

The static recording uses original camera pixels. Preserve the canonical
runtime's dewarping/rectification model and calibration revision so those pixels
cannot be confused with tracker image coordinates. Calibration and world-frame
changes remain explicit evidence boundaries.

A phone camera trajectory is not the tracked person's foot/body trajectory.
Identity, phone-to-body offset, reconstruction scale/alignment uncertainty, and
timing must be resolved before treating a comparison as calibration ground truth.
This work saves evidence and does not update live tracking calibration.

## Implementation lanes and completion

1. Backend: bounded compressed recorder, paired-session lifecycle, API,
   persistence, idempotent phone-bundle association, and focused tests.
2. Source/observer: active camera configuration and private authentication,
   canonical telemetry observer, calibration/runtime snapshots, and tests.
3. Browser: camera choice, coordinated start/stop, bounded clock/lease exchange,
   markers, interruption/retry handling, and saved evidence links.
4. Parent review: producer/consumer contracts and failure behavior, focused
   checks, a bounded actual RTSP/telemetry smoke, deployed browser inspection,
   and final documentation. Physical phone timing/pose accuracy remains a
   separate field validation.

No unrelated dirty work, branch changes, commits, inference/model changes, or
appliance release procedures are part of this work order.

## Verification

The deployed browser coordinator started Living Room, retried its start with
the same identifiers, maintained its heartbeat, saved a marker and stopped the
paired session. The original H.264 video decoded successfully at 1920x1080,
30 fps: 13.091766 seconds and 391 decoded frames. Its 392 encoded-buffer timing
records include parser caps, segment and pipeline-clock metadata; no packet
metadata was dropped, and 388 buffers carried source reference timestamps.

The observer retained 26 consecutive selected-source tracking publications
and 26 world snapshots, plus the initial calibration bundle and four successful
authenticated REST snapshots. Observer sequence order and tracking sequence
continuity passed. All 11 browser/server clock probes and the marker survived
finalization. The canonical Noesis run remained unchanged and reported ready.

A synthetic phone fixture exercised association and RGB frame preparation:
15 views reached ready, identical and manual-TAR retries returned HTTP 200,
and a changed archive returned HTTP 409 without altering the saved walk.
An invalid phone upload was retained separately and a corrected retry recovered
without invalidating healthy static evidence. All ten artifact links returned
HTTP 200; Chrome displayed the paired session and stopped video/tracking status.
The synthetic walk and two test sessions were removed after retaining their
bounded evidence outside the application's walk list; the four existing walks
remain untouched.

Focused recorder, source/observer, manager, upload and browser tests passed,
including slow startup, tracking failure after readiness, writer/queue failure,
stale-video shutdown, upload recovery and conflicting retry preservation.
The phone service is running on the existing HTTP/HTTPS listeners.

This proves capture, transport and association. The phone fixture was not a
simultaneous physical walk. Fold sensor acquisition, occupied-person matching,
phone-to-body offset, timing accuracy and pose/calibration accuracy still need
field evidence. Source reference timestamps are retained without asserting
verified synchronization with the canonical tracker's independent PTS origin.
