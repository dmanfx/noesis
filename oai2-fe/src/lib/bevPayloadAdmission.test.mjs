import assert from 'node:assert/strict';
import test from 'node:test';

import {
  admitBevFrame,
  admitBevStatus,
  bevMatchesFloorplan,
  clearBevForStatus,
} from './bevPayloadAdmission.js';

const frame = (overrides = {}) => ({
  type: 'bev-frame',
  cameraId: 'family-room',
  sourceId: 2,
  sourceEpoch: 0,
  frameId: 100,
  observedAtUs: 1_000_000,
  trackingPublicationSequence: 8,
  trackingOutboundSubmissionId: 20,
  cohort: {
    source_id: 2,
    source_epoch: 0,
    frame_id: 100,
    observed_at_us: 1_000_000,
    tracking_publication_sequence: 8,
    tracking_outbound_submission_id: 20,
  },
  frame_mode: 'camera_local',
  frame: 'camera_local_ground_m',
  world_frame: 'camera_local_ground_m',
  footpoints: [{ x: 1, y: 2 }],
  trails: [],
  floorplanSnapshotId: 'prior-a',
  floorplanSnapshotContentSha256: 'a'.repeat(64),
  ...overrides,
});

test('exact empty BEV frame clears people instead of retaining the previous array', () => {
  const next = frame({
    frameId: 101,
    observedAtUs: 1_100_000,
    trackingPublicationSequence: 9,
    trackingOutboundSubmissionId: 21,
    cohort: {
      source_id: 2,
      source_epoch: 0,
      frame_id: 101,
      observed_at_us: 1_100_000,
      tracking_publication_sequence: 9,
      tracking_outbound_submission_id: 21,
    },
    footpoints: [],
  });
  const admitted = admitBevFrame(frame(), next);
  assert.equal(admitted.admitted, true);
  assert.deepEqual(admitted.payload.footpoints, []);
});

test('late, malformed and unknown-frame payloads fail closed', () => {
  const previous = frame();
  const late = frame({
    frameId: 99,
    observedAtUs: 999_999,
    trackingPublicationSequence: 7,
    trackingOutboundSubmissionId: 19,
    cohort: {
      source_id: 2,
      source_epoch: 0,
      frame_id: 99,
      observed_at_us: 999_999,
      tracking_publication_sequence: 7,
      tracking_outbound_submission_id: 19,
    },
  });
  assert.equal(admitBevFrame(previous, late).reason, 'cohort_not_newer');
  assert.equal(admitBevFrame(previous, frame({ frame: 'mystery_space' })).admitted, false);
  assert.equal(admitBevFrame(previous, frame({ cohort: { source_id: 9 } })).admitted, false);
  assert.equal(admitBevFrame(previous, frame({ cohort: {} })).admitted, false);
  assert.equal(admitBevFrame(previous, frame({
    cohort: {
      source_id: 2,
      source_epoch: 0,
      frame_id: 100,
      observed_at_us: 1_000_000,
    },
  })).admitted, false);
  assert.equal(admitBevFrame(previous, frame({
    cohort: {
      source_id: 2,
      source_epoch: 0,
      frame_id: 100,
      observed_at_us: null,
      tracking_publication_sequence: 8,
    },
  })).admitted, false);
});

test('status clears points and trails', () => {
  const cleared = clearBevForStatus(frame(), {
    type: 'bev-status',
    camera_id: 'family-room',
    error: 'frame_mismatch',
  });
  assert.equal(cleared.cameraId, 'family-room');
  assert.deepEqual(cleared.footpoints, []);
  assert.deepEqual(cleared.trails, []);
  assert.deepEqual(cleared.droppedFootpoints, []);
  assert.equal(cleared.droppedFootpointCount, 0);
  const late = frame({
    observedAtUs: 999_999,
    trackingOutboundSubmissionId: 19,
    cohort: {
      source_id: 2,
      source_epoch: 0,
      frame_id: 100,
      observed_at_us: 999_999,
      tracking_publication_sequence: 8,
      tracking_outbound_submission_id: 19,
    },
  });
  assert.equal(admitBevFrame(cleared, late).reason, 'cohort_not_newer');

  const transportClosed = clearBevForStatus(undefined, {
    type: 'bev-status',
    cameraId: 'family-room',
    error: 'transport_closed',
  });
  assert.equal(admitBevFrame(transportClosed, frame({
    frameId: 1,
    observedAtUs: 2_000_000,
    trackingPublicationSequence: 1,
    trackingOutboundSubmissionId: 1,
    cohort: {
      source_id: 2,
      source_epoch: 0,
      frame_id: 1,
      observed_at_us: 2_000_000,
      tracking_publication_sequence: 1,
      tracking_outbound_submission_id: 1,
    },
  })).admitted, true);
});

test('an older or unbound status cannot clear a newer admitted frame', () => {
  const previous = frame({
    frameId: 101,
    observedAtUs: 1_100_000,
    trackingPublicationSequence: 9,
    trackingOutboundSubmissionId: 21,
    cohort: {
      source_id: 2,
      source_epoch: 0,
      frame_id: 101,
      observed_at_us: 1_100_000,
      tracking_publication_sequence: 9,
      tracking_outbound_submission_id: 21,
    },
  });
  const oldStatus = {
    type: 'bev-status',
    cameraId: 'family-room',
    sourceId: 2,
    sourceEpoch: 0,
    frameId: 100,
    observedAtUs: 1_000_000,
    trackingPublicationSequence: 8,
    trackingOutboundSubmissionId: 20,
    cohort: {
      source_id: 2,
      source_epoch: 0,
      frame_id: 100,
      observed_at_us: 1_000_000,
      tracking_publication_sequence: 8,
      tracking_outbound_submission_id: 20,
    },
    error: 'homography_failed',
  };
  assert.equal(admitBevStatus(previous, oldStatus).reason, 'cohort_not_newer');
  assert.equal(admitBevStatus(previous, {
    type: 'bev-status',
    cameraId: 'family-room',
    error: 'late_unbound_error',
  }).reason, 'cohort_missing');
  assert.equal(previous.footpoints.length, 1);
});

test('a status for the exact failed cohort clears that cohort', () => {
  const status = {
    type: 'bev-status',
    cameraId: 'family-room',
    sourceId: 2,
    sourceEpoch: 0,
    frameId: 101,
    observedAtUs: 1_100_000,
    trackingPublicationSequence: 9,
    trackingOutboundSubmissionId: 21,
    cohort: {
      source_id: 2,
      source_epoch: 0,
      frame_id: 101,
      observed_at_us: 1_100_000,
      tracking_publication_sequence: 9,
      tracking_outbound_submission_id: 21,
    },
    error: 'homography_failed',
  };
  const admission = admitBevStatus(frame({
    frameId: 101,
    observedAtUs: 1_100_000,
    trackingPublicationSequence: 9,
    trackingOutboundSubmissionId: 21,
    cohort: {
      source_id: 2,
      source_epoch: 0,
      frame_id: 101,
      observed_at_us: 1_100_000,
      tracking_publication_sequence: 9,
      tracking_outbound_submission_id: 21,
    },
  }), status);
  assert.equal(admission.admitted, true);
  assert.deepEqual(admission.payload.footpoints, []);
  assert.equal(admission.payload.error, 'homography_failed');
});

test('higher source epoch admits a same-source media-clock rewind and rejects the old epoch', () => {
  const rewound = frame({
    sourceEpoch: 1,
    frameId: 1,
    observedAtUs: 100_000,
    trackingPublicationSequence: 0,
    trackingOutboundSubmissionId: 1,
    cohort: {
      source_id: 2,
      source_epoch: 1,
      frame_id: 1,
      observed_at_us: 100_000,
      tracking_publication_sequence: 0,
      tracking_outbound_submission_id: 1,
    },
  });
  const admitted = admitBevFrame(frame(), rewound);
  assert.equal(admitted.admitted, true);
  assert.equal(admitted.payload.sourceEpoch, 1);

  const stale = frame({
    frameId: 110,
    observedAtUs: 2_000_000,
    trackingPublicationSequence: 10,
    trackingOutboundSubmissionId: 30,
    cohort: {
      source_id: 2,
      source_epoch: 0,
      frame_id: 110,
      observed_at_us: 2_000_000,
      tracking_publication_sequence: 10,
      tracking_outbound_submission_id: 30,
    },
  });
  assert.equal(admitBevFrame(rewound, stale).reason, 'source_epoch_stale');
  assert.equal(admitBevFrame(frame(), frame({
    cohort: {
      source_id: 2,
      source_epoch: 9,
      frame_id: 100,
      observed_at_us: 1_000_000,
      tracking_publication_sequence: 8,
      tracking_outbound_submission_id: 20,
    },
  })).reason, 'cohort_invalid');
});

test('higher source epoch status can clear a rewound failed cohort', () => {
  const status = {
    type: 'bev-status',
    cameraId: 'family-room',
    sourceId: 2,
    sourceEpoch: 1,
    frameId: 1,
    observedAtUs: 100_000,
    trackingPublicationSequence: 0,
    trackingOutboundSubmissionId: 1,
    cohort: {
      source_id: 2,
      source_epoch: 1,
      frame_id: 1,
      observed_at_us: 100_000,
      tracking_publication_sequence: 0,
      tracking_outbound_submission_id: 1,
    },
    error: 'render_failed_after_rewind',
  };
  const admitted = admitBevStatus(frame(), status);
  assert.equal(admitted.admitted, true);
  assert.equal(admitted.payload.sourceEpoch, 1);
  assert.deepEqual(admitted.payload.footpoints, []);
});

test('camera-local BEV must match the exact floorplan revision', () => {
  const fingerprint = 'b'.repeat(64);
  const bev = frame({ floorplanCalibrationFingerprint: fingerprint });
  assert.equal(bevMatchesFloorplan(bev, {
    snapshot_id: 'prior-a',
    snapshot_content_sha256: 'a'.repeat(64),
    calibration_fingerprint: fingerprint,
  }), true);
  assert.equal(bevMatchesFloorplan(bev, {
    snapshot_id: 'prior-b',
    snapshot_content_sha256: 'a'.repeat(64),
    calibration_fingerprint: fingerprint,
  }), false);
  assert.equal(bevMatchesFloorplan(bev, {
    snapshot_id: 'prior-a',
    snapshot_content_sha256: 'a'.repeat(64),
    calibration_fingerprint: 'c'.repeat(64),
  }), false);
  assert.equal(bevMatchesFloorplan(bev, null), false);
  assert.equal(bevMatchesFloorplan(frame({ floorplanCalibrationFingerprint: null }), {
    snapshot_id: 'prior-a',
    snapshot_content_sha256: 'a'.repeat(64),
    calibration_fingerprint: fingerprint,
  }), false);
});
