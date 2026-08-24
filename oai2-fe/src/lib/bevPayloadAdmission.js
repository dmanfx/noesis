const integer = (value, minimum) => (
  Number.isInteger(value) && value >= minimum ? value : null
);

export const bevCameraId = (payload) => String(
  payload?.cameraId ?? payload?.camId ?? payload?.camera_id ?? payload?.cam_id ?? ''
).trim();

export const bevCohort = (payload) => {
  if (!payload?.cohort || typeof payload.cohort !== 'object') return null;
  const cohort = payload.cohort;
  const sourceId = integer(payload?.sourceId, 0);
  const frameId = integer(payload?.frameId, 0);
  const observedAtUs = integer(payload?.observedAtUs, 1);
  const sequence = integer(
    payload?.trackingPublicationSequence,
    0,
  );
  const outboundSubmissionId = integer(
    payload?.trackingOutboundSubmissionId,
    1,
  );
  if ([sourceId, frameId, observedAtUs, sequence, outboundSubmissionId].some((value) => value === null)) {
    return null;
  }
  const mirrored = [
    ['sourceId', 'source_id', sourceId],
    ['frameId', 'frame_id', frameId],
    ['observedAtUs', 'observed_at_us', observedAtUs],
    ['trackingPublicationSequence', 'tracking_publication_sequence', sequence],
  ];
  for (const [topKey, cohortKey, expected] of mirrored) {
    if (payload[topKey] !== expected) return null;
    if (cohort[cohortKey] !== expected) return null;
  }
  if (cohort.tracking_outbound_submission_id !== outboundSubmissionId) {
    return null;
  }
  return { sourceId, frameId, observedAtUs, sequence, outboundSubmissionId };
};

const hasCohortMetadata = (payload) => (
  payload
  && typeof payload === 'object'
  && (
    Object.prototype.hasOwnProperty.call(payload, 'cohort')
    || Object.prototype.hasOwnProperty.call(payload, 'sourceId')
    || Object.prototype.hasOwnProperty.call(payload, 'frameId')
    || Object.prototype.hasOwnProperty.call(payload, 'observedAtUs')
    || Object.prototype.hasOwnProperty.call(payload, 'trackingPublicationSequence')
    || Object.prototype.hasOwnProperty.call(payload, 'trackingOutboundSubmissionId')
  )
);

const canonicalFrame = (payload) => {
  const mode = String(payload?.frame_mode ?? '').trim().toLowerCase();
  const frame = String(payload?.frame ?? '').trim().toLowerCase();
  const worldFrame = String(payload?.world_frame ?? '').trim().toLowerCase();
  if (mode === 'world') {
    return frame === 'backend_world_m'
      && (worldFrame === 'backend_world_m' || worldFrame === 'world');
  }
  if (mode === 'camera_local') {
    return frame === 'camera_local_ground_m'
      && worldFrame === 'camera_local_ground_m';
  }
  return false;
};

export const admitBevFrame = (previous, incoming) => {
  if (!incoming || incoming.type !== 'bev-frame') {
    return { admitted: false, reason: 'not_bev_frame', payload: null };
  }
  const cameraId = bevCameraId(incoming);
  if (!cameraId) return { admitted: false, reason: 'camera_missing', payload: null };
  if (!canonicalFrame(incoming)) {
    return { admitted: false, reason: 'coordinate_frame_invalid', payload: null };
  }
  const nextCohort = bevCohort(incoming);
  if (!nextCohort) return { admitted: false, reason: 'cohort_invalid', payload: null };

  const previousCohort = previous ? bevCohort(previous) : null;
  if (
    previousCohort
    && nextCohort.sourceId === previousCohort.sourceId
    && (
      nextCohort.observedAtUs <= previousCohort.observedAtUs
      || nextCohort.outboundSubmissionId <= previousCohort.outboundSubmissionId
    )
  ) {
    return { admitted: false, reason: 'cohort_not_newer', payload: null };
  }

  return {
    admitted: true,
    reason: 'admitted',
    payload: {
      ...incoming,
      cameraId,
      sourceId: nextCohort.sourceId,
      frameId: nextCohort.frameId,
      observedAtUs: nextCohort.observedAtUs,
      trackingPublicationSequence: nextCohort.sequence,
      trackingOutboundSubmissionId: nextCohort.outboundSubmissionId,
      footpoints: Array.isArray(incoming.footpoints) ? incoming.footpoints : [],
      trails: Array.isArray(incoming.trails) ? incoming.trails : [],
    },
  };
};

/**
 * Admit a BEV status only for the exact/current producer cohort.
 *
 * A status is allowed to clear a frame when it has the same cohort (the
 * renderer failed that attempt) or a strictly newer cohort. A cohort-less
 * status is only valid before a canonical frame exists; otherwise a delayed
 * error would erase newer dots/trails. Local transport resets use
 * clearBevForStatus directly and intentionally bypass producer admission.
 */
export const admitBevStatus = (previous, incoming) => {
  if (!incoming || incoming.type !== 'bev-status') {
    return { admitted: false, reason: 'not_bev_status', payload: null };
  }
  const cameraId = bevCameraId(incoming);
  if (!cameraId) return { admitted: false, reason: 'camera_missing', payload: null };

  const incomingHasCohort = hasCohortMetadata(incoming);
  const nextCohort = incomingHasCohort ? bevCohort(incoming) : null;
  if (incomingHasCohort && !nextCohort) {
    return { admitted: false, reason: 'cohort_invalid', payload: null };
  }

  const previousCohort = previous ? bevCohort(previous) : null;
  if (!nextCohort) {
    if (previousCohort) {
      return { admitted: false, reason: 'cohort_missing', payload: null };
    }
    return {
      admitted: true,
      reason: 'admitted_unbound_status',
      payload: clearBevForStatus(previous, { ...incoming, cameraId }),
    };
  }

  if (previousCohort && nextCohort.sourceId === previousCohort.sourceId) {
    const sameCohort = (
      nextCohort.observedAtUs === previousCohort.observedAtUs
      && nextCohort.outboundSubmissionId === previousCohort.outboundSubmissionId
    );
    const newerCohort = (
      nextCohort.observedAtUs > previousCohort.observedAtUs
      && nextCohort.outboundSubmissionId > previousCohort.outboundSubmissionId
    );
    if (!sameCohort && !newerCohort) {
      return { admitted: false, reason: 'cohort_not_newer', payload: null };
    }
  }

  return {
    admitted: true,
    reason: 'admitted',
    payload: clearBevForStatus(previous, { ...incoming, cameraId }),
  };
};

export const clearBevForStatus = (previous, status) => ({
  ...(previous && typeof previous === 'object' ? previous : {}),
  ...(status && typeof status === 'object' ? status : {}),
  type: 'bev-status',
  cameraId: bevCameraId(status) || bevCameraId(previous),
  footpoints: [],
  trails: [],
  droppedFootpoints: [],
  droppedFootpointCount: 0,
});

export const bevMatchesFloorplan = (bev, floorplan) => {
  if (String(bev?.frame_mode ?? '').toLowerCase() !== 'camera_local') return true;
  if (!floorplan || typeof floorplan !== 'object') return false;
  const pairs = [
    [bev?.floorplanSnapshotId, floorplan?.snapshot_id],
    [bev?.floorplanSnapshotContentSha256, floorplan?.snapshot_content_sha256],
    [bev?.floorplanCalibrationFingerprint, floorplan?.calibration_fingerprint],
  ];
  for (const [expected, actual] of pairs) {
    if (expected === undefined || expected === null || expected === '') return false;
    if (actual === undefined || actual === null || actual === '') return false;
    if (String(actual ?? '') !== String(expected)) return false;
  }
  return true;
};
