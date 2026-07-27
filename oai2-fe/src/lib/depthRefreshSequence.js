const text = (value) => String(value ?? '').trim();

/**
 * Builds the sole allowed fresh-floorplan continuation: a successful,
 * explicitly fresh depth response carrying a complete sealed snapshot
 * identity. Cache reads, errors, and partial identities produce no action.
 */
export function exactFloorplanContinuation(message, {
  gridResM = 0.04,
  maxExtentM = 20,
  nowMs = Date.now(),
} = {}) {
  if (!message || typeof message !== 'object') return null;
  if (
    message.type !== 'ma_depth_response'
    || message.ok === false
    || message.cache_only !== false
    || message.served_from_cache !== false
  ) {
    return null;
  }

  const payload = message.payload && typeof message.payload === 'object'
    ? message.payload
    : message;
  const camera = text(
    message.camera
    || message.cam_id
    || message.camera_id
    || message.cameraId
    || payload.camera
    || payload.camera_id,
  );
  const snapshotRef = text(payload.snapshot_ref);
  const snapshotId = text(payload.snapshot_id);
  const snapshotContentSha256 = text(
    payload.snapshot_content_sha256 ?? payload.content_sha256,
  );
  if (!camera || !snapshotRef || !snapshotId || !snapshotContentSha256) return null;

  return {
    identityKey: `${camera}|${snapshotRef}|${snapshotId}|${snapshotContentSha256}`,
    request: {
      camera,
      requestId: `bev-exact-${snapshotId}-${Math.floor(Number(nowMs) || Date.now())}`,
      maxAgeSec: 0,
      gridResM,
      maxExtentM,
      cacheOnly: false,
      snapshotRef,
      snapshotId,
      snapshotContentSha256,
    },
  };
}

/**
 * Classifies a floorplan response against one staged fresh-depth snapshot.
 * Only the exact request and all three immutable identity fields may commit the
 * staged pair. Unrelated cache/bootstrap responses remain unrelated, while any
 * failure on the exact request is explicit and leaves the visible pair intact.
 */
export function exactFloorplanResponseOutcome(pending, message, {
  renderable = false,
} = {}) {
  if (!pending || !message || typeof message !== 'object') {
    return { kind: 'unrelated' };
  }
  const requestId = text(message.request_id ?? message.requestId);
  if (!requestId || requestId !== text(pending.requestId)) {
    return { kind: 'unrelated' };
  }

  const responseError = text(message.error);
  if (responseError || message.ok === false) {
    return {
      kind: 'error',
      error: responseError || 'floorplan_generation_failed',
    };
  }
  if (!renderable) {
    return { kind: 'error', error: 'invalid_floorplan_response' };
  }

  const expected = [
    text(pending.snapshotRef),
    text(pending.snapshotId),
    text(pending.snapshotContentSha256),
  ];
  const actual = [
    text(message.snapshot_ref),
    text(message.snapshot_id),
    text(message.snapshot_content_sha256),
  ];
  if (
    expected.some((value) => !value)
    || actual.some((value) => !value)
    || expected.some((value, index) => value !== actual[index])
  ) {
    return { kind: 'error', error: 'snapshot_identity_mismatch' };
  }
  return { kind: 'commit' };
}

export function floorplanMatchesActiveDepth(activeSnapshotId, floorplan) {
  const active = text(activeSnapshotId);
  if (!active) return true;
  return text(floorplan?.snapshot_id) === active;
}

/**
 * Validates the immutable identity shared by one cached depth descriptor and
 * one cached floorplan. A passive cache read may commit only when all identity
 * fields and the capture timestamp match exactly.
 */
export function cachedSnapshotPairOutcome(depth, floorplan, {
  renderable = false,
} = {}) {
  if (!depth) return { kind: 'waiting', waitingFor: 'depth' };
  if (!floorplan) return { kind: 'waiting', waitingFor: 'floorplan' };

  const responseError = text(floorplan.error);
  if (responseError) {
    return { kind: 'error', error: responseError };
  }
  if (!renderable) {
    return { kind: 'error', error: 'invalid_cached_floorplan_response' };
  }

  const depthTimestamp = Number(depth.ts ?? depth.tsUs);
  const floorplanTimestamp = Number(floorplan.snapshot_ts);
  const expected = [
    text(depth.snapshotRef ?? depth.snapshot_ref),
    text(depth.snapshotId ?? depth.snapshot_id),
    text(depth.snapshotContentSha256 ?? depth.snapshot_content_sha256 ?? depth.content_sha256),
  ];
  const actual = [
    text(floorplan.snapshot_ref),
    text(floorplan.snapshot_id),
    text(floorplan.snapshot_content_sha256),
  ];
  if (
    !Number.isSafeInteger(depthTimestamp)
    || depthTimestamp <= 0
    || !Number.isSafeInteger(floorplanTimestamp)
    || floorplanTimestamp <= 0
    || expected.some((value) => !value)
    || actual.some((value) => !value)
  ) {
    return { kind: 'error', error: 'incomplete_cached_snapshot_identity' };
  }
  if (
    depthTimestamp !== floorplanTimestamp
    || expected.some((value, index) => value !== actual[index])
  ) {
    return { kind: 'error', error: 'cached_snapshot_identity_mismatch' };
  }
  return { kind: 'commit' };
}
