/**
 * Return the exact lifecycle key used to associate a canonical point with
 * its current head and trail history.
 *
 * The display id is intentionally not part of the primary key.  It is a
 * presentation label and can be reused by several tracker lifecycles.
 */
export const historyKeyForBevIdentity = ({
  historyKey,
  stableId,
  trackerId,
  trackerLifecycleGeneration,
} = {}) => {
  if (typeof historyKey === 'string' && historyKey.trim().length > 0) return historyKey.trim();
  const trackerNum = typeof trackerId === 'number' && Number.isFinite(trackerId) ? trackerId : null;
  const generationNum = typeof trackerLifecycleGeneration === 'number'
    && Number.isFinite(trackerLifecycleGeneration)
    ? trackerLifecycleGeneration
    : null;
  if (trackerNum !== null && trackerNum >= 0) {
    return generationNum !== null && generationNum >= 0
      ? `t:${trackerNum}:g:${generationNum}`
      : `t:${trackerNum}`;
  }
  const stableNum = typeof stableId === 'number' && Number.isFinite(stableId) ? stableId : null;
  if (stableNum !== null && stableNum > 0) return `s:${stableNum}`;
  return null;
};

const nonNegativeInteger = (value) => (
  Number.isInteger(value) && value >= 0 ? value : null
);

/** Return the exact producer timeline key carried by a canonical BEV frame. */
export const sourceTimelineKeyForBevPayload = (payload) => {
  if (!payload || typeof payload !== 'object') return null;
  const sourceId = nonNegativeInteger(payload.sourceId);
  const sourceEpoch = nonNegativeInteger(payload.sourceEpoch);
  const cohortSourceId = nonNegativeInteger(payload.cohort?.source_id);
  const cohortSourceEpoch = nonNegativeInteger(payload.cohort?.source_epoch);
  if (
    sourceId === null
    || sourceEpoch === null
    || cohortSourceId !== sourceId
    || cohortSourceEpoch !== sourceEpoch
  ) {
    return null;
  }
  return `${sourceId}:${sourceEpoch}`;
};

/**
 * Clear all frontend-owned visual continuity at a producer timeline boundary.
 * Floorplan/background state is intentionally excluded because it is bound to
 * coordinate-space revision, not to the source media clock.
 */
export const resetBevVisualContinuityState = ({
  heads,
  trails,
  frameCounterRef,
  clockRef,
}) => {
  heads?.clear?.();
  trails?.clear?.();
  if (frameCounterRef && typeof frameCounterRef === 'object') {
    frameCounterRef.current = 0;
  }
  if (clockRef && typeof clockRef === 'object') {
    clockRef.current = {};
  }
};

/** Purge one live head without touching the corresponding trail history. */
export const purgeCurrentBevHeadState = (state, historyKey) => {
  if (!(state instanceof Map)) return false;
  if (typeof historyKey !== 'string' || historyKey.length === 0) return false;
  return state.delete(historyKey);
};

/**
 * Purge heads named by an authoritative dropped-footpoint cohort.
 *
 * The producer may retain a trail across a temporary canonical-world gap;
 * therefore this helper only touches the live-head map.  Callers decide
 * separately whether a non-retained drop should also break local trail state.
 */
export const purgeCurrentBevHeadsForDropped = (state, droppedFootpoints) => {
  if (!(state instanceof Map) || !Array.isArray(droppedFootpoints)) return 0;
  let purged = 0;
  for (const point of droppedFootpoints) {
    const historyKey = historyKeyForBevIdentity(point);
    if (historyKey !== null && purgeCurrentBevHeadState(state, historyKey)) purged += 1;
  }
  return purged;
};

/**
 * Purge every live head absent from an authoritative exact-current cohort.
 *
 * Backend-world BEV frames are complete, not deltas.  Cohort membership is
 * therefore the scalable tombstone: it remains exact even when the producer
 * intentionally caps its richer dropped-footpoint diagnostic records.
 */
export const purgeCurrentBevHeadsMissingFromCohort = (state, seenHistoryKeys) => {
  if (!(state instanceof Map) || !(seenHistoryKeys instanceof Set)) return 0;
  let purged = 0;
  for (const historyKey of state.keys()) {
    if (!seenHistoryKeys.has(historyKey) && state.delete(historyKey)) purged += 1;
  }
  return purged;
};

/**
 * Store the current resolved BEV point independently of trail history.
 *
 * The canvas uses this state for the live head marker.  Keeping this small
 * admission step separate makes it impossible for the trail enablement or
 * trail sampling policy to accidentally discard a valid current point.
 */
export const updateCurrentBevHeadState = ({
  state,
  point,
  resolved,
  nowMs,
  historyKey,
  displayId,
  colorId,
}) => {
  if (!(state instanceof Map)) return false;
  if (typeof historyKey !== 'string' || historyKey.length === 0) return false;
  if (typeof displayId !== 'number' || !Number.isFinite(displayId)) return false;
  if (!resolved || !Number.isFinite(resolved.x) || !Number.isFinite(resolved.y)) return false;
  if (!Number.isFinite(nowMs) || !Number.isFinite(colorId)) return false;

  state.set(historyKey, {
    x: resolved.x,
    y: resolved.y,
    lastSeen: nowMs,
    stableId: `${displayId}`,
    colorId,
    worldAdmission: typeof point?.worldAdmission === 'string' ? point.worldAdmission : undefined,
  });
  return true;
};
