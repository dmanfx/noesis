import assert from 'node:assert/strict';
import test from 'node:test';

import {
  historyKeyForBevIdentity,
  purgeCurrentBevHeadState,
  purgeCurrentBevHeadsForDropped,
  purgeCurrentBevHeadsMissingFromCohort,
  resetBevVisualContinuityState,
  sourceTimelineKeyForBevPayload,
  updateCurrentBevHeadState,
} from './bevCurrentHead.js';

test('lifecycle history keys stay distinct when the display id is reused', () => {
  assert.equal(historyKeyForBevIdentity({ stableId: 9, trackerId: 4, trackerLifecycleGeneration: 1 }), 't:4:g:1');
  assert.equal(historyKeyForBevIdentity({ stableId: 9, trackerId: 4, trackerLifecycleGeneration: 2 }), 't:4:g:2');
  assert.notEqual(
    historyKeyForBevIdentity({ stableId: 9, trackerId: 4, trackerLifecycleGeneration: 1 }),
    historyKeyForBevIdentity({ stableId: 9, trackerId: 8, trackerLifecycleGeneration: 1 }),
  );
  assert.equal(historyKeyForBevIdentity({ historyKey: 't:4:g:1', stableId: 99, trackerId: 12 }), 't:4:g:1');
});

test('current canonical head state is retained without trail samples', () => {
  const state = new Map();
  const accepted = updateCurrentBevHeadState({
    state,
    point: { worldAdmission: 'accepted' },
    resolved: { x: 1.25, y: 4.5 },
    nowMs: 1234,
    historyKey: 't:7:g:3',
    displayId: 7,
    colorId: 142,
  });

  assert.equal(accepted, true);
  assert.deepEqual(state.get('t:7:g:3'), {
    x: 1.25,
    y: 4.5,
    lastSeen: 1234,
    stableId: '7',
    colorId: 142,
    worldAdmission: 'accepted',
  });
});

test('current head admission preserves exact coordinates and rejects incomplete identity', () => {
  const state = new Map();
  assert.equal(updateCurrentBevHeadState({
    state,
    point: {},
    resolved: { x: 2, y: 3 },
    nowMs: 10,
    historyKey: null,
    displayId: null,
    colorId: 1,
  }), false);
  assert.equal(state.size, 0);

  assert.equal(updateCurrentBevHeadState({
    state,
    point: {},
    resolved: { x: 2.125, y: 3.875 },
    nowMs: 11,
    historyKey: 's:4',
    displayId: 4,
    colorId: 1,
  }), true);
  assert.equal(state.get('s:4').x, 2.125);
  assert.equal(state.get('s:4').y, 3.875);
});

test('dropped canonical world point purges only its lifecycle head', () => {
  const state = new Map();
  for (const [historyKey, x] of [['t:4:g:1', 1], ['t:4:g:2', 2]]) {
    updateCurrentBevHeadState({
      state,
      point: { worldAdmission: 'accepted' },
      resolved: { x, y: x + 1 },
      nowMs: 100,
      historyKey,
      displayId: 9,
      colorId: 12,
    });
  }

  assert.equal(purgeCurrentBevHeadsForDropped(state, [{
    stableId: 9,
    trackerId: 4,
    trackerLifecycleGeneration: 1,
    reason: 'canonical_world_missing',
  }]), 1);
  assert.equal(state.has('t:4:g:1'), false);
  assert.equal(state.has('t:4:g:2'), true);

  assert.equal(purgeCurrentBevHeadState(state, 't:4:g:2'), true);
  assert.equal(state.size, 0);
});

test('exact cohort membership purges every absent head beyond the diagnostic cap', () => {
  const state = new Map();
  for (let index = 0; index < 65; index += 1) {
    updateCurrentBevHeadState({
      state,
      point: { worldAdmission: 'accepted' },
      resolved: { x: index, y: index + 1 },
      nowMs: 100,
      historyKey: `t:${index}:g:1`,
      displayId: index + 1,
      colorId: index,
    });
  }

  assert.equal(
    purgeCurrentBevHeadsMissingFromCohort(state, new Set()),
    65,
  );
  assert.equal(state.size, 0);
});

test('source timeline key requires exact top-level and cohort epoch parity', () => {
  assert.equal(sourceTimelineKeyForBevPayload({
    sourceId: 2,
    sourceEpoch: 4,
    cohort: { source_id: 2, source_epoch: 4 },
  }), '2:4');
  assert.equal(sourceTimelineKeyForBevPayload({
    sourceId: 2,
    sourceEpoch: 4,
    cohort: { source_id: 2, source_epoch: 3 },
  }), null);
  assert.equal(sourceTimelineKeyForBevPayload({ sourceId: 2, sourceEpoch: 4 }), null);
});

test('source timeline reset clears heads, trails, sampling phase, and media clock', () => {
  const heads = new Map([['t:7:g:1', { x: 1, y: 2 }]]);
  const trails = new Map([['t:7:g:1', { points: [{ x: 1, y: 2, t: 100 }] }]]);
  const frameCounterRef = { current: 7 };
  const clockRef = { current: { lastSrcMs: 10_000, lastSampleMs: 20_000 } };

  resetBevVisualContinuityState({ heads, trails, frameCounterRef, clockRef });

  assert.equal(heads.size, 0);
  assert.equal(trails.size, 0);
  assert.equal(frameCounterRef.current, 0);
  assert.deepEqual(clockRef.current, {});
});
