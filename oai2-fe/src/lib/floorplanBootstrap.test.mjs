import test from 'node:test';
import assert from 'node:assert/strict';

import { FloorplanBootstrapCoordinator } from './floorplanBootstrap.js';

const reply = (action, fields = {}) => ({
  type: 'floorplan_response',
  request_id: action.request.requestId,
  camera_id: action.request.camera,
  ...fields,
});

test('cache hit advances without a fresh capture', () => {
  const coordinator = new FloorplanBootstrapCoordinator();
  const first = coordinator.restart(['living-room', 'kitchen']);
  assert.equal(first.request.cacheOnly, true);
  assert.equal(first.request.gridResM, 0.04);

  const result = coordinator.handleResponse(reply(first), { renderable: true });
  assert.equal(result.completedCamera, 'living-room');
  assert.equal(result.action.request.camera, 'kitchen');
  assert.equal(result.action.request.cacheOnly, true);
});

test('cache miss advances without ever escalating to a fresh capture', () => {
  const coordinator = new FloorplanBootstrapCoordinator();
  const cache = coordinator.restart(['living-room', 'kitchen']);
  const miss = reply(cache, { error: 'no_cached_floorplan' });

  const result = coordinator.handleResponse(miss);
  assert.equal(result.failedCamera, 'living-room');
  assert.equal(result.error, 'no_cached_floorplan');
  assert.equal(result.action.request.camera, 'kitchen');
  assert.equal(result.action.request.cacheOnly, true);
  assert.equal(coordinator.handleResponse(miss).handled, false);
});

test('non-transient cache failure stops retrying that camera and continues the queue', () => {
  const coordinator = new FloorplanBootstrapCoordinator();
  const cache = coordinator.restart(['living-room', 'kitchen']);
  const failed = coordinator.handleResponse(reply(cache, { error: 'floorplan_generation_failed' }));

  assert.equal(failed.failedCamera, 'living-room');
  assert.equal(failed.error, 'floorplan_generation_failed');
  assert.equal(failed.action.request.camera, 'kitchen');
});

test('transient capture conflicts retry only within the configured bound', () => {
  const coordinator = new FloorplanBootstrapCoordinator({ maxTransientRetries: 1, retryDelayMs: 25 });
  const cache = coordinator.restart(['living-room', 'kitchen']);
  const retry = coordinator.handleResponse(reply(cache, { error: 'capture_event_busy' })).action;

  assert.equal(retry.request.camera, 'living-room');
  assert.equal(retry.request.cacheOnly, true);
  assert.equal(retry.delayMs, 25);

  const exhausted = coordinator.handleResponse(reply(retry, { error: 'capture_event_busy' }));
  assert.equal(exhausted.failedCamera, 'living-room');
  assert.equal(exhausted.action.request.camera, 'kitchen');
  assert.equal(exhausted.action.request.cacheOnly, true);
});

test('every action in a cache-miss run is cache-only', () => {
  const coordinator = new FloorplanBootstrapCoordinator();
  let action = coordinator.restart(['living-room', 'kitchen', 'family-room']);
  const actions = [];
  while (action) {
    actions.push(action);
    action = coordinator.handleResponse(
      reply(action, { error: 'no_cached_floorplan' }),
    ).action;
  }
  assert.equal(actions.length, 3);
  assert.ok(actions.every((candidate) => candidate.request.cacheOnly === true));
  assert.ok(actions.every((candidate) => candidate.request.gridResM === 0.04));
});

test('calibration restart is coalesced behind the active request', () => {
  const coordinator = new FloorplanBootstrapCoordinator();
  const original = coordinator.restart(['living-room', 'kitchen']);
  assert.equal(coordinator.restart(['family-room']), null);
  assert.deepEqual(coordinator.snapshot().pendingRestart, ['family-room']);

  const result = coordinator.handleResponse(reply(original), { renderable: true });
  assert.equal(result.action.request.camera, 'family-room');
  assert.equal(result.action.request.cacheOnly, true);
  assert.deepEqual(coordinator.snapshot().queue, []);
});
