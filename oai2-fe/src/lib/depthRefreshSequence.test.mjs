import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

import {
  cachedSnapshotPairOutcome,
  exactFloorplanContinuation,
  exactFloorplanResponseOutcome,
  floorplanMatchesActiveDepth,
} from './depthRefreshSequence.js';

const exactPayload = {
  snapshot_ref: 'depth/kitchen/42.npz',
  snapshot_id: 'snapshot-42',
  snapshot_content_sha256: 'a'.repeat(64),
};

test('successful fresh depth continues with one exact non-cache floorplan request', () => {
  const continuation = exactFloorplanContinuation({
    type: 'ma_depth_response',
    camera: 'kitchen',
    ok: true,
    cache_only: false,
    served_from_cache: false,
    payload: exactPayload,
  }, { nowMs: 1234 });

  assert.ok(continuation);
  assert.deepEqual(continuation.request, {
    camera: 'kitchen',
    requestId: 'bev-exact-snapshot-42-1234',
    maxAgeSec: 0,
    gridResM: 0.04,
    maxExtentM: 20,
    cacheOnly: false,
    snapshotRef: 'depth/kitchen/42.npz',
    snapshotId: 'snapshot-42',
    snapshotContentSha256: 'a'.repeat(64),
  });
});

test('cache responses and failed fresh responses never continue to generation', () => {
  assert.equal(exactFloorplanContinuation({
    type: 'ma_depth_response',
    camera: 'kitchen',
    ok: true,
    cache_only: true,
    served_from_cache: true,
    payload: exactPayload,
  }), null);
  assert.equal(exactFloorplanContinuation({
    type: 'ma_depth_response',
    camera: 'kitchen',
    ok: false,
    cache_only: false,
    served_from_cache: false,
    payload: exactPayload,
  }), null);
});

test('partial snapshot identity never falls back to latest', () => {
  assert.equal(exactFloorplanContinuation({
    type: 'ma_depth_response',
    camera: 'kitchen',
    ok: true,
    cache_only: false,
    served_from_cache: false,
    payload: {
      snapshot_ref: exactPayload.snapshot_ref,
      snapshot_id: exactPayload.snapshot_id,
    },
  }), null);
});

test('bulk descriptor content_sha256 maps to the exact floorplan identity', () => {
  const continuation = exactFloorplanContinuation({
    type: 'ma_depth_response',
    camera: 'kitchen',
    ok: true,
    cache_only: false,
    served_from_cache: false,
    payload: {
      snapshot_ref: exactPayload.snapshot_ref,
      snapshot_id: exactPayload.snapshot_id,
      content_sha256: exactPayload.snapshot_content_sha256,
    },
  }, { nowMs: 55 });
  assert.equal(
    continuation?.request.snapshotContentSha256,
    exactPayload.snapshot_content_sha256,
  );
});

const pendingFloorplan = {
  requestId: 'bev-exact-snapshot-42-1234',
  snapshotRef: exactPayload.snapshot_ref,
  snapshotId: exactPayload.snapshot_id,
  snapshotContentSha256: exactPayload.snapshot_content_sha256,
};

test('staged depth commits only for the matching renderable exact floorplan', () => {
  assert.deepEqual(exactFloorplanResponseOutcome(pendingFloorplan, {
    type: 'floorplan_response',
    request_id: pendingFloorplan.requestId,
    ...exactPayload,
  }, { renderable: true }), { kind: 'commit' });
});

test('cache/bootstrap floorplans cannot commit a staged fresh depth', () => {
  assert.deepEqual(exactFloorplanResponseOutcome(pendingFloorplan, {
    type: 'floorplan_response',
    request_id: 'bev-bootstrap-7',
    ...exactPayload,
  }, { renderable: true }), { kind: 'unrelated' });
});

test('matching request failures and identity drift preserve the visible pair', () => {
  assert.deepEqual(exactFloorplanResponseOutcome(pendingFloorplan, {
    type: 'floorplan_response',
    request_id: pendingFloorplan.requestId,
    error: 'rate_limited',
  }), { kind: 'error', error: 'rate_limited' });

  assert.deepEqual(exactFloorplanResponseOutcome(pendingFloorplan, {
    type: 'floorplan_response',
    request_id: pendingFloorplan.requestId,
    ...exactPayload,
    snapshot_id: 'wrong-snapshot',
  }, { renderable: true }), { kind: 'error', error: 'snapshot_identity_mismatch' });

  assert.deepEqual(exactFloorplanResponseOutcome(pendingFloorplan, {
    type: 'floorplan_response',
    request_id: pendingFloorplan.requestId,
    ...exactPayload,
  }, { renderable: false }), { kind: 'error', error: 'invalid_floorplan_response' });
});

test('cache floorplans cannot replace a visible depth from another snapshot', () => {
  assert.equal(
    floorplanMatchesActiveDepth('snapshot-42', { snapshot_id: 'snapshot-42' }),
    true,
  );
  assert.equal(
    floorplanMatchesActiveDepth('snapshot-42', { snapshot_id: 'snapshot-41' }),
    false,
  );
  assert.equal(
    floorplanMatchesActiveDepth('snapshot-42', {}),
    false,
  );
  assert.equal(
    floorplanMatchesActiveDepth(undefined, { snapshot_id: 'snapshot-41' }),
    true,
  );
});

const cachedDepth = {
  ts: 1_780_000_000_000_000,
  snapshotRef: exactPayload.snapshot_ref,
  snapshotId: exactPayload.snapshot_id,
  snapshotContentSha256: exactPayload.snapshot_content_sha256,
};

const cachedFloorplan = {
  snapshot_ts: cachedDepth.ts,
  snapshot_ref: cachedDepth.snapshotRef,
  snapshot_id: cachedDepth.snapshotId,
  snapshot_content_sha256: cachedDepth.snapshotContentSha256,
};

test('passive cache pair commits only with a complete exact snapshot identity', () => {
  assert.deepEqual(
    cachedSnapshotPairOutcome(cachedDepth, cachedFloorplan, { renderable: true }),
    { kind: 'commit' },
  );
  assert.deepEqual(
    cachedSnapshotPairOutcome(cachedDepth, {
      ...cachedFloorplan,
      snapshot_id: 'snapshot-older',
    }, { renderable: true }),
    { kind: 'error', error: 'cached_snapshot_identity_mismatch' },
  );
  assert.deepEqual(
    cachedSnapshotPairOutcome(cachedDepth, {
      ...cachedFloorplan,
      snapshot_content_sha256: undefined,
    }, { renderable: true }),
    { kind: 'error', error: 'incomplete_cached_snapshot_identity' },
  );
});

test('passive cache pair waits without exposing either unpaired side', () => {
  assert.deepEqual(
    cachedSnapshotPairOutcome(cachedDepth, null, { renderable: false }),
    { kind: 'waiting', waitingFor: 'floorplan' },
  );
  assert.deepEqual(
    cachedSnapshotPairOutcome(null, cachedFloorplan, { renderable: true }),
    { kind: 'waiting', waitingFor: 'depth' },
  );
  assert.deepEqual(
    cachedSnapshotPairOutcome(cachedDepth, cachedFloorplan, { renderable: false }),
    { kind: 'error', error: 'invalid_cached_floorplan_response' },
  );
});

test('floorplan transport fallback preserves the depth-panel 4 cm resolution', async () => {
  const source = await readFile(
    fileURLToPath(new URL('../hooks/useWebSocketClient.ts', import.meta.url)),
    'utf8',
  );
  assert.match(source, /grid_res_m:\s*options\?\.gridResM \?\? 0\.04/);
});
