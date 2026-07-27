import test from 'node:test';
import assert from 'node:assert/strict';

import {
  chooseDepthRange,
  computeMaskedRange,
  floorplanObservationCounts,
  integerExportScale,
  snapshotExportStem,
} from './depthQuality.js';

test('robust range excludes invalid, non-positive, and mask-invalid depth', () => {
  const values = Float32Array.from([
    Number.NaN, -5, 0, 1, 2, 3, 4, 1000,
  ]);
  const mask = Uint8Array.from([
    1, 1, 1, 1, 1, 1, 1, 0,
  ]);
  const range = computeMaskedRange(values, {
    mask,
    positiveOnly: true,
    lowPercentile: 25,
    highPercentile: 75,
  });

  assert.ok(range);
  assert.equal(range.sampleCount, 4);
  assert.equal(range.fullMin, 1);
  assert.equal(range.fullMax, 4);
  assert.equal(range.min, 1.75);
  assert.equal(range.max, 3.25);
});

test('p2-p98 limits isolated tails while preserving the full range', () => {
  const values = Float32Array.from({ length: 101 }, (_, idx) => idx === 100 ? 10_000 : idx + 1);
  const range = computeMaskedRange(values, {
    positiveOnly: true,
    lowPercentile: 2,
    highPercentile: 98,
  });

  assert.ok(range);
  assert.equal(range.fullMin, 1);
  assert.equal(range.fullMax, 10_000);
  assert.ok(range.min > 2 && range.min < 4);
  assert.ok(range.max < 101);
});

test('range mode selection keeps a valid locked range and rejects an inverted one', () => {
  const robust = { min: 1, max: 8 };
  const full = { min: 0.5, max: 30 };
  assert.deepEqual(chooseDepthRange('auto', robust, full, null), robust);
  assert.deepEqual(chooseDepthRange('full', robust, full, null), full);
  assert.deepEqual(chooseDepthRange('locked', robust, full, { min: 2, max: 5 }), { min: 2, max: 5 });
  assert.deepEqual(chooseDepthRange('locked', robust, full, { min: 5, max: 2 }), robust);
});

test('integer export scale stays integral and bounded', () => {
  assert.equal(integerExportScale(100, 200), 8);
  assert.equal(integerExportScale(300, 400), 5);
  assert.equal(integerExportScale(2500, 2000), 1);
});

test('snapshot export stem includes milliseconds and snapshot identity', () => {
  const stem = snapshotExportStem('Kitchen Camera', 'Heatmap', {
    depthTs: 123456789,
    now: new Date('2026-07-24T12:34:56.789Z'),
  });
  assert.equal(stem, '20260724123456789_kitchen-camera_heatmap_snapshot-123456789');
});

test('v8 observation metadata uses exact producer field names', () => {
  const counts = floorplanObservationCounts({
    observed_cells: 12,
    unknown_cells: 7,
    total_cells: 19,
    // A legacy lookalike must not be accepted as the producer contract.
    observed_cell_count: 999,
  }, Float32Array.from([0, 1, 1, 0]));
  assert.deepEqual(counts, {
    observed: 12,
    unknown: 7,
    total: 19,
    inferredWalkable: 2,
  });
});
