import test from 'node:test';
import assert from 'node:assert/strict';

import { computeObservedFloorplanViewport } from './floorplanViewport.js';

test('observed viewport crops the display in metric padding without changing the grid', () => {
  const mask = new Float32Array(10 * 20);
  mask[(4 * 20) + 8] = 1;
  mask[(5 * 20) + 10] = 1;

  const result = computeObservedFloorplanViewport({
    maskValues: mask,
    rows: 10,
    cols: 20,
    bounds: { min_x: -10, max_x: 10, min_z: -5, max_z: 5 },
    paddingM: 1,
  });

  assert.ok(result);
  assert.deepEqual(result.observedRect, { x: 8, y: 4, width: 3, height: 2 });
  assert.deepEqual(result.sourceRect, { x: 7, y: 3, width: 5, height: 4 });
  assert.equal(result.cropWidthM, 5);
  assert.equal(result.cropDepthM, 4);
  assert.equal(result.totalCount, 200);
  assert.equal(mask[(4 * 20) + 8], 1);
});

test('unknown-layer inversion crops around finite zero cells and rejects non-finite cells', () => {
  const unknown = Float32Array.from([
    1, 1, 1, 1,
    1, 0, Number.NaN, 1,
    1, 0, 0, 1,
    1, 1, 1, 1,
  ]);
  const result = computeObservedFloorplanViewport({
    maskValues: unknown,
    rows: 4,
    cols: 4,
    maskInvert: true,
    paddingM: 0,
  });

  assert.ok(result);
  assert.equal(result.observedCount, 3);
  assert.deepEqual(result.sourceRect, { x: 1, y: 1, width: 2, height: 2 });
});

test('observed viewport returns null when no cell is explicitly observed', () => {
  assert.equal(
    computeObservedFloorplanViewport({
      maskValues: new Float32Array(16),
      rows: 4,
      cols: 4,
    }),
    null,
  );
});
