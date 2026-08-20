import test from 'node:test';
import assert from 'node:assert/strict';

import { buildMaskedHeightfield } from './heightfieldModel.js';

test('heightfield preserves valid boundary triangles without crossing an unknown cell', () => {
  const heights = Float32Array.from([
    0, 0, 0,
    0, 1, 0,
    0, 0, 0,
  ]);
  const density = Float32Array.from([
    1, 1, 1,
    1, 0, 1,
    1, 1, 1,
  ]);
  const model = buildMaskedHeightfield({
    heightValues: heights,
    densityValues: density,
    rows: 3,
    cols: 3,
  });

  assert.ok(model);
  assert.equal(model.observedCount, 8);
  assert.equal(model.triangleCount, 4);
  for (const index of model.indices) {
    assert.notEqual(index, 4);
  }
});

test('heightfield emits two triangles per fully observed quad', () => {
  const model = buildMaskedHeightfield({
    heightValues: Float32Array.from([0, 0.2, 0.4, 0.6]),
    densityValues: Float32Array.from([1, 1, 1, 1]),
    rows: 2,
    cols: 2,
    bounds: { min_x: -1, max_x: 1, min_z: 2, max_z: 4 },
  });

  assert.ok(model);
  assert.equal(model.triangleCount, 2);
  assert.equal(model.positions[0], -0.5);
  assert.equal(model.positions[2], 3.5);
  assert.equal(model.positions[9], 0.5);
  assert.equal(model.positions[11], 2.5);
  assert.ok(Array.from(model.heights).every(
    (value, index) => Math.abs(value - [0, 0.2, 0.4, 0.6][index]) < 1e-6,
  ));
  assert.equal(model.aggregation.reducer, 'exact');
});

test('heightfield maps asymmetric far/right landmarks without a mirror or half-turn', () => {
  const heights = new Float32Array(3 * 4);
  heights[(2 * 4) + 0] = 0.25; // near-left
  heights[(0 * 4) + 0] = 1.25; // forward-left
  heights[(2 * 4) + 3] = 2.25; // near-right
  const density = new Float32Array(3 * 4);
  density.fill(1);
  const model = buildMaskedHeightfield({
    heightValues: heights,
    densityValues: density,
    rows: 3,
    cols: 4,
    bounds: { min_x: 0, max_x: 4, min_z: 0, max_z: 3 },
  });

  assert.ok(model);
  const nearLeft = (2 * 4) + 0;
  const forwardLeft = (0 * 4) + 0;
  const nearRight = (2 * 4) + 3;
  assert.deepEqual(
    [model.positions[nearLeft * 3], model.positions[(nearLeft * 3) + 2]],
    [0.5, 0.5],
  );
  assert.deepEqual(
    [model.positions[forwardLeft * 3], model.positions[(forwardLeft * 3) + 2]],
    [0.5, 2.5],
  );
  assert.deepEqual(
    [model.positions[nearRight * 3], model.positions[(nearRight * 3) + 2]],
    [3.5, 0.5],
  );
});

test('heightfield caps vertices and preserves supported peaks when downsampling', () => {
  const rows = 400;
  const cols = 400;
  const heights = new Float32Array(rows * cols);
  const density = new Float32Array(rows * cols);
  density.fill(1);
  heights[(rows - 1) * cols + (cols - 1)] = 2.5;
  heights[(rows - 1) * cols + (cols - 2)] = 2.5;

  const model = buildMaskedHeightfield({
    heightValues: heights,
    densityValues: density,
    rows,
    cols,
    maxVertices: 65_536,
  });

  assert.ok(model);
  assert.ok(model.rows * model.cols <= 65_536);
  assert.ok(model.downsampleFactor > 1);
  assert.equal(model.maxHeightM, 2.5);
});

test('heightfield downsampling suppresses an isolated source-cell spike', () => {
  const heights = new Float32Array(4 * 4);
  heights[0] = 8;
  const density = new Float32Array(4 * 4);
  density.fill(1);

  const model = buildMaskedHeightfield({
    heightValues: heights,
    densityValues: density,
    rows: 4,
    cols: 4,
    maxVertices: 4,
  });

  assert.ok(model);
  assert.equal(model.downsampleFactor, 2);
  assert.equal(model.maxHeightM, 0);
  assert.equal(model.aggregation.reducer, 'upper_p90');
});

test('heightfield downsampling preserves a single unknown source cell', () => {
  const rows = 4;
  const cols = 4;
  const heights = new Float32Array(rows * cols);
  const density = new Float32Array(rows * cols);
  density.fill(1);
  density[1 * cols + 1] = 0;

  const model = buildMaskedHeightfield({
    heightValues: heights,
    densityValues: density,
    rows,
    cols,
    maxVertices: 4,
  });

  assert.ok(model);
  assert.equal(model.downsampleFactor, 2);
  assert.deepEqual(Array.from(model.observed), [0, 1, 1, 1]);
  assert.deepEqual(Array.from(model.coverage), [0.75, 1, 1, 1]);
  assert.equal(model.aggregation.coverageThreshold, 1);
  assert.equal(model.aggregation.maxSourceBlockRows, 2);
  assert.equal(model.aggregation.maxSourceBlockCols, 2);
  assert.equal(model.observedCount, 3);
  assert.equal(model.triangleCount, 1);
  assert.ok(Array.from(model.indices).every((index) => index !== 0));
});

test('heightfield chooses the lower-discontinuity diagonal for a fully observed quad', () => {
  const model = buildMaskedHeightfield({
    heightValues: Float32Array.from([0, 4, 3, 0.1]),
    densityValues: Float32Array.from([1, 1, 1, 1]),
    rows: 2,
    cols: 2,
  });

  assert.ok(model);
  assert.deepEqual(Array.from(model.indices), [0, 2, 3, 0, 3, 1]);
});

test('heightfield downsampling places odd source blocks at their metric centroids', () => {
  const rows = 267;
  const cols = 267;
  const heights = new Float32Array(rows * cols);
  const density = new Float32Array(rows * cols);
  density.fill(1);
  const bounds = { min_x: -20, max_x: 20.05, min_z: 3, max_z: 43.05 };

  const model = buildMaskedHeightfield({
    heightValues: heights,
    densityValues: density,
    rows,
    cols,
    bounds,
    maxVertices: 65_536,
  });

  assert.ok(model);
  assert.equal(model.rows, 134);
  assert.equal(model.cols, 134);

  const targetRow = 1;
  const targetCol = 1;
  const sourceRow0 = Math.floor((targetRow * rows) / model.rows);
  const sourceRow1 = Math.floor(((targetRow + 1) * rows) / model.rows);
  const sourceCol0 = Math.floor((targetCol * cols) / model.cols);
  const sourceCol1 = Math.floor(((targetCol + 1) * cols) / model.cols);
  const sourceDx = (bounds.max_x - bounds.min_x) / cols;
  const sourceDz = (bounds.max_z - bounds.min_z) / rows;
  const idx = (targetRow * model.cols) + targetCol;

  assert.ok(Math.abs(
    model.positions[idx * 3]
      - (bounds.min_x + (((sourceCol0 + sourceCol1) * 0.5) * sourceDx)),
  ) < 1e-6);
  assert.ok(Math.abs(
    model.positions[(idx * 3) + 2]
      - (bounds.max_z - (((sourceRow0 + sourceRow1) * 0.5) * sourceDz)),
  ) < 1e-6);
});

test('heightfield presentation cutoff leaves overhead cells open', () => {
  const model = buildMaskedHeightfield({
    heightValues: Float32Array.from([0.4, 2.1, 0.7, 2.2]),
    densityValues: Float32Array.from([1, 1, 1, 1]),
    rows: 2,
    cols: 2,
    maxVisibleHeightM: 1.95,
  });

  assert.ok(model);
  assert.deepEqual(Array.from(model.observed), [1, 0, 1, 0]);
  assert.equal(model.observedCount, 2);
  assert.equal(model.triangleCount, 0);
  assert.ok(Math.abs(model.maxHeightM - 0.7) < 1e-6);
});
