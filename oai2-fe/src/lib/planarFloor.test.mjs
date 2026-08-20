import test from 'node:test';
import assert from 'node:assert/strict';
import { derivePlanarFloorMask } from './planarFloor.mjs';

const index = (row, col, cols) => (row * cols) + col;

test('planar floor fills an evidence-enclosed sampling gap', () => {
  const rows = 7;
  const cols = 7;
  const support = new Float32Array(rows * cols);
  for (let col = 1; col <= 5; col += 1) {
    support[index(1, col, cols)] = 1;
    support[index(5, col, cols)] = 1;
  }
  for (let row = 1; row <= 5; row += 1) {
    support[index(row, 1, cols)] = 1;
    support[index(row, 5, cols)] = 1;
  }

  const plane = derivePlanarFloorMask({ floorSupport: support, rows, cols, expansionCells: 0 });

  assert.equal(plane[index(3, 3, cols)], 1);
  assert.equal(plane[index(0, 0, cols)], 0);
  assert.equal(plane.reduce((sum, value) => sum + value, 0), 25);
});

test('bounded expansion does not turn an isolated sample into the whole grid', () => {
  const rows = 9;
  const cols = 9;
  const support = new Float32Array(rows * cols);
  support[index(4, 4, cols)] = 1;

  const plane = derivePlanarFloorMask({ floorSupport: support, rows, cols, expansionCells: 2 });

  assert.equal(plane[index(4, 4, cols)], 1);
  assert.equal(plane[index(1, 1, cols)], 0);
  assert.equal(plane.reduce((sum, value) => sum + value, 0), 25);
});

test('connected appendage is retained beyond the former authored rectangle', () => {
  const rows = 16;
  const cols = 24;
  const support = new Float32Array(rows * cols);
  for (let row = 4; row <= 11; row += 1) {
    for (let col = 4; col <= 12; col += 1) support[index(row, col, cols)] = 1;
  }
  for (let row = 7; row <= 9; row += 1) {
    for (let col = 13; col <= 20; col += 1) support[index(row, col, cols)] = 1;
  }

  const plane = derivePlanarFloorMask({ floorSupport: support, rows, cols, expansionCells: 0 });

  assert.equal(plane[index(8, 19, cols)], 1);
  assert.equal(plane[index(2, 20, cols)], 0);
  assert.equal(plane.reduce((sum, value) => sum + value, 0), 96);
});

test('tiny disconnected PCF fleck is not promoted into the floor plane', () => {
  const rows = 30;
  const cols = 30;
  const support = new Float32Array(rows * cols);
  for (let row = 8; row <= 17; row += 1) {
    for (let col = 8; col <= 17; col += 1) support[index(row, col, cols)] = 1;
  }
  support[index(2, 2, cols)] = 1;

  const plane = derivePlanarFloorMask({ floorSupport: support, rows, cols, expansionCells: 2 });

  assert.equal(plane[index(12, 12, cols)], 1);
  assert.equal(plane[index(2, 2, cols)], 0);
  assert.equal(plane.reduce((sum, value) => sum + value, 0), 196);
});
