import assert from 'node:assert/strict';
import test from 'node:test';

import { deriveCalibratedDepthNormals } from '../src/lib/depthNormals';

const vectorAt = (
  normals: Float32Array,
  width: number,
  x: number,
  y: number,
): [number, number, number] => {
  const index = (y * width + x) * 3;
  return [normals[index], normals[index + 1], normals[index + 2]];
};

test('calibrated normals recover a front-facing metric plane', () => {
  const width = 9;
  const height = 9;
  const depth = new Float32Array(width * height).fill(2);
  const mask = new Uint8Array(width * height).fill(1);
  const normals = deriveCalibratedDepthNormals(
    depth,
    mask,
    height,
    width,
    [100, 100, 4, 4],
  );
  const [nx, ny, nz] = vectorAt(normals, width, 4, 4);
  assert.ok(Math.abs(nx) < 1e-6);
  assert.ok(Math.abs(ny) < 1e-6);
  assert.ok(nz < -0.999999, `expected camera-facing -Z normal, got ${nx},${ny},${nz}`);
});

test('calibrated normals recover a horizontal floor instead of collapsing toward Z', () => {
  const width = 80;
  const height = 60;
  const fx = 80;
  const fy = 80;
  const cx = (width - 1) / 2;
  const cy = -20;
  const depth = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    const z = fy / (y - cy);
    for (let x = 0; x < width; x += 1) depth[y * width + x] = z;
  }
  const mask = new Uint8Array(width * height).fill(1);
  const normals = deriveCalibratedDepthNormals(
    depth,
    mask,
    height,
    width,
    [fx, fy, cx, cy],
  );
  const [nx, ny, nz] = vectorAt(normals, width, 40, 30);
  assert.ok(Math.abs(nx) < 1e-4);
  assert.ok(ny > 0.9999, `expected upward camera-Y normal, got ${nx},${ny},${nz}`);
  assert.ok(Math.abs(nz) < 1e-4);
});

test('calibrated normals leave pixels without complete local support invalid', () => {
  const width = 9;
  const height = 9;
  const depth = new Float32Array(width * height).fill(2);
  const mask = new Uint8Array(width * height).fill(1);
  mask[4 * width + 2] = 0;
  const normals = deriveCalibratedDepthNormals(
    depth,
    mask,
    height,
    width,
    [100, 100, 4, 4],
  );
  assert.deepEqual(vectorAt(normals, width, 4, 4), [0, 0, 0]);
});
