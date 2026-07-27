import test from 'node:test';
import assert from 'node:assert/strict';

import { buildCalibratedPointCloud } from './pointCloudModel.js';

test('backprojected points reproject to their source pixels', () => {
  const model = buildCalibratedPointCloud({
    depthValues: Float32Array.from([2, 2, 2, 2]),
    confidenceValues: Float32Array.from([1, 1, 1, 1]),
    maskValues: Uint8Array.from([1, 1, 1, 1]),
    width: 2,
    height: 2,
    intrinsics: [100, 120, 0.5, 0.5],
  });

  assert.ok(model);
  assert.equal(model.count, 4);
  const [fx, fy, cx, cy] = model.intrinsics;
  for (let idx = 0; idx < model.count; idx += 1) {
    const x = model.positions[idx * 3];
    const y = model.positions[idx * 3 + 1];
    const z = model.positions[idx * 3 + 2];
    const u = (x * fx / z) + cx;
    const v = (-y * fy / z) + cy;
    assert.ok(Math.abs(u - model.sourcePixels[idx * 2]) < 1e-6);
    assert.ok(Math.abs(v - model.sourcePixels[idx * 2 + 1]) < 1e-6);
  }
});

test('point cloud respects masks, confidence, and the hard point cap', () => {
  const width = 1000;
  const height = 1000;
  const depth = new Float32Array(width * height);
  const confidence = new Float32Array(width * height);
  const mask = new Uint8Array(width * height);
  depth.fill(2);
  confidence.fill(0.8);
  mask.fill(1);
  mask[0] = 0;
  confidence[1] = 0.1;

  const model = buildCalibratedPointCloud({
    depthValues: depth,
    confidenceValues: confidence,
    maskValues: mask,
    width,
    height,
    intrinsics: [500, 500, 500, 500],
    maxPoints: 10_000,
    minConfidence: 0.2,
  });

  assert.ok(model);
  assert.ok(model.stride >= 10);
  assert.ok(model.count <= 10_000);
  assert.ok(model.sourcePixels[0] !== 0 || model.sourcePixels[1] !== 0);
});

test('point cloud carries exact RGB from each retained source pixel', () => {
  const rgb = Uint8Array.from([
    10, 20, 30,
    40, 50, 60,
    70, 80, 90,
    100, 110, 120,
  ]);
  const model = buildCalibratedPointCloud({
    depthValues: Float32Array.from([2, 2, 2, 2]),
    maskValues: Uint8Array.from([1, 0, 1, 1]),
    rgbValues: rgb,
    rgbShape: [2, 2, 3],
    width: 2,
    height: 2,
    intrinsics: [100, 100, 0.5, 0.5],
  });

  assert.ok(model);
  assert.deepEqual(Array.from(model.sourcePixels), [0, 0, 0, 1, 1, 1]);
  assert.deepEqual(
    Array.from(model.rgbColors),
    [10, 20, 30, 70, 80, 90, 100, 110, 120],
  );
});

test('point cloud refuses mismatched RGB geometry instead of miscoloring points', () => {
  const model = buildCalibratedPointCloud({
    depthValues: Float32Array.from([2, 2, 2, 2]),
    rgbValues: new Uint8Array(12),
    rgbShape: [1, 4, 3],
    width: 2,
    height: 2,
    intrinsics: [100, 100, 0.5, 0.5],
  });

  assert.ok(model);
  assert.equal(model.rgbColors, null);
});

test('250k quality cap retains a dense bounded sample of a 1080p payload', () => {
  const width = 1920;
  const height = 1080;
  const depth = new Float32Array(width * height);
  const mask = new Uint8Array(width * height);
  depth.fill(3);
  mask.fill(1);
  const model = buildCalibratedPointCloud({
    depthValues: depth,
    maskValues: mask,
    width,
    height,
    intrinsics: [1000, 1000, 959.5, 539.5],
    maxPoints: 250_000,
  });

  assert.ok(model);
  assert.equal(model.stride, 3);
  assert.equal(model.count, 230_400);
  assert.ok(model.count <= 250_000);
});
