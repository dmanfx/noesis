import assert from 'node:assert/strict';
import test from 'node:test';

import { closeThinSurfaceNormalGapsInPlace } from '../src/lib/normalDisplay';

const setPixel = (
  rgba: Uint8ClampedArray,
  width: number,
  x: number,
  y: number,
  color: [number, number, number, number],
) => {
  rgba.set(color, (y * width + x) * 4);
};

const pixelAt = (
  rgba: Uint8ClampedArray,
  width: number,
  x: number,
  y: number,
): [number, number, number, number] => {
  const index = (y * width + x) * 4;
  return [
    rgba[index],
    rgba[index + 1],
    rgba[index + 2],
    rgba[index + 3],
  ];
};

test('surface display closes a thin transparent band without erasing its color boundary', () => {
  const width = 9;
  const rgba = new Uint8ClampedArray(width * 4);
  const left: [number, number, number, number] = [20, 80, 160, 255];
  const right: [number, number, number, number] = [200, 120, 40, 255];
  setPixel(rgba, width, 2, 0, left);
  setPixel(rgba, width, 6, 0, right);

  const filled = closeThinSurfaceNormalGapsInPlace(rgba, width, 1, 3);

  assert.equal(filled, 3);
  assert.deepEqual(pixelAt(rgba, width, 2, 0), left);
  assert.deepEqual(pixelAt(rgba, width, 3, 0), left);
  assert.deepEqual(pixelAt(rgba, width, 4, 0), right);
  assert.deepEqual(pixelAt(rgba, width, 5, 0), right);
  assert.deepEqual(pixelAt(rgba, width, 6, 0), right);
  assert.equal(pixelAt(rgba, width, 0, 0)[3], 0);
  assert.equal(pixelAt(rgba, width, 8, 0)[3], 0);
});

test('surface display does not grow normals into a broad unsupported region', () => {
  const width = 11;
  const rgba = new Uint8ClampedArray(width * 4);
  setPixel(rgba, width, 1, 0, [20, 80, 160, 255]);
  setPixel(rgba, width, 9, 0, [200, 120, 40, 255]);

  const filled = closeThinSurfaceNormalGapsInPlace(rgba, width, 1, 3);

  assert.equal(filled, 0);
  for (let x = 2; x < 9; x += 1) {
    assert.equal(pixelAt(rgba, width, x, 0)[3], 0);
  }
});

test('surface display rejects a mismatched RGBA shape', () => {
  assert.throws(
    () => closeThinSurfaceNormalGapsInPlace(new Uint8ClampedArray(7), 2, 1),
    /surface_normal_display_shape_invalid/,
  );
});
