import assert from 'node:assert/strict';
import test from 'node:test';

import {
  deriveCalibratedDepthNormals,
  derivePlaneAwareSurfaceNormals,
} from '../src/lib/depthNormals';

const vectorAt = (
  normals: ArrayLike<number>,
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

const meanFrontPlaneAngularError = (
  normals: ArrayLike<number>,
  width: number,
  height: number,
  border: number,
): number => {
  let errorTotal = 0;
  let count = 0;
  for (let y = border; y < height - border; y += 1) {
    for (let x = border; x < width - border; x += 1) {
      const [nx, ny, nz] = vectorAt(normals, width, x, y);
      const magnitude = Math.hypot(nx, ny, nz);
      if (magnitude < 0.5) continue;
      const cosine = Math.max(-1, Math.min(1, -nz / magnitude));
      errorTotal += Math.acos(cosine);
      count += 1;
    }
  }
  assert.ok(count > (width - border * 2) * (height - border * 2) * 0.75);
  return errorTotal / count;
};

const meanHorizontalNormalVariation = (
  normals: ArrayLike<number>,
  width: number,
  height: number,
  border: number,
): number => {
  let angleTotal = 0;
  let count = 0;
  for (let y = border; y < height - border; y += 1) {
    for (let x = border; x < width - border - 1; x += 1) {
      const left = vectorAt(normals, width, x, y);
      const right = vectorAt(normals, width, x + 1, y);
      const leftMagnitude = Math.hypot(...left);
      const rightMagnitude = Math.hypot(...right);
      if (leftMagnitude < 0.5 || rightMagnitude < 0.5) continue;
      const dot = Math.max(
        -1,
        Math.min(
          1,
          (
            left[0] * right[0]
            + left[1] * right[1]
            + left[2] * right[2]
          ) / (leftMagnitude * rightMagnitude),
        ),
      );
      angleTotal += Math.acos(dot);
      count += 1;
    }
  }
  assert.ok(count > 0);
  return angleTotal / count;
};

test('confidence-aware multiscale normals suppress deterministic depth jitter', () => {
  const width = 96;
  const height = 72;
  const depth = new Float32Array(width * height);
  let state = 0x12345678;
  for (let index = 0; index < depth.length; index += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const noise = (state / 0xffffffff - 0.5) * 0.024;
    depth[index] = 3 + noise;
  }
  const confidence = new Float32Array(depth.length).fill(0.9);
  const mask = new Uint8Array(depth.length).fill(1);
  const intrinsics: [number, number, number, number] = [700, 700, 47.5, 35.5];
  const raw = deriveCalibratedDepthNormals(
    depth,
    mask,
    height,
    width,
    intrinsics,
    {
      confidence,
      bilateralRadius: 0,
      sampleRadius: 1,
      normalSmoothingRadius: 0,
    },
  );
  const enhanced = deriveCalibratedDepthNormals(
    depth,
    mask,
    height,
    width,
    intrinsics,
    {
      confidence,
      bilateralRadius: 2,
      sampleRadius: 4,
      normalSmoothingRadius: 1,
    },
  );
  const rawError = meanFrontPlaneAngularError(raw, width, height, 4);
  const enhancedError = meanFrontPlaneAngularError(enhanced, width, height, 4);
  assert.ok(
    enhancedError < rawError * 0.35,
    `expected substantial angular-noise reduction, got raw=${rawError}, enhanced=${enhancedError}`,
  );
});

test('edge-aware normals preserve a hard depth step instead of rounding across it', () => {
  const width = 48;
  const height = 32;
  const depth = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      depth[y * width + x] = x < width / 2 ? 2 : 3;
    }
  }
  const confidence = new Float32Array(depth.length).fill(0.95);
  const mask = new Uint8Array(depth.length).fill(1);
  const normals = deriveCalibratedDepthNormals(
    depth,
    mask,
    height,
    width,
    [500, 500, 23.5, 15.5],
    {
      confidence,
      bilateralRadius: 2,
      sampleRadius: 4,
      normalSmoothingRadius: 1,
    },
  );

  for (const x of [10, 20, 27, 38]) {
    const [nx, ny, nz] = vectorAt(normals, width, x, 16);
    assert.ok(Math.abs(nx) < 1e-4 && Math.abs(ny) < 1e-4 && nz < -0.9999);
  }
  for (const x of [23, 24]) {
    const [nx, ny, nz] = vectorAt(normals, width, x, 16);
    const magnitude = Math.hypot(nx, ny, nz);
    assert.ok(
      magnitude < 0.5 || (Math.abs(nx) < 0.05 && nz < -0.98),
      `depth edge was rounded into a false surface normal at x=${x}: ${nx},${ny},${nz}`,
    );
  }
});

test('plane-aware surface normals make a noisy plane piecewise constant', () => {
  const width = 96;
  const height = 72;
  const depth = new Float32Array(width * height);
  let state = 0x9e3779b9;
  for (let index = 0; index < depth.length; index += 1) {
    state = (Math.imul(state, 1103515245) + 12345) >>> 0;
    depth[index] = 3 + (state / 0xffffffff - 0.5) * 0.03;
  }
  const confidence = new Float32Array(depth.length).fill(0.95);
  const mask = new Uint8Array(depth.length).fill(1);
  const detail = deriveCalibratedDepthNormals(
    depth,
    mask,
    height,
    width,
    [700, 700, 47.5, 35.5],
    {
      confidence,
      bilateralRadius: 2,
      sampleRadius: 4,
      normalSmoothingRadius: 1,
    },
  );
  const surface = derivePlaneAwareSurfaceNormals(
    detail,
    depth,
    confidence,
    mask,
    height,
    width,
  );
  const detailVariation = meanHorizontalNormalVariation(detail, width, height, 8);
  const surfaceVariation = meanHorizontalNormalVariation(surface, width, height, 8);
  assert.ok(
    surfaceVariation < detailVariation * 0.15,
    `expected a nearly constant surface field, got detail=${detailVariation}, surface=${surfaceVariation}`,
  );
  const surfaceError = meanFrontPlaneAngularError(surface, width, height, 8);
  assert.ok(surfaceError < 0.03, `surface plane tilted by ${surfaceError} radians`);
});

test('plane-aware surface normals do not merge distinct planes across a depth edge', () => {
  const width = 64;
  const height = 48;
  const pixelCount = width * height;
  const depth = new Float32Array(pixelCount);
  const confidence = new Float32Array(pixelCount).fill(0.95);
  const mask = new Uint8Array(pixelCount).fill(1);
  const detail = new Float32Array(pixelCount * 3);
  const leftNormal: [number, number, number] = [0, 0, -1];
  const rightNormal: [number, number, number] = [0.6, 0, -0.8];
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const pixelIndex = y * width + x;
      const expected = x < width / 2 ? leftNormal : rightNormal;
      depth[pixelIndex] = x < width / 2 ? 2 : 3;
      const jitter = ((x * 17 + y * 13) % 9 - 4) * 0.004;
      const nx = expected[0] + jitter;
      const ny = expected[1] - jitter * 0.5;
      const nz = expected[2];
      const magnitude = Math.hypot(nx, ny, nz);
      const normalIndex = pixelIndex * 3;
      detail[normalIndex] = nx / magnitude;
      detail[normalIndex + 1] = ny / magnitude;
      detail[normalIndex + 2] = nz / magnitude;
    }
  }
  const surface = derivePlaneAwareSurfaceNormals(
    detail,
    depth,
    confidence,
    mask,
    height,
    width,
  );
  const left = vectorAt(surface, width, 16, 24);
  const right = vectorAt(surface, width, 48, 24);
  const leftMagnitude = Math.hypot(...left);
  const rightMagnitude = Math.hypot(...right);
  const leftDot = (
    left[0] * leftNormal[0]
    + left[1] * leftNormal[1]
    + left[2] * leftNormal[2]
  ) / leftMagnitude;
  const rightDot = (
    right[0] * rightNormal[0]
    + right[1] * rightNormal[1]
    + right[2] * rightNormal[2]
  ) / rightMagnitude;
  assert.ok(leftDot > 0.999, `left plane changed orientation: ${left}`);
  assert.ok(rightDot > 0.999, `right plane changed orientation: ${right}`);
  assert.ok(
    (
      left[0] * right[0]
      + left[1] * right[1]
      + left[2] * right[2]
    ) / (leftMagnitude * rightMagnitude) < 0.9,
    'distinct planes were merged',
  );
});
