import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildVisibleFloorPlaneModel,
  deriveCameraDepthNormal,
  floorplanMatchesDepthSnapshot,
} from '../src/lib/visibleFloorPlane';

const WIDTH = 80;
const HEIGHT = 60;
const FX = 80;
const FY = 80;
const CX = (WIDTH - 1) / 2;
const CY = -20;

const flatFloorDepth = (): Float32Array => {
  const depth = new Float32Array(WIDTH * HEIGHT);
  for (let v = 0; v < HEIGHT; v += 1) {
    // With the UI camera convention y = -(v-cy)*z/fy, this is y=-1m.
    const z = FY / (v - CY);
    for (let u = 0; u < WIDTH; u += 1) depth[v * WIDTH + u] = z;
  }
  return depth;
};

test('metric depth cross-product recovers a flat floor normal', () => {
  const depth = flatFloorDepth();
  const mask = new Uint8Array(WIDTH * HEIGHT).fill(1);
  const normal = deriveCameraDepthNormal({
    depth,
    mask,
    width: WIDTH,
    height: HEIGHT,
    u: 40,
    v: 30,
    fx: FX,
    fy: FY,
    cx: CX,
    cy: CY,
  });

  assert.ok(normal);
  assert.ok(Math.abs(normal[1]) > 0.9999, `expected camera Y normal, got ${normal}`);
  assert.ok(Math.abs(normal[0]) < 1e-4);
  assert.ok(Math.abs(normal[2]) < 1e-4);
});

test('metric depth normal rejects an invalid neighbor', () => {
  const depth = flatFloorDepth();
  const mask = new Uint8Array(WIDTH * HEIGHT).fill(1);
  mask[30 * WIDTH + 39] = 0;

  assert.equal(deriveCameraDepthNormal({
    depth,
    mask,
    width: WIDTH,
    height: HEIGHT,
    u: 40,
    v: 30,
    fx: FX,
    fy: FY,
    cx: CX,
    cy: CY,
  }), null);
});

test('visible-floor extraction uses typed depth geometry without display normals', () => {
  const depth = flatFloorDepth();
  const result = buildVisibleFloorPlaneModel({
    cameraId: 'test-camera',
    depthEntry: {
      ts: 1_780_000_000_000_000,
      depth,
      conf: new Float32Array(WIDTH * HEIGHT).fill(1),
      mask: new Uint8Array(WIDTH * HEIGHT).fill(1),
      shape: [HEIGHT, WIDTH],
    },
    intrinsics: [FX, FY, CX, CY],
    extrinsics: [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ],
  });

  assert.equal(result.status, 'ready', result.message);
  assert.ok(result.model);
  assert.equal(result.model.normalSpace, 'camera');
  assert.ok(result.model.metrics.meanHorizontalDot > 0.9999);
  assert.ok(Math.abs(result.model.plane.worldHeightM + 1) < 1e-4);
  assert.ok(result.model.metrics.visibleFloorPixelCount > 1000);
});

test('floorplan footprint is used only for the exact depth snapshot identity', () => {
  const depthEntry = {
    ts: 1_780_000_000_000_000,
    depth: flatFloorDepth(),
    conf: new Float32Array(WIDTH * HEIGHT).fill(1),
    mask: new Uint8Array(WIDTH * HEIGHT).fill(1),
    shape: [HEIGHT, WIDTH] as [number, number],
    snapshotId: 'snapshot-a',
    snapshotRef: 'test-camera/snapshot-a.zarr',
    snapshotContentSha256: 'a'.repeat(64),
  };
  const exactFloorplan = {
    snapshot_ts: depthEntry.ts,
    snapshot_id: depthEntry.snapshotId,
    snapshot_ref: depthEntry.snapshotRef,
    snapshot_content_sha256: depthEntry.snapshotContentSha256,
    units: 'meters',
    bounds: { min_x: -5, max_x: 5, min_z: 0, max_z: 8 },
  };
  const common = {
    cameraId: 'test-camera',
    depthEntry,
    intrinsics: [FX, FY, CX, CY],
    extrinsics: [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ],
  };

  assert.equal(floorplanMatchesDepthSnapshot(depthEntry, exactFloorplan), true);
  const exact = buildVisibleFloorPlaneModel({ ...common, floorplan: exactFloorplan });
  assert.equal(exact.status, 'ready', exact.message);
  assert.equal(exact.model?.footprint.source, 'floorplan_bounds');

  const mismatchedFloorplan = { ...exactFloorplan, snapshot_id: 'snapshot-b' };
  assert.equal(floorplanMatchesDepthSnapshot(depthEntry, mismatchedFloorplan), false);
  const mismatched = buildVisibleFloorPlaneModel({
    ...common,
    floorplan: mismatchedFloorplan,
  });
  assert.equal(mismatched.status, 'ready', mismatched.message);
  assert.equal(mismatched.model?.footprint.source, 'visible_support');
  assert.match(mismatched.message, /snapshot identity does not match/);
});
