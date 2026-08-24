import assert from 'node:assert/strict';
import test from 'node:test';

import {
  resolveBevDisplayBounds,
  resolveBevMetricPoint,
} from './bevDisplayGeometry.js';

const pcfBounds = {
  min_x: -4,
  max_x: 4,
  min_z: 0,
  max_z: 8,
};

const producerDisplayBounds = {
  min_x: -8,
  max_x: 8,
  min_z: -4,
  max_z: 12,
};

test('camera-local dashboard uses producer displayBounds for exact out-of-PCF metric points', () => {
  const displayBounds = resolveBevDisplayBounds({
    floorplanBounds: pcfBounds,
    advertisedBounds: producerDisplayBounds,
  });
  assert.deepEqual(displayBounds, producerDisplayBounds);

  for (const x of [5, 7.5]) {
    const resolved = resolveBevMetricPoint({
      point: { x, y: 2, canonicalWorld: true, floorplanInside: false },
      displayBounds,
      normalizedBounds: pcfBounds,
    });
    assert.deepEqual(resolved, { x, y: 2, mapped: false });
  }
});

test('canonical points outside a coverage polygon follow the producer display contract', () => {
  const coverage = {
    boundaryToleranceM: 0,
    bounds: { min_x: -1, max_x: 3, min_z: 0, max_z: 4 },
    regions: [{ id: 'main', polygonXZ: [[0, 0], [1, 0], [1, 1], [0, 1]] }],
  };
  const displayBounds = resolveBevDisplayBounds({
    floorplanBounds: pcfBounds,
    advertisedBounds: { min_x: -4, max_x: 4, min_z: 0, max_z: 8 },
    coverageBounds: coverage.bounds,
    coverageToleranceM: coverage.boundaryToleranceM,
  });

  const canonical = resolveBevMetricPoint({
    point: { x: 2, y: 2, canonicalWorld: true, coverageInside: false },
    displayBounds,
    normalizedBounds: pcfBounds,
    coverage,
  });
  assert.deepEqual(canonical, { x: 2, y: 2, mapped: false });

  const legacy = resolveBevMetricPoint({
    point: { x: 2, y: 2, canonicalWorld: false, coverageInside: false },
    displayBounds,
    normalizedBounds: pcfBounds,
    coverage,
  });
  assert.equal(legacy, null);
});

test('a retained canonical trail remains displayable when the current footpoint is absent', () => {
  const coverage = {
    boundaryToleranceM: 0,
    bounds: { min_x: -1, max_x: 3, min_z: 0, max_z: 4 },
    regions: [{ id: 'main', polygonXZ: [[0, 0], [1, 0], [1, 1], [0, 1]] }],
  };
  const displayBounds = resolveBevDisplayBounds({
    floorplanBounds: pcfBounds,
    advertisedBounds: { min_x: -4, max_x: 4, min_z: 0, max_z: 8 },
    coverageBounds: coverage.bounds,
    coverageToleranceM: coverage.boundaryToleranceM,
  });

  // This is a retained trail sample, not a current footpoint.  Its producer
  // marker is the only authority available during the current-footpoint gap.
  const retainedCanonicalTrailPoint = resolveBevMetricPoint({
    point: { x: 2, y: 2, canonicalWorld: true, coverageInside: false },
    displayBounds,
    normalizedBounds: pcfBounds,
    coverage,
  });
  assert.deepEqual(retainedCanonicalTrailPoint, { x: 2, y: 2, mapped: false });

  // A legacy/unmarked retained sample cannot bypass the coverage gate.
  const unmarkedTrailPoint = resolveBevMetricPoint({
    point: { x: 2, y: 2, coverageInside: false },
    displayBounds,
    normalizedBounds: pcfBounds,
    coverage,
  });
  assert.equal(unmarkedTrailPoint, null);
});

test('a producer displayBounds narrower than the semantic raster is not accepted', () => {
  const displayBounds = resolveBevDisplayBounds({
    floorplanBounds: pcfBounds,
    advertisedBounds: { min_x: -2, max_x: 2, min_z: 1, max_z: 7 },
  });
  assert.deepEqual(displayBounds, pcfBounds);
});
