import test from 'node:test';
import assert from 'node:assert/strict';

import { detectOverheadBand } from './overheadPresentation.js';

test('detects a concentrated room-height shell and returns a clearance cutoff', () => {
  const heights = [];
  for (let index = 0; index < 800; index += 1) heights.push((index % 180) / 100);
  for (let index = 0; index < 180; index += 1) heights.push(2.1 + ((index % 5) * 0.008));

  const result = detectOverheadBand(Float32Array.from(heights));

  assert.ok(result);
  assert.ok(result.planeHeightM >= 2.1 && result.planeHeightM <= 2.15);
  assert.ok(result.cutoffM >= 1.94 && result.cutoffM <= 2.01);
  assert.ok(result.hiddenCount >= 180);
});

test('reads interleaved point heights by stride and offset', () => {
  const positions = [];
  for (let index = 0; index < 500; index += 1) {
    positions.push(index / 10, (index % 150) / 100, index / 20);
  }
  for (let index = 0; index < 120; index += 1) {
    positions.push(index / 10, 2.12 + ((index % 3) * 0.005), index / 20);
  }

  const result = detectOverheadBand(Float32Array.from(positions), { stride: 3, offset: 1 });

  assert.ok(result);
  assert.ok(result.cutoffM < 2.05);
  assert.equal(result.sampleCount, 620);
});

test('does not classify a broad high distribution as an overhead plane', () => {
  const heights = Float32Array.from(
    { length: 800 },
    (_unused, index) => 1.75 + ((index % 180) / 100),
  );

  assert.equal(detectOverheadBand(heights), null);
});

test('requires enough samples to suppress a shell', () => {
  assert.equal(detectOverheadBand(Float32Array.from([2.1, 2.1, 2.1])), null);
});
