import assert from 'node:assert/strict';
import test from 'node:test';

import {
  DEPTH_DIAGNOSTICS_METHOD,
  deriveDepthDiagnostics,
} from '../src/lib/depthDiagnostics';

test('client diagnostics summarize only masked finite positive depth samples', () => {
  const summary = deriveDepthDiagnostics(
    new Float32Array([1, 2, 3, 4, 0, Number.NaN, 9, 8]),
    new Float32Array([0.2, 0.4, 0.6, 0.8, 0.9, 0.9, Number.NaN, 1]),
    new Uint8Array([1, 1, 1, 1, 1, 1, 1, 0]),
    [2, 4],
  );

  assert.equal(summary.sample_count, 4);
  assert.equal(summary.valid_ratio, 0.5);
  assert.equal(summary.median, 2.5);
  assert.ok(Math.abs(summary.p10 - 1.3) < 1e-6);
  assert.ok(Math.abs(summary.p90 - 3.7) < 1e-6);
  assert.ok(Math.abs(summary.conf_mean - 0.5) < 1e-6);
  assert.equal(summary.method, DEPTH_DIAGNOSTICS_METHOD);
});
