import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  decodeFloat32,
  pcfRasterCompatibilityError,
} from './rasterEncoding.ts';

const fixture = JSON.parse(await readFile(
  new URL('../../../tests/fixtures/pcf_raster_encoding_v1.json', import.meta.url),
  'utf8',
));

test('decodes producer-pinned PCF float16, mask, category and legacy float32 grids', () => {
  assert.deepEqual(Array.from(decodeFloat32(fixture.float16.grid_b64)), [0, 0.5, 1.25, 2, -0.5, NaN]);
  assert.deepEqual(Array.from(decodeFloat32(fixture.mask.grid_b64)), [0, 1, 1, 1, 0, 1]);
  assert.deepEqual(Array.from(decodeFloat32(fixture.category.grid_b64)), [0, 1, 2, 2, 1, 0]);
  assert.deepEqual(Array.from(decodeFloat32(fixture.legacy_float32.grid_b64)), [0, 0.5, 1.25, 2, -0.5, 4]);
});

test('rejects unsupported PCF raster versions and a declared/payload encoding mismatch visibly', () => {
  const response = {
    scene_prior_only: true,
    display_source: 'pcf',
    scene_prior_diagnostic_meta: { raster_encoding_version: 1 },
    scene_prior_diagnostic_height: { ...fixture.float16, grid_shape: fixture.shape },
  };
  assert.equal(pcfRasterCompatibilityError(response), null);
  assert.match(pcfRasterCompatibilityError({
    ...response,
    scene_prior_diagnostic_meta: { raster_encoding_version: 2 },
  }), /Unsupported PCF raster encoding version 2/);
  assert.match(pcfRasterCompatibilityError({
    ...response,
    scene_prior_diagnostic_height: { ...fixture.float16, grid_encoding: 'f32-le-base64', grid_shape: fixture.shape },
  }), /Unsupported PCF raster encoding in scene_prior_diagnostic_height/);
  assert.match(pcfRasterCompatibilityError({
    ...response,
    scene_prior_diagnostic_height: { grid_b64: 'f64:AAAA', grid_shape: fixture.shape },
  }), /Unsupported PCF raster encoding in scene_prior_diagnostic_height/);
  assert.match(pcfRasterCompatibilityError({
    ...response,
    scene_prior_diagnostic_height: { ...fixture.float16, grid_shape: [2, 4] },
  }), /Invalid PCF raster size in scene_prior_diagnostic_height/);
});
