import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import test from 'node:test';

import {
  DEPTH_BULK_MAX_COMPONENT_BYTES,
  DEPTH_BULK_MAX_PIXELS,
  DEPTH_BULK_MAX_SNAPSHOT_BYTES,
} from '../src/lib/depthBulkLimits';

test('frontend dense-depth limits match the backend authority fixture', async () => {
  const payload = JSON.parse(await readFile(
    join(process.cwd(), '..', 'noesis_core', 'depth_bulk_limits.json'),
    'utf8',
  )) as Record<string, number>;
  assert.deepEqual(payload, {
    max_pixels: DEPTH_BULK_MAX_PIXELS,
    max_component_bytes: DEPTH_BULK_MAX_COMPONENT_BYTES,
    max_snapshot_bytes: DEPTH_BULK_MAX_SNAPSHOT_BYTES,
  });
});
