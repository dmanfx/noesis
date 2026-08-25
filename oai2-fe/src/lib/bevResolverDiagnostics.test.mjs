import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import test from 'node:test';
import ts from 'typescript';

const here = dirname(fileURLToPath(import.meta.url));
const source = await readFile(join(here, 'bevResolverDiagnostics.ts'), 'utf8');
const transpiled = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
}).outputText;
const moduleUrl = `data:text/javascript;base64,${Buffer.from(transpiled).toString('base64')}`;
const {
  admitResolverDiagnostics,
  requestLocalizationDetailsChange,
  validResolverCovariance,
} = await import(moduleUrl);

const covariance = [[0.04, 0.01], [0.01, 0.09]];
const diagnostic = {
  contract: 'noesis.world_resolver_diagnostics',
  version: 1,
  frameId: 42,
  sourceId: 20,
  sensorId: 2,
  worldFrame: 'backend_world_m',
  worldFrameRevision: 'world-r7',
  pcfRevision: 'vt-family-r3',
  floorplanSnapshotId: 'pcf-a',
  floorplanSnapshotContentSha256: 'a'.repeat(64),
  floorplanCalibrationFingerprint: 'b'.repeat(64),
  floorplanWorldFrame: 'backend_world_m',
  floorplanWorldFrameRevision: 'world-r7',
  resolved: {
    world: { x: 1, z: 3 },
    display: { x: 1, z: 3 },
    covarianceXZ: covariance,
  },
  candidates: [{
    id: 'floor-1',
    kind: 'floor_ray',
    world: { x: 1, z: 3 },
    display: { x: 1, z: 3 },
    covarianceXZ: covariance,
    selected: true,
    pcf: {
      insideExtent: false,
      insideAuthoredSpace: false,
      extentOutsideDistanceM: 0.867,
    },
  }],
};
const point = { frameId: 42, resolverDiagnostics: diagnostic };
const payload = {
  sourceId: 2,
  frameId: 42,
  canonicalWorldFrame: 'backend_world_m',
  canonicalWorldFrameRevision: 'world-r7',
  floorplanSnapshotId: 'pcf-a',
  floorplanSnapshotContentSha256: 'a'.repeat(64),
  floorplanCalibrationFingerprint: 'b'.repeat(64),
};

test('resolver admission accepts only the exact current cohort and revision', () => {
  const admitted = admitResolverDiagnostics(point, payload);
  assert.ok(admitted);
  assert.equal(admitted.frameId, 42);
  assert.deepEqual(admitted.resolved.covarianceXZ, covariance);
  assert.equal(admitted.candidates.length, 1);
  assert.equal(admitted.candidates[0].pcf.extentOutsideDistanceM, 0.867);
});

test('resolver admission clears on world or floorplan revision mismatch', () => {
  assert.equal(
    admitResolverDiagnostics(point, { ...payload, canonicalWorldFrameRevision: 'world-old' }),
    null,
  );
  assert.equal(
    admitResolverDiagnostics(point, { ...payload, floorplanSnapshotId: 'pcf-old' }),
    null,
  );
  assert.equal(
    admitResolverDiagnostics(point, { ...payload, sourceId: 3 }),
    null,
  );
  assert.equal(
    admitResolverDiagnostics({
      ...point,
      resolverDiagnostics: { ...diagnostic, floorplanWorldFrameRevision: 'world-old' },
    }, payload),
    null,
  );
  assert.equal(admitResolverDiagnostics({ frameId: 43, resolverDiagnostics: diagnostic }, payload), null);
  assert.equal(admitResolverDiagnostics({ frameId: 42 }, payload), null);
  assert.equal(
    admitResolverDiagnostics(point, {
      ...payload,
      canonicalWorldFrame: undefined,
      frame: 'backend_world_m',
    }),
    null,
  );
  assert.equal(
    admitResolverDiagnostics({
      ...point,
      resolverDiagnostics: { ...diagnostic, contract: 'resolver_diagnostics' },
    }, payload),
    null,
  );
});

test('covariance admission rejects asymmetric, indefinite, and non-finite matrices', () => {
  assert.equal(validResolverCovariance(covariance), true);
  assert.equal(validResolverCovariance([[1, 2], [0, 1]]), false);
  assert.equal(validResolverCovariance([[1, 2], [2, 1]]), false);
  assert.equal(validResolverCovariance([[1, Number.NaN], [Number.NaN, 1]]), false);
});

test('diagnostic admission rejects malformed numeric display evidence', () => {
  assert.equal(
    admitResolverDiagnostics({ ...point, resolverDiagnostics: undefined }, payload),
    null,
  );
  assert.equal(
    admitResolverDiagnostics({
      ...point,
      resolverDiagnostics: { ...diagnostic, disagreement: { distanceM: 'far' } },
    }, payload),
    null,
  );
});

test('localization detail requests update the controlled UI only after a successful send', () => {
  const updates = [];
  assert.equal(requestLocalizationDetailsChange(true, () => true, (value) => updates.push(value)), true);
  assert.deepEqual(updates, [true]);

  assert.equal(requestLocalizationDetailsChange(false, () => false, (value) => updates.push(value)), false);
  assert.deepEqual(updates, [true]);
});
