import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';

const sourceUrl = new URL('./camera.ts', import.meta.url);
const source = await readFile(sourceUrl, 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
  fileName: 'camera.ts',
}).outputText;
const camera = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);

test('calibration frame bindings contribute cameras without becoming a camera', () => {
  const discovered = camera.discoverCamerasFromPayloads([{
    data: {
      cameras: {
        K: { 'living-room': [1, 1, 1, 1], garage: [1, 1, 1, 1] },
        E: { 'living-room': [], garage: [] },
        frame_bindings: {
          'living-room': { contract: 'noesis.calibration.frame_binding' },
          garage: { contract: 'noesis.calibration.frame_binding' },
        },
      },
    },
  }]);

  assert.equal(discovered.includes('frame_bindings'), false);
  assert.equal(discovered.includes('garage'), true);
  assert.deepEqual(discovered.slice(0, 3), ['living-room', 'kitchen', 'family-room']);
});

test('flat dynamic camera entries remain discoverable', () => {
  const discovered = camera.discoverCamerasFromPayloads([{
    cameras: {
      garage: { intrinsics: [1, 1, 1, 1] },
    },
  }]);

  assert.equal(discovered.includes('garage'), true);
});
