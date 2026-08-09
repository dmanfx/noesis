import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import {
  ADE20K_LABELS,
  semanticColor,
  semanticColorRgb,
  summarizeClassIds,
  toggleSemanticClass,
} from './semanticSegmentation.js';

test('ADE20K contract exposes all 150 labels and one stable color per label', () => {
  assert.equal(ADE20K_LABELS.length, 150);
  assert.equal(ADE20K_LABELS[0], 'wall');
  assert.equal(ADE20K_LABELS[149], 'flag');

  const colors = ADE20K_LABELS.map((_, classId) => semanticColor(classId));
  assert.equal(new Set(colors).size, 150);
  assert.deepEqual(semanticColorRgb(0), [4, 42, 255]);
});

test('class summaries include only present labels in descending coverage order', () => {
  assert.deepEqual(summarizeClassIds(Uint8Array.from([3, 3, 0, 24, 3, 0])), [
    { classId: 3, count: 3, fraction: 0.5 },
    { classId: 0, count: 2, fraction: 2 / 6 },
    { classId: 24, count: 1, fraction: 1 / 6 },
  ]);
  assert.deepEqual(summarizeClassIds(null), []);
});

test('semantic class selection toggles classes independently', () => {
  assert.deepEqual(toggleSemanticClass([], 0), [0]);
  assert.deepEqual(toggleSemanticClass([0], 3), [0, 3]);
  assert.deepEqual(toggleSemanticClass([0, 3], 0), [3]);
  assert.deepEqual(toggleSemanticClass([3], 3), []);
});

test('image clicks use the same independent toggle behavior as the legend', async () => {
  const component = await readFile(new URL('../components/SemanticSegView.tsx', import.meta.url), 'utf8');
  assert.match(component, /onClick=\{togglePixel\}/);
  assert.match(component, /toggleSemanticClass\(current, inspected\.classId\)/);
});

test('semantic inference is manual-only and the saved evidence remains the initial display', async () => {
  const component = await readFile(new URL('../components/SemanticSegView.tsx', import.meta.url), 'utf8');
  assert.match(component, /public|semantic-seg|ASSET_ROOT/);
  assert.match(component, /const refreshCapture = async/);
  assert.match(component, /onClick=\{refreshCapture\}/);
  assert.match(component, /\/api\/diagnostics\/semantic-seg\/captures/);
  assert.match(component, /'Idempotency-Key': `semseg-\$\{model\}-\$\{crypto\.randomUUID\(\)\}`/);
  assert.equal(component.match(/\/api\/diagnostics\/semantic-seg\/captures/g)?.length, 1);
  assert.doesNotMatch(component, /onChange=\{refreshCapture\}/);
  assert.doesNotMatch(component, /onRequestDepth|WebSocket/);
  assert.match(component, /type ModelSize = 's' \| 'l'/);
  assert.doesNotMatch(component, /Nano|Medium|value: 'n'|value: 'm'/);
});

test('semantic tab exposes all three camera captures and camera-bound assets', async () => {
  const component = await readFile(new URL('../components/SemanticSegView.tsx', import.meta.url), 'utf8');
  assert.match(component, /Semantic segmentation camera/);
  assert.match(component, /Living Room/);
  assert.match(component, /Kitchen/);
  assert.match(component, /Family Room/);
  assert.match(component, /saved well-lit anchor/);
  assert.match(component, /staticSourceUrl\(model, camera\)/);
  assert.match(component, /loadClassMap\(activeClassMapUrl\)/);
});
