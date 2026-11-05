import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
const built = await build({ entryPoints: [new URL('./spatialCameraLayout.ts', import.meta.url).pathname], bundle: true, write: false, platform: 'node', format: 'esm' });
const { rasterFootprint, cameraQuadrantExpansion } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
const overlaps = (a, b) => a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;
const stage = { width: 1600, height: 900 }, home = { x: 550, y: 210, width: 500, height: 520 };

test('every camera stays in its current quadrant, clear of the painted home', () => {
  for (const [x, y, quadrant] of [[20, 20, 'top-left'], [1200, 20, 'top-right'], [20, 600, 'bottom-left'], [1200, 600, 'bottom-right']]) {
    const r = cameraQuadrantExpansion(stage, { x, y, width: 300, height: 250 }, home);
    assert(r); assert.equal(r.quadrant, quadrant); assert(!overlaps(r, home));
    const left = quadrant.endsWith('left'), top = quadrant.startsWith('top');
    assert(r.x >= (left ? 12 : 806)); assert(r.x + r.width <= (left ? 794 : 1588));
    assert(r.y >= (top ? 12 : 456)); assert(r.y + r.height <= (top ? 444 : 888));
    assert(r.width >= 180 && r.height >= 154);
  }
});

test('a moved camera uses its new quadrant and responds to viewport resizing', () => {
  const rest = { x: 1100, y: 600, width: 300, height: 250 };
  assert.equal(cameraQuadrantExpansion(stage, rest, home).quadrant, 'bottom-right');
  const smallHome = { x: 330, y: 180, width: 380, height: 280 };
  const r = cameraQuadrantExpansion({ width: 1024, height: 640 }, { x: 750, y: 350, width: 250, height: 230 }, smallHome);
  assert(r); assert.equal(r.quadrant, 'bottom-right'); assert(r.x + r.width <= 1012); assert(r.y + r.height <= 628);
  assert(!overlaps(r, smallHome));
});

test('prefers useful video space over a wide strip too short for controls', () => {
  const r = cameraQuadrantExpansion(stage, { x: 20, y: 20, width: 300, height: 250 }, home);
  assert(r); assert.equal(r.x, 12); assert.equal(r.y, 12); assert.equal(r.width, 526); assert.equal(r.height, 432);
});

test('a blocked quadrant does not force a panel through the home', () => {
  assert.equal(cameraQuadrantExpansion(stage, { x: 20, y: 20, width: 300, height: 250 }, { x: 0, y: 0, width: 1600, height: 900 }), null);
});

test('transparent padding does not claim home image space; complete cell edges do', () => {
  const pixels = new Uint8ClampedArray(4 * 4 * 4); pixels[(1 * 4 + 2) * 4 + 3] = 255;
  const corners = [[0, 0, 0], [4, 0, 0], [4, 0, 4], [0, 0, 4]];
  assert.deepEqual(rasterFootprint(4, 4, pixels, corners), { minX: 2, maxX: 3, minZ: 2, maxZ: 3 });
  assert.equal(rasterFootprint(4, 4, new Uint8ClampedArray(64), corners), null);
  pixels[(3 * 4 + 3) * 4 + 3] = 255;
  assert.deepEqual(rasterFootprint(4, 4, pixels, corners), { minX: 2, maxX: 4, minZ: 0, maxZ: 3 });
});

test('rotated and reflected rasters preserve the renderer affine without room-sized padding', () => {
  const pixels = new Uint8ClampedArray(4 * 4 * 4); pixels[(1 * 4 + 2) * 4 + 3] = 255;
  const corners = [[6, 0, 16], [2, 0, 20], [6, 0, 24], [10, 0, 20]];
  assert.deepEqual(rasterFootprint(4, 4, pixels, corners), { minX: 5, maxX: 7, minZ: 20, maxZ: 22 });
});
