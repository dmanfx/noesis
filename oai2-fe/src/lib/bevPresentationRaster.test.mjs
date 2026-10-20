import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';

const built = await build({ entryPoints: [new URL('./bevPresentationRaster.ts', import.meta.url).pathname],
  bundle: true, write: false, format: 'esm', platform: 'node' });
const { createBevPresentationRaster, combinedBevHeightRange, bevPaletteColor } = await import(
  `data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const f32 = values => ({ grid_shape: [3, 3], grid_b64: Buffer.from(new Float32Array(values).buffer).toString('base64') });
// Synthetic 3x3 static evidence; no live/physical placement claim.
const fixture = ({ heights = [0, 1, 2, 0, 1, 2, 0, 1, 2], observed = Array(9).fill(1), matrix = identity } = {}) => {
  const heightLayer = f32(heights);
  const raster = { cameraId: 'fixture', mapId: 'pcf-fixture', snapshotContentSha256: 'e'.repeat(64),
    layer: heightLayer, heightLayer, mask: f32(observed),
    surfaceRgb: { rgb_b64: Buffer.from(Array.from({ length: 27 }, (_, n) => n * 9)).toString('base64'),
      rgb_shape: [3, 3, 3], observed_b64: f32([1, 0, 0, 0, 1, 0, 0, 0, 1]).grid_b64 } };
  const map = { cameraId: raster.cameraId, mapId: raster.mapId, snapshotContentSha256: raster.snapshotContentSha256,
    gridShape: [3, 3], localBounds: { min_x: 0, max_x: 3, min_z: 0, max_z: 3 },
    sceneFromLocalColMajor: matrix,
    sceneCorners: [[0, 0, 0], [3, 0, 0], [3, 0, 3], [0, 0, 3]] };
  return { raster, map };
};
const pixel = (image, index) => Array.from(image.pixels.slice(index * 4, index * 4 + 4));
const textOf = node => typeof node === 'string' || typeof node === 'number' ? String(node)
  : Array.isArray(node) ? node.map(textOf).join('') : node && typeof node === 'object' ? textOf(node.props?.children ?? node.children) : '';

test('structure preserves observed geometry; height uses actual AGL with a shared metric range', () => {
  const { raster, map } = fixture({ observed: [1, 0, 1, 1, 1, 1, 1, 1, 1] });
  raster.layer = f32(Array(9).fill(.5));
  const decoded = createBevPresentationRaster(raster, map);
  const structure = decoded.render('structure', 'turbo', decoded.heightRange);
  assert.deepEqual(pixel(structure, 0), [142, 157, 174, 255]);
  assert.deepEqual(pixel(structure, 1), [0, 0, 0, 0]);
  assert.deepEqual(decoded.heightRange, { min: 0, max: 2 });
  const range = combinedBevHeightRange([decoded.heightRange, { min: -1, max: 4 }, null]);
  assert.deepEqual(range, { min: -1, max: 4 });
  for (const palette of ['turbo', 'viridis', 'gray']) {
    const image = decoded.render('height', palette, range);
    assert.deepEqual(pixel(image, 0), [...bevPaletteColor(palette, .2), 255]);
    assert.deepEqual(pixel(image, 2), [...bevPaletteColor(palette, .6), 255]);
  }
});

test('texture uses exact RGB samples and its own observation mask without filling holes from AGL', () => {
  const { raster, map } = fixture();
  const decoded = createBevPresentationRaster(raster, map);
  const image = decoded.render('texture', 'turbo', decoded.heightRange);
  assert.deepEqual(pixel(image, 0), [0, 9, 18, 255]);
  assert.deepEqual(pixel(image, 1), [0, 0, 0, 0]);
  assert.deepEqual(pixel(image, 4), [108, 117, 126, 255]);
  assert.deepEqual(pixel(image, 8), [216, 225, 234, 255]);
  const missing = createBevPresentationRaster({ ...raster, surfaceRgb: null }, map);
  assert.equal(missing.render('texture', 'turbo', missing.heightRange), null);
  const noAgl = createBevPresentationRaster({ ...raster, heightLayer: null }, map);
  assert.equal(noAgl.render('height', 'turbo', decoded.heightRange), null);
  assert.ok(noAgl.render('structure', 'turbo', null));
});

test('room normals derive metre-scaled PCF slopes and transform orientation exactly once into scene axes', () => {
  const { raster, map } = fixture();
  let decoded = createBevPresentationRaster(raster, map);
  const n = Math.round((1 / Math.sqrt(2) * .5 + .5) * 255);
  const inverse = Math.round((-1 / Math.sqrt(2) * .5 + .5) * 255);
  assert.deepEqual(pixel(decoded.render('normals', 'turbo', decoded.heightRange), 4), [inverse, n, 128, 255]);
  // Uniform scale and 90-degree yaw. The physical -localX slope becomes +sceneZ.
  const rotated = { ...map, sceneFromLocalColMajor: [0, 0, -2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 10, 20, 30, 1] };
  decoded = createBevPresentationRaster(raster, rotated);
  assert.deepEqual(pixel(decoded.render('normals', 'turbo', decoded.heightRange), 4), [128, n, n, 255]);
  // Height increases toward row zero (+localZ), giving a -Z normal component.
  const towardForward = fixture({ heights: [2, 2, 2, 1, 1, 1, 0, 0, 0] });
  decoded = createBevPresentationRaster(towardForward.raster, towardForward.map);
  assert.deepEqual(pixel(decoded.render('normals', 'turbo', decoded.heightRange), 4), [128, n, inverse, 255]);
  const isolated = fixture({ observed: [0, 0, 0, 0, 1, 0, 0, 0, 0] });
  decoded = createBevPresentationRaster(isolated.raster, isolated.map);
  assert.equal(decoded.render('normals', 'turbo', decoded.heightRange), null, 'unsupported derivatives stay unavailable');
});

test('changing layers and palettes reuses decoded static arrays', () => {
  const { raster, map } = fixture(); const originalAtob = globalThis.atob;
  let decodedCalls = 0;
  globalThis.atob = value => { decodedCalls++; return originalAtob(value); };
  try {
    const decoded = createBevPresentationRaster(raster, map);
    assert.equal(decodedCalls, 2, 'AGL and observation mask decoded once');
    for (const layer of ['structure', 'texture', 'height', 'normals']) for (const palette of ['turbo', 'gray', 'viridis']) {
      assert.ok(decoded.render(layer, palette, decoded.heightRange));
    }
    assert.equal(decodedCalls, 4, 'only RGB and its own mask add decodes');
    const dark = decoded.render('structure', 'turbo', decoded.heightRange);
    const light = decoded.render('structure', 'turbo', decoded.heightRange, 'light');
    assert.notDeepEqual(pixel(dark, 0), pixel(light, 0));
    assert.ok(pixel(light, 0).slice(0, 3).every(v => v < 150), 'light map keeps structure readable');
    assert.deepEqual(decoded.render('height', 'viridis', decoded.heightRange, 'light'),
      decoded.render('height', 'viridis', decoded.heightRange, 'dark'), 'scientific colors do not change with theme');
    assert.equal(decodedCalls, 4, 'theme changes reuse decoded arrays');
  } finally { globalThis.atob = originalAtob; }
});

test('actual component preserves overlays, caches static images, labels units, and reports unavailable layers', async () => {
  const { raster, map } = fixture();
  const person = { entityId: 'synthetic-person', stableId: 7, label: 'Person 7', position: [1, 0, 1],
    positionState: 'measured', observedAtUs: 2_000_000,
    trail: [{ position: [.5, 0, .5], observedAtUs: 1_000_000 }, { position: [1, 0, 1], observedAtUs: 2_000_000 }] };
  const display = { rasters: [raster], hasSnapshot: true, world: { maps: [map], people: [person], unavailable: [],
    bindings: {}, diagnostic: { accepted: true }, cursor: { publishedAtUs: 2_000_000 } } };
  const state = { calls: [], cleanup: [], memos: [], slot: 0, images: 0 };
  const ctx = new Proxy({ measureText: text => ({ width: text.length * 6 }),
    fillText(text) { state.calls.push(['label', text, this.fillStyle]); },
    drawImage() { state.calls.push(['image', this.globalAlpha]); },
  }, { get: (target, key) => target[key] ?? ((...args) => state.calls.push([key, ...args])) });
  state.canvas = { getBoundingClientRect: () => ({ width: 500, height: 300 }), getContext: () => ctx };
  const previous = { document: globalThis.document, window: globalThis.window, ResizeObserver: globalThis.ResizeObserver };
  globalThis.__bevLayerTest = state;
  globalThis.document = { createElement() { state.images++; return { getContext: () => ({
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }), putImageData() {},
  }) }; } };
  globalThis.window = { devicePixelRatio: 1, getComputedStyle: () => ({ getPropertyValue: key => key === '--bev-label-text' ? '#123456' : '' }) };
  globalThis.ResizeObserver = class { observe() {} disconnect() {} };
  try {
    const result = await build({ entryPoints: [new URL('../components/UnifiedBevView.tsx', import.meta.url).pathname],
      bundle: true, write: false, format: 'esm', platform: 'node', jsx: 'transform',
      jsxFactory: 'React.createElement', jsxFragment: 'React.Fragment', plugins: [{ name: 'layer-component', setup(builder) {
        builder.onResolve({ filter: /^react(?:\/jsx-runtime)?$/ }, args => ({ path: args.path, namespace: 'layer-test' }));
        builder.onLoad({ filter: /.*/, namespace: 'layer-test' }, args => ({ contents: args.path === 'react/jsx-runtime'
          ? 'export const jsx=(tag,props,key)=>({tag,props,key}); export const jsxs=jsx; export const Fragment="fragment";'
          : `
          const React={createElement:(tag,props,...children)=>({tag,props,children}),Fragment:'fragment'}; export default React;
          export const useRef=()=>({current:globalThis.__bevLayerTest.canvas});
          export const useMemo=(fn,deps)=>{const s=globalThis.__bevLayerTest;const slot=s.slot++;const old=s.memos[slot];
            if(old&&deps.every((d,i)=>Object.is(d,old.deps[i])))return old.value;
            const value=fn();s.memos[slot]={deps,value};return value;};
          export const useEffect=fn=>globalThis.__bevLayerTest.cleanup.push(fn());
        ` }));
      } }] });
    const { UnifiedBevView } = await import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
    const render = (presentation = {}, current = display) => {
      state.slot = 0; state.calls.length = 0;
      return UnifiedBevView({ display: current, connection: 'open', trailEnabled: true, presentation });
    };
    const before = JSON.stringify(display);
    const first = render();
    assert.equal(first.props['data-bev-layer'], 'structure');
    assert.equal(state.images, 1);
    assert.ok(state.calls.find(c => c[0] === 'arc'));
    assert.equal(state.calls.find(c => c[0] === 'label')[2], '#123456', 'canvas labels resolve theme tokens');
    const view = render({ bevLayer: 'height', opacity: .25, palette: 'viridis' });
    assert.ok(textOf(view).includes('Height above ground (metres)'));
    assert.ok(textOf(view).includes('0.00 m – 2.00 m'));
    assert.deepEqual(state.calls.find(c => c[0] === 'image'), ['image', .25]);
    assert.ok(state.calls.find(c => c[0] === 'arc'), 'canonical people draw above the chosen layer');
    render({ bevLayer: 'height', opacity: .8, palette: 'viridis', labels: false, people: false, trails: false });
    assert.equal(state.images, 2, 'opacity and overlay flags reuse the height image');
    assert.equal(state.calls.filter(c => ['arc', 'label', 'lineTo'].includes(c[0])).length, 0);
    render({ bevLayer: 'texture' }); render({ bevLayer: 'normals' }); render();
    assert.equal(state.images, 4, 'returning to Structure reuses the original rendered raster');
    const unavailable = render({ bevLayer: 'texture' }, { ...display, rasters: [{ ...raster, surfaceRgb: null }] });
    assert.equal(state.calls.filter(c => c[0] === 'image').length, 0, 'no silent substitute layer');
    assert.ok(textOf(unavailable).includes('Texture'));
    assert.ok(textOf(unavailable).includes(' unavailable for this admitted map'));
    assert.equal(JSON.stringify(display), before, 'presentation never changes canonical state');
    state.cleanup.forEach(fn => fn?.());
  } finally { Object.assign(globalThis, previous); delete globalThis.__bevLayerTest; }
});
