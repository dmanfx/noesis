import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const source = async (relativeUrl) => readFile(
  fileURLToPath(new URL(relativeUrl, import.meta.url)),
  'utf8',
);

test('primary floorplan is the observed texture-first metric orthophoto', async () => {
  const drawer = await source('../components/DepthDrawer.tsx');

  assert.match(
    drawer,
    /renderTextureFloorplanToCanvas\(\s*floorplanCompositeCanvasRef\.current,\s*structuralHeightLayer,\s*roomFootprintLayer,\s*measuredPerimeterLayer,\s*surfaceRgbLayer,/,
  );

  const primaryStart = drawer.indexOf(
    'renderTextureFloorplanToCanvas(\n          floorplanCompositeCanvasRef.current',
  );
  const structuralStart = drawer.indexOf(
    'renderStructuralFloorplanToCanvas(',
    primaryStart,
  );
  assert.ok(primaryStart >= 0 && structuralStart > primaryStart);
  const primaryRender = drawer.slice(primaryStart, structuralStart);

  assert.match(primaryRender, /sourceRect: displaySourceRectForLayer\(structuralHeightLayer\)/);
  assert.match(primaryRender, /background: '#000'/);
  assert.match(primaryRender, /imageSmoothing: false/);
  assert.doesNotMatch(primaryRender, /flipHorizontal|flipVertical/);
  assert.doesNotMatch(primaryRender, /renderLayerToCanvas|heightLayer|heightAglLayer|metricGridM/);
});

test('primary floorplan export uses the identical texture renderer and orientation', async () => {
  const drawer = await source('../components/DepthDrawer.tsx');
  const exportStart = drawer.indexOf(
    'const result = renderTextureFloorplanToCanvas(\n          canvas,\n          structuralHeightLayer',
  );
  const exportEnd = drawer.indexOf(
    "addFloorplanLayer('density-gray'",
    exportStart,
  );
  assert.ok(exportStart >= 0 && exportEnd > exportStart);
  const primaryExport = drawer.slice(exportStart, exportEnd);
  assert.match(primaryExport, /roomFootprintLayer,\s*measuredPerimeterLayer,\s*surfaceRgbLayer,/);
  assert.match(primaryExport, /imageSmoothing: false/);
  assert.doesNotMatch(primaryExport, /flipHorizontal|flipVertical/);
  assert.match(primaryExport, /label: 'floorplan-observed-texture-primary'/);
  assert.doesNotMatch(primaryExport, /renderLayerToCanvas|primaryFloorplanLayer|HEIGHT_CONTRAST_GAMMA/);
  assert.match(
    drawer,
    /floorplan_image_flip: activeTab === 'heatmap'[\s\S]*u: Boolean\(cameraFloorplan\?\.image_flip\?\.u\),[\s\S]*v: Boolean\(cameraFloorplan\?\.image_flip\?\.v\),/,
  );
});

test('texture renderer keeps inferred space black and bounds every repair to 0.12 m', async () => {
  const renderer = await source('./renderUtils.ts');
  const start = renderer.indexOf('export function renderTextureFloorplanToCanvas');
  const end = renderer.indexOf(
    'export function renderStructuralFloorplanToCanvas',
    start,
  );
  assert.ok(start >= 0 && end > start);
  const textureRenderer = renderer.slice(start, end);

  assert.match(textureRenderer, /percentileSorted\(luminanceSamples, 1\)/);
  assert.match(textureRenderer, /percentileSorted\(luminanceSamples, 99\)/);
  assert.match(textureRenderer, /nearestDistanceSquared\[index\] <= 9/);
  assert.match(textureRenderer, /footprint\[index\]\s*&& nearestSource\[index\] >= 0/);
  assert.match(textureRenderer, /0\.90 \* textureR/);
  assert.match(textureRenderer, /0\.10 \* \(heightR \/ 255\)/);
  assert.match(textureRenderer, /heightM \/ 1\.65/);
  assert.match(textureRenderer, /writeCompositePixel\(index, 2, 4, 7\)/);
  assert.match(textureRenderer, /measuredPerimeterValues\[index\] <= 0\.18/);
  assert.match(textureRenderer, /const measuredMix = 0\.35 \+ \(0\.30 \* measuredStrength\)/);
  assert.match(textureRenderer, /image\[offset\] = Math\.max\(image\[offset\], 70\)/);
  assert.match(textureRenderer, /const clearCanvas = \(\) =>/);
  assert.doesNotMatch(textureRenderer, /maskLayer: roomFootprint|fillStyle.*footprint/);
});

test('structural composite remains diagnostic and cannot style the primary canvas', async () => {
  const drawer = await source('../components/DepthDrawer.tsx');

  assert.match(
    drawer,
    /renderStructuralFloorplanToCanvas\(\s*structuralFloorplanCanvasRef\.current,/,
  );
  assert.match(drawer, /Structural Composite \(Diagnostic\)/);
  assert.match(
    drawer,
    /ref=\{floorplanCompositeCanvasRef\}\s*className="heatmap-canvas"/,
  );
  assert.doesNotMatch(
    drawer,
    /ref=\{floorplanCompositeCanvasRef\}\s*className="heatmap-canvas heatmap-canvas--semantic"/,
  );
});

test('primary viewport follows explicitly observed cells rather than inferred footprint', async () => {
  const drawer = await source('../components/DepthDrawer.tsx');

  assert.match(
    drawer,
    /const floorplanViewportMaskLayer = unknownLayer \?\? observationMaskLayer;/,
  );
  assert.match(drawer, /maskInvert: floorplanViewportMaskInvert/);
  assert.match(drawer, /paddingM: PRIMARY_FLOORPLAN_VIEWPORT_PADDING_M/);

  const viewportStart = drawer.indexOf('const floorplanDisplayViewport = useMemo');
  const viewportEnd = drawer.indexOf(
    'const displaySourceRectForLayer',
    viewportStart,
  );
  assert.ok(viewportStart >= 0 && viewportEnd > viewportStart);
  assert.doesNotMatch(
    drawer.slice(viewportStart, viewportEnd),
    /roomFootprintLayer/,
  );
});
