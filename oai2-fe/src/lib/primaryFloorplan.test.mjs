import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const source = async (relativeUrl) => readFile(
  fileURLToPath(new URL(relativeUrl, import.meta.url)),
  'utf8',
);

test('primary floorplan is observed scalar height with a clean Inferno render', async () => {
  const drawer = await source('../components/DepthDrawer.tsx');

  assert.match(drawer, /const HEIGHT_CONTRAST_PCT_LO = 3;/);
  assert.match(drawer, /const HEIGHT_CONTRAST_PCT_HI = 97;/);
  assert.match(drawer, /const HEIGHT_CONTRAST_GAMMA = 0\.9;/);
  assert.match(
    drawer,
    /const primaryFloorplanLayer = heightLayer\?\.grid_b64 && heightLayer\.grid_shape\s*\? heightLayer\s*: heightAglLayer;/,
  );
  assert.match(
    drawer,
    /renderLayerToCanvas\(\s*floorplanCompositeCanvasRef\.current,\s*primaryFloorplanLayer,\s*infernoColor,/,
  );

  const primaryStart = drawer.indexOf(
    'renderLayerToCanvas(\n          floorplanCompositeCanvasRef.current',
  );
  const structuralStart = drawer.indexOf(
    'renderStructuralFloorplanToCanvas(',
    primaryStart,
  );
  assert.ok(primaryStart >= 0 && structuralStart > primaryStart);
  const primaryRender = drawer.slice(primaryStart, structuralStart);

  assert.match(primaryRender, /maskLayer: renderObservationMaskLayer/);
  assert.match(primaryRender, /maskInvert: renderObservationMaskInvert/);
  assert.match(primaryRender, /unknownColor: PRIMARY_FLOORPLAN_UNKNOWN/);
  assert.match(primaryRender, /repairIsolatedMaskHoles: true/);
  assert.match(primaryRender, /unknownAltColor: PRIMARY_FLOORPLAN_UNKNOWN/);
  assert.match(primaryRender, /sourceRect: displaySourceRectForLayer\(primaryFloorplanLayer\)/);
  assert.match(primaryRender, /background: '#000'/);
  assert.match(primaryRender, /imageSmoothing: true/);
  assert.match(primaryRender, /gamma: HEIGHT_CONTRAST_GAMMA/);
  assert.doesNotMatch(primaryRender, /surfaceRgbLayer|wallSupportLayer|roomFootprintLayer|metricGridM/);
});

test('primary floorplan export preserves the reference-matched height gamma', async () => {
  const drawer = await source('../components/DepthDrawer.tsx');
  const exportStart = drawer.indexOf(
    'const result = renderLayerToCanvas(\n          canvas,\n          primaryFloorplanLayer',
  );
  const exportEnd = drawer.indexOf(
    "addFloorplanLayer('density-gray'",
    exportStart,
  );
  assert.ok(exportStart >= 0 && exportEnd > exportStart);
  assert.match(
    drawer.slice(exportStart, exportEnd),
    /gamma: HEIGHT_CONTRAST_GAMMA/,
  );
  assert.match(
    drawer.slice(exportStart, exportEnd),
    /repairIsolatedMaskHoles: true/,
  );
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
