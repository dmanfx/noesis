import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const source = async (relativeUrl) => readFile(
  fileURLToPath(new URL(relativeUrl, import.meta.url)),
  'utf8',
);

test('drawer autoload requests the canonical PCF payload without depth inference', async () => {
  const drawer = await source('../components/DepthDrawer.tsx');
  assert.match(drawer, /requestId: `drawer-pcf-/);
  assert.match(drawer, /cacheOnly: true,[\s\S]*scenePriorOnly: true/);
  assert.match(drawer, /\[calibrationEpoch, cameraFloorplan, open, selectedCamera, transportOpen\]/);
  assert.doesNotMatch(drawer, /onRequestDepthCached/);
});

test('every standard room raster aliases the scene-prior diagnostic contract', async () => {
  const drawer = await source('../components/DepthDrawer.tsx');
  const bindings = [
    ['densityLayer', 'scene_prior_diagnostic_density'],
    ['heightLayer', 'scene_prior_diagnostic_height'],
    ['heightAglLayer', 'scene_prior_diagnostic_height_agl'],
    ['distanceLayer', 'scene_prior_diagnostic_distance'],
    ['gradientLayer', 'scene_prior_diagnostic_gradient'],
    ['obstacleHeightLayer', 'scene_prior_diagnostic_obstacle_height'],
    ['walkableLayer', 'scene_prior_diagnostic_walkable'],
    ['structuralHeightLayer', 'scene_prior_diagnostic_structural_height'],
    ['surfaceRgbLayer', 'scene_prior_diagnostic_surface_rgb'],
  ];
  for (const [alias, field] of bindings) {
    assert.match(drawer, new RegExp(`const ${alias} = hasCanonicalPcf \\? cameraFloorplan\\?\\.${field}`));
  }
  assert.match(drawer, /const floorPlaneResult = scenePriorFloorResult/);
  assert.match(drawer, /const pcfConfidenceLayer = hasCanonicalPcf \? cameraFloorplan\?\.scene_prior_diagnostic_confidence/);
  assert.match(drawer, /<ScenePriorPointCloud3DView/);
  assert.match(drawer, /calibrationEpoch=\{calibrationEpoch\}/);
  assert.doesNotMatch(drawer, /<FusedPointCloud3DView|<CalibratedPointCloud3DView/);
});

test('inline BEV also selects the canonical PCF diagnostic layers', async () => {
  const bev = await source('../components/BevView.tsx');
  assert.match(bev, /floorplan\?\.scene_prior_only === true/);
  assert.match(bev, /floorplan\?\.scene_prior_diagnostic_walkable/);
  assert.match(bev, /floorplan\?\.scene_prior_diagnostic_obstacle_height/);
  assert.match(bev, /floorplan\?\.scene_prior_diagnostic_height_agl/);
  assert.match(bev, /floorplan\?\.scene_prior_diagnostic_density/);
});

test('manual refresh response completes without entering visible floorplan state', async () => {
  const app = await source('../App.tsx');
  const refreshBranchStart = app.indexOf('if (refreshResult.handled && refreshCamera) {');
  const admissionStart = app.indexOf('const existingFloorplan = floorplanDataRef.current[key];', refreshBranchStart);
  assert.ok(refreshBranchStart >= 0 && admissionStart > refreshBranchStart);
  const refreshBranch = app.slice(refreshBranchStart, admissionStart);
  assert.match(refreshBranch, /status: 'idle'/);
  assert.match(refreshBranch, /return;/);
  assert.doesNotMatch(refreshBranch, /setFloorplanData|setMaDepthData|setMaDiagnostics/);
});

test('PCF admission is sticky against later static-camera responses', async () => {
  const workflow = await source('./depthPanelWorkflow.ts');
  assert.match(workflow, /if \(incoming\.scene_prior_only === true\)[\s\S]*return false;/);
  assert.match(workflow, /payload\.scene_prior_only === true/);
  assert.match(workflow, /payload\.served_from_cache !== false/);
  assert.match(workflow, /completed: true/);
});
