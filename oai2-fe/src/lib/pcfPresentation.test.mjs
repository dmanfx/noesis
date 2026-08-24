import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import {
  CAMERA_GROUND_OBLIQUE_VIEW_DIRECTION,
  CAMERA_GROUND_TOP_DOWN_UP,
  CAMERA_GROUND_TOP_DOWN_VIEW_DIRECTION,
  cameraGroundScreenBasis,
  cameraLocalRasterToIsometric,
} from './cameraGroundPresentation.js';

const source = async (relativeUrl) => readFile(
  fileURLToPath(new URL(relativeUrl, import.meta.url)),
  'utf8',
);

test('all PCF 3D views preserve geometry and share an unmirrored screen basis', async () => {
  const { screenRight, screenUp, backward } = cameraGroundScreenBasis();
  assert.ok(screenRight[0] > 0.999);
  assert.ok(screenUp[2] > 0.8);
  const determinant = (
    (screenRight[0] * ((screenUp[1] * backward[2]) - (screenUp[2] * backward[1])))
    - (screenRight[1] * ((screenUp[0] * backward[2]) - (screenUp[2] * backward[0])))
    + (screenRight[2] * ((screenUp[0] * backward[1]) - (screenUp[1] * backward[0])))
  );
  assert.ok(Math.abs(determinant - 1) < 1e-12);

  const topDown = cameraGroundScreenBasis(
    CAMERA_GROUND_TOP_DOWN_VIEW_DIRECTION,
    CAMERA_GROUND_TOP_DOWN_UP,
  );
  assert.ok(Math.abs(topDown.screenRight[0] - 1) < 1e-12);
  assert.ok(Math.abs(topDown.screenRight[1]) < 1e-12);
  assert.ok(Math.abs(topDown.screenRight[2]) < 1e-12);
  assert.ok(Math.abs(topDown.screenUp[0]) < 1e-12);
  assert.ok(Math.abs(topDown.screenUp[1]) < 1e-12);
  assert.ok(Math.abs(topDown.screenUp[2] - 1) < 1e-12);
  assert.ok(CAMERA_GROUND_OBLIQUE_VIEW_DIRECTION[1] < 0);
  assert.ok(CAMERA_GROUND_OBLIQUE_VIEW_DIRECTION[2] > 0);

  for (const component of [
    'ScenePriorPointCloud3DView.tsx',
    'FusedPointCloud3DView.tsx',
    'CachedHeightfield3DView.tsx',
  ]) {
    const text = await source(`../components/${component}`);
    assert.match(text, /CAMERA_GROUND_OBLIQUE_VIEW_DIRECTION/);
    assert.doesNotMatch(text, /scale\.[xyz]\s*=\s*-/);
  }
  const visibleFloor = await source('../components/FloorPlane3DView.tsx');
  assert.match(visibleFloor, /CAMERA_GROUND_TOP_DOWN_UP/);
  assert.match(visibleFloor, /camera\.position\.set\(center\.x, center\.y - dist, center\.z\)/);
  assert.doesNotMatch(visibleFloor, /scale\.[xyz]\s*=\s*-/);
});

test('obstacle isometric view keeps forward above and camera-right to the right', async () => {
  const nearLeft = cameraLocalRasterToIsometric(2, 0, 2, 1);
  const forwardLeft = cameraLocalRasterToIsometric(0, 0, 2, 1);
  const nearRight = cameraLocalRasterToIsometric(2, 3, 2, 1);
  assert.ok(forwardLeft.y < nearLeft.y);
  assert.ok(nearRight.x > nearLeft.x);

  const renderer = await source('./extrudedFloorplan.ts');
  assert.match(renderer, /cameraLocalRasterToIsometric\(r, c, tileW, tileH\)/);
});

test('drawer autoload requests the canonical PCF payload without depth inference', async () => {
  const drawer = await source('../components/DepthDrawer.tsx');
  assert.match(drawer, /requestId: `drawer-pcf-/);
  assert.match(drawer, /cacheOnly: true,[\s\S]*scenePriorOnly: true/);
  assert.match(drawer, /\[calibrationEpoch, cameraFloorplan, open, selectedCamera, transportOpen\]/);
  assert.doesNotMatch(drawer, /onRequestDepthCached/);
});

test('renderer decodes lossless compact masks used by full-extent PCF payloads', async () => {
  const renderer = await source('./renderUtils.ts');
  assert.match(renderer, /base64\.startsWith\('bit:'\)/);
  assert.match(renderer, /Math\.ceil\(count \/ 8\)/);
  assert.match(renderer, /7 - \(index & 7\)/);
  assert.match(renderer, /base64\.startsWith\('u8:'\)/);
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
  assert.match(bev, /floorplan\?\.scene_prior_diagnostic_unknown/);
  assert.match(bev, /floorplan\?\.scene_prior_floor_supported/);
});

test('inline BEV renders the PCF walkable map with inferno semantics', async () => {
  const bev = await source('../components/BevView.tsx');
  assert.match(bev, /preferWalkableVisual && hasWalkable/);
  assert.match(bev, /selectFloorplanVisualSelection\(floorplanNow, isFloorplanCompatible, variant === 'inline'\)/);
  assert.match(bev, /const basePalette = infernoColor/);
  assert.match(bev, /maskLayer: observationMaskLayer/);
  assert.match(bev, /planarFloorSupportLayer/);
  assert.match(bev, /floorplan\?\.scene_prior_floor_supported/);
  assert.doesNotMatch(bev, /planarFloorGuardLayer/);
  assert.doesNotMatch(bev, /showFootprintGuide|footprintGuideLayer/);
  assert.match(bev, /showPlanarFloor: floorPlaneEnabled/);
  assert.match(bev, /planarFloorExpansionCells: BEV_PLANAR_FLOOR_EXPANSION_CELLS/);
  assert.match(bev, /renderCompositeWalkableObstacleToCanvas\([\s\S]*\.\.\.walkableRenderOptions/);
  assert.doesNotMatch(bev, /smoothBinaryPasses|BEV_WALKABLE_SMOOTHING_PASSES/);
  assert.doesNotMatch(bev, /repairIsolatedMaskHoles: true/);
  assert.doesNotMatch(bev, /inferredWalkableColor: BEV_INFERRED_WALKABLE_COLOR/);
  assert.match(bev, /ctx\.imageSmoothingEnabled = true/);
  assert.match(bev, /resolveBevDisplayBounds\(/);
  assert.match(bev, /advertisedBounds: advertised/);
  assert.doesNotMatch(bev, /BEV_DISPLAY_SAFETY_PADDING_M/);
  assert.match(bev, /const cameraOrigin = resolveForDraw\(0, 0\)/);
  assert.match(bev, /const normalizedPayloadBounds = useMemo/);
  assert.match(bev, /resolveBevMetricPoint\(/);
  assert.match(bev, /const smoothBaseImage = hasComposite \|\| baseKind === 'obstacle_height' \|\| isHeightVisual \|\| isWalkableVisual/);
  assert.doesNotMatch(bev, /baseKind === 'walkable' \? bwColor/);

  const renderer = await source('./renderUtils.ts');
  assert.match(renderer, /const floorColor: \[number, number, number\] = \[184, 180, 170\]/);
  assert.doesNotMatch(renderer, /const floorColor = infernoColor\(1\.0\)/);
});

test('each inline PCF BEV can toggle its regenerated floor plane', async () => {
  const bev = await source('../components/BevView.tsx');
  assert.match(bev, /const \[floorPlaneEnabled, setFloorPlaneEnabled\] = useState\(true\)/);
  assert.match(bev, /checked=\{floorPlaneEnabled\}/);
  assert.match(bev, /setFloorPlaneEnabled\(event\.target\.checked\)/);
  assert.match(bev, />\s*Floor plane\s*<\/label>/);
  assert.match(bev, /\{floorPlaneToggleLabel\}\s*\{toggleLabel\}/);

  const renderer = await source('./renderUtils.ts');
  assert.match(renderer, /optObj\?\.showPlanarFloor !== false/);
  assert.doesNotMatch(renderer, /showFootprintGuide|footprintGuideValues/);
});

test('heatmap PCF framing follows the complete reconstruction extent', async () => {
  const drawer = await source('../components/DepthDrawer.tsx');
  assert.match(drawer, /const floorplanViewportMaskLayer = reconstructionExtentLayer/);
  assert.match(drawer, /contentPaddingPx: PCF_DIAGNOSTIC_CONTENT_PADDING_PX/);
  assert.match(drawer, /drawCameraOriginMarker/);
  assert.match(drawer, /strokeText\('CAM'/);
  assert.match(drawer, /Room framing/);
  assert.match(drawer, /camera at bottom, \+Z forward/);
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
