import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const source = async (relativeUrl) => readFile(
  fileURLToPath(new URL(relativeUrl, import.meta.url)),
  'utf8',
);

const occurrenceCount = (text, pattern) => text.match(pattern)?.length ?? 0;

test('canonical PCF populates exactly the established four 3D views', async () => {
  const drawer = await source('../components/DepthDrawer.tsx');

  assert.equal(occurrenceCount(drawer, /setPrimitivesView\('/g), 4);
  assert.doesNotMatch(
    drawer,
    /setPrimitivesView\('(fusion-|scene-prior|scene-composite|room-cloud)/,
  );
  assert.match(drawer, /const primitivesHeightLayer = hasScenePrior \? pcfHeightLayer : undefined/);
  assert.match(
    drawer,
    /height: primitivesObstacleHeightLayer,[\s\S]*density: primitivesObstacleObservedLayer/,
  );
  assert.match(
    drawer,
    /hasScenePrior \? \([\s\S]*<ScenePriorPointCloud3DView/,
  );
  assert.match(drawer, /const floorPlaneResult = scenePriorFloorResult/);
  assert.doesNotMatch(drawer, /<FusedPointCloud3DView|<CalibratedPointCloud3DView/);
});

test('3D rendering remains read-only with respect to depth and floorplan inference', async () => {
  const renderers = await Promise.all([
    source('../components/CalibratedPointCloud3DView.tsx'),
    source('../components/CachedHeightfield3DView.tsx'),
    source('../components/FloorPlane3DView.tsx'),
    source('../components/FusedPointCloud3DView.tsx'),
    source('../components/ScenePriorPointCloud3DView.tsx'),
  ]);
  for (const renderer of renderers) {
    assert.doesNotMatch(
      renderer,
      /requestMapAnythingDepth|onRequestDepth|get_ma_depth|get_floorplan/,
    );
  }
});
