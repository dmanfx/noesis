import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const source = async (relativeUrl) => readFile(
  fileURLToPath(new URL(relativeUrl, import.meta.url)),
  'utf8',
);

test('3D drawer does not mount the redundant source-specific diagnostic modes', async () => {
  const drawer = await source('../components/DepthDrawer.tsx');
  assert.doesNotMatch(drawer, /<SceneFusionDiagnosticLayers/);
  assert.doesNotMatch(drawer, /Fused cloud|Prior surface|Live \+ prior|Live depth/);
});

test('joint dropdown recreates every Heatmap diagnostic view from scene-fusion layers', async () => {
  const component = await source('../components/SceneFusionDiagnosticLayers.tsx');
  const expectedTitles = [
    'Structural Composite (Joint Diagnostic)',
    'Density (Grayscale)',
    'Surface Height (Inferno)',
    'Surface Height (Contrast)',
    'Height Above Floor',
    'Camera Range (Viridis)',
    'Obstacle Height (Clean)',
    'Measured Walkable Support (Binary)',
    'Gradient (Edges)',
  ];

  for (const title of expectedTitles) assert.match(component, new RegExp(title.replace(/[()]/g, '\\$&')));
  assert.match(component, /scene_fusion_diagnostic_density/);
  assert.match(component, /scene_fusion_diagnostic_obstacle_height/);
  assert.match(component, /scene_fusion_diagnostic_surface_rgb/);
  assert.match(component, /diagnostic_layers\?\.bounds/);
  assert.match(component, /opening this view runs no inference or registration/);
  assert.doesNotMatch(component, /onRequestFloorplan|onRequestDepth|flipHorizontal|flipVertical/);
});
