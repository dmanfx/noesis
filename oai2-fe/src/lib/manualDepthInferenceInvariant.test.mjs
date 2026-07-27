import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const source = async (relativeUrl) => readFile(
  fileURLToPath(new URL(relativeUrl, import.meta.url)),
  'utf8',
);

const occurrenceCount = (text, pattern) => text.match(pattern)?.length ?? 0;

test('fresh depth remains routed only through the explicit Refresh handler', async () => {
  const [app, drawer] = await Promise.all([
    source('../App.tsx'),
    source('../components/DepthDrawer.tsx'),
  ]);

  assert.equal(
    occurrenceCount(app, /requestMapAnythingDepth\(camId,\s*'fresh'\)/g),
    1,
  );
  assert.equal(
    occurrenceCount(drawer, /onRequestDepthFresh\(selectedCamera\)/g),
    1,
  );
  assert.equal(
    occurrenceCount(drawer, /onRequestDepthCachedRef\.current\(selectedCamera\)/g),
    1,
  );
  assert.match(
    drawer,
    /\}, \[open, selectedCamera\]\);/,
  );

  const exportStart = drawer.indexOf('const handleSaveCurrentImages = useCallback');
  const renderStart = drawer.indexOf('\n  return (', exportStart);
  assert.ok(exportStart >= 0 && renderStart > exportStart);
  const exportImplementation = drawer.slice(exportStart, renderStart);
  assert.doesNotMatch(exportImplementation, /onRequestDepth(?:Fresh|Cached)/);
});

test('3D renderers contain no depth or floorplan request path', async () => {
  const sources = await Promise.all([
    source('../components/CalibratedPointCloud3DView.tsx'),
    source('../components/CachedHeightfield3DView.tsx'),
    source('../components/FloorPlane3DView.tsx'),
  ]);
  for (const renderer of sources) {
    assert.doesNotMatch(
      renderer,
      /requestMapAnythingDepth|onRequestDepth|get_ma_depth|get_floorplan/,
    );
  }
});

test('depth responses remain bound to the camera registered for the request ID', async () => {
  const [app, websocketClient] = await Promise.all([
    source('../App.tsx'),
    source('../hooks/useWebSocketClient.ts'),
  ]);

  assert.match(
    websocketClient,
    /expectedCameraId:\s*pendingRequest\.cameraId/,
  );
  assert.match(
    app,
    /responseCameraMismatch[\s\S]*depth_response_camera_mismatch/,
  );
  assert.match(
    app,
    /const storageKey = expectedStorageKey \?\? responseStorageKey/,
  );
});
