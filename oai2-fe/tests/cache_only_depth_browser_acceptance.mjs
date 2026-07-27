#!/usr/bin/env node
import { createRequire } from 'node:module';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright-core');

function argsFrom(argv) {
  const out = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith('--') || value === undefined) {
      throw new Error(`invalid argument at ${index}: ${key ?? ''}`);
    }
    out[key.slice(2)] = value;
  }
  return out;
}

const args = argsFrom(process.argv.slice(2));
for (const key of ['ui-url', 'harness-url', 'output-dir', 'chrome']) {
  if (!args[key]) throw new Error(`missing --${key}`);
}

const outputDir = path.resolve(args['output-dir']);
await mkdir(outputDir, { recursive: true });

const browser = await chromium.launch({
  executablePath: args.chrome,
  headless: true,
  args: [
    '--enable-webgl',
    '--ignore-gpu-blocklist',
    '--disable-dev-shm-usage',
  ],
});

const consoleRows = [];
const pageErrors = [];
const websocketFrames = [];
const networkRows = [];
const runMetrics = { contexts: [] };

function attachEvidence(page, contextName) {
  page.on('console', (message) => {
    consoleRows.push({
      context: contextName,
      type: message.type(),
      text: message.text(),
    });
  });
  page.on('pageerror', (error) => {
    pageErrors.push({ context: contextName, error: String(error) });
  });
  page.on('request', (request) => {
    const url = request.url();
    if (!url.includes('/api/v1/depth/')) return;
    networkRows.push({
      context: contextName,
      method: request.method(),
      resource_type: request.resourceType(),
      url,
    });
  });
  page.on('websocket', (socket) => {
    socket.on('framesent', (event) => {
      const payload = typeof event.payload === 'string'
        ? event.payload
        : Buffer.from(event.payload).toString('utf8');
      let parsed = null;
      try {
        parsed = JSON.parse(payload);
      } catch {}
      websocketFrames.push({
        context: contextName,
        direction: 'sent',
        payload: parsed ?? payload.slice(0, 400),
      });
    });
  });
}

async function waitForCanvas(page, selector, startedAt) {
  await page.waitForFunction(
    (canvasSelector) => {
      const canvas = document.querySelector(canvasSelector);
      if (!(canvas instanceof HTMLCanvasElement)) return false;
      if (canvas.width < 10 || canvas.height < 10) return false;
      try {
        return canvas.toDataURL('image/png').length > 1_000;
      } catch {
        return false;
      }
    },
    selector,
    { timeout: 30_000 },
  );
  return Math.round((performance.now() - startedAt) * 10) / 10;
}

async function canvasWebglInfo(page, selector) {
  return page.locator(selector).evaluate((canvas) => {
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) return { available: false };
    const extension = gl.getExtension('WEBGL_debug_renderer_info');
    return {
      available: true,
      version: gl.getParameter(gl.VERSION),
      renderer: extension
        ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL)
        : gl.getParameter(gl.RENDERER),
      vendor: extension
        ? gl.getParameter(extension.UNMASKED_VENDOR_WEBGL)
        : gl.getParameter(gl.VENDOR),
      drawing_buffer_width: gl.drawingBufferWidth,
      drawing_buffer_height: gl.drawingBufferHeight,
    };
  });
}

async function orbitAndSample(page, selector) {
  const canvas = page.locator(selector);
  const box = await canvas.boundingBox();
  if (!box) throw new Error(`canvas has no box: ${selector}`);
  const sample = page.evaluate(() => new Promise((resolve) => {
    const timestamps = [];
    const started = performance.now();
    const step = (timestamp) => {
      timestamps.push(timestamp);
      if (performance.now() - started < 1_500) {
        requestAnimationFrame(step);
        return;
      }
      const intervals = timestamps.slice(1).map((value, index) => (
        value - timestamps[index]
      )).sort((left, right) => left - right);
      const quantile = (ratio) => {
        if (!intervals.length) return null;
        return intervals[Math.min(
          intervals.length - 1,
          Math.floor((intervals.length - 1) * ratio),
        )];
      };
      const elapsed = timestamps.length > 1
        ? timestamps.at(-1) - timestamps[0]
        : 0;
      resolve({
        frame_count: timestamps.length,
        fps: elapsed > 0 ? ((timestamps.length - 1) * 1000) / elapsed : 0,
        frame_interval_p50_ms: quantile(0.5),
        frame_interval_p95_ms: quantile(0.95),
      });
    };
    requestAnimationFrame(step);
  }));
  await page.mouse.move(box.x + box.width * 0.48, box.y + box.height * 0.52);
  await page.mouse.down();
  for (let index = 1; index <= 24; index += 1) {
    await page.mouse.move(
      box.x + box.width * (0.48 + index * 0.006),
      box.y + box.height * (0.52 - index * 0.003),
    );
  }
  await page.mouse.up();
  return sample;
}

async function visualMetrics(page) {
  return page.evaluate(() => {
    const drawer = document.querySelector('.depth-drawer');
    const content = document.querySelector('.depth-drawer .content');
    const floorplanCell = [...document.querySelectorAll('.heatmap-cell')]
      .find((element) => element.textContent?.includes('Clean Floorplan · Primary'));
    const floorplanCanvas = floorplanCell?.querySelector('canvas');
    const cameraCell = [...document.querySelectorAll('.heatmap-cell')]
      .find((element) => element.textContent?.includes('Camera Depth · Primary'));
    const cameraCanvas = cameraCell?.querySelector('canvas');
    const bounds = (element) => {
      if (!(element instanceof HTMLElement)) return null;
      const rect = element.getBoundingClientRect();
      return {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
      };
    };
    const aspect = (element) => {
      if (!(element instanceof HTMLElement)) return null;
      const rect = element.getBoundingClientRect();
      const css = getComputedStyle(element).aspectRatio.split('/');
      const expected = Number(css[0]) / Number(css[1] || 1);
      const actual = rect.width / Math.max(1, rect.height);
      return {
        actual,
        expected,
        error_fraction: Number.isFinite(expected) && expected > 0
          ? Math.abs(actual - expected) / expected
          : null,
      };
    };
    return {
      viewport: { width: window.innerWidth, height: window.innerHeight },
      drawer: bounds(drawer),
      content_client_width: content?.clientWidth ?? null,
      content_scroll_width: content?.scrollWidth ?? null,
      horizontal_overflow_px: content
        ? Math.max(0, content.scrollWidth - content.clientWidth)
        : null,
      floorplan_canvas: bounds(floorplanCanvas),
      floorplan_aspect: aspect(floorplanCanvas),
      floorplan_width_occupancy: floorplanCanvas && content
        ? floorplanCanvas.getBoundingClientRect().width / content.clientWidth
        : null,
      camera_canvas: bounds(cameraCanvas),
      camera_aspect: aspect(cameraCanvas),
      camera_width_occupancy: cameraCanvas && content
        ? cameraCanvas.getBoundingClientRect().width / content.clientWidth
        : null,
    };
  });
}

async function openCachedDrawer(page) {
  await page.goto(args['ui-url'], { waitUntil: 'domcontentloaded' });
  await page.getByText('Connected', { exact: true }).waitFor({ timeout: 15_000 });
  await page.getByRole('button', { name: 'Depth', exact: true }).click();
  await page.locator('.depth-drawer.open').waitFor();
  await page.locator('#ma-depth-select').waitFor();
}

async function selectCachedCamera(page, camera) {
  await page.locator('#ma-depth-select').selectOption(camera);
  await page.waitForFunction(
    (expectedCamera) => {
      const select = document.querySelector('#ma-depth-select');
      const status = document.querySelector('.floorplan-status');
      const error = document.querySelector('.floorplan-error');
      return select?.value === expectedCamera
        && !error
        && status?.textContent?.includes('Showing cached floorplan');
    },
    camera,
    { timeout: 30_000 },
  );
  await page.waitForFunction(() => {
    const title = [...document.querySelectorAll('.heatmap-cell__title')]
      .find((element) => element.textContent?.includes('Camera Depth · Primary'));
    const canvas = title?.parentElement?.querySelector('canvas');
    return canvas instanceof HTMLCanvasElement
      && canvas.width >= 400
      && canvas.toDataURL('image/png').length > 10_000;
  }, undefined, { timeout: 30_000 });
}

async function runDesktop() {
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
    deviceScaleFactor: 1,
  });
  const page = await context.newPage();
  attachEvidence(page, 'desktop');
  await openCachedDrawer(page);
  const metrics = { name: 'desktop', cameras: {}, three_d: {} };

  for (const camera of ['living-room', 'kitchen', 'family-room']) {
    await selectCachedCamera(page, camera);
    metrics.cameras[camera] = await visualMetrics(page);
    await page.locator('.depth-drawer').screenshot({
      path: path.join(outputDir, `desktop-${camera}-heatmap.png`),
    });
  }

  await selectCachedCamera(page, 'living-room');
  const heightfieldStarted = performance.now();
  await page.getByRole('button', { name: '3D', exact: true }).click();
  const preview = page.getByLabel('Show live camera preview');
  if (await preview.isChecked()) await preview.uncheck();
  metrics.three_d.heightfield_construction_ms = await waitForCanvas(
    page,
    '.cached-heightfield-view canvas',
    heightfieldStarted,
  );
  metrics.three_d.heightfield_webgl = await canvasWebglInfo(
    page,
    '.cached-heightfield-view canvas',
  );
  await page.locator('.depth-drawer').screenshot({
    path: path.join(outputDir, 'desktop-living-room-heightfield.png'),
  });
  metrics.three_d.heightfield_orbit = await orbitAndSample(
    page,
    '.cached-heightfield-view canvas',
  );
  await page.locator('.depth-drawer').screenshot({
    path: path.join(outputDir, 'desktop-living-room-heightfield-orbit.png'),
  });

  const pointCloudStarted = performance.now();
  await page.getByRole('button', { name: 'Point cloud', exact: true }).click();
  metrics.three_d.point_cloud_construction_ms = await waitForCanvas(
    page,
    '.calibrated-point-cloud-view canvas',
    pointCloudStarted,
  );
  metrics.three_d.point_cloud_webgl = await canvasWebglInfo(
    page,
    '.calibrated-point-cloud-view canvas',
  );
  await page.locator('.depth-drawer').screenshot({
    path: path.join(outputDir, 'desktop-living-room-point-cloud.png'),
  });
  metrics.three_d.point_cloud_orbit = await orbitAndSample(
    page,
    '.calibrated-point-cloud-view canvas',
  );
  metrics.three_d.point_cloud_color_mode = await page
    .locator('#point-cloud-color-mode')
    .inputValue();
  metrics.three_d.point_cloud_submeta = await page
    .locator('.primitives-submeta')
    .filter({ hasText: 'Source 1920×1080' })
    .textContent();
  await page.locator('.depth-drawer').screenshot({
    path: path.join(outputDir, 'desktop-living-room-point-cloud-orbit.png'),
  });
  runMetrics.contexts.push(metrics);
  await context.close();
}

async function runNarrow() {
  const context = await browser.newContext({
    viewport: { width: 620, height: 900 },
    deviceScaleFactor: 1,
  });
  const page = await context.newPage();
  attachEvidence(page, 'narrow');
  await openCachedDrawer(page);
  await selectCachedCamera(page, 'living-room');
  const metrics = { name: 'narrow', heatmap: await visualMetrics(page), three_d: {} };
  await page.screenshot({
    path: path.join(outputDir, 'narrow-living-room-heatmap.png'),
  });

  await page.getByRole('button', { name: '3D', exact: true }).click();
  const preview = page.getByLabel('Show live camera preview');
  if (await preview.isChecked()) await preview.uncheck();
  await waitForCanvas(page, '.cached-heightfield-view canvas', performance.now());
  metrics.three_d.heightfield = await visualMetrics(page);
  await page.screenshot({
    path: path.join(outputDir, 'narrow-living-room-heightfield.png'),
  });
  await page.getByRole('button', { name: 'Point cloud', exact: true }).click();
  await waitForCanvas(page, '.calibrated-point-cloud-view canvas', performance.now());
  metrics.three_d.point_cloud = await visualMetrics(page);
  metrics.three_d.point_cloud_color_mode = await page
    .locator('#point-cloud-color-mode')
    .inputValue();
  await page.screenshot({
    path: path.join(outputDir, 'narrow-living-room-point-cloud.png'),
  });
  runMetrics.contexts.push(metrics);
  await context.close();
}

let acceptanceError = null;
try {
  await runDesktop();
  await runNarrow();
} catch (error) {
  acceptanceError = error instanceof Error ? error.stack || error.message : String(error);
} finally {
  await browser.close();
}

const sentPayloads = websocketFrames
  .filter((row) => row.direction === 'sent' && row.payload && typeof row.payload === 'object')
  .map((row) => row.payload);
const browserProof = {
  contract: 'noesis.depth-panel.cache-only-browser-acceptance.v1',
  ui_url: args['ui-url'],
  target_browser: {
    executable: args.chrome,
    product: 'Google Chrome',
  },
  websocket: {
    frames: websocketFrames,
    sent_type_counts: Object.fromEntries(
      [...new Set(sentPayloads.map((payload) => String(payload.type || '')))]
        .sort()
        .map((type) => [
          type,
          sentPayloads.filter((payload) => String(payload.type || '') === type).length,
        ]),
    ),
    fresh_depth_request_count: sentPayloads.filter(
      (payload) => payload.type === 'get_ma_depth',
    ).length,
    non_cache_floorplan_request_count: sentPayloads.filter(
      (payload) => payload.type === 'get_floorplan' && payload.cache_only !== true,
    ).length,
  },
  component_network_requests: networkRows,
  console: consoleRows,
  page_errors: pageErrors,
  metrics: runMetrics,
  error: acceptanceError,
};
await writeFile(
  path.join(outputDir, 'browser-acceptance.json'),
  `${JSON.stringify(browserProof, null, 2)}\n`,
  'utf8',
);

const harnessStatus = await fetch(`${args['harness-url']}/__harness/status`).then(
  (response) => {
    if (!response.ok) throw new Error(`harness status HTTP ${response.status}`);
    return response.json();
  },
);
await writeFile(
  path.join(outputDir, 'request-count-proof.json'),
  `${JSON.stringify(
    {
      browser: {
        fresh_depth_request_count:
          browserProof.websocket.fresh_depth_request_count,
        non_cache_floorplan_request_count:
          browserProof.websocket.non_cache_floorplan_request_count,
      },
      server: {
        fresh_depth_request_count: harnessStatus.fresh_request_count,
        non_cache_floorplan_request_count:
          harnessStatus.non_cache_floorplan_request_count,
        cache_only_request_count: harnessStatus.cache_only_request_count,
        component_request_count: harnessStatus.component_request_count,
      },
      pass: (
        browserProof.websocket.fresh_depth_request_count === 0
        && browserProof.websocket.non_cache_floorplan_request_count === 0
        && harnessStatus.fresh_request_count === 0
        && harnessStatus.non_cache_floorplan_request_count === 0
      ),
    },
    null,
    2,
  )}\n`,
  'utf8',
);

if (acceptanceError) throw new Error(acceptanceError);
if (pageErrors.length) throw new Error(`page errors: ${JSON.stringify(pageErrors)}`);
if (
  browserProof.websocket.fresh_depth_request_count !== 0
  || browserProof.websocket.non_cache_floorplan_request_count !== 0
  || harnessStatus.fresh_request_count !== 0
  || harnessStatus.non_cache_floorplan_request_count !== 0
) {
  throw new Error('cache-only request invariant failed');
}
console.log(JSON.stringify({
  status: 'pass',
  output_dir: outputDir,
  server_cache_only_requests: harnessStatus.cache_only_request_count,
  server_component_requests: harnessStatus.component_request_count,
  fresh_depth_requests: harnessStatus.fresh_request_count,
  non_cache_floorplan_requests: harnessStatus.non_cache_floorplan_request_count,
}, null, 2));
