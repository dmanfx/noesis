#!/usr/bin/env node
// Exercise the actual Three.js consumer, its existing Menon loader, and bounded orbit frames.
import { createRequire } from 'node:module';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import path from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, index, array) => {
  if (value.startsWith('--')) pairs.push([value.slice(2), array[index+1]]);
  return pairs;
}, []));
for (const name of ['dependency-root', 'browser-executable', 'output-dir']) {
  if (!args[name]) throw new Error(`--${name} is required`);
}
const require = createRequire(path.resolve(args['dependency-root'], 'package.json'));
const { chromium } = require('playwright-core');
const baseURL = args.url || 'http://127.0.0.1:8842';
const output = path.resolve(args['output-dir']);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ executablePath: args['browser-executable'], headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-webgl', '--ignore-gpu-blocklist', '--use-angle=gl'],
  timeout: 30_000 });
const report = { url: baseURL, browser: browser.version(), profiles: [], errors: [], warnings: [] };
try {
  const page = await browser.newPage({ viewport: { width: 1020, height: 1120 }, deviceScaleFactor: 1 });
  page.on('pageerror', (error) => report.errors.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') report.errors.push(message.text());
    if (message.type() === 'warning') report.warnings.push(message.text()); });
  const fetchedModels = [];
  page.on('response', (response) => { if (new URL(response.url()).pathname.endsWith('.glb')) fetchedModels.push(new URL(response.url()).pathname); });
  await page.goto(baseURL, { waitUntil: 'networkidle', timeout: 45_000 });
  await page.waitForFunction(() => window.roomMeshTrial?.metrics.loaded, { timeout: 45_000 });
  const trial = await (await page.request.get(`${baseURL}/trial.json`)).json();
  if (fetchedModels.length !== 1 || fetchedModels[0] !== trial.assets.find((row) => row.id === trial.default_asset).url) {
    throw new Error(`Initial page fetched unexpected model levels: ${JSON.stringify(fetchedModels)}`);
  }
  report.initial_model_requests = fetchedModels.slice();
  report.gpu = await page.evaluate(() => {
    const gl = document.querySelector('canvas').getContext('webgl2');
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    return { renderer: gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER),
      vendor: gl.getParameter(ext ? ext.UNMASKED_VENDOR_WEBGL : gl.VENDOR), version: gl.getParameter(gl.VERSION) };
  });
  for (const asset of trial.assets) {
    await page.selectOption('#asset', asset.id);
    await page.waitForFunction((expected) => window.roomMeshTrial?.metrics.loaded && window.roomMeshTrial.metrics.url === expected,
      asset.url, { timeout: 45_000 });
    await page.locator('canvas').screenshot({ path: path.join(output, `${asset.id}-fixed-rgb.png`) });
    await page.selectOption('#camera', 'phone_0064');
    await page.locator('canvas').screenshot({ path: path.join(output, `${asset.id}-walk-rgb.png`) });
    await page.selectOption('#mode', 'normal');
    await page.locator('canvas').screenshot({ path: path.join(output, `${asset.id}-walk-normals.png`) });
    await page.selectOption('#mode', 'rgb');
    await page.selectOption('#camera', 'fixed');
    await page.click('#benchmark');
    await page.waitForFunction(() => !!window.roomMeshTrial?.metrics.benchmark, { timeout: 12_000 });
    report.profiles.push(await page.evaluate(() => ({ ...window.roomMeshTrial.metrics,
      rendererMemory: { ...window.roomMeshTrial.rendererInfo.memory } })));
  }
  const balanced = trial.assets.find((row) => row.id === (args['compatible-asset'] || 'balanced'));
  if (!balanced) throw new Error('Requested compatible-loader asset is unavailable');
  await page.selectOption('#asset', balanced.id);
  await page.waitForFunction((url) => window.roomMeshTrial.metrics.loaded && window.roomMeshTrial.metrics.url === url,
    balanced.url, { timeout: 45_000 });
  // Prove the plain variant is accepted by Menon's existing embedded-image loader.
  report.existing_menon_loader = await page.evaluate(async (url) => {
    const { loadBoundedGlb } = await import('/menon-loader.js');
    const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
    const root = await loadBoundedGlb(bytes, { embeddedImages: true });
    let triangles = 0, materials = 0, textures = 0;
    root.traverse((node) => {
      if (!node.isMesh) return;
      triangles += (node.geometry.index?.count || node.geometry.attributes.position.count)/3;
      materials++; if (node.material.map) textures++;
      node.geometry.dispose(); node.material.map?.image?.close?.();
      node.material.map?.dispose(); node.material.dispose();
    });
    return { passed: true, bytes: bytes.byteLength, triangles, materials, textures };
  }, balanced.compatibility_url);
  await page.setViewportSize({ width: 412, height: 915 });
  await page.screenshot({ path: path.join(output, 'mobile-balanced.png'), fullPage: true });
  report.mobile = await page.evaluate(() => ({ width: innerWidth, documentWidth: document.documentElement.scrollWidth }));
  if (report.mobile.documentWidth > report.mobile.width) throw new Error('Mobile trial overflows horizontally');
  await page.setViewportSize({ width: 1020, height: 1120 });
  // The large source is fetched only for this explicit bounded comparison.
  await page.locator('details summary').click();
  await page.click('#source');
  await page.waitForFunction(() => window.roomMeshTrial?.metrics.loaded && window.roomMeshTrial.metrics.url === '/source.glb',
    { timeout: 60_000 });
  report.source = await page.evaluate(() => ({ ...window.roomMeshTrial.metrics }));
  await page.locator('canvas').screenshot({ path: path.join(output, 'original-fixed-rgb.png') });
  await page.click('#benchmark');
  await page.waitForFunction(() => !!window.roomMeshTrial?.metrics.benchmark, { timeout: 12_000 });
  report.source = await page.evaluate(() => ({ ...window.roomMeshTrial.metrics }));
  await page.selectOption('#camera', 'phone_0064');
  await page.locator('canvas').screenshot({ path: path.join(output, 'original-walk-rgb.png') });
  report.model_requests = fetchedModels;
  report.passed = report.errors.length === 0 && report.existing_menon_loader.passed;
} finally {
  await browser.close();
  await writeFile(path.join(output, 'browser_validation.json'), JSON.stringify(report, null, 2)+'\n');
}
console.log(JSON.stringify(report, null, 2));
if (!report.passed) process.exitCode = 1;
