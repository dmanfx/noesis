import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';

const built = await build({ entryPoints: [new URL('./pcfCameraProjection.ts', import.meta.url).pathname], bundle: true, write: false, format: 'esm', platform: 'node' });
const { buildPcfCameraProjection, validatePcfCameraMetadata, rasterizePcfCamera } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const input = () => ({ intrinsics: [160, 160, 48, 36], extrinsics: [...identity], binding: {
  contract: 'noesis.calibration.frame_binding', world_frame: { frame_id: 'backend_world_m', revision: 'pcf-fixture' },
  target_from_calibration_col_major: [...identity], transform_sha256: 'a'.repeat(64), scene_prior_id: 'pcf-fixture',
} });
const projection = (value = input()) => buildPcfCameraProjection('kitchen', 'pcf-fixture', 96, 72, value);
const rgb = count => new Uint8Array(count * 3).fill(128);
const metadata = () => ({ contract: 'noesis.scene_prior.camera_view', contract_version: 1, camera_id: 'kitchen', prior_id: 'pcf-fixture',
  source_coordinate_frame: 'backend_world_m', source_coordinate_frame_revision: 'pcf-fixture', frame_transform_sha256: 'a'.repeat(64),
  artifact_urls: { points_glb: '/api/v1/scene-priors/cameras/kitchen/artifacts/points_glb' }, points: { sha256: 'b'.repeat(64), size_bytes: 100 } });

const cloud = ({ zAt = () => 2, keep = () => true, color = [128, 128, 128], translateX = 0 } = {}) => {
  const points = [], colors = [];
  for (let v = 0; v < 72; v += 2) for (let u = 0; u < 96; u += 2) {
    if (!keep(u, v)) continue;
    const z = zAt(u, v); points.push((u - 48) / 160 * z + translateX, (v - 36) / 160 * z, z);
    colors.push(...(typeof color === 'function' ? color(u, v, z) : color));
  }
  return { points: new Float32Array(points), colors: new Uint8Array(colors) };
};
const render = (c = cloud(), p = projection()) => rasterizePcfCamera(c.points, c.colors, p);
const at = (u, v) => v * 96 + u;

test('surface retains the calibrated principal point and optical image Y down', () => {
  const r = render(cloud({ keep: (u, v) => u >= 52 && u <= 68 && v >= 40 && v <= 56 }));
  // Optical (X,Y,Z)=(.15,.15,2) must land at (60,48), below the principal point.
  assert.equal(r.depth[at(60, 48)], 2);
  assert(Number.isNaN(r.depth[at(60, 24)]));
  assert.deepEqual(r.range, { min: 2, max: 2 });
});

test('composes the exact target-to-calibration inverse once before optical projection', () => {
  const calibration = input(); calibration.binding.target_from_calibration_col_major[12] = 10;
  const p = projection(calibration);
  assert.equal(p.cameraFromWorld[12], -10);
  assert.equal(render(cloud({ translateX: 10 }), p).depth[at(48, 36)], 2);
});

test('nearest camera depth owns surface occlusion and perspective color', () => {
  const back = cloud({ zAt: () => 5, color: [255, 0, 0] });
  const front = cloud({ keep: (u, v) => u >= 24 && u <= 72 && v >= 18 && v <= 54, color: [0, 0, 255] });
  const r = render({ points: new Float32Array([...back.points, ...front.points, 0, 0, -1, 100, 100, 1]), colors: new Uint8Array([...back.colors, ...front.colors, 0, 255, 0, 0, 255, 0]) });
  assert.equal(r.depth[at(48, 36)], 2);
  assert.deepEqual(Array.from(r.rgb.slice(at(48, 36) * 4, at(48, 36) * 4 + 4)), [0, 0, 255, 255]);
  assert.equal(r.depth[at(12, 12)], 5);
});

test('camera-facing plane normals and all materials share continuous surface coverage', () => {
  const r = render();
  assert.deepEqual(Array.from(r.normals.slice(at(48, 36) * 4, at(48, 36) * 4 + 4)), [128, 128, 0, 255]);
  for (let v = 8; v < 64; v++) for (let u = 8; u < 88; u++) assert.equal(r.depth[at(u, v)], 2, 'interior surface hole');
  for (let i = 0; i < r.depth.length; i++) {
    assert.equal(r.rgb[i * 4 + 3] > 0, Number.isFinite(r.depth[i]));
    assert.equal(r.normals[i * 4 + 3] > 0, Number.isFinite(r.depth[i]));
  }
  assert.equal(r.normalCount, r.validCount);
  assert(r.triangleCount > 0);
});

test('tilted planes use perspective-correct inverse depth and optical-axis normals', () => {
  const r = render(cloud({ zAt: (u, v) => 1 / (.5 + .001 * (u - 48) + .0005 * (v - 36)), color: (_u, _v, z) => [Math.round(z * 90), 0, 0] }));
  const expected = [-.16, -.08, -.5], length = Math.hypot(...expected);
  for (const [u, v] of [[20, 20], [48, 36], [74, 50]]) {
    const i = at(u, v);
    assert(Math.abs(r.depth[i] - 1 / (.5 + .001 * (u - 48) + .0005 * (v - 36))) < 1e-5);
    assert(Math.abs(r.rgb[i * 4] - r.depth[i] * 90) <= 1, 'perspective RGB interpolation');
    const normal = Array.from(r.normals.slice(i * 4, i * 4 + 3), n => n / 255 * 2 - 1), norm = Math.hypot(...normal);
    assert(normal.reduce((s, n, axis) => s + n / norm * expected[axis] / length, 0) > .999);
  }
});

test('local plane fitting suppresses sample noise without changing depth or filling holes', () => {
  const r = render(cloud({ zAt: (u, v) => 2 + .008 * Math.sin(u * 1.7) * Math.cos(v * .7) }));
  const directions = new Set();
  for (let v = 18; v < 54; v++) for (let u = 18; u < 78; u++) {
    const i = at(u, v); if (!Number.isFinite(r.depth[i])) continue;
    assert(Math.abs(r.depth[i] - 2) < .009);
    assert(r.normals[i * 4 + 2] <= 2, 'plane normal faces camera');
    directions.add(Array.from(r.normals.slice(i * 4, i * 4 + 3)).join(','));
  }
  assert(directions.size < 12, 'coherent plane remains speckled');
});

test('large unsupported holes remain empty and surfaces do not extrapolate beyond support', () => {
  const r = render(cloud({ keep: (u, v) => !(u > 30 && u < 66 && v > 20 && v < 52) }));
  assert(Number.isNaN(r.depth[at(48, 36)]));
  assert.equal(r.normals[at(48, 36) * 4 + 3], 0);
  assert.equal(r.rgb[at(48, 36) * 4 + 3], 0);
  assert.equal(r.depth[at(12, 12)], 2);
  assert(Number.isNaN(r.depth[0]));
});

test('depth discontinuities cannot become interpolated connecting surfaces', () => {
  const r = render(cloud({ zAt: u => u < 48 ? 2 : 5 }));
  for (const z of r.depth) if (Number.isFinite(z)) assert(Math.min(Math.abs(z - 2), Math.abs(z - 5)) < 1e-5);
  assert.equal(r.depth[at(24, 36)], 2);
  assert.equal(r.depth[at(72, 36)], 5);
});

test('an isolated foreground sample blocks background interpolation without becoming a surface', () => {
  const back = cloud({ zAt: () => 5 });
  const r = render({ points: new Float32Array([...back.points, 0, 0, 1]), colors: new Uint8Array([...back.colors, 255, 255, 255]) });
  assert(Number.isNaN(r.depth[at(48, 36)]));
  assert.throws(() => rasterizePcfCamera(new Float32Array([0, 0, 2]), rgb(1), projection()), /no_visible_surface/);
});

test('refuses missing calibrated image size, intrinsics, revision and nonrigid transforms', () => {
  assert.throws(() => buildPcfCameraProjection('kitchen', 'pcf-fixture', 0, 7, input()));
  const missingK = input(); missingK.intrinsics = null; assert.throws(() => projection(missingK));
  const wrongRevision = input(); wrongRevision.binding.scene_prior_id = 'other'; assert.throws(() => projection(wrongRevision));
  const scaled = input(); scaled.binding.target_from_calibration_col_major[0] = 2; assert.throws(() => projection(scaled));
});

test('artifact admission binds exact camera, PCF revision, world frame, transform digest and byte inventory', () => {
  assert.equal(validatePcfCameraMetadata(metadata(), projection()).sizeBytes, 100);
  for (const patch of [{ camera_id: 'living-room' }, { prior_id: 'other' }, { source_coordinate_frame_revision: 'other' }, { frame_transform_sha256: 'c'.repeat(64) },
    { artifact_urls: { points_glb: 'https://example.com/points.glb' } }, { points: { sha256: 'b'.repeat(64), size_bytes: 33 * 1024 * 1024 } }]) {
    assert.throws(() => validatePcfCameraMetadata({ ...metadata(), ...patch }, projection()));
  }
});
