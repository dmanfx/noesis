import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultPresentationSettings, sanitizePresentationSettings, sanitizeSavedPresets, applyPresentationPreset, cameraDisplayMode, allCameraDisplayMode } from './dashboardPresentation.js';
test('untrusted saved preferences are bounded and cannot supply data or prototype keys', () => {
  const s = sanitizePresentationSettings({ theme: 'bad', bevLayer: 'normals', opacity: -4, comparison: 900,
    people: 'false', cameraModes: JSON.parse('{"__proto__":"depth","kitchen":"normals","living-room":"bad"}') });
  assert.equal(s.theme, 'dark'); assert.equal(s.opacity, 0); assert.equal(s.comparison, 100); assert.equal(s.people, true);
  assert.deepEqual(s.cameraModes, { kitchen: 'normals' }); assert.equal(s.bevLayer, 'normals');
  assert.deepEqual(sanitizePresentationSettings(null), defaultPresentationSettings());
});
test('presets keep appearance independent and individual cameras can override the default', () => {
  const current = { ...defaultPresentationSettings(), theme: 'light', cameraModes: { kitchen: 'video' }, comparison: 100 };
  const geometry = applyPresentationPreset(current, 'geometry');
  assert.equal(geometry.theme, 'light'); assert.equal(geometry.bevLayer, 'height'); assert.equal(cameraDisplayMode(geometry, 'kitchen'), 'depth');
  assert.equal(cameraDisplayMode({ ...geometry, cameraModes: { kitchen: 'normals' } }, 'kitchen'), 'normals');
  assert.equal(current.cameraModes.kitchen, 'video');
  assert.equal(geometry.comparison, 0, 'a color wipe must not conceal the chosen material');
  assert.equal(applyPresentationPreset(current, 'surface').comparison, 0);
});
test('global camera selection reflects actual panels rather than the saved default', () => {
  const current = { ...defaultPresentationSettings(), cameraModes: { kitchen: 'depth', retired: 'normals' } };
  assert.equal(allCameraDisplayMode(current, ['kitchen', 'living-room']), 'mixed');
  assert.equal(allCameraDisplayMode(current, ['kitchen']), 'depth');
  assert.equal(allCameraDisplayMode(current, ['living-room']), 'video');
  assert.equal(allCameraDisplayMode(current, []), 'video');
  assert.equal(cameraDisplayMode(current, 'new-camera'), 'video');
  for (const preset of ['monitor', 'geometry', 'surface']) {
    const next = applyPresentationPreset(current, preset);
    assert.equal(allCameraDisplayMode(next, ['kitchen', 'living-room', 'new-camera']), next.defaultCameraMode);
    assert.deepEqual(next.cameraModes, {});
  }
});
test('saved styles retain only six bounded data-free settings records', () => {
  const presets = sanitizeSavedPresets(Array.from({ length: 20 }, (_, i) => ({ id: `saved-${i}`, name: 'x'.repeat(60), settings: { snapshot: 'not allowed' } })));
  assert.equal(presets.length, 6); assert.equal(presets[0].name.length, 40); assert(!('snapshot' in presets[0].settings));
  assert.deepEqual(sanitizeSavedPresets([{ id: 'bad', name: 'test' }]), []);
});
