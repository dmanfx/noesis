export const PRESENTATION_STORAGE_KEY = 'noesis.dashboard.presentation.v1';
const modes = ['video', 'depth', 'normals'];
const pick = (value, choices, fallback) => choices.includes(value) ? value : fallback;
const bounded = (value, fallback, min, max) => typeof value === 'number' && Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : fallback;
export function defaultPresentationSettings() {
  return { theme: 'dark', bevLayer: 'structure', palette: 'viridis', opacity: 1,
    people: true, trails: true, cameras: true, labels: true,
    defaultCameraMode: 'video', cameraModes: {}, comparison: 0 };
}
export function sanitizePresentationSettings(value) {
  const d = defaultPresentationSettings();
  if (!value || typeof value !== 'object' || Array.isArray(value)) return d;
  const cameraModes = {};
  if (value.cameraModes && typeof value.cameraModes === 'object') {
    Object.entries(value.cameraModes).slice(0, 32).forEach(([camera, mode]) => {
      if (/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}$/.test(camera) && !['constructor', 'prototype', '__proto__'].includes(camera) && modes.includes(mode)) cameraModes[camera] = mode;
    });
  }
  return { theme: pick(value.theme, ['dark', 'light', 'contrast'], d.theme),
    bevLayer: pick(value.bevLayer, ['structure', 'texture', 'height', 'normals'], d.bevLayer),
    palette: pick(value.palette, ['turbo', 'viridis', 'gray'], d.palette), opacity: bounded(value.opacity, d.opacity, 0, 1),
    ...Object.fromEntries(['people', 'trails', 'cameras', 'labels'].map(key => [key, typeof value[key] === 'boolean' ? value[key] : d[key]])),
    defaultCameraMode: pick(value.defaultCameraMode, modes, d.defaultCameraMode), cameraModes,
    comparison: bounded(value.comparison, d.comparison, 0, 100) };
}
export function sanitizeSavedPresets(value) {
  if (!Array.isArray(value)) return [];
  const seen = new Set();
  return value.slice(0, 6).flatMap(p => {
    if (!p || typeof p.id !== 'string' || !/^saved-[a-z0-9-]{1,60}$/.test(p.id) || seen.has(p.id) || typeof p.name !== 'string' || !p.name.trim()) return [];
    seen.add(p.id);
    return [{ id: p.id, name: p.name.trim().slice(0, 40), settings: sanitizePresentationSettings(p.settings) }];
  });
}
export function applyPresentationPreset(current, preset) {
  const base = { ...sanitizePresentationSettings(current), people: true, cameras: true, labels: true, opacity: 1, cameraModes: {}, comparison: 0 };
  if (preset === 'geometry') return { ...base, bevLayer: 'height', defaultCameraMode: 'depth', trails: false };
  if (preset === 'surface') return { ...base, bevLayer: 'normals', defaultCameraMode: 'normals', trails: false };
  return { ...base, bevLayer: 'structure', defaultCameraMode: 'video', trails: true };
}
export function cameraDisplayMode(settings, camera) { return settings.cameraModes[camera] ?? settings.defaultCameraMode; }

/** Reflect the modes actually shown, including room overrides and new cameras. */
export function allCameraDisplayMode(settings, cameras) {
  const shown = new Set(cameras.map(camera => cameraDisplayMode(settings, camera)));
  return shown.size > 1 ? 'mixed' : shown.values().next().value ?? settings.defaultCameraMode;
}
