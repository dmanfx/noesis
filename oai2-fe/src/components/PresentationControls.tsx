import React, { useState } from 'react';
import { allCameraDisplayMode, type PresentationSettings, type SavedPresentationPreset, type CameraDisplayMode } from '../lib/dashboardPresentation.js';
import { ChoiceStrip } from './ChoiceStrip';

type Props = {
  settings: PresentationSettings; onChange: (patch: Partial<PresentationSettings>) => void;
  presetId: string; presets: SavedPresentationPreset[]; onPreset: (id: string) => void;
  onSave: (name: string) => void; onDelete: (id: string) => void;
  cameras: string[];
};
export function PresentationControls({ settings, onChange, presetId, presets, onPreset, onSave, onDelete, cameras }: Props) {
  const [saving, setSaving] = useState(false), [name, setName] = useState('');
  const selectedSavedStyle = presets.find(preset => preset.id === presetId);
  return <div className="presentation-toolbar" aria-label="Scene presentation">
    <ChoiceStrip label="Unified BEV" value={settings.bevLayer} onChange={value => onChange({ bevLayer: value as PresentationSettings['bevLayer'] })}
      options={[{ value: 'structure', label: 'Structure' }, { value: 'texture', label: 'Texture' }, { value: 'height', label: 'Height' }, { value: 'normals', label: 'Normals' }]} />
    <CameraDisplayControls settings={settings} cameras={cameras} onChange={onChange} label="All cameras" />
    <ChoiceStrip label="Appearance" value={settings.theme} onChange={value => onChange({ theme: value as PresentationSettings['theme'] })}
      options={[{ value: 'dark', label: 'Dark' }, { value: 'light', label: 'Light' }, { value: 'contrast', label: 'Contrast' }]} />
    <div className="presentation-style-group">
      <ChoiceStrip label="Style preset" value={presetId} onChange={onPreset}
        options={[{ value: 'monitor', label: 'Monitor' }, { value: 'geometry', label: 'Geometry' }, { value: 'surface', label: 'Surface' }]} />
      {(presetId === 'custom' || selectedSavedStyle) && <span className="presentation-current-style" title={selectedSavedStyle?.name}>
        {selectedSavedStyle ? `Saved: ${selectedSavedStyle.name}` : 'Custom'}
      </span>}
    </div>
    <details className="presentation-layer-menu"><summary>Layers &amp; styles</summary><div className="presentation-layer-options">
      <PresentationLayerOptions settings={settings} onChange={onChange} />
      <div className="presentation-style-manager">
        {presets.length > 0 && <ChoiceStrip label="Saved styles" value={presetId} options={presets.map(preset => ({ value: preset.id, label: preset.name }))} onChange={onPreset} />}
        <div className="presentation-style-actions">
          <button className="btn ghost" type="button" aria-expanded={saving} aria-controls="presentation-save-form" onClick={() => setSaving(v => !v)}>Save style</button>
          {presetId.startsWith('saved-') && <button className="btn ghost" type="button" onClick={() => onDelete(presetId)}>Delete style</button>}
        </div>
        {saving && <form id="presentation-save-form" className="presentation-save" onSubmit={e => { e.preventDefault(); if (name.trim()) { onSave(name.trim()); setName(''); setSaving(false); } }}>
          <label htmlFor="presentation-name">Style name</label><input id="presentation-name" value={name} maxLength={40} onChange={e => setName(e.target.value)} required />
          <button className="btn primary" type="submit">Save</button>
        </form>}
      </div>
    </div></details>
  </div>;
}
export function CameraDisplayControls({ settings, cameras, onChange, label = 'All camera displays' }: Pick<Props, 'settings' | 'cameras' | 'onChange'> & { label?: string }) {
  const mode = allCameraDisplayMode(settings, cameras);
  return <div className="presentation-camera-group">
    <ChoiceStrip label={label} value={mode}
      options={[{ value: 'video', label: 'Video' }, { value: 'depth', label: 'Depth' }, { value: 'normals', label: 'Normals' }]}
      onChange={value => onChange({ defaultCameraMode: value as CameraDisplayMode, cameraModes: {}, cameras: true, comparison: 0 })} />
    {mode === 'mixed' && <span className="presentation-current-style" role="status">Mixed</span>}
  </div>;
}
export function PresentationLayerOptions({ settings, onChange }: Pick<Props, 'settings' | 'onChange'>) {
  return <div className="presentation-options">
    <div className="presentation-overlay-toggles">{(['people', 'trails', 'cameras', 'labels'] as const).map(key => <label key={key}>
      <input type="checkbox" checked={settings[key]} onChange={e => onChange({ [key]: e.target.checked })} />{key[0].toUpperCase() + key.slice(1)}
    </label>)}</div>
    <ChoiceStrip label="Height / depth palette" value={settings.palette} onChange={value => onChange({ palette: value as PresentationSettings['palette'], bevLayer: 'height' })}
      options={[{ value: 'viridis', label: 'Viridis' }, { value: 'turbo', label: 'Turbo' }, { value: 'gray', label: 'Gray' }]} />
    <label>Map opacity <output>{Math.round(settings.opacity * 100)}%</output><input aria-label="Map opacity" type="range" min="0" max="100" value={Math.round(settings.opacity * 100)} onChange={e => onChange({ opacity: Number(e.target.value) / 100 })} /></label>
    <p className="inspector-note">Palette choices open Height and also color camera Depth. Normals keep their direction colors. People hides markers, labels and trails. These controls change the overview display.</p>
  </div>;
}
