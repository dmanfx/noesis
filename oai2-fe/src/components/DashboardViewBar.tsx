import React from 'react';
import { cameraLabel } from '../lib/camera';
import { ChoiceStrip } from './ChoiceStrip';

export type DashboardView = 'overview' | 'reconstruction' | 'height' | 'normals' | '3d';
export type ActiveDashboardView = DashboardView | 'diagnostics';

const views: Array<{ id: DashboardView; label: string; description: string }> = [
  { id: 'overview', label: 'Overview', description: 'Home map and live room cameras' },
  { id: 'reconstruction', label: 'Reconstruction', description: 'Observed PCF texture across the whole unified map' },
  { id: 'height', label: 'Height', description: 'Height above ground across the whole unified map' },
  { id: 'normals', label: 'Normals', description: 'PCF surface normals across the whole unified map' },
  { id: '3d', label: '3D', description: 'Existing room heightfield, point cloud, obstacles and visible floor' },
];

export function DashboardViewBar({ activeView, cameras, selectedRoom, onRoomChange, onViewChange, inspectorOpen, onToggleInspector }: {
  activeView: ActiveDashboardView;
  cameras: string[];
  selectedRoom: string;
  onRoomChange: (camera: string) => void;
  onViewChange: (view: DashboardView) => void;
  inspectorOpen: boolean;
  onToggleInspector: () => void;
}) {
  return <div className="dashboard-viewbar">
    <nav className="dashboard-viewbar__views" aria-label="Spatial views">
      {views.map(view => <button key={view.id} type="button" className={activeView === view.id ? 'is-active' : ''}
        aria-pressed={activeView === view.id} title={view.description}
        disabled={view.id === '3d' && !selectedRoom} onClick={() => onViewChange(view.id)}>{view.label}</button>)}
    </nav>
    <div className="dashboard-viewbar__context">
      <ChoiceStrip label="Room" value={selectedRoom} options={cameras.map(camera => ({ value: camera, label: cameraLabel(camera) }))} onChange={onRoomChange} />
      {!cameras.length && <span className="subtitle">No rooms available</span>}
      <button type="button" className={`btn ghost dashboard-inspector-toggle${inspectorOpen ? ' is-active' : ''}`}
        aria-expanded={inspectorOpen} aria-controls="context-inspector" onClick={onToggleInspector}>Inspector</button>
    </div>
  </div>;
}
