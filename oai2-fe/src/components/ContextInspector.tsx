import React from 'react';
import { cameraLabel } from '../lib/camera';
import type { DashboardView } from './DashboardViewBar';

export type InspectorTrack = {
  stable_id: number;
  display_name?: string | null;
  identity_kind?: string | null;
  zone?: string;
  dwell_time?: number;
  world_quality?: string;
  depth_status?: string | null;
};

export function ContextInspector({ context, selectedRoom, cameras, cameraStatuses, fps, vacancyText, occupancy, tracks,
  onSelectRoom, onHome, onClose, onViewChange, systemHealth, allOccupancy, allTracks, presentationControls }: {
  context: 'home' | 'room'; selectedRoom: string; cameras: string[];
  cameraStatuses: Record<string, string>; fps: Record<string, string>; vacancyText: Record<string, string>;
  occupancy: Record<string, number>; tracks: InspectorTrack[];
  onSelectRoom: (camera: string) => void; onHome: () => void; onClose: () => void;
  onViewChange: (view: DashboardView) => void;
  presentationControls?: React.ReactNode; systemHealth: React.ReactNode; allOccupancy: React.ReactNode; allTracks: React.ReactNode;
}) {
  const roomContext = context === 'room' && Boolean(selectedRoom);
  return <aside className="context-inspector" id="context-inspector" aria-label="Context inspector">
    <header className="context-inspector__header">
      <div><span className="eyebrow">Inspector</span><h2>{roomContext ? cameraLabel(selectedRoom) : 'Home overview'}</h2></div>
      <button type="button" className="btn ghost" aria-label="Close inspector" onClick={onClose}>×</button>
    </header>
    <div className="context-inspector__content">
      {roomContext ? <>
        <button type="button" className="inspector-home" onClick={onHome}>← Home overview</button>
        <dl className="inspector-facts">
          <div><dt>Camera</dt><dd>{cameraStatuses[selectedRoom] ?? 'unknown'}</dd></div>
          <div><dt>Source rate</dt><dd>{fps[selectedRoom]?.replace(/^FPS:\s*/, '') ?? '—'} FPS</dd></div>
          <div><dt>Vacancy</dt><dd>{vacancyText[selectedRoom] || '—'}</dd></div>
        </dl>
        <div className="inspector-section"><h3>Room views</h3><p className="inspector-note">Explore the saved PCF room reconstruction.</p>
          <div className="inspector-view-links">
            <button type="button" className="btn ghost" onClick={() => onViewChange('reconstruction')}>Reconstruction ↗</button>
            <button type="button" className="btn ghost" onClick={() => onViewChange('height')}>Room height ↗</button>
            <button type="button" className="btn ghost" onClick={() => onViewChange('normals')}>Room normals ↗</button>
            <button type="button" className="btn ghost" onClick={() => onViewChange('3d')}>3D room ↗</button>
          </div>
        </div>
        <div className="inspector-section"><h3>Zone occupancy</h3>
          {Object.keys(occupancy).length ? <dl className="inspector-facts">{Object.entries(occupancy).map(([zone, count]) => <div key={zone}><dt>{zone}</dt><dd>{count}</dd></div>)}</dl> : <p className="inspector-note">No occupancy data for this room.</p>}
        </div>
        <div className="inspector-section"><h3>Active track observations <span className="inspector-count">{tracks.length}</span></h3>
          {tracks.length ? <div className="inspector-tracks">{tracks.map((track, index) => <article key={`${track.stable_id}-${index}`}>
            <strong>{track.display_name || (Number.isFinite(track.stable_id) && track.stable_id > 0 ? `Person ${track.stable_id}` : 'Identity pending')}</strong>
            {track.identity_kind && <span>{track.identity_kind}</span>}
            <span>Zone {track.zone || '—'}{Number.isFinite(track.dwell_time) ? ` · ${track.dwell_time!.toFixed(1)} s dwell` : ''}</span>
            {track.world_quality && <span>World {track.world_quality}</span>}
            {track.depth_status && <span>Depth {track.depth_status}</span>}
          </article>)}</div> : <p className="inspector-note">No active track observations in this room.</p>}
        </div>
      </> : <>
        <p className="inspector-note">Select a room or a camera title to inspect its current observations and room views.</p>
        <div className="inspector-rooms">{cameras.map(camera => <button key={camera} type="button" onClick={() => onSelectRoom(camera)}>
          <strong>{cameraLabel(camera)}</strong><span>{cameraStatuses[camera] ?? 'unknown'} <span aria-hidden="true">↗</span></span>
        </button>)}</div>
      </>}
      {presentationControls && <details className="inspector-disclosure"><summary>Display layers &amp; camera modes</summary><div>{presentationControls}</div></details>}
      <details className="inspector-disclosure"><summary>System health &amp; controls</summary><div>{systemHealth}</div></details>
      <details className="inspector-disclosure"><summary>All zone occupancy</summary><div>{allOccupancy}</div></details>
      <details className="inspector-disclosure"><summary>All active track details</summary><div className="inspector-all-tracks">{allTracks}</div></details>
    </div>
  </aside>;
}
