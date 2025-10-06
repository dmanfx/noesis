import React from 'react';
import { colorForTrack } from '../lib/camera';

export const LegendPanel: React.FC<{ tracks: Array<{ track_id: number; stable_id?: number | null; camera_id: string }>; title?: string; }>
  = ({ tracks, title }) => {
  const list = [...(tracks || [])].sort((a,b) => (Number((a.stable_id ?? a.track_id) || 0)) - (Number((b.stable_id ?? b.track_id) || 0)));
  if (!list.length) return null;
  return (
    <div className="panel card">
      <div className="card-title">{title ?? 'Stable IDs'}</div>
      <div className="legend">
        {list.map(t => (
          <div key={(t.stable_id ?? t.track_id)} className="legend-row">
            <div className="dot" style={{ background: colorForTrack(Number((t.stable_id ?? t.track_id) || 0)) }} />
            <div className="mono">SID {(t.stable_id ?? t.track_id)}</div>
            <div className="spacer" />
            <div className="subtitle">{t.camera_id}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
