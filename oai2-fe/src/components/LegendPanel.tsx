import React from 'react';
import { colorForTrack } from '../lib/camera';

export const LegendPanel: React.FC<{ tracks: Array<{ track_id: number; camera_id: string }>; title?: string; }>
  = ({ tracks, title }) => {
  const list = [...(tracks || [])].sort((a,b) => a.track_id - b.track_id);
  if (!list.length) return null;
  return (
    <div className="panel card">
      <div className="card-title">{title ?? 'Track IDs'}</div>
      <div className="legend">
        {list.map(t => (
          <div key={t.track_id} className="legend-row">
            <div className="dot" style={{ background: colorForTrack(t.track_id) }} />
            <div className="mono">ID {t.track_id}</div>
            <div className="spacer" />
            <div className="subtitle">{t.camera_id}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

