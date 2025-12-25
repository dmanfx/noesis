import React from 'react';
import { colorForTrack, colorIdForPerson, detectCameraKey } from '../lib/camera';

export const LegendPanel: React.FC<{ tracks: Array<{ stable_id: number; camera_id: string }>; title?: string; }>
  = ({ tracks, title }) => {
  const list = [...(tracks || [])].sort((a,b) => (Number(a.stable_id || 0)) - (Number(b.stable_id || 0)));
  if (!list.length) return null;
  return (
    <div className="panel card">
      <div className="card-title">{title ?? 'Stable IDs'}</div>
      <div className="legend">
        {list.map(t => {
          const camKey = detectCameraKey(t.camera_id);
          const colorId = camKey ? colorIdForPerson(camKey, t.stable_id) : Number(t.stable_id || 0);
          const key = `p:${t.camera_id}:s:${t.stable_id}`;
          return (
          <div key={key} className="legend-row">
            <div className="dot" style={{ background: colorForTrack(colorId) }} />
            <div className="mono">SID {t.stable_id}</div>
            <div className="spacer" />
            <div className="subtitle">{t.camera_id}</div>
          </div>
        );})}
      </div>
    </div>
  );
};
