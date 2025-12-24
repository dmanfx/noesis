import React from 'react';
import { COLOR_NS_STRIDE, cameraOrder, colorForTrack, colorIdForPerson, detectCameraKey } from '../lib/camera';

const unknownCameraIndex = new Map<string, number>();
let nextUnknownIndex = cameraOrder.length;

const fallbackColorId = (cameraId: string, trackId: number | null | undefined): number => {
  const key = String(cameraId ?? '').toLowerCase().trim();
  let idx = unknownCameraIndex.get(key);
  if (idx === undefined) {
    idx = nextUnknownIndex;
    nextUnknownIndex += 1;
    unknownCameraIndex.set(key, idx);
  }
  const rawTrack = typeof trackId === 'number' && Number.isFinite(trackId) ? trackId : 0;
  const trackInt = Math.max(0, Math.floor(rawTrack));
  return (idx + 1) * COLOR_NS_STRIDE + trackInt;
};

export const LegendPanel: React.FC<{ tracks: Array<{ track_id: number; stable_id?: number | null; camera_id: string }>; title?: string; }>
  = ({ tracks, title }) => {
  const list = [...(tracks || [])].sort((a,b) => (Number((a.stable_id ?? a.track_id) || 0)) - (Number((b.stable_id ?? b.track_id) || 0)));
  if (!list.length) return null;
  return (
    <div className="panel card">
      <div className="card-title">{title ?? 'Stable IDs'}</div>
      <div className="legend">
        {list.map(t => {
          const camKey = detectCameraKey(t.camera_id);
          const colorId = camKey
            ? colorIdForPerson(camKey, t.stable_id, t.track_id)
            : fallbackColorId(t.camera_id, t.track_id);
          const key = t.stable_id !== null && t.stable_id !== undefined
            ? `p:${t.camera_id}:s:${t.stable_id}`
            : `p:${t.camera_id}:t:${t.track_id}`;
          return (
          <div key={key} className="legend-row">
            <div className="dot" style={{ background: colorForTrack(colorId) }} />
            <div className="mono">SID {(t.stable_id ?? t.track_id)}</div>
            <div className="spacer" />
            <div className="subtitle">{t.camera_id}</div>
          </div>
        );})}
      </div>
    </div>
  );
};
