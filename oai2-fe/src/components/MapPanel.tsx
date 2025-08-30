import React, { useEffect, useRef } from 'react';
import { CameraKey, cameraLabel, colorForTrack } from '../lib/camera';
import { TrailStore, drawTrails } from '../lib/trails';

export const MapPanel: React.FC<{
  store: TrailStore;
  visible: boolean;
}> = ({ store, visible }) => {
  const canvLiving = useRef<HTMLCanvasElement>(null);
  const canvKitchen = useRef<HTMLCanvasElement>(null);
  const canvFamily = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!visible) {
      [canvLiving.current, canvKitchen.current, canvFamily.current].forEach(c => c?.getContext('2d')?.clearRect(0, 0, c.width, c.height));
      return;
    }
    if (canvLiving.current) drawTrails(canvLiving.current, 'living-room', store, colorForTrack);
    if (canvKitchen.current) drawTrails(canvKitchen.current, 'kitchen', store, colorForTrack);
    if (canvFamily.current) drawTrails(canvFamily.current, 'family-room', store, colorForTrack);
  });

  const renderRow = (cam: CameraKey, ref: React.RefObject<HTMLCanvasElement>) => (
    <div className="vstack panel card">
      <div className="card-title">Top‑Down • {cameraLabel(cam)}</div>
      <div className="map-wrap">
        <canvas className="map" ref={ref} width={420} height={160} />
      </div>
    </div>
  );

  if (!visible) return null;

  return (
    <div className="vstack" style={{ gap: 16 }}>
      {renderRow('living-room', canvLiving)}
      {renderRow('kitchen', canvKitchen)}
      {renderRow('family-room', canvFamily)}
    </div>
  );
};

