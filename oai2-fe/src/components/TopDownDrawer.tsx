import React, { useEffect, useRef } from 'react';
import '../styles/topdown-drawer.css';
import { TrailStore, drawTrails } from '../lib/trails';
import { CameraKey, cameraLabel, colorForTrack } from '../lib/camera';

type Props = {
  open: boolean;
  onClose: () => void;
  store: TrailStore;
};

const CanvasBlock: React.FC<{ cam: CameraKey; store: TrailStore }>
  = ({ cam, store }) => {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = c.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width * dpr));
    const h = Math.max(1, Math.floor(rect.height * dpr));
    c.width = w; c.height = h;
    drawTrails(c, cam, store, colorForTrack);
  });
  return (
    <div className="td-cell">
      <div className="td-title">Top‑Down • {cameraLabel(cam)}</div>
      <canvas ref={ref} className="td-canvas" />
    </div>
  );
};

const TopDownDrawer: React.FC<Props> = ({ open, onClose, store }) => {
  return (
    <>
      <div className={open ? 'td-overlay open' : 'td-overlay'} onClick={onClose} />
      <aside className={open ? 'td-drawer open' : 'td-drawer'} aria-hidden={!open}>
        <header className="td-header">
          <h2>Top‑Down Views</h2>
          <button onClick={onClose} aria-label="Close">✕</button>
        </header>
        <div className="td-content">
          <div className="td-grid">
            <CanvasBlock cam="living-room" store={store} />
            <CanvasBlock cam="kitchen" store={store} />
            <CanvasBlock cam="family-room" store={store} />
          </div>
        </div>
      </aside>
    </>
  );
};

export default TopDownDrawer;

