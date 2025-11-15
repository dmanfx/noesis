import React, { useEffect, useMemo, useState } from 'react';
import '../styles/topdown-drawer.css';
import { CameraKey, cameraLabel } from '../lib/camera';

type BevMeta = {
  cameraId: string;
  mpp?: number;
  xMin?: number;
  xMax?: number;
  zMin?: number;
  zMax?: number;
  overlay?: boolean;
};

type ControlState = {
  mpp: string;
  xMin: string;
  xMax: string;
  zMin: string;
  zMax: string;
  overlay: boolean;
};

type Props = {
  open: boolean;
  onClose: () => void;
  images: Record<CameraKey, string | null>;
  meta: Record<CameraKey, BevMeta | undefined>;
  onUpdateConfig: (cam: CameraKey, cfg: { mpp: number; xMin: number; xMax: number; zMin: number; zMax: number }) => void;
  onToggleOverlay: (cam: CameraKey, enabled: boolean) => void;
};

const cameras: CameraKey[] = ['living-room', 'kitchen', 'family-room'];

const deriveState = (meta?: BevMeta): ControlState => ({
  mpp: meta?.mpp !== undefined ? meta.mpp.toFixed(3) : '0.05',
  xMin: meta?.xMin !== undefined ? meta.xMin.toFixed(2) : '-4.00',
  xMax: meta?.xMax !== undefined ? meta.xMax.toFixed(2) : '4.00',
  zMin: meta?.zMin !== undefined ? meta.zMin.toFixed(2) : '0.00',
  zMax: meta?.zMax !== undefined ? meta.zMax.toFixed(2) : '12.00',
  overlay: meta?.overlay !== undefined ? meta.overlay : true,
});

const sanitize = (value: string, fallback: number) => {
  const parsed = parseFloat(value);
  if (Number.isFinite(parsed)) return parsed;
  return fallback;
};

const BevPanel: React.FC<{
  cam: CameraKey;
  image: string | null;
  control: ControlState;
  onChange: (next: ControlState) => void;
  onUpdate: () => void;
  onToggleOverlay: (enabled: boolean) => void;
}> = ({ cam, image, control, onChange, onUpdate, onToggleOverlay }) => {
  const label = cameraLabel(cam);
  return (
    <div className="td-cell">
      <div className="td-title">Top-Down • {label}</div>
      <div className="td-image-wrap">
        {image ? <img src={image} alt={`${label} BEV`} /> : <div className="td-placeholder">Waiting for BEV…</div>}
      </div>
      <div className="td-controls">
        <label>
          m/px
          <input
            type="number"
            step="0.01"
            min="0.01"
            value={control.mpp}
            onChange={(e) => onChange({ ...control, mpp: e.target.value })}
          />
        </label>
        <label>
          X min (m)
          <input
            type="number"
            step="0.1"
            value={control.xMin}
            onChange={(e) => onChange({ ...control, xMin: e.target.value })}
          />
        </label>
        <label>
          X max (m)
          <input
            type="number"
            step="0.1"
            value={control.xMax}
            onChange={(e) => onChange({ ...control, xMax: e.target.value })}
          />
        </label>
        <label>
          Z min (m)
          <input
            type="number"
            step="0.1"
            value={control.zMin}
            onChange={(e) => onChange({ ...control, zMin: e.target.value })}
          />
        </label>
        <label>
          Z max (m)
          <input
            type="number"
            step="0.1"
            value={control.zMax}
            onChange={(e) => onChange({ ...control, zMax: e.target.value })}
          />
        </label>
        <button type="button" onClick={onUpdate}>
          Apply
        </button>
        <label className="td-overlay-toggle">
          <input
            type="checkbox"
            checked={control.overlay}
            onChange={(e) => {
              onChange({ ...control, overlay: e.target.checked });
              onToggleOverlay(e.target.checked);
            }}
          />
          Show overlay
        </label>
      </div>
    </div>
  );
};

const TopDownDrawer: React.FC<Props> = ({ open, onClose, images, meta, onUpdateConfig, onToggleOverlay }) => {
  const [controlState, setControlState] = useState<Record<CameraKey, ControlState>>(() => {
    const initial: Partial<Record<CameraKey, ControlState>> = {};
    cameras.forEach((cam) => {
      initial[cam] = deriveState(meta[cam]);
    });
    return initial as Record<CameraKey, ControlState>;
  });

  useEffect(() => {
    setControlState((prev) => {
      const next = { ...prev };
      cameras.forEach((cam) => {
        next[cam] = deriveState(meta[cam]);
      });
      return next;
    });
  }, [meta]);

  const handleUpdate = (cam: CameraKey) => {
    const ctrl = controlState[cam];
    const payload = {
      mpp: sanitize(ctrl.mpp, 0.05),
      xMin: sanitize(ctrl.xMin, -4),
      xMax: sanitize(ctrl.xMax, 4),
      zMin: sanitize(ctrl.zMin, 0),
      zMax: sanitize(ctrl.zMax, 12),
    };
    onUpdateConfig(cam, payload);
  };

  return (
    <>
      <div className={open ? 'td-overlay open' : 'td-overlay'} onClick={onClose} />
      <aside className={open ? 'td-drawer open' : 'td-drawer'} aria-hidden={!open}>
        <header className="td-header">
          <h2>Top-Down Views</h2>
          <button onClick={onClose} aria-label="Close">✕</button>
        </header>
        <div className="td-content">
          <div className="td-grid">
            {cameras.map((cam) => (
              <BevPanel
                key={cam}
                cam={cam}
                image={images[cam] ?? null}
                control={controlState[cam]}
                onChange={(next) => setControlState((prev) => ({ ...prev, [cam]: next }))}
                onUpdate={() => handleUpdate(cam)}
                onToggleOverlay={(en) => onToggleOverlay(cam, en)}
              />
            ))}
          </div>
        </div>
      </aside>
    </>
  );
};

export default TopDownDrawer;
