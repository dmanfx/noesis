import React, { useEffect, useRef, useState } from 'react';
import '../styles/topdown-drawer.css';
import { CameraKey } from '../lib/camera';
import { FloorplanResponse } from './DepthDrawer';
import BevView, { BevMeta } from './BevView';

type Props = {
  open: boolean;
  onClose: () => void;
  images: Record<CameraKey, string | null>;
  meta: Record<CameraKey, BevMeta | undefined>;
  floorplans: Record<string, FloorplanResponse>;
  tracks: Record<CameraKey, Array<{ track_id: number; stable_id?: number | null }>>;
  trailEnabled?: boolean;
  onUpdateConfig: (cam: CameraKey, cfg: { mpp: number; xMin: number; xMax: number; zMin: number; zMax: number }) => void;
  onToggleOverlay: (cam: CameraKey, enabled: boolean) => void;
  onCalibrateAll?: () => void;
};

const cameras: CameraKey[] = ['living-room', 'kitchen', 'family-room'];
const DEFAULT_WIDTH = 420;
const MAX_WIDTH = 1080;

const TopDownDrawer: React.FC<Props> = ({ open, onClose, meta, floorplans, tracks, onCalibrateAll, trailEnabled = true }) => {
  const [drawerWidth, setDrawerWidth] = useState<number>(DEFAULT_WIDTH);
  const drawerRef = useRef<HTMLElement | null>(null);
  const isResizingRef = useRef(false);
  const activePointerIdRef = useRef<number | null>(null);
  const previousUserSelectRef = useRef('');

  useEffect(() => {
    if (open) {
      setDrawerWidth(DEFAULT_WIDTH);
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;

    const handlePointerMove = (event: PointerEvent) => {
      if (!isResizingRef.current || activePointerIdRef.current !== event.pointerId) return;
      const drawer = drawerRef.current;
      if (!drawer) return;
      const rect = drawer.getBoundingClientRect();
      const candidate = rect.right - event.clientX;
      const viewportAllowance = Math.max(DEFAULT_WIDTH, Math.min(MAX_WIDTH, window.innerWidth - 80));
      const nextWidth = Math.max(DEFAULT_WIDTH, Math.min(candidate, viewportAllowance));
      if (Number.isFinite(nextWidth)) {
        setDrawerWidth(nextWidth);
      }
    };

    const stopResizing = (event: PointerEvent) => {
      if (!isResizingRef.current || activePointerIdRef.current !== event.pointerId) return;
      isResizingRef.current = false;
      activePointerIdRef.current = null;
      document.body.style.userSelect = previousUserSelectRef.current;
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', stopResizing);
    window.addEventListener('pointercancel', stopResizing);
    window.addEventListener('pointerleave', stopResizing);
    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', stopResizing);
      window.removeEventListener('pointercancel', stopResizing);
      window.removeEventListener('pointerleave', stopResizing);
    };
  }, [open]);

  useEffect(() => {
    if (!open && isResizingRef.current) {
      isResizingRef.current = false;
      activePointerIdRef.current = null;
      document.body.style.userSelect = previousUserSelectRef.current;
    }
  }, [open]);

  const handleResizePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 && event.pointerType === 'mouse') return;
    event.preventDefault();
    event.stopPropagation();
    if (!open) return;
    isResizingRef.current = true;
    activePointerIdRef.current = event.pointerId;
    previousUserSelectRef.current = document.body.style.userSelect;
    document.body.style.userSelect = 'none';
  };

  return (
    <>
      <div className={open ? 'td-overlay open' : 'td-overlay'} onClick={onClose} />
      <aside
        className={open ? 'td-drawer open' : 'td-drawer'}
        aria-hidden={!open}
        ref={drawerRef}
        style={{ width: drawerWidth }}
      >
        <div
          className="td-drawer-resize-handle"
          onPointerDown={handleResizePointerDown}
          role="presentation"
        />
        <header className="td-header">
          <h2>Top-Down Views</h2>
          {onCalibrateAll && (
            <button className="btn ghost" onClick={onCalibrateAll} title="Auto-calibrate poses from latest depth">
              Calibrate
            </button>
          )}
          <button onClick={onClose} aria-label="Close">✕</button>
        </header>
        <div className="td-content">
          <div className="td-grid">
            {cameras.map((cam) => (
              <BevView
                key={cam}
                cam={cam}
                meta={meta[cam]}
                floorplan={floorplans[cam]}
                tracks={tracks[cam]}
                trailEnabled={trailEnabled}
                variant="drawer"
              />
            ))}
          </div>
        </div>
      </aside>
    </>
  );
};

export default TopDownDrawer;
