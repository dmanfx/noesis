import React, { useEffect, useRef, useState } from 'react';
import '../styles/topdown-drawer.css';
import { CameraKey, cameraLabel } from '../lib/camera';
import { FloorplanResponse } from './DepthDrawer';
import { renderLayerToCanvas, infernoColor } from '../lib/renderUtils';

type BevMeta = {
  cameraId: string;
  // Metric coordinates of detected objects
  footpoints?: Array<{ x: number; y: number; method: string; trackId?: number }>;
  // Optional status/error fields from server
  error?: string;
  details?: string;
};

type Props = {
  open: boolean;
  onClose: () => void;
  images: Record<CameraKey, string | null>;
  meta: Record<CameraKey, BevMeta | undefined>;
  floorplans: Record<string, FloorplanResponse>;
  tracks: Record<CameraKey, Array<{ track_id: number; stable_id?: number | null }>>;
  onUpdateConfig: (cam: CameraKey, cfg: { mpp: number; xMin: number; xMax: number; zMin: number; zMax: number }) => void;
  onToggleOverlay: (cam: CameraKey, enabled: boolean) => void;
  onCalibrateAll?: () => void;
};

const cameras: CameraKey[] = ['living-room', 'kitchen', 'family-room'];
const DEFAULT_WIDTH = 420;
const MAX_WIDTH = 1080;

const BevPanel: React.FC<{
  cam: CameraKey;
  meta?: BevMeta;
  floorplan?: FloorplanResponse;
  tracks?: Array<{ track_id: number; stable_id?: number | null }>;
}> = ({ cam, meta, floorplan, tracks }) => {
  const label = cameraLabel(cam);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [overlayEnabled, setOverlayEnabled] = useState(true);

  // Smoothing state: Map<trackId, { x: number, y: number, lastSeen: number, velocity: {x:number, y:number} }>
  const smoothState = useRef<Map<number, { x: number; y: number; lastSeen: number; stableId?: string }>>(new Map());
  const animationFrameRef = useRef<number>();

  // Update smooth state when meta changes
  useEffect(() => {
    const now = Date.now();
    const state = smoothState.current;

    if (meta?.footpoints) {
      meta.footpoints.forEach(pt => {
        if (pt.trackId === undefined) return;

        const targetX = pt.x;
        const targetY = pt.y; // Backend Z

        // Find stable ID
        let sid = `${pt.trackId}`;
        if (tracks) {
          const t = tracks.find(tr => tr.track_id === pt.trackId);
          if (t && t.stable_id !== undefined && t.stable_id !== null) {
            sid = `${t.stable_id}`;
          }
        }

        state.set(pt.trackId, {
          x: targetX,
          y: targetY,
          lastSeen: now,
          stableId: sid
        });
      });
    }

    // Prune old tracks
    for (const [id, data] of state.entries()) {
      if (now - data.lastSeen > 1000) { // 1 second hysteresis
        state.delete(id);
      }
    }
  }, [meta, tracks]);

  // Render loop
  useEffect(() => {
    const render = () => {
      const cvs = canvasRef.current;
      if (!cvs) return;
      const ctx = cvs.getContext('2d');
      if (!ctx) return;

      // 1. Draw Background
      const heightLayer = floorplan?.height;
      const hasFloorplan = !!(heightLayer && heightLayer.grid_b64 && heightLayer.grid_shape);

      if (hasFloorplan) {
        // Note: renderLayerToCanvas clears the canvas internally and may resize it
        renderLayerToCanvas(cvs, heightLayer, infernoColor);
      } else {
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, cvs.width, cvs.height);
      }

      // Re-read dimensions after potential resize
      const width = cvs.width;
      const height = cvs.height;

      // 2. Coordinate System
      let xMin = -4;
      let xMax = 4;
      let zMin = 0;
      let zMax = 12;

      if (floorplan?.bounds && floorplan.scale_m_per_px) {
        const b = floorplan.bounds;
        if (typeof b.min_x === 'number' && typeof b.max_x === 'number' &&
          typeof b.min_z === 'number' && typeof b.max_z === 'number') {
          xMin = b.min_x;
          xMax = b.max_x;
          zMin = b.min_z;
          zMax = b.max_z;
        }
      }

      const drawX = (mx: number) => ((mx - xMin) / (xMax - xMin)) * width;
      // Flip X so left side of the image maps to left on canvas (previously mirrored)
      const drawXFlipped = (mx: number) => width - ((mx - xMin) / (xMax - xMin)) * width;
      const drawY = (mz: number) => height - ((mz - zMin) / (zMax - zMin)) * height;

      // 3. Draw Grid
      if (overlayEnabled) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        const startX = Math.ceil(xMin);
        for (let x = startX; x <= xMax; x++) {
          const u = drawXFlipped(x);
          ctx.moveTo(u, 0);
          ctx.lineTo(u, height);
        }
        const startZ = Math.ceil(zMin);
        for (let z = startZ; z <= zMax; z++) {
          const v = drawY(z);
          ctx.moveTo(0, v);
          ctx.lineTo(width, v);
        }
        ctx.stroke();

        // Origin
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        if (zMin <= 0 && zMax >= 0) {
          const v0 = drawY(0);
          ctx.moveTo(0, v0);
          ctx.lineTo(width, v0);
        }
        if (xMin <= 0 && xMax >= 0) {
          const u0 = drawXFlipped(0);
          ctx.moveTo(u0, 0);
          ctx.lineTo(u0, height);
        }
        ctx.stroke();
      }

      // 4. Draw Smoothed Dots
      const now = Date.now();
      smoothState.current.forEach((pt, id) => {
        const age = now - pt.lastSeen;
        if (age > 500) return; // Don't draw if too old (but keep in state for 1s)

        const px = drawXFlipped(pt.x);
        const py = drawY(pt.y);

        // Fade out if stale
        const alpha = Math.max(0, 1 - age / 500);

        ctx.globalAlpha = alpha;
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, 2 * Math.PI);
        ctx.fillStyle = '#0ff';
        ctx.fill();
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        if (pt.stableId) {
          ctx.fillStyle = '#fff';
          ctx.font = 'bold 12px sans-serif';
          ctx.shadowColor = 'black';
          ctx.shadowBlur = 4;
          ctx.fillText(pt.stableId, px + 8, py - 8);
          ctx.shadowBlur = 0;
        }
        ctx.globalAlpha = 1.0;
      });

      animationFrameRef.current = requestAnimationFrame(render);
    };

    render();
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [floorplan, overlayEnabled]);

  return (
    <div className="td-cell">
      <div className="td-title">
        {label}
        <span className="td-subtitle">{floorplan ? ' • Height Map' : ' • No Map Data'}</span>
      </div>
      <div className="td-image-wrap" style={{ background: '#000', display: 'flex', justifyContent: 'center', alignItems: 'center', position: 'relative' }}>
        {meta?.error ? (
          <div className="td-placeholder">BEV Error: {meta.error}{meta.details ? ` — ${meta.details}` : ''}</div>
        ) : (
          <canvas
            ref={canvasRef}
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          />
        )}
        <div style={{ position: 'absolute', bottom: 8, right: 8 }}>
          <label className="td-overlay-toggle" style={{ background: 'rgba(0,0,0,0.5)', padding: '4px 8px', borderRadius: 4, color: 'white', fontSize: '12px' }}>
            <input
              type="checkbox"
              checked={overlayEnabled}
              onChange={(e) => setOverlayEnabled(e.target.checked)}
              style={{ marginRight: 6 }}
            />
            Grid
          </label>
        </div>
      </div>
    </div>
  );
};

const TopDownDrawer: React.FC<Props> = ({ open, onClose, meta, floorplans, tracks, onCalibrateAll }) => {
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
              <BevPanel
                key={cam}
                cam={cam}
                meta={meta[cam]}
                floorplan={floorplans[cam]}
                tracks={tracks[cam]}
              />
            ))}
          </div>
        </div>
      </aside>
    </>
  );
};

export default TopDownDrawer;
