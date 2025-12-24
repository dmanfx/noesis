import React, { useEffect, useRef, useState } from 'react';
import { CameraKey, cameraLabel } from '../lib/camera';
import { FloorplanResponse } from './DepthDrawer';
import { renderLayerToCanvas, infernoColor } from '../lib/renderUtils';

export type BevMeta = {
  cameraId?: string;
  camId?: string;
  footpoints?: Array<{ x: number; y: number; method: string; trackId?: number; stableId?: number | null }>;
  xMin?: number;
  xMax?: number;
  zMin?: number;
  zMax?: number;
  error?: string;
  details?: string;
};

type BevViewProps = {
  cam: CameraKey;
  meta?: BevMeta;
  floorplan?: FloorplanResponse;
  tracks?: Array<{ track_id: number; stable_id?: number | null }>;
  trailEnabled?: boolean;
  variant?: 'drawer' | 'inline';
};

const DEFAULT_X_MIN = -4;
const DEFAULT_X_MAX = 4;
const DEFAULT_Z_MIN = 0;
const DEFAULT_Z_MAX = 12;

type TrailPoint = { x: number; y: number; t: number };
type TrailTrack = { points: TrailPoint[]; lastSeen: number; label: string; colorId: number };

const TRAIL_WINDOW_MS = 20000;
const TRAIL_MIN_DT_MS = 80;
const TRAIL_MIN_STEP_M = 0.05;
const TRAIL_GAP_MS = 650;
const TRAIL_MAX_TRACKS = 8;
const TRAIL_LINE_WIDTH = 2.5;
const TRAIL_MIN_ALPHA = 0.12;
const TRAIL_STALE_BLINK_START_MS = 700;
const TRAIL_STALE_BLINK_PERIOD_MS = 1400;

export const BevView: React.FC<BevViewProps> = ({
  cam,
  meta,
  floorplan,
  tracks,
  trailEnabled = true,
  variant = 'drawer'
}) => {
  const label = cameraLabel(cam);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [overlayEnabled, setOverlayEnabled] = useState(false);

  const smoothState = useRef<Map<number, { x: number; y: number; lastSeen: number; stableId?: string }>>(new Map());
  const trailsRef = useRef<Map<string, TrailTrack>>(new Map());
  const animationFrameRef = useRef<number>();
  const bgCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const bgKeyRef = useRef<string>('');
  const bgSizeRef = useRef<{ w: number; h: number }>({ w: 0, h: 0 });
  const metaRef = useRef<BevMeta | undefined>(meta);

  useEffect(() => {
    metaRef.current = meta;
  }, [meta]);

  useEffect(() => {
    const now = Date.now();
    const state = smoothState.current;
    const trails = trailsRef.current;

    if (meta?.footpoints) {
      meta.footpoints.forEach(pt => {
        if (pt.trackId === undefined) return;

        const targetX = pt.x;
        const targetY = pt.y;

        const stableFromMeta = typeof pt.stableId === 'number' && Number.isFinite(pt.stableId) ? pt.stableId : null;
        const stableFromTracks = tracks?.find(tr => tr.track_id === pt.trackId)?.stable_id ?? null;
        const stableId = stableFromMeta ?? stableFromTracks;
        const sid = (stableId !== null && stableId !== undefined) ? `${stableId}` : `${pt.trackId}`;

        state.set(pt.trackId, {
          x: targetX,
          y: targetY,
          lastSeen: now,
          stableId: sid
        });

        if (!trailEnabled) return;

        const hasStable = stableId !== null && stableId !== undefined;
        const key = hasStable ? `s:${stableId}` : `t:${pt.trackId}`;
        const colorId = hasStable ? Number(stableId) : Number(pt.trackId);
        const labelText = hasStable ? `${stableId}` : `${pt.trackId}`;

        // If stable-id becomes available after earlier track-id-only samples, migrate history.
        if (hasStable) {
          const trackKey = `t:${pt.trackId}`;
          if (!trails.has(key) && trails.has(trackKey)) {
            const existing = trails.get(trackKey);
            if (existing) {
              trails.delete(trackKey);
              trails.set(key, { ...existing, label: labelText, colorId });
            }
          }
        }

        const entry = trails.get(key) ?? { points: [], lastSeen: 0, label: labelText, colorId };
        const pts = entry.points;
        const last = pts.length ? pts[pts.length - 1] : null;
        const lastValid = last && Number.isFinite(last.x) && Number.isFinite(last.y) ? last : null;

        // Break polyline if we lost sight long enough to avoid teleport lines.
        if (entry.lastSeen > 0 && (now - entry.lastSeen) > TRAIL_GAP_MS) {
          pts.push({ x: Number.NaN, y: Number.NaN, t: now });
        }

        if (!lastValid) {
          pts.push({ x: targetX, y: targetY, t: now });
        } else {
          const dt = now - lastValid.t;
          const dist = Math.hypot(targetX - lastValid.x, targetY - lastValid.y);
          if (dt >= TRAIL_MIN_DT_MS || dist >= TRAIL_MIN_STEP_M) {
            pts.push({ x: targetX, y: targetY, t: now });
          } else {
            // Update head sample in-place so the trail stays responsive without oversampling.
            pts[pts.length - 1] = { x: targetX, y: targetY, t: now };
          }
        }

        entry.lastSeen = now;
        entry.label = labelText;
        entry.colorId = colorId;
        trails.set(key, entry);
      });
    }

    for (const [id, data] of state.entries()) {
      if (now - data.lastSeen > 1000) {
        state.delete(id);
      }
    }

    if (trailEnabled) {
      // Prune per-track points by time window and evict old/empty tracks.
      for (const [key, track] of trails.entries()) {
        const pts = track.points;
        if (!pts.length) {
          trails.delete(key);
          continue;
        }
        let cut = 0;
        while (cut < pts.length && (now - pts[cut].t) > TRAIL_WINDOW_MS) cut += 1;
        if (cut > 0) track.points = pts.slice(cut);
        while (track.points.length && (!Number.isFinite(track.points[0].x) || !Number.isFinite(track.points[0].y))) {
          track.points.shift();
        }
        if (track.points.length === 0) trails.delete(key);
      }

      if (trails.size > TRAIL_MAX_TRACKS) {
        const ordered = Array.from(trails.entries()).sort((a, b) => a[1].lastSeen - b[1].lastSeen);
        for (let i = 0; i < ordered.length - TRAIL_MAX_TRACKS; i += 1) {
          trails.delete(ordered[i][0]);
        }
      }
    } else {
      trails.clear();
    }
  }, [meta, tracks, trailEnabled]);

  useEffect(() => {
    const render = () => {
      const cvs = canvasRef.current;
      if (!cvs) return;
      const ctx = cvs.getContext('2d');
      if (!ctx) return;

      const metaNow = metaRef.current;
      const aspect = variant === 'inline' ? 2 : (4 / 3);

      const heightLayer = floorplan?.height;
      const hasFloorplan = !!(heightLayer && heightLayer.grid_b64 && heightLayer.grid_shape);

      // Cache the floorplan render so we don't re-decode base64 every animation frame.
      const dpr = window.devicePixelRatio || 1;
      const rect = cvs.getBoundingClientRect();
      const expectedW = Math.max(1, Math.round((rect.width || 1) * dpr));
      const expectedH = Math.max(1, Math.round((rect.height || 1) * dpr));
      const key = hasFloorplan
        ? `${floorplan?.snapshot_ts ?? floorplan?.ts ?? ''}:${heightLayer?.grid_shape?.join('x')}:${heightLayer?.value_min ?? ''}:${heightLayer?.value_max ?? ''}:${heightLayer?.grid_b64?.length ?? ''}:${aspect}`
        : `none:${aspect}`;

      const bg = bgCanvasRef.current ?? (bgCanvasRef.current = document.createElement('canvas'));
      const bgSize = bgSizeRef.current;
      const bgNeedsRedraw = bgKeyRef.current !== key || bgSize.w !== expectedW || bgSize.h !== expectedH;

      if (bgNeedsRedraw) {
        if (hasFloorplan) {
          renderLayerToCanvas(cvs, heightLayer, infernoColor, aspect);
        } else {
          cvs.width = expectedW;
          cvs.height = expectedH;
          ctx.fillStyle = '#111';
          ctx.fillRect(0, 0, cvs.width, cvs.height);
        }
        bg.width = cvs.width;
        bg.height = cvs.height;
        const bgCtx = bg.getContext('2d');
        bgCtx?.drawImage(cvs, 0, 0);
        bgKeyRef.current = key;
        bgSizeRef.current = { w: cvs.width, h: cvs.height };
      }

      // Start frame from cached background.
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, cvs.width, cvs.height);
      ctx.drawImage(bg, 0, 0);

      const width = cvs.width;
      const height = cvs.height;

      let xMin = DEFAULT_X_MIN;
      let xMax = DEFAULT_X_MAX;
      let zMin = DEFAULT_Z_MIN;
      let zMax = DEFAULT_Z_MAX;

      if (floorplan?.bounds && floorplan.scale_m_per_px) {
        const b = floorplan.bounds;
        if (typeof b.min_x === 'number' && typeof b.max_x === 'number' &&
          typeof b.min_z === 'number' && typeof b.max_z === 'number') {
          xMin = b.min_x;
          xMax = b.max_x;
          zMin = b.min_z;
          zMax = b.max_z;
        }
      } else if (
        metaNow &&
        typeof metaNow.xMin === 'number' && typeof metaNow.xMax === 'number' &&
        typeof metaNow.zMin === 'number' && typeof metaNow.zMax === 'number'
      ) {
        xMin = metaNow.xMin;
        xMax = metaNow.xMax;
        zMin = metaNow.zMin;
        zMax = metaNow.zMax;
      }

      const drawX = (mx: number) => ((mx - xMin) / (xMax - xMin)) * width;
      const drawXFlipped = (mx: number) => width - ((mx - xMin) / (xMax - xMin)) * width;
      const drawY = (mz: number) => height - ((mz - zMin) / (zMax - zMin)) * height;

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

      const now = Date.now();

      const hueForId = (id: number) => (id * 47) % 360;
      const hsla = (id: number, a: number) => `hsla(${hueForId(id)}, 80%, 60%, ${a})`;

      // Time-based pruning must run even when BEV meta updates stop, otherwise
      // the last-seen trail head can stick around indefinitely.
      try {
        const state = smoothState.current;
        for (const [id, data] of state.entries()) {
          if (now - data.lastSeen > 1000) {
            state.delete(id);
          }
        }
      } catch {
        // defensive
      }
      try {
        const trails = trailsRef.current;
        if (!trailEnabled) {
          trails.clear();
        } else {
          for (const [key, track] of trails.entries()) {
            const pts = track.points;
            if (!pts.length) {
              trails.delete(key);
              continue;
            }
            let cut = 0;
            while (cut < pts.length && (now - pts[cut].t) > TRAIL_WINDOW_MS) cut += 1;
            if (cut > 0) track.points = pts.slice(cut);
            while (track.points.length && (!Number.isFinite(track.points[0].x) || !Number.isFinite(track.points[0].y))) {
              track.points.shift();
            }
            if (track.points.length === 0) {
              trails.delete(key);
              continue;
            }
            if (now - track.lastSeen > (TRAIL_WINDOW_MS + 2000)) {
              trails.delete(key);
            }
          }
          if (trails.size > TRAIL_MAX_TRACKS) {
            const ordered = Array.from(trails.entries()).sort((a, b) => a[1].lastSeen - b[1].lastSeen);
            for (let i = 0; i < ordered.length - TRAIL_MAX_TRACKS; i += 1) {
              trails.delete(ordered[i][0]);
            }
          }
        }
      } catch {
        // defensive
      }

      if (trailEnabled) {
        const tracksToDraw = Array.from(trailsRef.current.values());
        for (const tr of tracksToDraw) {
          const pts = tr.points;
          if (!pts || pts.length < 2) continue;

          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';
          ctx.lineWidth = TRAIL_LINE_WIDTH;

          let prev: TrailPoint | null = null;
          for (let i = 0; i < pts.length; i += 1) {
            const p = pts[i];
            const isGap = !Number.isFinite(p.x) || !Number.isFinite(p.y);
            if (isGap) {
              prev = null;
              continue;
            }
            if (!prev) {
              prev = p;
              continue;
            }

            const ageMs = Math.max(0, now - p.t);
            const frac = Math.max(0, Math.min(1, 1 - (ageMs / TRAIL_WINDOW_MS)));
            const alpha = TRAIL_MIN_ALPHA + (1 - TRAIL_MIN_ALPHA) * frac;

            ctx.strokeStyle = hsla(tr.colorId, alpha);
            ctx.beginPath();
            ctx.moveTo(drawXFlipped(prev.x), drawY(prev.y));
            ctx.lineTo(drawXFlipped(p.x), drawY(p.y));
            ctx.stroke();
            prev = p;
          }

          let lastValid: TrailPoint | null = null;
          for (let i = pts.length - 1; i >= 0; i -= 1) {
            const p = pts[i];
            if (Number.isFinite(p.x) && Number.isFinite(p.y)) {
              lastValid = p;
              break;
            }
          }
          if (lastValid) {
            const ageMs = Math.max(0, now - lastValid.t);
            const frac = Math.max(0, Math.min(1, 1 - (ageMs / TRAIL_WINDOW_MS)));
            const alpha = TRAIL_MIN_ALPHA + (1 - TRAIL_MIN_ALPHA) * frac;
            const staleMs = Math.max(0, now - (tr.lastSeen || 0));
            const blinkPhase = (2 * Math.PI * (now % TRAIL_STALE_BLINK_PERIOD_MS)) / TRAIL_STALE_BLINK_PERIOD_MS;
            const blink = staleMs >= TRAIL_STALE_BLINK_START_MS
              ? (0.35 + 0.65 * (0.5 + 0.5 * Math.sin(blinkPhase)))
              : 1.0;
            const px = drawXFlipped(lastValid.x);
            const py = drawY(lastValid.y);

            ctx.fillStyle = hsla(tr.colorId, Math.min(1, (alpha * blink) + 0.25));
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.65)';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(px, py, 4.5, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
          }
        }
      }

      smoothState.current.forEach((pt) => {
        const age = now - pt.lastSeen;
        if (age > 500) return;

        const px = drawXFlipped(pt.x);
        const py = drawY(pt.y);

        const alpha = Math.max(0, 1 - age / 500);
        const sidNum = pt.stableId ? Number(pt.stableId) : Number.NaN;
        const colorId = Number.isFinite(sidNum) ? sidNum : 0;

        ctx.globalAlpha = alpha;
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, 2 * Math.PI);
        ctx.fillStyle = `hsl(${hueForId(colorId)}, 80%, 60%)`;
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
  }, [floorplan, overlayEnabled, trailEnabled, variant]);

  const subtitleText = floorplan ? 'Height Map' : 'No Map Data';
  const subtitle = ` • ${subtitleText}`;

  const toggleLabel = (
    <label
      className={variant === 'drawer' ? 'td-overlay-toggle' : 'bev-grid-toggle'}
      style={variant === 'drawer' ? { background: 'rgba(0,0,0,0.5)', padding: '4px 8px', borderRadius: 4, color: 'white', fontSize: '12px' } : undefined}
    >
      <input
        type="checkbox"
        checked={overlayEnabled}
        onChange={(e) => setOverlayEnabled(e.target.checked)}
        style={{ marginRight: 6 }}
      />
      Grid
    </label>
  );

  const canvasContent = meta?.error ? (
    <div className={variant === 'drawer' ? 'td-placeholder' : 'bev-inline-placeholder'}>
      BEV Error: {meta.error}{meta.details ? ` — ${meta.details}` : ''}
    </div>
  ) : (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: '100%', objectFit: 'contain' }}
    />
  );

  if (variant === 'inline') {
    return (
      <div className="bev-inline-card">
        <div className="bev-inline-head">
          <span className="bev-inline-label">{label}</span>
          <span className="bev-inline-subtitle">{subtitleText}</span>
        </div>
        <div className="bev-inline-body">
          {canvasContent}
          <div className="bev-grid-toggle-wrap">
            {toggleLabel}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="td-cell">
      <div className="td-title">
        {label}
        <span className="td-subtitle">{subtitle}</span>
      </div>
      <div className="td-image-wrap" style={{ background: '#000', display: 'flex', justifyContent: 'center', alignItems: 'center', position: 'relative' }}>
        {canvasContent}
        <div style={{ position: 'absolute', bottom: 8, right: 8 }}>
          {toggleLabel}
        </div>
      </div>
    </div>
  );
};

export default BevView;
