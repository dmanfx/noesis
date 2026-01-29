import React, { useEffect, useRef, useState } from 'react';
import { CameraKey, cameraLabel, colorIdForPerson, identityKeyForPerson } from '../lib/camera';
import { FloorplanResponse } from './DepthDrawer';
import { renderLayerToCanvas, infernoColor } from '../lib/renderUtils';

export type BevMeta = {
  cameraId?: string;
  camId?: string;
  footpoints?: Array<{ x: number; y: number; method: string; stableId: number }>;
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
  trailEnabled?: boolean;
  variant?: 'drawer' | 'inline';
};

const DEFAULT_X_MIN = -4;
const DEFAULT_X_MAX = 4;
const DEFAULT_Z_MIN = 0;
const DEFAULT_Z_MAX = 12;

type TrailPoint = { x: number; y: number; t: number };
type TrailTrack = { points: TrailPoint[]; lastSeen: number; label: string; colorId: number };
type ContentRect = { x: number; y: number; w: number; h: number };

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
  trailEnabled = true,
  variant = 'drawer'
}) => {
  const label = cameraLabel(cam);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [overlayEnabled, setOverlayEnabled] = useState(false);

  const smoothState = useRef<Map<number, { x: number; y: number; lastSeen: number; stableId?: string; colorId: number }>>(new Map());
  const trailsRef = useRef<Map<string, TrailTrack>>(new Map());
  const animationFrameRef = useRef<number>();
  const bgCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const bgKeyRef = useRef<string>('');
  const bgSizeRef = useRef<{ w: number; h: number }>({ w: 0, h: 0 });
  const bgContentRectRef = useRef<ContentRect | null>(null);
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
        const targetX = pt.x;
        const targetY = pt.y;

        const stableNum = typeof pt.stableId === 'number' && Number.isFinite(pt.stableId) ? pt.stableId : null;
        if (stableNum === null || stableNum <= 0) return;
        const colorId = colorIdForPerson(cam, stableNum);

        state.set(stableNum, {
          x: targetX,
          y: targetY,
          lastSeen: now,
          stableId: `${stableNum}`,
          colorId
        });

        if (!trailEnabled) return;

        const key = identityKeyForPerson(cam, stableNum);
        const labelText = `${stableNum}`;

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
  }, [meta, trailEnabled]);

  useEffect(() => {
    const render = () => {
      const cvs = canvasRef.current;
      if (!cvs) return;
      const ctx = cvs.getContext('2d');
      if (!ctx) return;

      const metaNow = metaRef.current;
      const aspect = variant === 'inline' ? 2 : (4 / 3);
      const fitMode = 'contain';

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

      if (floorplan?.bounds && metaNow?.footpoints?.length) {
        let minX = Number.POSITIVE_INFINITY;
        let maxX = Number.NEGATIVE_INFINITY;
        let minZ = Number.POSITIVE_INFINITY;
        let maxZ = Number.NEGATIVE_INFINITY;

        metaNow.footpoints.forEach((pt) => {
          if (!Number.isFinite(pt.x) || !Number.isFinite(pt.y)) return;
          minX = Math.min(minX, pt.x);
          maxX = Math.max(maxX, pt.x);
          minZ = Math.min(minZ, pt.y);
          maxZ = Math.max(maxZ, pt.y);
        });

        if (Number.isFinite(minX) && Number.isFinite(maxX) && Number.isFinite(minZ) && Number.isFinite(maxZ)) {
          const spanX = Math.max(1e-3, xMax - xMin);
          const spanZ = Math.max(1e-3, zMax - zMin);
          const padX = Math.max(0.5, spanX * 0.05);
          const padZ = Math.max(0.5, spanZ * 0.05);
          let nextXMin = Math.min(xMin, minX - padX);
          let nextXMax = Math.max(xMax, maxX + padX);
          let nextZMin = Math.min(zMin, minZ - padZ);
          let nextZMax = Math.max(zMax, maxZ + padZ);

          const maxExpand = 2.5;
          const maxSpanX = spanX * maxExpand;
          const maxSpanZ = spanZ * maxExpand;
          if ((nextXMax - nextXMin) > maxSpanX) {
            const cx = (nextXMax + nextXMin) * 0.5;
            nextXMin = cx - maxSpanX * 0.5;
            nextXMax = cx + maxSpanX * 0.5;
          }
          if ((nextZMax - nextZMin) > maxSpanZ) {
            const cz = (nextZMax + nextZMin) * 0.5;
            nextZMin = cz - maxSpanZ * 0.5;
            nextZMax = cz + maxSpanZ * 0.5;
          }

          xMin = nextXMin;
          xMax = nextXMax;
          zMin = nextZMin;
          zMax = nextZMax;
        }
      }

      const boundsSpanX = Math.max(1e-6, xMax - xMin);
      const boundsSpanZ = Math.max(1e-6, zMax - zMin);
      const boundsAspect = boundsSpanX / boundsSpanZ;

      const heightLayer = floorplan?.height;
      const hasFloorplan = !!(heightLayer && heightLayer.grid_b64 && heightLayer.grid_shape);

      // Cache the floorplan render so we don't re-decode base64 every animation frame.
      const dpr = window.devicePixelRatio || 1;
      const rect = cvs.getBoundingClientRect();
      const canvasWidthCss = Math.max(1, rect.width || 1);
      const canvasHeightCss = Math.max(1, rect.height || 1);
      const expectedW = Math.max(1, Math.round(canvasWidthCss * dpr));
      const expectedH = Math.max(1, Math.round(canvasHeightCss * dpr));
      let padCss = 0;
      if (hasFloorplan && Array.isArray(heightLayer?.grid_shape)) {
        const [rows, cols] = heightLayer.grid_shape;
        if (rows && cols) {
          const canvasAspect = canvasWidthCss / canvasHeightCss;
          let contentW = canvasWidthCss;
          let contentH = canvasHeightCss;
          if (canvasAspect > boundsAspect) {
            contentH = canvasHeightCss;
            contentW = canvasHeightCss * boundsAspect;
          } else {
            contentW = canvasWidthCss;
            contentH = canvasWidthCss / boundsAspect;
          }
          const cellPx = Math.min(contentW / cols, contentH / rows);
          padCss = Math.max(0, cellPx * 0.5);
        }
      }

      const key = hasFloorplan
        ? `${floorplan?.snapshot_ts ?? floorplan?.ts ?? ''}:${heightLayer?.grid_shape?.join('x')}:${heightLayer?.value_min ?? ''}:${heightLayer?.value_max ?? ''}:${heightLayer?.grid_b64?.length ?? ''}:${aspect}:${fitMode}:${boundsAspect.toFixed(6)}:${padCss.toFixed(3)}`
        : `none:${aspect}:${fitMode}:${boundsAspect.toFixed(6)}:${padCss.toFixed(3)}`;

      const bg = bgCanvasRef.current ?? (bgCanvasRef.current = document.createElement('canvas'));
      const bgSize = bgSizeRef.current;
      const bgNeedsRedraw = bgKeyRef.current !== key || bgSize.w !== expectedW || bgSize.h !== expectedH;

      if (bgNeedsRedraw) {
        if (hasFloorplan) {
          const rendered = renderLayerToCanvas(cvs, heightLayer, infernoColor, {
            fit: fitMode,
            forceAspect: boundsAspect,
            contentPaddingPx: padCss
          });
          bgContentRectRef.current = rendered?.contentRectPx ?? { x: 0, y: 0, w: cvs.width, h: cvs.height };
        } else {
          cvs.width = expectedW;
          cvs.height = expectedH;
          ctx.fillStyle = '#111';
          ctx.fillRect(0, 0, cvs.width, cvs.height);
          bgContentRectRef.current = { x: 0, y: 0, w: cvs.width, h: cvs.height };
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
      const contentRect = bgContentRectRef.current ?? { x: 0, y: 0, w: width, h: height };

      const drawX = (mx: number) => contentRect.x + ((mx - xMin) / (xMax - xMin)) * contentRect.w;
      const drawY = (mz: number) => contentRect.y + contentRect.h - ((mz - zMin) / (zMax - zMin)) * contentRect.h;
      const inBounds = (mx: number, mz: number) => mx >= xMin && mx <= xMax && mz >= zMin && mz <= zMax;

      if (overlayEnabled) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        const startX = Math.ceil(xMin);
        for (let x = startX; x <= xMax; x++) {
          const u = drawX(x);
          ctx.moveTo(u, contentRect.y);
          ctx.lineTo(u, contentRect.y + contentRect.h);
        }
        const startZ = Math.ceil(zMin);
        for (let z = startZ; z <= zMax; z++) {
          const v = drawY(z);
          ctx.moveTo(contentRect.x, v);
          ctx.lineTo(contentRect.x + contentRect.w, v);
        }
        ctx.stroke();

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        if (zMin <= 0 && zMax >= 0) {
          const v0 = drawY(0);
          ctx.moveTo(contentRect.x, v0);
          ctx.lineTo(contentRect.x + contentRect.w, v0);
        }
        if (xMin <= 0 && xMax >= 0) {
          const u0 = drawX(0);
          ctx.moveTo(u0, contentRect.y);
          ctx.lineTo(u0, contentRect.y + contentRect.h);
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
            const isGap = !Number.isFinite(p.x) || !Number.isFinite(p.y) || !inBounds(p.x, p.y);
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
            ctx.moveTo(drawX(prev.x), drawY(prev.y));
            ctx.lineTo(drawX(p.x), drawY(p.y));
            ctx.stroke();
            prev = p;
          }

          let lastValid: TrailPoint | null = null;
          for (let i = pts.length - 1; i >= 0; i -= 1) {
            const p = pts[i];
            if (Number.isFinite(p.x) && Number.isFinite(p.y) && inBounds(p.x, p.y)) {
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
            const px = drawX(lastValid.x);
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
        if (!Number.isFinite(pt.x) || !Number.isFinite(pt.y) || !inBounds(pt.x, pt.y)) return;

        const px = drawX(pt.x);
        const py = drawY(pt.y);

        const alpha = Math.max(0, 1 - age / 500);
        const colorId = Number.isFinite(pt.colorId) ? pt.colorId : 0;

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
