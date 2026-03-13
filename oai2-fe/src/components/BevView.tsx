import React, { useEffect, useMemo, useRef, useState } from 'react';
import { CameraKey, cameraLabel, colorIdForPerson } from '../lib/camera';
import { FloorplanResponse } from './DepthDrawer';
import { renderLayerToCanvas, renderCompositeWalkableObstacleToCanvas, infernoColor, bwColor } from '../lib/renderUtils';
import type { BevFrameMode as CoordFrameMode } from '../lib/coordTransforms';
import { isCameraLocalFrame, isWorldFrame } from '../lib/coordTransforms';
import {
  type BevTrailConfig,
  type TrailPoint,
  type TrailTrack,
  computeSceneUnitsPerPx,
  normalizeBevTrailConfig,
  pruneTrailCollection,
  upsertTrailSample,
} from '../lib/bevTrails';

export type BevMeta = {
  type?: string;
  cameraId?: string;
  camId?: string;
  ts?: number;
  frame?: string;
  world_frame?: string;
  frame_mode?: string;
  units?: string;
  s_obj_to_m?: number;
  footpoints?: Array<{
    x: number;
    y: number;
    method: string;
    stableId?: number | null;
    trackerId?: number | null;
    anchorSource?: string | null;
    anchorQuality?: string | null;
    anchorReason?: string | null;
  }>;
  trails?: Array<{
    stableId?: number | null;
    trackerId?: number | null;
    points?: Array<{
      x: number;
      y: number;
      t: number;
    }>;
  }>;
  xMin?: number;
  xMax?: number;
  zMin?: number;
  zMax?: number;
  error?: string;
  details?: string;
  fallbackActive?: boolean;
  fallbackTrackCount?: number;
  fallbackSources?: string[];
  fallbackReasonCounts?: Record<string, number>;
  trail_smoothing_owner?: 'frontend' | 'backend' | 'none';
  bev_world_points_smoothed?: boolean;
};

export type BevFrameMode = CoordFrameMode;

type BevViewProps = {
  cam: CameraKey;
  meta?: BevMeta;
  floorplan?: FloorplanResponse;
  // 'world' means points are in MENON scene/world frame.
  // 'camera_local_legacy' keeps existing floorplan-local behavior.
  coordMode?: BevFrameMode;
  trailEnabled?: boolean;
  trailConfig?: Partial<BevTrailConfig>;
  debug?: boolean;
  variant?: 'drawer' | 'inline';
};

const DEFAULT_X_MIN = -4;
const DEFAULT_X_MAX = 4;
const DEFAULT_Z_MIN = 0;
const DEFAULT_Z_MAX = 12;

type ContentRect = { x: number; y: number; w: number; h: number };

const TRAIL_LINE_WIDTH = 2.5;

type TrailClock = {
  lastSrcMs?: number;
  lastSampleMs?: number;
};

// Keep source-clock smoothing from lagging visible motion by multiple seconds.
const MAX_TRAIL_CLOCK_SKEW_MS = 200;

const floorplanHasRenderableGrid = (floorplan?: FloorplanResponse | null): boolean => Boolean(
  floorplan?.walkable?.grid_b64 ||
  floorplan?.obstacle_height?.grid_b64 ||
  floorplan?.height?.grid_b64 ||
  floorplan?.density?.grid_b64 ||
  floorplan?.distance?.grid_b64
);

const normalizePayloadTsMs = (rawTs: unknown): number | null => {
  const value = Number(rawTs);
  if (!Number.isFinite(value) || value <= 0) return null;
  // 2020+ epoch in nanoseconds.
  if (value >= 1e17) return value / 1e6;
  // Typical epoch microseconds.
  if (value >= 1e14) return value / 1e3;
  // Typical epoch milliseconds.
  if (value >= 1e11) return value;
  // Epoch seconds.
  if (value >= 1e9) return value * 1e3;
  // Non-epoch stream PTS domain: keep as milliseconds for delta-only use.
  return value;
};

const resolveTrailSampleNowMs = (rawTs: unknown, arrivalNowMs: number, clock: TrailClock): number => {
  const srcMs = normalizePayloadTsMs(rawTs);
  if (srcMs === null) {
    clock.lastSrcMs = undefined;
    clock.lastSampleMs = arrivalNowMs;
    return arrivalNowMs;
  }

  const prevSrcMs = clock.lastSrcMs;
  const prevSampleMs = clock.lastSampleMs;
  clock.lastSrcMs = srcMs;

  if (!Number.isFinite(Number(prevSrcMs)) || !Number.isFinite(Number(prevSampleMs))) {
    clock.lastSampleMs = arrivalNowMs;
    return arrivalNowMs;
  }

  let deltaMs = srcMs - Number(prevSrcMs);
  if (!Number.isFinite(deltaMs) || deltaMs < 0) {
    clock.lastSampleMs = arrivalNowMs;
    return arrivalNowMs;
  }
  deltaMs = Math.min(deltaMs, 1000);

  const candidate = Number(prevSampleMs) + deltaMs;
  if (!Number.isFinite(candidate)) {
    clock.lastSampleMs = arrivalNowMs;
    return arrivalNowMs;
  }
  const clamped = Math.max(
    arrivalNowMs - MAX_TRAIL_CLOCK_SKEW_MS,
    Math.min(arrivalNowMs + MAX_TRAIL_CLOCK_SKEW_MS, candidate)
  );
  clock.lastSampleMs = clamped;
  return clamped;
};

const formatFallbackReason = (reason: string): string => {
  switch (reason) {
    case 'pose_meta_missing':
      return 'pose metadata missing';
    case 'pose_keypoints_unusable':
      return 'pose keypoints unusable';
    case 'height_lock_missing':
      return 'height lock missing';
    case 'pose_anchor_unavailable':
      return 'pose anchor unavailable';
    case 'pose_anchor_projection_failed':
      return 'pose anchor projection failed';
    default:
      return reason.replace(/_/g, ' ');
  }
};

export const BevView: React.FC<BevViewProps> = ({
  cam,
  meta,
  floorplan,
  coordMode = 'world',
  trailEnabled = true,
  trailConfig,
  debug = false,
  variant = 'drawer'
}) => {
  const label = cameraLabel(cam);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [overlayEnabled, setOverlayEnabled] = useState(false);

  const smoothState = useRef<Map<string, { x: number; y: number; lastSeen: number; stableId?: string; colorId: number }>>(new Map());
  const trailsRef = useRef<Map<string, TrailTrack>>(new Map());
  const animationFrameRef = useRef<number>();
  const bgCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const bgKeyRef = useRef<string>('');
  const bgSizeRef = useRef<{ w: number; h: number }>({ w: 0, h: 0 });
  const bgContentRectRef = useRef<ContentRect | null>(null);
  const historyKeyForPoint = (stableId?: number | null, trackerId?: number | null): string | null => {
    const trackerNum = typeof trackerId === 'number' && Number.isFinite(trackerId) ? trackerId : null;
    if (trackerNum !== null && trackerNum >= 0) return `t:${trackerNum}`;
    const stableNum = typeof stableId === 'number' && Number.isFinite(stableId) ? stableId : null;
    if (stableNum !== null && stableNum > 0) return `s:${stableNum}`;
    return null;
  };

  const displayIdForPoint = (stableId?: number | null, trackerId?: number | null): number | null => {
    const stableNum = typeof stableId === 'number' && Number.isFinite(stableId) ? stableId : null;
    if (stableNum !== null && stableNum > 0) return stableNum;
    const trackerNum = typeof trackerId === 'number' && Number.isFinite(trackerId) ? trackerId : null;
    if (trackerNum !== null && trackerNum >= 0) return trackerNum;
    return null;
  };

  const retainedFloorplanRef = useRef<FloorplanResponse | undefined>(floorplan);
  const trailSceneUnitsPerPxRef = useRef<number>(1.0);
  const trailFrameCounterRef = useRef<number>(0);
  const trailSpaceKeyRef = useRef<string>('');
  const trailClockRef = useRef<TrailClock>({});
  const metaRef = useRef<BevMeta | undefined>(meta);
  const trailSmoothingOwner = useMemo(() => {
    const owner = meta?.trail_smoothing_owner;
    if (owner === 'frontend' || owner === 'backend' || owner === 'none') return owner;
    return meta?.bev_world_points_smoothed ? 'backend' : 'frontend';
  }, [meta?.trail_smoothing_owner, meta?.bev_world_points_smoothed]);
  const frontendOwnsTrailSmoothing = trailSmoothingOwner === 'frontend';
  const resolvedTrailConfig = useMemo(
    () => {
      const normalized = normalizeBevTrailConfig({ ...trailConfig, enabled: trailEnabled });
      if (frontendOwnsTrailSmoothing) return normalized;
      return {
        ...normalized,
        smooth_tau_s: 0.0,
        head_smooth_tau_s: 0.0,
        trail_smooth_tau_s: 0.0,
        max_speed_px_per_s: 0.0,
      };
    },
    [frontendOwnsTrailSmoothing, trailConfig, trailEnabled]
  );
  const displayFloorplan = (floorplanHasRenderableGrid(floorplan) && !floorplan?.error)
    ? floorplan
    : (retainedFloorplanRef.current ?? floorplan);

  useEffect(() => {
    metaRef.current = meta;
  }, [meta]);

  useEffect(() => {
    if (floorplanHasRenderableGrid(floorplan) && !floorplan?.error) {
      retainedFloorplanRef.current = floorplan;
      return;
    }
    if (!retainedFloorplanRef.current) {
      retainedFloorplanRef.current = floorplan;
    }
  }, [floorplan]);

  useEffect(() => {
    const bounds = displayFloorplan?.bounds;
    const boundsKey = bounds
      ? [
          Number(bounds.min_x).toFixed(3),
          Number(bounds.max_x).toFixed(3),
          Number(bounds.min_z).toFixed(3),
          Number(bounds.max_z).toFixed(3),
        ].join(',')
      : 'none';
    const nextSpaceKey = [
      coordMode,
      String(displayFloorplan?.frame || '').trim().toLowerCase(),
      String(displayFloorplan?.units || '').trim().toLowerCase(),
      boundsKey,
    ].join('|');

    if (trailSpaceKeyRef.current && trailSpaceKeyRef.current !== nextSpaceKey) {
      trailsRef.current.clear();
      smoothState.current.clear();
      trailFrameCounterRef.current = 0;
      trailClockRef.current = {};
    }
    trailSpaceKeyRef.current = nextSpaceKey;
  }, [
    coordMode,
    displayFloorplan?.frame,
    displayFloorplan?.units,
    displayFloorplan?.bounds?.min_x,
    displayFloorplan?.bounds?.max_x,
    displayFloorplan?.bounds?.min_z,
    displayFloorplan?.bounds?.max_z,
  ]);

  useEffect(() => {
    const arrivalNow = Date.now();
    const sampleNow = resolveTrailSampleNowMs(meta?.ts, arrivalNow, trailClockRef.current);
    const effectiveNow = Math.max(sampleNow, arrivalNow - MAX_TRAIL_CLOCK_SKEW_MS);
    const state = smoothState.current;
    const trails = trailsRef.current;
    const cfg = resolvedTrailConfig;
    const useBackendTrails = !frontendOwnsTrailSmoothing && Array.isArray(meta?.trails);

    if (!cfg.enabled) {
      trails.clear();
      state.clear();
      trailFrameCounterRef.current = 0;
      trailClockRef.current = {};
      return;
    }

    const points = Array.isArray(meta?.footpoints) ? meta.footpoints : [];
    if (useBackendTrails) {
      trails.clear();
      const seenIds = new Set<string>();
      points.forEach(pt => {
        const stableNum = typeof pt.stableId === 'number' && Number.isFinite(pt.stableId) ? pt.stableId : null;
        const trackerNum = typeof pt.trackerId === 'number' && Number.isFinite(pt.trackerId) ? pt.trackerId : null;
        const historyKey = historyKeyForPoint(stableNum, trackerNum);
        const displayId = displayIdForPoint(stableNum, trackerNum);
        if (historyKey === null || displayId === null) return;
        seenIds.add(historyKey);
        state.set(historyKey, {
          x: Number(pt.x),
          y: Number(pt.y),
          lastSeen: arrivalNow,
          stableId: `${displayId}`,
          colorId: colorIdForPerson(cam, displayId),
        });
      });
      for (const [id, data] of state.entries()) {
        if (!seenIds.has(id) && (arrivalNow - data.lastSeen > 1000)) {
          state.delete(id);
        }
      }
      return;
    }

    if (points.length) {
      trailFrameCounterRef.current += 1;
      const doSample = (trailFrameCounterRef.current % cfg.draw_stride) === 0;
      const sceneUnitsPerPx = trailSceneUnitsPerPxRef.current;

      points.forEach(pt => {
        const targetX = pt.x;
        const targetY = pt.y;

        const stableNum = typeof pt.stableId === 'number' && Number.isFinite(pt.stableId) ? pt.stableId : null;
        const trackerNum = typeof pt.trackerId === 'number' && Number.isFinite(pt.trackerId) ? pt.trackerId : null;
        const historyKey = historyKeyForPoint(stableNum, trackerNum);
        const displayId = displayIdForPoint(stableNum, trackerNum);
        if (historyKey === null || displayId === null) return;
        const colorId = colorIdForPerson(cam, displayId);
        const labelText = `${displayId}`;

        const entry = trails.get(historyKey) ?? { points: [], lastSeen: 0, label: labelText, colorId };
        const update = upsertTrailSample(entry, {
          nowMs: effectiveNow,
          x: targetX,
          y: targetY,
          doSample,
          sceneUnitsPerPx,
          cfg,
        });

        state.set(historyKey, {
          x: update.headX,
          y: update.headY,
          lastSeen: arrivalNow,
          stableId: `${displayId}`,
          colorId,
        });

        entry.label = labelText;
        entry.colorId = colorId;
        trails.set(historyKey, entry);
      });
    }

    for (const [id, data] of state.entries()) {
      if (arrivalNow - data.lastSeen > 1000) {
        state.delete(id);
      }
    }

    pruneTrailCollection(trails, arrivalNow, cfg);
  }, [cam, meta, resolvedTrailConfig]);

  useEffect(() => {
    const render = () => {
      const cvs = canvasRef.current;
      if (!cvs) return;
      const ctx = cvs.getContext('2d');
      if (!ctx) return;

      const metaNow = metaRef.current;
      const floorplanNow = displayFloorplan;
      const aspect = variant === 'inline' ? 2 : (4 / 3);
      const fitMode = 'contain';

      const floorplanFrame = floorplanNow?.frame;
      const hasFloorplanFrame = typeof floorplanFrame === 'string' && floorplanFrame.trim().length > 0;
      const isFloorplanWorld = isWorldFrame(floorplanFrame);
      const isFloorplanCameraLocal = isCameraLocalFrame(floorplanFrame);
      const floorplanUnits = String(floorplanNow?.units || '').trim().toLowerCase();
      const hasSceneUnits = floorplanUnits === 'scene';
      const isFloorplanCompatible =
        coordMode === 'world'
          ? ((!hasFloorplanFrame || isFloorplanWorld || isFloorplanCameraLocal) && hasSceneUnits)
          : !floorplanFrame || isFloorplanCameraLocal;
      const useBoundsFromFloorplan = coordMode === 'world'
        ? (hasSceneUnits && (isFloorplanWorld || isFloorplanCameraLocal))
        : (!hasFloorplanFrame || isFloorplanCameraLocal);

      let xMin = DEFAULT_X_MIN;
      let xMax = DEFAULT_X_MAX;
      let zMin = DEFAULT_Z_MIN;
      let zMax = DEFAULT_Z_MAX;

      if (floorplanNow?.bounds && useBoundsFromFloorplan) {
        const b = floorplanNow.bounds;
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

      const boundsSpanX = Math.max(1e-6, xMax - xMin);
      const boundsSpanZ = Math.max(1e-6, zMax - zMin);
      const boundsAspect = boundsSpanX / boundsSpanZ;

      const walkableLayer = floorplanNow?.walkable;
      const obstacleHeightLayer = floorplanNow?.obstacle_height;
      const heightLayer = floorplanNow?.height;
      const hasWalkable = isFloorplanCompatible && !!(walkableLayer && walkableLayer.grid_b64 && walkableLayer.grid_shape);
      const hasObstacleHeight = isFloorplanCompatible && !!(obstacleHeightLayer && obstacleHeightLayer.grid_b64 && obstacleHeightLayer.grid_shape);
      const hasHeight = isFloorplanCompatible && !!(heightLayer && heightLayer.grid_b64 && heightLayer.grid_shape);
      const forceLegacyHeightInline = variant === 'inline' && cam === 'kitchen' && hasHeight;

      const hasComposite = !forceLegacyHeightInline && hasWalkable && hasObstacleHeight;
      const baseLayer = forceLegacyHeightInline
        ? heightLayer
        : (hasWalkable ? walkableLayer : (hasObstacleHeight ? obstacleHeightLayer : heightLayer));
      const baseKind = forceLegacyHeightInline
        ? 'height'
        : (hasComposite ? 'composite' : (hasWalkable ? 'walkable' : (hasObstacleHeight ? 'obstacle_height' : 'height')));
      const basePalette = baseKind === 'walkable' ? bwColor : infernoColor;
      const hasFloorplan = !!(baseLayer && baseLayer.grid_b64 && baseLayer.grid_shape);

      // Cache the floorplan render so we don't re-decode base64 every animation frame.
      const dpr = window.devicePixelRatio || 1;
      const rect = cvs.getBoundingClientRect();
      const canvasWidthCss = Math.max(1, rect.width || 1);
      const canvasHeightCss = Math.max(1, rect.height || 1);
      const expectedW = Math.max(1, Math.round(canvasWidthCss * dpr));
      const expectedH = Math.max(1, Math.round(canvasHeightCss * dpr));
      let padCss = 0;
      if (hasFloorplan && Array.isArray(baseLayer?.grid_shape)) {
        const [rows, cols] = baseLayer.grid_shape;
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
        ? `${baseKind}:${floorplanNow?.snapshot_ts ?? floorplanNow?.ts ?? ''}:${baseLayer?.grid_shape?.join('x')}:${baseLayer?.value_min ?? ''}:${baseLayer?.value_max ?? ''}:${baseLayer?.grid_b64?.length ?? ''}:${hasComposite ? (obstacleHeightLayer?.grid_b64?.length ?? '') : ''}:${aspect}:${fitMode}:${boundsAspect.toFixed(6)}:${padCss.toFixed(3)}`
        : `none:${aspect}:${fitMode}:${boundsAspect.toFixed(6)}:${padCss.toFixed(3)}`;

      const bg = bgCanvasRef.current ?? (bgCanvasRef.current = document.createElement('canvas'));
      const bgSize = bgSizeRef.current;
      const bgNeedsRedraw = bgKeyRef.current !== key || bgSize.w !== expectedW || bgSize.h !== expectedH;

      if (bgNeedsRedraw) {
        if (hasFloorplan) {
          const rendered = hasComposite
            ? renderCompositeWalkableObstacleToCanvas(cvs, walkableLayer, obstacleHeightLayer, {
              fit: fitMode,
              forceAspect: boundsAspect,
              contentPaddingPx: padCss
            })
            : renderLayerToCanvas(cvs, baseLayer, basePalette, {
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
      trailSceneUnitsPerPxRef.current = computeSceneUnitsPerPx({
        xMin,
        xMax,
        zMin,
        zMax,
        widthPx: contentRect.w,
        heightPx: contentRect.h,
      });

      const drawX = (mx: number) => contentRect.x + ((mx - xMin) / (xMax - xMin)) * contentRect.w;
      const drawY = (mz: number) => contentRect.y + contentRect.h - ((mz - zMin) / (zMax - zMin)) * contentRect.h;
      const inBounds = (mx: number, mz: number) => mx >= xMin && mx <= xMax && mz >= zMin && mz <= zMax;

      const footpoints = Array.isArray(metaNow?.footpoints) ? metaNow.footpoints : [];
      let finitePointCount = 0;
      let inBoundsPointCount = 0;
      for (const p of footpoints) {
        const px = Number(p?.x);
        const py = Number(p?.y);
        if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
        finitePointCount += 1;
        if (inBounds(px, py)) inBoundsPointCount += 1;
      }

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
      const trailCfg = resolvedTrailConfig;
      const trailWindowMs = Math.max(100, trailCfg.window_s * 1000.0);

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
        if (!trailCfg.enabled) {
          trails.clear();
        } else {
          pruneTrailCollection(trails, now, trailCfg);
        }
      } catch {
        // defensive
      }

      if (trailCfg.enabled) {
        const backendTracks = !frontendOwnsTrailSmoothing && Array.isArray(metaNow?.trails)
          ? metaNow.trails
            .map((tr) => {
              const stableNum = typeof tr?.stableId === 'number' && Number.isFinite(tr.stableId) ? tr.stableId : null;
              const trackerNum = typeof tr?.trackerId === 'number' && Number.isFinite(tr.trackerId) ? tr.trackerId : null;
              const displayId = displayIdForPoint(stableNum, trackerNum);
              if (displayId === null || !Array.isArray(tr?.points)) return null;
              const points = tr.points
                .map((p) => ({
                  x: Number(p?.x),
                  y: Number(p?.y),
                  t: Number(p?.t),
                }))
                .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.t));
              if (points.length < 2) return null;
              return {
                points,
                lastSeen: Number(points[points.length - 1]?.t) || now,
                label: `${displayId}`,
                colorId: colorIdForPerson(cam, displayId),
              } as TrailTrack;
            })
            .filter((tr): tr is TrailTrack => tr !== null)
          : null;
        const tracksToDraw = backendTracks ?? Array.from(trailsRef.current.values());
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
            const frac = Math.max(0, Math.min(1, 1 - (ageMs / trailWindowMs)));
            const alpha = trailCfg.min_alpha + (1 - trailCfg.min_alpha) * frac;

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
            const frac = Math.max(0, Math.min(1, 1 - (ageMs / trailWindowMs)));
            const alpha = trailCfg.min_alpha + (1 - trailCfg.min_alpha) * frac;
            const staleMs = Math.max(0, now - (tr.lastSeen || 0));
            let blink = 1.0;
            if (trailCfg.stale_head_blink_enabled && staleMs >= trailCfg.stale_blink_start_ms) {
              const blinkPhase = (2 * Math.PI * (now % trailCfg.stale_blink_period_ms)) / trailCfg.stale_blink_period_ms;
              blink = 0.35 + 0.65 * (0.5 + 0.5 * Math.sin(blinkPhase));
            }
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
        ctx.arc(px, py, 3.8, 0, 2 * Math.PI);
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

      // Camera marker at the bottom-center to preserve the BEV forward-view convention.
      const camPx = drawX((xMin + xMax) * 0.5);
      const camPy = drawY(zMin);
      ctx.fillStyle = 'rgba(255, 215, 64, 0.95)';
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.65)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(camPx - 10, camPy);
      ctx.lineTo(camPx + 10, camPy);
      ctx.lineTo(camPx, camPy - 16);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(camPx, camPy);
      ctx.lineTo(camPx, camPy - 26);
      ctx.stroke();

      if (finitePointCount > 0 && inBoundsPointCount === 0) {
        ctx.fillStyle = 'rgba(244, 67, 54, 0.9)';
        ctx.font = 'bold 11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Tracks are outside floorplan bounds', contentRect.x + (contentRect.w / 2), contentRect.y + 16);
        ctx.textAlign = 'left';
      }

      if (debug) {
        const lines = [
          `mode=${coordMode}`,
          `frame=${String(floorplanFrame || 'none')}`,
          `units=${String(metaNow?.units || floorplanNow?.units || 'unknown')}`,
          `pts=${finitePointCount} in=${inBoundsPointCount}`,
          `x:[${xMin.toFixed(2)},${xMax.toFixed(2)}] z:[${zMin.toFixed(2)},${zMax.toFixed(2)}]`,
        ];
        const panelPad = 6;
        const lineH = 13;
        const panelW = 250;
        const panelH = panelPad * 2 + lines.length * lineH;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.62)';
        ctx.fillRect(contentRect.x + 8, contentRect.y + 8, panelW, panelH);
        ctx.fillStyle = '#d8f5d1';
        ctx.font = '11px monospace';
        for (let i = 0; i < lines.length; i += 1) {
          ctx.fillText(lines[i], contentRect.x + 8 + panelPad, contentRect.y + 8 + panelPad + ((i + 1) * lineH) - 3);
        }
      }

      animationFrameRef.current = requestAnimationFrame(render);
    };

    render();
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [coordMode, debug, floorplan, overlayEnabled, resolvedTrailConfig, variant]);

  const floorplanFrame = displayFloorplan?.frame;
  const hasFloorplanFrame = typeof floorplanFrame === 'string' && floorplanFrame.trim().length > 0;
  const isFloorplanWorld = isWorldFrame(floorplanFrame);
  const isFloorplanCameraLocal = isCameraLocalFrame(floorplanFrame);
  const floorplanUnits = String(displayFloorplan?.units || '').trim().toLowerCase();
  const hasSceneUnits = floorplanUnits === 'scene';
  const isFloorplanCompatible =
    coordMode === 'world'
      ? ((!hasFloorplanFrame || isFloorplanWorld || isFloorplanCameraLocal) && hasSceneUnits)
      : !floorplanFrame || isFloorplanCameraLocal;

  const hasWalkableLayer = isFloorplanCompatible && !!(displayFloorplan?.walkable?.grid_b64 && displayFloorplan?.walkable?.grid_shape);
  const hasObstacleHeightLayer = isFloorplanCompatible && !!(displayFloorplan?.obstacle_height?.grid_b64 && displayFloorplan?.obstacle_height?.grid_shape);
  const hasHeightLayer = isFloorplanCompatible && !!(displayFloorplan?.height?.grid_b64 && displayFloorplan?.height?.grid_shape);
  const forceLegacyHeightInline = variant === 'inline' && cam === 'kitchen' && hasHeightLayer;
  const floorplanHasImage = hasWalkableLayer || hasObstacleHeightLayer || hasHeightLayer;

  const isFrameMismatch = coordMode === 'world' && isFloorplanCameraLocal;
  const baseLabel = forceLegacyHeightInline
    ? 'Height Map'
    : (hasWalkableLayer ? 'Walkable Map' : (hasObstacleHeightLayer ? 'Obstacle Height' : 'Height Map'));
  const subtitleText = floorplanHasImage ? (isFrameMismatch ? `${baseLabel} (local-floorplan fallback)` : baseLabel) : 'No Map Data';
  const subtitle = ` • ${subtitleText}`;
  const fallbackActive = Boolean(meta?.fallbackActive);
  const fallbackTrackCount = Number.isFinite(Number(meta?.fallbackTrackCount))
    ? Math.max(0, Number(meta?.fallbackTrackCount))
    : 0;
  const fallbackReasonEntries = Object.entries(meta?.fallbackReasonCounts ?? {})
    .filter(([, count]) => Number.isFinite(Number(count)) && Number(count) > 0)
    .sort((a, b) => Number(b[1]) - Number(a[1]));
  const fallbackReasonText = fallbackReasonEntries.length
    ? fallbackReasonEntries
        .slice(0, 2)
        .map(([reason, count]) => {
          const label = formatFallbackReason(String(reason));
          const numericCount = Number(count);
          return numericCount > 1 ? `${label} x${numericCount}` : label;
        })
        .join(', ')
    : 'pose-first anchor unavailable';
  const fallbackBanner = fallbackActive ? (
    <div className="bev-anchor-banner" role="status" aria-live="polite">
      <strong>Legacy fallback</strong>
      <span>
        {fallbackTrackCount > 0 ? `${fallbackTrackCount} track${fallbackTrackCount === 1 ? '' : 's'} on bbox-bottom` : 'bbox-bottom anchor'}
        {fallbackReasonText ? ` • ${fallbackReasonText}` : ''}
      </span>
    </div>
  ) : null;

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
          {fallbackBanner}
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
        {fallbackBanner}
        {canvasContent}
        <div style={{ position: 'absolute', bottom: 8, right: 8 }}>
          {toggleLabel}
        </div>
      </div>
    </div>
  );
};

export default BevView;
