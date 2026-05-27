import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CameraKey, cameraLabel, colorIdForPerson } from '../lib/camera';
import { FloorplanResponse } from './DepthDrawer';
import { renderLayerToCanvas, renderCompositeWalkableObstacleToCanvas, infernoColor, bwColor, decodeFloat32 } from '../lib/renderUtils';
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
    displaySource?: string | null;
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
  bev_points_smoothed?: boolean;
  bev_world_points_smoothed?: boolean;
};

export type BevFrameMode = CoordFrameMode;

type BevViewProps = {
  cam: CameraKey;
  meta?: BevMeta;
  floorplan?: FloorplanResponse;
  // 'world' means producer-owned metric world points, optionally projected to camera-local display.
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
type MetricBounds = { min_x: number; max_x: number; min_z: number; max_z: number };
type FloorplanSurfaceMap = {
  rows: number;
  cols: number;
  valid: Uint8Array;
  nearestRow: Int16Array;
  nearestCol: Int16Array;
  maxSnapCells: number;
};
type ResolvedMetricPoint = { x: number; y: number; mapped: boolean };
type FloorplanVisualSelection = {
  walkableLayer?: FloorplanResponse['walkable'];
  obstacleHeightLayer?: FloorplanResponse['obstacle_height'];
  heightLayer?: FloorplanResponse['height'];
  densityLayer?: FloorplanResponse['density'];
  distanceLayer?: FloorplanResponse['distance'];
  baseLayer?: FloorplanResponse['height'];
  baseKind: 'composite' | 'walkable' | 'obstacle_height' | 'height' | 'none';
  hasComposite: boolean;
  hasFloorplan: boolean;
};

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

const isMetricUnits = (units?: string): boolean => {
  const value = String(units || '').trim().toLowerCase();
  return value === 'm' || value === 'meter' || value === 'meters';
};

const isSceneUnits = (units?: string): boolean => String(units || '').trim().toLowerCase() === 'scene';

const hasCompatibleFloorplanUnits = (units?: string): boolean => isSceneUnits(units) || isMetricUnits(units);

const selectFloorplanVisualSelection = (
  floorplan: FloorplanResponse | undefined,
  isCompatible = true,
  preferHeightVisual = false
): FloorplanVisualSelection => {
  const walkableLayer = floorplan?.walkable;
  const obstacleHeightLayer = floorplan?.obstacle_height;
  const heightLayer = floorplan?.height;
  const densityLayer = floorplan?.density;
  const distanceLayer = floorplan?.distance;
  const hasWalkable = isCompatible && !!(walkableLayer?.grid_b64 && walkableLayer?.grid_shape);
  const hasObstacleHeight = isCompatible && !!(obstacleHeightLayer?.grid_b64 && obstacleHeightLayer?.grid_shape);
  const hasHeight = isCompatible && !!(heightLayer?.grid_b64 && heightLayer?.grid_shape);
  const hasComposite = hasWalkable && hasObstacleHeight;
  const useHeightVisual = preferHeightVisual && hasHeight;
  const baseLayer = useHeightVisual
    ? heightLayer
    : (hasComposite
      ? walkableLayer
      : (hasWalkable ? walkableLayer : (hasObstacleHeight ? obstacleHeightLayer : heightLayer)));
  const baseKind = useHeightVisual
    ? 'height'
    : (hasComposite
      ? 'composite'
      : (hasWalkable ? 'walkable' : (hasObstacleHeight ? 'obstacle_height' : (hasHeight ? 'height' : 'none'))));
  return {
    walkableLayer,
    obstacleHeightLayer,
    heightLayer,
    densityLayer,
    distanceLayer,
    baseLayer,
    baseKind,
    hasComposite: useHeightVisual ? false : hasComposite,
    hasFloorplan: !!(baseLayer?.grid_b64 && baseLayer?.grid_shape),
  };
};

const floorplanBoundsForMode = (
  floorplan: FloorplanResponse | undefined,
  coordMode: BevFrameMode
): MetricBounds | null => {
  const bounds = floorplan?.bounds;
  if (!bounds) return null;
  const minX = Number(bounds.min_x);
  const maxX = Number(bounds.max_x);
  const minZ = Number(bounds.min_z);
  const maxZ = Number(bounds.max_z);
  if (![minX, maxX, minZ, maxZ].every(Number.isFinite)) return null;

  const floorplanFrame = floorplan?.frame;
  const hasFloorplanFrame = typeof floorplanFrame === 'string' && floorplanFrame.trim().length > 0;
  const isFloorplanWorld = isWorldFrame(floorplanFrame);
  const isFloorplanCameraLocal = isCameraLocalFrame(floorplanFrame);
  const floorplanUnits = String(floorplan?.units || '').trim().toLowerCase();
  const hasFloorplanCompatibleUnits = hasCompatibleFloorplanUnits(floorplanUnits);
  const useBounds = coordMode === 'world'
    ? (hasFloorplanCompatibleUnits && (isFloorplanWorld || isFloorplanCameraLocal))
    : (!hasFloorplanFrame || isFloorplanCameraLocal);
  if (!useBounds) return null;
  return { min_x: minX, max_x: maxX, min_z: minZ, max_z: maxZ };
};

const buildFloorplanSurfaceMap = (floorplan: FloorplanResponse | undefined): FloorplanSurfaceMap | null => {
  if (!floorplan) return null;
  const visual = selectFloorplanVisualSelection(floorplan, true);
  const candidates = [
    visual.baseLayer,
    visual.obstacleHeightLayer,
    visual.walkableLayer,
    visual.heightLayer,
    visual.densityLayer,
    visual.distanceLayer,
  ].filter((layer) => layer?.grid_b64 && Array.isArray(layer.grid_shape));
  const base = candidates[0];
  if (!base?.grid_shape) return null;
  const [rowsRaw, colsRaw] = base.grid_shape;
  const rows = Number(rowsRaw);
  const cols = Number(colsRaw);
  if (!Number.isFinite(rows) || !Number.isFinite(cols) || rows <= 0 || cols <= 0) return null;
  const cellCount = rows * cols;
  const valid = new Uint8Array(cellCount);
  let markedCells = 0;

  const markLayer = (layer: typeof base | undefined, accepts: (value: number) => boolean): number => {
    if (!layer?.grid_b64 || !Array.isArray(layer.grid_shape)) return 0;
    const [layerRows, layerCols] = layer.grid_shape;
    if (Number(layerRows) !== rows || Number(layerCols) !== cols) return 0;
    const values = decodeFloat32(layer.grid_b64);
    if (!values || values.length < cellCount) return 0;
    let count = 0;
    for (let i = 0; i < cellCount; i += 1) {
      if (!valid[i] && accepts(values[i])) {
        valid[i] = 1;
        count += 1;
      }
    }
    return count;
  };

  if (visual.baseKind === 'composite') {
    markedCells += markLayer(visual.walkableLayer, (v) => Number.isFinite(v) && v > 0.5) ?? 0;
    markedCells += markLayer(visual.obstacleHeightLayer, (v) => Number.isFinite(v) && v > 0.05) ?? 0;
  } else if (visual.baseKind === 'walkable') {
    markedCells += markLayer(visual.walkableLayer, (v) => Number.isFinite(v) && v > 0.5) ?? 0;
  } else if (visual.baseKind === 'obstacle_height') {
    markedCells += markLayer(visual.obstacleHeightLayer, (v) => Number.isFinite(v) && v > 0.05) ?? 0;
  } else if (visual.baseKind === 'height') {
    markedCells += markLayer(visual.densityLayer, (v) => Number.isFinite(v) && v > 1e-6) ?? 0;
    markedCells += markLayer(visual.distanceLayer, (v) => Number.isFinite(v) && v > 0.0) ?? 0;
    markedCells += markLayer(visual.heightLayer, (v) => Number.isFinite(v) && v > 1e-6) ?? 0;
  }

  if (markedCells <= 0 || !valid.some((v) => v > 0)) return null;

  const nearestRow = new Int16Array(cellCount);
  const nearestCol = new Int16Array(cellCount);
  nearestRow.fill(-1);
  nearestCol.fill(-1);
  const queue = new Int32Array(cellCount);
  let head = 0;
  let tail = 0;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const idx = (row * cols) + col;
      if (!valid[idx]) continue;
      nearestRow[idx] = row;
      nearestCol[idx] = col;
      queue[tail] = idx;
      tail += 1;
    }
  }

  const offsets = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
    [-1, -1],
    [-1, 1],
    [1, -1],
    [1, 1],
  ];
  while (head < tail) {
    const idx = queue[head];
    head += 1;
    const row = Math.floor(idx / cols);
    const col = idx % cols;
    for (const [dr, dc] of offsets) {
      const rr = row + dr;
      const cc = col + dc;
      if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) continue;
      const nextIdx = (rr * cols) + cc;
      if (nearestRow[nextIdx] >= 0) continue;
      nearestRow[nextIdx] = nearestRow[idx];
      nearestCol[nextIdx] = nearestCol[idx];
      queue[tail] = nextIdx;
      tail += 1;
    }
  }

  return {
    rows,
    cols,
    valid,
    nearestRow,
    nearestCol,
    maxSnapCells: Math.max(2, Math.round(Math.min(rows, cols) * 0.08)),
  };
};

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
  const displayBounds = useMemo(
    () => floorplanBoundsForMode(displayFloorplan, coordMode),
    [
      coordMode,
      displayFloorplan?.frame,
      displayFloorplan?.units,
      displayFloorplan?.bounds?.min_x,
      displayFloorplan?.bounds?.max_x,
      displayFloorplan?.bounds?.min_z,
      displayFloorplan?.bounds?.max_z,
    ]
  );
  const displaySurfaceMap = useMemo(
    () => buildFloorplanSurfaceMap(displayFloorplan),
    [
      displayFloorplan?.walkable?.grid_b64,
      displayFloorplan?.obstacle_height?.grid_b64,
      displayFloorplan?.height?.grid_b64,
      displayFloorplan?.density?.grid_b64,
      displayFloorplan?.distance?.grid_b64,
    ]
  );
  const resolveDisplayPoint = useCallback((x: number, y: number): ResolvedMetricPoint | null => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    if (!displayBounds) return { x, y, mapped: false };

    const spanX = Math.max(1e-6, displayBounds.max_x - displayBounds.min_x);
    const spanZ = Math.max(1e-6, displayBounds.max_z - displayBounds.min_z);
    const colFloat = ((x - displayBounds.min_x) / spanX) * (displaySurfaceMap?.cols ?? 1);
    const rowFloat = (1.0 - ((y - displayBounds.min_z) / spanZ)) * (displaySurfaceMap?.rows ?? 1);

    if (!displaySurfaceMap) {
      if (x < displayBounds.min_x || x > displayBounds.max_x || y < displayBounds.min_z || y > displayBounds.max_z) return null;
      return { x, y, mapped: false };
    }

    const rawCol = Math.floor(colFloat);
    const rawRow = Math.floor(rowFloat);
    const col = Math.max(0, Math.min(displaySurfaceMap.cols - 1, rawCol));
    const row = Math.max(0, Math.min(displaySurfaceMap.rows - 1, rawRow));
    const edgeDistanceCells = Math.max(
      rawCol < 0 ? -rawCol : 0,
      rawCol >= displaySurfaceMap.cols ? rawCol - displaySurfaceMap.cols + 1 : 0,
      rawRow < 0 ? -rawRow : 0,
      rawRow >= displaySurfaceMap.rows ? rawRow - displaySurfaceMap.rows + 1 : 0
    );
    const idx = (row * displaySurfaceMap.cols) + col;
    if (displaySurfaceMap.valid[idx] && edgeDistanceCells === 0) {
      return { x, y, mapped: false };
    }

    const nearestRow = displaySurfaceMap.nearestRow[idx];
    const nearestCol = displaySurfaceMap.nearestCol[idx];
    if (nearestRow < 0 || nearestCol < 0) return null;
    const snapDistanceCells = Math.max(Math.abs(nearestRow - row), Math.abs(nearestCol - col), edgeDistanceCells);
    if (snapDistanceCells > displaySurfaceMap.maxSnapCells) return null;

    return {
      x: displayBounds.min_x + ((nearestCol + 0.5) / displaySurfaceMap.cols) * spanX,
      y: displayBounds.max_z - ((nearestRow + 0.5) / displaySurfaceMap.rows) * spanZ,
      mapped: true,
    };
  }, [displayBounds, displaySurfaceMap]);

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
      String(displayFloorplan?.snapshot_ts ?? displayFloorplan?.ts ?? ''),
      String(displayFloorplan?.walkable?.grid_b64?.length ?? ''),
      String(displayFloorplan?.obstacle_height?.grid_b64?.length ?? ''),
      String(displayFloorplan?.height?.grid_b64?.length ?? ''),
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
    displayFloorplan?.snapshot_ts,
    displayFloorplan?.ts,
    displayFloorplan?.walkable?.grid_b64,
    displayFloorplan?.obstacle_height?.grid_b64,
    displayFloorplan?.height?.grid_b64,
  ]);

  useEffect(() => {
    const arrivalNow = Date.now();
    const sampleNow = resolveTrailSampleNowMs(meta?.ts, arrivalNow, trailClockRef.current);
    const effectiveNow = Math.max(sampleNow, arrivalNow - MAX_TRAIL_CLOCK_SKEW_MS);
    const state = smoothState.current;
    const trails = trailsRef.current;
    const cfg = resolvedTrailConfig;
    const useBackendTrails = !frontendOwnsTrailSmoothing && Array.isArray(meta?.trails);

    // MINIMAL strengthening of the world-mode contract (per approved plan + design decisions):
    // When the producer owns trails (trail_smoothing_owner=backend and world mode), we must render the
    // emitted meta.trails directly and bypass all FE-side upsert/prune/smoothing. This is the only
    // supported path for canonical world BEV. The FE reconstruction path is kept only for legacy
    // camera_local payloads during transition.
    if (useBackendTrails) {
      // Fast path: trust the producer trails (already smoothed, keyed by tracker-local identity, in the
      // declared frame). No local history mutation.
      // (The drawing code later already has the `backendTracks` branch that consumes metaNow.trails verbatim.)
    }

    if (!cfg.enabled) {
      trails.clear();
      state.clear();
      trailFrameCounterRef.current = 0;
      trailClockRef.current = {};
      return;
    }

    const points = Array.isArray(meta?.footpoints) ? meta.footpoints : [];
    const dropUnresolvedPoint = (pt: typeof points[number]) => {
      const stableNum = typeof pt?.stableId === 'number' && Number.isFinite(pt.stableId) ? pt.stableId : null;
      const trackerNum = typeof pt?.trackerId === 'number' && Number.isFinite(pt.trackerId) ? pt.trackerId : null;
      const historyKey = historyKeyForPoint(stableNum, trackerNum);
      if (historyKey === null) return;
      state.delete(historyKey);
      trails.delete(historyKey);
    };

    if (useBackendTrails) {
      trails.clear();
      const seenIds = new Set<string>();
      points.forEach(pt => {
        const px = Number(pt.x);
        const py = Number(pt.y);
        const resolved = resolveDisplayPoint(px, py);
        if (!resolved) {
          dropUnresolvedPoint(pt);
          return;
        }
        const stableNum = typeof pt.stableId === 'number' && Number.isFinite(pt.stableId) ? pt.stableId : null;
        const trackerNum = typeof pt.trackerId === 'number' && Number.isFinite(pt.trackerId) ? pt.trackerId : null;
        const historyKey = historyKeyForPoint(stableNum, trackerNum);
        const displayId = displayIdForPoint(stableNum, trackerNum);
        if (historyKey === null || displayId === null) return;
        seenIds.add(historyKey);
        state.set(historyKey, {
          x: resolved.x,
          y: resolved.y,
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
        const resolved = resolveDisplayPoint(Number(pt.x), Number(pt.y));
        if (!resolved) {
          dropUnresolvedPoint(pt);
          return;
        }
        const targetX = resolved.x;
        const targetY = resolved.y;

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
  }, [cam, meta, resolveDisplayPoint, resolvedTrailConfig]);

  useEffect(() => {
    const render = () => {
      const cvs = canvasRef.current;
      if (!cvs) return;
      const ctx = cvs.getContext('2d');
      if (!ctx) return;

      const metaNow = metaRef.current;
      const floorplanNow = displayFloorplan;
      const aspect = variant === 'inline' ? 1.2 : (4 / 3);
      const fitMode = 'contain';

      const floorplanFrame = floorplanNow?.frame;
      const hasFloorplanFrame = typeof floorplanFrame === 'string' && floorplanFrame.trim().length > 0;
      const isFloorplanWorld = isWorldFrame(floorplanFrame);
      const isFloorplanCameraLocal = isCameraLocalFrame(floorplanFrame);
      const floorplanUnits = String(floorplanNow?.units || '').trim().toLowerCase();
      const hasFloorplanCompatibleUnits = hasCompatibleFloorplanUnits(floorplanUnits);
      const isFloorplanCompatible =
        coordMode === 'world'
          ? ((!hasFloorplanFrame || isFloorplanWorld || isFloorplanCameraLocal) && hasFloorplanCompatibleUnits)
          : !floorplanFrame || isFloorplanCameraLocal;

      let xMin = DEFAULT_X_MIN;
      let xMax = DEFAULT_X_MAX;
      let zMin = DEFAULT_Z_MIN;
      let zMax = DEFAULT_Z_MAX;

      if (displayBounds) {
        xMin = displayBounds.min_x;
        xMax = displayBounds.max_x;
        zMin = displayBounds.min_z;
        zMax = displayBounds.max_z;
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

      const visual = selectFloorplanVisualSelection(floorplanNow, isFloorplanCompatible, variant === 'inline');
      const walkableLayer = visual.walkableLayer;
      const obstacleHeightLayer = visual.obstacleHeightLayer;
      const baseLayer = visual.baseLayer;
      const baseKind = visual.baseKind;
      const hasComposite = visual.hasComposite;
      const basePalette = baseKind === 'walkable' ? bwColor : infernoColor;
      const hasFloorplan = visual.hasFloorplan;

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
      const resolveForDraw = (mx: number, mz: number): ResolvedMetricPoint | null => resolveDisplayPoint(mx, mz);
      const inBounds = (mx: number, mz: number) => resolveForDraw(mx, mz) !== null;

      const footpoints = Array.isArray(metaNow?.footpoints) ? metaNow.footpoints : [];
      let finitePointCount = 0;
      let inBoundsPointCount = 0;
      for (const p of footpoints) {
        const px = Number(p?.x);
        const py = Number(p?.y);
        if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
        finitePointCount += 1;
        if (resolveForDraw(px, py)) inBoundsPointCount += 1;
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
                .map((p) => {
                  if (!Number.isFinite(p.x) || !Number.isFinite(p.y) || !Number.isFinite(p.t)) return null;
                  const resolved = resolveForDraw(p.x, p.y);
                  if (!resolved) return { x: Number.NaN, y: Number.NaN, t: p.t };
                  return { x: resolved.x, y: resolved.y, t: p.t };
                })
                .filter((p): p is TrailPoint => p !== null);
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
            const resolved = Number.isFinite(p.x) && Number.isFinite(p.y) ? resolveForDraw(p.x, p.y) : null;
            const isGap = !resolved;
            if (isGap) {
              prev = null;
              continue;
            }
            const drawPoint = { ...p, x: resolved.x, y: resolved.y };
            if (!prev) {
              prev = drawPoint;
              continue;
            }

            const ageMs = Math.max(0, now - drawPoint.t);
            const frac = Math.max(0, Math.min(1, 1 - (ageMs / trailWindowMs)));
            const alpha = trailCfg.min_alpha + (1 - trailCfg.min_alpha) * frac;

            ctx.strokeStyle = hsla(tr.colorId, alpha);
            ctx.beginPath();
            ctx.moveTo(drawX(prev.x), drawY(prev.y));
            ctx.lineTo(drawX(drawPoint.x), drawY(drawPoint.y));
            ctx.stroke();
            prev = drawPoint;
          }

          let lastValid: TrailPoint | null = null;
          for (let i = pts.length - 1; i >= 0; i -= 1) {
            const p = pts[i];
            const resolved = Number.isFinite(p.x) && Number.isFinite(p.y) ? resolveForDraw(p.x, p.y) : null;
            if (resolved) {
              lastValid = { ...p, x: resolved.x, y: resolved.y };
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
        const resolved = Number.isFinite(pt.x) && Number.isFinite(pt.y) ? resolveForDraw(pt.x, pt.y) : null;
        if (!resolved) return;

        const px = drawX(resolved.x);
        const py = drawY(resolved.y);

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

      if (debug && finitePointCount > 0 && inBoundsPointCount === 0) {
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
  }, [coordMode, debug, displayBounds, floorplan, overlayEnabled, resolveDisplayPoint, resolvedTrailConfig, variant]);

  const floorplanFrame = displayFloorplan?.frame;
  const hasFloorplanFrame = typeof floorplanFrame === 'string' && floorplanFrame.trim().length > 0;
  const isFloorplanWorld = isWorldFrame(floorplanFrame);
  const isFloorplanCameraLocal = isCameraLocalFrame(floorplanFrame);
  const floorplanUnits = String(displayFloorplan?.units || '').trim().toLowerCase();
  const hasFloorplanCompatibleUnits = hasCompatibleFloorplanUnits(floorplanUnits);
  const isFloorplanCompatible =
    coordMode === 'world'
      ? ((!hasFloorplanFrame || isFloorplanWorld || isFloorplanCameraLocal) && hasFloorplanCompatibleUnits)
      : !floorplanFrame || isFloorplanCameraLocal;

  const visualSelection = selectFloorplanVisualSelection(displayFloorplan, isFloorplanCompatible, variant === 'inline');
  const hasWalkableLayer = visualSelection.baseKind === 'walkable' || visualSelection.baseKind === 'composite';
  const hasObstacleHeightLayer = visualSelection.baseKind === 'obstacle_height';
  const floorplanHasImage = visualSelection.hasFloorplan;

  const isFrameMismatch = coordMode === 'world' && isFloorplanCameraLocal;
  const baseLabel = hasWalkableLayer ? 'Walkable Map' : (hasObstacleHeightLayer ? 'Obstacle Height' : 'Height Map');
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
