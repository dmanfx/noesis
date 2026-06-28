import { PointerEvent as ReactPointerEvent, memo, useCallback, useEffect, useMemo, useRef, useState, RefObject } from 'react';
import { CameraKey, cameraIndex, cameraLabel, detectCameraKey } from '../lib/camera';
import type { MosaicLayout } from '../hooks/useWebSocketClient';
import '../styles/depth-drawer.css';

type DepthEntry = {
  ts: number;
  depth_b64: string;
  conf_b64?: string;
  mask_b64?: string;
  shape: [number, number];
  normals_b64?: string;
  normals_shape?: [number, number, number];
  normals_dtype?: 'float16' | 'float32';
  normals_space?: 'camera' | 'world';
  normals_error?: string;
};

export type DepthMetaEntry = {
  tsUs?: number;
  servedFromCache?: boolean;
  requestId?: string;
  error?: string;
  // Raw camera id as reported by backend (may differ from UI key if remapped)
  sourceCameraId?: string;
};

type DiagnosticsEntry = {
  summary: {
    median?: number;
    p10?: number;
    p90?: number;
    conf_mean?: number;
    valid_ratio?: number;
    sample_count?: number;
    method?: string;
  };
  ts: number;
};

export type FloorplanLayer = {
  grid_b64?: string;
  grid_shape?: [number, number];
  value_min?: number;
  value_max?: number;
};

export type FloorplanResponse = {
  type?: string;
  request_id?: string;
  camera_id?: string;
  ts?: number;
  snapshot_ts?: number | null;
  served_from_cache?: boolean;
  cache_only?: boolean;
  frame?: string;
  bounds?: { min_x?: number; max_x?: number; min_z?: number; max_z?: number };
  scale_m_per_px?: number;
  scale_scene_per_px?: number;
  units?: string;
  s_obj_to_m?: number;
  grid_res_scene?: number;
  max_extent_scene?: number;
  point_count?: number;
  error?: string;
  density?: FloorplanLayer;
  height?: FloorplanLayer;
  height_agl?: FloorplanLayer;
  distance?: FloorplanLayer;
  gradient?: FloorplanLayer;
  obstacle_height?: FloorplanLayer;
  walkable?: FloorplanLayer;
  height_agl_meta?: { floor_y?: number; floor_estimate?: unknown };
};

type FloorplanRequestOptions = {
  camera?: string;
  requestId?: string;
  maxAgeSec?: number;
  gridResM?: number;
  maxExtentM?: number;
  cacheOnly?: boolean;
};

interface DepthDrawerProps {
  open: boolean;
  onClose: () => void;
  diagnostics: Record<string, DiagnosticsEntry>;
  depthData: Record<string, DepthEntry>;
  depthMeta?: Record<string, DepthMetaEntry>;
  onRequestDepthFresh: (cameraId: string) => void;
  onRequestDepthCached: (cameraId: string) => void;
  onHeatmapReady: (cameraId: string) => void;
  floorplans: Record<string, FloorplanResponse>;
  onRequestFloorplan: (options: FloorplanRequestOptions) => string | void;
  availableCameras: string[];
  mosaicLayout?: MosaicLayout | null;
  videoRef?: RefObject<HTMLVideoElement>;
  calibrationEpoch?: number;
}

import {
  applyCanvasSize,
  decodeFloat32,
  decodeFloat16,
  decodeUint8,
  grayscaleColor,
  bwColor,
  infernoColor,
  renderCompositeWalkableObstacleToCanvas,
  renderLayerToCanvas,
  turboColor,
  viridisColor
} from '../lib/renderUtils';
import { buildExtrudedFloorplanModel, DEFAULT_OBSTACLE_SETTINGS, renderExtrudedFloorplanToCanvas } from '../lib/extrudedFloorplan';
import { getExtrinsicsAny, getIntrinsicsAny } from '../lib/calibration';
import { buildVisibleFloorPlaneModel } from '../lib/visibleFloorPlane';
import FloorPlane3DView from './FloorPlane3DView';

const DEFAULT_WIDTH = 700;
const MAX_WIDTH = 960;
const WIDE_ASPECT = 16 / 9;
const EXTRUDED_ASPECT = 4 / 3;

// Display-only settings for a high-contrast height visualization (floor vs countertops).
const HEIGHT_CONTRAST_PCT_LO = 5;
const HEIGHT_CONTRAST_PCT_HI = 95;
const HEIGHT_CONTRAST_GAMMA = 1.0;
const HEIGHT_CONTRAST_DENSITY_THRESH = 1e-6;

// Display-only: height above estimated floor (AGL). Use a fixed range so floor vs countertop pops.
const HEIGHT_AGL_VIEW_MIN_M = 0.0;
const HEIGHT_AGL_VIEW_MAX_M = 1.2;

function percentileSorted(sorted: number[], pct: number): number {
  if (!sorted.length) return 0;
  const p = Math.min(100, Math.max(0, pct));
  if (sorted.length === 1) return sorted[0];
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const frac = idx - lo;
  return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
}

function generateGradient(palette: (t: number) => [number, number, number], steps = 12): string {
  const stops: string[] = [];
  for (let i = 0; i <= steps; i += 1) {
    const t = i / steps;
    const [r, g, b] = palette(t);
    stops.push(`rgb(${r}, ${g}, ${b}) ${(t * 100).toFixed(1)}%`);
  }
  return `linear-gradient(to top, ${stops.join(', ')})`;
}

const turboGradient = generateGradient(turboColor);
const viridisGradient = generateGradient((t) => viridisColor(t));
const infernoGradient = generateGradient(infernoColor);
const densityGradient = 'linear-gradient(to top, rgb(0,0,0) 0%, rgb(255,255,255) 100%)';

function formatNumber(value?: number | null, digits = 2): string {
  if (value === undefined || value === null || Number.isNaN(value)) return 'n/a';
  const factor = 10 ** digits;
  return `${Math.round(value * factor) / factor}`;
}

const DepthDrawer = memo(function DepthDrawer({
  open,
  onClose,
  diagnostics,
  depthData,
  depthMeta = {},
  onRequestDepthFresh,
  onRequestDepthCached,
  onHeatmapReady,
  floorplans,
  onRequestFloorplan,
  availableCameras,
  mosaicLayout,
  videoRef,
  calibrationEpoch = 0,
}: DepthDrawerProps) {
  // Show cameras from either available list or present depth data (union)
  const cameras = useMemo(() => {
    const set = new Set<string>();
    (availableCameras || []).forEach((c) => { if (c) set.add(c); });
    Object.keys(depthData || {}).forEach((c) => { if (c) set.add(c); });
    return Array.from(set).sort();
  }, [availableCameras, depthData]);
  const [activeTab, setActiveTab] = useState<'heatmap' | 'normals' | '3d' | 'stats' | 'histogram' | 'metrics'>('heatmap');
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [drawerWidth, setDrawerWidth] = useState<number>(DEFAULT_WIDTH);
  const drawerRef = useRef<HTMLDivElement | null>(null);
  const isResizingRef = useRef(false);
  const previousUserSelectRef = useRef('');
  const activePointerIdRef = useRef<number | null>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const normalsCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const histogramCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const densityCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const heightCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const heightContrastCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const heightAglCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const distanceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const gradientCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const obstacleHeightCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const walkableCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const floorplanCompositeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamPreviewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const extrudedCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const floorPlaneCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [primitivesView, setPrimitivesView] = useState<'obstacles' | 'visible-floor'>('obstacles');
  const [primitivesShowPreview, setPrimitivesShowPreview] = useState(true);
  const [primitivesShowBoxes, setPrimitivesShowBoxes] = useState(true);
  const [primitivesSelectedBoxId, setPrimitivesSelectedBoxId] = useState<string | null>(null);
  const [primitivesHeightExaggeration, setPrimitivesHeightExaggeration] = useState(1.6);
  const [primitivesMinHeightM, setPrimitivesMinHeightM] = useState(DEFAULT_OBSTACLE_SETTINGS.minHeightM);
  const [primitivesMinDensity, setPrimitivesMinDensity] = useState(DEFAULT_OBSTACLE_SETTINGS.minDensity);
  const [primitivesMinFootprintM2, setPrimitivesMinFootprintM2] = useState(DEFAULT_OBSTACLE_SETTINGS.minFootprintM2);
  const [primitivesMaxBoxes, setPrimitivesMaxBoxes] = useState(DEFAULT_OBSTACLE_SETTINGS.maxBoxes);
  const [primitivesMaxCells, setPrimitivesMaxCells] = useState(DEFAULT_OBSTACLE_SETTINGS.maxCells);
  const [floorplanStatus, setFloorplanStatus] = useState<'idle' | 'loading' | 'checking'>('idle');
  const [floorplanRequest, setFloorplanRequest] = useState<string>('');
  const [heatmapRange, setHeatmapRange] = useState<{ min: number; max: number } | null>(null);
  const floorplanWarmupTimersRef = useRef<number[]>([]);
  const floorplanPrefetchScheduledRef = useRef<Set<string>>(new Set());
  const floorplanPrefetchedRef = useRef<Set<string>>(new Set());
  const onRequestFloorplanRef = useRef(onRequestFloorplan);
  const heatmapNotifiedRef = useRef<Set<string>>(new Set());

  // Floorplan selection (declared early to avoid TDZ in hooks below)
  const cameraFloorplan = floorplans[selectedCamera];
  const densityLayer = cameraFloorplan?.density;
  const heightLayer = cameraFloorplan?.height;
  const heightAglLayer = cameraFloorplan?.height_agl;
  const distanceLayer = cameraFloorplan?.distance;
  const gradientLayer = cameraFloorplan?.gradient;
  const obstacleHeightLayer = cameraFloorplan?.obstacle_height;
  const walkableLayer = cameraFloorplan?.walkable;
  const floorplanError = cameraFloorplan?.error ?? null;
  const floorplanServedFromCache = cameraFloorplan?.served_from_cache ?? false;
  const hasDensity = !!(densityLayer && densityLayer.grid_b64 && densityLayer.grid_shape);
  const hasHeight = !!(heightLayer && heightLayer.grid_b64 && heightLayer.grid_shape);
  const hasHeightAgl = !!(heightAglLayer && heightAglLayer.grid_b64 && heightAglLayer.grid_shape);
  const hasDistance = !!(distanceLayer && distanceLayer.grid_b64 && distanceLayer.grid_shape);
  const hasGradient = !!(gradientLayer && gradientLayer.grid_b64 && gradientLayer.grid_shape);
  const hasObstacleHeight = !!(obstacleHeightLayer && obstacleHeightLayer.grid_b64 && obstacleHeightLayer.grid_shape);
  const hasWalkable = !!(walkableLayer && walkableLayer.grid_b64 && walkableLayer.grid_shape);
  const heightBase = heightLayer?.value_min ?? null;
  const heightMaxRaw = heightLayer?.value_max ?? null;
  const heightSpan = (heightBase !== null && heightMaxRaw !== null) ? Math.max(0, heightMaxRaw - heightBase) : null;
  const heightContrastRange = useMemo(() => {
    if (!heightLayer?.grid_b64 || !heightLayer?.grid_shape) return null;
    const [rows, cols] = heightLayer.grid_shape;
    if (!rows || !cols) return null;
    const heightValues = decodeFloat32(heightLayer.grid_b64);
    if (!heightValues || heightValues.length < rows * cols) return null;

    let densityValues: Float32Array | null = null;
    if (densityLayer?.grid_b64 && densityLayer?.grid_shape) {
      const [dRows, dCols] = densityLayer.grid_shape;
      if (dRows === rows && dCols === cols) {
        densityValues = decodeFloat32(densityLayer.grid_b64);
      }
    }

    const samples: number[] = [];
    const n = rows * cols;
    for (let idx = 0; idx < n; idx += 1) {
      const v = heightValues[idx];
      if (!Number.isFinite(v)) continue;
      if (densityValues) {
        const d = densityValues[idx];
        if (!Number.isFinite(d) || d <= HEIGHT_CONTRAST_DENSITY_THRESH) continue;
      }
      samples.push(v);
    }
    if (samples.length < 16) return null;
    samples.sort((a, b) => a - b);

    const lo = percentileSorted(samples, HEIGHT_CONTRAST_PCT_LO);
    const hi = percentileSorted(samples, HEIGHT_CONTRAST_PCT_HI);
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return null;
    return { min: lo, max: hi };
  }, [heightLayer?.grid_b64, heightLayer?.grid_shape, densityLayer?.grid_b64, densityLayer?.grid_shape]);
  const obstacleHeightMax = obstacleHeightLayer?.value_max ?? null;
  const distanceMin = distanceLayer?.value_min;
  const distanceMax = distanceLayer?.value_max;
  const distanceMid = distanceMin !== undefined && distanceMax !== undefined ? (distanceMin + distanceMax) / 2 : undefined;
  const spanX = cameraFloorplan?.bounds ? Math.abs((cameraFloorplan.bounds.max_x ?? 0) - (cameraFloorplan.bounds.min_x ?? 0)) : undefined;
  const spanZ = cameraFloorplan?.bounds ? Math.abs((cameraFloorplan.bounds.max_z ?? 0) - (cameraFloorplan.bounds.min_z ?? 0)) : undefined;
  const floorplanStatusText = useMemo(() => {
    if (!open || (activeTab !== 'heatmap' && activeTab !== '3d')) return '';
    if (floorplanStatus === 'loading') return 'Refreshing depth view...';
    if (floorplanStatus === 'checking') return 'Checking cached floorplan...';
    if (floorplanError) return `Floorplan error: ${floorplanError}`;
    if (cameraFloorplan?.served_from_cache) return 'Showing cached floorplan. Press refresh to regenerate.';
    if (cameraFloorplan) return 'Floorplan generated from the latest depth snapshot.';
    return 'Waiting for floorplan data.';
  }, [open, activeTab, floorplanStatus, floorplanError, cameraFloorplan]);
  const clearCanvasElement = useCallback((canvas: HTMLCanvasElement | null) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  useEffect(() => {
    onRequestFloorplanRef.current = onRequestFloorplan;
  }, [onRequestFloorplan]);

  const renderTopdownLayer = useCallback(
    (canvas: HTMLCanvasElement | null, layer: FloorplanLayer | undefined, palette: (t: number) => [number, number, number]) => {
      renderLayerToCanvas(canvas, layer, palette, { fit: 'contain' });
    },
    []
  );

  const requestFloorplan = useCallback((mode: 'cache-only' | 'regenerate') => {
    if (!open || !selectedCamera) return;
    const requestFn = onRequestFloorplanRef.current;
    if (!requestFn) return;
    const cacheOnly = mode === 'cache-only';
    setFloorplanStatus(cacheOnly ? 'checking' : 'loading');
    const req = {
      camera: selectedCamera,
      requestId: Date.now().toString(),
      // Regenerate ignores staleness by passing maxAgeSec=0
      maxAgeSec: cacheOnly ? undefined : 0,
      gridResM: 0.15,
      maxExtentM: 20,
      cacheOnly,
    };
    try { console.debug('[UI] floorplan request', { mode, ...req }); } catch { }
    const requestId = requestFn(req);
    if (typeof requestId === 'string' && requestId.length) {
      setFloorplanRequest(requestId);
    } else {
      setFloorplanRequest('');
      setFloorplanStatus('idle');
    }
  }, [open, selectedCamera]);

  useEffect(() => {
    if (!cameras.length) {
      setSelectedCamera('');
      return;
    }
    if (!selectedCamera || !cameras.includes(selectedCamera)) {
      setSelectedCamera(cameras[0]);
    }
  }, [cameras, selectedCamera]);

  useEffect(() => {
    if (!open || !selectedCamera) return;
    onRequestDepthCached(selectedCamera);
  }, [open, selectedCamera, onRequestDepthCached]);

  useEffect(() => {
    if (open) {
      setDrawerWidth(DEFAULT_WIDTH);
    }
  }, [open]);

  const depthEntry = selectedCamera ? depthData[selectedCamera] : undefined;
  const depthMetaEntry = selectedCamera && depthMeta ? depthMeta[selectedCamera] : undefined;
  const summaryEntry = selectedCamera ? diagnostics[selectedCamera] : undefined;
  const normalsInfo = useMemo(() => {
    if (!depthEntry) {
      return {
        available: false,
        height: 0,
        width: 0,
        dtype: '',
        space: '',
        error: null as string | null,
      };
    }
    const shape = depthEntry.normals_shape;
    let height = depthEntry.shape[0];
    let width = depthEntry.shape[1];
    if (Array.isArray(shape) && shape.length >= 2) {
      height = Number(shape[0]) || height;
      width = Number(shape[1]) || width;
    }
    return {
      available: Boolean(depthEntry.normals_b64),
      height,
      width,
      dtype: depthEntry.normals_dtype || '',
      space: depthEntry.normals_space || '',
      error: depthEntry.normals_error ?? null,
    };
  }, [depthEntry]);
  const normalsStatusText = useMemo(() => {
    if (!depthEntry) return 'Waiting for depth snapshot.';
    if (normalsInfo.error) return `Normals error: ${normalsInfo.error}`;
    if (!normalsInfo.available) return 'Normals not attached to this snapshot.';
    const spaceLabel = normalsInfo.space || 'camera';
    const dtypeLabel = normalsInfo.dtype || 'float16';
    return `Normals attached (${spaceLabel}, ${dtypeLabel}).`;
  }, [depthEntry, normalsInfo]);

  useEffect(() => {
    const canvas = heatmapCanvasRef.current;
    if (!canvas || !depthEntry || activeTab !== 'heatmap') return;

    const [height, width] = depthEntry.shape;
    const depthArray = decodeFloat32(depthEntry.depth_b64);
    if (!depthArray || depthArray.length < width * height) {
      setHeatmapRange(null);
      clearCanvasElement(canvas);
      return;
    }

    const confArray = decodeFloat32(depthEntry.conf_b64);
    const maskArray = decodeUint8(depthEntry.mask_b64);

    let minDepth = Number.POSITIVE_INFINITY;
    let maxDepth = Number.NEGATIVE_INFINITY;
    const total = width * height;
    for (let i = 0; i < total; i += 1) {
      const d = depthArray[i];
      const maskOk = !maskArray || maskArray[i] > 0;
      if (!Number.isFinite(d) || !maskOk) continue;
      if (d < minDepth) minDepth = d;
      if (d > maxDepth) maxDepth = d;
    }

    // Fallback when all valid samples are zero/negative: use the raw finite span
    if (!Number.isFinite(minDepth) || !Number.isFinite(maxDepth)) {
      minDepth = Number.POSITIVE_INFINITY;
      maxDepth = Number.NEGATIVE_INFINITY;
      for (let i = 0; i < total; i += 1) {
        const d = depthArray[i];
        if (!Number.isFinite(d)) continue;
        if (d < minDepth) minDepth = d;
        if (d > maxDepth) maxDepth = d;
      }
    }

    if (!Number.isFinite(minDepth) || !Number.isFinite(maxDepth)) {
      clearCanvasElement(canvas);
      setHeatmapRange(null);
      return;
    }

    let range = maxDepth - minDepth;
    if (range <= 0) {
      range = Math.max(Math.abs(maxDepth) || 1, 1);
      maxDepth = minDepth + range;
    }
    setHeatmapRange({ min: minDepth, max: maxDepth });

    const offscreen = document.createElement('canvas');
    offscreen.width = width;
    offscreen.height = height;
    const offCtx = offscreen.getContext('2d');
    if (!offCtx) return;
    const imageData = offCtx.createImageData(width, height);
    const data = imageData.data;

    for (let i = 0; i < total; i += 1) {
      const d = depthArray[i];
      const idx = i * 4;
      const maskOk = !maskArray || maskArray[i] > 0;
      if (!Number.isFinite(d) || d <= 0 || !maskOk) {
        data[idx + 3] = 0;
        continue;
      }
      const norm = Math.min(1, Math.max(0, (d - minDepth) / range));
      const [r, g, b] = turboColor(norm);
      let alpha = 0.9;
      if (confArray && confArray.length > i) {
        const conf = Math.max(0, Math.min(1, confArray[i]));
        alpha = 0.3 + conf * 0.7;
      }
      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = Math.round(alpha * 255);
    }

    offCtx.putImageData(imageData, 0, 0);

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const rect = canvas.getBoundingClientRect();
    const targetWidth = rect.width || canvas.clientWidth || width;
    const targetHeight = targetWidth / WIDE_ASPECT;  // Force 16:9
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(targetWidth * dpr));
    canvas.height = Math.max(1, Math.round(targetHeight * dpr));

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, targetWidth, targetHeight);
    ctx.imageSmoothingEnabled = true;
    // Stretch to 16:9 (minimal effect since native matches, but consistent)
    ctx.drawImage(offscreen, 0, 0, targetWidth, targetHeight);
    ctx.restore();
  }, [depthEntry, activeTab, drawerWidth, clearCanvasElement]);

  useEffect(() => {
    const canvas = normalsCanvasRef.current;
    if (!canvas || activeTab !== 'normals') return;
    if (!depthEntry || !depthEntry.normals_b64 || !normalsInfo.available) {
      clearCanvasElement(canvas);
      return;
    }
    const height = normalsInfo.height;
    const width = normalsInfo.width;
    if (!height || !width) {
      clearCanvasElement(canvas);
      return;
    }
    const dtype = (normalsInfo.dtype || 'float16').toLowerCase();
    const normalsArray = dtype === 'float16'
      ? decodeFloat16(depthEntry.normals_b64)
      : decodeFloat32(depthEntry.normals_b64);
    if (!normalsArray || normalsArray.length < width * height * 3) {
      clearCanvasElement(canvas);
      return;
    }
    const offscreen = document.createElement('canvas');
    offscreen.width = width;
    offscreen.height = height;
    const offCtx = offscreen.getContext('2d');
    if (!offCtx) return;
    const imageData = offCtx.createImageData(width, height);
    const data = imageData.data;
    const total = width * height;
    for (let i = 0; i < total; i += 1) {
      const base = i * 3;
      const nx = normalsArray[base];
      const ny = normalsArray[base + 1];
      const nz = normalsArray[base + 2];
      const idx = i * 4;
      if (!Number.isFinite(nx) || !Number.isFinite(ny) || !Number.isFinite(nz)) {
        data[idx + 3] = 0;
        continue;
      }
      const mag = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (!Number.isFinite(mag) || mag <= 0.0001) {
        data[idx + 3] = 0;
        continue;
      }
      const r = Math.round((Math.min(1, Math.max(-1, nx)) * 0.5 + 0.5) * 255);
      const g = Math.round((Math.min(1, Math.max(-1, ny)) * 0.5 + 0.5) * 255);
      const b = Math.round((Math.min(1, Math.max(-1, nz)) * 0.5 + 0.5) * 255);
      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = 255;
    }
    offCtx.putImageData(imageData, 0, 0);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const rect = canvas.getBoundingClientRect();
    const targetWidth = rect.width || canvas.clientWidth || width;
    const targetHeight = targetWidth / WIDE_ASPECT;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(targetWidth * dpr));
    canvas.height = Math.max(1, Math.round(targetHeight * dpr));
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, targetWidth, targetHeight);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(offscreen, 0, 0, targetWidth, targetHeight);
    ctx.restore();
  }, [depthEntry, activeTab, normalsInfo, drawerWidth, clearCanvasElement]);

  // Avoid showing stale topdown canvases when switching cameras.
  useEffect(() => {
    if (!open || activeTab !== 'heatmap') return;
    clearCanvasElement(densityCanvasRef.current);
    clearCanvasElement(heightCanvasRef.current);
    clearCanvasElement(distanceCanvasRef.current);
  }, [open, activeTab, selectedCamera, clearCanvasElement]);

  // When switching cameras, try to load the cached floorplan for that camera.
  useEffect(() => {
    if (!open || (activeTab !== 'heatmap' && activeTab !== '3d') || !selectedCamera) return;
    requestFloorplan('cache-only');
  }, [open, activeTab, selectedCamera, requestFloorplan]);

  // Notify backend once per camera when a heatmap already in local state is rendered.
  useEffect(() => {
    if (!open || activeTab !== 'heatmap') return;
    if (!selectedCamera || !depthEntry) return;
    if (heatmapNotifiedRef.current.has(selectedCamera)) return;
    try {
      onHeatmapReady(selectedCamera);
      heatmapNotifiedRef.current.add(selectedCamera);
    } catch { }
  }, [open, activeTab, selectedCamera, depthEntry, onHeatmapReady]);

  useEffect(() => {
    setPrimitivesSelectedBoxId(null);
  }, [selectedCamera]);

  const drawStreamPreview = useCallback(() => {
    const canvas = streamPreviewCanvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const rect = canvas.getBoundingClientRect();
    const targetW = rect.width || canvas.clientWidth || 1;
    const targetH = targetW / WIDE_ASPECT;
    const dpr = window.devicePixelRatio || 1;
    const nextW = Math.max(1, Math.round(targetW * dpr));
    const nextH = Math.max(1, Math.round(targetH * dpr));
    if (canvas.width !== nextW || canvas.height !== nextH) {
      canvas.width = nextW;
      canvas.height = nextH;
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, targetW, targetH);

    const video = videoRef?.current ?? null;
    const layout = mosaicLayout ?? null;
    const cols = layout?.cols ?? null;
    const rows = layout?.rows ?? null;
    const camKey = detectCameraKey(selectedCamera) as CameraKey | null;
    if (!video || !video.videoWidth || !video.videoHeight || !cols || !rows || !camKey) {
      ctx.fillStyle = '#0d1320';
      ctx.fillRect(0, 0, targetW, targetH);
      ctx.fillStyle = 'rgba(255,255,255,0.65)';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Preview unavailable', targetW / 2, targetH / 2);
      ctx.restore();
      return;
    }

    const sourceId = cameraIndex(camKey);
    let tileIndex = sourceId;
    const sources = layout?.sources || [];
    if (sources.length) {
      const idx = sources.findIndex((src) => src.source_id === sourceId);
      if (idx >= 0) tileIndex = idx;
    }
    if (layout?.source_count && tileIndex >= layout.source_count) {
      ctx.restore();
      return;
    }

    const c = tileIndex % cols;
    const r = Math.floor(tileIndex / cols);
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const sx = Math.round((c * vw) / cols);
    const sy = Math.round((r * vh) / rows);
    const sw = Math.round(((c + 1) * vw) / cols) - sx;
    const sh = Math.round(((r + 1) * vh) / rows) - sy;
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, targetW, targetH);
    ctx.restore();
  }, [mosaicLayout, selectedCamera, videoRef]);

  useEffect(() => {
    if (!open || activeTab !== '3d' || !primitivesShowPreview) return;
    let raf = 0;
    const tick = () => {
      drawStreamPreview();
      raf = window.requestAnimationFrame(tick);
    };
    raf = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(raf);
  }, [open, activeTab, primitivesShowPreview, drawStreamPreview]);

  useEffect(() => {
    if (!open || activeTab !== '3d' || primitivesShowPreview) return;
    clearCanvasElement(streamPreviewCanvasRef.current);
  }, [open, activeTab, primitivesShowPreview, clearCanvasElement]);

  const primitivesModel = useMemo(() => {
    if (!cameraFloorplan || cameraFloorplan.error) return null;
    return buildExtrudedFloorplanModel(
      {
        bounds: cameraFloorplan.bounds,
        height: cameraFloorplan.height,
        density: cameraFloorplan.density,
      },
      {
        maxCells: primitivesMaxCells,
        minHeightM: primitivesMinHeightM,
        minDensity: primitivesMinDensity,
        minFootprintM2: primitivesMinFootprintM2,
        maxBoxes: primitivesMaxBoxes,
      }
    );
  }, [cameraFloorplan, primitivesMaxBoxes, primitivesMaxCells, primitivesMinDensity, primitivesMinFootprintM2, primitivesMinHeightM]);

  const obstacleBoxes = primitivesModel?.boxes ?? [];
  const selectedIntrinsics = useMemo(
    () => (selectedCamera ? getIntrinsicsAny(selectedCamera) : null),
    [selectedCamera, calibrationEpoch]
  );
  const selectedExtrinsics = useMemo(
    () => (selectedCamera ? getExtrinsicsAny(selectedCamera) : null),
    [selectedCamera, calibrationEpoch]
  );
  const visibleFloorResult = useMemo(() => buildVisibleFloorPlaneModel({
    cameraId: selectedCamera,
    depthEntry,
    intrinsics: selectedIntrinsics,
    extrinsics: selectedExtrinsics,
    floorplan: cameraFloorplan,
  }), [
    selectedCamera,
    depthEntry,
    selectedIntrinsics,
    selectedExtrinsics,
    cameraFloorplan,
  ]);
  const visibleFloorModel = visibleFloorResult.model;
  const handleFloorPlaneCanvasReady = useCallback((canvas: HTMLCanvasElement | null) => {
    floorPlaneCanvasRef.current = canvas;
  }, []);

  useEffect(() => {
    if (primitivesSelectedBoxId && obstacleBoxes.every((b) => b.id !== primitivesSelectedBoxId)) {
      setPrimitivesSelectedBoxId(null);
    }
  }, [obstacleBoxes, primitivesSelectedBoxId]);

  useEffect(() => {
    if (!open || activeTab !== '3d' || primitivesView !== 'obstacles') return;
    renderExtrudedFloorplanToCanvas(
      extrudedCanvasRef.current,
      primitivesModel,
      {
        forceAspect: EXTRUDED_ASPECT,
        heightExaggeration: primitivesHeightExaggeration,
        minHeightM: primitivesMinHeightM,
        minDensity: primitivesMinDensity,
        showBoxes: primitivesShowBoxes,
        selectedBoxId: primitivesSelectedBoxId,
        palette: infernoColor,
        background: '#0d1320',
      }
    );
  }, [
    open,
    activeTab,
    primitivesModel,
    primitivesHeightExaggeration,
    primitivesMinHeightM,
    primitivesMinDensity,
    primitivesShowBoxes,
    primitivesSelectedBoxId,
    primitivesView,
    drawerWidth,
  ]);

  useEffect(() => {
    if (!depthEntry) {
      setHeatmapRange(null);
    }
  }, [depthEntry]);

  useEffect(() => {
    const canvas = histogramCanvasRef.current;
    if (!canvas || !depthEntry || activeTab !== 'histogram') return;
    const confArray = decodeFloat32(depthEntry.conf_b64);
    if (!confArray) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }
    const bins = new Array(10).fill(0);
    for (let i = 0; i < confArray.length; i += 1) {
      const val = Math.max(0, Math.min(0.999, confArray[i] || 0));
      const bin = Math.floor(val * bins.length);
      bins[bin] += 1;
    }
    const maxBin = Math.max(...bins, 1);
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const barWidth = canvas.width / bins.length;
    bins.forEach((count, idx) => {
      const height = (count / maxBin) * canvas.height;
      ctx.fillStyle = '#58d68d';
      ctx.fillRect(idx * barWidth + 4, canvas.height - height, barWidth - 8, height);
      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      ctx.font = '12px system-ui';
      ctx.fillText(`${idx / 10}-${(idx + 1) / 10}`, idx * barWidth + 6, canvas.height - 6);
    });
  }, [depthEntry, activeTab]);

  const summary = summaryEntry?.summary;
  const metrics = useMemo(() => {
    if (!summary) return [] as Array<{ label: string; type: 'bar' | 'text'; text: string; fraction?: number }>;
    const confidence = summary.conf_mean !== undefined ? Math.max(0, Math.min(1, summary.conf_mean)) : null;
    const coverage = summary.valid_ratio !== undefined ? Math.max(0, Math.min(1, summary.valid_ratio)) : null;
    return [
      {
        label: 'Confidence',
        type: 'bar' as const,
        fraction: confidence ?? 0,
        text: confidence !== null ? `${Math.round(confidence * 100)}%` : 'n/a'
      },
      {
        label: 'Valid Coverage',
        type: 'bar' as const,
        fraction: coverage ?? 0,
        text: coverage !== null ? `${Math.round(coverage * 100)}%` : 'n/a'
      },
      {
        label: 'Median Depth',
        type: 'text' as const,
        text: summary.median !== undefined ? `${summary.median.toFixed(2)} m` : 'n/a'
      },
      {
        label: 'Depth Range',
        type: 'text' as const,
        text: summary.p10 !== undefined && summary.p90 !== undefined ? `${summary.p10.toFixed(2)}–${summary.p90.toFixed(2)} m` : 'n/a'
      },
      {
        label: 'Samples',
        type: 'text' as const,
        text: summary.sample_count !== undefined ? `${summary.sample_count}` : 'n/a'
      },
      {
        label: 'Method',
        type: 'text' as const,
        text: summary.method ? summary.method.toUpperCase() : (confidence !== null && confidence >= 0.5 ? 'MDE' : 'FLOOR')
      }
    ];
  }, [summary]);

  const renderScale = (gradient: string, min?: number, _mid?: number, max?: number, unit = '') => (
    <div className="color-scale">
      <span className="color-scale__label color-scale__label--max">{formatNumber(max)}{unit}</span>
      <div className="color-scale__bar" style={{ background: gradient }} />
      <span className="color-scale__label color-scale__label--min">{formatNumber(min)}{unit}</span>
    </div>
  );

  useEffect(() => {
    if (!cameras.length) {
      if (selectedCamera !== '') {
        setSelectedCamera('');
      }
      return;
    }
    if (!selectedCamera || !cameras.includes(selectedCamera)) {
      setSelectedCamera(cameras[0]);
    }
  }, [cameras, selectedCamera]);

  useEffect(() => {
    return () => {
      floorplanWarmupTimersRef.current.forEach((id) => window.clearTimeout(id));
      floorplanWarmupTimersRef.current = [];
      floorplanPrefetchScheduledRef.current.clear();
      floorplanPrefetchedRef.current.clear();
    };
  }, []);

  useEffect(() => {
    if (!open) {
      floorplanPrefetchedRef.current.clear();
      floorplanPrefetchScheduledRef.current.clear();
      floorplanWarmupTimersRef.current.forEach((id) => window.clearTimeout(id));
      floorplanWarmupTimersRef.current = [];
    }
  }, [open]);

  useEffect(() => {
    if (!open || !cameras.length) return;
    const requestFn = onRequestFloorplanRef.current;
    if (!requestFn) return;

    const spacingMs = 60;
    const ts = Date.now();
    const missing = cameras.filter((cam) => {
      if (!cam) return false;
      if (floorplans[cam]) return false;
      if (floorplanPrefetchedRef.current.has(cam)) return false;
      if (floorplanPrefetchScheduledRef.current.has(cam)) return false;
      return true;
    });

    if (!missing.length) return;

    try { console.debug('[UI] warmup floorplan cache', missing); } catch { }

    const baseDelay = floorplanWarmupTimersRef.current.length * spacingMs;
    missing.forEach((cam, idx) => {
      floorplanPrefetchScheduledRef.current.add(cam);
      const requestId = `drawer-cache-${cam}-${ts}-${idx}`;
      if (cam === selectedCamera) {
        setFloorplanStatus('checking');
        setFloorplanRequest(requestId);
      }
      const timer = window.setTimeout(() => {
        floorplanPrefetchScheduledRef.current.delete(cam);
        floorplanPrefetchedRef.current.add(cam);
        try {
          requestFn({
            camera: cam,
            requestId,
            gridResM: 0.15,
            maxExtentM: 20,
            cacheOnly: true,
          });
        } catch { }
      }, baseDelay + (idx * spacingMs));
      floorplanWarmupTimersRef.current.push(timer);
    });
  }, [open, cameras, floorplans, selectedCamera]);

  useEffect(() => {
    if ((activeTab !== 'heatmap' && activeTab !== '3d') || !open) return;

    if (floorplanRequest && !cameraFloorplan) {
      if (activeTab === 'heatmap') {
        clearCanvasElement(densityCanvasRef.current);
        clearCanvasElement(heightCanvasRef.current);
        clearCanvasElement(heightContrastCanvasRef.current);
        clearCanvasElement(heightAglCanvasRef.current);
        clearCanvasElement(distanceCanvasRef.current);
        clearCanvasElement(gradientCanvasRef.current);
        clearCanvasElement(obstacleHeightCanvasRef.current);
        clearCanvasElement(walkableCanvasRef.current);
        clearCanvasElement(floorplanCompositeCanvasRef.current);
      }
      return;
    }

    if (floorplanError) {
      if (activeTab === 'heatmap') {
        clearCanvasElement(densityCanvasRef.current);
        clearCanvasElement(heightCanvasRef.current);
        clearCanvasElement(heightContrastCanvasRef.current);
        clearCanvasElement(heightAglCanvasRef.current);
        clearCanvasElement(distanceCanvasRef.current);
        clearCanvasElement(gradientCanvasRef.current);
        clearCanvasElement(obstacleHeightCanvasRef.current);
        clearCanvasElement(walkableCanvasRef.current);
        clearCanvasElement(floorplanCompositeCanvasRef.current);
      }
      setFloorplanStatus('idle');
      setFloorplanRequest('');
      return;
    }

    if (cameraFloorplan) {
      if (activeTab === 'heatmap') {
        renderTopdownLayer(densityCanvasRef.current, densityLayer, grayscaleColor);
        renderTopdownLayer(heightCanvasRef.current, heightLayer, infernoColor);
        // A display-only contrast view: clamp to a small height range and hide unobserved cells.
        const contrastMin = heightContrastRange?.min ?? 0;
        const contrastMax = heightContrastRange?.max ?? (heightMaxRaw ?? 1);
        renderLayerToCanvas(heightContrastCanvasRef.current, heightLayer, turboColor, {
          fit: 'contain',
          valueMin: contrastMin,
          valueMax: contrastMax,
          gamma: HEIGHT_CONTRAST_GAMMA,
          maskLayer: densityLayer,
          maskThreshold: HEIGHT_CONTRAST_DENSITY_THRESH,
        });
        // Height above estimated floor (AGL): fixed range for clarity.
        renderLayerToCanvas(heightAglCanvasRef.current, heightAglLayer, turboColor, {
          fit: 'contain',
          valueMin: HEIGHT_AGL_VIEW_MIN_M,
          valueMax: HEIGHT_AGL_VIEW_MAX_M,
          gamma: 1.0,
          maskLayer: densityLayer,
          maskThreshold: HEIGHT_CONTRAST_DENSITY_THRESH,
        });
        renderTopdownLayer(distanceCanvasRef.current, distanceLayer, viridisColor);
        renderTopdownLayer(obstacleHeightCanvasRef.current, obstacleHeightLayer, infernoColor);
        renderTopdownLayer(walkableCanvasRef.current, walkableLayer, bwColor);
        renderCompositeWalkableObstacleToCanvas(
          floorplanCompositeCanvasRef.current,
          walkableLayer,
          obstacleHeightLayer,
          { fit: 'contain' }
        );
        renderTopdownLayer(gradientCanvasRef.current, gradientLayer, viridisColor);
      }

      if (!floorplanRequest || !cameraFloorplan.request_id || cameraFloorplan.request_id === floorplanRequest) {
        setFloorplanStatus('idle');
        setFloorplanRequest('');
      }
    }
  }, [activeTab, open, cameraFloorplan, densityLayer, heightLayer, heightAglLayer, heightContrastRange, heightMaxRaw, distanceLayer, gradientLayer, obstacleHeightLayer, walkableLayer, renderTopdownLayer, clearCanvasElement, floorplanRequest, drawerWidth, floorplanError]);

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

  const handleResizePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 && event.pointerType === 'mouse') return;
    event.preventDefault();
    event.stopPropagation();
    if (!open) return;
    isResizingRef.current = true;
    activePointerIdRef.current = event.pointerId;
    previousUserSelectRef.current = document.body.style.userSelect;
    document.body.style.userSelect = 'none';
  };

  const saveBlob = useCallback((blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, []);

  const saveCanvas = useCallback((canvas: HTMLCanvasElement | null, filename: string) => {
    if (!canvas || canvas.width <= 0 || canvas.height <= 0) return;
    try {
      canvas.toBlob((blob) => {
        if (!blob) return;
        saveBlob(blob, filename);
      }, 'image/png');
    } catch (err) {
      console.warn('Unable to save depth drawer canvas', filename, err);
    }
  }, [saveBlob]);

  const handleSaveCurrentImages = useCallback(() => {
    if (!selectedCamera) return;
    const safePart = (value: string) => (value || 'unknown')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '') || 'unknown';
    const stamp = new Date().toISOString()
      .replace(/\.\d{3}Z$/, 'Z')
      .replace(/[:.]/g, '')
      .replace('T', '_')
      .replace('Z', '');
    const prefix = `${stamp}_${safePart(selectedCamera)}_${activeTab}`;
    const canvases: Array<{ label: string; canvas: HTMLCanvasElement | null }> = [];
    if (activeTab === 'heatmap') {
      canvases.push(
        { label: 'camera-depth-turbo', canvas: heatmapCanvasRef.current },
        { label: 'density-gray', canvas: densityCanvasRef.current },
        { label: 'height-inferno', canvas: heightCanvasRef.current },
        { label: 'height-contrast', canvas: heightContrastCanvasRef.current },
        { label: 'height-agl-turbo', canvas: heightAglCanvasRef.current },
        { label: 'distance-viridis', canvas: distanceCanvasRef.current },
        { label: 'obstacle-height-clean', canvas: obstacleHeightCanvasRef.current },
        { label: 'walkable-binary', canvas: walkableCanvasRef.current },
        { label: 'floorplan-composite', canvas: floorplanCompositeCanvasRef.current },
        { label: 'gradient-edges', canvas: gradientCanvasRef.current },
      );
    } else if (activeTab === 'normals') {
      canvases.push({ label: 'normals-rgb', canvas: normalsCanvasRef.current });
    } else if (activeTab === '3d') {
      canvases.push({ label: 'stream-preview', canvas: streamPreviewCanvasRef.current });
      if (primitivesView === 'visible-floor') {
        canvases.push({ label: 'visible-floor-plane', canvas: floorPlaneCanvasRef.current });
      } else {
        canvases.push({ label: 'extruded-floorplan', canvas: extrudedCanvasRef.current });
      }
    } else if (activeTab === 'histogram') {
      canvases.push({ label: 'histogram', canvas: histogramCanvasRef.current });
    }

    const imageNames: string[] = [];
    canvases.forEach(({ label, canvas }) => {
      if (!canvas || canvas.width <= 0 || canvas.height <= 0) return;
      const filename = `${prefix}_${safePart(label)}.png`;
      imageNames.push(filename);
      saveCanvas(canvas, filename);
    });

    if (!imageNames.length) return;
    const manifest = {
      saved_at: new Date().toISOString(),
      camera: selectedCamera,
      tab: activeTab,
      images: imageNames,
      depth_ts_us: depthEntry?.ts ?? null,
      depth_served_from_cache: depthMetaEntry?.servedFromCache ?? null,
      depth_source_camera_id: depthMetaEntry?.sourceCameraId ?? null,
      floorplan_ts_us: cameraFloorplan?.ts ?? null,
      floorplan_snapshot_ts_us: cameraFloorplan?.snapshot_ts ?? null,
      floorplan_served_from_cache: cameraFloorplan?.served_from_cache ?? null,
      floor_plane_status: activeTab === '3d' ? visibleFloorResult.status : undefined,
    };
    saveBlob(
      new Blob([JSON.stringify(manifest, null, 2)], { type: 'application/json' }),
      `${prefix}_manifest.json`,
    );
  }, [activeTab, cameraFloorplan, depthEntry, depthMetaEntry, primitivesView, saveBlob, saveCanvas, selectedCamera, visibleFloorResult.status]);

  return (
    <>
      <div className={`depth-drawer-overlay ${open ? 'open' : ''}`} onClick={onClose} />
      <aside
        className={`depth-drawer ${open ? 'open' : ''}`}
        ref={drawerRef}
        style={{ width: drawerWidth }}
      >
        <div
          className="depth-drawer-resize-handle"
          onPointerDown={handleResizePointerDown}
          role="presentation"
        />
        <header>
          <h2>MapAnything Depth</h2>
          <button onClick={onClose} aria-label="Close depth drawer">×</button>
        </header>
        <div className="tabs">
          <button className={activeTab === 'heatmap' ? 'active' : ''} onClick={() => setActiveTab('heatmap')}>Heatmap</button>
          <button className={activeTab === 'normals' ? 'active' : ''} onClick={() => setActiveTab('normals')}>Normals</button>
          <button className={activeTab === '3d' ? 'active' : ''} onClick={() => setActiveTab('3d')}>3D</button>
          <button className={activeTab === 'stats' ? 'active' : ''} onClick={() => setActiveTab('stats')}>Stats</button>
          <button className={activeTab === 'histogram' ? 'active' : ''} onClick={() => setActiveTab('histogram')}>Histogram</button>
          <button className={activeTab === 'metrics' ? 'active' : ''} onClick={() => setActiveTab('metrics')}>Metrics</button>
        </div>
        <div className="content">
          {!cameras.length && <p>No MapAnything diagnostics received yet.</p>}
          {cameras.length > 0 && (
            <div className={`drawer-toolbar ${(activeTab === 'heatmap' || activeTab === '3d' || activeTab === 'normals') ? 'drawer-toolbar--heatmap' : ''}`}>
              <label className="drawer-toolbar__camera" htmlFor="ma-depth-select">
                <span className="drawer-toolbar__label">Camera</span>
                <select
                  id="ma-depth-select"
                  value={selectedCamera}
                  onChange={(ev) => setSelectedCamera(ev.target.value)}
                >
                  {cameras.map((cam) => {
                    const key = detectCameraKey(cam);
                    const label = key ? cameraLabel(key) : cam;
                    return <option key={cam} value={cam}>{label}</option>;
                  })}
                </select>
              </label>
              {(activeTab === 'heatmap' || activeTab === '3d' || activeTab === 'normals') && (
                <div className="drawer-toolbar__actions">
                  <button
                    type="button"
                    className="btn-icon"
                    onClick={handleSaveCurrentImages}
                    disabled={!selectedCamera}
                    aria-label="Save current depth images"
                    title="Save current depth images"
                  >
                    <svg viewBox="0 0 16 16" aria-hidden="true">
                      <path
                        fill="currentColor"
                        d="M8.75 2.75a.75.75 0 0 0-1.5 0v5.69L5.53 6.72a.75.75 0 0 0-1.06 1.06l3 3a.75.75 0 0 0 1.06 0l3-3a.75.75 0 0 0-1.06-1.06L8.75 8.44V2.75ZM3 11.5a.75.75 0 0 1 .75.75v.75h8.5v-.75a.75.75 0 0 1 1.5 0v1.5a.75.75 0 0 1-.75.75H3a.75.75 0 0 1-.75-.75v-1.5A.75.75 0 0 1 3 11.5Z"
                      />
                    </svg>
                  </button>
                  <button
                    type="button"
                    className="btn-icon"
                    onClick={() => {
                      if (!selectedCamera) return;
                      try { console.debug('[UI] refresh depth', selectedCamera); } catch { }
                      onRequestDepthFresh(selectedCamera);
                      if (activeTab !== 'normals') {
                        requestFloorplan('cache-only');
                      }
                    }}
                    disabled={floorplanStatus === 'loading' && activeTab !== 'normals'}
                    aria-label="Refresh depth frame"
                    title={(floorplanStatus === 'loading' && activeTab !== 'normals') ? 'Refreshing depth view...' : undefined}
                  >
                    <svg viewBox="0 0 16 16" aria-hidden="true">
                      <path
                        fill="currentColor"
                        d="M8 2a5.5 5.5 0 0 1 3.804 9.49l1.068 1.068a.75.75 0 1 1-1.06 1.06l-2.5-2.5a.75.75 0 0 1 0-1.06l2.5-2.5a.75.75 0 1 1 1.06 1.06L11.66 9.19A4 4 0 1 0 8 12.5a.75.75 0 1 1 0 1.5A5.5 5.5 0 1 1 8 2Z"
                      />
                    </svg>
                  </button>
                </div>
              )}
            </div>
          )}

          {activeTab === 'heatmap' && (
            <>
              {floorplanError && <p className="floorplan-error">Error: {floorplanError}</p>}
              {floorplanStatusText && !floorplanError && (
                <p className="floorplan-status">{floorplanStatusText}</p>
              )}
              <div className="heatmap-grid">
                <div className="heatmap-cell heatmap-cell--left">
                  <div className="heatmap-cell__scale">
                    {renderScale(turboGradient, heatmapRange?.min ?? undefined, heatmapRange ? (heatmapRange.min + heatmapRange.max) / 2 : undefined, heatmapRange?.max ?? undefined, ' m')}
                  </div>
                  <div className="heatmap-cell__body">
                    <div className="heatmap-cell__title">Camera Heatmap (Turbo)</div>
                    <canvas
                      ref={heatmapCanvasRef}
                      className="heatmap-canvas"
                      style={{ aspectRatio: `${WIDE_ASPECT}` }}
                    />
                  </div>
                </div>
                <div className="heatmap-cell heatmap-cell--left">
                  <div className="heatmap-cell__body">
                    <div className="heatmap-cell__title">Density (Grayscale)</div>
                    <canvas ref={densityCanvasRef} className="heatmap-canvas" style={{ aspectRatio: `${WIDE_ASPECT}` }} />
                  </div>
                  <div className="heatmap-cell__scale">
                    {renderScale(densityGradient, 0, 0.5, 1)}
                  </div>
                </div>
                <div className="heatmap-cell heatmap-cell--right">
                  <div className="heatmap-cell__scale">
                    {renderScale(
                      infernoGradient,
                      heightSpan !== null ? 0 : undefined,
                      heightSpan !== null ? Math.max(0, heightSpan / 2) : undefined,
                      heightSpan !== null ? heightSpan : undefined,
                      ' m'
                    )}
                  </div>
                  <div className="heatmap-cell__body">
                    <div className="heatmap-cell__title">Height (Inferno)</div>
                    <canvas ref={heightCanvasRef} className="heatmap-canvas" style={{ aspectRatio: `${WIDE_ASPECT}` }} />
                  </div>
                </div>
                <div className="heatmap-cell heatmap-cell--right">
                  <div className="heatmap-cell__body">
                    <div className="heatmap-cell__title">Height (Contrast)</div>
                    <canvas ref={heightContrastCanvasRef} className="heatmap-canvas" style={{ aspectRatio: `${WIDE_ASPECT}` }} />
                  </div>
                  <div className="heatmap-cell__scale">
                    {renderScale(
                      turboGradient,
                      heightContrastRange ? heightContrastRange.min : undefined,
                      heightContrastRange ? (heightContrastRange.min + heightContrastRange.max) / 2 : undefined,
                      heightContrastRange ? heightContrastRange.max : undefined,
                      ' m'
                    )}
                  </div>
                </div>
                <div className="heatmap-cell heatmap-cell--left">
                  <div className="heatmap-cell__scale">
                    {renderScale(turboGradient, HEIGHT_AGL_VIEW_MIN_M, (HEIGHT_AGL_VIEW_MIN_M + HEIGHT_AGL_VIEW_MAX_M) / 2, HEIGHT_AGL_VIEW_MAX_M, ' m')}
                  </div>
                  <div className="heatmap-cell__body">
                    <div className="heatmap-cell__title">Height (AGL Turbo)</div>
                    <canvas ref={heightAglCanvasRef} className="heatmap-canvas" style={{ aspectRatio: `${WIDE_ASPECT}` }} />
                  </div>
                </div>
                <div className="heatmap-cell heatmap-cell--right">
                  <div className="heatmap-cell__scale">
                    {renderScale(viridisGradient, distanceMin ?? undefined, distanceMid ?? undefined, distanceMax ?? undefined, ' m')}
                  </div>
                  <div className="heatmap-cell__body">
                    <div className="heatmap-cell__title">Distance (Viridis)</div>
                    <canvas ref={distanceCanvasRef} className="heatmap-canvas" style={{ aspectRatio: `${WIDE_ASPECT}` }} />
                  </div>
                </div>
                <div className="heatmap-cell heatmap-cell--left">
                  <div className="heatmap-cell__scale">
                    {renderScale(
                      infernoGradient,
                      0,
                      obstacleHeightMax !== null ? Math.max(0, obstacleHeightMax / 2) : undefined,
                      obstacleHeightMax !== null ? obstacleHeightMax : undefined,
                      ' m'
                    )}
                  </div>
                  <div className="heatmap-cell__body">
                    <div className="heatmap-cell__title">Obstacle Height (Clean)</div>
                    <canvas ref={obstacleHeightCanvasRef} className="heatmap-canvas" style={{ aspectRatio: `${WIDE_ASPECT}` }} />
                  </div>
                </div>
                <div className="heatmap-cell heatmap-cell--right">
                  <div className="heatmap-cell__body">
                    <div className="heatmap-cell__title">Walkable (Binary)</div>
                    <canvas ref={walkableCanvasRef} className="heatmap-canvas" style={{ aspectRatio: `${WIDE_ASPECT}` }} />
                  </div>
                  <div className="heatmap-cell__scale">
                    {renderScale(densityGradient, 0, 0.5, 1)}
                  </div>
                </div>
                <div className="heatmap-cell heatmap-cell--left">
                  <div className="heatmap-cell__body">
                    <div className="heatmap-cell__title">Floorplan (Composite)</div>
                    <canvas ref={floorplanCompositeCanvasRef} className="heatmap-canvas" style={{ aspectRatio: `${WIDE_ASPECT}` }} />
                  </div>
                </div>
                <div className="heatmap-cell heatmap-cell--right">
                  <div className="heatmap-cell__body">
                    <div className="heatmap-cell__title">Gradient (Edges)</div>
                    <canvas ref={gradientCanvasRef} className="heatmap-canvas" style={{ aspectRatio: `${WIDE_ASPECT}` }} />
                  </div>
                  <div className="heatmap-cell__scale">
                    {renderScale(viridisGradient, 0, 0.5, 1)}
                  </div>
                </div>
              </div>

              {/* Meta information below the grid */}
              <div className="heatmap-meta">
                <div className="heatmap-meta__item">
                  {depthEntry ? (() => {
                    const parts = [`Updated ${new Date(depthEntry.ts / 1000).toLocaleTimeString()}`];
                    if (depthMetaEntry && typeof depthMetaEntry.servedFromCache === 'boolean') {
                      parts.push(depthMetaEntry.servedFromCache ? 'from cache' : 'fresh');
                    }
                    parts.push(`Resolution ${depthEntry.shape[1]}×${depthEntry.shape[0]}`);
                    return parts.join(' · ');
                  })() : 'No depth frame cached yet for this camera.'}
                </div>
                <div className="heatmap-meta__item">
                  {hasDensity ? 'Normalized point density' : 'Waiting for density data'}
                </div>
                <div className="heatmap-meta__item">
                  {hasHeight ? 'Highest surface per cell.' : 'Waiting for height data'}
                </div>
                <div className="heatmap-meta__item">
                  {hasHeightAgl ? 'Height above estimated floor (AGL).' : 'Waiting for AGL height'}
                </div>
                <div className="heatmap-meta__item">
                  {hasDistance ? 'Average distance from camera.' : 'Waiting for distance data'}
                </div>
                <div className="heatmap-meta__item">
                  {hasGradient ? 'Height gradient magnitude (edges).' : 'Waiting for gradient layer'}
                </div>
                <div className="heatmap-meta__item">
                  {hasObstacleHeight ? 'Clean obstacle height above floor.' : 'Waiting for clean obstacle layer'}
                </div>
                <div className="heatmap-meta__item">
                  {hasWalkable ? 'Walkable mask: white=floor, black=obstacle.' : 'Waiting for walkable mask'}
                </div>
              </div>

              {cameraFloorplan && !floorplanError && (
                <p className="floorplan-meta">
                  Points: {cameraFloorplan.point_count ?? 0}
                  {spanX !== undefined && spanZ !== undefined ? ` · Span ${formatNumber(spanX)}m × ${formatNumber(spanZ)}m` : ''}
                  {cameraFloorplan.ts
                    ? ` · Updated ${new Date(cameraFloorplan.ts / 1000).toLocaleTimeString()}${floorplanServedFromCache ? ' (cached)' : ''}`
                    : ''}
                </p>
              )}
            </>
          )}

          {activeTab === 'normals' && (
            <>
              {normalsInfo.error && <p className="floorplan-error">Normals error: {normalsInfo.error}</p>}
              {normalsStatusText && !normalsInfo.error && (
                <p className="floorplan-status">{normalsStatusText}</p>
              )}
              <div className="heatmap-single">
                <div className="heatmap-cell__body">
                  <div className="heatmap-cell__title">Normals (RGB)</div>
                  <canvas
                    ref={normalsCanvasRef}
                    className="heatmap-canvas"
                    style={{ aspectRatio: `${WIDE_ASPECT}` }}
                  />
                </div>
              </div>
              <p className="floorplan-meta">RGB encodes X/Y/Z normals mapped from −1..1.</p>
            </>
          )}

          {activeTab === '3d' && (
            <>
              {floorplanError && <p className="floorplan-error">Error: {floorplanError}</p>}
              {floorplanStatusText && !floorplanError && (
                <p className="floorplan-status">{floorplanStatusText}</p>
              )}

              <div className="primitives-view-toggle" role="tablist" aria-label="3D view">
                <button
                  type="button"
                  className={primitivesView === 'obstacles' ? 'active' : ''}
                  onClick={() => setPrimitivesView('obstacles')}
                >
                  Obstacles
                </button>
                <button
                  type="button"
                  className={primitivesView === 'visible-floor' ? 'active' : ''}
                  onClick={() => setPrimitivesView('visible-floor')}
                >
                  Visible floor
                </button>
              </div>

              <div className="primitives-layout">
                <div className="primitives-panel">
                  <div className="primitives-panel__title">Camera Tile (ROI-style)</div>
                  <canvas
                    ref={streamPreviewCanvasRef}
                    className="primitives-canvas"
                    style={{ aspectRatio: `${WIDE_ASPECT}` }}
                  />
                  <label className="primitives-toggle">
                    <input
                      type="checkbox"
                      checked={primitivesShowPreview}
                      onChange={(e) => setPrimitivesShowPreview(e.target.checked)}
                    />
                    <span>Live preview</span>
                  </label>
                  <div className="primitives-hint">
                    Cropped from the mosaic, like the ROI editor.
                  </div>
                </div>

                {primitivesView === 'obstacles' ? (
                  <div className="primitives-panel">
                    <div className="primitives-panel__title">Extruded Floorplan Primitives</div>
                    <canvas
                      ref={extrudedCanvasRef}
                      className="primitives-canvas primitives-canvas--extruded"
                      style={{ aspectRatio: `${EXTRUDED_ASPECT}` }}
                    />
                    <div className="primitives-submeta">
                      {primitivesModel
                        ? `Grid ${primitivesModel.cols}×${primitivesModel.rows} · max height ${formatNumber(primitivesModel.maxHeightM)} m`
                        : 'Waiting for floorplan grids…'}
                    </div>
                  </div>
                ) : (
                  <div className="primitives-panel">
                    <div className="primitives-panel__title">Visible Floor Plane</div>
                    <div className="floor-plane-view-wrap">
                      <FloorPlane3DView
                        model={visibleFloorModel}
                        onCanvasReady={handleFloorPlaneCanvasReady}
                      />
                      {!visibleFloorModel && (
                        <div className="floor-plane-empty">{visibleFloorResult.message}</div>
                      )}
                    </div>
                    <div className="primitives-submeta">
                      {visibleFloorModel
                        ? `Support ${visibleFloorModel.metrics.estimatedVisibleFloorPixelCount} px · footprint ${visibleFloorModel.footprintMesh.cellCount} cells · ${formatNumber(visibleFloorModel.footprintMesh.areaM2, 1)} m²`
                        : visibleFloorResult.message}
                    </div>
                  </div>
                )}
              </div>

              {primitivesView === 'obstacles' && (
                <>
                  <details className="primitives-details">
                    <summary>Obstacle boxes ({obstacleBoxes.length})</summary>
                    <div className="primitives-boxes">
                      <label className="primitives-toggle primitives-toggle--inline">
                        <input
                          type="checkbox"
                          checked={primitivesShowBoxes}
                          onChange={(e) => setPrimitivesShowBoxes(e.target.checked)}
                        />
                        <span>Draw boxes</span>
                      </label>
                      {obstacleBoxes.length === 0 ? (
                        <div className="primitives-empty">No obstacle clusters detected at current thresholds.</div>
                      ) : (
                        <div className="primitives-box-list">
                          {obstacleBoxes.map((box, idx) => (
                            <button
                              key={box.id}
                              type="button"
                              className={primitivesSelectedBoxId === box.id ? 'primitives-box active' : 'primitives-box'}
                              onClick={() => setPrimitivesSelectedBoxId((prev) => (prev === box.id ? null : box.id))}
                            >
                              <span className="primitives-box__idx">#{idx + 1}</span>
                              <span className="primitives-box__dims">
                                {formatNumber(box.widthM, 2)}×{formatNumber(box.depthM, 2)} m
                              </span>
                              <span className="primitives-box__h">h={formatNumber(box.heightM, 2)} m</span>
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  </details>

                  <details className="primitives-details">
                    <summary>3D settings (noise filter)</summary>
                    <div className="primitives-settings">
                      <div className="primitives-row">
                        <label>Min height</label>
                        <input
                          type="range"
                          min={0.05}
                          max={1.2}
                          step={0.05}
                          value={primitivesMinHeightM}
                          onChange={(e) => setPrimitivesMinHeightM(parseFloat(e.target.value))}
                        />
                        <span className="primitives-value">{formatNumber(primitivesMinHeightM, 2)} m</span>
                      </div>
                      <div className="primitives-row">
                        <label>Min footprint</label>
                        <input
                          type="range"
                          min={0.05}
                          max={2.0}
                          step={0.05}
                          value={primitivesMinFootprintM2}
                          onChange={(e) => setPrimitivesMinFootprintM2(parseFloat(e.target.value))}
                        />
                        <span className="primitives-value">{formatNumber(primitivesMinFootprintM2, 2)} m²</span>
                      </div>
                      <div className="primitives-row">
                        <label>Min density</label>
                        <input
                          type="range"
                          min={0}
                          max={0.3}
                          step={0.01}
                          value={primitivesMinDensity}
                          onChange={(e) => setPrimitivesMinDensity(parseFloat(e.target.value))}
                        />
                        <span className="primitives-value">{formatNumber(primitivesMinDensity, 2)}</span>
                      </div>
                      <div className="primitives-row">
                        <label>Height exaggeration</label>
                        <input
                          type="range"
                          min={0.6}
                          max={3.0}
                          step={0.1}
                          value={primitivesHeightExaggeration}
                          onChange={(e) => setPrimitivesHeightExaggeration(parseFloat(e.target.value))}
                        />
                        <span className="primitives-value">{formatNumber(primitivesHeightExaggeration, 1)}×</span>
                      </div>
                      <div className="primitives-row">
                        <label>Quality</label>
                        <input
                          type="range"
                          min={60}
                          max={220}
                          step={10}
                          value={primitivesMaxCells}
                          onChange={(e) => setPrimitivesMaxCells(parseInt(e.target.value, 10))}
                        />
                        <span className="primitives-value">{primitivesMaxCells} max</span>
                      </div>
                      <div className="primitives-row">
                        <label>Max boxes</label>
                        <input
                          type="range"
                          min={3}
                          max={24}
                          step={1}
                          value={primitivesMaxBoxes}
                          onChange={(e) => setPrimitivesMaxBoxes(parseInt(e.target.value, 10))}
                        />
                        <span className="primitives-value">{primitivesMaxBoxes}</span>
                      </div>
                    </div>
                  </details>
                </>
              )}

              {primitivesView === 'visible-floor' && visibleFloorModel && (
                <details className="primitives-details">
                  <summary>Floor plane fit</summary>
                  <div className="floor-plane-metrics">
                    <div><span>Frame</span><strong>{visibleFloorModel.frame}</strong></div>
                    <div><span>World height</span><strong>{formatNumber(visibleFloorModel.plane.worldHeightM, 3)} m</strong></div>
                    <div><span>Area</span><strong>{formatNumber(visibleFloorModel.metrics.planeAreaM2, 1)} m²</strong></div>
                    <div><span>Source</span><strong>{visibleFloorModel.footprint.source.replace(/_/g, ' ')}</strong></div>
                    <div><span>Cells</span><strong>{visibleFloorModel.footprintMesh.cellCount}</strong></div>
                    <div><span>Inliers</span><strong>{visibleFloorModel.metrics.visibleFloorPixelCount}</strong></div>
                    <div><span>Confidence</span><strong>{formatNumber(visibleFloorModel.metrics.meanConfidence, 2)}</strong></div>
                    <div><span>Normal dot</span><strong>{formatNumber(visibleFloorModel.metrics.meanHorizontalDot, 2)}</strong></div>
                  </div>
                </details>
              )}
            </>
          )}

          {activeTab === 'stats' && summary && (
            <div className="stat-grid">
              <div className="stat-card">
                <h4>Median Depth</h4>
                <strong>{summary.median !== undefined ? summary.median.toFixed(2) + ' m' : 'n/a'}</strong>
              </div>
              <div className="stat-card">
                <h4>10–90% Range</h4>
                <strong>
                  {summary.p10 !== undefined && summary.p90 !== undefined
                    ? `${summary.p10.toFixed(2)}–${summary.p90.toFixed(2)} m`
                    : 'n/a'}
                </strong>
              </div>
              <div className="stat-card">
                <h4>Confidence</h4>
                <strong>{summary.conf_mean !== undefined ? `${Math.round(summary.conf_mean * 100)}%` : 'n/a'}</strong>
              </div>
              <div className="stat-card">
                <h4>Valid Coverage</h4>
                <strong>{summary.valid_ratio !== undefined ? `${Math.round(summary.valid_ratio * 100)}%` : 'n/a'}</strong>
              </div>
              <div className="stat-card">
                <h4>Samples</h4>
                <strong>{summary.sample_count !== undefined ? summary.sample_count : 'n/a'}</strong>
              </div>
              <div className="stat-card">
                <h4>Method</h4>
                <strong>{summary.method ? summary.method.toUpperCase() : 'AUTO'}</strong>
              </div>
            </div>
          )}

          {activeTab === 'histogram' && (
            <canvas ref={histogramCanvasRef} className="histogram" />
          )}

          {activeTab === 'metrics' && (
            <ul className="metrics-list">
              {metrics.map(item => (
                <li key={item.label}>
                  <strong>{item.label}</strong>
                  {item.type === 'bar' ? (
                    <>
                      <div className="metric-bar">
                        <div className="metric-bar-fill" style={{ width: `${Math.round((item.fraction || 0) * 100)}%` }} />
                      </div>
                      <span className="metric-bar-label">{item.text}</span>
                    </>
                  ) : (
                    <div>{item.text}</div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>
    </>
  );
});

export default DepthDrawer;
export type DepthDrawerEntry = DepthEntry;
export type DepthDiagnosticsEntry = DiagnosticsEntry;
