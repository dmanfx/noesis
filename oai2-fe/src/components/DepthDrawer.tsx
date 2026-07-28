import { PointerEvent as ReactPointerEvent, memo, useCallback, useEffect, useMemo, useRef, useState, RefObject } from 'react';
import { CameraKey, cameraIndex, cameraLabel, detectCameraKey } from '../lib/camera';
import type { MosaicLayout } from '../hooks/useWebSocketClient';
import '../styles/depth-drawer.css';

type DepthEntry = {
  ts: number;
  depth: Float32Array;
  conf: Float32Array;
  mask: Uint8Array;
  rgb?: Uint8Array;
  rgb_shape?: [number, number, number];
  normals?: Float32Array;
  surface_normals?: Int8Array;
  shape: [number, number];
  normals_shape?: [number, number, number];
  surface_normals_shape?: [number, number, number];
  normals_dtype?: 'float32';
  normals_space?: 'camera' | 'world';
  normals_error?: string;
  snapshotId?: string;
  snapshotRef?: string;
  snapshotContentSha256?: string;
};

export type DepthMetaEntry = {
  tsUs?: number;
  servedFromCache?: boolean;
  requestId?: string;
  error?: string;
  snapshotRef?: string;
  snapshotId?: string;
  snapshotContentSha256?: string;
  rgbComponentSha256?: string;
  // Raw camera id as reported by backend (may differ from UI key if remapped)
  sourceCameraId?: string;
  transferBytes?: number;
  transferDurationMs?: number;
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

export type FloorplanRgbLayer = {
  rgb_b64?: string;
  rgb_shape?: [number, number, number];
  observed_b64?: string;
};

export type FloorplanResponse = {
  type?: string;
  request_id?: string;
  camera_id?: string;
  snapshot_ref?: string;
  snapshot_id?: string;
  snapshot_content_sha256?: string;
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
  observed?: FloorplanLayer;
  unknown?: FloorplanLayer;
  inferred_walkable?: FloorplanLayer;
  structural_height?: FloorplanLayer;
  surface_observed?: FloorplanLayer;
  room_footprint?: FloorplanLayer;
  wall_support?: FloorplanLayer;
  room_boundary?: FloorplanLayer;
  surface_rgb?: FloorplanRgbLayer;
  observation_meta?: {
    observed_cells?: number;
    unknown_cells?: number;
    total_cells?: number;
    [key: string]: unknown;
  };
  height_agl_meta?: { floor_y?: number; floor_estimate?: unknown };
};

export type DepthRefreshEntry = {
  phase: 'requesting-depth' | 'requesting-floorplan' | 'error';
  error?: string;
};

export type DepthCachePairEntry = {
  phase: 'requesting-depth' | 'waiting-depth' | 'waiting-floorplan' | 'error';
  error?: string;
};

interface DepthDrawerProps {
  open: boolean;
  onClose: () => void;
  diagnostics: Record<string, DiagnosticsEntry>;
  depthData: Record<string, DepthEntry>;
  depthMeta?: Record<string, DepthMetaEntry>;
  onRequestDepthFresh: (cameraId: string) => void;
  onRequestDepthCached: (cameraId: string) => void;
  floorplans: Record<string, FloorplanResponse>;
  refreshState?: Record<string, DepthRefreshEntry>;
  cachePairState?: Record<string, DepthCachePairEntry>;
  availableCameras: string[];
  mosaicLayout?: MosaicLayout | null;
  videoRef?: RefObject<HTMLVideoElement>;
  calibrationEpoch?: number;
}

import {
  decodeFloat32,
  grayscaleColor,
  bwColor,
  infernoColor,
  renderLayerToCanvas,
  renderStructuralFloorplanToCanvas,
  turboColor,
  viridisColor
} from '../lib/renderUtils';
import {
  chooseDepthRange,
  computeMaskedRange,
  floorplanObservationCounts,
  integerExportScale,
  snapshotExportStem,
  type SimpleRange,
} from '../lib/depthQuality.js';
import { computeObservedFloorplanViewport } from '../lib/floorplanViewport.js';
import { buildExtrudedFloorplanModel, DEFAULT_OBSTACLE_SETTINGS, renderExtrudedFloorplanToCanvas } from '../lib/extrudedFloorplan';
import { getExtrinsicsAny, getIntrinsicsAny } from '../lib/calibration';
import { buildVisibleFloorPlaneModel } from '../lib/visibleFloorPlane';
import FloorPlane3DView from './FloorPlane3DView';
import CachedHeightfield3DView from './CachedHeightfield3DView';
import CalibratedPointCloud3DView from './CalibratedPointCloud3DView';

const DEFAULT_WIDTH = 700;
const MAX_WIDTH = 960;
const WIDE_ASPECT = 16 / 9;
const EXTRUDED_ASPECT = 4 / 3;

// Display-only settings for a high-contrast height visualization (floor vs countertops).
const HEIGHT_CONTRAST_PCT_LO = 3;
const HEIGHT_CONTRAST_PCT_HI = 97;
const HEIGHT_CONTRAST_GAMMA = 0.9;
const HEIGHT_CONTRAST_DENSITY_THRESH = 1e-6;

// Display-only: height above estimated floor (AGL). Use a fixed range so floor vs countertop pops.
const HEIGHT_AGL_VIEW_MIN_M = 0.0;
const HEIGHT_AGL_VIEW_MAX_M = 1.2;
const HEIGHT_AGL_ROOM_MAX_M = 2.5;
const UNKNOWN_CELL_COLOR: [number, number, number, number] = [16, 22, 32, 255];
const UNKNOWN_CELL_ALT_COLOR: [number, number, number, number] = [35, 44, 58, 255];
const PRIMARY_FLOORPLAN_UNKNOWN: [number, number, number, number] = [0, 0, 0, 255];
const PRIMARY_FLOORPLAN_VIEWPORT_PADDING_M = 0.3;

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

function layerAspect(layer?: FloorplanLayer): number {
  const rows = Number(layer?.grid_shape?.[0]);
  const cols = Number(layer?.grid_shape?.[1]);
  return Number.isFinite(rows) && Number.isFinite(cols) && rows > 0 && cols > 0
    ? cols / rows
    : WIDE_ASPECT;
}

function layerRange(layer?: FloorplanLayer): SimpleRange | null {
  const min = Number(layer?.value_min);
  const max = Number(layer?.value_max);
  return Number.isFinite(min) && Number.isFinite(max) && max > min
    ? { min, max }
    : null;
}

const DepthDrawer = memo(function DepthDrawer({
  open,
  onClose,
  diagnostics,
  depthData,
  depthMeta = {},
  onRequestDepthFresh,
  onRequestDepthCached,
  floorplans,
  refreshState = {},
  cachePairState = {},
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
  const [normalsView, setNormalsView] = useState<'surface' | 'detail'>('surface');
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [drawerWidth, setDrawerWidth] = useState<number>(DEFAULT_WIDTH);
  const onRequestDepthCachedRef = useRef(onRequestDepthCached);
  const drawerRef = useRef<HTMLDivElement | null>(null);
  const isResizingRef = useRef(false);
  const previousUserSelectRef = useRef('');
  const activePointerIdRef = useRef<number | null>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const heatmapSourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rgbSourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rgbDepthOverlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rgbDepthOverlaySourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const normalsCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const normalsSourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
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
  const structuralFloorplanCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamPreviewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const extrudedCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const floorPlaneCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const heightfieldCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const pointCloudCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const exportSequenceRef = useRef(0);
  const [primitivesView, setPrimitivesView] = useState<'obstacles' | 'heightfield' | 'point-cloud' | 'visible-floor'>('heightfield');
  const [pointCloudColorMode, setPointCloudColorMode] = useState<'rgb' | 'depth' | 'confidence'>('rgb');
  const [primitivesShowPreview, setPrimitivesShowPreview] = useState(true);
  const [primitivesShowBoxes, setPrimitivesShowBoxes] = useState(true);
  const [primitivesSelectedBoxId, setPrimitivesSelectedBoxId] = useState<string | null>(null);
  const [primitivesHeightExaggeration, setPrimitivesHeightExaggeration] = useState(1.6);
  const [primitivesMinHeightM, setPrimitivesMinHeightM] = useState(DEFAULT_OBSTACLE_SETTINGS.minHeightM);
  const [primitivesMinDensity, setPrimitivesMinDensity] = useState(DEFAULT_OBSTACLE_SETTINGS.minDensity);
  const [primitivesMinFootprintM2, setPrimitivesMinFootprintM2] = useState(DEFAULT_OBSTACLE_SETTINGS.minFootprintM2);
  const [primitivesMaxBoxes, setPrimitivesMaxBoxes] = useState(DEFAULT_OBSTACLE_SETTINGS.maxBoxes);
  const [primitivesMaxCells, setPrimitivesMaxCells] = useState(DEFAULT_OBSTACLE_SETTINGS.maxCells);
  const [heatmapRangeMode, setHeatmapRangeMode] = useState<'auto' | 'full' | 'locked'>('auto');
  const [rgbOverlayOpacity, setRgbOverlayOpacity] = useState(0.48);
  const [lockedHeatmapRanges, setLockedHeatmapRanges] = useState<Record<string, SimpleRange>>({});
  const [aglRangeMode, setAglRangeMode] = useState<'furniture' | 'room' | 'auto'>('furniture');
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  // Floorplan selection (declared early to avoid TDZ in hooks below)
  const cameraFloorplan = floorplans[selectedCamera];
  const structuralFloorplanBounds = useMemo(() => {
    const bounds = cameraFloorplan?.bounds;
    const minX = Number(bounds?.min_x);
    const maxX = Number(bounds?.max_x);
    const minZ = Number(bounds?.min_z);
    const maxZ = Number(bounds?.max_z);
    if (
      !Number.isFinite(minX)
      || !Number.isFinite(maxX)
      || !Number.isFinite(minZ)
      || !Number.isFinite(maxZ)
      || maxX <= minX
      || maxZ <= minZ
    ) {
      return undefined;
    }
    return { min_x: minX, max_x: maxX, min_z: minZ, max_z: maxZ };
  }, [cameraFloorplan?.bounds]);
  const selectedRefresh = refreshState[selectedCamera];
  const selectedCachePair = cachePairState[selectedCamera];
  const refreshPending = selectedRefresh?.phase === 'requesting-depth'
    || selectedRefresh?.phase === 'requesting-floorplan';
  const refreshError = selectedRefresh?.phase === 'error'
    ? selectedRefresh.error || 'unknown_refresh_error'
    : '';
  const cachePairError = !refreshPending && selectedCachePair?.phase === 'error'
    ? selectedCachePair.error || 'cached_pair_unavailable'
    : '';
  const densityLayer = cameraFloorplan?.density;
  const heightLayer = cameraFloorplan?.height;
  const heightAglLayer = cameraFloorplan?.height_agl;
  const distanceLayer = cameraFloorplan?.distance;
  const gradientLayer = cameraFloorplan?.gradient;
  const obstacleHeightLayer = cameraFloorplan?.obstacle_height;
  const walkableLayer = cameraFloorplan?.walkable;
  const observedLayer = cameraFloorplan?.observed;
  const unknownLayer = cameraFloorplan?.unknown;
  const inferredWalkableLayer = cameraFloorplan?.inferred_walkable;
  const structuralHeightLayer = cameraFloorplan?.structural_height;
  const surfaceObservedLayer = cameraFloorplan?.surface_observed;
  const roomFootprintLayer = cameraFloorplan?.room_footprint;
  const wallSupportLayer = cameraFloorplan?.wall_support;
  const roomBoundaryLayer = cameraFloorplan?.room_boundary;
  const surfaceRgbLayer = cameraFloorplan?.surface_rgb;
  // v8 exposes unknown explicitly. Renderers invert that binary mask so an
  // unknown cell never falls through as a valid zero-height/obstacle cell.
  const observationMaskLayer = observedLayer ?? densityLayer;
  const renderObservationMaskLayer = unknownLayer ?? observationMaskLayer;
  const renderObservationMaskInvert = Boolean(unknownLayer);
  const floorplanViewportMaskLayer = unknownLayer ?? observationMaskLayer;
  const floorplanViewportMaskInvert = Boolean(unknownLayer);
  const primaryFloorplanLayer = heightLayer?.grid_b64 && heightLayer.grid_shape
    ? heightLayer
    : heightAglLayer;
  const floorplanError = cameraFloorplan?.error ?? null;
  const floorplanServedFromCache = cameraFloorplan?.served_from_cache ?? false;
  const hasDensity = !!(densityLayer && densityLayer.grid_b64 && densityLayer.grid_shape);
  const hasHeight = !!(heightLayer && heightLayer.grid_b64 && heightLayer.grid_shape);
  const hasHeightAgl = !!(heightAglLayer && heightAglLayer.grid_b64 && heightAglLayer.grid_shape);
  const hasDistance = !!(distanceLayer && distanceLayer.grid_b64 && distanceLayer.grid_shape);
  const hasGradient = !!(gradientLayer && gradientLayer.grid_b64 && gradientLayer.grid_shape);
  const hasObstacleHeight = !!(obstacleHeightLayer && obstacleHeightLayer.grid_b64 && obstacleHeightLayer.grid_shape);
  const hasWalkable = !!(walkableLayer && walkableLayer.grid_b64 && walkableLayer.grid_shape);
  const hasObserved = !!(observedLayer?.grid_b64 && observedLayer.grid_shape);
  const hasUnknown = !!(unknownLayer?.grid_b64 && unknownLayer.grid_shape);
  const hasInferredWalkable = !!(inferredWalkableLayer?.grid_b64 && inferredWalkableLayer.grid_shape);
  const observationCounts = useMemo(() => floorplanObservationCounts(
    cameraFloorplan?.observation_meta,
    inferredWalkableLayer?.grid_b64
      ? decodeFloat32(inferredWalkableLayer.grid_b64)
      : null,
  ), [
    cameraFloorplan?.observation_meta,
    inferredWalkableLayer?.grid_b64,
  ]);
  const floorplanDisplayViewport = useMemo(() => {
    if (!floorplanViewportMaskLayer?.grid_b64 || !floorplanViewportMaskLayer.grid_shape) return null;
    const [rows, cols] = floorplanViewportMaskLayer.grid_shape;
    const values = decodeFloat32(floorplanViewportMaskLayer.grid_b64);
    if (!values || values.length < rows * cols) return null;
    return computeObservedFloorplanViewport({
      maskValues: values,
      rows,
      cols,
      maskInvert: floorplanViewportMaskInvert,
      maskThreshold: HEIGHT_CONTRAST_DENSITY_THRESH,
      bounds: cameraFloorplan?.bounds,
      paddingM: PRIMARY_FLOORPLAN_VIEWPORT_PADDING_M,
      fallbackPaddingCells: 3,
    });
  }, [
    cameraFloorplan?.bounds,
    floorplanViewportMaskInvert,
    floorplanViewportMaskLayer?.grid_b64,
    floorplanViewportMaskLayer?.grid_shape,
  ]);
  const displaySourceRectForLayer = useCallback((layer?: FloorplanLayer) => {
    if (!floorplanDisplayViewport || !layer?.grid_shape || !floorplanViewportMaskLayer?.grid_shape) {
      return undefined;
    }
    if (
      layer.grid_shape[0] !== floorplanViewportMaskLayer.grid_shape[0]
      || layer.grid_shape[1] !== floorplanViewportMaskLayer.grid_shape[1]
    ) {
      return undefined;
    }
    return floorplanDisplayViewport.sourceRect;
  }, [floorplanDisplayViewport, floorplanViewportMaskLayer?.grid_shape]);
  const displayLayerAspect = useCallback((layer?: FloorplanLayer) => {
    const sourceRect = displaySourceRectForLayer(layer);
    return sourceRect ? sourceRect.width / sourceRect.height : layerAspect(layer);
  }, [displaySourceRectForLayer]);
  const heightMaxRaw = heightLayer?.value_max ?? null;
  const densityRange = layerRange(densityLayer);
  const heightRange = layerRange(heightLayer);
  const distanceRange = layerRange(distanceLayer);
  const gradientRange = layerRange(gradientLayer);
  const obstacleHeightRange = layerRange(obstacleHeightLayer);
  const heightContrastRange = useMemo(() => {
    if (!heightLayer?.grid_b64 || !heightLayer?.grid_shape) return null;
    const [rows, cols] = heightLayer.grid_shape;
    if (!rows || !cols) return null;
    const heightValues = decodeFloat32(heightLayer.grid_b64);
    if (!heightValues || heightValues.length < rows * cols) return null;

    let observedValues: Float32Array | null = null;
    if (observationMaskLayer?.grid_b64 && observationMaskLayer?.grid_shape) {
      const [maskRows, maskCols] = observationMaskLayer.grid_shape;
      if (maskRows === rows && maskCols === cols) {
        observedValues = decodeFloat32(observationMaskLayer.grid_b64);
      }
    }

    const samples: number[] = [];
    const n = rows * cols;
    for (let idx = 0; idx < n; idx += 1) {
      const v = heightValues[idx];
      if (!Number.isFinite(v)) continue;
      if (observedValues) {
        const observed = observedValues[idx];
        if (!Number.isFinite(observed) || observed <= HEIGHT_CONTRAST_DENSITY_THRESH) continue;
      }
      samples.push(v);
    }
    if (samples.length < 16) return null;
    samples.sort((a, b) => a - b);

    const lo = percentileSorted(samples, HEIGHT_CONTRAST_PCT_LO);
    const hi = percentileSorted(samples, HEIGHT_CONTRAST_PCT_HI);
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return null;
    return { min: lo, max: hi };
  }, [
    heightLayer?.grid_b64,
    heightLayer?.grid_shape,
    observationMaskLayer?.grid_b64,
    observationMaskLayer?.grid_shape,
  ]);
  const heightAglAutoRange = useMemo(() => {
    if (!heightAglLayer?.grid_b64 || !heightAglLayer.grid_shape) return null;
    const values = decodeFloat32(heightAglLayer.grid_b64);
    if (!values) return null;
    let mask: Float32Array | null = null;
    if (
      observationMaskLayer?.grid_b64
      && observationMaskLayer.grid_shape?.[0] === heightAglLayer.grid_shape[0]
      && observationMaskLayer.grid_shape?.[1] === heightAglLayer.grid_shape[1]
    ) {
      mask = decodeFloat32(observationMaskLayer.grid_b64);
    }
    return computeMaskedRange(values, {
      mask,
      maskThreshold: HEIGHT_CONTRAST_DENSITY_THRESH,
      lowPercentile: 2,
      highPercentile: 98,
    });
  }, [
    heightAglLayer?.grid_b64,
    heightAglLayer?.grid_shape,
    observationMaskLayer?.grid_b64,
    observationMaskLayer?.grid_shape,
  ]);
  const heightAglRange = useMemo<SimpleRange>(() => {
    if (aglRangeMode === 'room') return { min: 0, max: HEIGHT_AGL_ROOM_MAX_M };
    if (aglRangeMode === 'auto' && heightAglAutoRange) {
      return { min: heightAglAutoRange.min, max: heightAglAutoRange.max };
    }
    return { min: HEIGHT_AGL_VIEW_MIN_M, max: HEIGHT_AGL_VIEW_MAX_M };
  }, [aglRangeMode, heightAglAutoRange]);
  const primaryFloorplanRange = useMemo<SimpleRange | null>(() => {
    if (primaryFloorplanLayer === heightLayer) {
      return heightContrastRange ?? heightRange;
    }
    if (heightAglAutoRange) {
      return {
        min: heightAglAutoRange.min,
        max: heightAglAutoRange.max,
      };
    }
    return primaryFloorplanLayer ? heightAglRange : null;
  }, [
    heightAglAutoRange,
    heightAglRange,
    heightContrastRange,
    heightLayer,
    heightRange,
    primaryFloorplanLayer,
  ]);
  const spanX = cameraFloorplan?.bounds ? Math.abs((cameraFloorplan.bounds.max_x ?? 0) - (cameraFloorplan.bounds.min_x ?? 0)) : undefined;
  const spanZ = cameraFloorplan?.bounds ? Math.abs((cameraFloorplan.bounds.max_z ?? 0) - (cameraFloorplan.bounds.min_z ?? 0)) : undefined;
  const floorplanStatusText = useMemo(() => {
    if (!open || (activeTab !== 'heatmap' && activeTab !== '3d')) return '';
    if (selectedRefresh?.phase === 'requesting-depth') {
      return 'Capturing a fresh depth snapshot; the previous coherent view remains visible.';
    }
    if (selectedRefresh?.phase === 'requesting-floorplan') {
      return 'Building the exact matching floorplan; the previous coherent view remains visible.';
    }
    if (selectedCachePair?.phase === 'requesting-depth') {
      return 'Checking cached depth; the current coherent view remains visible.';
    }
    if (selectedCachePair?.phase === 'waiting-floorplan') {
      return 'Cached depth is staged while waiting for its exact cached floorplan. No inference will run.';
    }
    if (selectedCachePair?.phase === 'waiting-depth') {
      return 'Cached floorplan is staged while waiting for its exact cached depth. No inference will run.';
    }
    if (floorplanError) return `Floorplan error: ${floorplanError}`;
    if (cameraFloorplan?.served_from_cache) return 'Showing cached floorplan. Press refresh to regenerate.';
    if (cameraFloorplan) return 'Floorplan generated from the latest depth snapshot.';
    return 'Waiting for floorplan data.';
  }, [
    open,
    activeTab,
    selectedRefresh?.phase,
    selectedCachePair?.phase,
    floorplanError,
    cameraFloorplan,
  ]);
  const panelErrorText = refreshError
    ? `Refresh failed: ${refreshError}`
    : cachePairError
      ? `Cached pair unavailable: ${cachePairError}. The previous coherent view was preserved.`
      : floorplanError
        ? `Floorplan error: ${floorplanError}`
        : '';
  const clearCanvasElement = useCallback((canvas: HTMLCanvasElement | null) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const renderTopdownLayer = useCallback(
    (
      canvas: HTMLCanvasElement | null,
      layer: FloorplanLayer | undefined,
      palette: (t: number) => [number, number, number],
      showInferredWalkable = false,
    ) => {
      renderLayerToCanvas(canvas, layer, palette, {
        fit: 'contain',
        maskLayer: renderObservationMaskLayer,
        maskThreshold: HEIGHT_CONTRAST_DENSITY_THRESH,
        maskInvert: renderObservationMaskInvert,
        unknownColor: UNKNOWN_CELL_COLOR,
        unknownAltColor: UNKNOWN_CELL_ALT_COLOR,
        sourceRect: displaySourceRectForLayer(layer),
        ...(showInferredWalkable ? { inferredWalkableLayer } : {}),
        imageSmoothing: false,
      });
    },
    [
      inferredWalkableLayer,
      displaySourceRectForLayer,
      renderObservationMaskInvert,
      renderObservationMaskLayer,
    ]
  );

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
    onRequestDepthCachedRef.current = onRequestDepthCached;
  }, [onRequestDepthCached]);

  useEffect(() => {
    if (!open || !selectedCamera) return;
    // The websocket hook owns transport state and may provide a new callback
    // identity after unrelated dashboard renders. Cache reads are a
    // transition effect: one per drawer-open/camera-selection transition, not
    // one per callback identity.
    onRequestDepthCachedRef.current(selectedCamera);
  }, [open, selectedCamera]);

  useEffect(() => {
    if (open) {
      setDrawerWidth(DEFAULT_WIDTH);
    }
  }, [open]);

  const depthEntry = selectedCamera ? depthData[selectedCamera] : undefined;
  const depthMetaEntry = selectedCamera && depthMeta ? depthMeta[selectedCamera] : undefined;
  const summaryEntry = selectedCamera ? diagnostics[selectedCamera] : undefined;
  const decodedDepth = useMemo(() => {
    if (!depthEntry) return null;
    const [height, width] = depthEntry.shape;
    const count = height * width;
    if (!height || !width || count <= 0) return null;
    const depth = depthEntry.depth;
    if (!depth || depth.length < count) return null;
    const confidence = depthEntry.conf;
    const mask = depthEntry.mask;
    const rgbShape = depthEntry.rgb_shape;
    const rgb = (
      depthEntry.rgb
      && Array.isArray(rgbShape)
      && rgbShape[0] === height
      && rgbShape[1] === width
      && rgbShape[2] === 3
      && depthEntry.rgb.length >= count * 3
    )
      ? depthEntry.rgb
      : null;
    return {
      height,
      width,
      count,
      depth,
      confidence: confidence && confidence.length >= count ? confidence : null,
      mask: mask && mask.length >= count ? mask : null,
      rgb,
      rgbShape: rgb ? [height, width, 3] as [number, number, number] : null,
    };
  }, [depthEntry]);
  const heatmapRanges = useMemo(() => {
    if (!decodedDepth) return { robust: null, full: null };
    const common = {
      mask: decodedDepth.mask,
      positiveOnly: true,
    };
    const robust = computeMaskedRange(decodedDepth.depth, {
      ...common,
      lowPercentile: 2,
      highPercentile: 98,
    });
    const full = computeMaskedRange(decodedDepth.depth, {
      ...common,
      lowPercentile: 0,
      highPercentile: 100,
    });
    return { robust, full };
  }, [decodedDepth]);
  const heatmapRange = useMemo(
    () => chooseDepthRange(
      heatmapRangeMode,
      heatmapRanges.robust,
      heatmapRanges.full,
      lockedHeatmapRanges[selectedCamera] ?? null,
    ),
    [heatmapRangeMode, heatmapRanges, lockedHeatmapRanges, selectedCamera],
  );
  const depthAspect = decodedDepth
    ? decodedDepth.width / decodedDepth.height
    : WIDE_ASPECT;
  useEffect(() => {
    if (decodedDepth && !decodedDepth.rgb && pointCloudColorMode === 'rgb') {
      setPointCloudColorMode('depth');
    }
  }, [decodedDepth, pointCloudColorMode]);
  const selectedNormals = normalsView === 'surface'
    ? depthEntry?.surface_normals
    : depthEntry?.normals;
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
    const shape = normalsView === 'surface'
      ? depthEntry.surface_normals_shape
      : depthEntry.normals_shape;
    let height = depthEntry.shape[0];
    let width = depthEntry.shape[1];
    if (Array.isArray(shape) && shape.length >= 2) {
      height = Number(shape[0]) || height;
      width = Number(shape[1]) || width;
    }
    return {
      available: Boolean(selectedNormals),
      height,
      width,
      dtype: depthEntry.normals_dtype || '',
      space: depthEntry.normals_space || '',
      error: depthEntry.normals_error ?? null,
    };
  }, [depthEntry, normalsView, selectedNormals]);
  const normalsStatusText = useMemo(() => {
    if (!depthEntry) return 'Waiting for depth snapshot.';
    if (normalsInfo.error) return `Normals error: ${normalsInfo.error}`;
    if (!normalsInfo.available) {
      return normalsView === 'surface'
        ? 'Plane-aware surface normals are unavailable for this snapshot.'
        : 'Detail normals are unavailable for this snapshot.';
    }
    const spaceLabel = normalsInfo.space || 'camera';
    const dtypeLabel = normalsInfo.dtype || 'float32';
    const viewLabel = normalsView === 'surface' ? 'Plane-aware surface' : 'Edge-aware detail';
    return `${viewLabel} normals attached (${spaceLabel}, ${dtypeLabel}).`;
  }, [depthEntry, normalsInfo, normalsView]);

  useEffect(() => {
    const canvas = heatmapCanvasRef.current;
    if (!canvas || activeTab !== 'heatmap') return;
    if (!decodedDepth || !heatmapRange) {
      heatmapSourceCanvasRef.current = null;
      rgbSourceCanvasRef.current = null;
      rgbDepthOverlaySourceCanvasRef.current = null;
      clearCanvasElement(canvas);
      clearCanvasElement(rgbDepthOverlayCanvasRef.current);
      return;
    }

    const {
      height,
      width,
      count,
      depth,
      confidence,
      mask,
      rgb,
    } = decodedDepth;
    const range = heatmapRange.max - heatmapRange.min;
    const offscreen = document.createElement('canvas');
    offscreen.width = width;
    offscreen.height = height;
    const offCtx = offscreen.getContext('2d');
    if (!offCtx) return;
    const imageData = offCtx.createImageData(width, height);
    const data = imageData.data;

    for (let i = 0; i < count; i += 1) {
      const d = depth[i];
      const idx = i * 4;
      const maskOk = !mask || mask[i] > 0;
      if (!Number.isFinite(d) || d <= 0 || !maskOk) {
        data[idx + 3] = 0;
        continue;
      }
      const norm = Math.min(1, Math.max(0, (d - heatmapRange.min) / range));
      const [r, g, b] = turboColor(norm);
      let alpha = 0.9;
      if (confidence) {
        const conf = Math.max(0, Math.min(1, confidence[i]));
        alpha = 0.3 + conf * 0.7;
      }
      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = Math.round(alpha * 255);
    }

    offCtx.putImageData(imageData, 0, 0);
    heatmapSourceCanvasRef.current = offscreen;

    const drawSourceToDisplay = (
      target: HTMLCanvasElement | null,
      source: HTMLCanvasElement,
    ) => {
      if (!target) return;
      const targetCtx = target.getContext('2d');
      if (!targetCtx) return;
      const rect = target.getBoundingClientRect();
      const targetWidth = rect.width || target.clientWidth || width;
      const targetHeight = targetWidth / (width / height);
      const dpr = window.devicePixelRatio || 1;
      target.width = Math.max(1, Math.round(targetWidth * dpr));
      target.height = Math.max(1, Math.round(targetHeight * dpr));
      targetCtx.save();
      targetCtx.scale(dpr, dpr);
      targetCtx.clearRect(0, 0, targetWidth, targetHeight);
      targetCtx.imageSmoothingEnabled = true;
      targetCtx.imageSmoothingQuality = 'high';
      targetCtx.drawImage(source, 0, 0, targetWidth, targetHeight);
      targetCtx.restore();
    };
    drawSourceToDisplay(canvas, offscreen);

    if (rgb) {
      const rgbSource = document.createElement('canvas');
      rgbSource.width = width;
      rgbSource.height = height;
      const rgbCtx = rgbSource.getContext('2d');
      if (rgbCtx) {
        const rgbImage = rgbCtx.createImageData(width, height);
        for (let idx = 0; idx < count; idx += 1) {
          const source = idx * 3;
          const target = idx * 4;
          rgbImage.data[target] = rgb[source];
          rgbImage.data[target + 1] = rgb[source + 1];
          rgbImage.data[target + 2] = rgb[source + 2];
          rgbImage.data[target + 3] = 255;
        }
        rgbCtx.putImageData(rgbImage, 0, 0);
        rgbSourceCanvasRef.current = rgbSource;

        const overlaySource = document.createElement('canvas');
        overlaySource.width = width;
        overlaySource.height = height;
        const overlayCtx = overlaySource.getContext('2d');
        if (overlayCtx) {
          overlayCtx.drawImage(rgbSource, 0, 0);
          overlayCtx.globalAlpha = rgbOverlayOpacity;
          overlayCtx.drawImage(offscreen, 0, 0);
          overlayCtx.globalAlpha = 1;
          rgbDepthOverlaySourceCanvasRef.current = overlaySource;
          drawSourceToDisplay(rgbDepthOverlayCanvasRef.current, overlaySource);
        }
      }
    } else {
      rgbSourceCanvasRef.current = null;
      rgbDepthOverlaySourceCanvasRef.current = null;
      clearCanvasElement(rgbDepthOverlayCanvasRef.current);
    }
  }, [
    activeTab,
    clearCanvasElement,
    decodedDepth,
    drawerWidth,
    heatmapRange,
    rgbOverlayOpacity,
  ]);

  useEffect(() => {
    const canvas = normalsCanvasRef.current;
    if (!canvas || activeTab !== 'normals') return;
    if (!selectedNormals || !normalsInfo.available) {
      normalsSourceCanvasRef.current = null;
      clearCanvasElement(canvas);
      return;
    }
    const height = normalsInfo.height;
    const width = normalsInfo.width;
    if (!height || !width) {
      normalsSourceCanvasRef.current = null;
      clearCanvasElement(canvas);
      return;
    }
    const normalsArray = selectedNormals;
    if (!normalsArray || normalsArray.length < width * height * 3) {
      normalsSourceCanvasRef.current = null;
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
    const componentScale = normalsView === 'surface' ? 1 / 127 : 1;
    for (let i = 0; i < total; i += 1) {
      const base = i * 3;
      const nx = normalsArray[base] * componentScale;
      const ny = normalsArray[base + 1] * componentScale;
      const nz = normalsArray[base + 2] * componentScale;
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
    normalsSourceCanvasRef.current = offscreen;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const rect = canvas.getBoundingClientRect();
    const targetWidth = rect.width || canvas.clientWidth || width;
    const targetHeight = targetWidth / (width / height);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(targetWidth * dpr));
    canvas.height = Math.max(1, Math.round(targetHeight * dpr));
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, targetWidth, targetHeight);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(offscreen, 0, 0, targetWidth, targetHeight);
    ctx.restore();
  }, [
    activeTab,
    clearCanvasElement,
    drawerWidth,
    normalsInfo,
    normalsView,
    selectedNormals,
  ]);

  // Avoid showing stale topdown canvases when switching cameras.
  useEffect(() => {
    if (!open || activeTab !== 'heatmap') return;
    clearCanvasElement(densityCanvasRef.current);
    clearCanvasElement(heightCanvasRef.current);
    clearCanvasElement(distanceCanvasRef.current);
  }, [open, activeTab, selectedCamera, clearCanvasElement]);

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
        // Geometry is metric height above the fitted floor. Raw camera-Y
        // height is convention-sensitive and can invert or offset obstacles.
        height: cameraFloorplan.height_agl ?? cameraFloorplan.height,
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
  const handleHeightfieldCanvasReady = useCallback((canvas: HTMLCanvasElement | null) => {
    heightfieldCanvasRef.current = canvas;
  }, []);
  const handlePointCloudCanvasReady = useCallback((canvas: HTMLCanvasElement | null) => {
    pointCloudCanvasRef.current = canvas;
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
    const canvas = histogramCanvasRef.current;
    if (!canvas || !depthEntry || activeTab !== 'histogram') return;
    const confArray = decodedDepth?.confidence;
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
  }, [depthEntry, decodedDepth, activeTab]);

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
    if ((activeTab !== 'heatmap' && activeTab !== '3d') || !open) return;

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
        clearCanvasElement(structuralFloorplanCanvasRef.current);
      }
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
          maskLayer: renderObservationMaskLayer,
          maskThreshold: HEIGHT_CONTRAST_DENSITY_THRESH,
          maskInvert: renderObservationMaskInvert,
          unknownColor: UNKNOWN_CELL_COLOR,
          unknownAltColor: UNKNOWN_CELL_ALT_COLOR,
          sourceRect: displaySourceRectForLayer(heightLayer),
          imageSmoothing: false,
        });
        // Height above estimated floor (AGL): user-selected physical range.
        renderLayerToCanvas(heightAglCanvasRef.current, heightAglLayer, turboColor, {
          fit: 'contain',
          valueMin: heightAglRange.min,
          valueMax: heightAglRange.max,
          gamma: 1.0,
          maskLayer: renderObservationMaskLayer,
          maskThreshold: HEIGHT_CONTRAST_DENSITY_THRESH,
          maskInvert: renderObservationMaskInvert,
          unknownColor: UNKNOWN_CELL_COLOR,
          unknownAltColor: UNKNOWN_CELL_ALT_COLOR,
          sourceRect: displaySourceRectForLayer(heightAglLayer),
          imageSmoothing: false,
        });
        renderTopdownLayer(distanceCanvasRef.current, distanceLayer, viridisColor);
        renderTopdownLayer(obstacleHeightCanvasRef.current, obstacleHeightLayer, infernoColor);
        renderTopdownLayer(walkableCanvasRef.current, walkableLayer, bwColor, true);
        renderLayerToCanvas(
          floorplanCompositeCanvasRef.current,
          primaryFloorplanLayer,
          infernoColor,
          {
            fit: 'contain',
            ...(primaryFloorplanRange
              ? {
                valueMin: primaryFloorplanRange.min,
                valueMax: primaryFloorplanRange.max,
              }
              : {}),
            gamma: HEIGHT_CONTRAST_GAMMA,
            maskLayer: renderObservationMaskLayer,
            maskThreshold: HEIGHT_CONTRAST_DENSITY_THRESH,
            maskInvert: renderObservationMaskInvert,
            unknownColor: PRIMARY_FLOORPLAN_UNKNOWN,
            unknownAltColor: PRIMARY_FLOORPLAN_UNKNOWN,
            repairIsolatedMaskHoles: true,
            sourceRect: displaySourceRectForLayer(primaryFloorplanLayer),
            background: '#000',
            imageSmoothing: true,
          },
        );
        renderStructuralFloorplanToCanvas(
          structuralFloorplanCanvasRef.current,
          structuralHeightLayer,
          roomFootprintLayer,
          surfaceObservedLayer,
          wallSupportLayer,
          roomBoundaryLayer,
          surfaceRgbLayer,
          {
            fit: 'contain',
            unknownColor: UNKNOWN_CELL_COLOR,
            unknownAltColor: UNKNOWN_CELL_ALT_COLOR,
            sourceRect: displaySourceRectForLayer(structuralHeightLayer),
            bounds: structuralFloorplanBounds,
            metricGridM: 1,
            imageSmoothing: false,
          }
        );
        renderTopdownLayer(gradientCanvasRef.current, gradientLayer, viridisColor);
      }
    }
  }, [activeTab, open, cameraFloorplan, densityLayer, renderObservationMaskLayer, renderObservationMaskInvert, heightLayer, heightAglLayer, heightContrastRange, heightMaxRaw, heightAglRange, primaryFloorplanLayer, primaryFloorplanRange, distanceLayer, gradientLayer, obstacleHeightLayer, walkableLayer, structuralHeightLayer, roomFootprintLayer, surfaceObservedLayer, wallSupportLayer, roomBoundaryLayer, surfaceRgbLayer, structuralFloorplanBounds, renderTopdownLayer, clearCanvasElement, drawerWidth, floorplanError, diagnosticsOpen, displaySourceRectForLayer]);

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
    const savedAt = new Date();
    const stem = snapshotExportStem(selectedCamera, activeTab, {
      depthTs: depthEntry?.ts ?? null,
      floorplanTs: cameraFloorplan?.snapshot_ts ?? cameraFloorplan?.ts ?? null,
      now: savedAt,
    });
    const prefix = `${stem}_export-${++exportSequenceRef.current}`;
    type ExportRaster = {
      label: string;
      canvas: HTMLCanvasElement | null;
      source: 'payload' | 'floorplan-grid' | 'display';
      valueRange?: SimpleRange | null;
      integerScale?: number;
    };
    const canvases: ExportRaster[] = [];

    const addFloorplanLayer = (
      label: string,
      layer: FloorplanLayer | undefined,
      palette: (t: number) => [number, number, number],
      valueRange?: SimpleRange | null,
      showInferredWalkable = false,
    ) => {
      if (!layer?.grid_shape || !layer.grid_b64) return;
      const [rows, cols] = layer.grid_shape;
      if (!rows || !cols) return;
      const scale = integerExportScale(rows, cols);
      const canvas = document.createElement('canvas');
      const result = renderLayerToCanvas(canvas, layer, palette, {
        fit: 'stretch',
        targetWidthPx: cols * scale,
        targetHeightPx: rows * scale,
        pixelRatio: 1,
        imageSmoothing: false,
        maskLayer: renderObservationMaskLayer,
        maskThreshold: HEIGHT_CONTRAST_DENSITY_THRESH,
        maskInvert: renderObservationMaskInvert,
        unknownColor: UNKNOWN_CELL_COLOR,
        unknownAltColor: UNKNOWN_CELL_ALT_COLOR,
        ...(showInferredWalkable ? { inferredWalkableLayer } : {}),
        ...(valueRange ? { valueMin: valueRange.min, valueMax: valueRange.max } : {}),
      });
      if (!result) return;
      canvases.push({
        label,
        canvas,
        source: 'floorplan-grid',
        valueRange: result.valueRange ?? valueRange ?? null,
        integerScale: scale,
      });
    };

    if (activeTab === 'heatmap') {
      canvases.push({
        label: 'camera-depth-turbo',
        canvas: heatmapSourceCanvasRef.current,
        source: 'payload',
        valueRange: heatmapRange,
      });
      if (rgbSourceCanvasRef.current && rgbDepthOverlaySourceCanvasRef.current) {
        canvases.push({
          label: 'exact-snapshot-rgb',
          canvas: rgbSourceCanvasRef.current,
          source: 'payload',
        });
        canvases.push({
          label: 'exact-snapshot-rgb-depth-overlay',
          canvas: rgbDepthOverlaySourceCanvasRef.current,
          source: 'payload',
          valueRange: heatmapRange,
        });
      }
      if (primaryFloorplanLayer?.grid_shape && primaryFloorplanLayer.grid_b64) {
        const [rows, cols] = primaryFloorplanLayer.grid_shape;
        const sourceRect = displaySourceRectForLayer(primaryFloorplanLayer);
        const outputRows = sourceRect?.height ?? rows;
        const outputCols = sourceRect?.width ?? cols;
        const scale = integerExportScale(outputRows, outputCols);
        const canvas = document.createElement('canvas');
        const result = renderLayerToCanvas(
          canvas,
          primaryFloorplanLayer,
          infernoColor,
          {
            fit: 'stretch',
            targetWidthPx: outputCols * scale,
            targetHeightPx: outputRows * scale,
            pixelRatio: 1,
            imageSmoothing: true,
            maskLayer: renderObservationMaskLayer,
            maskThreshold: HEIGHT_CONTRAST_DENSITY_THRESH,
            maskInvert: renderObservationMaskInvert,
            unknownColor: PRIMARY_FLOORPLAN_UNKNOWN,
            unknownAltColor: PRIMARY_FLOORPLAN_UNKNOWN,
            repairIsolatedMaskHoles: true,
            sourceRect,
            ...(primaryFloorplanRange
              ? {
                valueMin: primaryFloorplanRange.min,
                valueMax: primaryFloorplanRange.max,
              }
              : {}),
            gamma: HEIGHT_CONTRAST_GAMMA,
          },
        );
        if (result) {
          canvases.push({
            label: 'floorplan-observed-height-primary',
            canvas,
            source: 'floorplan-grid',
            valueRange: result.valueRange ?? primaryFloorplanRange,
            integerScale: scale,
          });
        }
      }
      addFloorplanLayer('density-gray', densityLayer, grayscaleColor);
      addFloorplanLayer('height-inferno', heightLayer, infernoColor);
      addFloorplanLayer('height-contrast', heightLayer, turboColor, heightContrastRange);
      addFloorplanLayer('height-agl-turbo', heightAglLayer, turboColor, heightAglRange);
      addFloorplanLayer('distance-viridis', distanceLayer, viridisColor);
      addFloorplanLayer('obstacle-height-clean', obstacleHeightLayer, infernoColor);
      addFloorplanLayer('walkable-binary', walkableLayer, bwColor, null, true);
      addFloorplanLayer('gradient-edges', gradientLayer, viridisColor);

      const compositeLayer = structuralHeightLayer;
      if (compositeLayer?.grid_shape && compositeLayer.grid_b64) {
        const [rows, cols] = compositeLayer.grid_shape;
        const scale = integerExportScale(rows, cols);
        const canvas = document.createElement('canvas');
        const result = renderStructuralFloorplanToCanvas(
          canvas,
          structuralHeightLayer,
          roomFootprintLayer,
          surfaceObservedLayer,
          wallSupportLayer,
          roomBoundaryLayer,
          surfaceRgbLayer,
          {
            fit: 'stretch',
            targetWidthPx: cols * scale,
            targetHeightPx: rows * scale,
            pixelRatio: 1,
            imageSmoothing: false,
            unknownColor: UNKNOWN_CELL_COLOR,
            unknownAltColor: UNKNOWN_CELL_ALT_COLOR,
            bounds: structuralFloorplanBounds,
            metricGridM: 1,
          },
        );
        if (result) {
          canvases.push({
            label: 'floorplan-structural-height',
            canvas,
            source: 'floorplan-grid',
            valueRange: result.valueRange,
            integerScale: scale,
          });
        }
      }
    } else if (activeTab === 'normals') {
      canvases.push({
        label: normalsView === 'surface'
          ? 'surface-normals-rgb'
          : 'detail-normals-rgb',
        canvas: normalsSourceCanvasRef.current,
        source: 'payload',
      });
    } else if (activeTab === '3d') {
      canvases.push({ label: 'stream-preview', canvas: streamPreviewCanvasRef.current, source: 'display' });
      if (primitivesView === 'visible-floor') {
        canvases.push({ label: 'visible-floor-plane', canvas: floorPlaneCanvasRef.current, source: 'display' });
      } else if (primitivesView === 'heightfield') {
        canvases.push({ label: 'cached-heightfield', canvas: heightfieldCanvasRef.current, source: 'display' });
      } else if (primitivesView === 'point-cloud') {
        canvases.push({ label: `calibrated-point-cloud-${pointCloudColorMode}`, canvas: pointCloudCanvasRef.current, source: 'display' });
      } else {
        canvases.push({ label: 'extruded-floorplan', canvas: extrudedCanvasRef.current, source: 'display' });
      }
    } else if (activeTab === 'histogram') {
      canvases.push({ label: 'histogram', canvas: histogramCanvasRef.current, source: 'display' });
    }

    const imageNames: string[] = [];
    const exports: Array<Record<string, unknown>> = [];
    canvases.forEach(({ label, canvas, source, valueRange, integerScale }) => {
      if (!canvas || canvas.width <= 0 || canvas.height <= 0) return;
      const filename = `${prefix}_${safePart(label)}.png`;
      imageNames.push(filename);
      exports.push({
        filename,
        label,
        source,
        width_px: canvas.width,
        height_px: canvas.height,
        integer_scale: integerScale ?? null,
        value_range: valueRange
          ? { min: valueRange.min, max: valueRange.max, units: 'm' }
          : null,
      });
      saveCanvas(canvas, filename);
    });

    if (!imageNames.length) return;
    const manifest = {
      saved_at: savedAt.toISOString(),
      camera: selectedCamera,
      tab: activeTab,
      images: imageNames,
      exports,
      depth_ts_us: depthEntry?.ts ?? null,
      depth_served_from_cache: depthMetaEntry?.servedFromCache ?? null,
      depth_source_camera_id: depthMetaEntry?.sourceCameraId ?? null,
      depth_snapshot_ref: depthMetaEntry?.snapshotRef ?? null,
      depth_snapshot_id: depthMetaEntry?.snapshotId ?? null,
      depth_snapshot_content_sha256: depthMetaEntry?.snapshotContentSha256 ?? null,
      exact_snapshot_rgb_component_sha256: depthMetaEntry?.rgbComponentSha256 ?? null,
      depth_transfer_bytes: depthMetaEntry?.transferBytes ?? null,
      depth_transfer_duration_ms: depthMetaEntry?.transferDurationMs ?? null,
      exact_snapshot_rgb_available: Boolean(decodedDepth?.rgb),
      exact_snapshot_rgb_shape: decodedDepth?.rgbShape ?? null,
      rgb_depth_overlay_opacity: activeTab === 'heatmap' && decodedDepth?.rgb
        ? rgbOverlayOpacity
        : null,
      depth_range_mode: activeTab === 'heatmap' ? heatmapRangeMode : null,
      depth_range: activeTab === 'heatmap' ? heatmapRange : null,
      normals_view: activeTab === 'normals' ? normalsView : null,
      agl_range_mode: activeTab === 'heatmap' ? aglRangeMode : null,
      agl_range: activeTab === 'heatmap' ? heightAglRange : null,
      floorplan_ts_us: cameraFloorplan?.ts ?? null,
      floorplan_snapshot_ts_us: cameraFloorplan?.snapshot_ts ?? null,
      floorplan_snapshot_ref: cameraFloorplan?.snapshot_ref ?? null,
      floorplan_snapshot_id: cameraFloorplan?.snapshot_id ?? null,
      floorplan_snapshot_content_sha256: cameraFloorplan?.snapshot_content_sha256 ?? null,
      floorplan_served_from_cache: cameraFloorplan?.served_from_cache ?? null,
      floorplan_display_viewport: activeTab === 'heatmap' && floorplanDisplayViewport
        ? {
          mode: 'observed_cells_crop_v2',
          source_rect_grid: floorplanDisplayViewport.sourceRect,
          crop_width_m: floorplanDisplayViewport.cropWidthM,
          crop_depth_m: floorplanDisplayViewport.cropDepthM,
          full_width_m: floorplanDisplayViewport.fullWidthM,
          full_depth_m: floorplanDisplayViewport.fullDepthM,
          authoritative_grid_unchanged: true,
        }
        : null,
      floor_plane_status: activeTab === '3d' ? visibleFloorResult.status : undefined,
    };
    saveBlob(
      new Blob([JSON.stringify(manifest, null, 2)], { type: 'application/json' }),
      `${prefix}_manifest.json`,
    );
  }, [
    activeTab,
    aglRangeMode,
    cameraFloorplan,
    densityLayer,
    depthEntry,
    depthMetaEntry,
    decodedDepth,
    distanceLayer,
    floorplanDisplayViewport,
    displaySourceRectForLayer,
    gradientLayer,
    heatmapRange,
    heatmapRangeMode,
    heightAglLayer,
    heightAglRange,
    heightContrastRange,
    heightLayer,
    obstacleHeightLayer,
    observationMaskLayer,
    normalsView,
    renderObservationMaskLayer,
    renderObservationMaskInvert,
    inferredWalkableLayer,
    roomBoundaryLayer,
    roomFootprintLayer,
    structuralFloorplanBounds,
    structuralHeightLayer,
    surfaceObservedLayer,
    surfaceRgbLayer,
    wallSupportLayer,
    pointCloudColorMode,
    primaryFloorplanLayer,
    primaryFloorplanRange,
    primitivesView,
    rgbOverlayOpacity,
    saveBlob,
    saveCanvas,
    selectedCamera,
    visibleFloorResult.status,
    walkableLayer,
  ]);

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
                    }}
                    disabled={refreshPending}
                    aria-label="Refresh depth frame"
                    title={refreshPending ? 'A coherent depth/floorplan refresh is already running.' : undefined}
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
              {panelErrorText && (
                <p className="floorplan-error">{panelErrorText}</p>
              )}
              {floorplanStatusText && !panelErrorText && (
                <p className="floorplan-status">{floorplanStatusText}</p>
              )}
              <div className="depth-view-controls">
                <label>
                  <span>Depth range</span>
                  <select
                    value={heatmapRangeMode}
                    onChange={(event) => {
                      const mode = event.target.value as 'auto' | 'full' | 'locked';
                      if (mode === 'locked' && selectedCamera && heatmapRange) {
                        setLockedHeatmapRanges((previous) => ({
                          ...previous,
                          [selectedCamera]: { ...heatmapRange },
                        }));
                      }
                      setHeatmapRangeMode(mode);
                    }}
                  >
                    <option value="auto">Auto (p2–p98)</option>
                    <option value="full">Full valid range</option>
                    <option value="locked">Locked</option>
                  </select>
                </label>
                {heatmapRangeMode === 'locked' && heatmapRange && (
                  <div className="depth-range-inputs">
                    <label>
                      <span>Min m</span>
                      <input
                        type="number"
                        step="0.05"
                        value={heatmapRange.min}
                        onChange={(event) => {
                          const min = Number(event.target.value);
                          if (!Number.isFinite(min) || min >= heatmapRange.max) return;
                          setLockedHeatmapRanges((previous) => ({
                            ...previous,
                            [selectedCamera]: { min, max: heatmapRange.max },
                          }));
                        }}
                      />
                    </label>
                    <label>
                      <span>Max m</span>
                      <input
                        type="number"
                        step="0.05"
                        value={heatmapRange.max}
                        onChange={(event) => {
                          const max = Number(event.target.value);
                          if (!Number.isFinite(max) || max <= heatmapRange.min) return;
                          setLockedHeatmapRanges((previous) => ({
                            ...previous,
                            [selectedCamera]: { min: heatmapRange.min, max },
                          }));
                        }}
                      />
                    </label>
                  </div>
                )}
                {decodedDepth?.rgb && (
                  <label>
                    <span>RGB overlay {Math.round(rgbOverlayOpacity * 100)}%</span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={rgbOverlayOpacity}
                      onChange={(event) => setRgbOverlayOpacity(Number(event.target.value))}
                    />
                  </label>
                )}
                <span className="depth-view-controls__meta">
                  {heatmapRanges.robust
                    ? `${heatmapRanges.robust.sampleCount.toLocaleString()} mask-valid positive pixels`
                    : 'No valid depth pixels'}
                </span>
              </div>

              <div className="heatmap-grid heatmap-grid--primary">
                <div className="heatmap-cell heatmap-cell--left heatmap-cell--primary">
                  <div className="heatmap-cell__scale">
                    {renderScale(turboGradient, heatmapRange?.min ?? undefined, heatmapRange ? (heatmapRange.min + heatmapRange.max) / 2 : undefined, heatmapRange?.max ?? undefined, ' m')}
                  </div>
                  <div className="heatmap-cell__body">
                    <div className="heatmap-cell__title">Camera Depth · Primary</div>
                    <canvas
                      ref={heatmapCanvasRef}
                      className="heatmap-canvas"
                      style={{ aspectRatio: `${depthAspect}` }}
                    />
                  </div>
                </div>
                {decodedDepth?.rgb && (
                  <div className="heatmap-cell heatmap-cell--primary heatmap-cell--no-scale">
                    <div className="heatmap-cell__body">
                      <div className="heatmap-cell__title">Exact Snapshot RGB + Depth</div>
                      <canvas
                        ref={rgbDepthOverlayCanvasRef}
                        className="heatmap-canvas"
                        style={{ aspectRatio: `${depthAspect}` }}
                      />
                      <div className="floorplan-class-keys">
                        <div className="unknown-cell-key">
                          Digest-verified RGB and depth from one snapshot identity
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                <div className="heatmap-cell heatmap-cell--primary heatmap-cell--no-scale">
                  <div className="heatmap-cell__body">
                    <div className="heatmap-cell__title">Observed Height Floorplan · Primary</div>
                    <canvas
                      ref={floorplanCompositeCanvasRef}
                      className="heatmap-canvas"
                      style={{ aspectRatio: `${displayLayerAspect(primaryFloorplanLayer)}` }}
                    />
                    <div className="floorplan-class-keys">
                      {floorplanDisplayViewport && (
                        <div className="unknown-cell-key">
                          Observed crop {formatNumber(floorplanDisplayViewport.cropWidthM, 1)}×{formatNumber(floorplanDisplayViewport.cropDepthM, 1)} m
                          {' '}· full grid {formatNumber(floorplanDisplayViewport.fullWidthM, 1)}×{formatNumber(floorplanDisplayViewport.fullDepthM, 1)} m
                        </div>
                      )}
                      <div className="unknown-cell-key">
                        Black = unknown · Inferno = observed scalar height
                        {primaryFloorplanRange
                          ? ` (${formatNumber(primaryFloorplanRange.min, 2)}–${formatNumber(primaryFloorplanRange.max, 2)} m)`
                          : ''}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <details
                className="depth-diagnostics"
                open={diagnosticsOpen}
                onToggle={(event) => setDiagnosticsOpen(event.currentTarget.open)}
              >
                <summary>Diagnostic layers</summary>
                <div className="depth-diagnostics__toolbar">
                  <label>
                    <span>AGL range</span>
                    <select
                      value={aglRangeMode}
                      onChange={(event) => setAglRangeMode(event.target.value as 'furniture' | 'room' | 'auto')}
                    >
                      <option value="furniture">Furniture (0–1.2 m)</option>
                      <option value="room">Room (0–2.5 m)</option>
                      <option value="auto">Auto (p2–p98)</option>
                    </select>
                  </label>
                </div>
                <div className="heatmap-grid">
                  <div className="heatmap-cell heatmap-cell--no-scale">
                    <div className="heatmap-cell__body">
                      <div className="heatmap-cell__title">Structural Composite (Diagnostic)</div>
                      <canvas
                        ref={structuralFloorplanCanvasRef}
                        className="heatmap-canvas heatmap-canvas--semantic"
                        style={{ aspectRatio: `${displayLayerAspect(structuralHeightLayer)}` }}
                      />
                      <div className="floorplan-class-keys">
                        <div className="unknown-cell-key">
                          Inferred footprint, surface RGB, furniture contours, and wall support
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="heatmap-cell heatmap-cell--left">
                    <div className="heatmap-cell__body">
                      <div className="heatmap-cell__title">Density (Grayscale)</div>
                      <canvas ref={densityCanvasRef} className="heatmap-canvas" style={{ aspectRatio: `${displayLayerAspect(densityLayer)}` }} />
                    </div>
                    <div className="heatmap-cell__scale">
                      {renderScale(densityGradient, densityRange?.min, densityRange ? (densityRange.min + densityRange.max) / 2 : undefined, densityRange?.max)}
                    </div>
                  </div>
                  <div className="heatmap-cell heatmap-cell--right">
                    <div className="heatmap-cell__scale">
                      {renderScale(infernoGradient, heightRange?.min, heightRange ? (heightRange.min + heightRange.max) / 2 : undefined, heightRange?.max, ' m')}
                    </div>
                    <div className="heatmap-cell__body">
                      <div className="heatmap-cell__title">Raw Height (Inferno)</div>
                      <canvas ref={heightCanvasRef} className="heatmap-canvas" style={{ aspectRatio: `${displayLayerAspect(heightLayer)}` }} />
                    </div>
                  </div>
                  <div className="heatmap-cell heatmap-cell--right">
                    <div className="heatmap-cell__body">
                      <div className="heatmap-cell__title">Raw Height (Contrast)</div>
                      <canvas ref={heightContrastCanvasRef} className="heatmap-canvas" style={{ aspectRatio: `${displayLayerAspect(heightLayer)}` }} />
                    </div>
                    <div className="heatmap-cell__scale">
                      {renderScale(turboGradient, heightContrastRange?.min, heightContrastRange ? (heightContrastRange.min + heightContrastRange.max) / 2 : undefined, heightContrastRange?.max, ' m')}
                    </div>
                  </div>
                  <div className="heatmap-cell heatmap-cell--left">
                    <div className="heatmap-cell__scale">
                      {renderScale(turboGradient, heightAglRange.min, (heightAglRange.min + heightAglRange.max) / 2, heightAglRange.max, ' m')}
                    </div>
                    <div className="heatmap-cell__body">
                      <div className="heatmap-cell__title">Height Above Floor</div>
                      <canvas ref={heightAglCanvasRef} className="heatmap-canvas" style={{ aspectRatio: `${displayLayerAspect(heightAglLayer)}` }} />
                    </div>
                  </div>
                  <div className="heatmap-cell heatmap-cell--right">
                    <div className="heatmap-cell__scale">
                      {renderScale(viridisGradient, distanceRange?.min, distanceRange ? (distanceRange.min + distanceRange.max) / 2 : undefined, distanceRange?.max, ' m')}
                    </div>
                    <div className="heatmap-cell__body">
                      <div className="heatmap-cell__title">Distance (Viridis)</div>
                      <canvas ref={distanceCanvasRef} className="heatmap-canvas" style={{ aspectRatio: `${displayLayerAspect(distanceLayer)}` }} />
                    </div>
                  </div>
                  <div className="heatmap-cell heatmap-cell--left">
                    <div className="heatmap-cell__scale">
                      {renderScale(infernoGradient, obstacleHeightRange?.min, obstacleHeightRange ? (obstacleHeightRange.min + obstacleHeightRange.max) / 2 : undefined, obstacleHeightRange?.max, ' m')}
                    </div>
                    <div className="heatmap-cell__body">
                      <div className="heatmap-cell__title">Obstacle Height (Clean)</div>
                      <canvas ref={obstacleHeightCanvasRef} className="heatmap-canvas" style={{ aspectRatio: `${displayLayerAspect(obstacleHeightLayer)}` }} />
                    </div>
                  </div>
                  <div className="heatmap-cell heatmap-cell--right">
                    <div className="heatmap-cell__body">
                      <div className="heatmap-cell__title">Walkable (Binary)</div>
                      <canvas ref={walkableCanvasRef} className="heatmap-canvas heatmap-canvas--semantic" style={{ aspectRatio: `${displayLayerAspect(walkableLayer)}` }} />
                    </div>
                    <div className="heatmap-cell__scale">
                      {renderScale(densityGradient, 0, 0.5, 1)}
                    </div>
                  </div>
                  <div className="heatmap-cell heatmap-cell--right">
                    <div className="heatmap-cell__body">
                      <div className="heatmap-cell__title">Gradient (Edges)</div>
                      <canvas ref={gradientCanvasRef} className="heatmap-canvas" style={{ aspectRatio: `${displayLayerAspect(gradientLayer)}` }} />
                    </div>
                    <div className="heatmap-cell__scale">
                      {renderScale(viridisGradient, gradientRange?.min, gradientRange ? (gradientRange.min + gradientRange.max) / 2 : undefined, gradientRange?.max)}
                    </div>
                  </div>
                </div>
              </details>

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
                <div className="heatmap-meta__item">
                  {hasObserved && hasUnknown
                    ? `Observed ${observationCounts.observed ?? 'n/a'} · unknown ${observationCounts.unknown ?? 'n/a'}${hasInferredWalkable ? ` · inferred walkable ${observationCounts.inferredWalkable}` : ''}`
                    : 'Legacy observation mask fallback in use'}
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
              <div className="primitives-view-toggle" role="tablist" aria-label="Normals view">
                <button
                  type="button"
                  role="tab"
                  aria-selected={normalsView === 'surface'}
                  className={normalsView === 'surface' ? 'active' : ''}
                  onClick={() => setNormalsView('surface')}
                >
                  Surface
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={normalsView === 'detail'}
                  className={normalsView === 'detail' ? 'active' : ''}
                  onClick={() => setNormalsView('detail')}
                >
                  Detail
                </button>
              </div>
              {panelErrorText && <p className="floorplan-error">{panelErrorText}</p>}
              {normalsInfo.error && !panelErrorText && (
                <p className="floorplan-error">Normals error: {normalsInfo.error}</p>
              )}
              {normalsStatusText && !normalsInfo.error && !panelErrorText && (
                <p className="floorplan-status">{normalsStatusText}</p>
              )}
              <div className="heatmap-single">
                <div className="heatmap-cell__body">
                  <div className="heatmap-cell__title">
                    {normalsView === 'surface'
                      ? 'Plane-Aware Surface Normals (RGB)'
                      : 'Edge-Aware Detail Normals (RGB)'}
                  </div>
                  <canvas
                    ref={normalsCanvasRef}
                    className="heatmap-canvas"
                    style={{ aspectRatio: `${normalsInfo.width && normalsInfo.height ? normalsInfo.width / normalsInfo.height : WIDE_ASPECT}` }}
                  />
                </div>
              </div>
              <p className="floorplan-meta">
                {normalsView === 'surface'
                  ? 'Regularized from coherent calibrated depth surfaces. Planes share one robust orientation; uncertain, curved, and boundary regions retain detail normals.'
                  : 'Variance-preserving metric depth normals for inspecting local model behavior and fine structure.'}
                {' '}RGB encodes X/Y/Z orientation from −1..1; transparent pixels lack valid local support.
              </p>
            </>
          )}

          {activeTab === '3d' && (
            <>
              {panelErrorText && (
                <p className="floorplan-error">{panelErrorText}</p>
              )}
              {floorplanStatusText && !panelErrorText && (
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
                  className={primitivesView === 'heightfield' ? 'active' : ''}
                  onClick={() => setPrimitivesView('heightfield')}
                >
                  Heightfield
                </button>
                <button
                  type="button"
                  className={primitivesView === 'point-cloud' ? 'active' : ''}
                  onClick={() => setPrimitivesView('point-cloud')}
                >
                  Point cloud
                </button>
                <button
                  type="button"
                  className={primitivesView === 'visible-floor' ? 'active' : ''}
                  onClick={() => setPrimitivesView('visible-floor')}
                >
                  Visible floor
                </button>
              </div>

              <label className="primitives-toggle primitives-toggle--preview">
                <input
                  type="checkbox"
                  checked={primitivesShowPreview}
                  onChange={(e) => setPrimitivesShowPreview(e.target.checked)}
                />
                <span>Show live camera preview</span>
              </label>

              <div className={`primitives-layout ${primitivesShowPreview ? '' : 'primitives-layout--single'}`}>
                {primitivesShowPreview && (
                  <div className="primitives-panel">
                    <div className="primitives-panel__title">Camera Tile (ROI-style)</div>
                    <canvas
                      ref={streamPreviewCanvasRef}
                      className="primitives-canvas"
                      style={{ aspectRatio: `${WIDE_ASPECT}` }}
                    />
                    <div className="primitives-hint">
                      Cropped from the mosaic, like the ROI editor.
                    </div>
                  </div>
                )}

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
                ) : primitivesView === 'heightfield' ? (
                  <div className="primitives-panel">
                    <div className="primitives-panel__title">Observed Height Above Floor</div>
                    <div className="floor-plane-view-wrap">
                      <CachedHeightfield3DView
                        floorplan={cameraFloorplan}
                        heightLayer={heightAglLayer}
                        densityLayer={observationMaskLayer}
                        heightExaggeration={primitivesHeightExaggeration}
                        onCanvasReady={handleHeightfieldCanvasReady}
                      />
                      {!hasHeightAgl && (
                        <div className="floor-plane-empty">Waiting for cached height_agl and density grids.</div>
                      )}
                    </div>
                    <div className="primitives-submeta">
                      {hasHeightAgl
                        ? `Grid ${heightAglLayer?.grid_shape?.[1] ?? 0}×${heightAglLayer?.grid_shape?.[0] ?? 0} · highest observed surface per cell · unknown cells omitted · zero plane is fitted floor · ${formatNumber(primitivesHeightExaggeration, 1)}× height`
                        : 'Waiting for plane-relative height data…'}
                    </div>
                  </div>
                ) : primitivesView === 'point-cloud' ? (
                  <div className="primitives-panel">
                    <div className="primitives-panel__title">Calibrated Dense Point Cloud</div>
                    <div className="floor-plane-view-wrap">
                      <CalibratedPointCloud3DView
                        depthValues={decodedDepth?.depth}
                        confidenceValues={decodedDepth?.confidence}
                        maskValues={decodedDepth?.mask}
                        rgbValues={decodedDepth?.rgb}
                        rgbShape={decodedDepth?.rgbShape}
                        width={decodedDepth?.width}
                        height={decodedDepth?.height}
                        intrinsics={selectedIntrinsics}
                        floorModel={visibleFloorModel}
                        colorMode={pointCloudColorMode}
                        onCanvasReady={handlePointCloudCanvasReady}
                      />
                      {(!decodedDepth || !selectedIntrinsics) && (
                        <div className="floor-plane-empty">
                          {!decodedDepth ? 'Waiting for cached dense depth.' : 'Waiting for camera intrinsics.'}
                        </div>
                      )}
                    </div>
                    <div className="primitives-submeta">
                      {decodedDepth && selectedIntrinsics
                        ? `Source ${decodedDepth.width}×${decodedDepth.height} · bounded to 250,000 mask-valid points · ${decodedDepth.rgb ? 'exact snapshot RGB available' : 'exact snapshot RGB unavailable'} · camera-local metric frame`
                        : 'Depth and camera calibration are required.'}
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

              {primitivesView === 'heightfield' && (
                <details className="primitives-details">
                  <summary>Heightfield settings</summary>
                  <div className="primitives-settings">
                    <div className="primitives-row">
                      <label>Height exaggeration</label>
                      <input
                        type="range"
                        min={0.5}
                        max={3}
                        step={0.1}
                        value={primitivesHeightExaggeration}
                        onChange={(e) => setPrimitivesHeightExaggeration(parseFloat(e.target.value))}
                      />
                      <span className="primitives-value">{formatNumber(primitivesHeightExaggeration, 1)}×</span>
                    </div>
                  </div>
                </details>
              )}

              {primitivesView === 'point-cloud' && (
                <details className="primitives-details">
                  <summary>Point-cloud settings</summary>
                  <div className="primitives-settings">
                    <div className="primitives-row">
                      <label htmlFor="point-cloud-color-mode">Color by</label>
                      <select
                        id="point-cloud-color-mode"
                        value={pointCloudColorMode}
                        onChange={(event) => setPointCloudColorMode(event.target.value as 'rgb' | 'depth' | 'confidence')}
                      >
                        <option value="rgb" disabled={!decodedDepth?.rgb}>Exact snapshot RGB</option>
                        <option value="depth">Depth (p2–p98)</option>
                        <option value="confidence">Confidence (p2–p98)</option>
                      </select>
                      <span className="primitives-value">{pointCloudColorMode}</span>
                    </div>
                  </div>
                </details>
              )}

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
