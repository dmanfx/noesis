import { PointerEvent as ReactPointerEvent, memo, useCallback, useEffect, useMemo, useRef, useState, RefObject } from 'react';
import { CameraKey, cameraIndex, cameraLabel, detectCameraKey } from '../lib/camera';
import type { MosaicLayout } from '../hooks/useWebSocketClient';
import SemanticSegView from './SemanticSegView';
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

export type SceneFusionPoints = {
  frame?: string;
  units?: string;
  point_count?: number;
  positions_f32_b64?: string;
  colors_rgb_u8_b64?: string;
  confidence_f32_b64?: string;
  provenance_u8_b64?: string;
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
  scene_prior_only?: boolean;
  display_source?: string;
  frame?: string;
  bounds?: { min_x?: number; max_x?: number; min_z?: number; max_z?: number };
  image_flip?: { u?: boolean; v?: boolean };
  scale_m_per_px?: number;
  scale_scene_per_px?: number;
  units?: string;
  s_obj_to_m?: number;
  grid_res_scene?: number;
  max_extent_scene?: number;
  point_count?: number;
  error?: string;
  scene_prior_error?: string;
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
  measured_perimeter?: FloorplanLayer;
  wall_segments?: Array<{
    space?: string;
    start?: [number, number];
    end?: [number, number];
    length_m?: number;
    measured_coverage?: number;
    confidence?: number;
  }>;
  surface_rgb?: FloorplanRgbLayer;
  observation_meta?: {
    observed_cells?: number;
    unknown_cells?: number;
    total_cells?: number;
    [key: string]: unknown;
  };
  height_agl_meta?: { floor_y?: number; floor_estimate?: unknown };
  scene_static_height_agl?: FloorplanLayer;
  scene_static_observed?: FloorplanLayer;
  scene_static_confidence?: FloorplanLayer;
  scene_composite_height_agl?: FloorplanLayer;
  scene_composite_observed?: FloorplanLayer;
  scene_composite_source?: FloorplanLayer;
  scene_prior_diagnostic_density?: FloorplanLayer;
  scene_prior_diagnostic_height?: FloorplanLayer;
  scene_prior_diagnostic_height_agl?: FloorplanLayer;
  scene_prior_diagnostic_distance?: FloorplanLayer;
  scene_prior_diagnostic_gradient?: FloorplanLayer;
  scene_prior_diagnostic_obstacle_height?: FloorplanLayer;
  scene_prior_diagnostic_obstacle_mask?: FloorplanLayer;
  scene_prior_diagnostic_walkable?: FloorplanLayer;
  scene_prior_diagnostic_observed?: FloorplanLayer;
  scene_prior_diagnostic_unknown?: FloorplanLayer;
  scene_prior_diagnostic_inferred_walkable?: FloorplanLayer;
  scene_prior_diagnostic_structural_height?: FloorplanLayer;
  scene_prior_diagnostic_surface_observed?: FloorplanLayer;
  scene_prior_diagnostic_room_footprint?: FloorplanLayer;
  scene_prior_diagnostic_reconstruction_extent?: FloorplanLayer;
  scene_prior_diagnostic_wall_support?: FloorplanLayer;
  scene_prior_diagnostic_room_boundary?: FloorplanLayer;
  scene_prior_diagnostic_measured_perimeter?: FloorplanLayer;
  scene_prior_diagnostic_surface_rgb?: FloorplanRgbLayer;
  scene_prior_diagnostic_confidence?: FloorplanLayer;
  scene_prior_floor_supported?: FloorplanLayer;
  scene_prior_floor_height?: FloorplanLayer;
  scene_prior_diagnostic_meta?: {
    contract?: string;
    contract_version?: number;
    source?: string;
    bounds?: FloorplanResponse['bounds'];
    grid_shape?: [number, number];
    raster_orientation?: string;
    point_count?: number;
  };
  scene_prior_meta?: {
    contract?: string;
    contract_version?: number;
    status?: string;
    prior_id?: string;
    space_id?: string;
    mode?: string;
    source_type?: string;
    source_model?: string;
    composition_policy?: string;
    live_observed_cells?: number;
    static_available_cells?: number;
    static_fill_cells?: number;
    composite_observed_cells?: number;
    display_source?: string;
    quality?: {
      passed?: boolean;
      source_point_count?: number;
      selected_point_count?: number;
      authored_cell_count?: number;
      observed_cell_count?: number;
      floor_supported_cell_count?: number;
      obstacle_cell_count?: number;
      authored_observed_fraction?: number;
      authored_floor_supported_fraction?: number;
      alignment_status?: string;
    };
    reason?: string;
  };
  scene_fusion_height_agl?: FloorplanLayer;
  scene_fusion_observed?: FloorplanLayer;
  scene_fusion_confidence?: FloorplanLayer;
  scene_fusion_obstacle_height?: FloorplanLayer;
  scene_fusion_obstacle_mask?: FloorplanLayer;
  scene_fusion_floor_supported?: FloorplanLayer;
  scene_fusion_floor_height?: FloorplanLayer;
  scene_fusion_provenance?: FloorplanLayer;
  scene_fusion_diagnostic_density?: FloorplanLayer;
  scene_fusion_diagnostic_height?: FloorplanLayer;
  scene_fusion_diagnostic_height_agl?: FloorplanLayer;
  scene_fusion_diagnostic_distance?: FloorplanLayer;
  scene_fusion_diagnostic_gradient?: FloorplanLayer;
  scene_fusion_diagnostic_obstacle_height?: FloorplanLayer;
  scene_fusion_diagnostic_walkable?: FloorplanLayer;
  scene_fusion_diagnostic_structural_height?: FloorplanLayer;
  scene_fusion_diagnostic_surface_observed?: FloorplanLayer;
  scene_fusion_diagnostic_room_footprint?: FloorplanLayer;
  scene_fusion_diagnostic_wall_support?: FloorplanLayer;
  scene_fusion_diagnostic_room_boundary?: FloorplanLayer;
  scene_fusion_diagnostic_surface_rgb?: FloorplanRgbLayer;
  scene_fusion_points?: SceneFusionPoints;
  scene_fusion_error?: string;
  scene_fusion_meta?: {
    contract?: string;
    contract_version?: number;
    fusion_id?: string;
    status?: string;
    diagnostic_only?: boolean;
    space_id?: string;
    camera_id?: string;
    inference?: string;
    quality?: {
      fixed_to_phone_median_m?: number;
      fixed_overlap_0p25m?: number;
      phone_overlap_0p25m?: number;
    };
    source_counts?: {
      fixed_only?: number;
      phone_only?: number;
      fixed_phone_agreement?: number;
    };
    diagnostic_layers?: {
      contract?: string;
      contract_version?: number;
      source?: string;
      derived_at_catalog_load?: boolean;
      mapanything_inference_triggered?: boolean;
      registration_triggered?: boolean;
      bounds?: FloorplanResponse['bounds'];
      grid_shape?: [number, number];
      raster_orientation?: string;
      fixed_anchor_view_count?: number;
      phone_view_count?: number;
      predicted_anchor_geometry_used_in_fusion?: boolean;
    };
  };
};

export type DepthRefreshEntry = {
  phase: 'requesting-depth' | 'requesting-floorplan' | 'error';
  error?: string;
};

type ProductionDepthRefreshEntry = {
  status: 'idle' | 'capturing-floorplan' | 'loading-depth' | 'error';
  requestId?: string;
  snapshotTsUs?: number;
  error?: string;
};

type FloorplanRequestOptions = {
  camera?: string;
  requestId?: string;
  maxAgeSec?: number;
  gridResM?: number;
  maxExtentM?: number;
  cacheOnly?: boolean;
  scenePriorOnly?: boolean;
};

interface DepthDrawerProps {
  open: boolean;
  onClose: () => void;
  depthData: Record<string, DepthEntry>;
  onRequestDepthFresh?: (cameraId: string) => void;
  onRefreshDepthPanel?: (cameraId: string) => void;
  transportOpen?: boolean;
  floorplans: Record<string, FloorplanResponse>;
  refreshState?: Record<string, DepthRefreshEntry | ProductionDepthRefreshEntry>;
  onRequestFloorplan?: (options: FloorplanRequestOptions) => string | void;
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
  renderHeightfieldNormalsToCanvas,
  renderStructuralFloorplanToCanvas,
  renderTextureFloorplanToCanvas,
  renderLayerToCanvas,
  turboColor,
  viridisColor
} from '../lib/renderUtils';
import {
  chooseDepthRange,
  computeMaskedRange,
  integerExportScale,
  snapshotExportStem,
  type SimpleRange,
} from '../lib/depthQuality.js';
import { computeObservedFloorplanViewport } from '../lib/floorplanViewport.js';
import { buildExtrudedFloorplanModel, DEFAULT_OBSTACLE_SETTINGS, renderExtrudedFloorplanToCanvas } from '../lib/extrudedFloorplan';
import { buildScenePriorFloorPlaneModel } from '../lib/scenePriorFloorPlane';
import FloorPlane3DView from './FloorPlane3DView';
import CachedHeightfield3DView from './CachedHeightfield3DView';
import ScenePriorPointCloud3DView from './ScenePriorPointCloud3DView';

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
const UNKNOWN_CELL_ALT_COLOR: [number, number, number, number] = [20, 28, 39, 255];
const PRIMARY_FLOORPLAN_VIEWPORT_PADDING_M = 1.0;
const PCF_DIAGNOSTIC_CONTENT_PADDING_PX = 36;
const INFERRED_CELL_COLOR: [number, number, number, number] = [35, 57, 67, 255];
const INFERRED_CELL_ALT_COLOR: [number, number, number, number] = [41, 68, 79, 255];

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
  depthData,
  onRequestDepthFresh,
  onRefreshDepthPanel,
  transportOpen = true,
  floorplans,
  refreshState = {},
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
  const [activeTab, setActiveTab] = useState<'heatmap' | 'normals' | '3d' | 'semantic' | 'histogram' | 'metrics'>('heatmap');
  const [normalsView, setNormalsView] = useState<'surface' | 'detail'>('surface');
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [drawerWidth, setDrawerWidth] = useState<number>(DEFAULT_WIDTH);
  const onRequestFloorplanRef = useRef(onRequestFloorplan);
  const scenePriorProbedRef = useRef(new Set<string>());
  const drawerRef = useRef<HTMLDivElement | null>(null);
  const isResizingRef = useRef(false);
  const previousUserSelectRef = useRef('');
  const activePointerIdRef = useRef<number | null>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const heatmapSourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
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
  const [primitivesShowPreview, setPrimitivesShowPreview] = useState(true);
  const [primitivesShowBoxes, setPrimitivesShowBoxes] = useState(true);
  const [primitivesSelectedBoxId, setPrimitivesSelectedBoxId] = useState<string | null>(null);
  const [primitivesHeightExaggeration, setPrimitivesHeightExaggeration] = useState(1);
  const [primitivesMinHeightM, setPrimitivesMinHeightM] = useState(DEFAULT_OBSTACLE_SETTINGS.minHeightM);
  const [primitivesMinDensity, setPrimitivesMinDensity] = useState(DEFAULT_OBSTACLE_SETTINGS.minDensity);
  const [primitivesMinFootprintM2, setPrimitivesMinFootprintM2] = useState(DEFAULT_OBSTACLE_SETTINGS.minFootprintM2);
  const [primitivesMaxBoxes, setPrimitivesMaxBoxes] = useState(DEFAULT_OBSTACLE_SETTINGS.maxBoxes);
  const [primitivesMaxCells, setPrimitivesMaxCells] = useState(DEFAULT_OBSTACLE_SETTINGS.maxCells);
  const [heatmapRangeMode, setHeatmapRangeMode] = useState<'auto' | 'full' | 'locked'>('auto');
  const [lockedHeatmapRanges, setLockedHeatmapRanges] = useState<Record<string, SimpleRange>>({});
  const [aglRangeMode, setAglRangeMode] = useState<'furniture' | 'room' | 'auto'>('furniture');
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  // Floorplan selection (declared early to avoid TDZ in hooks below)
  const cameraFloorplan = floorplans[selectedCamera];
  const detailedFloorplanBounds = useMemo(() => {
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
  const selectedRefreshPhase = selectedRefresh && 'phase' in selectedRefresh
    ? selectedRefresh.phase
    : selectedRefresh?.status === 'capturing-floorplan'
      ? 'requesting-floorplan'
      : selectedRefresh?.status === 'loading-depth'
        ? 'requesting-depth'
        : selectedRefresh?.status === 'error'
          ? 'error'
          : undefined;
  const anyRefreshPending = Object.values(refreshState).some((entry) => (
    ('phase' in entry && (
      entry.phase === 'requesting-depth' || entry.phase === 'requesting-floorplan'
    ))
    || ('status' in entry && (
      entry.status === 'capturing-floorplan' || entry.status === 'loading-depth'
    ))
  ));
  const refreshError = selectedRefreshPhase === 'error'
    ? selectedRefresh.error || 'unknown_refresh_error'
    : '';
  const hasCanonicalPcf = Boolean(
    cameraFloorplan?.scene_prior_only === true
    && cameraFloorplan?.display_source === 'pcf'
  );
  const densityLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_density : undefined;
  const heightLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_height : undefined;
  const heightAglLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_height_agl : undefined;
  const distanceLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_distance : undefined;
  const gradientLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_gradient : undefined;
  const obstacleHeightLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_obstacle_height : undefined;
  const walkableLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_walkable : undefined;
  const observedLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_observed : undefined;
  const unknownLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_unknown : undefined;
  const inferredWalkableLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_inferred_walkable : undefined;
  const structuralHeightLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_structural_height : undefined;
  const surfaceObservedLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_surface_observed : undefined;
  const roomFootprintLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_room_footprint : undefined;
  const reconstructionExtentLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_reconstruction_extent : undefined;
  const wallSupportLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_wall_support : undefined;
  const roomBoundaryLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_room_boundary : undefined;
  const measuredPerimeterLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_measured_perimeter : undefined;
  const surfaceRgbLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_surface_rgb : undefined;
  const scenePriorFloorHeightLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_floor_height : undefined;
  const scenePriorFloorSupportedLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_floor_supported : undefined;
  const pcfHeightLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_height_agl : undefined;
  const pcfObservedLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_observed : undefined;
  const pcfConfidenceLayer = hasCanonicalPcf ? cameraFloorplan?.scene_prior_diagnostic_confidence : undefined;
  const scenePriorMeta = hasCanonicalPcf ? cameraFloorplan?.scene_prior_meta : undefined;
  // v8 exposes unknown explicitly. Renderers invert that binary mask so an
  // unknown cell never falls through as a valid zero-height/obstacle cell.
  const observationMaskLayer = observedLayer ?? densityLayer;
  const renderObservationMaskLayer = unknownLayer ?? observationMaskLayer;
  const renderObservationMaskInvert = Boolean(unknownLayer);
  // Frame every PCF view around the complete admitted reconstruction. The
  // authored room footprint remains a semantic overlay, not a crop boundary.
  const floorplanViewportMaskLayer = reconstructionExtentLayer
    ?? roomFootprintLayer
    ?? inferredWalkableLayer
    ?? unknownLayer
    ?? observationMaskLayer;
  const floorplanViewportMaskInvert = floorplanViewportMaskLayer === unknownLayer
    && Boolean(unknownLayer);
  const floorplanError = cameraFloorplan?.error ?? null;
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
  const hasScenePrior = !!(pcfHeightLayer?.grid_b64 && pcfHeightLayer.grid_shape);
  const primitivesHeightLayer = hasScenePrior ? pcfHeightLayer : undefined;
  const primitivesObservedLayer = hasScenePrior ? pcfObservedLayer : undefined;
  const primitivesObstacleHeightLayer = hasScenePrior ? obstacleHeightLayer : undefined;
  const primitivesObstacleObservedLayer = hasScenePrior
    ? cameraFloorplan?.scene_prior_diagnostic_obstacle_mask
    : undefined;
  const observationCounts = useMemo(() => {
    const countPositive = (layer?: FloorplanLayer) => {
      const values = layer?.grid_b64 ? decodeFloat32(layer.grid_b64) : null;
      if (!values) return null;
      let count = 0;
      for (const value of values) {
        if (Number.isFinite(value) && value > 0.5) count += 1;
      }
      return count;
    };
    return {
      observed: countPositive(observedLayer),
      unknown: countPositive(unknownLayer),
      inferredWalkable: countPositive(inferredWalkableLayer) ?? 0,
    };
  }, [
    inferredWalkableLayer?.grid_b64,
    observedLayer?.grid_b64,
    unknownLayer?.grid_b64,
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
  const spanX = cameraFloorplan?.bounds ? Math.abs((cameraFloorplan.bounds.max_x ?? 0) - (cameraFloorplan.bounds.min_x ?? 0)) : undefined;
  const spanZ = cameraFloorplan?.bounds ? Math.abs((cameraFloorplan.bounds.max_z ?? 0) - (cameraFloorplan.bounds.min_z ?? 0)) : undefined;
  const floorplanStatusText = useMemo(() => {
    if (!open || (activeTab !== 'heatmap' && activeTab !== '3d')) return '';
    if (selectedRefreshPhase === 'requesting-depth') {
      return 'Capturing a static comparison snapshot; canonical PCF remains visible.';
    }
    if (selectedRefreshPhase === 'requesting-floorplan') {
      return 'Capturing a static comparison snapshot; canonical PCF remains visible.';
    }
    if (floorplanError) return `Floorplan error: ${floorplanError}`;
    if (hasCanonicalPcf) {
      return `Showing canonical PCF room reconstruction${scenePriorMeta?.prior_id ? ` · ${scenePriorMeta.prior_id}` : ''}.`;
    }
    return 'Loading the canonical PCF room reconstruction.';
  }, [
    open,
    activeTab,
    selectedRefreshPhase,
    floorplanError,
    hasCanonicalPcf,
    scenePriorMeta?.prior_id,
  ]);
  const panelErrorText = refreshError
    ? `Refresh failed: ${refreshError}`
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
  const drawCameraOriginMarker = useCallback((
    canvas: HTMLCanvasElement | null,
    layer: FloorplanLayer | undefined,
    rendered: ReturnType<typeof renderLayerToCanvas>,
  ) => {
    if (!canvas || !layer?.grid_shape || !rendered || !detailedFloorplanBounds) return;
    const [rows, cols] = layer.grid_shape;
    const spanX = detailedFloorplanBounds.max_x - detailedFloorplanBounds.min_x;
    const spanZ = detailedFloorplanBounds.max_z - detailedFloorplanBounds.min_z;
    if (
      rows <= 0
      || cols <= 0
      || spanX <= 0
      || spanZ <= 0
      || 0 < detailedFloorplanBounds.min_x
      || 0 > detailedFloorplanBounds.max_x
      || 0 < detailedFloorplanBounds.min_z
      || 0 > detailedFloorplanBounds.max_z
    ) {
      return;
    }
    const source = rendered.sourceRectGrid;
    const content = rendered.contentRectPx;
    const cameraColumn = ((0 - detailedFloorplanBounds.min_x) / spanX) * cols;
    const cameraRow = ((detailedFloorplanBounds.max_z - 0) / spanZ) * rows;
    const normalizedX = (cameraColumn - source.x) / source.width;
    const normalizedY = (cameraRow - source.y) / source.height;
    if (normalizedX < 0 || normalizedX > 1 || normalizedY < 0 || normalizedY > 1) return;

    const x = content.x + (normalizedX * content.w);
    const y = content.y + (normalizedY * content.h);
    const dpr = Math.max(1, canvas.width / Math.max(1, canvas.getBoundingClientRect().width));
    const size = 8 * dpr;
    const context = canvas.getContext('2d');
    if (!context) return;
    context.save();
    context.fillStyle = 'rgba(255, 215, 64, 0.98)';
    context.strokeStyle = 'rgba(0, 0, 0, 0.82)';
    context.lineWidth = Math.max(1.5, 1.5 * dpr);
    context.beginPath();
    context.moveTo(x, y - (size * 1.45));
    context.lineTo(x + size, y + (size * 0.45));
    context.lineTo(x - size, y + (size * 0.45));
    context.closePath();
    context.fill();
    context.stroke();
    context.beginPath();
    context.moveTo(x, y - (size * 1.45));
    context.lineTo(x, y - (size * 3.0));
    context.stroke();
    context.font = `bold ${Math.round(9 * dpr)}px sans-serif`;
    context.fillStyle = 'rgba(255, 224, 102, 0.98)';
    context.strokeStyle = 'rgba(0, 0, 0, 0.82)';
    context.lineWidth = Math.max(2, 2.5 * dpr);
    context.strokeText('CAM', x + (size * 1.25), y + (size * 0.25));
    context.fillText('CAM', x + (size * 1.25), y + (size * 0.25));
    context.restore();
  }, [detailedFloorplanBounds]);

  const renderTopdownLayer = useCallback(
    (
      canvas: HTMLCanvasElement | null,
      layer: FloorplanLayer | undefined,
      palette: (t: number) => [number, number, number],
      showInferredWalkable = false,
    ) => {
      const rendered = renderLayerToCanvas(canvas, layer, palette, {
        fit: 'contain',
        maskLayer: renderObservationMaskLayer,
        maskThreshold: HEIGHT_CONTRAST_DENSITY_THRESH,
        maskInvert: renderObservationMaskInvert,
        unknownColor: UNKNOWN_CELL_COLOR,
        unknownAltColor: UNKNOWN_CELL_ALT_COLOR,
        sourceRect: displaySourceRectForLayer(layer),
        ...(showInferredWalkable ? { inferredWalkableLayer } : {}),
        contentPaddingPx: PCF_DIAGNOSTIC_CONTENT_PADDING_PX,
        imageSmoothing: false,
      });
      drawCameraOriginMarker(canvas, layer, rendered);
      return rendered;
    },
    [
      drawCameraOriginMarker,
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
    onRequestFloorplanRef.current = onRequestFloorplan;
  }, [onRequestFloorplan]);

  useEffect(() => {
    scenePriorProbedRef.current.clear();
  }, [calibrationEpoch]);

  useEffect(() => {
    if (!open || !selectedCamera || !transportOpen || !onRequestFloorplanRef.current) return;
    if (scenePriorProbedRef.current.has(selectedCamera)) return;
    const hasCanonicalPcf = Boolean(
      cameraFloorplan?.scene_prior_only === true
      && cameraFloorplan?.display_source === 'pcf'
      && cameraFloorplan?.scene_prior_diagnostic_height_agl?.grid_b64
    );
    if (hasCanonicalPcf) return;
    scenePriorProbedRef.current.add(selectedCamera);
    onRequestFloorplanRef.current({
      camera: selectedCamera,
      requestId: `drawer-pcf-${selectedCamera}-${Date.now()}`,
      gridResM: 0.025,
      maxExtentM: 20,
      cacheOnly: true,
      scenePriorOnly: true,
    });
  }, [calibrationEpoch, cameraFloorplan, open, selectedCamera, transportOpen]);

  useEffect(() => {
    if (open && transportOpen) return;
    scenePriorProbedRef.current.clear();
  }, [open, transportOpen]);

  useEffect(() => {
    if (open) {
      setDrawerWidth(DEFAULT_WIDTH);
    }
  }, [open]);

  const heatmapRanges = useMemo(() => {
    if (!heightAglLayer?.grid_b64 || !heightAglLayer.grid_shape) {
      return { robust: null, full: null };
    }
    const values = decodeFloat32(heightAglLayer.grid_b64);
    let mask: Float32Array | null = null;
    if (
      observationMaskLayer?.grid_b64
      && observationMaskLayer.grid_shape?.[0] === heightAglLayer.grid_shape[0]
      && observationMaskLayer.grid_shape?.[1] === heightAglLayer.grid_shape[1]
    ) {
      mask = decodeFloat32(observationMaskLayer.grid_b64);
    }
    const common = { mask, maskThreshold: HEIGHT_CONTRAST_DENSITY_THRESH };
    const robust = computeMaskedRange(values, {
      ...common,
      lowPercentile: 2,
      highPercentile: 98,
    });
    const full = computeMaskedRange(values, {
      ...common,
      lowPercentile: 0,
      highPercentile: 100,
    });
    return { robust, full };
  }, [
    heightAglLayer?.grid_b64,
    heightAglLayer?.grid_shape,
    observationMaskLayer?.grid_b64,
    observationMaskLayer?.grid_shape,
  ]);
  const heatmapRange = useMemo(
    () => chooseDepthRange(
      heatmapRangeMode,
      heatmapRanges.robust,
      heatmapRanges.full,
      lockedHeatmapRanges[selectedCamera] ?? null,
    ),
    [heatmapRangeMode, heatmapRanges, lockedHeatmapRanges, selectedCamera],
  );
  const depthAspect = displayLayerAspect(heightAglLayer);
  const normalsInfo = useMemo(() => {
    if (!heightAglLayer?.grid_shape || !heightAglLayer.grid_b64) {
      return {
        available: false,
        height: 0,
        width: 0,
        dtype: '',
        space: '',
        error: null as string | null,
      };
    }
    return {
      available: true,
      height: heightAglLayer.grid_shape[0],
      width: heightAglLayer.grid_shape[1],
      dtype: 'derived float32',
      space: 'camera-local ground',
      error: null as string | null,
    };
  }, [heightAglLayer?.grid_b64, heightAglLayer?.grid_shape]);
  const normalsStatusText = useMemo(() => {
    if (!normalsInfo.available) return 'Waiting for the canonical PCF heightfield.';
    return normalsView === 'surface'
      ? 'PCF surface normals derived from the smoothed room heightfield.'
      : 'PCF detail normals derived from the native 2.5 cm room heightfield.';
  }, [normalsInfo.available, normalsView]);

  useEffect(() => {
    const canvas = heatmapCanvasRef.current;
    if (!canvas || activeTab !== 'heatmap') return;
    if (!heightAglLayer?.grid_b64 || !heightAglLayer.grid_shape || !heatmapRange) {
      heatmapSourceCanvasRef.current = null;
      clearCanvasElement(canvas);
      return;
    }
    const [rows, columns] = heightAglLayer.grid_shape;
    const offscreen = document.createElement('canvas');
    renderLayerToCanvas(offscreen, heightAglLayer, turboColor, {
      fit: 'stretch',
      targetWidthPx: columns,
      targetHeightPx: rows,
      pixelRatio: 1,
      valueMin: heatmapRange.min,
      valueMax: heatmapRange.max,
      maskLayer: renderObservationMaskLayer,
      maskThreshold: HEIGHT_CONTRAST_DENSITY_THRESH,
      maskInvert: renderObservationMaskInvert,
      unknownColor: UNKNOWN_CELL_COLOR,
      unknownAltColor: UNKNOWN_CELL_ALT_COLOR,
      sourceRect: displaySourceRectForLayer(heightAglLayer),
      contentPaddingPx: PCF_DIAGNOSTIC_CONTENT_PADDING_PX,
      imageSmoothing: false,
    });
    heatmapSourceCanvasRef.current = offscreen;
    const rendered = renderLayerToCanvas(canvas, heightAglLayer, turboColor, {
      fit: 'contain',
      valueMin: heatmapRange.min,
      valueMax: heatmapRange.max,
      maskLayer: renderObservationMaskLayer,
      maskThreshold: HEIGHT_CONTRAST_DENSITY_THRESH,
      maskInvert: renderObservationMaskInvert,
      unknownColor: UNKNOWN_CELL_COLOR,
      unknownAltColor: UNKNOWN_CELL_ALT_COLOR,
      sourceRect: displaySourceRectForLayer(heightAglLayer),
      imageSmoothing: false,
    });
    drawCameraOriginMarker(canvas, heightAglLayer, rendered);
  }, [
    activeTab,
    clearCanvasElement,
    displaySourceRectForLayer,
    drawCameraOriginMarker,
    drawerWidth,
    heightAglLayer,
    heatmapRange,
    renderObservationMaskInvert,
    renderObservationMaskLayer,
  ]);

  useEffect(() => {
    const canvas = normalsCanvasRef.current;
    if (!canvas || activeTab !== 'normals') return;
    if (!normalsInfo.available) {
      normalsSourceCanvasRef.current = null;
      clearCanvasElement(canvas);
      return;
    }
    normalsSourceCanvasRef.current = renderHeightfieldNormalsToCanvas(
      canvas,
      heightAglLayer,
      observedLayer,
      {
        resolutionM: Number(cameraFloorplan?.scale_m_per_px) || 0.025,
        smoothingRadius: normalsView === 'surface' ? 1 : 0,
      },
    );
  }, [
    activeTab,
    cameraFloorplan?.scale_m_per_px,
    clearCanvasElement,
    drawerWidth,
    heightAglLayer,
    normalsInfo,
    normalsView,
    observedLayer,
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
        // Canonical PCF obstacle support only. Static-camera products remain
        // stored for future comparison and never enter presentation state.
        height: primitivesObstacleHeightLayer,
        density: primitivesObstacleObservedLayer,
      },
      {
        maxCells: primitivesMaxCells,
        minHeightM: primitivesMinHeightM,
        minDensity: primitivesMinDensity,
        minFootprintM2: primitivesMinFootprintM2,
        maxBoxes: primitivesMaxBoxes,
      }
    );
  }, [cameraFloorplan, primitivesObstacleHeightLayer, primitivesObstacleObservedLayer, primitivesMaxBoxes, primitivesMaxCells, primitivesMinDensity, primitivesMinFootprintM2, primitivesMinHeightM]);

  const obstacleBoxes = primitivesModel?.boxes ?? [];
  const scenePriorFloorResult = useMemo(() => buildScenePriorFloorPlaneModel({
    cameraId: selectedCamera,
    floorplan: cameraFloorplan,
    heightLayer: scenePriorFloorHeightLayer,
    observedLayer: scenePriorFloorSupportedLayer,
  }), [
    selectedCamera,
    cameraFloorplan,
    scenePriorFloorHeightLayer,
    scenePriorFloorSupportedLayer,
  ]);
  const floorPlaneResult = scenePriorFloorResult;
  const visibleFloorModel = floorPlaneResult.model;
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
    if (!canvas || activeTab !== 'histogram') return;
    const confArray = pcfConfidenceLayer?.grid_b64
      ? decodeFloat32(pcfConfidenceLayer.grid_b64)
      : null;
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
  }, [activeTab, pcfConfidenceLayer?.grid_b64]);

  const metrics = useMemo(() => {
    if (!hasCanonicalPcf || !scenePriorMeta) {
      return [] as Array<{ label: string; type: 'bar' | 'text'; text: string; fraction?: number }>;
    }
    const confidenceValues = pcfConfidenceLayer?.grid_b64
      ? decodeFloat32(pcfConfidenceLayer.grid_b64)
      : null;
    let confidenceSum = 0;
    let confidenceCount = 0;
    if (confidenceValues) {
      for (const value of confidenceValues) {
        if (!Number.isFinite(value) || value <= 0) continue;
        confidenceSum += value;
        confidenceCount += 1;
      }
    }
    const confidence = confidenceCount > 0
      ? Math.max(0, Math.min(1, confidenceSum / confidenceCount))
      : null;
    const coverageRaw = scenePriorMeta.quality?.authored_observed_fraction;
    const floorSupportRaw = scenePriorMeta.quality?.authored_floor_supported_fraction;
    const coverage = Number.isFinite(coverageRaw)
      ? Math.max(0, Math.min(1, Number(coverageRaw)))
      : null;
    const floorSupport = Number.isFinite(floorSupportRaw)
      ? Math.max(0, Math.min(1, Number(floorSupportRaw)))
      : null;
    return [
      {
        label: 'PCF Evidence Confidence',
        type: 'bar' as const,
        fraction: confidence ?? 0,
        text: confidence !== null ? `${Math.round(confidence * 100)}%` : 'n/a'
      },
      {
        label: 'PCF Observed Coverage',
        type: 'bar' as const,
        fraction: coverage ?? 0,
        text: coverage !== null ? `${Math.round(coverage * 100)}%` : 'n/a'
      },
      {
        label: 'PCF Floor Support',
        type: 'bar' as const,
        fraction: floorSupport ?? 0,
        text: floorSupport !== null ? `${Math.round(floorSupport * 100)}%` : 'n/a'
      },
      {
        label: 'Selected Points',
        type: 'text' as const,
        text: Number(scenePriorMeta.quality?.selected_point_count ?? 0).toLocaleString()
      },
      {
        label: 'Prior Revision',
        type: 'text' as const,
        text: scenePriorMeta.prior_id ?? 'n/a'
      },
      {
        label: 'Method',
        type: 'text' as const,
        text: 'MAPANYTHING + DA3 PRIOR-CONDITIONED FUSION'
      }
    ];
  }, [hasCanonicalPcf, scenePriorMeta, pcfConfidenceLayer?.grid_b64]);

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
        const heightContrastRendered = renderLayerToCanvas(heightContrastCanvasRef.current, heightLayer, turboColor, {
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
          contentPaddingPx: PCF_DIAGNOSTIC_CONTENT_PADDING_PX,
          imageSmoothing: false,
        });
        drawCameraOriginMarker(heightContrastCanvasRef.current, heightLayer, heightContrastRendered);
        // Height above estimated floor (AGL): user-selected physical range.
        const heightAglRendered = renderLayerToCanvas(heightAglCanvasRef.current, heightAglLayer, turboColor, {
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
          contentPaddingPx: PCF_DIAGNOSTIC_CONTENT_PADDING_PX,
          imageSmoothing: false,
        });
        drawCameraOriginMarker(heightAglCanvasRef.current, heightAglLayer, heightAglRendered);
        renderTopdownLayer(distanceCanvasRef.current, distanceLayer, viridisColor);
        renderTopdownLayer(obstacleHeightCanvasRef.current, obstacleHeightLayer, infernoColor);
        renderTopdownLayer(walkableCanvasRef.current, walkableLayer, bwColor, true);
        const textureFloorplanRendered = renderTextureFloorplanToCanvas(
          floorplanCompositeCanvasRef.current,
          structuralHeightLayer,
          roomFootprintLayer,
          measuredPerimeterLayer,
          surfaceRgbLayer,
          {
            fit: 'contain',
            sourceRect: displaySourceRectForLayer(structuralHeightLayer),
            background: '#000',
            contentPaddingPx: PCF_DIAGNOSTIC_CONTENT_PADDING_PX,
            imageSmoothing: false,
            // Floorplan grids are already serialized in final camera-local
            // orientation. image_flip is diagnostic-only and must not be
            // applied again at the display boundary.
          },
        );
        drawCameraOriginMarker(
          floorplanCompositeCanvasRef.current,
          structuralHeightLayer,
          textureFloorplanRendered,
        );
        const structuralFloorplanRendered = renderStructuralFloorplanToCanvas(
          structuralFloorplanCanvasRef.current,
          structuralHeightLayer,
          roomFootprintLayer,
          surfaceObservedLayer,
          wallSupportLayer,
          roomBoundaryLayer,
          surfaceRgbLayer,
          {
            fit: 'contain',
            sourceRect: displaySourceRectForLayer(structuralHeightLayer),
            bounds: detailedFloorplanBounds,
            metricGridM: 1,
            contentPaddingPx: PCF_DIAGNOSTIC_CONTENT_PADDING_PX,
            imageSmoothing: false,
          }
        );
        drawCameraOriginMarker(
          structuralFloorplanCanvasRef.current,
          structuralHeightLayer,
          structuralFloorplanRendered,
        );
        renderTopdownLayer(gradientCanvasRef.current, gradientLayer, viridisColor);
      }
    }
  }, [activeTab, open, cameraFloorplan, densityLayer, renderObservationMaskLayer, renderObservationMaskInvert, inferredWalkableLayer, heightLayer, heightAglLayer, heightContrastRange, heightMaxRaw, heightAglRange, distanceLayer, gradientLayer, obstacleHeightLayer, walkableLayer, structuralHeightLayer, roomFootprintLayer, surfaceObservedLayer, wallSupportLayer, roomBoundaryLayer, measuredPerimeterLayer, surfaceRgbLayer, renderTopdownLayer, clearCanvasElement, drawerWidth, floorplanError, diagnosticsOpen, displaySourceRectForLayer, detailedFloorplanBounds, drawCameraOriginMarker]);

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
      depthTs: null,
      floorplanTs: cameraFloorplan?.snapshot_ts ?? cameraFloorplan?.ts ?? null,
      now: savedAt,
    });
    const prefix = `${stem}_export-${++exportSequenceRef.current}`;
    type ExportRaster = {
      label: string;
      canvas: HTMLCanvasElement | null;
      source: 'payload' | 'floorplan-grid' | 'display' | 'scene-fusion-diagnostic';
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
      maskLayer: FloorplanLayer | undefined = renderObservationMaskLayer,
      maskInvert = renderObservationMaskInvert,
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
        maskLayer,
        maskThreshold: HEIGHT_CONTRAST_DENSITY_THRESH,
        maskInvert,
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
        label: 'pcf-room-height-primary',
        canvas: heatmapSourceCanvasRef.current,
        source: 'floorplan-grid',
        valueRange: heatmapRange,
      });
      if (structuralHeightLayer?.grid_shape && structuralHeightLayer.grid_b64) {
        const [rows, cols] = structuralHeightLayer.grid_shape;
        const sourceRect = displaySourceRectForLayer(structuralHeightLayer);
        const outputRows = sourceRect?.height ?? rows;
        const outputCols = sourceRect?.width ?? cols;
        const scale = integerExportScale(outputRows, outputCols);
        const canvas = document.createElement('canvas');
        const result = renderTextureFloorplanToCanvas(
          canvas,
          structuralHeightLayer,
          roomFootprintLayer,
          measuredPerimeterLayer,
          surfaceRgbLayer,
          {
            fit: 'stretch',
            targetWidthPx: outputCols * scale,
            targetHeightPx: outputRows * scale,
            pixelRatio: 1,
            imageSmoothing: false,
            sourceRect,
            background: '#000',
            // Match the on-screen final-oriented raster exactly.
          },
        );
        if (result) {
          canvases.push({
            label: 'floorplan-observed-texture-primary',
            canvas,
            source: 'floorplan-grid',
            valueRange: result.valueRange,
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
      addFloorplanLayer('measured-perimeter', measuredPerimeterLayer, bwColor);
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
            bounds: detailedFloorplanBounds,
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
        canvases.push({
          label: 'pcf-visible-floor',
          canvas: floorPlaneCanvasRef.current,
          source: 'display',
        });
      } else if (primitivesView === 'heightfield') {
        canvases.push({
          label: 'pcf-heightfield',
          canvas: heightfieldCanvasRef.current,
          source: 'display',
        });
      } else if (primitivesView === 'point-cloud') {
        canvases.push({
          label: 'pcf-point-cloud',
          canvas: pointCloudCanvasRef.current,
          source: 'display',
        });
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
      presentation_source: 'pcf',
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
      floorplan_image_flip: activeTab === 'heatmap'
        ? {
          u: Boolean(cameraFloorplan?.image_flip?.u),
          v: Boolean(cameraFloorplan?.image_flip?.v),
        }
        : null,
      floorplan_display_viewport: activeTab === 'heatmap' && floorplanDisplayViewport
        ? {
          mode: 'observed_footprint_crop_v1',
          source_rect_grid: floorplanDisplayViewport.sourceRect,
          crop_width_m: floorplanDisplayViewport.cropWidthM,
          crop_depth_m: floorplanDisplayViewport.cropDepthM,
          full_width_m: floorplanDisplayViewport.fullWidthM,
          full_depth_m: floorplanDisplayViewport.fullDepthM,
          authoritative_grid_unchanged: true,
        }
        : null,
      floor_plane_status: activeTab === '3d' ? floorPlaneResult.status : undefined,
      scene_prior_3d_source: activeTab === '3d' && hasScenePrior
        ? {
          prior_id: scenePriorMeta?.prior_id ?? null,
          source_type: scenePriorMeta?.source_type ?? null,
          source_model: scenePriorMeta?.source_model ?? null,
        }
        : null,
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
    detailedFloorplanBounds,
    displaySourceRectForLayer,
    distanceLayer,
    floorplanDisplayViewport,
    gradientLayer,
    heatmapRange,
    heatmapRangeMode,
    hasScenePrior,
    heightAglLayer,
    heightAglRange,
    heightContrastRange,
    heightLayer,
    obstacleHeightLayer,
    observationMaskLayer,
    measuredPerimeterLayer,
    normalsView,
    renderObservationMaskLayer,
    renderObservationMaskInvert,
    inferredWalkableLayer,
    primitivesView,
    saveBlob,
    saveCanvas,
    selectedCamera,
    structuralHeightLayer,
    roomFootprintLayer,
    surfaceObservedLayer,
    wallSupportLayer,
    roomBoundaryLayer,
    scenePriorMeta,
    surfaceRgbLayer,
    floorPlaneResult.status,
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
          <h2>PCF Room Reconstruction</h2>
          <button onClick={onClose} aria-label="Close depth drawer">×</button>
        </header>
        <div className="tabs">
          <button className={activeTab === 'heatmap' ? 'active' : ''} onClick={() => setActiveTab('heatmap')}>Heatmap</button>
          <button className={activeTab === 'normals' ? 'active' : ''} onClick={() => setActiveTab('normals')}>Normals</button>
          <button className={activeTab === '3d' ? 'active' : ''} onClick={() => setActiveTab('3d')}>3D</button>
          <button className={activeTab === 'semantic' ? 'active' : ''} onClick={() => setActiveTab('semantic')}>Sem-seg</button>
          <button className={activeTab === 'histogram' ? 'active' : ''} onClick={() => setActiveTab('histogram')}>Histogram</button>
          <button className={activeTab === 'metrics' ? 'active' : ''} onClick={() => setActiveTab('metrics')}>Metrics</button>
        </div>
        <div className="content">
          {!cameras.length && activeTab !== 'semantic' && <p>No room cameras are available yet.</p>}
          {cameras.length > 0 && activeTab !== 'semantic' && (
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
                      (onRequestDepthFresh ?? onRefreshDepthPanel)?.(selectedCamera);
                    }}
                    disabled={!selectedCamera || !transportOpen || anyRefreshPending}
                    aria-label="Capture static comparison frame"
                    title={
                      !transportOpen
                        ? 'WebSocket is disconnected.'
                        : anyRefreshPending
                          ? 'A static comparison capture is already running.'
                          : 'Capture a static-camera comparison snapshot; PCF remains displayed'
                    }
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
                  <span>PCF height range</span>
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
                <span className="depth-view-controls__meta">
                  {heatmapRanges.robust
                    ? `${heatmapRanges.robust.sampleCount.toLocaleString()} observed PCF cells`
                    : 'Waiting for PCF height cells'}
                </span>
              </div>

              <div className="heatmap-grid heatmap-grid--primary">
                <div className="heatmap-cell heatmap-cell--left heatmap-cell--primary">
                  <div className="heatmap-cell__scale">
                    {renderScale(turboGradient, heatmapRange?.min ?? undefined, heatmapRange ? (heatmapRange.min + heatmapRange.max) / 2 : undefined, heatmapRange?.max ?? undefined, ' m')}
                  </div>
                  <div className="heatmap-cell__body">
                    <div className="heatmap-cell__title">PCF Room Height · Primary</div>
                    <canvas
                      ref={heatmapCanvasRef}
                      className="heatmap-canvas"
                      style={{ aspectRatio: `${depthAspect}` }}
                    />
                  </div>
                </div>
                <div className="heatmap-cell heatmap-cell--primary heatmap-cell--no-scale">
                  <div className="heatmap-cell__body">
                    <div className="heatmap-cell__title">PCF Observed Textured Floorplan · Primary</div>
                    <canvas
                      ref={floorplanCompositeCanvasRef}
                      className="heatmap-canvas"
                      style={{ aspectRatio: `${displayLayerAspect(structuralHeightLayer)}` }}
                    />
                    <div className="floorplan-class-keys">
                      {floorplanDisplayViewport && (
                        <div className="unknown-cell-key">
                          Room framing {formatNumber(floorplanDisplayViewport.cropWidthM, 1)}×{formatNumber(floorplanDisplayViewport.cropDepthM, 1)} m
                          {' '}· full grid {formatNumber(floorplanDisplayViewport.fullWidthM, 1)}×{formatNumber(floorplanDisplayViewport.fullDepthM, 1)} m
                        </div>
                      )}
                      <div className="unknown-cell-key">
                        Texture = observed horizontal surfaces · subtle color = fixed 0–1.65 m structural height
                      </div>
                      <div className="unknown-cell-key">
                        Black = unsupported · gray outline = inferred footprint · light edge = measured wall support
                      </div>
                      <div className="unknown-cell-key">
                        Sampling gaps are repaired only within 0.12 m
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
                  {hasCanonicalPcf
                    ? `PCF prior ${scenePriorMeta?.prior_id ?? 'unknown'} · ${heightAglLayer?.grid_shape?.[1] ?? 0}×${heightAglLayer?.grid_shape?.[0] ?? 0} · camera at bottom, +Z forward`
                    : 'Waiting for the canonical PCF scene prior.'}
                </div>
                <div className="heatmap-meta__item">
                  {hasDensity ? 'PCF normalized point density' : 'Waiting for PCF density data'}
                </div>
                <div className="heatmap-meta__item">
                  {hasHeight ? 'PCF highest supported surface per cell.' : 'Waiting for PCF height data'}
                </div>
                <div className="heatmap-meta__item">
                  {hasHeightAgl ? 'PCF height above fitted floor (AGL).' : 'Waiting for PCF AGL height'}
                </div>
                <div className="heatmap-meta__item">
                  {hasDistance ? 'PCF cell range from the static camera pose.' : 'Waiting for PCF distance data'}
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
                    : 'Waiting for PCF observation support'}
                </div>
              </div>

              {cameraFloorplan && !floorplanError && (
                <p className="floorplan-meta">
                  Points: {cameraFloorplan.point_count ?? 0}
                  {spanX !== undefined && spanZ !== undefined ? ` · Span ${formatNumber(spanX)}m × ${formatNumber(spanZ)}m` : ''}
                  {cameraFloorplan.ts
                    ? ` · Prior created ${new Date(cameraFloorplan.ts / 1000).toLocaleString()}`
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
                      ? 'PCF Smoothed Surface Normals (RGB)'
                      : 'PCF Native-Grid Detail Normals (RGB)'}
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
                  ? 'Derived from the canonical PCF heightfield after one-cell spatial smoothing.'
                  : 'Derived directly from the canonical 2.5 cm PCF heightfield.'}
                {' '}RGB encodes camera-right/up/forward orientation from −1..1; transparent pixels lack PCF support.
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
                    <div className="primitives-panel__title">PCF Extruded Floorplan Primitives</div>
                    <canvas
                      ref={extrudedCanvasRef}
                      className="primitives-canvas primitives-canvas--extruded"
                      style={{ aspectRatio: `${EXTRUDED_ASPECT}` }}
                    />
                    <div className="primitives-submeta">
                      {primitivesModel
                        ? `Prior-conditioned fusion · grid ${primitivesModel.cols}×${primitivesModel.rows} · max height ${formatNumber(primitivesModel.maxHeightM)} m`
                        : 'Waiting for the canonical PCF obstacle grid…'}
                    </div>
                  </div>
                ) : primitivesView === 'heightfield' ? (
                  <div className="primitives-panel">
                    <div className="primitives-panel__title">PCF Room Height Above Floor</div>
                    <div className="floor-plane-view-wrap">
                      <CachedHeightfield3DView
                        floorplan={cameraFloorplan}
                        heightLayer={primitivesHeightLayer}
                        densityLayer={primitivesObservedLayer}
                        heightExaggeration={primitivesHeightExaggeration}
                        onCanvasReady={handleHeightfieldCanvasReady}
                      />
                      {!primitivesHeightLayer?.grid_b64 && (
                        <div className="floor-plane-empty">Waiting for height and observation grids.</div>
                      )}
                    </div>
                    <div className="primitives-submeta">
                      {primitivesHeightLayer?.grid_b64
                        ? `PCF ${scenePriorMeta?.prior_id ?? 'unknown'} · grid ${primitivesHeightLayer.grid_shape?.[1] ?? 0}×${primitivesHeightLayer.grid_shape?.[0] ?? 0} · highest supported surface per cell · unknown cells omitted · zero plane is fitted floor · ${formatNumber(primitivesHeightExaggeration, 1)}× height`
                        : 'Waiting for the canonical PCF heightfield…'}
                    </div>
                  </div>
                ) : primitivesView === 'point-cloud' ? (
                  <div className="primitives-panel">
                    <div className="primitives-panel__title">PCF Conditioned Fusion Point Cloud</div>
                    <div className="floor-plane-view-wrap">
                      {hasScenePrior ? (
                        <ScenePriorPointCloud3DView
                          cameraId={selectedCamera}
                          expectedPriorId={scenePriorMeta?.prior_id}
                          calibrationEpoch={calibrationEpoch}
                          onCanvasReady={handlePointCloudCanvasReady}
                        />
                      ) : null}
                      {!hasScenePrior && (
                        <div className="floor-plane-empty">
                          No canonical PCF point cloud is assigned to this camera.
                        </div>
                      )}
                    </div>
                    <div className="primitives-submeta">
                      {hasScenePrior
                        ? `MapAnything + DA3 prior-conditioned consensus · ${scenePriorMeta?.quality?.selected_point_count ?? 0} selected points · backend world transformed to this camera · prior ${scenePriorMeta?.prior_id ?? 'unknown'}`
                        : 'A PCF scene prior is required.'}
                    </div>
                  </div>
                ) : (
                  <div className="primitives-panel">
                    <div className="primitives-panel__title">PCF Visible Floor Support</div>
                    <div className="floor-plane-view-wrap">
                      <FloorPlane3DView
                        model={visibleFloorModel}
                        onCanvasReady={handleFloorPlaneCanvasReady}
                      />
                      {!visibleFloorModel && (
                        <div className="floor-plane-empty">{floorPlaneResult.message}</div>
                      )}
                    </div>
                    <div className="primitives-submeta">
                      {visibleFloorModel
                        ? `PCF floor support ${visibleFloorModel.metrics.estimatedVisibleFloorPixelCount} cells · footprint ${visibleFloorModel.footprintMesh.cellCount} cells · ${formatNumber(visibleFloorModel.footprintMesh.areaM2, 1)} m² · prior ${scenePriorMeta?.prior_id ?? 'unknown'}`
                        : floorPlaneResult.message}
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

          {activeTab === 'semantic' && <SemanticSegView />}

          {activeTab === 'histogram' && (
            <>
              <p className="floorplan-status">PCF evidence-confidence distribution</p>
              <canvas ref={histogramCanvasRef} className="histogram" />
            </>
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
