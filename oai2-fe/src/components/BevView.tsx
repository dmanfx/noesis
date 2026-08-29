import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CameraKey, cameraLabel, colorIdForPerson } from '../lib/camera';
import { FloorplanResponse } from './DepthDrawer';
import { renderLayerToCanvas, renderCompositeWalkableObstacleToCanvas, infernoColor } from '../lib/renderUtils';
import type { BevFrameMode as CoordFrameMode } from '../lib/coordTransforms';
import { isCameraLocalFrame, isWorldFrame } from '../lib/coordTransforms';
import {
  type BevTrailConfig,
  type TrailPoint,
  type TrailTrack,
  computeSceneUnitsPerPx,
  computeSegmentAlpha,
  computeTrailAgeAlpha,
  normalizeBevTrailConfig,
  pruneTrailCollection,
  upsertTrailSample,
} from '../lib/bevTrails';
import {
  resolveBevDisplayBounds,
  resolveBevRenderBounds,
  resolveBevMetricPoint,
} from '../lib/bevDisplayGeometry';
import {
  admitResolverDiagnostics,
  resolverDiagnosticDisplayPoint,
  type ResolverCandidate,
  type ResolverDiagnostics,
} from '../lib/bevResolverDiagnostics';

export type BevMeta = {
  type?: string;
  cameraId?: string;
  camId?: string;
  camera_id?: string;
  cam_id?: string;
  ts?: number;
  sourceId?: number;
  frameId?: number;
  observedAtUs?: number;
  trackingPublicationSequence?: number;
  trackingOutboundSubmissionId?: number;
  cohort?: {
    source_id?: number;
    frame_id?: number;
    observed_at_us?: number;
    tracking_publication_sequence?: number;
    tracking_outbound_submission_id?: number;
  };
  frame?: string;
  world_frame?: string;
  canonicalWorldFrame?: string;
  canonicalWorldFrameRevision?: string | null;
  canonicalWorldTransformSha256?: string | null;
  frame_mode?: string;
  units?: string;
  s_obj_to_m?: number;
  footpoints?: Array<{
    x: number;
    y: number;
    floorplanX?: number;
    floorplanZ?: number;
    normX?: number;
    normY?: number;
    normZ?: number;
    floorplanInside?: boolean;
    coverageInside?: boolean;
    coverageRegion?: string | null;
    gridCol?: number | null;
    gridRow?: number | null;
    gridCell?: number[];
    method: string;
    stableId?: number | null;
    trackerId?: number | null;
    trackerLifecycleGeneration?: number | null;
    anchorSource?: string | null;
    anchorQuality?: string | null;
    anchorReason?: string | null;
    displaySource?: string | null;
    canonicalWorld?: boolean;
    worldAdmission?: 'accepted' | 'predicted' | 'held' | null;
    rawX?: number;
    rawY?: number;
    alignmentDebug?: {
      candidates?: Array<{
        name?: string;
        rayFloor?: { x?: number; z?: number; normX?: number; normY?: number; insideBounds?: boolean; floorplanInside?: boolean };
        mapanythingDepth?: { x?: number; z?: number; normX?: number; normY?: number; insideBounds?: boolean; floorplanInside?: boolean };
        depthVsRayDeltaM?: number;
      }>;
      chosen?: { x?: number; z?: number; normX?: number; normY?: number; insideBounds?: boolean; floorplanInside?: boolean; displaySource?: string };
    };
    resolverDiagnostics?: ResolverDiagnostics;
  }>;
  trails?: Array<{
    stableId?: number | null;
    trackerId?: number | null;
    trackerLifecycleGeneration?: number | null;
    canonicalWorld?: boolean;
    points?: Array<{
      x: number;
      y: number;
      t: number;
      floorplanX?: number;
      floorplanZ?: number;
      normX?: number;
      normY?: number;
      normZ?: number;
      floorplanInside?: boolean;
      coverageInside?: boolean;
      coverageRegion?: string | null;
      gridCol?: number | null;
      gridRow?: number | null;
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
  floorplanCoordinateSpace?: string;
  floorplanBounds?: MetricBounds;
  displayBounds?: MetricBounds;
  coverageEnvelope?: {
    contract?: string;
    contractVersion?: number;
    frame?: string;
    units?: string;
    cameraId?: string;
    boundaryToleranceM?: number;
    bounds?: MetricBounds;
    regions?: Array<{
      id?: string;
      polygonXZ?: Array<[number, number]>;
    }>;
  };
  floorplanGridShape?: number[] | null;
  floorplanGridResM?: number | null;
  floorplanSnapshotId?: string | null;
  floorplanSnapshotContentSha256?: string | null;
  floorplanCalibrationFingerprint?: string | null;
  droppedFootpoints?: Array<{
    x?: number;
    y?: number;
    normX?: number;
    normY?: number;
    floorplanInside?: boolean;
    coverageInside?: boolean;
    coverageRegion?: string | null;
    stableId?: number | null;
    trackerId?: number | null;
    trackerLifecycleGeneration?: number | null;
    trailSegmentId?: number | null;
    reason?: string;
  }>;
  droppedFootpointCount?: number;
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
  /** Global dashboard capability state, synchronized by the WebSocket server. */
  resolverComparisonEnabled?: boolean;
  onResolverComparisonChange?: (enabled: boolean) => void;
  variant?: 'drawer' | 'inline';
};

const DEFAULT_X_MIN = -4;
const DEFAULT_X_MAX = 4;
const DEFAULT_Z_MIN = 0;
const DEFAULT_Z_MAX = 12;

type ContentRect = { x: number; y: number; w: number; h: number };
type MetricBounds = { min_x: number; max_x: number; min_z: number; max_z: number };
type ResolvedMetricPoint = { x: number; y: number; mapped: boolean };
type ProjectedMetricPoint = { px: number; py: number; clipped: boolean };
type NormalizedPayloadPoint = {
  x?: number;
  y?: number;
  normX?: number;
  normY?: number;
  floorplanInside?: boolean | null;
  coverageInside?: boolean | null;
  coverageRegion?: string | null;
  canonicalWorld?: boolean;
};
type CoverageRegion = { id: string; polygonXZ: Array<[number, number]> };
type CoverageEnvelope = {
  cameraId: string;
  boundaryToleranceM: number;
  bounds: MetricBounds;
  regions: CoverageRegion[];
};
type FloorplanVisualSelection = {
  walkableLayer?: FloorplanResponse['walkable'];
  obstacleHeightLayer?: FloorplanResponse['obstacle_height'];
  heightLayer?: FloorplanResponse['height'];
  densityLayer?: FloorplanResponse['density'];
  distanceLayer?: FloorplanResponse['distance'];
  observationMaskLayer?: FloorplanResponse['observed'];
  observationMaskInvert: boolean;
  planarFloorSupportLayer?: FloorplanResponse['scene_prior_floor_supported'];
  baseLayer?: FloorplanResponse['height'];
  baseKind: 'composite' | 'walkable' | 'obstacle_height' | 'height' | 'none';
  hasComposite: boolean;
  hasFloorplan: boolean;
};

const TRAIL_LINE_WIDTH = 2.5;
const TRAIL_HEAD_RADIUS = 4.5;
const TRAIL_CONNECTOR_MIN_PX = 2.0;

type TrailClock = {
  lastSrcMs?: number;
  lastSampleMs?: number;
};

type HeightRenderTuning = {
  lowPct: number;
  highPct: number;
  gamma: number;
  densityCutoff: number;
  smoothing: boolean;
};

// Keep source-clock smoothing from lagging visible motion by multiple seconds.
const MAX_TRAIL_CLOCK_SKEW_MS = 200;
const DEFAULT_HEIGHT_RENDER_TUNING: HeightRenderTuning = {
  lowPct: 5,
  highPct: 95,
  gamma: 1.0,
  densityCutoff: 0.0,
  smoothing: true,
};
const BEV_WALKABLE_MASK_THRESHOLD = 1e-6;
const BEV_UNKNOWN_CELL_COLOR: [number, number, number, number] = [16, 22, 32, 255];
const BEV_UNKNOWN_CELL_ALT_COLOR: [number, number, number, number] = [20, 28, 39, 255];
const BEV_PLANAR_FLOOR_EXPANSION_CELLS = 2;

const clampNumber = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

const floorplanHasRenderableGrid = (floorplan?: FloorplanResponse | null): boolean => Boolean(
  floorplan?.scene_prior_diagnostic_walkable?.grid_b64 ||
  floorplan?.scene_prior_diagnostic_obstacle_height?.grid_b64 ||
  floorplan?.scene_prior_diagnostic_height_agl?.grid_b64 ||
  floorplan?.scene_prior_diagnostic_density?.grid_b64 ||
  floorplan?.scene_prior_diagnostic_distance?.grid_b64 ||
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
  preferWalkableVisual = false
): FloorplanVisualSelection => {
  const canonicalPcf = floorplan?.scene_prior_only === true
    && floorplan?.display_source === 'pcf';
  const walkableLayer = canonicalPcf
    ? floorplan?.scene_prior_diagnostic_walkable
    : floorplan?.walkable;
  const obstacleHeightLayer = canonicalPcf
    ? floorplan?.scene_prior_diagnostic_obstacle_height
    : floorplan?.obstacle_height;
  const heightLayer = canonicalPcf
    ? floorplan?.scene_prior_diagnostic_height_agl
    : floorplan?.height;
  const densityLayer = canonicalPcf
    ? floorplan?.scene_prior_diagnostic_density
    : floorplan?.density;
  const distanceLayer = canonicalPcf
    ? floorplan?.scene_prior_diagnostic_distance
    : floorplan?.distance;
  const observedLayer = canonicalPcf
    ? floorplan?.scene_prior_diagnostic_observed
    : floorplan?.observed;
  const unknownLayer = canonicalPcf
    ? floorplan?.scene_prior_diagnostic_unknown
    : floorplan?.unknown;
  const planarFloorSupportLayer = canonicalPcf
    ? floorplan?.scene_prior_floor_supported
    : undefined;
  const hasWalkable = isCompatible && !!(walkableLayer?.grid_b64 && walkableLayer?.grid_shape);
  const hasObstacleHeight = isCompatible && !!(obstacleHeightLayer?.grid_b64 && obstacleHeightLayer?.grid_shape);
  const hasHeight = isCompatible && !!(heightLayer?.grid_b64 && heightLayer?.grid_shape);
  const hasComposite = hasWalkable && hasObstacleHeight;
  const useWalkableVisual = preferWalkableVisual && hasWalkable;
  const baseLayer = useWalkableVisual
    ? walkableLayer
    : (hasComposite
      ? walkableLayer
      : (hasWalkable ? walkableLayer : (hasObstacleHeight ? obstacleHeightLayer : heightLayer)));
  const baseKind = useWalkableVisual
    ? 'walkable'
    : (hasComposite
      ? 'composite'
      : (hasWalkable ? 'walkable' : (hasObstacleHeight ? 'obstacle_height' : (hasHeight ? 'height' : 'none'))));
  return {
    walkableLayer,
    obstacleHeightLayer,
    heightLayer,
    densityLayer,
    distanceLayer,
    observationMaskLayer: canonicalPcf
      ? (observedLayer ?? densityLayer)
      : (unknownLayer ?? observedLayer ?? densityLayer),
    observationMaskInvert: canonicalPcf ? false : Boolean(unknownLayer),
    planarFloorSupportLayer,
    baseLayer,
    baseKind,
    hasComposite,
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

const rawFloorplanBounds = (floorplan: FloorplanResponse | undefined): MetricBounds | null => {
  const bounds = floorplan?.bounds;
  if (!bounds) return null;
  const minX = Number(bounds.min_x);
  const maxX = Number(bounds.max_x);
  const minZ = Number(bounds.min_z);
  const maxZ = Number(bounds.max_z);
  if (![minX, maxX, minZ, maxZ].every(Number.isFinite)) return null;
  if (maxX <= minX || maxZ <= minZ) return null;
  return { min_x: minX, max_x: maxX, min_z: minZ, max_z: maxZ };
};

const rawPayloadBounds = (meta: BevMeta | undefined): MetricBounds | null => {
  const minX = Number(meta?.xMin);
  const maxX = Number(meta?.xMax);
  const minZ = Number(meta?.zMin);
  const maxZ = Number(meta?.zMax);
  if (![minX, maxX, minZ, maxZ].every(Number.isFinite)) return null;
  if (maxX <= minX || maxZ <= minZ) return null;
  return { min_x: minX, max_x: maxX, min_z: minZ, max_z: maxZ };
};

const parseMetricBounds = (raw: unknown): MetricBounds | null => {
  if (!raw || typeof raw !== 'object') return null;
  const value = raw as Partial<MetricBounds>;
  const minX = Number(value.min_x);
  const maxX = Number(value.max_x);
  const minZ = Number(value.min_z);
  const maxZ = Number(value.max_z);
  if (![minX, maxX, minZ, maxZ].every(Number.isFinite)) return null;
  if (maxX <= minX || maxZ <= minZ) return null;
  return { min_x: minX, max_x: maxX, min_z: minZ, max_z: maxZ };
};

const parseCoverageEnvelope = (meta: BevMeta | undefined): CoverageEnvelope | null => {
  const raw = meta?.coverageEnvelope;
  if (!raw || typeof raw !== 'object') return null;
  if (
    raw.contract !== 'noesis.bev.coverage_envelopes' ||
    raw.contractVersion !== 1 ||
    !isCameraLocalFrame(raw.frame) ||
    !isMetricUnits(raw.units)
  ) {
    return null;
  }
  const cameraId = String(raw.cameraId || '').trim();
  const metaCameraId = String(meta?.cameraId || meta?.camId || '').trim();
  if (!cameraId || (metaCameraId && cameraId !== metaCameraId)) return null;
  const boundaryToleranceM = Number(raw.boundaryToleranceM);
  if (
    !Number.isFinite(boundaryToleranceM) ||
    boundaryToleranceM < 0 ||
    boundaryToleranceM > 2
  ) {
    return null;
  }
  const bounds = parseMetricBounds(raw.bounds);
  if (!bounds || !Array.isArray(raw.regions) || raw.regions.length < 1 || raw.regions.length > 16) {
    return null;
  }
  const regions: CoverageRegion[] = [];
  const ids = new Set<string>();
  for (const rawRegion of raw.regions) {
    const id = String(rawRegion?.id || '').trim();
    const rawPolygon = rawRegion?.polygonXZ;
    if (!id || ids.has(id) || !Array.isArray(rawPolygon) || rawPolygon.length < 3 || rawPolygon.length > 64) {
      return null;
    }
    const polygonXZ: Array<[number, number]> = [];
    for (const rawPoint of rawPolygon) {
      if (!Array.isArray(rawPoint) || rawPoint.length !== 2) return null;
      const x = Number(rawPoint[0]);
      const z = Number(rawPoint[1]);
      if (!Number.isFinite(x) || !Number.isFinite(z)) return null;
      if (
        x < bounds.min_x - 1e-6 ||
        x > bounds.max_x + 1e-6 ||
        z < bounds.min_z - 1e-6 ||
        z > bounds.max_z + 1e-6
      ) {
        return null;
      }
      polygonXZ.push([x, z]);
    }
    ids.add(id);
    regions.push({ id, polygonXZ });
  }
  return { cameraId, boundaryToleranceM, bounds, regions };
};

const boundsNearlyEqual = (a: MetricBounds | null, b: MetricBounds | null, eps = 1e-3): boolean => {
  if (!a || !b) return false;
  return (
    Math.abs(a.min_x - b.min_x) <= eps &&
    Math.abs(a.max_x - b.max_x) <= eps &&
    Math.abs(a.min_z - b.min_z) <= eps &&
    Math.abs(a.max_z - b.max_z) <= eps
  );
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
  resolverComparisonEnabled: resolverComparisonEnabledProp,
  onResolverComparisonChange,
  variant = 'drawer'
}) => {
  const label = cameraLabel(cam);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [overlayEnabled, setOverlayEnabled] = useState(false);
  const [lookPanelOpen, setLookPanelOpen] = useState(false);
  const [heightRenderTuning, setHeightRenderTuning] = useState<HeightRenderTuning>(DEFAULT_HEIGHT_RENDER_TUNING);
  const [floorPlaneEnabled, setFloorPlaneEnabled] = useState(true);
  // The resolver comparison is deliberately presentation-only.  Query-param
  // debug remains a useful initial override for development, while a normal
  // dashboard user can toggle the bounded overlay without changing runtime
  // estimation, publication cadence, or trails.
  const [localResolverComparisonEnabled, setLocalResolverComparisonEnabled] = useState(() => Boolean(debug));
  const resolverComparisonEnabled = resolverComparisonEnabledProp ?? localResolverComparisonEnabled;
  const setResolverComparisonEnabled = (enabled: boolean) => {
    if (resolverComparisonEnabledProp !== undefined) {
      onResolverComparisonChange?.(enabled);
      return;
    }
    setLocalResolverComparisonEnabled(enabled);
  };

  const smoothState = useRef<Map<string, { x: number; y: number; lastSeen: number; stableId?: string; colorId: number; worldAdmission?: string }>>(new Map());
  const trailsRef = useRef<Map<string, TrailTrack>>(new Map());
  const animationFrameRef = useRef<number>();
  const bgCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const bgKeyRef = useRef<string>('');
  const bgSizeRef = useRef<{ w: number; h: number }>({ w: 0, h: 0 });
  const bgContentRectRef = useRef<ContentRect | null>(null);
  const historyKeyForPoint = (
    stableId?: number | null,
    trackerId?: number | null,
    trackerLifecycleGeneration?: number | null
  ): string | null => {
    const trackerNum = typeof trackerId === 'number' && Number.isFinite(trackerId) ? trackerId : null;
    const generationNum = typeof trackerLifecycleGeneration === 'number' && Number.isFinite(trackerLifecycleGeneration)
      ? trackerLifecycleGeneration
      : null;
    if (trackerNum !== null && trackerNum >= 0) {
      return generationNum !== null && generationNum >= 0
        ? `t:${trackerNum}:g:${generationNum}`
        : `t:${trackerNum}`;
    }
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
  const coverageEnvelopeContractKey = JSON.stringify(meta?.coverageEnvelope ?? null);
  const coverageEnvelope = useMemo(
    () => parseCoverageEnvelope(meta),
    [coverageEnvelopeContractKey, meta?.cameraId, meta?.camId]
  );
  const displayBounds = useMemo(
    () => {
      const floorplanBounds = floorplanBoundsForMode(displayFloorplan, coordMode);
      const advertised = parseMetricBounds(meta?.displayBounds) ?? rawPayloadBounds(meta);
      return resolveBevDisplayBounds({
        floorplanBounds,
        advertisedBounds: advertised,
        coverageBounds: coverageEnvelope?.bounds,
        coverageToleranceM: coverageEnvelope?.boundaryToleranceM,
      });
    },
    [
      coordMode,
      coverageEnvelope,
      displayFloorplan?.frame,
      displayFloorplan?.units,
      displayFloorplan?.bounds?.min_x,
      displayFloorplan?.bounds?.max_x,
      displayFloorplan?.bounds?.min_z,
      displayFloorplan?.bounds?.max_z,
      meta?.displayBounds,
      meta?.xMin,
      meta?.xMax,
      meta?.zMin,
      meta?.zMax,
    ]
  );
  const resolveDisplayPoint = useCallback((x: number, y: number): ResolvedMetricPoint | null => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    if (!displayBounds) return { x, y, mapped: false };

    if (x < displayBounds.min_x || x > displayBounds.max_x || y < displayBounds.min_z || y > displayBounds.max_z) {
      return null;
    }
    return { x, y, mapped: false };
  }, [displayBounds]);
  const normalizedPayloadBounds = useMemo(
    () => parseMetricBounds(meta?.floorplanBounds)
      ?? floorplanBoundsForMode(displayFloorplan, coordMode)
      ?? displayBounds,
    [
      coordMode,
      displayBounds,
      displayFloorplan?.frame,
      displayFloorplan?.units,
      displayFloorplan?.bounds?.min_x,
      displayFloorplan?.bounds?.max_x,
      displayFloorplan?.bounds?.min_z,
      displayFloorplan?.bounds?.max_z,
      meta?.floorplanBounds,
    ]
  );

  const resolvePayloadPoint = useCallback((pt: NormalizedPayloadPoint | null | undefined): ResolvedMetricPoint | null => {
    const resolved = resolveBevMetricPoint({
      point: pt,
      displayBounds,
      normalizedBounds: normalizedPayloadBounds,
      coverage: coverageEnvelope,
    });
    if (!resolved) return null;
    return resolveDisplayPoint(resolved.x, resolved.y)
      ? resolved
      : null;
  }, [coverageEnvelope, displayBounds, normalizedPayloadBounds, resolveDisplayPoint]);

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
    const canonicalPcf = displayFloorplan?.scene_prior_only === true
      && displayFloorplan?.display_source === 'pcf';
    const spaceWalkable = canonicalPcf
      ? displayFloorplan?.scene_prior_diagnostic_walkable
      : displayFloorplan?.walkable;
    const spaceObstacle = canonicalPcf
      ? displayFloorplan?.scene_prior_diagnostic_obstacle_height
      : displayFloorplan?.obstacle_height;
    const spaceHeight = canonicalPcf
      ? displayFloorplan?.scene_prior_diagnostic_height_agl
      : displayFloorplan?.height;
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
      String(displayFloorplan?.snapshot_id || ''),
      String(displayFloorplan?.snapshot_content_sha256 || ''),
      String(displayFloorplan?.calibration_fingerprint || ''),
      boundsKey,
      String(displayFloorplan?.snapshot_ts ?? displayFloorplan?.ts ?? ''),
      String(spaceWalkable?.grid_b64?.length ?? ''),
      String(spaceObstacle?.grid_b64?.length ?? ''),
      String(spaceHeight?.grid_b64?.length ?? ''),
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
    displayFloorplan?.snapshot_id,
    displayFloorplan?.snapshot_content_sha256,
    displayFloorplan?.calibration_fingerprint,
    displayFloorplan?.bounds?.min_x,
    displayFloorplan?.bounds?.max_x,
    displayFloorplan?.bounds?.min_z,
    displayFloorplan?.bounds?.max_z,
    displayFloorplan?.snapshot_ts,
    displayFloorplan?.ts,
    displayFloorplan?.walkable?.grid_b64,
    displayFloorplan?.obstacle_height?.grid_b64,
    displayFloorplan?.height?.grid_b64,
    displayFloorplan?.scene_prior_only,
    displayFloorplan?.display_source,
    displayFloorplan?.scene_prior_diagnostic_walkable?.grid_b64,
    displayFloorplan?.scene_prior_diagnostic_obstacle_height?.grid_b64,
    displayFloorplan?.scene_prior_diagnostic_height_agl?.grid_b64,
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
    const purgePriorTrackerGenerations = (trackerId: number | null, keepKey: string) => {
      if (trackerId === null || trackerId < 0) return;
      const prefix = `t:${trackerId}:g:`;
      for (const key of state.keys()) {
        if (key !== keepKey && key.startsWith(prefix)) state.delete(key);
      }
      for (const key of trails.keys()) {
        if (key !== keepKey && key.startsWith(prefix)) trails.delete(key);
      }
    };
    const dropUnresolvedPoint = (pt: typeof points[number]) => {
      const stableNum = typeof pt?.stableId === 'number' && Number.isFinite(pt.stableId) ? pt.stableId : null;
      const trackerNum = typeof pt?.trackerId === 'number' && Number.isFinite(pt.trackerId) ? pt.trackerId : null;
      const historyKey = historyKeyForPoint(stableNum, trackerNum, pt?.trackerLifecycleGeneration);
      if (historyKey === null) return;
      state.delete(historyKey);
      trails.delete(historyKey);
    };

    if (useBackendTrails) {
      trails.clear();
      const seenIds = new Set<string>();
      points.forEach(pt => {
        const resolved = resolvePayloadPoint(pt);
        if (!resolved) {
          dropUnresolvedPoint(pt);
          return;
        }
        const stableNum = typeof pt.stableId === 'number' && Number.isFinite(pt.stableId) ? pt.stableId : null;
        const trackerNum = typeof pt.trackerId === 'number' && Number.isFinite(pt.trackerId) ? pt.trackerId : null;
        const historyKey = historyKeyForPoint(stableNum, trackerNum, pt?.trackerLifecycleGeneration);
        const displayId = displayIdForPoint(stableNum, trackerNum);
        if (historyKey === null || displayId === null) return;
        purgePriorTrackerGenerations(trackerNum, historyKey);
        seenIds.add(historyKey);
        state.set(historyKey, {
          x: resolved.x,
          y: resolved.y,
          lastSeen: arrivalNow,
          stableId: `${displayId}`,
          colorId: colorIdForPerson(cam, displayId),
          worldAdmission: typeof pt.worldAdmission === 'string' ? pt.worldAdmission : undefined,
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
        const resolved = resolvePayloadPoint(pt);
        if (!resolved) {
          dropUnresolvedPoint(pt);
          return;
        }
        const targetX = resolved.x;
        const targetY = resolved.y;

        const stableNum = typeof pt.stableId === 'number' && Number.isFinite(pt.stableId) ? pt.stableId : null;
        const trackerNum = typeof pt.trackerId === 'number' && Number.isFinite(pt.trackerId) ? pt.trackerId : null;
        const historyKey = historyKeyForPoint(stableNum, trackerNum, pt?.trackerLifecycleGeneration);
        const displayId = displayIdForPoint(stableNum, trackerNum);
        if (historyKey === null || displayId === null) return;
        purgePriorTrackerGenerations(trackerNum, historyKey);
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
          worldAdmission: typeof pt.worldAdmission === 'string' ? pt.worldAdmission : undefined,
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
  }, [cam, meta, resolvePayloadPoint, resolvedTrailConfig]);

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
      const floorplanMetricBounds = rawFloorplanBounds(floorplanNow);
      const renderBounds = resolveBevRenderBounds({
        floorplanBounds: isFloorplanCompatible && floorplanHasRenderableGrid(floorplanNow)
          ? floorplanMetricBounds
          : null,
        displayBounds,
        coverageBounds: coverageEnvelope?.bounds,
        coverageToleranceM: coverageEnvelope?.boundaryToleranceM,
      });

      let xMin = DEFAULT_X_MIN;
      let xMax = DEFAULT_X_MAX;
      let zMin = DEFAULT_Z_MIN;
      let zMax = DEFAULT_Z_MAX;

      if (renderBounds) {
        xMin = renderBounds.min_x;
        xMax = renderBounds.max_x;
        zMin = renderBounds.min_z;
        zMax = renderBounds.max_z;
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
      const densityLayer = visual.densityLayer;
      const observationMaskLayer = visual.observationMaskLayer;
      const observationMaskInvert = visual.observationMaskInvert;
      const planarFloorSupportLayer = visual.planarFloorSupportLayer;
      const baseLayer = visual.baseLayer;
      const baseKind = visual.baseKind;
      const hasComposite = visual.hasComposite;
      const basePalette = infernoColor;
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

      const isHeightVisual = baseKind === 'height';
      const isWalkableVisual = baseKind === 'walkable';
      const smoothBaseImage = hasComposite || baseKind === 'obstacle_height' || isHeightVisual || isWalkableVisual;
      const heightLowPct = clampNumber(heightRenderTuning.lowPct, 0, Math.min(99, heightRenderTuning.highPct - 1));
      const heightHighPct = clampNumber(heightRenderTuning.highPct, Math.max(1, heightLowPct + 1), 100);
      const heightGamma = clampNumber(heightRenderTuning.gamma, 0.25, 3.0);
      const densityCutoff = clampNumber(heightRenderTuning.densityCutoff, 0.0, 0.5);
      const layerRenderKey = isHeightVisual
        ? `${heightLowPct}:${heightHighPct}:${heightGamma.toFixed(3)}:${densityCutoff.toFixed(4)}:${heightRenderTuning.smoothing ? 1 : 0}`
        : (isWalkableVisual
          ? `inferno-walkable:${observationMaskInvert ? 'invert' : 'normal'}:${observationMaskLayer?.grid_b64?.length ?? ''}`
          : 'layer');
      const walkableRenderOptions = isWalkableVisual
        ? {
            maskLayer: observationMaskLayer,
            maskThreshold: BEV_WALKABLE_MASK_THRESHOLD,
            maskInvert: observationMaskInvert,
            unknownColor: BEV_UNKNOWN_CELL_COLOR,
            unknownAltColor: BEV_UNKNOWN_CELL_ALT_COLOR,
            planarFloorSupportLayer,
            planarFloorExpansionCells: BEV_PLANAR_FLOOR_EXPANSION_CELLS,
            showPlanarFloor: floorPlaneEnabled,
          }
        : {};
      const coverageRenderKey = coverageEnvelope
        ? `${coverageEnvelope.boundaryToleranceM}:${coverageEnvelope.regions
          .map(region => `${region.id}:${region.polygonXZ.map(point => point.join(',')).join(';')}`)
          .join('|')}`
        : 'none';
      const metricBoundsKey = floorplanMetricBounds
        ? `${floorplanMetricBounds.min_x}:${floorplanMetricBounds.max_x}:${floorplanMetricBounds.min_z}:${floorplanMetricBounds.max_z}`
        : 'none';
      const displayBoundsKey = `${xMin}:${xMax}:${zMin}:${zMax}`;
      const planarFloorKey = planarFloorSupportLayer?.grid_b64
        ? `${floorPlaneEnabled ? 'visible' : 'hidden'}:${planarFloorSupportLayer.grid_b64.length}:${planarFloorSupportLayer.grid_b64.slice(0, 16)}:${planarFloorSupportLayer.grid_b64.slice(-16)}`
        : 'none';
      const key = hasFloorplan
        ? `${baseKind}:${floorplanNow?.snapshot_id ?? ''}:${floorplanNow?.snapshot_content_sha256 ?? ''}:${floorplanNow?.calibration_fingerprint ?? ''}:${floorplanNow?.snapshot_ts ?? floorplanNow?.ts ?? ''}:${baseLayer?.grid_shape?.join('x')}:${baseLayer?.value_min ?? ''}:${baseLayer?.value_max ?? ''}:${baseLayer?.grid_b64?.length ?? ''}:${hasComposite ? (obstacleHeightLayer?.grid_b64?.length ?? '') : ''}:${isHeightVisual ? (densityLayer?.grid_b64?.length ?? '') : ''}:${aspect}:${fitMode}:${boundsAspect.toFixed(6)}:${padCss.toFixed(3)}:${smoothBaseImage ? 'smooth' : 'sharp'}:${layerRenderKey}:${metricBoundsKey}:${displayBoundsKey}:${coverageRenderKey}:${planarFloorKey}`
        : `none:${aspect}:${fitMode}:${boundsAspect.toFixed(6)}:${padCss.toFixed(3)}:${metricBoundsKey}:${displayBoundsKey}:${coverageRenderKey}`;

      const bg = bgCanvasRef.current ?? (bgCanvasRef.current = document.createElement('canvas'));
      const bgSize = bgSizeRef.current;
      const bgNeedsRedraw = bgKeyRef.current !== key || bgSize.w !== expectedW || bgSize.h !== expectedH;

      if (bgNeedsRedraw) {
        const placeFloorplanInMetricBounds = Boolean(
          hasFloorplan
          && floorplanMetricBounds
          && renderBounds
          && !boundsNearlyEqual(floorplanMetricBounds, renderBounds)
        );
        if (coverageEnvelope || placeFloorplanInMetricBounds) {
          cvs.width = expectedW;
          cvs.height = expectedH;
          ctx.setTransform(1, 0, 0, 1, 0, 0);
          ctx.fillStyle = '#111';
          ctx.fillRect(0, 0, cvs.width, cvs.height);

          const canvasAspect = cvs.width / Math.max(1, cvs.height);
          let contentW = cvs.width;
          let contentH = cvs.height;
          let contentX = 0;
          let contentY = 0;
          if (canvasAspect > boundsAspect) {
            contentW = contentH * boundsAspect;
            contentX = (cvs.width - contentW) * 0.5;
          } else {
            contentH = contentW / boundsAspect;
            contentY = (cvs.height - contentH) * 0.5;
          }
          const coverageContentRect: ContentRect = {
            x: contentX,
            y: contentY,
            w: contentW,
            h: contentH,
          };
          bgContentRectRef.current = coverageContentRect;

          // The margin is deliberately rendered as unknown space. It makes the
          // authored PCF grid boundary visible without pretending the room was
          // measured outside that grid.
          const checkerSize = Math.max(8, Math.round(10 * dpr));
          ctx.save();
          ctx.beginPath();
          ctx.rect(
            coverageContentRect.x,
            coverageContentRect.y,
            coverageContentRect.w,
            coverageContentRect.h,
          );
          ctx.clip();
          for (
            let checkerY = coverageContentRect.y;
            checkerY < coverageContentRect.y + coverageContentRect.h;
            checkerY += checkerSize
          ) {
            for (
              let checkerX = coverageContentRect.x;
              checkerX < coverageContentRect.x + coverageContentRect.w;
              checkerX += checkerSize
            ) {
              const checkerCol = Math.floor((checkerX - coverageContentRect.x) / checkerSize);
              const checkerRow = Math.floor((checkerY - coverageContentRect.y) / checkerSize);
              const color = ((checkerRow + checkerCol) & 1)
                ? BEV_UNKNOWN_CELL_ALT_COLOR
                : BEV_UNKNOWN_CELL_COLOR;
              ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${(color[3] ?? 255) / 255})`;
              ctx.fillRect(checkerX, checkerY, checkerSize, checkerSize);
            }
          }
          ctx.restore();

          if (hasFloorplan && floorplanMetricBounds) {
            const floorplanSpanX = floorplanMetricBounds.max_x - floorplanMetricBounds.min_x;
            const floorplanSpanZ = floorplanMetricBounds.max_z - floorplanMetricBounds.min_z;
            const floorplanAspect = floorplanSpanX / floorplanSpanZ;
            const destinationX = coverageContentRect.x
              + (((floorplanMetricBounds.min_x - xMin) / boundsSpanX) * coverageContentRect.w);
            const destinationY = coverageContentRect.y
              + coverageContentRect.h
              - (((floorplanMetricBounds.max_z - zMin) / boundsSpanZ) * coverageContentRect.h);
            const destinationW = (floorplanSpanX / boundsSpanX) * coverageContentRect.w;
            const destinationH = (floorplanSpanZ / boundsSpanZ) * coverageContentRect.h;
            const floorplanCanvas = document.createElement('canvas');
            const renderOptions = {
              fit: 'contain' as const,
              forceAspect: floorplanAspect,
              contentPaddingPx: 0,
              targetWidthPx: Math.max(1, Math.round(destinationW)),
              targetHeightPx: Math.max(1, Math.round(destinationH)),
              pixelRatio: 1,
            };
            if (hasComposite) {
              renderCompositeWalkableObstacleToCanvas(
                floorplanCanvas,
                walkableLayer,
                obstacleHeightLayer,
                {
                  ...renderOptions,
                  imageSmoothing: true,
                  ...walkableRenderOptions,
                }
              );
            } else {
              renderLayerToCanvas(
                floorplanCanvas,
                baseLayer,
                basePalette,
                {
                  ...renderOptions,
                  imageSmoothing: isHeightVisual ? heightRenderTuning.smoothing : smoothBaseImage,
                  ...walkableRenderOptions,
                  ...(isHeightVisual ? {
                    valueMinPercentile: heightLowPct,
                    valueMaxPercentile: heightHighPct,
                    gamma: heightGamma,
                    maskLayer: densityLayer,
                    maskThreshold: densityCutoff,
                  } : {})
                }
              );
            }
            ctx.save();
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            ctx.drawImage(
              floorplanCanvas,
              destinationX,
              destinationY,
              destinationW,
              destinationH
            );
            ctx.restore();
          }

          if (coverageEnvelope) {
            ctx.save();
            ctx.lineWidth = Math.max(1.5, dpr);
            ctx.setLineDash([6 * dpr, 4 * dpr]);
            ctx.strokeStyle = 'rgba(77, 220, 255, 0.78)';
            ctx.fillStyle = 'rgba(77, 220, 255, 0.85)';
            ctx.font = `${Math.max(10, 10 * dpr)}px sans-serif`;
            for (const region of coverageEnvelope.regions) {
              ctx.beginPath();
              region.polygonXZ.forEach(([mx, mz], index) => {
                const px = coverageContentRect.x + (((mx - xMin) / boundsSpanX) * coverageContentRect.w);
                const py = coverageContentRect.y + coverageContentRect.h
                  - (((mz - zMin) / boundsSpanZ) * coverageContentRect.h);
                if (index === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
              });
              ctx.closePath();
              ctx.stroke();
              const centroidX = region.polygonXZ.reduce((sum, point) => sum + point[0], 0) / region.polygonXZ.length;
              const centroidZ = region.polygonXZ.reduce((sum, point) => sum + point[1], 0) / region.polygonXZ.length;
              const labelX = coverageContentRect.x + (((centroidX - xMin) / boundsSpanX) * coverageContentRect.w);
              const labelY = coverageContentRect.y + coverageContentRect.h
                - (((centroidZ - zMin) / boundsSpanZ) * coverageContentRect.h);
              ctx.fillText(region.id, labelX + (4 * dpr), labelY - (4 * dpr));
            }
            ctx.restore();
          }
        } else if (hasFloorplan) {
          const rendered = hasComposite
            ? renderCompositeWalkableObstacleToCanvas(cvs, walkableLayer, obstacleHeightLayer, {
              fit: fitMode,
              forceAspect: boundsAspect,
              contentPaddingPx: padCss,
              imageSmoothing: true,
              ...walkableRenderOptions,
            })
            : renderLayerToCanvas(cvs, baseLayer, basePalette, {
              fit: fitMode,
              forceAspect: boundsAspect,
              contentPaddingPx: padCss,
              imageSmoothing: isHeightVisual ? heightRenderTuning.smoothing : smoothBaseImage,
              ...walkableRenderOptions,
              ...(isHeightVisual ? {
                valueMinPercentile: heightLowPct,
                valueMaxPercentile: heightHighPct,
                gamma: heightGamma,
                maskLayer: densityLayer,
                maskThreshold: densityCutoff,
              } : {})
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
      const activeDrawBounds: MetricBounds = { min_x: xMin, max_x: xMax, min_z: zMin, max_z: zMax };
      const floorplanLayerBounds = rawFloorplanBounds(floorplanNow);
      const payloadLayerBounds = rawPayloadBounds(metaNow);

      const projectForBounds = (bounds: MetricBounds, mx: number, mz: number, clamp = false): ProjectedMetricPoint | null => {
        const spanX = bounds.max_x - bounds.min_x;
        const spanZ = bounds.max_z - bounds.min_z;
        if (!Number.isFinite(spanX) || !Number.isFinite(spanZ) || spanX <= 0 || spanZ <= 0) return null;
        const nxRaw = (mx - bounds.min_x) / spanX;
        const nzRaw = (mz - bounds.min_z) / spanZ;
        if (!clamp && (nxRaw < 0 || nxRaw > 1 || nzRaw < 0 || nzRaw > 1)) return null;
        const nx = clampNumber(nxRaw, 0, 1);
        const nz = clampNumber(nzRaw, 0, 1);
        return {
          px: contentRect.x + nx * contentRect.w,
          py: contentRect.y + contentRect.h - nz * contentRect.h,
          clipped: nx !== nxRaw || nz !== nzRaw,
        };
      };

      const drawArrow = (x0: number, y0: number, x1: number, y1: number, color: string, labelText: string) => {
        const angle = Math.atan2(y1 - y0, x1 - x0);
        const head = 5;
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - head * Math.cos(angle - Math.PI / 6), y1 - head * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(x1 - head * Math.cos(angle + Math.PI / 6), y1 - head * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
        ctx.font = 'bold 9px sans-serif';
        ctx.lineWidth = 3;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.72)';
        ctx.strokeText(labelText, x1 + 3, y1 - 3);
        ctx.fillStyle = color;
        ctx.fillText(labelText, x1 + 3, y1 - 3);
      };

      const drawOriginLayer = (
        bounds: MetricBounds | null,
        labelText: string,
        legendText: string,
        color: string,
        labelOffsetY: number,
        dash: number[]
      ): boolean => {
        if (!bounds) return false;
        const origin = projectForBounds(bounds, 0, 0, true);
        if (!origin) return false;
        const xAxis = projectForBounds(bounds, 0, 0, false);

        ctx.save();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.setLineDash(dash);
        ctx.globalAlpha = 0.78;
        ctx.beginPath();
        if (bounds.min_z <= 0 && bounds.max_z >= 0) {
          const z0 = projectForBounds(bounds, bounds.min_x, 0, false);
          const z1 = projectForBounds(bounds, bounds.max_x, 0, false);
          if (z0 && z1) {
            ctx.moveTo(z0.px, z0.py);
            ctx.lineTo(z1.px, z1.py);
          }
        }
        if (bounds.min_x <= 0 && bounds.max_x >= 0) {
          const x0 = projectForBounds(bounds, 0, bounds.min_z, false);
          const x1 = projectForBounds(bounds, 0, bounds.max_z, false);
          if (x0 && x1) {
            ctx.moveTo(x0.px, x0.py);
            ctx.lineTo(x1.px, x1.py);
          }
        }
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;

        const markerSize = origin.clipped ? 6 : 5;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.72)';
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(origin.px, origin.py - markerSize);
        ctx.lineTo(origin.px + markerSize, origin.py);
        ctx.lineTo(origin.px, origin.py + markerSize);
        ctx.lineTo(origin.px - markerSize, origin.py);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        if (xAxis && !origin.clipped) {
          const arrowLen = 28;
          const xEnd = Math.min(contentRect.x + contentRect.w - 18, origin.px + arrowLen);
          const zEnd = Math.max(contentRect.y + 18, origin.py - arrowLen);
          if (xEnd > origin.px + 8) drawArrow(origin.px, origin.py, xEnd, origin.py, color, '+X');
          if (zEnd < origin.py - 8) drawArrow(origin.px, origin.py, origin.px, zEnd, color, '+Z');
        }

        const label = `${labelText}${origin.clipped ? ' off' : ''}`;
        const labelX = clampNumber(origin.px + 8, contentRect.x + 4, contentRect.x + contentRect.w - 86);
        const labelY = clampNumber(origin.py + labelOffsetY, contentRect.y + 12, contentRect.y + contentRect.h - 5);
        ctx.font = 'bold 10px sans-serif';
        ctx.lineWidth = 3;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.78)';
        ctx.strokeText(label, labelX, labelY);
        ctx.fillStyle = color;
        ctx.fillText(label, labelX, labelY);

        ctx.restore();
        return Boolean(legendText);
      };

      const drawOriginLegend = (rows: Array<{ text: string; color: string }>) => {
        if (!rows.length) return;
        const pad = 5;
        const lineH = 12;
        const legendW = 128;
        const legendH = pad * 2 + rows.length * lineH;
        const legendX = contentRect.x + 8;
        const legendY = contentRect.y + contentRect.h - legendH - 8;
        ctx.save();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.66)';
        ctx.fillRect(legendX, legendY, legendW, legendH);
        ctx.font = '10px monospace';
        rows.forEach((row, idx) => {
          const y = legendY + pad + ((idx + 1) * lineH) - 3;
          ctx.fillStyle = row.color;
          ctx.fillText(row.text, legendX + pad, y);
        });
        ctx.restore();
      };

      const resolverCandidateColor = (candidate: ResolverCandidate): string => {
        switch (String(candidate.kind || '').toLowerCase()) {
          case 'floor_ray': return 'rgba(77, 220, 255, 0.95)';
          case 'registered_depth': return 'rgba(255, 119, 95, 0.95)';
          case 'pose_scale': return 'rgba(210, 153, 255, 0.95)';
          case 'gravity_reconstruction': return 'rgba(168, 230, 106, 0.95)';
          default: return 'rgba(255, 215, 64, 0.95)';
        }
      };

      const drawResolverCovariance = (
        point: { x: number; z: number },
        covariance: [[number, number], [number, number]] | undefined,
        color: string,
      ) => {
        if (!covariance) return;
        const projected = projectForBounds(activeDrawBounds, point.x, point.z, false);
        if (!projected) return;
        const a = Number(covariance[0][0]);
        const b = Number(covariance[0][1]);
        const d = Number(covariance[1][1]);
        if (![a, b, d].every(Number.isFinite)) return;
        const halfTrace = (a + d) * 0.5;
        const discriminant = Math.sqrt(Math.max(0, ((a - d) * 0.5) ** 2 + b ** 2));
        const lambdaMajor = Math.max(0, halfTrace + discriminant);
        const lambdaMinor = Math.max(0, halfTrace - discriminant);
        let vx = 1;
        let vz = 0;
        if (Math.abs(b) > 1e-8) {
          vx = lambdaMajor - d;
          vz = b;
        } else if (d > a) {
          vx = 0;
          vz = 1;
        }
        const vectorLength = Math.hypot(vx, vz) || 1;
        vx /= vectorLength;
        vz /= vectorLength;
        const pxPerX = contentRect.w / Math.max(1e-6, xMax - xMin);
        const pxPerZ = contentRect.h / Math.max(1e-6, zMax - zMin);
        const radiusMajor = clampNumber(Math.sqrt(lambdaMajor) * 2 * Math.max(pxPerX, pxPerZ), 2, 72);
        const radiusMinor = clampNumber(Math.sqrt(lambdaMinor) * 2 * Math.min(pxPerX, pxPerZ), 2, 72);
        if (!Number.isFinite(radiusMajor) || !Number.isFinite(radiusMinor)) return;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.52;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.ellipse(
          projected.px,
          projected.py,
          radiusMajor,
          radiusMinor,
          Math.atan2(-vz * pxPerZ, vx * pxPerX),
          0,
          2 * Math.PI,
        );
        ctx.stroke();
        ctx.restore();
      };

      const drawResolverComparison = () => {
        if (!resolverComparisonEnabled || !metaNow) return;
        let drawn = false;
        const legendRows: Array<{ text: string; color: string }> = [
          { text: 'solid canonical', color: 'rgba(255,255,255,0.95)' },
          { text: 'candidate', color: 'rgba(77,220,255,0.95)' },
          { text: 'legacy', color: 'rgba(255,255,255,0.9)' },
          { text: '2σ covariance', color: 'rgba(168,230,106,0.9)' },
        ];

        for (const point of footpoints) {
          const diagnostics = admitResolverDiagnostics(point, metaNow);
          if (!diagnostics) continue;
          const canonical = resolvePayloadPoint(point);
          // The comparison layer may only decorate an already-rendered
          // canonical point.  It must never manufacture a dot from a
          // diagnostic when the canonical BEV point is absent.
          const canonicalPoint = canonical;
          if (!canonicalPoint) continue;
          const canonicalProjection = projectForBounds(activeDrawBounds, canonicalPoint.x, canonicalPoint.y, false);
          if (!canonicalProjection) continue;
          drawn = true;

          const selectedId = diagnostics.selectedId;
          const selectedKind = diagnostics.selectedKind;
          const selectedCandidate = (diagnostics.candidates ?? []).find((candidate) => (
            candidate.selected === true
            || (selectedId !== null && candidate.id === selectedId)
          ));

          ctx.save();
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
          ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(canonicalProjection.px, canonicalProjection.py, 7, 0, 2 * Math.PI);
          ctx.fill();
          ctx.stroke();
          ctx.restore();

          for (const candidate of diagnostics.candidates ?? []) {
            const candidatePoint = resolverDiagnosticDisplayPoint(candidate);
            if (!candidatePoint) continue;
            const candidateProjection = projectForBounds(activeDrawBounds, candidatePoint.x, candidatePoint.z, false);
            if (!candidateProjection) continue;
            const color = resolverCandidateColor(candidate);
            const isSelected = Boolean(candidate.selected)
              || (selectedId !== null && candidate.id === selectedId)
              || (selectedId === null && selectedKind !== null && candidate.kind === selectedKind);
            ctx.save();
            ctx.strokeStyle = color;
            ctx.fillStyle = isSelected ? color : 'rgba(0,0,0,0.12)';
            ctx.lineWidth = isSelected ? 2 : 1.5;
            ctx.beginPath();
            ctx.arc(candidateProjection.px, candidateProjection.py, isSelected ? 5 : 4, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
            if (candidateProjection.px !== canonicalProjection.px || candidateProjection.py !== canonicalProjection.py) {
              ctx.strokeStyle = color;
              ctx.globalAlpha = 0.5;
              ctx.lineWidth = 1;
              ctx.setLineDash([3, 3]);
              ctx.beginPath();
              ctx.moveTo(candidateProjection.px, candidateProjection.py);
              ctx.lineTo(canonicalProjection.px, canonicalProjection.py);
              ctx.stroke();
            }
            ctx.restore();
            drawResolverCovariance(candidatePoint, candidate.covarianceXZ, color);
          }

          const legacyPoint = resolverDiagnosticDisplayPoint(diagnostics.legacy);
          if (legacyPoint) {
            const legacyProjection = projectForBounds(activeDrawBounds, legacyPoint.x, legacyPoint.z, false);
            if (legacyProjection) {
              ctx.save();
              ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
              ctx.lineWidth = 2;
              ctx.setLineDash([5, 3]);
              ctx.beginPath();
              ctx.arc(legacyProjection.px, legacyProjection.py, 8, 0, 2 * Math.PI);
              ctx.stroke();
              ctx.beginPath();
              ctx.moveTo(legacyProjection.px, legacyProjection.py);
              ctx.lineTo(canonicalProjection.px, canonicalProjection.py);
              ctx.stroke();
              ctx.restore();
            }
          }

          const resolvedDiagnostic = resolverDiagnosticDisplayPoint(diagnostics.resolved);
          if (resolvedDiagnostic) {
            drawResolverCovariance(resolvedDiagnostic, diagnostics.resolved?.covarianceXZ, 'rgba(168, 230, 106, 0.95)');
          }

          const pcf = selectedCandidate?.pcf;
          const pcfText = pcf
            ? `pcf=${pcf.insideExtent === false && pcf.extentOutsideDistanceM !== undefined
              ? `extent+${pcf.extentOutsideDistanceM.toFixed(2)}m`
              : pcf.insideAuthoredSpace === false
                ? 'outside'
                : pcf.observedConfidence !== undefined
                  ? pcf.observedConfidence.toFixed(2)
                  : 'ok'}`
            : null;
          const labelParts = [
            selectedKind || selectedId,
            diagnostics.decision || diagnostics.reason,
            diagnostics.disagreement?.distanceM !== undefined
              ? `Δ=${diagnostics.disagreement.distanceM.toFixed(2)}m`
              : null,
            pcfText,
          ].filter((value): value is string => Boolean(value));
          if (labelParts.length) {
            const labelText = labelParts.join(' • ').slice(0, 120);
            ctx.save();
            ctx.font = '10px monospace';
            const labelWidth = Math.min(260, ctx.measureText(labelText).width + 10);
            const labelX = clampNumber(canonicalProjection.px + 10, contentRect.x + 3, contentRect.x + contentRect.w - labelWidth - 3);
            const labelY = clampNumber(canonicalProjection.py - 10, contentRect.y + 13, contentRect.y + contentRect.h - 3);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.74)';
            ctx.fillRect(labelX, labelY - 11, labelWidth, 14);
            ctx.fillStyle = '#f3f6ff';
            ctx.fillText(labelText, labelX + 5, labelY - 1);
            ctx.restore();
          }
        }

        if (drawn) {
          const pad = 5;
          const lineH = 12;
          const legendW = 122;
          const legendH = pad * 2 + legendRows.length * lineH;
          const legendX = contentRect.x + contentRect.w - legendW - 8;
          const legendY = contentRect.y + 8;
          ctx.save();
          ctx.fillStyle = 'rgba(0,0,0,0.66)';
          ctx.fillRect(legendX, legendY, legendW, legendH);
          ctx.font = '10px monospace';
          legendRows.forEach((row, index) => {
            ctx.fillStyle = row.color;
            ctx.fillText(row.text, legendX + pad, legendY + pad + ((index + 1) * lineH) - 3);
          });
          ctx.restore();
        }
      };

      const footpoints = Array.isArray(metaNow?.footpoints) ? metaNow.footpoints : [];
      const droppedFootpoints = Array.isArray(metaNow?.droppedFootpoints) ? metaNow.droppedFootpoints : [];
      let finitePointCount = 0;
      let inBoundsPointCount = 0;
      for (const p of footpoints) {
        const px = Number(p?.x);
        const py = Number(p?.y);
        const nx = Number(p?.normX);
        const ny = Number(p?.normY);
        if ((!Number.isFinite(px) || !Number.isFinite(py)) && (!Number.isFinite(nx) || !Number.isFinite(ny))) continue;
        finitePointCount += 1;
        if (resolvePayloadPoint(p)) inBoundsPointCount += 1;
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

        const originRows: Array<{ text: string; color: string }> = [];
        const floorColor = 'rgba(124, 255, 147, 0.95)';
        const drawColor = 'rgba(89, 218, 255, 0.95)';
        const payloadColor = 'rgba(255, 93, 206, 0.95)';
        if (drawOriginLayer(floorplanLayerBounds, 'F0 floor', 'F0 floor', floorColor, -24, [5, 4])) {
          originRows.push({ text: 'F0 floorplan', color: floorColor });
        }
        if (drawOriginLayer(activeDrawBounds, 'T0 track', 'T0 track', drawColor, -10, [])) {
          originRows.push({ text: 'T0 track/trail', color: drawColor });
        }
        if (
          payloadLayerBounds &&
          !boundsNearlyEqual(payloadLayerBounds, activeDrawBounds) &&
          drawOriginLayer(payloadLayerBounds, 'P0 payload', 'P0 payload', payloadColor, 6, [2, 3])
        ) {
          originRows.push({ text: 'P0 payload', color: payloadColor });
        }
        drawOriginLegend(originRows);
      }

      const now = Date.now();
      const trailCfg = resolvedTrailConfig;
      const trailWindowMs = Math.max(100, trailCfg.window_s * 1000.0);

      const hueForId = (id: number) => (id * 47) % 360;
      const hsla = (id: number, a: number) => `hsla(${hueForId(id)}, 80%, 60%, ${a})`;
      const drawDebugCross = (mx: number, mz: number, color: string, size = 4) => {
        const resolved = resolveForDraw(mx, mz);
        if (!resolved) return;
        const px = drawX(resolved.x);
        const py = drawY(resolved.y);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(px - size, py);
        ctx.lineTo(px + size, py);
        ctx.moveTo(px, py - size);
        ctx.lineTo(px, py + size);
        ctx.stroke();
      };

      const drawDroppedMarker = (pt: NormalizedPayloadPoint, labelText: string) => {
        const metricX = Number(pt?.x);
        const metricY = Number(pt?.y);
        if (
          coverageEnvelope &&
          Number.isFinite(metricX) &&
          Number.isFinite(metricY) &&
          metricX >= xMin &&
          metricX <= xMax &&
          metricY >= zMin &&
          metricY <= zMax
        ) {
          const px = drawX(metricX);
          const py = drawY(metricY);
          ctx.save();
          ctx.fillStyle = 'rgba(244, 67, 54, 0.92)';
          ctx.strokeStyle = 'rgba(0, 0, 0, 0.78)';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(px, py, 5, 0, 2 * Math.PI);
          ctx.fill();
          ctx.stroke();
          ctx.font = 'bold 10px sans-serif';
          ctx.lineWidth = 3;
          ctx.strokeText(labelText, px + 7, py - 7);
          ctx.fillStyle = '#ff8f8f';
          ctx.fillText(labelText, px + 7, py - 7);
          ctx.restore();
          return;
        }
        const nx = Number(pt?.normX);
        const ny = Number(pt?.normY);
        if (!Number.isFinite(nx) || !Number.isFinite(ny)) return;
        const px = contentRect.x + clampNumber(nx, 0, 1) * contentRect.w;
        const py = contentRect.y + clampNumber(ny, 0, 1) * contentRect.h;
        ctx.save();
        ctx.fillStyle = 'rgba(244, 67, 54, 0.92)';
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.78)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        ctx.font = 'bold 10px sans-serif';
        ctx.lineWidth = 3;
        ctx.strokeText(labelText, px + 7, py - 7);
        ctx.fillStyle = '#ff8f8f';
        ctx.fillText(labelText, px + 7, py - 7);
        ctx.restore();
      };

      if (debug) {
        droppedFootpoints.slice(0, 8).forEach((pt, idx) => {
          const stableNum = typeof pt?.stableId === 'number' && Number.isFinite(pt.stableId) ? pt.stableId : null;
          const trackerNum = typeof pt?.trackerId === 'number' && Number.isFinite(pt.trackerId) ? pt.trackerId : null;
          const displayId = displayIdForPoint(stableNum, trackerNum);
          drawDroppedMarker(pt, displayId === null ? `drop${idx + 1}` : `${displayId} off`);
        });
        for (const p of footpoints) {
          const dbg = p?.alignmentDebug;
          if (!dbg || !Array.isArray(dbg.candidates)) continue;
          for (const cand of dbg.candidates) {
            const rayX = Number(cand?.rayFloor?.x);
            const rayZ = Number(cand?.rayFloor?.z);
            const depthX = Number(cand?.mapanythingDepth?.x);
            const depthZ = Number(cand?.mapanythingDepth?.z);
            const hasRay = Number.isFinite(rayX) && Number.isFinite(rayZ);
            const hasDepth = Number.isFinite(depthX) && Number.isFinite(depthZ);
            if (hasRay) drawDebugCross(rayX, rayZ, 'rgba(80, 200, 255, 0.72)', 3.5);
            if (hasDepth) drawDebugCross(depthX, depthZ, 'rgba(255, 96, 80, 0.74)', 3.5);
            if (hasRay && hasDepth) {
              const rr = resolveForDraw(rayX, rayZ);
              const dr = resolveForDraw(depthX, depthZ);
              if (rr && dr) {
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(drawX(rr.x), drawY(rr.y));
                ctx.lineTo(drawX(dr.x), drawY(dr.y));
                ctx.stroke();
              }
            }
          }
        }
      }

      // Resolver comparison is a bounded, exact-cohort overlay.  It is
      // intentionally drawn after the canonical point/trail inputs have been
      // resolved and cannot mutate either path.
      drawResolverComparison();

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
        const canonicalTrailKeys = new Set(
          (Array.isArray(metaNow?.footpoints) ? metaNow.footpoints : [])
            .filter((point) => point?.canonicalWorld === true)
            .map((point) => historyKeyForPoint(
              typeof point?.stableId === 'number' && Number.isFinite(point.stableId) ? point.stableId : null,
              typeof point?.trackerId === 'number' && Number.isFinite(point.trackerId) ? point.trackerId : null,
              point?.trackerLifecycleGeneration,
            ))
            .filter((key): key is string => key !== null),
        );
        const backendTracks = !frontendOwnsTrailSmoothing && Array.isArray(metaNow?.trails)
          ? metaNow.trails
            .map((tr) => {
              const stableNum = typeof tr?.stableId === 'number' && Number.isFinite(tr.stableId) ? tr.stableId : null;
              const trackerNum = typeof tr?.trackerId === 'number' && Number.isFinite(tr.trackerId) ? tr.trackerId : null;
              const displayId = displayIdForPoint(stableNum, trackerNum);
              if (displayId === null || !Array.isArray(tr?.points)) return null;
              const hasExplicitCanonicalMarker = typeof tr?.canonicalWorld === 'boolean';
              const legacyCanonicalKey = historyKeyForPoint(
                stableNum,
                trackerNum,
                tr?.trackerLifecycleGeneration,
              );
              const canonicalWorld = hasExplicitCanonicalMarker
                ? tr.canonicalWorld === true
                : legacyCanonicalKey !== null && canonicalTrailKeys.has(legacyCanonicalKey);
              const points = tr.points
                .map((p) => ({
                  x: Number(p?.x),
                  y: Number(p?.y),
                  t: Number(p?.t),
                  normX: Number(p?.normX),
                  normY: Number(p?.normY),
                  floorplanInside: p?.floorplanInside,
                  coverageInside: p?.coverageInside,
                  coverageRegion: p?.coverageRegion,
                  canonicalWorld,
                }))
                .map((p) => {
                  if (!Number.isFinite(p.t)) return null;
                  const hasMetric = Number.isFinite(p.x) && Number.isFinite(p.y);
                  const hasNorm = Number.isFinite(p.normX) && Number.isFinite(p.normY);
                  if (!hasMetric && !hasNorm) return null;
                  const resolved = resolvePayloadPoint(p);
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
        const headByLabel = new Map<string, { x: number; y: number; colorId: number; lastSeen: number; worldAdmission?: string }>();
        for (const [, data] of smoothState.current.entries()) {
          if (!data.stableId) continue;
          headByLabel.set(String(data.stableId), {
            x: data.x,
            y: data.y,
            colorId: data.colorId,
            lastSeen: data.lastSeen,
            worldAdmission: data.worldAdmission,
          });
        }
        const drawnHeadLabels = new Set<string>();

        const drawTrailHead = (label: string, colorId: number, head: { x: number; y: number; lastSeen: number; worldAdmission?: string }, lastDrawn: TrailPoint | null) => {
          const headResolved = resolveForDraw(head.x, head.y);
          if (!headResolved) return;
          const headAlpha = computeTrailAgeAlpha(now, head.lastSeen, trailWindowMs, trailCfg.min_alpha);
          const staleMs = Math.max(0, now - head.lastSeen);
          let blink = 1.0;
          if (trailCfg.stale_head_blink_enabled && staleMs >= trailCfg.stale_blink_start_ms) {
            const blinkPhase = (2 * Math.PI * (now % trailCfg.stale_blink_period_ms)) / trailCfg.stale_blink_period_ms;
            blink = 0.35 + 0.65 * (0.5 + 0.5 * Math.sin(blinkPhase));
          }
          const px = drawX(headResolved.x);
          const py = drawY(headResolved.y);

          const held = head.worldAdmission === 'held';
          const predicted = head.worldAdmission === 'predicted';
          if (lastDrawn && !held) {
            const tailPx = drawX(lastDrawn.x);
            const tailPy = drawY(lastDrawn.y);
            const gapPx = Math.hypot(px - tailPx, py - tailPy);
            if (gapPx >= TRAIL_CONNECTOR_MIN_PX) {
              ctx.strokeStyle = hsla(colorId, Math.min(1, headAlpha * 0.85));
              ctx.lineWidth = TRAIL_LINE_WIDTH * 0.9;
              ctx.beginPath();
              ctx.moveTo(tailPx, tailPy);
              ctx.lineTo(px, py);
              ctx.stroke();
            }
          }

          ctx.fillStyle = held
            ? 'rgba(255, 193, 7, 0.18)'
            : hsla(colorId, Math.min(1, (headAlpha * blink) + (predicted ? 0.08 : 0.25)));
          ctx.strokeStyle = held
            ? 'rgba(255, 193, 7, 0.95)'
            : predicted
              ? hsla(colorId, Math.min(1, headAlpha * 0.95))
              : 'rgba(0, 0, 0, 0.65)';
          ctx.lineWidth = held ? 2.25 : predicted ? 2.0 : 1.5;
          ctx.setLineDash(held ? [3, 2] : predicted ? [2, 2] : []);
          ctx.beginPath();
          ctx.arc(px, py, held ? TRAIL_HEAD_RADIUS + 1.5 : TRAIL_HEAD_RADIUS, 0, 2 * Math.PI);
          ctx.fill();
          ctx.stroke();
          ctx.setLineDash([]);
          drawnHeadLabels.add(label);
        };

        for (const tr of tracksToDraw) {
          const pts = tr.points;
          if (!pts || pts.length < 1) continue;

          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';

          let prev: TrailPoint | null = null;
          let lastDrawn: TrailPoint | null = null;
          for (let i = 0; i < pts.length; i += 1) {
            const p = pts[i];
            const resolved = Number.isFinite(p.x) && Number.isFinite(p.y) ? resolveForDraw(p.x, p.y) : null;
            if (!resolved) {
              prev = null;
              continue;
            }
            const drawPoint = { ...p, x: resolved.x, y: resolved.y };
            if (!prev) {
              prev = drawPoint;
              lastDrawn = drawPoint;
              continue;
            }

            const alpha = computeSegmentAlpha(now, prev.t, drawPoint.t, trailWindowMs, trailCfg.min_alpha);
            const frac = Math.max(0, Math.min(1, alpha));
            const lineWidth = TRAIL_LINE_WIDTH * (0.55 + 0.45 * frac);

            ctx.strokeStyle = hsla(tr.colorId, alpha);
            ctx.lineWidth = lineWidth;
            ctx.beginPath();
            ctx.moveTo(drawX(prev.x), drawY(prev.y));
            ctx.lineTo(drawX(drawPoint.x), drawY(drawPoint.y));
            ctx.stroke();
            prev = drawPoint;
            lastDrawn = drawPoint;
          }

          const head = headByLabel.get(tr.label);
          if (head) {
            drawTrailHead(tr.label, tr.colorId, head, lastDrawn);
          }
        }

        for (const [label, head] of headByLabel.entries()) {
          if (drawnHeadLabels.has(label)) continue;
          drawTrailHead(label, head.colorId, head, null);
        }
      }

      smoothState.current.forEach((pt) => {
        const age = now - pt.lastSeen;
        if (age > 500) return;
        const resolved = Number.isFinite(pt.x) && Number.isFinite(pt.y) ? resolveForDraw(pt.x, pt.y) : null;
        if (!resolved) return;

        if (pt.stableId) {
          const px = drawX(resolved.x);
          const py = drawY(resolved.y);
          ctx.fillStyle = '#fff';
          ctx.font = 'bold 12px sans-serif';
          ctx.shadowColor = 'black';
          ctx.shadowBlur = 4;
          ctx.fillText(pt.stableId, px + 8, py - 8);
          ctx.shadowBlur = 0;
        }
      });

      // PCF is camera-local: draw the actual camera origin (0, 0), which sits
      // near the bottom while preserving any measured space behind the camera.
      const cameraOrigin = resolveForDraw(0, 0) ?? { x: (xMin + xMax) * 0.5, y: zMin, mapped: false };
      const camPx = drawX(cameraOrigin.x);
      const camPy = drawY(cameraOrigin.y);

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
        ctx.fillText(
          coverageEnvelope
            ? 'Tracks are outside the camera coverage envelope'
            : 'Tracks are outside floorplan bounds',
          contentRect.x + (contentRect.w / 2),
          contentRect.y + 16
        );
        ctx.textAlign = 'left';
      }

      if (debug) {
        const lines = [
          `mode=${coordMode}`,
          `frame=${String(floorplanFrame || 'none')}`,
          `units=${String(metaNow?.units || floorplanNow?.units || 'unknown')}`,
          `coverage=${coverageEnvelope ? coverageEnvelope.regions.map(region => region.id).join('+') : 'floorplan'}`,
          `pts=${finitePointCount} in=${inBoundsPointCount} drop=${droppedFootpoints.length}`,
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
  }, [coordMode, coverageEnvelope, debug, displayBounds, floorPlaneEnabled, floorplan, heightRenderTuning, overlayEnabled, resolveDisplayPoint, resolvePayloadPoint, resolverComparisonEnabled, resolvedTrailConfig, variant]);

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
  const hasPlanarFloorLayer = Boolean(
    visualSelection.planarFloorSupportLayer?.grid_b64,
  );
  const floorplanHasImage = visualSelection.hasFloorplan;

  const isFrameMismatch = coordMode === 'world' && isFloorplanCameraLocal;
  const baseLabel = hasPlanarFloorLayer
    ? 'PCF Floor Plane'
    : visualSelection.baseKind === 'composite'
    ? 'Footprint Map'
    : (hasWalkableLayer ? 'Walkable Map' : (hasObstacleHeightLayer ? 'Obstacle Height' : 'Height Map'));
  const floorplanError = String(floorplan?.error || '').trim();
  const floorplanPending = ['no_cached_floorplan', 'capture_event_busy', 'rate_limited'].includes(floorplanError);
  const floorplanErrorLabel = floorplanError.replaceAll('_', ' ');
  const subtitleText = floorplanHasImage
    ? (isFrameMismatch ? `${baseLabel} (local-floorplan fallback)` : baseLabel)
    : (floorplanPending
        ? 'Preparing Map...'
        : (floorplanError ? `Map Error: ${floorplanErrorLabel}` : 'No Map Data'));
  const subtitle = ` • ${subtitleText}`;
  const hasHeightLookControls = variant === 'inline' && visualSelection.baseKind === 'height';
  const updateHeightTuning = (patch: Partial<HeightRenderTuning>) => {
    setHeightRenderTuning((prev) => {
      const next = { ...prev, ...patch };
      next.lowPct = clampNumber(Number(next.lowPct), 0, 99);
      next.highPct = clampNumber(Number(next.highPct), 1, 100);
      if (next.lowPct >= next.highPct) {
        if (Object.prototype.hasOwnProperty.call(patch, 'lowPct')) {
          next.lowPct = Math.max(0, next.highPct - 1);
        } else {
          next.highPct = Math.min(100, next.lowPct + 1);
        }
      }
      next.gamma = clampNumber(Number(next.gamma), 0.25, 3.0);
      next.densityCutoff = clampNumber(Number(next.densityCutoff), 0.0, 0.5);
      next.smoothing = Boolean(next.smoothing);
      return next;
    });
  };
  const resetHeightTuning = () => setHeightRenderTuning(DEFAULT_HEIGHT_RENDER_TUNING);
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
  const floorPlaneToggleLabel = variant === 'inline'
    && hasPlanarFloorLayer ? (
      <label className="bev-grid-toggle">
        <input
          type="checkbox"
          checked={floorPlaneEnabled}
          onChange={(event) => setFloorPlaneEnabled(event.target.checked)}
          style={{ marginRight: 6 }}
        />
        Floor plane
      </label>
    ) : null;
  const resolverComparisonToggleLabel = (
    <label
      className="bev-grid-toggle"
      title="Show the current resolver candidates and uncertainty without changing the canonical BEV point"
    >
      <input
        type="checkbox"
        checked={resolverComparisonEnabled}
        onChange={(event) => setResolverComparisonEnabled(event.target.checked)}
        style={{ marginRight: 6 }}
        aria-label="Localization details"
      />
      Localization details
    </label>
  );
  const lookControls = hasHeightLookControls ? (
    <div className="bev-look-control-wrap">
      <button
        type="button"
        className={`bev-look-button${lookPanelOpen ? ' is-active' : ''}`}
        onClick={() => setLookPanelOpen((value) => !value)}
        aria-expanded={lookPanelOpen}
      >
        Look
      </button>
      {lookPanelOpen && (
        <div className="bev-look-panel">
          <label className="bev-look-row">
            <span>Low</span>
            <input
              type="range"
              min="0"
              max="30"
              step="1"
              value={heightRenderTuning.lowPct}
              onChange={(event) => updateHeightTuning({ lowPct: Number(event.target.value) })}
            />
            <output>{heightRenderTuning.lowPct.toFixed(0)}%</output>
          </label>
          <label className="bev-look-row">
            <span>High</span>
            <input
              type="range"
              min="70"
              max="100"
              step="1"
              value={heightRenderTuning.highPct}
              onChange={(event) => updateHeightTuning({ highPct: Number(event.target.value) })}
            />
            <output>{heightRenderTuning.highPct.toFixed(0)}%</output>
          </label>
          <label className="bev-look-row">
            <span>Gamma</span>
            <input
              type="range"
              min="0.4"
              max="2.2"
              step="0.05"
              value={heightRenderTuning.gamma}
              onChange={(event) => updateHeightTuning({ gamma: Number(event.target.value) })}
            />
            <output>{heightRenderTuning.gamma.toFixed(2)}</output>
          </label>
          <label className="bev-look-row">
            <span>Mask</span>
            <input
              type="range"
              min="0"
              max="0.2"
              step="0.005"
              value={heightRenderTuning.densityCutoff}
              onChange={(event) => updateHeightTuning({ densityCutoff: Number(event.target.value) })}
            />
            <output>{heightRenderTuning.densityCutoff.toFixed(3)}</output>
          </label>
          <div className="bev-look-actions">
            <label>
              <input
                type="checkbox"
                checked={heightRenderTuning.smoothing}
                onChange={(event) => updateHeightTuning({ smoothing: event.target.checked })}
              />
              Smooth
            </label>
            <button type="button" onClick={resetHeightTuning}>Reset</button>
          </div>
        </div>
      )}
    </div>
  ) : null;

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
          {lookControls}
          <div className="bev-grid-toggle-wrap">
            {floorPlaneToggleLabel}
            {toggleLabel}
            {resolverComparisonToggleLabel}
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
          {resolverComparisonToggleLabel}
        </div>
      </div>
    </div>
  );
};

export default BevView;
