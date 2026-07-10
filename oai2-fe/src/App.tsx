import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { StreamPanel } from './components/StreamPanel';
// MapPanel moved into a drawer
// import { MapPanel } from './components/MapPanel';
import { ControlsPanel } from './components/ControlsPanel';
// LegendPanel removed; we now inline the legend dot next to each ID
import { TelemetryPanel } from './telemetry/TelemetryPanel';
import { TelemetryProvider, useTelemetry } from './telemetry/TelemetryContext';
import { TrailStore } from './lib/trails';
import { cameraOrder, colorForTrack, cameraLabel, detectCameraKey, CameraKey, colorIdForPerson, identityKeyForPerson, discoverCamerasFromPayloads } from './lib/camera';
import { getExtrinsics, getIntrinsics4, extractPoseFromExtrinsics, forwardXZFromExtrinsics } from './lib/calibration';
import { isCameraLocalFrame, projectWorldPointToCameraLocal, resolveBevFrameModeFromPayload } from './lib/coordTransforms';
import { useWebSocketClient, StatsPayload, MosaicLayout } from './hooks/useWebSocketClient';
import { useWebRTCClient } from './hooks/useWebRTCClient';
import { StreamMode } from './components/StreamPanel';
import DepthDrawer, { DepthDiagnosticsEntry, DepthDrawerEntry, DepthMetaEntry, FloorplanResponse } from './components/DepthDrawer';
import { BevView, BevMeta, type BevFrameMode } from './components/BevView';
import { MosaicCropCanvas } from './components/MosaicCropCanvas';
import type { BevTrailConfig } from './lib/bevTrails';
import RoiEditorDrawer from './components/RoiEditorDrawer';
import { HouseholdIdentityDrawer } from './components/HouseholdIdentityDrawer';
import SettingsCorner from './components/SettingsCorner';
import { LatencyCard } from './components/LatencyCard';
import { LatencyMetrics } from './types/latency';

const wsHost = import.meta.env.VITE_WS_HOST || window.location.hostname;
const wsPort = Number(import.meta.env.VITE_WS_PORT || 6008);
const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws';
const WS_URL = import.meta.env.VITE_WS_URL || `${wsProto}://${wsHost}:${wsPort}`;
const restHost = window.location.hostname || '127.0.0.1';
const restPort = Number(import.meta.env.VITE_REST_PORT || 8080);
const restProto = window.location.protocol === 'https:' ? 'https' : 'http';
const REST_URL = import.meta.env.VITE_REST_URL || (import.meta.env.DEV ? '' : `${restProto}://${restHost}:${restPort}`);
const streamDisplayCams: CameraKey[] = ['living-room'];

type ExpandedView = { kind: 'mosaic' } | { kind: 'camera'; camera: CameraKey };

type CameraPoseSummary = {
  x: number;
  y: number;
  z: number;
  heightAboveFloor?: number | null;
  forwardFx?: number | null;
  forwardFz?: number | null;
};

const labelForCameraId = (camId: string): string => {
  const key = detectCameraKey(camId);
  if (key) return cameraLabel(key);
  return camId;
};

const normalizeCameraIdKey = (value: unknown): string => String(value ?? '').toLowerCase().trim();

const floorplanHasRenderableGrid = (floorplan?: FloorplanResponse | null): boolean => Boolean(
  floorplan?.walkable?.grid_b64 ||
  floorplan?.obstacle_height?.grid_b64 ||
  floorplan?.height?.grid_b64 ||
  floorplan?.density?.grid_b64 ||
  floorplan?.distance?.grid_b64
);

const buildMosaicCameraIdToSlotKey = (layout: MosaicLayout | null): Record<string, CameraKey> => {
  const map: Record<string, CameraKey> = {};
  if (!layout || !Array.isArray(layout.sources)) return map;

  for (const src of layout.sources) {
    const sourceId = Number((src as any).source_id);
    const cameraIdRaw = (src as any).camera_id || (src as any).cameraId;
    const cameraId = normalizeCameraIdKey(cameraIdRaw);

    // Prefer an actual camera identifier from the feed (supports dynamic 4th/5th+ cameras)
    // Fall back to the legacy fixed-slot key only for the original three source_ids.
    let slotKey: CameraKey | null = null;

    if (cameraId) {
      // If we have a real camera_id string, use it directly as the display key for dynamic cameras.
      // This is the key fix for newly discovered rooms appearing in mosaic_layout.
      slotKey = cameraId as CameraKey;
    }

    if (!slotKey && Number.isFinite(sourceId)) {
      // Legacy fallback for the first three tiler slots (0/1/2) using the old cameraOrder
      slotKey = cameraOrder[sourceId as 0 | 1 | 2] || null;
    }

    if (!slotKey) continue;

    // Populate many alias forms so resolveDisplayCameraKey has the best chance
    if (cameraId) {
      map[cameraId] = slotKey;
      map[cameraIdRaw] = slotKey;
    }
    if (Number.isFinite(sourceId)) {
      map[String(sourceId)] = slotKey;
      map[`rtsp_${sourceId}`] = slotKey;
      map[`camera_${sourceId}`] = slotKey;
      map[`source_${sourceId}`] = slotKey;
    }
  }
  return map;
};

function Dashboard() {
  const { publish } = useTelemetry();

  // Item 4: BEV state now keyed by discovered cameras (seeded with the original three).
  const [bevMeta, setBevMeta] = useState<Record<CameraKey, BevMeta | undefined>>({ 'living-room': undefined, 'kitchen': undefined, 'family-room': undefined });
  const bevMetaRef = useRef<Record<CameraKey, BevMeta | undefined>>({ 'living-room': undefined, 'kitchen': undefined, 'family-room': undefined });
  const [bevMetaRaw, setBevMetaRaw] = useState<Record<CameraKey, BevMeta | undefined>>({ 'living-room': undefined, 'kitchen': undefined, 'family-room': undefined });
  const bevMetaRawRef = useRef<Record<CameraKey, BevMeta | undefined>>({ 'living-room': undefined, 'kitchen': undefined, 'family-room': undefined });

  // FPS tracking (exponential over short window)
  const [fps, setFps] = useState<{ [k: string]: string }>({ 'living-room': 'FPS: 0.0', 'kitchen': 'FPS: 0.0', 'family-room': 'FPS: 0.0' });
  const [fpsSeries, setFpsSeries] = useState<{ [k in CameraKey]: number[] }>({ 'living-room': [], 'kitchen': [], 'family-room': [] });

  const updateFps = useCallback((key: CameraKey, fpsValRaw: number, tsOverride?: number) => {
    const ts = tsOverride ?? Date.now();
    const safe = Number.isFinite(fpsValRaw) ? Math.max(0, Number(fpsValRaw)) : 0;
    const rounded = Number(safe.toFixed(1));
    setFps(prev => ({ ...prev, [key]: `FPS: ${rounded.toFixed(1)}` }));
    setFpsSeries(prev => {
      const arr = [...(prev[key] || [])];
      arr.push(rounded);
      if (arr.length > 120) arr.shift();
      return { ...prev, [key]: arr } as any;
    });
    publish({ group: `Camera ${key.replace('-', ' ')}`, key: 'FPS', value: rounded, ts });
  }, [publish]);

  const handleWebrtcFps = useCallback((fpsVal: number) => {
    streamDisplayCams.forEach((cam) => updateFps(cam, fpsVal));
  }, [updateFps]);

  // Occupancy/Tracks (transitions removed)
  const [occupancy, setOccupancy] = useState<string>('');
  const [trackDetailsHtml, setTrackDetailsHtml] = useState<string>('');
  // Transitions removed from UI
  type ActiveTrack = {
    stable_id: number;
    tracker_id?: number;
    id_display?: string;
    display_name?: string | null;
    resident_uuid?: string | null;
    identity_kind?: string | null;
    identity_state?: string | null;
    reid_confidence?: number | null;
    embedding_present?: boolean | null;
    overlap_permit?: boolean | null;
    camera_id: string;
    zone?: string;
    center?: [number, number];
    dwell_time?: number;
    velocity?: [number, number];
    world?: [number, number, number];
    world_valid?: boolean;
    world_source?: string;
    world_quality?: string;
    world_quality_reason?: string;
    depth_status?: string | null;
    depth_anchor_source?: string | null;
    depth_anchor_m?: number | null;
    depth_used_m?: number | null;
    depth_center_m?: number | null;
    depth_median_m?: number | null;
    depth_sample_count?: number | null;
    depth_valid_fraction?: number | null;
  };
  const [tracksByCamera, setTracksByCamera] = useState<Record<string, ActiveTrack[]>>({});
  const [occByCamKey, setOccByCamKey] = useState<Record<CameraKey, Record<string, number>>>({ 'living-room': {}, 'kitchen': {}, 'family-room': {} });
  // Vacancy timer state
  const [vacancyText, setVacancyText] = useState<Record<CameraKey, string>>({ 'living-room': '', 'kitchen': '', 'family-room': '' });
  const zeroSinceRef = useRef<Record<CameraKey, number | null>>({ 'living-room': null, 'kitchen': null, 'family-room': null });

  const householdLiveTracks = useMemo(() => {
    const out: ActiveTrack[] = [];
    for (const tracks of Object.values(tracksByCamera)) {
      if (!Array.isArray(tracks)) continue;
      for (const t of tracks) out.push(t);
    }
    return out;
  }, [tracksByCamera]);

  const [trailEnabled, setTrailEnabled] = useState<boolean>(true);
  const [bevTrailConfig, setBevTrailConfig] = useState<Partial<BevTrailConfig>>({});
  const trailStoreRef = useRef(new TrailStore());
  const prevActiveRef = useRef<Record<CameraKey, Set<string>>>(
    { 'living-room': new Set(), 'kitchen': new Set(), 'family-room': new Set() }
  );
  // Item 4 (config + discovery): seed with the original three for backward compat,
  // then grow from real feeds (calibration, mosaic_layout, floorplan, bev meta).
  const [knownCameras, setKnownCameras] = useState<CameraKey[]>(['living-room', 'kitchen', 'family-room']);
  const knownCamerasRef = useRef<CameraKey[]>(['living-room', 'kitchen', 'family-room']);

  const [bevFrameModeByCam, setBevFrameModeByCam] = useState<Record<CameraKey, BevFrameMode>>({
    'living-room': 'world',
    'kitchen': 'world',
    'family-room': 'world',
  });
  const bevFrameModeByCamRef = useRef<Record<CameraKey, BevFrameMode>>({
    'living-room': 'world',
    'kitchen': 'world',
    'family-room': 'world',
  });

  const [telemetryOpen, setTelemetryOpen] = useState(false);
  const [depthDrawerOpen, setDepthDrawerOpen] = useState(false);
  const [peopleDrawerOpen, setPeopleDrawerOpen] = useState(false);
  const [maDiagnostics, setMaDiagnostics] = useState<Record<string, DepthDiagnosticsEntry>>({});
  const [maDepthData, setMaDepthData] = useState<Record<string, DepthDrawerEntry>>({});
  const depthMetaRef = useRef<Record<string, DepthMetaEntry>>({});
  const [maDepthMeta, setMaDepthMeta] = useState<Record<string, DepthMetaEntry>>({});
  const [floorplanData, setFloorplanData] = useState<Record<string, FloorplanResponse>>({});
  const floorplanDataRef = useRef<Record<string, FloorplanResponse>>({});
  const [cameraStatuses, setCameraStatuses] = useState<Record<CameraKey, string>>({
    'living-room': 'unknown',
    'kitchen': 'unknown',
    'family-room': 'unknown'
  });
  const [cameraPoses, setCameraPoses] = useState<Record<CameraKey, CameraPoseSummary | null>>({
    'living-room': null,
    'kitchen': null,
    'family-room': null
  });
  const [availableCameras, setAvailableCameras] = useState<string[]>([]);
  const [expandedView, setExpandedView] = useState<ExpandedView | null>(null);
  const [roiDrawerOpen, setRoiDrawerOpen] = useState(false);
  const [mosaicLayout, setMosaicLayout] = useState<MosaicLayout | null>(null);
  const [analyticsReloadCount, setAnalyticsReloadCount] = useState<number>(0);
  const [pipelineLatency, setPipelineLatency] = useState<LatencyMetrics | null>(null);
  const [latencyByCamKey, setLatencyByCamKey] = useState<Record<CameraKey, LatencyMetrics | null>>({
    'living-room': null,
    'kitchen': null,
    'family-room': null,
  });
  const maDiagThrottleRef = useRef<Record<string, number>>({});
  const lastCalibrationSignatureRef = useRef<string>('');
  const [calibrationEpoch, setCalibrationEpoch] = useState<number>(0);
  const lastDepthFloorplanTsRef = useRef<Record<string, number>>({});
  const mosaicCameraIdToSlotKeyRef = useRef<Record<string, CameraKey>>({});
  const floorplanWarmupTimersRef = useRef<number[]>([]);
  const floorplanWarmupScheduledRef = useRef(false);

  useEffect(() => {
    floorplanDataRef.current = floorplanData;
  }, [floorplanData]);

  // Stream mode is fixed to WebRTC (former JPEG toggle removed)
  const streamMode: StreamMode = 'webrtc';
  const [webrtcError, setWebrtcError] = useState<string | null>(null);
  const queryFlags = useMemo(() => new URLSearchParams(window.location.search), []);
  const bevDebugEnabled = useMemo(() => {
    const raw = queryFlags.get('bevDebug') ?? queryFlags.get('debugBev') ?? queryFlags.get('bev_debug') ?? '';
    return raw === '1' || raw.toLowerCase() === 'true';
  }, [queryFlags]);
  const showWorldBevRow = useMemo(() => {
    const raw = queryFlags.get('worldView') ?? queryFlags.get('world_view') ?? '';
    return raw === '1' || raw.toLowerCase() === 'true';
  }, [queryFlags]);

  const resolveDisplayCameraKey = useCallback((rawId: unknown): CameraKey | null => {
    const id = normalizeCameraIdKey(rawId);
    if (!id) return null;

    // 1. Mosaic layout mapping (primary for source_id / camera_id from stats)
    const mapped = mosaicCameraIdToSlotKeyRef.current[id];
    if (mapped) return mapped;

    // 2. Legacy three-camera detector
    const legacy = detectCameraKey(id);
    if (legacy) return legacy;

    // 3. Dynamic discovery path (Codex finding fix): if we've already seen this camera via
    //    knownCameras growth, accept the normalized form as a first-class dynamic key.
    const known = knownCamerasRef.current;
    if (known.includes(id as CameraKey)) return id as CameraKey;

    // 4. Last resort: treat any other normalized identifier as a dynamic camera key
    //    (this is what discoverCamerasFromPayloads already does for new rooms).
    return id as CameraKey;
  }, []);

  const resolveBevFrameMode = (payload: BevMeta, fallbackMode: BevFrameMode): BevFrameMode => (
    resolveBevFrameModeFromPayload(payload, fallbackMode)
  );

  // Item 4 (config + discovery): called whenever we receive calibration, floorplan, stats with layout, or bev meta.
  // Extends the known camera list so BEV panels can appear for newly discovered rooms without code changes.
  const updateKnownCamerasFromPayload = (payload: any) => {
    const next = discoverCamerasFromPayloads([payload]);
    const current = knownCamerasRef.current;
    const added = next.filter(c => !current.includes(c));
    if (added.length > 0) {
      const updated = [...current, ...added];
      knownCamerasRef.current = updated;
      setKnownCameras(updated);
      // Lazily ensure per-cam state exists for any newly discovered cameras so the dynamic BEV panels render without undefined entries.
      added.forEach((c) => ensureCameraState(c));
    }
  };

  // Lazy initializer for per-camera state when discovery adds a new room. Keeps all the map/ref structures in sync
  // so <BevView cam={newKey}> and the various per-cam telemetry/floorplan objects never see missing keys.
  const ensureCameraState = (camKey: CameraKey) => {
    // Refs (synchronous)
    if (!bevMetaRef.current[camKey]) bevMetaRef.current[camKey] = undefined;
    if (!bevMetaRawRef.current[camKey]) bevMetaRawRef.current[camKey] = undefined;
    if (!bevFrameModeByCamRef.current[camKey]) bevFrameModeByCamRef.current[camKey] = 'world';
    if (!floorplanDataRef.current[camKey]) floorplanDataRef.current[camKey] = undefined as any;
    if (!prevActiveRef.current[camKey]) prevActiveRef.current[camKey] = new Set();
    if (!zeroSinceRef.current[camKey]) zeroSinceRef.current[camKey] = null;
    if (!lastDepthFloorplanTsRef.current[camKey]) lastDepthFloorplanTsRef.current[camKey] = 0;
    if (!maDiagThrottleRef.current[camKey]) maDiagThrottleRef.current[camKey] = 0;

    // State updaters (functional so they see latest)
    setBevMeta((prev) => (prev[camKey] ? prev : { ...prev, [camKey]: undefined }));
    setBevMetaRaw((prev) => (prev[camKey] ? prev : { ...prev, [camKey]: undefined }));
    setBevFrameModeByCam((prev) => (prev[camKey] ? prev : { ...prev, [camKey]: 'world' }));
    setFloorplanData((prev) => (prev[camKey] ? prev : { ...prev, [camKey]: undefined as any }));
    setCameraStatuses((prev) => (prev[camKey] ? prev : { ...prev, [camKey]: 'unknown' }));
    setCameraPoses((prev) => (prev[camKey] !== undefined ? prev : { ...prev, [camKey]: null }));
    setLatencyByCamKey((prev) => (prev[camKey] !== undefined ? prev : { ...prev, [camKey]: null }));
    setFpsSeries((prev) => (prev[camKey] ? prev : { ...prev, [camKey]: [] }));
    setOccByCamKey((prev) => (prev[camKey] ? prev : { ...prev, [camKey]: {} }));
    setVacancyText((prev) => (prev[camKey] ? prev : { ...prev, [camKey]: '' }));
  };

  const setBevFrameMode = (camKey: CameraKey, nextMode: BevFrameMode) => {
    const prevMode = bevFrameModeByCamRef.current[camKey] || 'world';
    if (prevMode === nextMode) return;
    bevFrameModeByCamRef.current[camKey] = nextMode;
    setBevFrameModeByCam((prev) => ({ ...prev, [camKey]: nextMode }));
    const camTrailStore = trailStoreRef.current.trails[camKey];
    if (camTrailStore && typeof camTrailStore === 'object') {
      for (const k in camTrailStore) {
        delete camTrailStore[k];
      }
    }
  };

  const setTrackPointFromWorld = (key: CameraKey, activeKey: string, colorId: number, tw: ActiveTrack): boolean => {
    if (!Array.isArray(tw.world) || tw.world.length < 3) return false;
    const w0 = Number(tw.world?.[0]);
    const w1 = Number(tw.world?.[1]);
    const w2 = Number(tw.world?.[2]);
    if ((tw.world_valid === false) || !Number.isFinite(w0) || !Number.isFinite(w1) || !Number.isFinite(w2)) return false;

    const fallbackFrame = floorplanDataRef.current[key]?.frame;
    const renderCameraLocal = isCameraLocalFrame(fallbackFrame);
    if (renderCameraLocal) {
      const projected = projectWorldPointToCameraLocal(key, w0, w1, w2);
      if (!projected) return false;
      trailStoreRef.current.push(key, activeKey, projected, colorId);
      return true;
    }

    trailStoreRef.current.push(key, activeKey, { x: w0, y: w2 }, colorId);
    return true;
  };

  const mergeBevMetaPayload = (prevPayload: BevMeta | undefined, payload: BevMeta): BevMeta => {
    if (!prevPayload) return payload;
    const merged: BevMeta = { ...prevPayload, ...payload };
    if (!Array.isArray(payload.footpoints) && Array.isArray(prevPayload.footpoints)) {
      merged.footpoints = prevPayload.footpoints;
    }
    if (!Array.isArray(payload.trails) && Array.isArray(prevPayload.trails)) {
      merged.trails = prevPayload.trails;
    }
    if (typeof payload.xMin !== 'number' && typeof prevPayload.xMin === 'number') merged.xMin = prevPayload.xMin;
    if (typeof payload.xMax !== 'number' && typeof prevPayload.xMax === 'number') merged.xMax = prevPayload.xMax;
    if (typeof payload.zMin !== 'number' && typeof prevPayload.zMin === 'number') merged.zMin = prevPayload.zMin;
    if (typeof payload.zMax !== 'number' && typeof prevPayload.zMax === 'number') merged.zMax = prevPayload.zMax;
    if (payload.type === 'bev-frame' && !Object.prototype.hasOwnProperty.call(payload, 'error')) {
      delete merged.error;
    }
    if (payload.type === 'bev-frame' && !Object.prototype.hasOwnProperty.call(payload, 'details')) {
      delete merged.details;
    }
    if (payload.type === 'bev-frame' && !Object.prototype.hasOwnProperty.call(payload, 'fallbackActive')) {
      merged.fallbackActive = false;
      delete merged.fallbackTrackCount;
      delete merged.fallbackSources;
      delete merged.fallbackReasonCounts;
    }
    return merged;
  };

  const normalizeBevMetaForDisplay = useCallback((cam: CameraKey, payload: BevMeta, mode: BevFrameMode): BevMeta => {
    const isWorldMode = mode === 'world';
    if (!isWorldMode) return payload;

    const floorplan = floorplanDataRef.current[cam];
    const fallbackFrame = floorplan?.frame;
    if (!isCameraLocalFrame(fallbackFrame)) return payload;

    const points = payload.footpoints;
    const trails = payload.trails;
    if ((!Array.isArray(points) || !points.length) && (!Array.isArray(trails) || !trails.length)) return payload;

    let didProject = false;
    const projectedPoints = Array.isArray(points) ? points.map((point) => {
      const wx = Number(point?.x);
      const wz = Number(point?.y);
      if (!Number.isFinite(wx) || !Number.isFinite(wz)) return point;

      const projected = projectWorldPointToCameraLocal(
        cam,
        wx,
        0,
        wz
      );
      if (!projected) return point;

      didProject = true;
      return { ...point, x: projected.x, y: projected.y };
    }) : points;

    const projectedTrails = Array.isArray(trails) ? trails.map((trail) => {
      if (!Array.isArray(trail?.points)) return trail;
      let projectedAny = false;
      const nextPoints = trail.points.map((point) => {
        const wx = Number(point?.x);
        const wz = Number(point?.y);
        if (!Number.isFinite(wx) || !Number.isFinite(wz)) return point;
        const projected = projectWorldPointToCameraLocal(cam, wx, 0, wz);
        if (!projected) return point;
        projectedAny = true;
        didProject = true;
        return { ...point, x: projected.x, y: projected.y };
      });
      return projectedAny ? { ...trail, points: nextPoints } : trail;
    }) : trails;

    if (!didProject) return payload;

    return {
      ...payload,
      frame: String(fallbackFrame || '').trim() || payload.frame,
      units: String(floorplan?.units || '').trim() || payload.units,
      footpoints: projectedPoints,
      trails: projectedTrails,
    };
  }, []);

  const onStats = (payload: StatsPayload) => {
    // System status/uptime
    const now = Date.now();
    publish({ group: 'System', key: 'Uptime', value: Math.floor((payload.uptime ?? 0)), ts: now });

    const nextLayout = payload.pipeline?.mosaic_layout;
    if (nextLayout) {
      updateKnownCamerasFromPayload(payload); // Item 4 discovery from layout
      setMosaicLayout(nextLayout);
      mosaicCameraIdToSlotKeyRef.current = buildMosaicCameraIdToSlotKey(nextLayout);
    }
    const reloadCount = payload.pipeline?.analytics_reload_count;
    if (typeof reloadCount === 'number') {
      setAnalyticsReloadCount(reloadCount);
    }

    // Latency metrics (Option A: NVDS built-in latency measurement surfaced in stats)
    const pipeLat = payload.pipeline?.latency_ms as LatencyMetrics | undefined;
    setPipelineLatency(pipeLat ?? null);
    if (pipeLat && pipeLat.enabled && typeof pipeLat.p95 === 'number' && Number.isFinite(pipeLat.p95)) {
      publish({ group: 'System', key: 'Latency p95 (ms)', value: Number(pipeLat.p95.toFixed(1)), ts: now });
    }

    const cameras = payload.cameras || {};
    setAvailableCameras(Object.keys(cameras));
    const statusUpdates: Partial<Record<CameraKey, string>> = {};
    const globalOcc: Record<string, number> = {};
    let allTracks: any[] = [];
    const perCamTracks: Record<string, any[]> = {};
    // Build per-cam accumulators from the live known set (supports dynamic panels / newly discovered cameras).
    const currentKnown = knownCamerasRef.current;
    const perKeyOcc: Record<CameraKey, Record<string, number>> = Object.fromEntries(currentKnown.map(k => [k, {}])) as any;
    // const perKeyTransCount: Record<CameraKey, number> = Object.fromEntries(currentKnown.map(k => [k, 0])) as any;

    const seenNow: Record<CameraKey, Set<string>> = Object.fromEntries(currentKnown.map(k => [k, new Set()])) as any;

    const nextLatencyByKey: Record<CameraKey, LatencyMetrics | null> = Object.fromEntries(currentKnown.map(k => [k, null])) as any;

    for (const camId in cameras) {
      const c = cameras[camId];
      const track = c?.tracking;
      const camKey = resolveDisplayCameraKey(camId);
      if (!camKey) continue;
      if (camKey === 'living-room' || camKey === 'kitchen' || camKey === 'family-room') {
        const statusText = typeof c?.status === 'string' && c.status.trim().length > 0
          ? c.status
          : (typeof c?.frame_count === 'number' && c.frame_count > 0 ? 'running' : 'unknown');
        statusUpdates[camKey] = statusText;
      }

      const camLat = c?.latency_ms as LatencyMetrics | undefined;
      if (camLat) {
        nextLatencyByKey[camKey] = camLat;
        if (camLat.enabled && typeof camLat.p95 === 'number' && Number.isFinite(camLat.p95)) {
          publish({ group: `Camera ${camKey.replace('-', ' ')}`, key: 'Latency p95 (ms)', value: Number(camLat.p95.toFixed(1)), ts: now });
        }
      }

      if (track?.occupancy) {
        for (const z in track.occupancy) globalOcc[z] = (globalOcc[z] || 0) + (track.occupancy as any)[z];
        perKeyOcc[camKey] = track.occupancy;
      }
      if (Array.isArray(track?.active_tracks)) {
        perCamTracks[camId] = track!.active_tracks;
        allTracks = allTracks.concat(track!.active_tracks);
        // trails
        if (trailEnabled) {
          for (const t of track!.active_tracks as ActiveTrack[]) {
            const key = camKey as CameraKey;
            const stableId = Number(t.stable_id);
            if (!Number.isFinite(stableId) || stableId <= 0) continue;
            const activeKey = identityKeyForPerson(key, stableId);
            const colorId = colorIdForPerson(key, stableId);

            if (!prevActiveRef.current[key]?.has(activeKey)) {
              trailStoreRef.current.pushBreak(key, activeKey, colorId);
            }
            // Prefer front-end world space when BEV payload indicates world-frame output.
            const isWorldMode = (bevFrameModeByCamRef.current[key] || 'world') === 'world';
            if (isWorldMode) {
              const hadWorld = setTrackPointFromWorld(key, activeKey, colorId, t as ActiveTrack);
              if (hadWorld) {
                seenNow[key].add(activeKey);
                continue;
              }
              // If world frame is active but the track has no valid world point, skip it for consistency.
              continue;
            }

            const center = t.center;
            if (!Array.isArray(center) || center.length < 2) continue;
            const cx = Number(center[0]);
            const cy = Number(center[1]);
            if (!Number.isFinite(cx) || !Number.isFinite(cy)) continue;
            trailStoreRef.current.push(key, activeKey, { x: cx, y: cy }, colorId);
            seenNow[key].add(activeKey);
          }
        }
      }
      // Transitions disabled
    }

    setLatencyByCamKey(nextLatencyByKey);
    prevActiveRef.current = seenNow;

    if (Object.keys(statusUpdates).length > 0) {
      setCameraStatuses(prev => ({ ...prev, ...statusUpdates }));
    }

    // Occupancy HTML
    let occHtml = '<ul style="margin:0;padding-left:16px">';
    const sortedOcc = Object.entries(globalOcc).sort(([, a], [, b]) => b - a);
    if (sortedOcc.length) {
      sortedOcc.forEach(([zone, cnt]) => {
        occHtml += `<li><strong>${zone}:</strong> <span style="display:inline-block;min-width:3ch;text-align:right;">${cnt}</span></li>`;
        publish({ group: 'Occupancy', key: zone, value: cnt, ts: now });
      });
    } else {
      occHtml += '<li>No occupancy data.</li>';
      publish({ group: 'Occupancy', key: 'none', value: 0, ts: now });
    }
    occHtml += '</ul>';
    setOccupancy(occHtml);

    // Track details HTML
    let tracksHtml = '';
    if (allTracks.length) {
      const findCamForTrack = (track: any): CameraKey | undefined => {
        const keyFromTrack = resolveDisplayCameraKey(String(track.camera_id || ''));
        return keyFromTrack ?? undefined;
      };

      const metricForTrack = (track: any): { x: number; y: number } | null => {
        const camKey = findCamForTrack(track);
        if (!camKey) return null;
        const meta = bevMetaRef.current[camKey];
        if (!meta?.footpoints) return null;
        const fp = (meta.footpoints as any[]).find((p) => Number(p.stableId) === Number(track.stable_id));
        if (!fp) return null;
        const mx = Number(fp.x);
        const mz = Number(fp.y);
        if (!Number.isFinite(mx) || !Number.isFinite(mz)) return null;
        return { x: mx, y: mz };
      };

      allTracks.sort((a, b) => Number(a.stable_id || 0) - Number(b.stable_id || 0)).forEach(t => {
        const dwell = t.dwell_time?.toFixed(1) ?? '0.0';
        const center = t.center || ['N/A', 'N/A'];
        const vel = t.velocity || [0, 0];
        const speed = Math.sqrt(vel[0] ** 2 + vel[1] ** 2).toFixed(1);
        const camKey = findCamForTrack(t);
        const dotColor = camKey
          ? colorForTrack(colorIdForPerson(camKey, t.stable_id))
          : colorForTrack(Number(t.stable_id || 0));
        const metric = metricForTrack(t);

        const metricText = metric ? `[${metric.x.toFixed(2)} m, ${metric.y.toFixed(2)} m]` : 'N/A';
        const depthUsed = typeof t.depth_used_m === 'number' && Number.isFinite(t.depth_used_m)
          ? `${t.depth_used_m.toFixed(2)} m`
          : 'N/A';
        const depthStatus = t.depth_status || 'missing';
        const depthAnchor = t.depth_anchor_source ? `, ${t.depth_anchor_source}` : '';
        const idDisplay = t.display_name
          || t.id_display
          || (typeof t.tracker_id === 'number'
              ? `[${t.tracker_id}] | [${t.stable_id ?? 'N/A'}]`
              : String(t.stable_id ?? 'N/A'));
        const kindLabel = t.identity_kind || t.identity_state || '';
        const kindBit = kindLabel ? ` · ${kindLabel}` : '';
        tracksHtml += `<div><strong><span class="dot" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${dotColor};margin-right:6px;vertical-align:middle;"></span>${t.display_name ? String(t.display_name) : `ID ${idDisplay}`}${kindBit}:</strong><br/>Zone: ${t.zone || '-'}, Dwell: <span style="display:inline-block; min-width:4ch; text-align:right;">${dwell}</span>s<br/>Metric (BEV): ${metricText}<br/>Depth Used: ${depthUsed} (${depthStatus}${depthAnchor})<br/>Pixel Pos: [${(typeof center[0] === 'number' ? Number(center[0]).toFixed(3) : center[0])}, ${(typeof center[1] === 'number' ? Number(center[1]).toFixed(3) : center[1])}], Speed: <span style="display:inline-block; min-width:4ch; text-align:right;">${speed}</span> px/s</div>`;
      });
    } else {
      tracksHtml = '<span>No active tracks.</span>';
    }
    setTrackDetailsHtml(tracksHtml);
    setTracksByCamera(perCamTracks);
    setOccByCamKey(perKeyOcc);
    // Update zero-since timestamps when occupancy changes
    try {
      (Object.keys(perKeyOcc) as CameraKey[]).forEach((k) => {
        const sum = Object.values(perKeyOcc[k] || {}).reduce((a, b) => a + (b || 0), 0);
        if (sum === 0) {
          if (!zeroSinceRef.current[k]) zeroSinceRef.current[k] = Date.now();
        } else {
          zeroSinceRef.current[k] = null;
          setVacancyText(prev => ({ ...prev, [k]: '' }));
        }
      });
    } catch {
      // defensive
    }

    publish({ group: 'Tracking', key: 'Active Tracks', value: allTracks.length, ts: now });
  };

  const handleCalibrationBundle = useCallback((bundle: any) => {
    if (!bundle || typeof bundle !== 'object') return;

    updateKnownCamerasFromPayload(bundle); // Item 4 discovery

    let signature = '';
    try {
      signature = JSON.stringify({ cameras: bundle.cameras ?? {}, align: bundle.align ?? {} });
    } catch {
      signature = '';
    }
    if (signature && signature === lastCalibrationSignatureRef.current) {
      return;
    }
    if (signature) {
      lastCalibrationSignatureRef.current = signature;
      setCalibrationEpoch((v) => v + 1);
    }

    const now = Date.now();
    const publishNumeric = (group: string, key: string, raw: unknown, digits = 3) => {
      if (typeof raw !== 'number' || !Number.isFinite(raw)) return;
      publish({ group, key, value: Number(raw.toFixed(digits)), ts: now });
    };

    const alignData = bundle.align ?? {};
    const floorYRaw = typeof alignData.floor_y === 'number'
      ? alignData.floor_y
      : typeof alignData.floorY === 'number'
        ? alignData.floorY
        : null;
    if (floorYRaw !== null && Number.isFinite(floorYRaw)) {
      publishNumeric('MapAnything Align', 'Floor Y (m)', floorYRaw, 3);
    }
    const units = alignData.units ?? {};
    const scaleRaw = typeof units.s_obj_to_m === 'number' ? units.s_obj_to_m : null;
    if (scaleRaw !== null && Number.isFinite(scaleRaw)) {
      publishNumeric('MapAnything Align', 'Units (s_obj_to_m)', scaleRaw, 5);
    }

    const poseByCam: Record<CameraKey, CameraPoseSummary | null> = {
      'living-room': null,
      'kitchen': null,
      'family-room': null
    };

    cameraOrder.forEach((camKey) => {
      const label = cameraLabel(camKey);
      const intr = getIntrinsics4(camKey);
      if (intr && intr.length >= 4) {
        const [fx, fy, cx, cy] = intr.map((val) => Number(val));
        publishNumeric('MapAnything Intrinsics', `${label} fx`, fx, 1);
        publishNumeric('MapAnything Intrinsics', `${label} fy`, fy, 1);
        publishNumeric('MapAnything Intrinsics', `${label} cx`, cx, 1);
        publishNumeric('MapAnything Intrinsics', `${label} cy`, cy, 1);
        if (typeof fx === 'number' && Number.isFinite(fx) && Math.abs(fx) > 1e-6) {
          publishNumeric('MapAnything Pixel Scale', `${label} m/px @1m (H)`, 1 / fx, 4);
        }
        if (typeof fy === 'number' && Number.isFinite(fy) && Math.abs(fy) > 1e-6) {
          publishNumeric('MapAnything Pixel Scale', `${label} m/px @1m (V)`, 1 / fy, 4);
        }
      }

      const E = getExtrinsics(camKey);
      if (E && Array.isArray(E) && E.length === 16) {
        let poseSummary: CameraPoseSummary | null = null;
        const pose = extractPoseFromExtrinsics(E);
        if (pose) {
          const [px, py, pz] = pose.Cw;
          publishNumeric('MapAnything Pose', `${label} Position X (m)`, px, 2);
          publishNumeric('MapAnything Pose', `${label} Position Y (m)`, py, 2);
          publishNumeric('MapAnything Pose', `${label} Position Z (m)`, pz, 2);
          let heightAboveFloor: number | null = null;
          if (floorYRaw !== null && Number.isFinite(floorYRaw)) {
            const h = py - floorYRaw;
            publishNumeric('MapAnything Pose', `${label} Height Above Floor (m)`, h, 2);
            heightAboveFloor = h;
          }
          poseSummary = {
            x: px,
            y: py,
            z: pz,
            heightAboveFloor,
            forwardFx: null,
            forwardFz: null
          };
        }
        const forward = forwardXZFromExtrinsics(E);
        if (forward) {
          publishNumeric('MapAnything Pose', `${label} Forward X`, forward.fx, 3);
          publishNumeric('MapAnything Pose', `${label} Forward Z`, forward.fz, 3);
          if (poseSummary) {
            poseSummary.forwardFx = forward.fx;
            poseSummary.forwardFz = forward.fz;
          } else {
            poseSummary = {
              x: NaN,
              y: NaN,
              z: NaN,
              heightAboveFloor: null,
              forwardFx: forward.fx,
              forwardFz: forward.fz
            };
          }
        }
        poseByCam[camKey] = poseSummary;
      }
    });
    setCameraPoses(poseByCam);
  }, [lastCalibrationSignatureRef, publish, setCameraPoses]);

  const handleMADiagnostics = (payload: any) => {
    const camId = payload?.cam_id || payload?.cameraId;
    if (!camId) return;
    setMaDiagnostics(prev => ({
      ...prev,
      [camId]: {
        summary: payload.summary || {},
        ts: payload.ts || Date.now()
      }
    }));

    const now = Date.now();
    const throttleKey = `diag:${camId}`;
    const last = maDiagThrottleRef.current[throttleKey] || 0;
    if (now - last < 4000) return;
    maDiagThrottleRef.current[throttleKey] = now;

    const label = labelForCameraId(camId);
    const summary = payload.summary || {};
    const publishDiag = (keySuffix: string, raw: unknown, digits = 2, allowString = false) => {
      if (allowString && typeof raw === 'string') {
        publish({ group: 'MapAnything Diagnostics', key: `${label} ${keySuffix}`, value: raw, ts: now });
        return;
      }
      if (typeof raw !== 'number' || !Number.isFinite(raw)) return;
      publish({ group: 'MapAnything Diagnostics', key: `${label} ${keySuffix}`, value: Number(raw.toFixed(digits)), ts: now });
    };

    publishDiag('Median Depth (m)', summary.median, 2);
    publishDiag('Depth p10 (m)', summary.p10, 2);
    publishDiag('Depth p90 (m)', summary.p90, 2);
    publishDiag('Confidence Mean', summary.conf_mean, 3);
    publishDiag('Valid Ratio', summary.valid_ratio, 3);
    publishDiag('Valid Samples', summary.sample_count, 0);
    if (summary.method) {
      publishDiag('Method', summary.method, 2, true);
    }
  };

  const handleMADepth = (message: any) => {
    if (!message) return;
    const parseTimestampUs = (value: any): number | undefined => {
      if (typeof value === 'number' && Number.isFinite(value)) return Number(value);
      if (typeof value === 'string') {
        const parsed = Number(value);
        if (Number.isFinite(parsed)) return parsed;
      }
      return undefined;
    };
    const candidatePayload = (message.payload && typeof message.payload === 'object') ? message.payload : message;
    const camSource = message.camera || message.cam_id || message.cameraId || message.camera_id || message.camId;
    const payloadCamera = candidatePayload?.camera || candidatePayload?.cam_id || candidatePayload?.cameraId || candidatePayload?.camera_id;
    const rawCamId = String(camSource || payloadCamera || '')
      .trim();
    if (!rawCamId || (message?.ok === false && !candidatePayload?.depth_b64)) return;
    const shape = candidatePayload?.shape || candidatePayload?.depth_shape;
    if (!Array.isArray(shape) || shape.length !== 2) return;
    const depthB64 = candidatePayload.depth_b64 || candidatePayload.depth_z_b64;
    if (!depthB64) return;
    const confB64 = candidatePayload.conf_b64 || candidatePayload.conf;
    let maskB64 = candidatePayload.mask_b64 || candidatePayload.mask;
    const normalsB64 = typeof candidatePayload.normals_b64 === 'string' ? candidatePayload.normals_b64 : undefined;
    const normalsShapeRaw = candidatePayload.normals_shape;
    let normalsShape: [number, number, number] | undefined;
    if (Array.isArray(normalsShapeRaw) && normalsShapeRaw.length >= 3) {
      const h = Number(normalsShapeRaw[0]) || 0;
      const w = Number(normalsShapeRaw[1]) || 0;
      const c = Number(normalsShapeRaw[2]) || 0;
      if (h > 0 && w > 0 && c > 0) {
        normalsShape = [h, w, c];
      }
    }
    const normalsDtypeRaw = typeof candidatePayload.normals_dtype === 'string' ? candidatePayload.normals_dtype : undefined;
    const normalsDtype = normalsDtypeRaw === 'float16' || normalsDtypeRaw === 'float32' ? normalsDtypeRaw : undefined;
    const normalsSpaceRaw = typeof candidatePayload.normals_space === 'string' ? candidatePayload.normals_space : undefined;
    const normalsSpace = normalsSpaceRaw === 'camera' || normalsSpaceRaw === 'world' ? normalsSpaceRaw : undefined;
    const normalsError = typeof candidatePayload.normals_error === 'string' ? candidatePayload.normals_error : undefined;
    if (Array.isArray(maskB64)) {
      const maskArr = Uint8Array.from(maskB64.map((v: any) => (v ? 1 : 0)));
      maskB64 = btoa(String.fromCharCode(...maskArr));
    }
    const tsFromResponse = parseTimestampUs(message.ts_us);
    const tsFromPayload = parseTimestampUs(candidatePayload?.ts);
    const tsUs = tsFromResponse ?? tsFromPayload ?? Math.floor(Date.now() * 1000);
    const resolvedCamKey = resolveDisplayCameraKey(rawCamId);
    const storageKey = resolvedCamKey ?? rawCamId;
    const prevTs = depthMetaRef.current[storageKey]?.tsUs ?? 0;
    if (prevTs && tsUs < prevTs) return;
    const entry: DepthDrawerEntry = {
      ts: tsUs,
      depth_b64: depthB64,
      conf_b64: confB64,
      mask_b64: maskB64,
      shape: [Number(shape[0]) || 0, Number(shape[1]) || 0]
    };
    if (normalsB64) entry.normals_b64 = normalsB64;
    if (normalsShape) entry.normals_shape = normalsShape;
    if (normalsDtype) entry.normals_dtype = normalsDtype;
    if (normalsSpace) entry.normals_space = normalsSpace;
    if (normalsError) entry.normals_error = normalsError;
    if (!entry.shape[0] || !entry.shape[1]) return;
    setMaDepthData(prev => ({ ...prev, [storageKey]: entry }));

    const meta: DepthMetaEntry = {
      tsUs,
      servedFromCache: typeof message.served_from_cache === 'boolean' ? message.served_from_cache : undefined,
      requestId: message.request_id || message.requestId || undefined,
      error: typeof message.error === 'string' ? message.error : undefined,
      sourceCameraId: rawCamId,
    };
    depthMetaRef.current[storageKey] = meta;
    setMaDepthMeta(prev => ({ ...prev, [storageKey]: meta }));

    const label = labelForCameraId(storageKey);
    const now = Date.now();
    publish({
      group: 'MapAnything Depth',
      key: `${label} Resolution`,
      value: `${entry.shape[0]}x${entry.shape[1]}`,
      ts: now
    });
    const tsForAge = tsFromPayload ?? tsUs;
    if (typeof tsForAge === 'number' && Number.isFinite(tsForAge)) {
      const ageMs = Math.max(0, now - Math.floor(tsForAge / 1000));
      publish({
        group: 'MapAnything Depth',
        key: `${label} Snapshot Age (s)`,
        value: Number((ageMs / 1000).toFixed(1)),
        ts: now
      });
    }
  };

  const handleFloorplan = (payload: any) => {
    updateKnownCamerasFromPayload(payload); // Item 4 discovery
    if (!payload || payload.type !== 'floorplan_response') return;
    const camRaw = payload.camera_id || payload.camera || (Array.isArray(payload.cameras) && payload.cameras[0]);
    const camId = camRaw ? String(camRaw) : '';
    if (!camId) return;

    // Normalize key to match UI components (e.g. 'kitchen_camera' -> 'kitchen')
    const key = detectCameraKey(camId) || camId;

    setFloorplanData(prev => {
      const nextPayload = payload as FloorplanResponse;
      const existing = prev[key];
      const hasExistingRenderableGrid = floorplanHasRenderableGrid(existing);
      const nextHasRenderableGrid = floorplanHasRenderableGrid(nextPayload);
      if (hasExistingRenderableGrid && (nextPayload?.error || !nextHasRenderableGrid)) {
        return prev;
      }
      return { ...prev, [key]: nextPayload };
    });

    const label = labelForCameraId(key);
    const now = Date.now();
    const scaleScene = Number((payload as any).scale_scene_per_px);
    const scaleMetric = Number(payload.scale_m_per_px);
    if (Number.isFinite(scaleScene)) {
      publish({
        group: 'MapAnything Floorplan',
        key: `${label} Scale (scene/px)`,
        value: Number(scaleScene.toFixed(4)),
        ts: now
      });
    } else if (Number.isFinite(scaleMetric)) {
      publish({
        group: 'MapAnything Floorplan',
        key: `${label} Scale (m/px)`,
        value: Number(scaleMetric.toFixed(4)),
        ts: now
      });
    }
    const bounds = payload.bounds || {};
    const boundsUnitLabel = String((payload as any).units || '').toLowerCase() === 'scene' ? 'scene' : 'm';
    if (
      typeof bounds.min_x === 'number' && Number.isFinite(bounds.min_x) &&
      typeof bounds.max_x === 'number' && Number.isFinite(bounds.max_x)
    ) {
      publish({
        group: 'MapAnything Floorplan',
        key: `${label} X Span (${boundsUnitLabel})`,
        value: Number((bounds.max_x - bounds.min_x).toFixed(2)),
        ts: now
      });
    }
    if (
      typeof bounds.min_z === 'number' && Number.isFinite(bounds.min_z) &&
      typeof bounds.max_z === 'number' && Number.isFinite(bounds.max_z)
    ) {
      publish({
        group: 'MapAnything Floorplan',
        key: `${label} Z Span (${boundsUnitLabel})`,
        value: Number((bounds.max_z - bounds.min_z).toFixed(2)),
        ts: now
      });
    }
  };

  const handleBevMeta = useCallback((payload: BevMeta) => {
    if (!payload) return;
    updateKnownCamerasFromPayload(payload); // Item 4 discovery
    const cam = resolveDisplayCameraKey((payload.cameraId || payload.camId || '').toString());
    if (!cam) return;
    const prevMode = bevFrameModeByCamRef.current[cam] || 'world';
    const nextMode = resolveBevFrameMode(payload, prevMode);
    setBevFrameMode(cam, nextMode);

    const prevRaw = bevMetaRawRef.current[cam];
    const mergedRaw = mergeBevMetaPayload(prevRaw, payload);
    bevMetaRawRef.current = { ...bevMetaRawRef.current, [cam]: mergedRaw };
    setBevMetaRaw((prev) => ({ ...prev, [cam]: mergedRaw }));

    const normalizedPayload = normalizeBevMetaForDisplay(cam, mergedRaw, nextMode);
    bevMetaRef.current = { ...bevMetaRef.current, [cam]: normalizedPayload };
    setBevMeta((prev) => ({ ...prev, [cam]: normalizedPayload }));
  }, [mergeBevMetaPayload, normalizeBevMetaForDisplay, resolveBevFrameMode, resolveDisplayCameraKey]);

  useEffect(() => {
    let changed = false;
    const next: Record<CameraKey, BevMeta | undefined> = { ...bevMetaRef.current };

    // Process all currently known cameras (supports dynamic discovery of new rooms).
    knownCamerasRef.current.forEach((cam) => {
      const raw = bevMetaRawRef.current[cam];
      if (!raw) return;
      const mode = bevFrameModeByCamRef.current[cam] || 'world';
      const normalized = normalizeBevMetaForDisplay(cam, raw, mode);
      if (next[cam] !== normalized) {
        next[cam] = normalized;
        changed = true;
      }
    });

    if (!changed) return;
    bevMetaRef.current = next;
    setBevMeta((prev) => ({ ...prev, ...next }));
  }, [floorplanData, normalizeBevMetaForDisplay]);

  // WebRTC handler refs (to break circular dependency with useWebSocketClient)
  const webrtcHandleAnswerRef = useRef<(sdp: string) => Promise<void>>(() => Promise.resolve());
  const webrtcHandleIceCandidateRef = useRef<(candidate: RTCIceCandidateInit) => Promise<void>>(() => Promise.resolve());

  const {
    status,
    sendClearStats,
    sendTrailToggle,
    sendDetectionConfig,
    sendDetectionToggle,
    requestMapAnythingDepth,
    requestFloorplan,
    notifyMaHeatmapReady,
    sendAutoCalibrate,
    sendWebRTCOffer,
    sendWebRTCIceCandidate,
  } = useWebSocketClient(WS_URL, {
    onBevMeta: handleBevMeta,
    onStats,
    onTrailToggle: (en) => {
      setTrailEnabled(en);
      setBevTrailConfig((prev) => ({ ...prev, enabled: en }));
    },
    onTrailSettings: (config) => {
      if (!config || typeof config !== 'object') return;
      const next = config as Partial<BevTrailConfig>;
      setBevTrailConfig((prev) => ({ ...prev, ...next }));
      if (typeof next.enabled === 'boolean') {
        setTrailEnabled(next.enabled);
      }
    },
    onCalibration: handleCalibrationBundle,
    onMADiagnostics: handleMADiagnostics,
    onMADepth: handleMADepth,
    onFloorplan: handleFloorplan,
    onAutoCalibrateResult: (payload) => {
      setIsCalibrating(false);
      const ok = payload?.ok;
      const updated = Array.isArray(payload?.updated) ? payload.updated : [];
      const err = typeof payload?.error === 'string' ? payload.error : '';
      if (ok && updated.length) {
        setCalibrateToast({ text: `Calibrated: ${updated.join(', ')}`, kind: 'success', ts: Date.now() });
      } else {
        setCalibrateToast({ text: err || 'Calibration failed', kind: 'error', ts: Date.now() });
      }
      window.setTimeout(() => setCalibrateToast(null), 3000);
    },
    // WebRTC signaling handlers (use refs to avoid circular dependency)
    onWebRTCAnswer: (sdp) => webrtcHandleAnswerRef.current(sdp),
    onWebRTCIceCandidate: (candidate) => webrtcHandleIceCandidateRef.current(candidate),
    onWebRTCError: (error) => {
      console.error('[WebRTC] Error:', error);
      setWebrtcError(error);
    },
  });

  const requestFloorplanRef = useRef(requestFloorplan);
  useEffect(() => {
    requestFloorplanRef.current = requestFloorplan;
  }, [requestFloorplan]);

  // Initialize WebRTC client hook
  const webrtc = useWebRTCClient(
    status === 'open' ? { sendOffer: sendWebRTCOffer, sendIceCandidate: sendWebRTCIceCandidate } : null,
    {
      debug: true,
      onVideoFps: handleWebrtcFps,
    }
  );

  // Populate refs after webrtc hook is initialized
  useEffect(() => {
    webrtcHandleAnswerRef.current = webrtc.handleAnswer;
    webrtcHandleIceCandidateRef.current = webrtc.handleIceCandidate;
  }, [webrtc.handleAnswer, webrtc.handleIceCandidate]);

  // Auto-connect WebRTC when the websocket is open
  // Use a ref to track if we've already initiated connection
  const webrtcConnectedRef = useRef(false);
  useEffect(() => {
    if (status === 'open') {
      // Only connect if we haven't already
      if (!webrtcConnectedRef.current) {
        webrtcConnectedRef.current = true;
        webrtc.connect().catch((err) => {
          console.error('[WebRTC] Connect failed:', err);
          setWebrtcError(String(err));
          webrtcConnectedRef.current = false; // Allow retry on failure
        });
      }
    }
    // Reset the ref and close the peer connection whenever the socket drops
    if (status !== 'open') {
      webrtcConnectedRef.current = false;
      webrtc.disconnect();
    }
  }, [status, webrtc.connect, webrtc.disconnect]);

  const requestDepthFresh = useCallback((camId: string) => {
    if (!camId) return;
    requestMapAnythingDepth(camId, 'fresh');
  }, [requestMapAnythingDepth]);

  const requestDepthCached = useCallback((camId: string) => {
    if (!camId) return;
    requestMapAnythingDepth(camId, 'cache-only');
  }, [requestMapAnythingDepth]);

  const handleRequestFloorplan = useCallback((options?: { camera?: string; requestId?: string; maxAgeSec?: number; gridResM?: number; maxExtentM?: number; cacheOnly?: boolean }) => {
    return requestFloorplanRef.current(options);
  }, []);

  useEffect(() => {
    if (status !== 'open') {
      floorplanWarmupScheduledRef.current = false;
      floorplanWarmupTimersRef.current.forEach((id) => window.clearTimeout(id));
      floorplanWarmupTimersRef.current = [];
      return;
    }
    if (floorplanWarmupScheduledRef.current) return;
    floorplanWarmupScheduledRef.current = true;

    const timers: number[] = [];
    cameraOrder.forEach((cam, idx) => {
      const timer = window.setTimeout(() => {
        handleRequestFloorplan({
          camera: cam,
          requestId: `bev-cache-${cam}-${Date.now()}`,
          maxAgeSec: 600,
          gridResM: 0.15,
          maxExtentM: 20,
          cacheOnly: true
        });
      }, idx * 120);
      timers.push(timer);
    });
    floorplanWarmupTimersRef.current = timers;
    return () => {
      timers.forEach((id) => window.clearTimeout(id));
    };
  }, [status, handleRequestFloorplan]);

  useEffect(() => {
    if (status !== 'open' || calibrationEpoch <= 0) return;
    const timers: number[] = [];
    cameraOrder.forEach((cam, idx) => {
      const timer = window.setTimeout(() => {
        handleRequestFloorplan({
          camera: cam,
          requestId: `bev-calib-cache-${cam}-${Date.now()}`,
          maxAgeSec: 600,
          gridResM: 0.15,
          maxExtentM: 20,
          cacheOnly: true,
        });
      }, idx * 140);
      timers.push(timer);
    });
    return () => {
      timers.forEach((id) => window.clearTimeout(id));
    };
  }, [calibrationEpoch, handleRequestFloorplan, status]);

  useEffect(() => {
    Object.entries(maDepthMeta || {}).forEach(([camId, meta]) => {
      if (!meta || typeof meta.tsUs !== 'number') return;
      // If the depth payload was remapped to a different UI camera key, do not auto-regenerate
      // floorplans off that signal; it can overwrite a good cached floorplan with a mismatched one.
      if (meta.sourceCameraId && meta.sourceCameraId !== camId) return;
      const prev = lastDepthFloorplanTsRef.current[camId] || 0;
      if (meta.tsUs <= prev) return;
      lastDepthFloorplanTsRef.current[camId] = meta.tsUs;
      handleRequestFloorplan({
        camera: camId,
        requestId: `bev-cache-after-depth-${camId}-${meta.tsUs}`,
        maxAgeSec: 600,
        gridResM: 0.15,
        maxExtentM: 20,
        cacheOnly: true
      });
    });
  }, [maDepthMeta, handleRequestFloorplan]);

  // Live EST/EDT clock for top bar
  const [estTime, setEstTime] = useState<string>("");
  useEffect(() => {
    const fmt = new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
      hour12: true,
      timeZoneName: 'short'
    });
    const tick = () => setEstTime(fmt.format(new Date()));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  // Vacency timer updater (tick every second)
  useEffect(() => {
    const formatVacancy = (secs: number) => {
      if (secs < 60) return `${secs} s`;
      const m = Math.floor(secs / 60);
      const s = secs % 60;
      return `${m} m ${s} s`;
    };

    const id = setInterval(() => {
      const now = Date.now();
      setVacancyText(prev => {
        let changed = false;
        const next: Record<CameraKey, string> = { ...prev } as any;
        // Process the live known set so newly discovered cameras get vacancy text handling.
        knownCamerasRef.current.forEach(k => {
          const started = zeroSinceRef.current[k];
          if (!started) {
            if (next[k] !== '') { next[k] = ''; changed = true; }
            return;
          }
          const elapsed = Math.floor((now - started) / 1000) - 15; // start after 15s
          if (elapsed >= 0) {
            const txt = formatVacancy(elapsed);
            if (next[k] !== txt) { next[k] = txt; changed = true; }
          } else {
            if (next[k] !== '') { next[k] = ''; changed = true; }
          }
        });
        return changed ? next : prev;
      });
    }, 1000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const connectionChip = useMemo(() => {
    let color = 'var(--bad)';
    let text = 'Disconnected';
    if (status === 'open') {
      color = 'var(--good)';
      text = 'Connected';
    } else if (status === 'connecting') {
      color = 'var(--warn)';
      text = 'Connecting…';
    } else if (status === 'closed') {
      color = 'var(--bad)';
      text = 'Closed';
    } else if (status === 'error') {
      color = 'var(--bad)';
      text = 'Error';
    }
    return <span className="chip"><span className="status-dot" style={{ background: color }} />{text}</span>;
  }, [status]);

  const timeChip = useMemo(() => (
    estTime ? <span className="chip mono" title="Current time (US Eastern)">{estTime}</span> : null
  ), [estTime]);

  const expandedCamera = expandedView?.kind === 'camera' ? expandedView.camera : null;
  const mosaicExpanded = expandedView?.kind === 'mosaic';
  const anyExpanded = expandedView !== null;
  const requestAppFullscreen = useCallback(() => {
    const root = document.documentElement;
    if (!document.fullscreenElement && root?.requestFullscreen) {
      root.requestFullscreen().catch(() => undefined);
    }
  }, []);
  const exitAppFullscreen = useCallback(() => {
    if (document.fullscreenElement && document.exitFullscreen) {
      document.exitFullscreen().catch(() => undefined);
    }
  }, []);
  const collapseStreams = useCallback(() => {
    setExpandedView(null);
    exitAppFullscreen();
  }, [exitAppFullscreen]);
  const handleToggleMosaicExpand = useCallback(() => {
    const next: ExpandedView | null = mosaicExpanded ? null : { kind: 'mosaic' };
    setExpandedView(next);
    if (next) {
      requestAppFullscreen();
    } else {
      exitAppFullscreen();
    }
  }, [exitAppFullscreen, mosaicExpanded, requestAppFullscreen]);
  const handleToggleCameraExpand = useCallback((cameraKey: CameraKey) => {
    const isCurrent = expandedCamera === cameraKey;
    const next: ExpandedView | null = isCurrent ? null : { kind: 'camera', camera: cameraKey };
    setExpandedView(next);
    if (next) {
      requestAppFullscreen();
    } else {
      exitAppFullscreen();
    }
  }, [expandedCamera, exitAppFullscreen, requestAppFullscreen]);

  const [calibrateToast, setCalibrateToast] = useState<{ text: string; kind: 'info' | 'success' | 'error'; ts: number } | null>(null);
  const [isCalibrating, setIsCalibrating] = useState<boolean>(false);

  useEffect(() => {
    if (status !== 'open') setIsCalibrating(false);
  }, [status]);

  const handleAutoCalibrateAll = useCallback(() => {
    if (status !== 'open' || isCalibrating) return;
    setCalibrateToast({ text: 'Calibrating…', kind: 'info', ts: Date.now() });
    setIsCalibrating(true);
    sendAutoCalibrate();
  }, [isCalibrating, sendAutoCalibrate, status]);

  useEffect(() => {
    if (!anyExpanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') collapseStreams();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [anyExpanded, collapseStreams]);

  useEffect(() => {
    if (anyExpanded) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
      exitAppFullscreen();
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [anyExpanded, exitAppFullscreen]);

  useEffect(() => {
    const onFullscreenChange = () => {
      if (!document.fullscreenElement && anyExpanded) {
        setExpandedView(null);
      }
    };
    document.addEventListener('fullscreenchange', onFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', onFullscreenChange);
  }, [anyExpanded]);

  return (
    <div className={`shell${mosaicExpanded ? ' shell--streams-expanded' : ''}`}>
      <header className="topbar">
        <div className="brand">
          <div className="logo" />
          <div>
            <div className="title">OAI² Console</div>
            <div className="subtitle">Spatial perception, calmly presented</div>
          </div>
        </div>
        <div className="spacer" />
        {timeChip}
        {connectionChip}
        <button className="btn ghost" onClick={() => setTelemetryOpen(v => !v)}>Telemetry</button>
        <button className="btn ghost" onClick={() => { setDepthDrawerOpen(v => !v); setRoiDrawerOpen(false); setPeopleDrawerOpen(false); }}>Depth</button>
        <button className="btn ghost" onClick={() => { setRoiDrawerOpen(v => !v); setDepthDrawerOpen(false); setPeopleDrawerOpen(false); }}>ROIs</button>
        <button
          className="btn ghost"
          onClick={() => { setPeopleDrawerOpen((v) => !v); setDepthDrawerOpen(false); setRoiDrawerOpen(false); }}
          title="Enroll and manage household residents"
        >
          People
        </button>
      </header>
      <main className="main">
        <section className={`streams${mosaicExpanded ? ' streams--expanded' : ''}`}>
          <div className={`stream-tiler${streamDisplayCams.length === 1 ? ' stream-tiler--single' : ''}`}>
            {streamDisplayCams.map((cam) => (
              <StreamPanel
                key={`stream-${cam}`}
                camera={cam}
                title="Mosaic"
                blob={null}
                fpsText={fps[cam]}
                fpsSeries={fpsSeries[cam]}
                vacancyText={vacancyText[cam]}
                isExpanded={mosaicExpanded}
                onToggleExpand={handleToggleMosaicExpand}
                tileCameras={knownCameras}
                expandedTileCamera={expandedCamera}
                onToggleTileExpand={handleToggleCameraExpand}
                mosaicLayout={mosaicLayout}
                streamMode={streamMode}
                videoRef={streamMode === 'webrtc' ? webrtc.videoRef : undefined}
              />
            ))}
          </div>
          <div className="bev-row">
            {knownCameras.map((cam) => (
              <BevView
                key={`bev-${cam}`}
                cam={cam}
                meta={bevMeta[cam]}
                floorplan={floorplanData[cam]}
                coordMode={bevFrameModeByCam[cam]}
                trailEnabled={trailEnabled}
                trailConfig={bevTrailConfig}
                debug={bevDebugEnabled}
                variant="inline"
              />
            ))}
          </div>
          {showWorldBevRow && (
            <div className="bev-row">
              {knownCameras.map((cam) => (
                <BevView
                  key={`bev-world-${cam}`}
                  cam={cam}
                  meta={bevMetaRaw[cam]}
                  coordMode={bevFrameModeByCam[cam]}
                  trailEnabled={trailEnabled}
                  trailConfig={bevTrailConfig}
                  debug={bevDebugEnabled}
                  variant="inline"
                />
              ))}
            </div>
          )}
        </section>

        <section className="side">
          <ControlsPanel
            trailEnabled={trailEnabled}
            setTrailEnabled={setTrailEnabled}
            onSendTrail={sendTrailToggle}
            onClearStats={sendClearStats}
            onSendDetectionConfig={sendDetectionConfig}
            onSendDetectionToggle={sendDetectionToggle}
          />

          <LatencyCard
            pipeline={pipelineLatency}
            perCam={latencyByCamKey}
          />

          <div className="panel card">
            <div className="card-title">Zone Occupancy</div>
            <div dangerouslySetInnerHTML={{ __html: occupancy }} />
          </div>

          { /* Legend removed; color dot is shown inline with each ID */}

          <div className="panel card">
            <div className="card-title">Active Tracks</div>
            <div dangerouslySetInnerHTML={{ __html: trackDetailsHtml }} />
          </div>

          { /* Transitions panel removed */}
        </section>
      </main>
      <footer className="footer">
        <span>WebSocket: {status}</span>
        <span className="spacer" />
        <span className="subtitle">RTSP to WebRTC mosaic</span>
      </footer>

      <DepthDrawer
        open={depthDrawerOpen}
        onClose={() => setDepthDrawerOpen(false)}
        diagnostics={maDiagnostics}
        depthData={maDepthData}
        depthMeta={maDepthMeta}
        onRequestDepthFresh={requestDepthFresh}
        onRequestDepthCached={requestDepthCached}
        onHeatmapReady={notifyMaHeatmapReady}
        floorplans={floorplanData}
        onRequestFloorplan={handleRequestFloorplan}
        availableCameras={availableCameras}
        mosaicLayout={mosaicLayout}
        videoRef={webrtc.videoRef}
        calibrationEpoch={calibrationEpoch}
      />
      <RoiEditorDrawer
        open={roiDrawerOpen}
        onClose={() => setRoiDrawerOpen(false)}
        restBaseUrl={REST_URL}
        mosaicLayout={mosaicLayout}
        videoRef={webrtc.videoRef}
        analyticsReloadCount={analyticsReloadCount}
      />
      <HouseholdIdentityDrawer
        open={peopleDrawerOpen}
        onClose={() => setPeopleDrawerOpen(false)}
        restBaseUrl={REST_URL}
        liveTracks={householdLiveTracks}
      />

      {telemetryOpen && (
        <TelemetryPanel
          onClose={() => setTelemetryOpen(false)}
          cameraStatuses={cameraStatuses}
          cameraPoses={cameraPoses}
        />
      )}
      <SettingsCorner
        connected={status === 'open'}
        calibrating={isCalibrating}
        onCalibrate={handleAutoCalibrateAll}
      />
      {expandedCamera && (
        <div className="individual-fullscreen" role="dialog" aria-modal="true" aria-label={`${cameraLabel(expandedCamera)} fullscreen`}>
          <div className="individual-fullscreen__toolbar">
            <div className="individual-fullscreen__title">{cameraLabel(expandedCamera)}</div>
            <button
              type="button"
              className="individual-fullscreen__close"
              onClick={collapseStreams}
              title="Close fullscreen"
              aria-label="Close fullscreen"
            >
              <span aria-hidden="true">X</span>
            </button>
          </div>
          <div className="individual-fullscreen__content">
            <div className="individual-fullscreen__video-panel">
              <MosaicCropCanvas
                camera={expandedCamera}
                mosaicLayout={mosaicLayout}
                videoRef={webrtc.videoRef}
                className="individual-fullscreen__video"
              />
            </div>
            <div className="individual-fullscreen__bev-panel">
              <BevView
                cam={expandedCamera}
                meta={bevMeta[expandedCamera]}
                floorplan={floorplanData[expandedCamera]}
                coordMode={bevFrameModeByCam[expandedCamera]}
                trailEnabled={trailEnabled}
                trailConfig={bevTrailConfig}
                debug={bevDebugEnabled}
                variant="inline"
              />
            </div>
          </div>
        </div>
      )}
      {calibrateToast && (
        <div
          className={`toast toast--${calibrateToast.kind}`}
          style={{
            position: 'fixed',
            bottom: 20,
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '10px 14px',
            borderRadius: 8,
            background: calibrateToast.kind === 'success' ? '#2ecc71' : calibrateToast.kind === 'error' ? '#e74c3c' : '#34495e',
            color: '#fff',
            boxShadow: '0 6px 18px rgba(0,0,0,0.25)',
            zIndex: 9999,
            minWidth: '220px',
            textAlign: 'center'
          }}
        >
          {calibrateToast.text}
        </div>
      )}
    </div>
  );
}

const App: React.FC = () => (
  <TelemetryProvider>
    <Dashboard />
  </TelemetryProvider>
);

export default App;
