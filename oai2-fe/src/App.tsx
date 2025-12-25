import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { StreamPanel } from './components/StreamPanel';
// MapPanel moved into a drawer
// import { MapPanel } from './components/MapPanel';
import { ControlsPanel } from './components/ControlsPanel';
// LegendPanel removed; we now inline the legend dot next to each ID
import { TelemetryPanel } from './telemetry/TelemetryPanel';
import { TelemetryProvider, useTelemetry } from './telemetry/TelemetryContext';
import { TrailStore } from './lib/trails';
import { cameraOrder, colorForTrack, cameraLabel, detectCameraKey, CameraKey, colorIdForPerson, identityKeyForPerson } from './lib/camera';
import { getExtrinsics, worldToCamera, getIntrinsics4, extractPoseFromExtrinsics, forwardXZFromExtrinsics } from './lib/calibration';
import { useWebSocketClient, StatsPayload, DepthRequestStrategy, MosaicLayout } from './hooks/useWebSocketClient';
import { useWebRTCClient } from './hooks/useWebRTCClient';
import { StreamMode } from './components/StreamPanel';
import DepthDrawer, { DepthDiagnosticsEntry, DepthDrawerEntry, DepthMetaEntry, FloorplanResponse } from './components/DepthDrawer';
import { BevView, BevMeta } from './components/BevView';
import RoiEditorDrawer from './components/RoiEditorDrawer';
import SettingsCorner from './components/SettingsCorner';

const wsHost = import.meta.env.VITE_WS_HOST || window.location.hostname;
const wsPort = Number(import.meta.env.VITE_WS_PORT || 6008);
const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws';
const WS_URL = import.meta.env.VITE_WS_URL || `${wsProto}://${wsHost}:${wsPort}`;
// Prefer same-origin `/api/...` (Vite dev proxy or prod reverse-proxy) to avoid CORS.
// Override with `VITE_REST_URL` if the REST API is intentionally hosted elsewhere.
const REST_URL = import.meta.env.VITE_REST_URL || '';
const streamDisplayCams: CameraKey[] = ['living-room'];

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

const buildMosaicCameraIdToSlotKey = (layout: MosaicLayout | null): Record<string, CameraKey> => {
  const map: Record<string, CameraKey> = {};
  if (!layout || !Array.isArray(layout.sources)) return map;
  for (const src of layout.sources) {
    const sourceId = Number((src as any).source_id);
    if (!Number.isFinite(sourceId)) continue;
    const slotKey = cameraOrder[sourceId as 0 | 1 | 2];
    if (!slotKey) continue;
    const cameraId = normalizeCameraIdKey((src as any).camera_id);
    if (cameraId) map[cameraId] = slotKey;
    map[String(sourceId)] = slotKey;
    map[`rtsp_${sourceId}`] = slotKey;
    map[`camera_${sourceId}`] = slotKey;
    map[`source_${sourceId}`] = slotKey;
  }
  return map;
};

function Dashboard() {
  const { publish } = useTelemetry();

  // Stream image blobs
  const [streams, setStreams] = useState<{ [k: string]: Blob | null }>({ 'living-room': null, 'kitchen': null, 'family-room': null });
  const [bevMeta, setBevMeta] = useState<Record<CameraKey, BevMeta | undefined>>({ 'living-room': undefined, 'kitchen': undefined, 'family-room': undefined });
  const bevMetaRef = useRef<Record<CameraKey, BevMeta | undefined>>({ 'living-room': undefined, 'kitchen': undefined, 'family-room': undefined });

  // FPS tracking (exponential over short window)
  const [fps, setFps] = useState<{ [k: string]: string }>({ 'living-room': 'FPS: 0.0', 'kitchen': 'FPS: 0.0', 'family-room': 'FPS: 0.0' });
  const [fpsSeries, setFpsSeries] = useState<{ [k in CameraKey]: number[] }>({ 'living-room': [], 'kitchen': [], 'family-room': [] });
  const fpsHistory = useRef<{ [k: string]: number[] }>({ 'living-room': [], 'kitchen': [], 'family-room': [] });

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

  const computeFps = useCallback((key: CameraKey) => {
    const hist = fpsHistory.current[key];
    const now = Date.now();
    hist.push(now);
    const maxN = 30; if (hist.length > maxN) hist.shift();
    if (hist.length >= 2) {
      const span = (hist[hist.length - 1] - hist[0]) / 1000;
      if (span > 0) {
        const fpsVal = (hist.length - 1) / span;
        updateFps(key, fpsVal, now);
      }
    }
  }, [updateFps]);

  const handleWebrtcFps = useCallback((fpsVal: number) => {
    streamDisplayCams.forEach((cam) => updateFps(cam, fpsVal));
  }, [updateFps]);

  // Occupancy/Tracks (transitions removed)
  const [occupancy, setOccupancy] = useState<string>('');
  const [trackDetailsHtml, setTrackDetailsHtml] = useState<string>('');
  // Transitions removed from UI
  type ActiveTrack = {
    stable_id: number;
    camera_id: string;
    zone?: string;
    center?: [number, number];
    dwell_time?: number;
    velocity?: [number, number];
    world?: [number, number, number];
    world_valid?: boolean;
  };
  const [tracksByCamera, setTracksByCamera] = useState<Record<string, ActiveTrack[]>>({});
  const [tracksByCamKey, setTracksByCamKey] = useState<Record<CameraKey, ActiveTrack[]>>({ 'living-room': [], 'kitchen': [], 'family-room': [] });
  const [occByCamKey, setOccByCamKey] = useState<Record<CameraKey, Record<string, number>>>({ 'living-room': {}, 'kitchen': {}, 'family-room': {} });
  // Vacancy timer state
  const [vacancyText, setVacancyText] = useState<Record<CameraKey, string>>({ 'living-room': '', 'kitchen': '', 'family-room': '' });
  const zeroSinceRef = useRef<Record<CameraKey, number | null>>({ 'living-room': null, 'kitchen': null, 'family-room': null });

  const [trailEnabled, setTrailEnabled] = useState<boolean>(true);
  const trailStoreRef = useRef(new TrailStore());
  const prevActiveRef = useRef<Record<CameraKey, Set<string>>>(
    { 'living-room': new Set(), 'kitchen': new Set(), 'family-room': new Set() }
  );
  // Switch cams to world-space top-down when available
  const usingWorldKitchenRef = useRef<boolean>(false);
  const usingWorldLivingRef = useRef<boolean>(false);
  const usingWorldFamilyRef = useRef<boolean>(false);

  const [telemetryOpen, setTelemetryOpen] = useState(false);
  const [depthDrawerOpen, setDepthDrawerOpen] = useState(false);
  const [maDiagnostics, setMaDiagnostics] = useState<Record<string, DepthDiagnosticsEntry>>({});
  const [maDepthData, setMaDepthData] = useState<Record<string, DepthDrawerEntry>>({});
  const depthMetaRef = useRef<Record<string, DepthMetaEntry>>({});
  const [maDepthMeta, setMaDepthMeta] = useState<Record<string, DepthMetaEntry>>({});
  const [floorplanData, setFloorplanData] = useState<Record<string, FloorplanResponse>>({});
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
  const [expandedCamera, setExpandedCamera] = useState<CameraKey | null>(null);
  const [roiDrawerOpen, setRoiDrawerOpen] = useState(false);
  const [mosaicLayout, setMosaicLayout] = useState<MosaicLayout | null>(null);
  const [analyticsReloadCount, setAnalyticsReloadCount] = useState<number>(0);
  const maDiagThrottleRef = useRef<Record<string, number>>({});
  const lastCalibrationSignatureRef = useRef<string>('');
  const lastDepthFloorplanTsRef = useRef<Record<string, number>>({});
  const mosaicCameraIdToSlotKeyRef = useRef<Record<string, CameraKey>>({});

  // Stream mode is fixed to WebRTC (former JPEG toggle removed)
  const streamMode: StreamMode = 'webrtc';
  const [webrtcError, setWebrtcError] = useState<string | null>(null);

  const resolveDisplayCameraKey = useCallback((rawId: unknown): CameraKey | null => {
    const id = normalizeCameraIdKey(rawId);
    if (!id) return null;
    const mapped = mosaicCameraIdToSlotKeyRef.current[id];
    if (mapped) return mapped;
    return detectCameraKey(id);
  }, []);

  const onStats = (payload: StatsPayload) => {
    // System status/uptime
    const now = Date.now();
    publish({ group: 'System', key: 'Uptime', value: Math.floor((payload.uptime ?? 0)), ts: now });

    const nextLayout = payload.pipeline?.mosaic_layout;
    if (nextLayout) {
      setMosaicLayout(nextLayout);
      mosaicCameraIdToSlotKeyRef.current = buildMosaicCameraIdToSlotKey(nextLayout);
    }
    const reloadCount = payload.pipeline?.analytics_reload_count;
    if (typeof reloadCount === 'number') {
      setAnalyticsReloadCount(reloadCount);
    }

    const cameras = payload.cameras || {};
    setAvailableCameras(Object.keys(cameras));
    const statusUpdates: Partial<Record<CameraKey, string>> = {};
    const globalOcc: Record<string, number> = {};
    let allTracks: any[] = [];
    // Transitions disabled; keep placeholder for compatibility
    let allTrans: any[] = [];
    const perCamTracks: Record<string, any[]> = {};
    const perKeyTracks: Record<CameraKey, any[]> = { 'living-room': [], 'kitchen': [], 'family-room': [] };
    const perKeyOcc: Record<CameraKey, Record<string, number>> = { 'living-room': {}, 'kitchen': {}, 'family-room': {} };
    // const perKeyTransCount: Record<CameraKey, number> = { 'living-room': 0, 'kitchen': 0, 'family-room': 0 };

    const seenNow: Record<CameraKey, Set<string>> = {
      'living-room': new Set(), 'kitchen': new Set(), 'family-room': new Set()
    };

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
            // Prefer camera-local space for kitchen when provided (x_cam,z_cam → canvas x,y)
            const tw: any = t as any;
            const hasWorld = Array.isArray(tw.world) && tw.world.length >= 3 && !!tw.world_valid;
            if (key === 'kitchen' && hasWorld) {
              if (!usingWorldKitchenRef.current) {
                // First time we see valid world for kitchen, clear old pixel trails for that cam
                try { (trailStoreRef.current.trails as any)['kitchen'] = {}; } catch { }
                usingWorldKitchenRef.current = true;
              }
              const w = tw.world as [number, number, number];
              const E = getExtrinsics('kitchen');
              if (E) {
                const pc = worldToCamera(E, w);
                if (pc) {
                  const xCam = pc[0];
                  const zCam = pc[2];
                  const depth = Math.abs(zCam);
                  trailStoreRef.current.push(key, activeKey, { x: xCam, y: depth }, colorId);
                }
              } else {
                // Fallback to world XZ if extrinsics not loaded
                trailStoreRef.current.push(key, activeKey, { x: Number(w[0] || 0), y: Number(w[2] || 0) }, colorId);
              }
              seenNow[key].add(activeKey);
            } else if (key === 'living-room' && hasWorld) {
              if (!usingWorldLivingRef.current) {
                try { (trailStoreRef.current.trails as any)['living-room'] = {}; } catch { }
                usingWorldLivingRef.current = true;
              }
              const w = tw.world as [number, number, number];
              const E = getExtrinsics('living-room');
              if (E) {
                const pc = worldToCamera(E, w);
                if (pc) {
                  const xCam = pc[0];
                  const zCam = pc[2];
                  const depth = Math.abs(zCam);
                  trailStoreRef.current.push(key, activeKey, { x: xCam, y: depth }, colorId);
                }
              }
              seenNow[key].add(activeKey);
            } else if (key === 'family-room' && hasWorld) {
              if (!usingWorldFamilyRef.current) {
                try { (trailStoreRef.current.trails as any)['family-room'] = {}; } catch { }
                usingWorldFamilyRef.current = true;
              }
              const w = tw.world as [number, number, number];
              const E = getExtrinsics('family-room');
              if (E) {
                const pc = worldToCamera(E, w);
                if (pc) {
                  const xCam = pc[0];
                  const zCam = pc[2];
                  const depth = Math.abs(zCam);
                  trailStoreRef.current.push(key, activeKey, { x: xCam, y: depth }, colorId);
                }
              }
              seenNow[key].add(activeKey);
            } else {
              const center = t.center; if (!Array.isArray(center) || center.length < 2) continue;
              trailStoreRef.current.push(key, activeKey, { x: center[0]!, y: center[1]! }, colorId);
              seenNow[key].add(activeKey);
            }
          }
        }
        perKeyTracks[camKey] = track!.active_tracks;
      }
      // Transitions disabled
    }

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
        tracksHtml += `<div><strong><span class="dot" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${dotColor};margin-right:6px;vertical-align:middle;"></span>SID ${t.stable_id ?? 'N/A'}:</strong><br/>Zone: ${t.zone || '-'}, Dwell: <span style="display:inline-block; min-width:4ch; text-align:right;">${dwell}</span>s<br/>Metric (BEV): ${metricText}<br/>Pixel Pos: [${(typeof center[0] === 'number' ? Number(center[0]).toFixed(3) : center[0])}, ${(typeof center[1] === 'number' ? Number(center[1]).toFixed(3) : center[1])}], Speed: <span style="display:inline-block; min-width:4ch; text-align:right;">${speed}</span> px/s</div>`;
      });
    } else {
      tracksHtml = '<span>No active tracks.</span>';
    }
    setTrackDetailsHtml(tracksHtml);
    setTracksByCamera(perCamTracks);
    setTracksByCamKey(perKeyTracks);
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

  const onImage = (cam: CameraKey, blob: Blob) => {
    setStreams(prev => ({ ...prev, [cam]: blob }));
    computeFps(cam);
  };

  const handleCalibrationBundle = useCallback((bundle: any) => {
    if (!bundle || typeof bundle !== 'object') return;

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
    if (!payload || payload.type !== 'floorplan_response') return;
    const camRaw = payload.camera_id || payload.camera || (Array.isArray(payload.cameras) && payload.cameras[0]);
    const camId = camRaw ? String(camRaw) : '';
    if (!camId) return;

    // Normalize key to match UI components (e.g. 'kitchen_camera' -> 'kitchen')
    const key = detectCameraKey(camId) || camId;

    setFloorplanData(prev => ({ ...prev, [key]: payload as FloorplanResponse }));

    const label = labelForCameraId(key);
    const now = Date.now();
    if (typeof payload.scale_m_per_px === 'number' && Number.isFinite(payload.scale_m_per_px)) {
      publish({
        group: 'MapAnything Floorplan',
        key: `${label} Scale (m/px)`,
        value: Number(payload.scale_m_per_px.toFixed(4)),
        ts: now
      });
    }
    const bounds = payload.bounds || {};
    if (
      typeof bounds.min_x === 'number' && Number.isFinite(bounds.min_x) &&
      typeof bounds.max_x === 'number' && Number.isFinite(bounds.max_x)
    ) {
      publish({
        group: 'MapAnything Floorplan',
        key: `${label} X Span (m)`,
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
        key: `${label} Z Span (m)`,
        value: Number((bounds.max_z - bounds.min_z).toFixed(2)),
        ts: now
      });
    }
  };

  const handleBevMeta = useCallback((payload: BevMeta) => {
    if (!payload) return;
    const cam = resolveDisplayCameraKey((payload.cameraId || payload.camId || '').toString());
    if (!cam) return;
    bevMetaRef.current = { ...bevMetaRef.current, [cam]: payload };
    setBevMeta((prev) => ({ ...prev, [cam]: payload }));
  }, [resolveDisplayCameraKey]);

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
    onImage,
    onBevMeta: handleBevMeta,
    onStats,
    onTrailToggle: (en) => setTrailEnabled(en),
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
    requestMapAnythingDepth(camId, 'cache-first');
  }, [requestMapAnythingDepth]);

  const handleRequestFloorplan = useCallback((options?: { camera?: string; requestId?: string; maxAgeSec?: number; gridResM?: number; maxExtentM?: number; cacheOnly?: boolean }) => {
    return requestFloorplan(options);
  }, [requestFloorplan]);

  useEffect(() => {
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
    return () => {
      timers.forEach((id) => window.clearTimeout(id));
    };
  }, [handleRequestFloorplan]);

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
        requestId: `bev-refresh-${camId}-${meta.tsUs}`,
        // Prefer cache-first behavior to avoid regenerating topdowns while snapshots are being written.
        maxAgeSec: 600,
        gridResM: 0.15,
        maxExtentM: 20,
        cacheOnly: false
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
        (['living-room', 'kitchen', 'family-room'] as CameraKey[]).forEach(k => {
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

  const streamsExpanded = Boolean(expandedCamera);
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
    setExpandedCamera(null);
    exitAppFullscreen();
  }, [exitAppFullscreen]);
  const handleToggleExpand = useCallback((cameraKey: CameraKey) => {
    const next = expandedCamera === cameraKey ? null : cameraKey;
    setExpandedCamera(next);
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
    if (!expandedCamera) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') collapseStreams();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [collapseStreams, expandedCamera]);

  useEffect(() => {
    if (streamsExpanded) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
      exitAppFullscreen();
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [exitAppFullscreen, streamsExpanded]);

  useEffect(() => {
    const onFullscreenChange = () => {
      if (!document.fullscreenElement && streamsExpanded) {
        setExpandedCamera(null);
      }
    };
    document.addEventListener('fullscreenchange', onFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', onFullscreenChange);
  }, [streamsExpanded]);

  return (
    <div className={`shell${streamsExpanded ? ' shell--streams-expanded' : ''}`}>
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
        <button className="btn ghost" onClick={() => setDepthDrawerOpen(v => !v)}>Depth</button>
        <button className="btn ghost" onClick={() => setRoiDrawerOpen(v => !v)}>ROIs</button>
      </header>
      <main className="main">
        <section className={`streams${streamsExpanded ? ' streams--expanded' : ''}`}>
          <div className={`stream-tiler${streamDisplayCams.length === 1 ? ' stream-tiler--single' : ''}`}>
            {streamDisplayCams.map((cam) => (
              <StreamPanel
                key={`stream-${cam}`}
                camera={cam}
                blob={streams[cam]}
                fpsText={fps[cam]}
                fpsSeries={fpsSeries[cam]}
                vacancyText={vacancyText[cam]}
                isExpanded={expandedCamera === cam}
                onToggleExpand={handleToggleExpand}
                streamMode={streamMode}
                videoRef={streamMode === 'webrtc' ? webrtc.videoRef : undefined}
              />
            ))}
          </div>
          <div className="bev-row">
            {cameraOrder.map((cam) => (
              <BevView
                key={`bev-${cam}`}
                cam={cam}
                meta={bevMeta[cam]}
                floorplan={floorplanData[cam]}
                tracks={tracksByCamKey[cam]}
                trailEnabled={trailEnabled}
                variant="inline"
              />
            ))}
          </div>
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
        <span className="subtitle">Use the Fullscreen button on any stream</span>
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
      />
      <RoiEditorDrawer
        open={roiDrawerOpen}
        onClose={() => setRoiDrawerOpen(false)}
        restBaseUrl={REST_URL}
        mosaicLayout={mosaicLayout}
        videoRef={webrtc.videoRef}
        analyticsReloadCount={analyticsReloadCount}
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
