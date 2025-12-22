import { PointerEvent as ReactPointerEvent, memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { cameraLabel, detectCameraKey } from '../lib/camera';
import '../styles/depth-drawer.css';

type DepthEntry = {
  ts: number;
  depth_b64: string;
  conf_b64?: string;
  mask_b64?: string;
  shape: [number, number];
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
  bounds?: { min_x?: number; max_x?: number; min_z?: number; max_z?: number };
  scale_m_per_px?: number;
  point_count?: number;
  error?: string;
  density?: FloorplanLayer;
  height?: FloorplanLayer;
  distance?: FloorplanLayer;
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
}

import {
  applyCanvasSize,
  decodeFloat32,
  decodeUint8,
  grayscaleColor,
  infernoColor,
  renderLayerToCanvas,
  turboColor,
  viridisColor
} from '../lib/renderUtils';

const DEFAULT_WIDTH = 700;
const MAX_WIDTH = 960;
const WIDE_ASPECT = 16 / 9;

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

const DepthDrawer = memo(function DepthDrawer({ open, onClose, diagnostics, depthData, depthMeta = {}, onRequestDepthFresh, onRequestDepthCached, onHeatmapReady, floorplans, onRequestFloorplan, availableCameras }: DepthDrawerProps) {
  // Show cameras from either available list or present depth data (union)
  const cameras = useMemo(() => {
    const set = new Set<string>();
    (availableCameras || []).forEach((c) => { if (c) set.add(c); });
    Object.keys(depthData || {}).forEach((c) => { if (c) set.add(c); });
    return Array.from(set).sort();
  }, [availableCameras, depthData]);
  const [activeTab, setActiveTab] = useState<'heatmap' | 'stats' | 'histogram' | 'metrics'>('heatmap');
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [drawerWidth, setDrawerWidth] = useState<number>(DEFAULT_WIDTH);
  const drawerRef = useRef<HTMLDivElement | null>(null);
  const isResizingRef = useRef(false);
  const previousUserSelectRef = useRef('');
  const activePointerIdRef = useRef<number | null>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const histogramCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const densityCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const heightCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const distanceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [floorplanStatus, setFloorplanStatus] = useState<'idle' | 'loading' | 'checking'>('idle');
  const [floorplanRequest, setFloorplanRequest] = useState<string>('');
  const [heatmapRange, setHeatmapRange] = useState<{ min: number; max: number } | null>(null);
  const warmupScheduledRef = useRef(false);
  const warmupTimersRef = useRef<number[]>([]);
  const floorplanWarmupTimersRef = useRef<number[]>([]);
  const floorplanPrefetchScheduledRef = useRef<Set<string>>(new Set());
  const floorplanPrefetchedRef = useRef<Set<string>>(new Set());
  const onRequestFloorplanRef = useRef(onRequestFloorplan);
  const heatmapNotifiedRef = useRef<Set<string>>(new Set());

  // Floorplan selection (declared early to avoid TDZ in hooks below)
  const cameraFloorplan = floorplans[selectedCamera];
  const densityLayer = cameraFloorplan?.density;
  const heightLayer = cameraFloorplan?.height;
  const distanceLayer = cameraFloorplan?.distance;
  const floorplanError = cameraFloorplan?.error ?? null;
  const floorplanServedFromCache = cameraFloorplan?.served_from_cache ?? false;
  const hasDensity = !!(densityLayer && densityLayer.grid_b64 && densityLayer.grid_shape);
  const hasHeight = !!(heightLayer && heightLayer.grid_b64 && heightLayer.grid_shape);
  const hasDistance = !!(distanceLayer && distanceLayer.grid_b64 && distanceLayer.grid_shape);
  const heightBase = heightLayer?.value_min ?? null;
  const heightMaxRaw = heightLayer?.value_max ?? null;
  const heightSpan = (heightBase !== null && heightMaxRaw !== null) ? Math.max(0, heightMaxRaw - heightBase) : null;
  const distanceMin = distanceLayer?.value_min;
  const distanceMax = distanceLayer?.value_max;
  const distanceMid = distanceMin !== undefined && distanceMax !== undefined ? (distanceMin + distanceMax) / 2 : undefined;
  const spanX = cameraFloorplan?.bounds ? Math.abs((cameraFloorplan.bounds.max_x ?? 0) - (cameraFloorplan.bounds.min_x ?? 0)) : undefined;
  const spanZ = cameraFloorplan?.bounds ? Math.abs((cameraFloorplan.bounds.max_z ?? 0) - (cameraFloorplan.bounds.min_z ?? 0)) : undefined;
  const floorplanStatusText = useMemo(() => {
    if (!open || activeTab !== 'heatmap') return '';
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
      renderLayerToCanvas(canvas, layer, palette, WIDE_ASPECT);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, selectedCamera]);

  useEffect(() => {
    if (open) {
      setDrawerWidth(DEFAULT_WIDTH);
    }
  }, [open]);

  const depthEntry = selectedCamera ? depthData[selectedCamera] : undefined;
  const depthMetaEntry = selectedCamera && depthMeta ? depthMeta[selectedCamera] : undefined;
  const summaryEntry = selectedCamera ? diagnostics[selectedCamera] : undefined;

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

  // Avoid showing stale topdown canvases when switching cameras.
  useEffect(() => {
    if (!open || activeTab !== 'heatmap') return;
    clearCanvasElement(densityCanvasRef.current);
    clearCanvasElement(heightCanvasRef.current);
    clearCanvasElement(distanceCanvasRef.current);
  }, [open, activeTab, selectedCamera, clearCanvasElement]);

  // When switching cameras, try to load the cached floorplan for that camera.
  useEffect(() => {
    if (!open || activeTab !== 'heatmap' || !selectedCamera) return;
    requestFloorplan('cache-only');
  }, [open, activeTab, selectedCamera, requestFloorplan]);

  // Notify backend once per camera when heatmap is rendered (cache-first or fresh)
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
      warmupTimersRef.current.forEach((id) => window.clearTimeout(id));
      warmupTimersRef.current = [];
      warmupScheduledRef.current = false;
      floorplanWarmupTimersRef.current.forEach((id) => window.clearTimeout(id));
      floorplanWarmupTimersRef.current = [];
      floorplanPrefetchScheduledRef.current.clear();
      floorplanPrefetchedRef.current.clear();
    };
  }, []);

  const scheduleDepthBatch = useCallback((cameraList: string[], spacingMs = 200) => {
    const unique = Array.from(new Set(cameraList)).filter(Boolean);
    if (!unique.length) return;
    warmupTimersRef.current.forEach((id) => window.clearTimeout(id));
    warmupTimersRef.current = unique.map((cam, idx) => window.setTimeout(() => onRequestDepthCached(cam), idx * spacingMs));
  }, [onRequestDepthCached]);


  useEffect(() => {
    if (open && availableCameras.length && !warmupScheduledRef.current) {
      // Warm up by requesting fresh depth for all available cameras
      try { console.debug('[UI] warmup depth batch', availableCameras); } catch { }
      scheduleDepthBatch(availableCameras);
      warmupScheduledRef.current = true;
    }
    if (!open) {
      warmupScheduledRef.current = false;
      warmupTimersRef.current.forEach((id) => window.clearTimeout(id));
      warmupTimersRef.current = [];
    }
  }, [open, availableCameras, scheduleDepthBatch]);

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
    if (activeTab !== 'heatmap' || !open) return;

    if (floorplanRequest && !cameraFloorplan) {
      clearCanvasElement(densityCanvasRef.current);
      clearCanvasElement(heightCanvasRef.current);
      clearCanvasElement(distanceCanvasRef.current);
      return;
    }

    if (floorplanError) {
      clearCanvasElement(densityCanvasRef.current);
      clearCanvasElement(heightCanvasRef.current);
      clearCanvasElement(distanceCanvasRef.current);
      setFloorplanStatus('idle');
      setFloorplanRequest('');
      return;
    }

    if (cameraFloorplan) {
      renderTopdownLayer(densityCanvasRef.current, densityLayer, grayscaleColor);
      renderTopdownLayer(heightCanvasRef.current, heightLayer, infernoColor);
      renderTopdownLayer(distanceCanvasRef.current, distanceLayer, viridisColor);

      if (!floorplanRequest || !cameraFloorplan.request_id || cameraFloorplan.request_id === floorplanRequest) {
        setFloorplanStatus('idle');
        setFloorplanRequest('');
      }
    }
  }, [activeTab, open, cameraFloorplan, densityLayer, heightLayer, distanceLayer, renderTopdownLayer, clearCanvasElement, floorplanRequest, drawerWidth, floorplanError]);

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
          <button className={activeTab === 'stats' ? 'active' : ''} onClick={() => setActiveTab('stats')}>Stats</button>
          <button className={activeTab === 'histogram' ? 'active' : ''} onClick={() => setActiveTab('histogram')}>Histogram</button>
          <button className={activeTab === 'metrics' ? 'active' : ''} onClick={() => setActiveTab('metrics')}>Metrics</button>
        </div>
        <div className="content">
          {!cameras.length && <p>No MapAnything diagnostics received yet.</p>}
          {cameras.length > 0 && (
            <div className={`drawer-toolbar ${activeTab === 'heatmap' ? 'drawer-toolbar--heatmap' : ''}`}>
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
              {activeTab === 'heatmap' && (
                <div className="drawer-toolbar__actions">
                  <button
                    type="button"
                    className="btn-icon"
                    onClick={() => {
                      if (!selectedCamera) return;
                      try { console.debug('[UI] refresh depth', selectedCamera); } catch { }
                      onRequestDepthFresh(selectedCamera);
                      requestFloorplan('regenerate');
                    }}
                    disabled={floorplanStatus === 'loading'}
                    aria-label="Refresh depth frame"
                    title={floorplanStatus === 'loading' ? 'Refreshing depth view...' : undefined}
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
                <div className="heatmap-cell heatmap-cell--right">
                  <div className="heatmap-cell__body">
                    <div className="heatmap-cell__title">Density (Grayscale)</div>
                    <canvas ref={densityCanvasRef} className="heatmap-canvas" style={{ aspectRatio: `${WIDE_ASPECT}` }} />
                  </div>
                  <div className="heatmap-cell__scale">
                    {renderScale(densityGradient, 0, 0.5, 1)}
                  </div>
                </div>
                <div className="heatmap-cell heatmap-cell--left">
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
                    <div className="heatmap-cell__title">Distance (Viridis)</div>
                    <canvas ref={distanceCanvasRef} className="heatmap-canvas" style={{ aspectRatio: `${WIDE_ASPECT}` }} />
                  </div>
                  <div className="heatmap-cell__scale">
                    {renderScale(viridisGradient, distanceMin ?? undefined, distanceMid ?? undefined, distanceMax ?? undefined, ' m')}
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
                  {hasDistance ? 'Average distance from camera.' : 'Waiting for distance data'}
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
