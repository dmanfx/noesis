import { PointerEvent as ReactPointerEvent, memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import '../styles/depth-drawer.css';

type DepthEntry = {
  ts: number;
  depth_b64: string;
  conf_b64?: string;
  mask_b64?: string;
  shape: [number, number];
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

type FloorplanLayer = {
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
};

interface DepthDrawerProps {
  open: boolean;
  onClose: () => void;
  diagnostics: Record<string, DiagnosticsEntry>;
  depthData: Record<string, DepthEntry>;
  onRequestDepth: (cameraId: string) => void;
  floorplans: Record<string, FloorplanResponse>;
  onRequestFloorplan: (options: FloorplanRequestOptions) => string | void;
}

const VIRIDIS = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 36],
];

const DEFAULT_WIDTH = 700;
const MAX_WIDTH = 960;

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

const applyCanvasSize = (canvas: HTMLCanvasElement) => {
  const rect = canvas.getBoundingClientRect();
  const width = rect.width || canvas.clientWidth || 1;
  const height = rect.height || canvas.clientHeight || width;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(width * dpr));
  canvas.height = Math.max(1, Math.round(height * dpr));
  return { width, height, dpr };
};

function decodeFloat32(base64?: string): Float32Array | null {
  if (!base64) return null;
  try {
    const binary = atob(base64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return new Float32Array(bytes.buffer);
  } catch (err) {
    console.error('Failed to decode float32 payload', err);
    return null;
  }
}

function decodeUint8(base64?: string): Uint8Array | null {
  if (!base64) return null;
  try {
    const binary = atob(base64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  } catch (err) {
    console.error('Failed to decode uint8 payload', err);
    return null;
  }
}

function viridisColor(t: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * (VIRIDIS.length - 1);
  const idx = Math.floor(scaled);
  const frac = scaled - idx;
  const a = VIRIDIS[idx];
  const b = VIRIDIS[Math.min(idx + 1, VIRIDIS.length - 1)];
  const r = Math.round(a[0] + (b[0] - a[0]) * frac);
  const g = Math.round(a[1] + (b[1] - a[1]) * frac);
  const bl = Math.round(a[2] + (b[2] - a[2]) * frac);
  return [r, g, bl];
}

function grayscaleColor(t: number): [number, number, number] {
  const v = Math.round(255 * (1 - Math.min(1, Math.max(0, t))));
  return [v, v, v];
}

function infernoColor(t: number): [number, number, number] {
  const stops: Array<[number, [number, number, number]]> = [
    [0, [0, 0, 4]],
    [0.2, [35, 6, 59]],
    [0.4, [99, 23, 94]],
    [0.6, [159, 43, 73]],
    [0.8, [218, 83, 32]],
    [1, [252, 255, 164]],
  ];
  const clamped = Math.min(1, Math.max(0, t));
  for (let i = 0; i < stops.length - 1; i += 1) {
    const [posA, colorA] = stops[i];
    const [posB, colorB] = stops[i + 1];
    if (clamped >= posA && clamped <= posB) {
      const frac = (clamped - posA) / (posB - posA || 1);
      const r = Math.round(colorA[0] + (colorB[0] - colorA[0]) * frac);
      const g = Math.round(colorA[1] + (colorB[1] - colorA[1]) * frac);
      const b = Math.round(colorA[2] + (colorB[2] - colorA[2]) * frac);
      return [r, g, b];
    }
  }
  const last = stops[stops.length - 1][1];
  return [last[0], last[1], last[2]];
}

function turboColor(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  const r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
  const g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
  const b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (115.67994485 + x * (-87.60200647 + x * 26.70740952))));
  const clamp = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255);
  return [clamp(r), clamp(g), clamp(b)];
}

const DepthDrawer = memo(function DepthDrawer({ open, onClose, diagnostics, depthData, onRequestDepth, floorplans, onRequestFloorplan }: DepthDrawerProps) {
  const cameras = useMemo(() => Object.keys(diagnostics).sort(), [diagnostics]);
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
  const [floorplanStatus, setFloorplanStatus] = useState<'idle' | 'loading'>('idle');
  const [floorplanRequest, setFloorplanRequest] = useState<string>('');
  const [heatmapRange, setHeatmapRange] = useState<{ min: number; max: number } | null>(null);
  const [heatmapAspect, setHeatmapAspect] = useState<number | null>(null);

  // Floorplan selection (declared early to avoid TDZ in hooks below)
  const cameraFloorplan = floorplans[selectedCamera];
  const densityLayer = cameraFloorplan?.density;
  const heightLayer = cameraFloorplan?.height;
  const distanceLayer = cameraFloorplan?.distance;
  const floorplanError = cameraFloorplan?.error;
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
  const clearCanvasElement = useCallback((canvas: HTMLCanvasElement | null) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const renderTopdownLayer = useCallback(
    (canvas: HTMLCanvasElement | null, layer: FloorplanLayer | undefined, palette: (t: number) => [number, number, number]) => {
      if (!canvas) return;
      if (!layer || !layer.grid_b64 || !layer.grid_shape) {
        clearCanvasElement(canvas);
        return;
      }
      const [rows, cols] = layer.grid_shape;
      if (!rows || !cols) {
        clearCanvasElement(canvas);
        return;
      }
      const values = decodeFloat32(layer.grid_b64);
      if (!values || values.length < rows * cols) {
        clearCanvasElement(canvas);
        return;
      }

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const offscreen = document.createElement('canvas');
      offscreen.width = cols;
      offscreen.height = rows;
      const offCtx = offscreen.getContext('2d');
      if (!offCtx) return;

      const imageData = offCtx.createImageData(cols, rows);
      const data = imageData.data;
      const min = layer.value_min ?? 0;
      const max = layer.value_max ?? 1;
      const denom = max - min === 0 ? 1 : max - min;
      for (let idx = 0; idx < values.length; idx += 1) {
        const norm = Math.min(1, Math.max(0, (values[idx] - min) / denom));
        const [r, g, b] = palette(norm);
        const offset = idx * 4;
        data[offset] = r;
        data[offset + 1] = g;
        data[offset + 2] = b;
        data[offset + 3] = 255;
      }
      offCtx.putImageData(imageData, 0, 0);

      const { width, height, dpr } = applyCanvasSize(canvas);

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, width, height);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(offscreen, 0, 0, width, height);
      ctx.restore();

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.strokeStyle = '#ff4d4d';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(1, 1, width - 2, height - 2);
      ctx.restore();
    },
    [clearCanvasElement]
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
    if (!open || !selectedCamera) return;
    onRequestDepth(selectedCamera);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, selectedCamera]);

  useEffect(() => {
    if (open) {
      setDrawerWidth(DEFAULT_WIDTH);
    }
  }, [open]);

  const depthEntry = selectedCamera ? depthData[selectedCamera] : undefined;
  const summaryEntry = selectedCamera ? diagnostics[selectedCamera] : undefined;

  useEffect(() => {
    const canvas = heatmapCanvasRef.current;
    if (!canvas || !depthEntry || activeTab !== 'heatmap') return;

    const [height, width] = depthEntry.shape;
    const depthArray = decodeFloat32(depthEntry.depth_b64);
    if (!depthArray || depthArray.length < width * height) {
      setHeatmapRange(null);
      setHeatmapAspect(null);
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
      if (!Number.isFinite(d) || d <= 0 || !maskOk) continue;
      if (d < minDepth) minDepth = d;
      if (d > maxDepth) maxDepth = d;
    }

    if (!Number.isFinite(minDepth) || !Number.isFinite(maxDepth) || maxDepth <= minDepth) {
      clearCanvasElement(canvas);
      setHeatmapRange(null);
      setHeatmapAspect(null);
      return;
    }

    const range = maxDepth - minDepth;
    setHeatmapRange({ min: minDepth, max: maxDepth });
    const aspect = width / height;
    setHeatmapAspect(aspect);

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
    const targetHeight = rect.height || canvas.clientHeight || targetWidth / aspect;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(targetWidth * dpr));
    canvas.height = Math.max(1, Math.round(targetHeight * dpr));

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, targetWidth, targetHeight);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(offscreen, 0, 0, targetWidth, targetHeight);
    ctx.restore();
  }, [depthEntry, activeTab, drawerWidth, clearCanvasElement]);

  useEffect(() => {
    if (!depthEntry) {
      setHeatmapRange(null);
      setHeatmapAspect(null);
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

  // (moved above for TDZ safety)

  const renderScale = (gradient: string, min?: number, _mid?: number, max?: number, unit = '') => (
    <div className="color-scale">
      <span className="color-scale__label color-scale__label--max">{formatNumber(max)}{unit}</span>
      <div className="color-scale__bar" style={{ background: gradient }} />
      <span className="color-scale__label color-scale__label--min">{formatNumber(min)}{unit}</span>
    </div>
  );


  const fetchFloorplan = useCallback(() => {
    if (!onRequestFloorplan || !open || !selectedCamera) return;
    setFloorplanStatus('loading');
    const requestId = onRequestFloorplan({
      camera: selectedCamera,
      requestId: Date.now().toString(),
      maxAgeSec: 60,
      gridResM: 0.5,
      maxExtentM: 20,
    });
    if (typeof requestId === 'string' && requestId.length) {
      setFloorplanRequest(requestId);
    } else {
      setFloorplanRequest('');
      setFloorplanStatus('idle');
    }
  }, [onRequestFloorplan, open, selectedCamera]);


  useEffect(() => {
    if (activeTab !== 'heatmap' || !open) return;

    if (floorplanRequest && !cameraFloorplan) {
      clearCanvasElement(densityCanvasRef.current);
      clearCanvasElement(heightCanvasRef.current);
      clearCanvasElement(distanceCanvasRef.current);
      return;
    }

    if (cameraFloorplan?.error) {
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
  }, [activeTab, open, cameraFloorplan, densityLayer, heightLayer, distanceLayer, renderTopdownLayer, clearCanvasElement, floorplanRequest, drawerWidth]);

  useEffect(() => {
    if (activeTab !== 'heatmap' || !open || !selectedCamera) return;
    const id = window.setInterval(() => {
      if (floorplanStatus !== 'loading') {
        fetchFloorplan();
      }
    }, 30000);
    return () => window.clearInterval(id);
  }, [activeTab, open, selectedCamera, floorplanStatus, fetchFloorplan]);

  useEffect(() => {
    if (activeTab !== 'heatmap' || !open || !selectedCamera) return;
    fetchFloorplan();
  }, [selectedCamera, activeTab, open, fetchFloorplan]);

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
                  {cameras.map((cam) => (
                    <option key={cam} value={cam}>{cam}</option>
                  ))}
                </select>
              </label>
              {activeTab === 'heatmap' && (
                <div className="drawer-toolbar__actions">
                  <button
                    type="button"
                    className="btn-icon"
                    onClick={() => selectedCamera && onRequestDepth(selectedCamera)}
                    disabled={!depthEntry}
                    aria-label="Refresh depth frame"
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
                      style={{ aspectRatio: heatmapAspect ? `${heatmapAspect}` : undefined }}
                    />
                  </div>
                </div>
                <div className="heatmap-cell heatmap-cell--right">
                  <div className="heatmap-cell__body">
                    <div className="heatmap-cell__title">Density (Grayscale)</div>
                    <canvas ref={densityCanvasRef} className="heatmap-canvas" />
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
                    <canvas ref={heightCanvasRef} className="heatmap-canvas" />
                  </div>
                </div>
                <div className="heatmap-cell heatmap-cell--right">
                  <div className="heatmap-cell__body">
                    <div className="heatmap-cell__title">Distance (Viridis)</div>
                    <canvas ref={distanceCanvasRef} className="heatmap-canvas" />
                  </div>
                  <div className="heatmap-cell__scale">
                    {renderScale(viridisGradient, distanceMin ?? undefined, distanceMid ?? undefined, distanceMax ?? undefined, ' m')}
                  </div>
                </div>
              </div>

              {/* Meta information below the grid */}
              <div className="heatmap-meta">
                <div className="heatmap-meta__item">
                  {depthEntry
                    ? `Updated ${new Date(depthEntry.ts / 1000).toLocaleTimeString()} · Resolution ${depthEntry.shape[1]}×${depthEntry.shape[0]}`
                    : 'No depth frame cached yet for this camera.'}
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
                  {cameraFloorplan.ts ? ` · Updated ${new Date(cameraFloorplan.ts / 1000).toLocaleTimeString()}` : ''}
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
