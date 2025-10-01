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

export type FloorplanEntry = {
  type?: string;
  request_id?: string;
  grid_b64?: string;
  grid_shape?: [number, number];
  bounds?: { min_x?: number; max_x?: number; min_z?: number; max_z?: number };
  scale_m_per_px?: number;
  use_height?: boolean;
  point_count?: number;
  ts?: number;
  error?: string;
  value_min?: number;
  value_max?: number;
  cameras?: string[];
};

type FloorplanRequestOptions = {
  cameras?: string[];
  useHeight?: boolean;
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
  floorplan: FloorplanEntry | null;
  onRequestFloorplan: (options: FloorplanRequestOptions) => string | void;
}

const VIRIDIS = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 36],
];

const DEFAULT_WIDTH = 400;
const MAX_WIDTH = 960;

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

const DepthDrawer = memo(function DepthDrawer({ open, onClose, diagnostics, depthData, onRequestDepth, floorplan, onRequestFloorplan }: DepthDrawerProps) {
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
  const floorplanCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [showFloorplan, setShowFloorplan] = useState(false);
  const [useHeightColormap, setUseHeightColormap] = useState(false);
  const [floorplanStatus, setFloorplanStatus] = useState<'idle' | 'loading'>('idle');
  const [floorplanRequest, setFloorplanRequest] = useState<string>('');
  const [floorplanCache, setFloorplanCache] = useState<{ density?: FloorplanEntry; height?: FloorplanEntry }>({});

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
    if (!depthArray || depthArray.length < width * height) return;
    const confArray = decodeFloat32(depthEntry.conf_b64);
    const maskArray = decodeUint8(depthEntry.mask_b64);

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;
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
      ctx.clearRect(0, 0, width, height);
      return;
    }
    const range = maxDepth - minDepth;
    for (let i = 0; i < total; i += 1) {
      const d = depthArray[i];
      const idx = i * 4;
      const maskOk = !maskArray || maskArray[i] > 0;
      if (!Number.isFinite(d) || d <= 0 || !maskOk) {
        data[idx + 3] = 0;
        continue;
      }
      const norm = Math.min(1, Math.max(0, (d - minDepth) / range));
      const [r, g, b] = viridisColor(norm);
      let alpha = 0.8;
      if (confArray && confArray.length > i) {
        const conf = Math.max(0, Math.min(1, confArray[i]));
        alpha = 0.25 + conf * 0.75;
      }
      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = Math.round(alpha * 255);
    }
    ctx.putImageData(imageData, 0, 0);
  }, [depthEntry, activeTab]);

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

  const activeFloorplan = useMemo(() => (useHeightColormap ? floorplanCache.height : floorplanCache.density), [floorplanCache, useHeightColormap]);
  const floorplanError = activeFloorplan?.error;

  const clearFloorplanCanvas = useCallback(() => {
    const canvas = floorplanCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const renderFloorplan = useCallback((entry: FloorplanEntry) => {
    const canvas = floorplanCanvasRef.current;
    if (!canvas || !entry || !entry.grid_b64 || !entry.grid_shape || entry.error) return;
    const [rows, cols] = entry.grid_shape;
    if (!rows || !cols) return;
    const values = decodeFloat32(entry.grid_b64);
    if (!values || values.length < rows * cols) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = 400;
    canvas.height = 400;

    const offscreen = document.createElement('canvas');
    offscreen.width = cols;
    offscreen.height = rows;
    const offCtx = offscreen.getContext('2d');
    if (!offCtx) return;

    const imageData = offCtx.createImageData(cols, rows);
    const buffer = imageData.data;
    const count = cols * rows;
    const isHeight = !!entry.use_height;

    let valueMin = typeof entry.value_min === 'number' && Number.isFinite(entry.value_min) ? entry.value_min : undefined;
    let valueMax = typeof entry.value_max === 'number' && Number.isFinite(entry.value_max) ? entry.value_max : undefined;

    if (isHeight && (valueMin === undefined || valueMax === undefined || valueMax <= valueMin)) {
      let minObserved = Number.POSITIVE_INFINITY;
      let maxObserved = Number.NEGATIVE_INFINITY;
      for (let i = 0; i < count; i += 1) {
        const val = values[i];
        if (!Number.isFinite(val)) continue;
        if (val < minObserved) minObserved = val;
        if (val > maxObserved) maxObserved = val;
      }
      if (!Number.isFinite(minObserved) || !Number.isFinite(maxObserved) || maxObserved <= minObserved) {
        minObserved = 0;
        maxObserved = 1;
      }
      valueMin = minObserved;
      valueMax = maxObserved;
    }

    if (!isHeight) {
      valueMin = 0;
      valueMax = 1;
    }

    const range = (valueMax ?? 1) - (valueMin ?? 0);
    const safeRange = range <= 1e-6 ? 1 : range;

    for (let idx = 0; idx < count; idx += 1) {
      const valRaw = values[idx];
      const val = Number.isFinite(valRaw) ? valRaw : 0;
      let r = 255;
      let g = 255;
      let b = 255;
      if (isHeight) {
        const norm = Math.max(0, Math.min(1, ((val - (valueMin ?? 0)) / safeRange)));
        const [vr, vg, vb] = viridisColor(norm);
        r = vr;
        g = vg;
        b = vb;
      } else {
        const norm = Math.max(0, Math.min(1, val));
        const gray = Math.round(255 * (1 - norm));
        r = gray;
        g = gray;
        b = gray;
      }
      const offset = idx * 4;
      buffer[offset] = r;
      buffer[offset + 1] = g;
      buffer[offset + 2] = b;
      buffer[offset + 3] = 255;
    }

    offCtx.putImageData(imageData, 0, 0);

    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);

    // Draw bounding box outline
    ctx.strokeStyle = '#ff4d4d';
    ctx.lineWidth = 2;
    ctx.strokeRect(1, 1, canvas.width - 2, canvas.height - 2);

    const bounds = entry.bounds || {};
    const widthMeters = typeof bounds.max_x === 'number' && typeof bounds.min_x === 'number' ? bounds.max_x - bounds.min_x : 0;
    const heightMeters = typeof bounds.max_z === 'number' && typeof bounds.min_z === 'number' ? bounds.max_z - bounds.min_z : 0;
    ctx.fillStyle = 'rgba(255,255,255,0.82)';
    ctx.fillRect(8, 8, 150, 36);
    ctx.fillStyle = '#111';
    ctx.font = '12px system-ui';
    ctx.fillText(`X span: ${widthMeters.toFixed(1)} m`, 14, 22);
    ctx.fillText(`Z span: ${heightMeters.toFixed(1)} m`, 14, 36);
    ctx.fillText(isHeight ? 'Height colormap' : 'Density', canvas.width - 130, 22);

    if (widthMeters > 0.01) {
      const barMeters = widthMeters >= 15 ? 5 : widthMeters >= 7 ? 2 : 1;
      const pixelsPerMeter = canvas.width / widthMeters;
      const barPx = Math.max(12, Math.min(canvas.width - 40, pixelsPerMeter * barMeters));
      const barX = 24;
      const barY = canvas.height - 32;
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      ctx.fillRect(barX, barY, barPx, 8);
      ctx.strokeStyle = '#222';
      ctx.lineWidth = 1;
      ctx.strokeRect(barX, barY, barPx, 8);
      ctx.fillStyle = '#111';
      ctx.fillText(`${barMeters} m`, barX, barY - 4);
    }

    ctx.restore();
  }, []);

  useEffect(() => {
    if (!floorplan || floorplan.type !== 'floorplan_response') return;
    const key = floorplan.use_height ? 'height' : 'density';
    setFloorplanCache(prev => ({ ...prev, [key === 'height' ? 'height' : 'density']: floorplan }));
    if (!floorplan.request_id || floorplan.request_id === floorplanRequest || floorplanRequest === 'pending') {
      setFloorplanStatus('idle');
      setFloorplanRequest('');
    }
  }, [floorplan, floorplanRequest]);

  const fetchFloorplan = useCallback((opts?: { useHeight?: boolean }) => {
    if (!onRequestFloorplan || !open) return;
    setFloorplanCache({});
    clearFloorplanCanvas();
    setFloorplanStatus('loading');
    const targetUseHeight = opts?.useHeight ?? useHeightColormap;
    const cameraList = selectedCamera ? [selectedCamera] : [];
    const requestId = onRequestFloorplan({
      cameras: cameraList,
      useHeight: targetUseHeight,
      requestId: Date.now().toString(),
      maxAgeSec: 60,
      gridResM: 0.5,
      maxExtentM: 20,
    });
    if (typeof requestId === 'string' && requestId.length) {
      setFloorplanRequest(requestId);
    } else if (requestId === undefined) {
      setFloorplanRequest('pending');
    } else {
      setFloorplanRequest('');
      setFloorplanStatus('idle');
    }
  }, [onRequestFloorplan, open, selectedCamera, useHeightColormap, clearFloorplanCanvas]);

  useEffect(() => {
    if (!showFloorplan) return;
    setFloorplanCache({});
    clearFloorplanCanvas();
  }, [selectedCamera, showFloorplan, clearFloorplanCanvas]);

  useEffect(() => {
    if (!showFloorplan || activeTab !== 'heatmap' || !open) {
      if (!showFloorplan) {
        setFloorplanStatus('idle');
        setFloorplanRequest('');
      }
      return;
    }
    if (activeFloorplan && !activeFloorplan.error) {
      renderFloorplan(activeFloorplan);
    } else if (!activeFloorplan) {
      clearFloorplanCanvas();
    }
  }, [showFloorplan, activeTab, open, activeFloorplan, renderFloorplan, clearFloorplanCanvas]);

  useEffect(() => {
    if (!showFloorplan || activeTab !== 'heatmap' || !open) return;
    const id = window.setInterval(() => {
      fetchFloorplan();
    }, 30000);
    return () => window.clearInterval(id);
  }, [showFloorplan, activeTab, open, fetchFloorplan]);

  useEffect(() => {
    if (!showFloorplan || activeTab !== 'heatmap' || !open) return;
    fetchFloorplan();
  }, [selectedCamera, showFloorplan, activeTab, open, fetchFloorplan]);

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
            <div className="camera-select">
              <label htmlFor="ma-depth-select">Camera</label>
              <select
                id="ma-depth-select"
                value={selectedCamera}
                onChange={(ev) => setSelectedCamera(ev.target.value)}
              >
                {cameras.map((cam) => (
                  <option key={cam} value={cam}>{cam}</option>
                ))}
              </select>
            </div>
          )}

          {activeTab === 'heatmap' && (
            <>
              {depthEntry ? (
                <>
                  <canvas ref={heatmapCanvasRef} className="heatmap" />
                  <p style={{ fontSize: '12px', opacity: 0.7 }}>
                    Updated {new Date(depthEntry.ts / 1000).toLocaleTimeString()} · resolution {depthEntry.shape[1]}×{depthEntry.shape[0]}
                  </p>
                  <button className="btn ghost" onClick={() => selectedCamera && onRequestDepth(selectedCamera)}>Refresh</button>
                </>
              ) : (
                <p style={{ fontSize: '12px', opacity: 0.7 }}>No depth frame cached yet for this camera.</p>
              )}
              <div style={{ marginTop: '20px', borderTop: '1px solid rgba(255,255,255,0.15)', paddingTop: '12px' }}>
                <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px' }}>
                    <input
                      type="checkbox"
                      checked={showFloorplan}
                      onChange={(ev) => {
                        const next = ev.target.checked;
                        setShowFloorplan(next);
                        if (next) {
                          setFloorplanStatus('loading');
                        }
                        if (next) {
                          const cached = useHeightColormap ? floorplanCache.height : floorplanCache.density;
                          if (cached && !cached.error) {
                            renderFloorplan(cached);
                          }
                        } else {
                          clearFloorplanCanvas();
                          setFloorplanStatus('idle');
                          setFloorplanRequest('');
                        }
                      }}
                    />
                    Show Floorplan (Top-down)
                  </label>
                  {showFloorplan && (
                    <>
                      <button className="btn ghost" onClick={() => fetchFloorplan()}>Refresh Floorplan</button>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px' }}>
                        <input
                          type="checkbox"
                          checked={useHeightColormap}
                          onChange={(ev) => {
                            const next = ev.target.checked;
                            setUseHeightColormap(next);
                            if (showFloorplan) {
                              const cached = next ? floorplanCache.height : floorplanCache.density;
                              if (cached && !cached.error) {
                                renderFloorplan(cached);
                              }
                              fetchFloorplan({ useHeight: next });
                            }
                          }}
                        />
                        Height Colormap
                      </label>
                    </>
                  )}
                </div>
                {showFloorplan && (
                  <div style={{ marginTop: '12px' }}>
                    <canvas
                      ref={floorplanCanvasRef}
                      width={400}
                      height={400}
                      style={{ width: '100%', maxWidth: '400px', border: '1px solid rgba(255,255,255,0.2)', background: '#111' }}
                    />
                    {floorplanStatus === 'loading' && (
                      <p style={{ fontSize: '12px', marginTop: '6px' }}>Loading floorplan…</p>
                    )}
                    {floorplanError && (
                      <p style={{ fontSize: '12px', color: '#ff5c5c', marginTop: '6px' }}>Error: {floorplanError}</p>
                    )}
                    {!floorplanError && activeFloorplan && floorplanStatus !== 'loading' && (
                      <p style={{ fontSize: '12px', opacity: 0.75, marginTop: '6px' }}>
                        Points: {activeFloorplan.point_count ?? 0} · Updated {activeFloorplan.ts ? new Date(activeFloorplan.ts / 1000).toLocaleTimeString() : 'n/a'}
                      </p>
                    )}
                    {!floorplanError && !activeFloorplan && floorplanStatus !== 'loading' && (
                      <p style={{ fontSize: '12px', opacity: 0.75, marginTop: '6px' }}>Enable and refresh to generate the floorplan.</p>
                    )}
                  </div>
                )}
              </div>
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
