import { memo, useEffect, useMemo, useRef, useState } from 'react';
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

interface DepthDrawerProps {
  open: boolean;
  onClose: () => void;
  diagnostics: Record<string, DiagnosticsEntry>;
  depthData: Record<string, DepthEntry>;
  onRequestDepth: (cameraId: string) => void;
}

const VIRIDIS = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 36],
];

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

const DepthDrawer = memo(function DepthDrawer({ open, onClose, diagnostics, depthData, onRequestDepth }: DepthDrawerProps) {
  const cameras = useMemo(() => Object.keys(diagnostics).sort(), [diagnostics]);
  const [activeTab, setActiveTab] = useState<'heatmap' | 'stats' | 'histogram' | 'metrics'>('heatmap');
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const heatmapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const histogramCanvasRef = useRef<HTMLCanvasElement | null>(null);

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

  return (
    <>
      <div className={`depth-drawer-overlay ${open ? 'open' : ''}`} onClick={onClose} />
      <aside className={`depth-drawer ${open ? 'open' : ''}`}>
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

          {activeTab === 'heatmap' && depthEntry && (
            <>
              <canvas ref={heatmapCanvasRef} className="heatmap" />
              <p style={{ fontSize: '12px', opacity: 0.7 }}>
                Updated {new Date(depthEntry.ts / 1000).toLocaleTimeString()} · resolution {depthEntry.shape[1]}×{depthEntry.shape[0]}
              </p>
              <button className="btn ghost" onClick={() => selectedCamera && onRequestDepth(selectedCamera)}>Refresh</button>
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
