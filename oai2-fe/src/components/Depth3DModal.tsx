import React, { useMemo, useRef } from 'react';
import { createPortal } from 'react-dom';
import Depth3DView, { Depth3DViewHandle } from './Depth3DView';
import { HeightMapResponse } from '../types/heightMap';
import '../styles/depth-3d.css';

type ColorMode = 'height' | 'distance' | 'density' | 'none';
type GeometrySource = 'height' | 'distance';

type Depth3DModalProps = {
  open: boolean;
  cameraId?: string;
  cameraLabel?: string;
  heightMap?: HeightMapResponse;
  loading?: boolean;
  error?: string | null;
  onClose: () => void;
  onRefresh: () => void;
  autoRefresh: boolean;
  onToggleAutoRefresh: (enabled: boolean) => void;
  exaggeration: number;
  onChangeExaggeration: (value: number) => void;
  objectThresholdM: number;
  onChangeObjectThreshold: (value: number) => void;
  showEdges: boolean;
  onToggleEdges: (enabled: boolean) => void;
  colorBy: ColorMode;
  onChangeColorBy: (value: ColorMode) => void;
  geometrySource: GeometrySource;
  onChangeGeometrySource: (value: GeometrySource) => void;
  requestedGridResM?: number;
  updatedAt?: number | null;
};

const Depth3DModal: React.FC<Depth3DModalProps> = ({
  open,
  cameraId,
  cameraLabel,
  heightMap,
  loading = false,
  error = null,
  onClose,
  onRefresh,
  autoRefresh,
  onToggleAutoRefresh,
  exaggeration,
  onChangeExaggeration,
  objectThresholdM,
  onChangeObjectThreshold,
  showEdges,
  onToggleEdges,
  colorBy,
  onChangeColorBy,
  geometrySource,
  onChangeGeometrySource,
  requestedGridResM,
  updatedAt,
}) => {
  const viewRef = useRef<Depth3DViewHandle | null>(null);
  const status = useMemo(() => {
    if (error) return error;
    if (loading) return 'Loading height map…';
    if (!heightMap?.data || !heightMap.width || !heightMap.height) return 'No height map yet for this camera.';
    return `Resolution ${heightMap.width}×${heightMap.height} · ${heightMap.served_from_cache ? 'cached' : 'fresh'}`;
  }, [error, heightMap, loading]);

  const zRange = useMemo(() => {
    if (!heightMap) return null;
    if (typeof heightMap.z_min === 'number' && typeof heightMap.z_max === 'number') {
      const floor = typeof heightMap.z_offset === 'number' ? heightMap.z_offset : heightMap.z_min;
      const top = heightMap.z_max;
      if (typeof floor === 'number' && typeof top === 'number') {
        const relMax = top - floor;
        if (Number.isFinite(relMax)) {
          return `0.00–${relMax.toFixed(2)} m above floor`;
        }
      }
      return `${heightMap.z_min.toFixed(2)}–${heightMap.z_max.toFixed(2)} m`;
    }
    return null;
  }, [heightMap]);

  const effectiveGridRes = useMemo(() => {
    return heightMap?.grid_res_m ?? requestedGridResM ?? 0.1;
  }, [heightMap, requestedGridResM]);

  const zMeta = useMemo(() => {
    if (!heightMap) return { span: '—', zMin: undefined as number | undefined, zMax: undefined as number | undefined };
    const zMin = typeof heightMap.z_min === 'number' ? heightMap.z_min : undefined;
    const zMax = typeof heightMap.z_max === 'number' ? heightMap.z_max : undefined;
    const span = zMin !== undefined && zMax !== undefined ? `${(zMax - zMin).toFixed(2)} m` : '—';
    return { span, zMin, zMax };
  }, [heightMap]);

  const gridSizeText = useMemo(() => {
    if (heightMap?.width && heightMap?.height) {
      return `${heightMap.width} × ${heightMap.height}`;
    }
    return '—';
  }, [heightMap]);

  const updatedText = useMemo(() => {
    if (updatedAt) {
      return new Date(updatedAt).toLocaleTimeString();
    }
    if (heightMap?.meta?.generated_at) {
      return heightMap.meta.generated_at;
    }
    return '—';
  }, [heightMap, updatedAt]);

  if (!open) return null;

  return createPortal(
    <div className="depth3d-overlay" role="dialog" aria-modal="true">
      <div className="depth3d-modal">
        <header className="depth3d-header">
          <div>
            <div className="depth3d-title">3D Depth View{cameraLabel ? ` – ${cameraLabel}` : cameraId ? ` – ${cameraId}` : ''}</div>
            <div className="depth3d-subtitle">{status}</div>
          </div>
          <div className="depth3d-controls">
            <button type="button" className="btn ghost" onClick={() => viewRef.current?.setTopDownView()} aria-label="Top-down view">Top</button>
            <button type="button" className="btn ghost" onClick={() => viewRef.current?.setIsometricView()} aria-label="Isometric view">Iso</button>
            <button type="button" className="btn ghost" onClick={() => viewRef.current?.resetView()} aria-label="Reset view">Reset</button>
            <button type="button" className="btn ghost" onClick={onRefresh} disabled={loading}>Refresh</button>
            <label className="depth3d-autorefresh">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => onToggleAutoRefresh(e.target.checked)}
              />
              Auto (10s)
            </label>
            <button type="button" className="btn ghost danger" onClick={onClose} aria-label="Close 3D modal">✕</button>
          </div>
        </header>
        <div className="depth3d-body">
          <div className="depth3d-canvas">
            <Depth3DView
              ref={viewRef}
              heightMap={heightMap}
              exaggeration={exaggeration}
              objectThresholdM={objectThresholdM}
              showEdges={showEdges}
              colorBy={colorBy}
              geometrySource={geometrySource}
              requestedGridResM={requestedGridResM}
            />
            {error && <div className="depth3d-toast error">{error}</div>}
            {loading && !error && <div className="depth3d-toast">Loading…</div>}
          </div>
          <aside className="depth3d-sidebar">
            <div className="depth3d-info">
              <div className="depth3d-meta">
                <div className="depth3d-meta__label">Updated</div>
                <div className="depth3d-meta__value">{updatedText}</div>
              </div>
              <div className="depth3d-meta">
                <div className="depth3d-meta__label">Height range</div>
                <div className="depth3d-meta__value">{zRange || '—'}</div>
              </div>
              <div className="depth3d-meta">
                <div className="depth3d-meta__label">Grid</div>
                <div className="depth3d-meta__value">{gridSizeText}</div>
              </div>
              <div className="depth3d-meta">
                <div className="depth3d-meta__label">Grid res</div>
                <div className="depth3d-meta__value">{`${effectiveGridRes.toFixed(2)} m`}</div>
              </div>
              <div className="depth3d-meta">
                <div className="depth3d-meta__label">Z span</div>
                <div className="depth3d-meta__value">{zMeta.span}</div>
              </div>
              <div className="depth3d-meta">
                <div className="depth3d-meta__label">Data source</div>
                <div className="depth3d-meta__value">
                  {heightMap?.meta?.source || 'mapanything_height_bev_v1'}
                </div>
              </div>
              {heightMap?.meta?.generated_at && (
                <div className="depth3d-meta">
                  <div className="depth3d-meta__label">Generated</div>
                  <div className="depth3d-meta__value">{heightMap.meta.generated_at}</div>
                </div>
              )}
            </div>
            <div className="depth3d-control">
              <div className="depth3d-meta__label">Height exaggeration</div>
              <input
                type="range"
                min={0.5}
                max={4}
                step={0.1}
                value={exaggeration}
                onChange={(e) => onChangeExaggeration(parseFloat(e.target.value))}
              />
              <div className="depth3d-meta__value">{exaggeration.toFixed(1)}×</div>
            </div>
            <div className="depth3d-control">
              <div className="depth3d-meta__label">Floor tolerance</div>
              <input
                type="range"
                min={0}
                max={0.2}
                step={0.01}
                value={objectThresholdM}
                onChange={(e) => onChangeObjectThreshold(parseFloat(e.target.value))}
              />
              <div className="depth3d-meta__value">{objectThresholdM.toFixed(2)} m</div>
            </div>
            <div className="depth3d-control">
              <div className="depth3d-meta__label">Color by</div>
              <select
                value={colorBy}
                onChange={(e) => onChangeColorBy(e.target.value as ColorMode)}
              >
                <option value="height">Height</option>
                <option value="distance">Distance</option>
                <option value="density">Density</option>
                <option value="none">None</option>
              </select>
            </div>
            <label className="depth3d-checkbox">
              <input
                type="checkbox"
                checked={showEdges}
                onChange={(e) => onToggleEdges(e.target.checked)}
              />
              Show mesh edges
            </label>
            {import.meta.env.DEV && (
              <div className="depth3d-control">
                <div className="depth3d-meta__label">Geometry source</div>
                <select
                  value={geometrySource}
                  onChange={(e) => onChangeGeometrySource(e.target.value as GeometrySource)}
                >
                  <option value="height">Height data</option>
                  <option value="distance">Distance data</option>
                </select>
              </div>
            )}
          </aside>
        </div>
      </div>
    </div>,
    document.body
  );
};

export default Depth3DModal;
