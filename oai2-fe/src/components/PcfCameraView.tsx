import React, { memo, useEffect, useMemo, useRef, useState } from 'react';
import { getCameraProjectionInput } from '../lib/calibration';
import { buildPcfCameraProjection } from '../lib/pcfCameraProjection';
import { loadPcfCamera, type LoadedPcfCamera } from '../lib/pcfCameraClient';
import { bevPaletteColor } from '../lib/bevPresentationRaster';
import type { ScientificPalette } from '../lib/dashboardPresentation.js';
import '../styles/camera-snapshot.css';

/** Optical-camera PCF projection, independent of the unified map's raster views. */
export const PcfCameraView = memo(function PcfCameraView({ camera, priorId, calibrationEpoch, imageWidth, imageHeight,
  connected, mode, palette, comparison, onComparisonChange, active }: {
  camera: string; priorId?: string; calibrationEpoch: number; imageWidth?: number; imageHeight?: number;
  connected: boolean; mode: 'depth' | 'normals'; palette: ScientificPalette; comparison: number;
  onComparisonChange?: (value: number) => void; active: boolean;
}) {
  const [retry, setRetry] = useState(0);
  const identity = `${camera}:${priorId}:${calibrationEpoch}:${imageWidth}:${imageHeight}:${retry}`;
  const [load, setLoad] = useState<{ identity: string; result?: LoadedPcfCamera; error?: string }>({ identity: '' });
  const canvasRef = useRef<HTMLCanvasElement>(null), rgbRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!connected || !priorId || !imageWidth || !imageHeight) { setLoad({ identity }); return; }
    let current = true;
    setLoad({ identity });
    try {
      const projection = buildPcfCameraProjection(camera, priorId, imageWidth, imageHeight, getCameraProjectionInput(camera));
      const request = loadPcfCamera(projection);
      request.promise.then(result => { if (current) setLoad({ identity, result }); }, error => { if (current) setLoad({ identity, error: error.message }); });
      return () => { current = false; request.cancel(); };
    } catch (error) { setLoad({ identity, error: error instanceof Error ? error.message : String(error) }); }
  }, [connected, identity]);
  const result = load.identity === identity ? load.result : undefined;
  const error = load.identity === identity ? load.error : undefined;
  const pixels = useMemo(() => {
    if (!active || !result) return null;
    if (mode === 'normals') return result.normalCount ? result.normals : null;
    const rgba = new Uint8ClampedArray(result.width * result.height * 4), { min, max } = result.range;
    for (let i = 0; i < result.depth.length; i++) {
      const z = result.depth[i]; if (!Number.isFinite(z)) continue;
      rgba.set([...bevPaletteColor(palette, max === min ? .5 : (z - min) / (max - min)), 255], i * 4);
    }
    return rgba;
  }, [active, result, mode, palette]);
  useEffect(() => {
    for (const [canvas, data] of [[canvasRef.current, pixels], [rgbRef.current, active && result?.hasColor ? result.rgb : null]] as const) {
      if (!canvas) continue;
      canvas.width = result?.width ?? 1; canvas.height = result?.height ?? 1;
      const ctx = canvas.getContext('2d');
      if (ctx && data) { const image = ctx.createImageData(canvas.width, canvas.height); image.data.set(data); ctx.putImageData(image, 0, 0); }
    }
  }, [active, pixels, result]);
  const available = Boolean(pixels), rgbAvailable = Boolean(available && result?.hasColor);
  const wipe = rgbAvailable && Number.isFinite(comparison) ? Math.max(0, Math.min(100, comparison)) : 0;
  const label = mode === 'depth' ? 'Depth' : 'Normals';
  const message = error === 'pcf_camera_http_403' ? 'Camera-view PCF access is waiting for the gateway update.'
    : error ? 'Camera-view PCF could not be loaded.'
    : result && !pixels ? 'Insufficient PCF surface support for this camera view.'
    : !connected ? 'Waiting for connection.' : 'Loading the saved PCF camera view…';
  const gradient = Array.from({ length: 13 }, (_, i) => `rgb(${bevPaletteColor(palette, i / 12).join(' ')})`).join(', ');
  return <div className="camera-snapshot pcf-camera-view" hidden={!active} data-camera-mode={mode} data-pcf-revision={result?.priorId}
    data-pcf-valid-pixels={result?.validCount} data-pcf-normal-pixels={result?.normalCount} data-pcf-triangles={result?.triangleCount} data-pcf-error={error}>
    <div className="camera-snapshot__viewport pcf-camera-viewport">
      <canvas ref={canvasRef} className="camera-snapshot__canvas" aria-label={`${camera} PCF camera ${label.toLowerCase()}`} hidden={!available} />
      <canvas ref={rgbRef} className="camera-snapshot__canvas camera-snapshot__rgb" style={{ clipPath: `inset(0 ${100 - wipe}% 0 0)` }} hidden={!rgbAvailable} aria-hidden="true" />
      {rgbAvailable && wipe > 0 && wipe < 100 && <div className="camera-snapshot__divider" style={{ left: `${wipe}%` }} aria-hidden="true" />}
    </div>
    <div className="camera-snapshot__source" title={result ? `PCF revision: ${result.priorId}. Locally interpolated surface; unsupported areas remain empty.` : undefined}><strong>Camera POV · {label.toLowerCase()}</strong><span>PCF surface · static</span></div>
    {!available && <div className="camera-snapshot__empty" role="status"><strong>{label} {error || result ? 'unavailable' : 'loading'}</strong><span title={error}>{message}</span>
      {error && <button type="button" className="btn ghost" disabled={!connected} onClick={() => setRetry(value => value + 1)}>Retry camera view</button>}
    </div>}
    {available && result && <div className="camera-snapshot__details">
      {mode === 'depth' ? <div className="camera-snapshot__legend" aria-label="Optical camera depth in metres"><span>{result.range.min.toFixed(2)} m</span><span className="camera-snapshot__ramp" style={{ background: `linear-gradient(90deg, ${gradient})` }} /><span>{result.range.max.toFixed(2)} m</span></div>
        : <div className="camera-snapshot__normal-legend" aria-label="Camera normals: red X right, green Y down, blue Z forward"><span><i className="camera-snapshot__axis-x" />X right</span><span><i className="camera-snapshot__axis-y" />Y down</span><span><i className="camera-snapshot__axis-z" />Z forward</span><span>−1 to +1</span></div>}
      <label className="camera-snapshot__comparison"><span>{rgbAvailable ? 'PCF color' : 'PCF color unavailable'}</span><input type="range" min={0} max={100} value={wipe} disabled={!rgbAvailable || !onComparisonChange}
        aria-label={`${camera} camera PCF color comparison`} onChange={event => onComparisonChange?.(Number(event.currentTarget.value))} /><output>{rgbAvailable ? `${Math.round(wipe)}%` : '—'}</output></label>
    </div>}
  </div>;
});
