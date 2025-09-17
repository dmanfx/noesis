import React, { useEffect, useMemo, useRef, useState } from 'react';
import { CameraKey, cameraLabel } from '../lib/camera';

export const StreamPanel: React.FC<{
  camera: CameraKey;
  title?: string;
  blob?: Blob | null;
  fpsText?: string;
  fpsSeries?: number[];
  vacancyText?: string;
  // Expand and Lock are controlled by parent
  isExpanded?: boolean;
  onToggleExpand?: (camera: CameraKey) => void;
  showLock?: boolean;
  locked?: boolean;
  onToggleLock?: () => void;
}> = ({ camera, title, blob, fpsText, fpsSeries = [], vacancyText, isExpanded = false, onToggleExpand, showLock = false, locked = false, onToggleLock }) => {
  const [url, setUrl] = useState<string>('');
  const imgRef = useRef<HTMLImageElement>(null);
  const viewRef = useRef<HTMLDivElement>(null);
  const sparkRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!blob) return;
    const objectUrl = URL.createObjectURL(blob);
    if (imgRef.current) { imgRef.current.src = objectUrl; }
    const prev = url;
    setUrl(objectUrl);
    if (prev) URL.revokeObjectURL(prev);
    return () => URL.revokeObjectURL(objectUrl);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [blob]);

  const label = useMemo(() => title ?? cameraLabel(camera), [camera, title]);

  // Draw FPS sparkline in dedicated bar below the video
  useEffect(() => {
    const c = sparkRef.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth * dpr;
    const h = c.clientHeight * dpr;
    c.width = w; c.height = h;
    const ctx = c.getContext('2d'); if (!ctx) return;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = '#3aa0ff';
    ctx.lineWidth = 1.25 * dpr;
    const values = fpsSeries.slice(-60);
    if (!values.length) return;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = Math.max(1e-3, max - min);
    const step = w / Math.max(1, values.length - 1);
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = i * step;
      const y = h - ((v - min) / range) * (h - 2*dpr) - dpr;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }, [fpsSeries]);

  return (
    <div className="panel stream-card">
      <div className="stream-head">
        <div className="chip" title={camera}>{label}</div>
        <div className="chip mono" style={{ color: '#9db1c8' }}>{fpsText ?? 'FPS: 0.0'}</div>
        {vacancyText ? (
          <div
            className="chip mono"
            style={{ color: '#9db1c8' }}
            title={`Vacant for ${vacancyText}`}
          >
            {vacancyText}
          </div>
        ) : null}
        <div className="spacer" />
        <div className="stream-tools">
          {showLock ? (
            <button
              className="btn ghost"
              onClick={onToggleLock}
              title={locked ? 'Unlock stream order' : 'Lock stream order'}
              aria-pressed={locked}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                {locked ? (
                  <>
                    <path d="M7 10V7a5 5 0 0 1 10 0v3" stroke="#cfe0ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <rect x="5" y="10" width="14" height="10" rx="2" stroke="#cfe0ff" strokeWidth="2"/>
                  </>
                ) : (
                  <>
                    <path d="M7 10V7a5 5 0 0 1 9 0" stroke="#9db1c8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <rect x="5" y="10" width="14" height="10" rx="2" stroke="#9db1c8" strokeWidth="2"/>
                  </>
                )}
              </svg>
            </button>
          ) : null}
          <button
            className="btn ghost"
            onClick={() => onToggleExpand?.(camera)}
            title="Toggle expand"
          >
            {isExpanded ? 'Collapse' : 'Expand'}
          </button>
        </div>
      </div>
      <div className="stream-view" ref={viewRef}>
        <img ref={imgRef} alt={`${label} stream`} />
      </div>
      <div className="sparkbar"><canvas ref={sparkRef} /></div>
    </div>
  );
};
