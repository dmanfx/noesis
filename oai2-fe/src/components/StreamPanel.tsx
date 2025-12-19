import React, { useEffect, useMemo, useRef, useState } from 'react';
import { CameraKey, cameraLabel } from '../lib/camera';

export type StreamMode = 'jpeg' | 'webrtc';

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
  // Stream mode: 'jpeg' uses blob/img, 'webrtc' uses videoRef
  streamMode?: StreamMode;
  // Video ref for WebRTC mode (from useWebRTCClient)
  videoRef?: React.RefObject<HTMLVideoElement>;
}> = ({
  camera,
  title,
  blob,
  fpsText,
  fpsSeries = [],
  vacancyText,
  isExpanded = false,
  onToggleExpand,
  streamMode = 'jpeg',
  videoRef: externalVideoRef,
}) => {
    const [url, setUrl] = useState<string>('');
    const [aspect, setAspect] = useState<string | null>(null);
    const imgRef = useRef<HTMLImageElement>(null);
    const internalVideoRef = useRef<HTMLVideoElement>(null);
    const viewRef = useRef<HTMLDivElement>(null);
    const sparkRef = useRef<HTMLCanvasElement>(null);

    // Use external video ref if provided, otherwise use internal
    const videoRef = externalVideoRef || internalVideoRef;

    useEffect(() => {
      if (streamMode !== 'jpeg' || !blob) return;
      const objectUrl = URL.createObjectURL(blob);
      if (imgRef.current) { imgRef.current.src = objectUrl; }
      const prev = url;
      setUrl(objectUrl);
      if (prev) URL.revokeObjectURL(prev);
      return () => URL.revokeObjectURL(objectUrl);
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [blob, streamMode]);

    useEffect(() => {
      if (streamMode !== 'jpeg') return;
      const img = imgRef.current;
      if (!img) return;

      const update = () => {
        const w = img.naturalWidth || 0;
        const h = img.naturalHeight || 0;
        if (w > 0 && h > 0) setAspect(`${w} / ${h}`);
      };
      img.addEventListener('load', update);
      update();
      return () => img.removeEventListener('load', update);
    }, [streamMode, blob]);

    useEffect(() => {
      if (streamMode !== 'webrtc') return;
      const v = videoRef.current;
      if (!v) return;

      const update = () => {
        const w = v.videoWidth || 0;
        const h = v.videoHeight || 0;
        if (w > 0 && h > 0) setAspect(`${w} / ${h}`);
      };
      v.addEventListener('loadedmetadata', update);
      v.addEventListener('resize', update);
      update();
      return () => {
        v.removeEventListener('loadedmetadata', update);
        v.removeEventListener('resize', update);
      };
    }, [streamMode, videoRef]);

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
        const y = h - ((v - min) / range) * (h - 2 * dpr) - dpr;
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
            {streamMode === 'webrtc' && (
              <span className="chip" style={{ color: '#4caf50', fontSize: '0.7em' }}>WebRTC</span>
            )}
            <button
              className="btn ghost"
              onClick={() => onToggleExpand?.(camera)}
              title="Toggle expand"
            >
              {isExpanded ? 'Collapse' : 'Expand'}
            </button>
          </div>
        </div>
        <div
          className="stream-view"
          ref={viewRef}
          style={aspect ? ({ ['--stream-aspect' as any]: aspect } as React.CSSProperties) : undefined}
        >
          {streamMode === 'jpeg' ? (
            <img ref={imgRef} alt={`${label} stream`} />
          ) : (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
            />
          )}
        </div>
        <div className="sparkbar"><canvas ref={sparkRef} /></div>
      </div>
    );
  };
