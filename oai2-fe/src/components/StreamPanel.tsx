import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CameraKey, cameraLabel } from '../lib/camera';
import { containRect, resolveMosaicTileCrop, type MosaicLayoutLike } from '../lib/mosaic';

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
  tileCameras?: CameraKey[];
  expandedTileCamera?: CameraKey | null;
  onToggleTileExpand?: (camera: CameraKey) => void;
  mosaicLayout?: MosaicLayoutLike | null;
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
  tileCameras = [],
  expandedTileCamera = null,
  onToggleTileExpand,
  mosaicLayout,
  streamMode = 'jpeg',
  videoRef: externalVideoRef,
}) => {
    const [url, setUrl] = useState<string>('');
    const [aspect, setAspect] = useState<string | null>(null);
    const imgRef = useRef<HTMLImageElement>(null);
    const internalVideoRef = useRef<HTMLVideoElement>(null);
    const viewRef = useRef<HTMLDivElement>(null);
    const sparkRef = useRef<HTMLCanvasElement>(null);
    const [viewMetrics, setViewMetrics] = useState({ viewW: 0, viewH: 0, videoW: 0, videoH: 0 });

    // Use external video ref if provided, otherwise use internal
    const videoRef = externalVideoRef || internalVideoRef;

    const updateViewMetrics = useCallback(() => {
      const view = viewRef.current;
      if (!view) return;
      const rect = view.getBoundingClientRect();
      const v = videoRef.current;
      const next = {
        viewW: Math.max(0, Math.round(rect.width)),
        viewH: Math.max(0, Math.round(rect.height)),
        videoW: Math.max(0, Math.round(v?.videoWidth || 0)),
        videoH: Math.max(0, Math.round(v?.videoHeight || 0)),
      };
      setViewMetrics((prev) => (
        prev.viewW === next.viewW &&
        prev.viewH === next.viewH &&
        prev.videoW === next.videoW &&
        prev.videoH === next.videoH
          ? prev
          : next
      ));
    }, [videoRef]);

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
      v.addEventListener('loadeddata', updateViewMetrics);
      v.addEventListener('loadedmetadata', updateViewMetrics);
      v.addEventListener('resize', updateViewMetrics);
      update();
      updateViewMetrics();
      return () => {
        v.removeEventListener('loadedmetadata', update);
        v.removeEventListener('resize', update);
        v.removeEventListener('loadeddata', updateViewMetrics);
        v.removeEventListener('loadedmetadata', updateViewMetrics);
        v.removeEventListener('resize', updateViewMetrics);
      };
    }, [streamMode, updateViewMetrics, videoRef]);

    useEffect(() => {
      const view = viewRef.current;
      if (!view) return;
      updateViewMetrics();
      const resizeObserver = typeof ResizeObserver !== 'undefined'
        ? new ResizeObserver(updateViewMetrics)
        : null;
      resizeObserver?.observe(view);
      window.addEventListener('resize', updateViewMetrics);
      return () => {
        resizeObserver?.disconnect();
        window.removeEventListener('resize', updateViewMetrics);
      };
    }, [updateViewMetrics]);

    const label = useMemo(() => title ?? cameraLabel(camera), [camera, title]);

    const tileButtons = useMemo(() => {
      if (streamMode !== 'webrtc' || !mosaicLayout || !tileCameras.length) return [];
      const { viewW, viewH, videoW, videoH } = viewMetrics;
      if (viewW <= 0 || viewH <= 0 || videoW <= 0 || videoH <= 0) return [];
      const mediaRect = containRect(viewW, viewH, videoW, videoH);
      const size = 34;
      return tileCameras.flatMap((cam) => {
        const crop = resolveMosaicTileCrop(mosaicLayout, cam, videoW, videoH);
        if (!crop) return [];
        const right = mediaRect.x + ((crop.sx + crop.sw) / videoW) * mediaRect.w;
        const top = mediaRect.y + (crop.sy / videoH) * mediaRect.h;
        const minLeft = mediaRect.x + 8;
        const maxLeft = mediaRect.x + mediaRect.w - size - 8;
        const minTop = mediaRect.y + 8;
        const maxTop = mediaRect.y + mediaRect.h - size - 8;
        return [{
          camera: cam,
          label: cameraLabel(cam),
          left: Math.min(maxLeft, Math.max(minLeft, right - size - 10)),
          top: Math.min(maxTop, Math.max(minTop, top + 10)),
          size,
        }];
      });
    }, [mosaicLayout, streamMode, tileCameras, viewMetrics]);

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
          <div className="chip" title={title ?? camera}>{label}</div>
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
            {onToggleExpand ? (
              <button
                type="button"
                className={`icon-btn mosaic-fullscreen-toggle${isExpanded ? ' is-active' : ''}`}
                onClick={() => onToggleExpand(camera)}
                title={isExpanded ? 'Exit mosaic fullscreen' : 'Fullscreen mosaic'}
                aria-label={isExpanded ? 'Exit mosaic fullscreen' : 'Fullscreen mosaic'}
              >
                <span className="fullscreen-corners" aria-hidden="true" />
              </button>
            ) : null}
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
          {tileButtons.length > 0 && (
            <div className="mosaic-tile-actions" aria-hidden={false}>
              {tileButtons.map((button) => (
                <button
                  key={`tile-expand-${button.camera}`}
                  type="button"
                  className={`mosaic-tile-expand${expandedTileCamera === button.camera ? ' is-active' : ''}`}
                  style={{ left: button.left, top: button.top, width: button.size, height: button.size }}
                  onClick={() => onToggleTileExpand?.(button.camera)}
                  title={`Fullscreen ${button.label}`}
                  aria-label={`Fullscreen ${button.label}`}
                >
                  <span className="fullscreen-corners" aria-hidden="true" />
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="sparkbar"><canvas ref={sparkRef} /></div>
      </div>
    );
  };
