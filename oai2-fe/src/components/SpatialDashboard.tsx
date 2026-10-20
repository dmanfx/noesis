import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { UnifiedBevView } from './UnifiedBevView';
import { MosaicCropCanvas } from './MosaicCropCanvas';
import { PcfCameraView } from './PcfCameraView';
import { ChoiceStrip } from './ChoiceStrip';
import type { FloorplanResponse } from './DepthDrawer';
import { cameraDisplayMode, defaultPresentationSettings, type PresentationSettings, type CameraDisplayMode } from '../lib/dashboardPresentation.js';
import { cameraLabel, detectCameraKey, type CameraKey } from '../lib/camera';
import { normalizeMosaicCameraId, resolveMosaicTileIndex, type MosaicLayoutLike } from '../lib/mosaic';
import { fitUnifiedBevViewport, type UnifiedBevDisplayState } from '../lib/unifiedBevConsumer.js';
import { projectUnifiedBevMapPoint } from '../lib/unifiedBevWorld.js';
import type { BevTrailConfig } from '../lib/bevTrails';
import '../styles/spatial-dashboard.css';

type Point = { x: number; y: number };
type Props = {
  display: UnifiedBevDisplayState; connection: string; cameras: CameraKey[];
  mosaicLayout: MosaicLayoutLike | null; videoRef: React.RefObject<HTMLVideoElement>;
  videoConnected: boolean; mosaicFps?: number | null; fps: Record<string, string>; vacancyText: Record<string, string>;
  trailEnabled: boolean; trailConfig?: Partial<BevTrailConfig>; debug?: boolean;
  selectedRoom?: string; onSelectRoom?: (camera: string) => void;
  active?: boolean; presentation?: PresentationSettings;
  floorplans?: Record<string, FloorplanResponse>;
  calibrationEpoch?: number;
  onCameraModeChange?: (camera: string, mode: CameraDisplayMode) => void;
  onComparisonChange?: (value: number) => void; onLoadSnapshot?: (camera: string) => void;
  onToggleCameras?: () => void;
};

/** One decoded mosaic stays mounted; all floating views reuse its existing tile crops. */
export function SpatialDashboard(props: Props) {
  const { display, cameras, mosaicLayout, videoRef, videoConnected } = props;
  const presentation = props.presentation ?? defaultPresentationSettings();
  const root = useRef<HTMLElement>(null);
  const panels = useRef(new Map<string, HTMLElement>());
  const [size, setSize] = useState({ width: 0, height: 0, mapX: 0, mapY: 0, mapW: 0, mapH: 0 });
  const [positions, setPositions] = useState<Record<string, Point>>({});
  const [expanded, setExpanded] = useState<string | null>(null);
  const [fullscreen, setFullscreen] = useState<string | null>(null);
  const [hidden, setHidden] = useState(false);
  const camerasHidden = hidden || !presentation.cameras;
  const [highlight, setHighlight] = useState<string | null>(null);
  const [notice, setNotice] = useState('');
  const [videoReady, setVideoReady] = useState(false);
  const compact = size.width > 0 && size.width < 720;
  const panelW = compact ? Math.max(180, size.width - 24) : Math.min(488, Math.max(309, size.width * .382));
  const panelH = panelW * 9 / 16 + 118;

  useLayoutEffect(() => {
    const element = root.current;
    const stage = element?.querySelector('.unified-bev-canvas-wrap');
    const canvas = element?.querySelector<HTMLCanvasElement>('.unified-bev-canvas-wrap > canvas');
    if (!stage || !canvas) return;
    const measure = () => {
      const s = stage.getBoundingClientRect(), c = canvas.getBoundingClientRect();
      setSize({ width: s.width, height: s.height, mapX: c.x - s.x, mapY: c.y - s.y, mapW: c.width, mapH: c.height });
    };
    const observer = new ResizeObserver(measure); observer.observe(stage); observer.observe(canvas); measure();
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const video = videoRef.current; if (!video) return;
    const ready = () => setVideoReady(video.readyState >= 2 && video.videoWidth > 0);
    video.addEventListener('loadeddata', ready); video.addEventListener('emptied', ready); ready();
    return () => { video.removeEventListener('loadeddata', ready); video.removeEventListener('emptied', ready); };
  }, [videoRef]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') setExpanded(null); };
    const onFullscreen = () => setFullscreen(document.fullscreenElement?.getAttribute('data-camera') ?? null);
    window.addEventListener('keydown', onKey); document.addEventListener('fullscreenchange', onFullscreen);
    return () => { window.removeEventListener('keydown', onKey); document.removeEventListener('fullscreenchange', onFullscreen); };
  }, []);

  useEffect(() => {
    if (!camerasHidden) return;
    setExpanded(null); setHighlight(null);
    const element = document.fullscreenElement;
    if (element && root.current?.contains(element)) {
      void document.exitFullscreen().catch(() => setNotice('Exit fullscreen to return to the dashboard.'));
    }
  }, [camerasHidden]);

  // These are presentation anchors from already admitted static maps, never world estimates.
  const viewport = fitUnifiedBevViewport(display.world, size.mapW, size.mapH);
  const anchors = new Map<string, Point>();
  cameras.forEach(camera => {
    const map = display.world.maps.find(m => normalizeMosaicCameraId(m.cameraId) === normalizeMosaicCameraId(camera));
    if (!map || !viewport) return;
    // The admitted local map's origin is the calibrated camera's floor footprint.
    // Reuse its revision-bound scene transform, rather than estimating a room center.
    const cameraFootprint = projectUnifiedBevMapPoint(map, [0, 0]);
    const [x, y] = viewport.point(cameraFootprint); anchors.set(camera, { x: x + size.mapX, y: y + size.mapY });
  });
  const defaultPositions = new Map<string, Point>();
  // Order each side by its room's vertical position. Static corners keep this stable as people move.
  const staticViewport = fitUnifiedBevViewport({ ...display.world, people: [] }, size.mapW, size.mapH);
  const staticAnchors = cameras.map((camera, i) => {
    const map = display.world.maps.find(m => normalizeMosaicCameraId(m.cameraId) === normalizeMosaicCameraId(camera));
    const points = map?.sceneCorners;
    const center = points?.length ? points.reduce((a, p) => [a[0] + p[0] / points.length, 0, a[2] + p[2] / points.length] as [number, number, number], [0, 0, 0] as [number, number, number]) : null;
    const point = center && staticViewport ? staticViewport.point(center) : null;
    return { camera, side: point ? (point[0] < size.mapW / 2 ? 'left' : 'right') : (i % 2 ? 'right' : 'left'), y: point?.[1] ?? i };
  });
  for (const side of ['left', 'right']) {
    const group = staticAnchors.filter(a => a.side === side).sort((a, b) => a.y - b.y);
    group.forEach((a, index) => defaultPositions.set(a.camera, {
      x: side === 'left' ? 14 : Math.max(14, size.width - panelW - 14),
      y: 18 + index * Math.max(panelH + 14, (size.height - panelH - 36) / Math.max(1, group.length - 1)),
    }));
  }
  const position = (camera: string): Point => {
    const p = positions[camera] ?? defaultPositions.get(camera) ?? { x: 14, y: 18 };
    return { x: Math.max(0, Math.min(Math.max(0, size.width - panelW), p.x)), y: Math.max(0, Math.min(Math.max(0, size.height - panelH), p.y)) };
  };
  const drag = (event: React.PointerEvent<HTMLElement>, camera: string) => {
    if (event.button !== 0 || (event.target as HTMLElement).closest('button') || expanded === camera || fullscreen === camera || compact) return;
    const header = event.currentTarget, start = position(camera), x = event.clientX, y = event.clientY;
    header.setPointerCapture(event.pointerId);
    const move = (e: PointerEvent) => setPositions(prev => ({ ...prev, [camera]: {
      x: Math.max(0, Math.min(size.width - panelW, start.x + e.clientX - x)),
      y: Math.max(0, Math.min(size.height - panelH, start.y + e.clientY - y)),
    } }));
    const stop = () => { header.removeEventListener('pointermove', move); header.removeEventListener('pointerup', stop); header.removeEventListener('pointercancel', stop); };
    header.addEventListener('pointermove', move); header.addEventListener('pointerup', stop); header.addEventListener('pointercancel', stop);
  };
  const openFullscreen = async (camera: string) => {
    setNotice('');
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await panels.current.get(camera)?.requestFullscreen();
    } catch { setNotice('Fullscreen is unavailable in this browser. Use Expand to inspect the stream.'); }
  };

  const rows = Math.max(...['left', 'right'].map(side => staticAnchors.filter(a => a.side === side).length), 1);
  return <section className="spatial-dashboard" ref={root} aria-label="Spatial dashboard"
    style={{ '--spatial-stage-min-height': `${Math.max(590, rows * (panelH + 14) + 36)}px` } as React.CSSProperties}>
    <video ref={videoRef} className="spatial-mosaic-source" autoPlay playsInline muted aria-hidden="true" />
    <div className="spatial-toolbar">
      <span className="subtitle">Home / Spatial overview{videoConnected && props.mosaicFps != null ? ` · Mosaic ${props.mosaicFps.toFixed(1)} FPS` : ''}</span>
      <div><button className="btn ghost" onClick={() => { if (props.onToggleCameras) props.onToggleCameras(); else setHidden(v => !v); setExpanded(null); }}>{camerasHidden ? 'Show cameras' : 'Hide cameras'}</button>
        <button className="btn ghost" onClick={() => { setPositions({}); setExpanded(null); }}>Reset layout</button></div>
    </div>
    <UnifiedBevView display={display} connection={props.connection} trailEnabled={props.trailEnabled} trailConfig={props.trailConfig} debug={props.debug} presentation={presentation}>
      <div className={`spatial-overlays${compact ? ' spatial-overlays--compact' : ''}`}>
        {!compact && !camerasHidden && <svg className="spatial-connectors" aria-hidden="true">
          {cameras.map(camera => { const a = anchors.get(camera), p = position(camera); return a && <g key={camera} className={highlight === camera ? 'is-highlighted' : ''}>
            <line x1={p.x + panelW / 2} y1={p.y + panelH / 2} x2={a.x} y2={a.y} /><circle cx={a.x} cy={a.y} r="4" />
            {presentation.labels && <text x={a.x + 9} y={a.y - 8}>{cameraLabel(detectCameraKey(camera) ?? camera)}</text>}
          </g>; })}
        </svg>}
        {expanded && !camerasHidden && <button className="spatial-scrim" aria-label="Close expanded camera" onClick={() => setExpanded(null)} />}
        {cameras.map(camera => {
          const p = position(camera), isExpanded = expanded === camera;
          const mode = cameraDisplayMode(presentation, camera);
          const pcf = props.floorplans?.[camera];
          const snapshotError = pcf?.error;
          const available = resolveMosaicTileIndex(mosaicLayout, camera) !== null;
          const videoStatus = !available ? 'Camera tile unavailable' : !videoConnected || !videoReady ? 'Waiting for video' : null;
          return <article key={camera} data-camera={camera} ref={el => { if (el) panels.current.set(camera, el); else panels.current.delete(camera); }}
            hidden={camerasHidden} className={`spatial-camera${isExpanded ? ' spatial-camera--expanded' : ''}${props.selectedRoom === camera ? ' spatial-camera--selected' : ''}`}
            style={{ left: p.x, top: p.y, width: panelW }} aria-label={`${cameraLabel(camera)} camera`}
            onMouseEnter={() => setHighlight(camera)} onMouseLeave={() => setHighlight(null)} onFocus={() => setHighlight(camera)} onBlur={() => setHighlight(null)}>
            <header onPointerDown={event => drag(event, camera)}>
              {props.onSelectRoom ? <button type="button" className="spatial-camera-title spatial-camera-select"
                aria-label={`Inspect ${cameraLabel(camera)}`} aria-pressed={props.selectedRoom === camera}
                onClick={() => props.onSelectRoom?.(camera)}>{cameraLabel(camera)}</button> : <span className="spatial-camera-title">{cameraLabel(camera)}</span>}
              <div className="spatial-camera-actions"><button className="btn ghost" onClick={() => setExpanded(isExpanded ? null : camera)} aria-label={`${isExpanded ? 'Collapse' : 'Expand'} ${cameraLabel(camera)}`} aria-expanded={isExpanded}>{isExpanded ? 'Collapse' : 'Expand'}</button>
                <button className="btn ghost" onClick={() => void openFullscreen(camera)} aria-label={`Fullscreen ${cameraLabel(camera)}`} title="Fullscreen"><span className="fullscreen-corners" aria-hidden="true" /></button></div>
            </header>
            <div className="spatial-camera-mode">
              <ChoiceStrip compact label={`${cameraLabel(camera)} display`} value={mode}
                options={[{ value: 'video', label: 'Video' }, { value: 'depth', label: 'Depth' }, { value: 'normals', label: 'Normals' }]}
                onChange={value => props.onCameraModeChange?.(camera, value as CameraDisplayMode)} />
              {mode !== 'video' && (snapshotError
                  ? <button className="btn ghost snapshot-retry" type="button" aria-label={`Retry PCF for ${cameraLabel(camera)}`}
                    title={snapshotError} onClick={() => props.onLoadSnapshot?.(camera)} disabled={props.connection !== 'open' || !props.onLoadSnapshot}>Retry saved</button>
                  : <span className="snapshot-load-state" role="status">{pcf ? 'Saved PCF' : props.connection === 'open' ? 'Loading PCF…' : 'Waiting for connection'}</span>)}
            </div>
            <div className={`spatial-camera-media${mode !== 'video' ? ' spatial-camera-media--snapshot' : ''}`}>
              <MosaicCropCanvas camera={camera} mosaicLayout={mosaicLayout} videoRef={videoRef} className="spatial-camera-video" active={!camerasHidden && props.active !== false && mode === 'video'} />
              {mode === 'video' && videoStatus && <div className="spatial-video-status">{videoStatus}</div>}
              <PcfCameraView camera={camera} priorId={pcf?.snapshot_id} calibrationEpoch={props.calibrationEpoch ?? 0}
                imageWidth={mosaicLayout?.frame_w} imageHeight={mosaicLayout?.frame_h} connected={props.connection === 'open'}
                mode={mode === 'video' ? 'depth' : mode} palette={presentation.palette} comparison={presentation.comparison} onComparisonChange={props.onComparisonChange}
                active={!camerasHidden && props.active !== false && mode !== 'video'} />
            </div>
            <footer><span>{mode === 'video' ? `${videoStatus ? 'Unavailable' : 'WebRTC'} · ${props.fps[camera] ?? 'FPS: —'}` : `Camera POV · ${mode === 'depth' ? 'Optical depth' : 'Surface normals'} · static PCF`}</span><span>{mode === 'video' && props.vacancyText[camera] ? `Vacant ${props.vacancyText[camera]}` : ''}</span></footer>
          </article>;
        })}
      </div>
    </UnifiedBevView>
    {notice && <div className="spatial-notice" role="status">{notice}</div>}
  </section>;
}
