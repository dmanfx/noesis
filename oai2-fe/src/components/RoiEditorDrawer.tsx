import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { MosaicLayout } from '../hooks/useWebSocketClient';
import '../styles/roi-editor.css';

type RoiPoint = [number, number];

type RoiDef = {
  id: string;
  description?: string | null;
  points_px: RoiPoint[];
};

type RoiStream = {
  stream_id: string;
  label?: string | null;
  enable: boolean;
  rois: RoiDef[];
};

type RoiStage = {
  stage: string;
  config_width?: number | null;
  config_height?: number | null;
  defaults?: Record<string, unknown>;
  streams: RoiStream[];
  config_path?: string | null;
};

type TileCrop = {
  sx: number;
  sy: number;
  sw: number;
  sh: number;
  rows: number;
  cols: number;
  tileIndex: number;
};

type Props = {
  open: boolean;
  onClose: () => void;
  restBaseUrl: string;
  mosaicLayout?: MosaicLayout | null;
  videoRef?: React.RefObject<HTMLVideoElement>;
  analyticsReloadCount?: number;
};

type FreezeSnapshot = {
  width: number;
  height: number;
  canvas: HTMLCanvasElement;
};

const DEFAULT_WIDTH = 560;
const MAX_WIDTH = 1120;
const HISTORY_LIMIT = 50;

const COLOR_PALETTE = [
  '#e76f51',
  '#2a9d8f',
  '#f4a261',
  '#e9c46a',
  '#457b9d',
  '#f07167',
  '#2d6a4f',
  '#e63946',
  '#00afb9',
  '#277da1',
];

const roundPoint = (pt: RoiPoint): RoiPoint => [Math.round(pt[0]), Math.round(pt[1])];

const cloneStream = (stream: RoiStream): RoiStream => ({
  ...stream,
  rois: stream.rois.map((roi) => ({
    ...roi,
    points_px: roi.points_px.map((pt) => [pt[0], pt[1]] as RoiPoint),
  })),
});

const hashString = (value: string): number => {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) >>> 0;
  }
  return hash;
};

const colorForId = (id: string): string => {
  const idx = hashString(id) % COLOR_PALETTE.length;
  return COLOR_PALETTE[idx];
};

const pointInPolygon = (point: RoiPoint, polygon: RoiPoint[]): boolean => {
  const [x, y] = point;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const xi = polygon[i][0];
    const yi = polygon[i][1];
    const xj = polygon[j][0];
    const yj = polygon[j][1];
    const intersect = (yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-9) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
};

const distance = (a: RoiPoint, b: RoiPoint): number => {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return Math.hypot(dx, dy);
};

const RoiEditorDrawer: React.FC<Props> = ({ open, onClose, restBaseUrl, mosaicLayout, videoRef, analyticsReloadCount }) => {
  const [drawerWidth, setDrawerWidth] = useState<number>(DEFAULT_WIDTH);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [stage, setStage] = useState<RoiStage | null>(null);
  const [selectedStreamId, setSelectedStreamId] = useState<string>('');
  const [activeRoiId, setActiveRoiId] = useState<string | null>(null);
  const [drawing, setDrawing] = useState<boolean>(false);
  const [frozenByStream, setFrozenByStream] = useState<Record<string, boolean>>({});
  const [dirtyStreams, setDirtyStreams] = useState<Record<string, boolean>>({});
  const [statusMsg, setStatusMsg] = useState<string>('');

  const baseCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawerRef = useRef<HTMLElement | null>(null);
  const freezeSnapshotsRef = useRef<Record<string, FreezeSnapshot>>({});
  const isResizingRef = useRef(false);
  const activePointerIdRef = useRef<number | null>(null);
  const previousUserSelectRef = useRef('');
  const dragRef = useRef<{ mode: 'none' | 'vertex' | 'shape'; roiId?: string; vertexIndex?: number; start?: RoiPoint; origin?: RoiPoint[]; historyPushed?: boolean; }>(
    { mode: 'none' }
  );
  const historyRef = useRef<Record<string, { past: RoiStream[]; future: RoiStream[] }>>({});

  const restBase = (restBaseUrl || '').replace(/\/$/, '');

  useEffect(() => {
    if (open) {
      setDrawerWidth(DEFAULT_WIDTH);
    }
  }, [open]);

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

  const handleResizePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 && event.pointerType === 'mouse') return;
    event.preventDefault();
    event.stopPropagation();
    if (!open) return;
    isResizingRef.current = true;
    activePointerIdRef.current = event.pointerId;
    previousUserSelectRef.current = document.body.style.userSelect;
    document.body.style.userSelect = 'none';
  };

  const configWidth = stage?.config_width || mosaicLayout?.frame_w || 1920;
  const configHeight = stage?.config_height || mosaicLayout?.frame_h || 1080;
  const aspect = `${configWidth} / ${configHeight}`;

  const streams = stage?.streams || [];

  const selectedStream = useMemo(() => {
    if (!selectedStreamId) return streams[0];
    return streams.find((s) => s.stream_id === selectedStreamId) || streams[0];
  }, [streams, selectedStreamId]);

  const rois = selectedStream?.rois || [];
  const freezeFrame = !!(selectedStream && frozenByStream[selectedStream.stream_id]);

  const updateStream = useCallback((streamId: string, updater: (stream: RoiStream) => RoiStream) => {
    setStage((prev) => {
      if (!prev) return prev;
      const nextStreams = prev.streams.map((stream) => {
        if (stream.stream_id !== streamId) return stream;
        return updater(stream);
      });
      return { ...prev, streams: nextStreams };
    });
  }, []);

  const markDirty = useCallback((streamId: string) => {
    setDirtyStreams((prev) => ({ ...prev, [streamId]: true }));
  }, []);

  const pushHistory = useCallback((streamId: string) => {
    const stream = streams.find((s) => s.stream_id === streamId);
    if (!stream) return;
    const entry = historyRef.current[streamId] || { past: [], future: [] };
    entry.past.push(cloneStream(stream));
    if (entry.past.length > HISTORY_LIMIT) {
      entry.past.shift();
    }
    entry.future = [];
    historyRef.current[streamId] = entry;
  }, [streams]);

  const applyStreamSnapshot = useCallback((streamId: string, snapshot: RoiStream) => {
    setStage((prev) => {
      if (!prev) return prev;
      const nextStreams = prev.streams.map((stream) => (stream.stream_id === streamId ? cloneStream(snapshot) : stream));
      return { ...prev, streams: nextStreams };
    });
    markDirty(streamId);
  }, [markDirty]);

  const undo = useCallback(() => {
    if (!selectedStream) return;
    const entry = historyRef.current[selectedStream.stream_id];
    if (!entry || entry.past.length === 0) return;
    entry.future.push(cloneStream(selectedStream));
    const snapshot = entry.past.pop();
    if (snapshot) {
      applyStreamSnapshot(selectedStream.stream_id, snapshot);
    }
  }, [applyStreamSnapshot, selectedStream]);

  const redo = useCallback(() => {
    if (!selectedStream) return;
    const entry = historyRef.current[selectedStream.stream_id];
    if (!entry || entry.future.length === 0) return;
    entry.past.push(cloneStream(selectedStream));
    const snapshot = entry.future.pop();
    if (snapshot) {
      applyStreamSnapshot(selectedStream.stream_id, snapshot);
    }
  }, [applyStreamSnapshot, selectedStream]);

  const fetchRois = useCallback(async () => {
    if (!open) return;
    setLoading(true);
    setError('');
      setStatusMsg('');
    try {
      const resp = await fetch(`${restBase}/api/v1/analytics/rois?stage=exclude`, { method: 'GET' });
      if (!resp.ok) {
        throw new Error(`REST ${resp.status}`);
      }
      const data = (await resp.json()) as RoiStage;
      setStage(data);
      setDirtyStreams({});
      historyRef.current = {};
      if (data.streams && data.streams.length) {
        setSelectedStreamId((prev) => {
          if (prev && data.streams.some((stream) => stream.stream_id === prev)) return prev;
          return data.streams[0].stream_id;
        });
      }
    } catch (err) {
      setError(`Failed to load ROIs (${String(err)})`);
    } finally {
      setLoading(false);
    }
  }, [open, restBase]);

  useEffect(() => {
    if (open) {
      fetchRois();
    }
  }, [open, fetchRois]);

  useEffect(() => {
    if (!open) return;
    if (!selectedStreamId && streams.length) {
      setSelectedStreamId(streams[0].stream_id);
    }
  }, [open, selectedStreamId, streams]);

  useEffect(() => {
    if (!selectedStream) return;
    setActiveRoiId(selectedStream.rois[0]?.id || null);
    setDrawing(false);
  }, [selectedStream?.stream_id]);

  const tileCrop = useMemo<TileCrop | null>(() => {
    if (!selectedStream) return null;
    const layout = mosaicLayout || undefined;
    const cols = layout?.cols ?? null;
    const rows = layout?.rows ?? null;
    if (!cols || !rows) return null;
    const sourceId = Number(selectedStream.stream_id);
    let tileIndex = Number.isFinite(sourceId) ? sourceId : 0;
    const sources = layout?.sources || [];
    if (sources.length) {
      const idx = sources.findIndex((source) => source.source_id === sourceId);
      if (idx >= 0) tileIndex = idx;
    }
    if (layout?.source_count && tileIndex >= layout.source_count) return null;
    const video = videoRef?.current;
    if (!video || !video.videoWidth || !video.videoHeight) return null;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const c = tileIndex % cols;
    const r = Math.floor(tileIndex / cols);
    const x0 = Math.round((c * vw) / cols);
    const x1 = Math.round(((c + 1) * vw) / cols);
    const y0 = Math.round((r * vh) / rows);
    const y1 = Math.round(((r + 1) * vh) / rows);
    return { sx: x0, sy: y0, sw: x1 - x0, sh: y1 - y0, rows, cols, tileIndex };
  }, [mosaicLayout, selectedStream, videoRef]);

  const drawBaseFrame = useCallback(() => {
    const canvas = baseCanvasRef.current;
    const ctx = canvas?.getContext('2d');
    const video = videoRef?.current;
    if (!canvas || !ctx || !video || !tileCrop) return;
    if (!video.videoWidth || !video.videoHeight) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, tileCrop.sx, tileCrop.sy, tileCrop.sw, tileCrop.sh, 0, 0, canvas.width, canvas.height);
  }, [tileCrop, videoRef]);

  const drawFrozenSnapshot = useCallback((streamId: string) => {
    const snapshot = freezeSnapshotsRef.current[streamId];
    const canvas = baseCanvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!snapshot || !canvas || !ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(snapshot.canvas, 0, 0, snapshot.width, snapshot.height, 0, 0, canvas.width, canvas.height);
  }, []);

  const captureFreezeSnapshot = useCallback((streamId: string) => {
    const base = baseCanvasRef.current;
    if (!base) return;
    const snapshotCanvas = document.createElement('canvas');
    snapshotCanvas.width = base.width;
    snapshotCanvas.height = base.height;
    const snapCtx = snapshotCanvas.getContext('2d');
    if (!snapCtx) return;
    snapCtx.drawImage(base, 0, 0);
    freezeSnapshotsRef.current[streamId] = {
      width: base.width,
      height: base.height,
      canvas: snapshotCanvas,
    };
  }, []);

  const handleToggleFreeze = useCallback(() => {
    if (!selectedStream) return;
    const streamId = selectedStream.stream_id;
    setFrozenByStream((prev) => {
      const currentlyFrozen = !!prev[streamId];
      const nextFrozen = !currentlyFrozen;
      if (nextFrozen) {
        // Ensure the base canvas has a current frame before capturing.
        drawBaseFrame();
        captureFreezeSnapshot(streamId);
      } else {
        delete freezeSnapshotsRef.current[streamId];
      }
      return { ...prev, [streamId]: nextFrozen };
    });
  }, [captureFreezeSnapshot, drawBaseFrame, selectedStream]);

  useEffect(() => {
    if (!open) return;
    if (freezeFrame) {
      return;
    }
    let raf = 0;
    const tick = () => {
      drawBaseFrame();
      raf = window.requestAnimationFrame(tick);
    };
    raf = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(raf);
  }, [open, freezeFrame, drawBaseFrame]);

  useEffect(() => {
    const canvas = baseCanvasRef.current;
    if (!canvas) return;
    canvas.width = configWidth;
    canvas.height = configHeight;
  }, [configWidth, configHeight]);

  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    canvas.width = configWidth;
    canvas.height = configHeight;
  }, [configWidth, configHeight]);

  useEffect(() => {
    if (!open || !selectedStream) return;
    if (!freezeFrame) return;
    drawFrozenSnapshot(selectedStream.stream_id);
  }, [drawFrozenSnapshot, freezeFrame, open, selectedStream]);

  const drawOverlay = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!selectedStream) return;

    rois.forEach((roi) => {
      if (!roi.points_px.length) return;
      const color = colorForId(roi.id);
      const isActive = roi.id === activeRoiId;
      ctx.beginPath();
      roi.points_px.forEach((pt, idx) => {
        if (idx === 0) ctx.moveTo(pt[0], pt[1]);
        else ctx.lineTo(pt[0], pt[1]);
      });
      if (!drawing || !isActive) {
        ctx.closePath();
      }
      ctx.lineWidth = isActive ? 3 : 2;
      ctx.strokeStyle = color;
      ctx.stroke();
      ctx.fillStyle = `${color}33`;
      if (!drawing || !isActive) {
        ctx.fill();
      }

      roi.points_px.forEach((pt, idx) => {
        const radius = isActive ? 6 : 4;
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], radius, 0, Math.PI * 2);
        ctx.fillStyle = idx === 0 ? '#ffffff' : color;
        ctx.fill();
        ctx.strokeStyle = '#050a11';
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      if (roi.points_px.length) {
        const label = roi.id;
        const anchor = roi.points_px[0];
        ctx.font = '12px sans-serif';
        ctx.fillStyle = color;
        ctx.fillText(label, anchor[0] + 8, anchor[1] - 8);
      }
    });
  }, [activeRoiId, drawing, rois, selectedStream]);

  useEffect(() => {
    drawOverlay();
  }, [drawOverlay]);

  const getCanvasPoint = useCallback((event: React.PointerEvent<HTMLCanvasElement>): RoiPoint | null => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
    return [x, y];
  }, []);

  const hitRadius = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return 8;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return 8;
    const scale = canvas.width / rect.width;
    return 8 * scale;
  }, []);

  const clampPoint = useCallback((pt: RoiPoint): RoiPoint => {
    const x = Math.min(configWidth, Math.max(0, pt[0]));
    const y = Math.min(configHeight, Math.max(0, pt[1]));
    return [x, y];
  }, [configHeight, configWidth]);

  const updateRoiPoints = useCallback((streamId: string, roiId: string, points: RoiPoint[]) => {
    const clamped = points.map(clampPoint);
    updateStream(streamId, (stream) => {
      const nextRois = stream.rois.map((roi) => (roi.id === roiId ? { ...roi, points_px: clamped } : roi));
      return { ...stream, rois: nextRois };
    });
    markDirty(streamId);
  }, [clampPoint, markDirty, updateStream]);

  const ensureUniqueId = useCallback((stream: RoiStream, desiredId: string, currentId?: string) => {
    const base = desiredId.trim() || 'ROI';
    if (base === currentId) return base;
    const ids = new Set(stream.rois.map((roi) => roi.id));
    if (!ids.has(base)) return base;
    let idx = 2;
    while (ids.has(`${base}-${idx}`)) idx += 1;
    return `${base}-${idx}`;
  }, []);

  const nextRoiId = useCallback((stream: RoiStream) => {
    const ids = stream.rois.map((roi) => roi.id);
    let max = 0;
    ids.forEach((id) => {
      const match = /^RF(\d+)$/.exec(id);
      if (match) max = Math.max(max, Number(match[1]));
    });
    if (max > 0) return `RF${max + 1}`;
    return `ROI-${stream.rois.length + 1}`;
  }, []);

  const handleNewRoi = useCallback(() => {
    if (!selectedStream) return;
    const id = nextRoiId(selectedStream);
    pushHistory(selectedStream.stream_id);
    updateStream(selectedStream.stream_id, (stream) => ({
      ...stream,
      rois: [...stream.rois, { id, description: '', points_px: [] }],
    }));
    setActiveRoiId(id);
    setDrawing(true);
    markDirty(selectedStream.stream_id);
  }, [markDirty, nextRoiId, pushHistory, selectedStream, updateStream]);

  const handleDeleteRoi = useCallback((roiId: string) => {
    if (!selectedStream) return;
    pushHistory(selectedStream.stream_id);
    updateStream(selectedStream.stream_id, (stream) => ({
      ...stream,
      rois: stream.rois.filter((roi) => roi.id !== roiId),
    }));
    if (activeRoiId === roiId) setActiveRoiId(null);
    markDirty(selectedStream.stream_id);
  }, [activeRoiId, markDirty, pushHistory, selectedStream, updateStream]);

  const handleRenameRoi = useCallback((roiId: string, nextId: string) => {
    if (!selectedStream) return;
    const unique = ensureUniqueId(selectedStream, nextId, roiId);
    updateStream(selectedStream.stream_id, (stream) => ({
      ...stream,
      rois: stream.rois.map((roi) => (roi.id === roiId ? { ...roi, id: unique } : roi)),
    }));
    if (activeRoiId === roiId) setActiveRoiId(unique);
    markDirty(selectedStream.stream_id);
  }, [activeRoiId, ensureUniqueId, markDirty, selectedStream, updateStream]);

  const handleDescriptionChange = useCallback((roiId: string, desc: string) => {
    if (!selectedStream) return;
    updateStream(selectedStream.stream_id, (stream) => ({
      ...stream,
      rois: stream.rois.map((roi) => (roi.id === roiId ? { ...roi, description: desc } : roi)),
    }));
    markDirty(selectedStream.stream_id);
  }, [markDirty, selectedStream, updateStream]);

  const handleStreamEnable = useCallback((enabled: boolean) => {
    if (!selectedStream) return;
    pushHistory(selectedStream.stream_id);
    updateStream(selectedStream.stream_id, (stream) => ({
      ...stream,
      enable: enabled,
    }));
    markDirty(selectedStream.stream_id);
  }, [markDirty, pushHistory, selectedStream, updateStream]);

  const handlePointerDown = useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!selectedStream) return;
    if (event.button !== 0 && event.pointerType === 'mouse') return;
    const pt = getCanvasPoint(event);
    if (!pt) return;

    const radius = hitRadius();
    const active = rois.find((roi) => roi.id === activeRoiId) || rois[0];

    if (drawing && active) {
      const pts = active.points_px;
      if (pts.length >= 3 && (event.detail > 1 || distance(pts[0], pt) <= radius * 1.6)) {
        setDrawing(false);
        drawOverlay();
        return;
      }
      pushHistory(selectedStream.stream_id);
      updateRoiPoints(selectedStream.stream_id, active.id, [...pts, roundPoint(pt)]);
      return;
    }

    if (active && active.points_px.length) {
      const vertexIndex = active.points_px.findIndex((p) => distance(p, pt) <= radius);
      if (vertexIndex >= 0) {
        dragRef.current = { mode: 'vertex', roiId: active.id, vertexIndex, start: pt, historyPushed: true };
        pushHistory(selectedStream.stream_id);
        overlayCanvasRef.current?.setPointerCapture(event.pointerId);
        return;
      }
    }

    const hitRoi = rois.find((roi) => roi.points_px.length >= 3 && pointInPolygon(pt, roi.points_px));
    if (hitRoi) {
      setActiveRoiId(hitRoi.id);
      dragRef.current = { mode: 'shape', roiId: hitRoi.id, start: pt, origin: hitRoi.points_px.map((p) => [p[0], p[1]] as RoiPoint), historyPushed: true };
      pushHistory(selectedStream.stream_id);
      overlayCanvasRef.current?.setPointerCapture(event.pointerId);
      return;
    }

    setActiveRoiId(active?.id || null);
  }, [activeRoiId, drawOverlay, drawing, getCanvasPoint, hitRadius, pushHistory, rois, selectedStream, updateRoiPoints]);

  const handlePointerMove = useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (drag.mode === 'none' || !selectedStream || !drag.roiId || !drag.start) return;
    const pt = getCanvasPoint(event);
    if (!pt) return;

    if (drag.mode === 'vertex' && drag.vertexIndex !== undefined) {
      const roi = rois.find((r) => r.id === drag.roiId);
      if (!roi) return;
      const nextPoints = roi.points_px.map((p, idx) => (idx === drag.vertexIndex ? roundPoint(pt) : p));
      updateRoiPoints(selectedStream.stream_id, roi.id, nextPoints);
    }

    if (drag.mode === 'shape' && drag.origin) {
      const dx = pt[0] - drag.start[0];
      const dy = pt[1] - drag.start[1];
      const nextPoints = drag.origin.map((p) => [p[0] + dx, p[1] + dy] as RoiPoint);
      updateRoiPoints(selectedStream.stream_id, drag.roiId, nextPoints);
    }
  }, [getCanvasPoint, rois, selectedStream, updateRoiPoints]);

  const handlePointerUp = useCallback((event?: React.PointerEvent<HTMLCanvasElement>) => {
    dragRef.current = { mode: 'none' };
    if (event?.pointerId !== undefined) {
      try {
        overlayCanvasRef.current?.releasePointerCapture(event.pointerId);
      } catch {
        // ignore
      }
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (!open) return;
      if (event.key === 'Escape' && drawing && selectedStream && activeRoiId) {
        handleDeleteRoi(activeRoiId);
        setDrawing(false);
      }
      if ((event.key === 'Delete' || event.key === 'Backspace') && !drawing && activeRoiId) {
        handleDeleteRoi(activeRoiId);
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z') {
        if (event.shiftKey) redo();
        else undo();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [activeRoiId, drawing, handleDeleteRoi, open, redo, selectedStream, undo]);

  const applyChanges = useCallback(async () => {
    if (!selectedStream || !stage) return;
    setLoading(true);
    setStatusMsg('Applying...');
    setError('');
    try {
      const completeRois = selectedStream.rois.filter((roi) => roi.points_px.length >= 3);
      const dropped = selectedStream.rois.length - completeRois.length;
      const payload = {
        stage: stage.stage || 'exclude',
        streams: [{
          stream_id: selectedStream.stream_id,
          label: selectedStream.label || null,
          enable: selectedStream.enable,
          rois: completeRois.map((roi) => ({
            id: roi.id,
            description: roi.description || null,
            points_px: roi.points_px.map((pt) => [pt[0], pt[1]]),
          })),
        }],
      };
      const resp = await fetch(`${restBase}/api/v1/analytics/rois`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        throw new Error(`REST ${resp.status}`);
      }
      const data = (await resp.json()) as RoiStage & { reloaded?: boolean };
      setStage(data);
      setDirtyStreams((prev) => ({ ...prev, [selectedStream.stream_id]: false }));
      historyRef.current[selectedStream.stream_id] = { past: [], future: [] };
      if (dropped) {
        setStatusMsg(`Applied (${dropped} incomplete ROI ignored).`);
      } else {
        setStatusMsg(data.reloaded ? 'Applied + reloaded.' : 'Applied.');
      }
    } catch (err) {
      setError(`Apply failed (${String(err)})`);
      setStatusMsg('');
    } finally {
      setLoading(false);
    }
  }, [restBase, selectedStream, stage]);

  const tileStatus = useMemo(() => {
    if (!mosaicLayout) return 'Waiting for layout metadata.';
    if (!mosaicLayout.cols || !mosaicLayout.rows) return 'Tiler rows/cols not set. Set explicit layout to edit.';
    if (!tileCrop) return 'Mosaic tile not ready yet.';
    return '';
  }, [mosaicLayout, tileCrop]);

  const canApply = !!(selectedStream && dirtyStreams[selectedStream.stream_id]);

  return (
    <>
      <div className={open ? 'roi-overlay open' : 'roi-overlay'} onClick={onClose} />
      <aside
        className={open ? 'roi-drawer open' : 'roi-drawer'}
        aria-hidden={!open}
        ref={drawerRef}
        style={{ width: drawerWidth }}
      >
        <div className="roi-drawer-resize" onPointerDown={handleResizePointerDown} role="presentation" />
        <header className="roi-header">
          <div>
            <h2>ROI Editor</h2>
            <div className="roi-subtitle">Exclude zones (nvdsroiexclude)</div>
          </div>
          <div className="roi-header-actions">
            <button className="btn ghost" onClick={fetchRois} disabled={loading}>Reload</button>
            <button onClick={onClose} aria-label="Close">✕</button>
          </div>
        </header>

        <div className="roi-content">
          <section className="roi-panel">
            <div className="roi-row">
              <label>Camera</label>
              <select
                value={selectedStream?.stream_id || ''}
                onChange={(e) => setSelectedStreamId(e.target.value)}
                disabled={!streams.length}
              >
                {streams.map((stream) => (
                  <option key={stream.stream_id} value={stream.stream_id}>
                    {stream.label || `Stream ${stream.stream_id}`}
                  </option>
                ))}
              </select>
            </div>
            <div className="roi-row">
              <label>Enabled</label>
              <input
                type="checkbox"
                checked={selectedStream?.enable ?? false}
                onChange={(e) => handleStreamEnable(e.target.checked)}
                disabled={!selectedStream}
              />
            </div>
            <div className="roi-row">
              <label>Reload Count</label>
              <span className="roi-chip">{analyticsReloadCount ?? 0}</span>
            </div>
            {statusMsg && <div className="roi-status">{statusMsg}</div>}
            {error && <div className="roi-error">{error}</div>}
          </section>

          <section className="roi-editor">
            <div className="roi-canvas-wrap" style={{ ['--roi-aspect' as any]: aspect }}>
              <canvas ref={baseCanvasRef} className="roi-canvas" />
              <canvas
                ref={overlayCanvasRef}
                className="roi-canvas roi-overlay-canvas"
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerLeave={handlePointerUp}
              />
              {tileStatus && <div className="roi-canvas-status">{tileStatus}</div>}
            </div>
            <div className="roi-toolbar">
              <button className="btn" onClick={handleNewRoi} disabled={!selectedStream}>New ROI</button>
              <button className="btn ghost" onClick={handleToggleFreeze} disabled={!selectedStream}>
                {freezeFrame ? 'Resume' : 'Freeze'}
              </button>
              <button className="btn ghost" onClick={undo} disabled={!selectedStream}>Undo</button>
              <button className="btn ghost" onClick={redo} disabled={!selectedStream}>Redo</button>
              <button className="btn primary" onClick={applyChanges} disabled={!canApply || loading}>Apply</button>
            </div>
          </section>

          <section className="roi-list">
            <div className="roi-list-header">
              <div>ROIs</div>
              <div className="roi-hint">{drawing ? 'Click to add points, click first point to close.' : 'Select and edit polygons.'}</div>
            </div>
            {rois.length === 0 && <div className="roi-empty">No ROIs yet.</div>}
            {rois.map((roi) => (
              <div key={roi.id} className={roi.id === activeRoiId ? 'roi-item active' : 'roi-item'}>
                <div className="roi-swatch" style={{ background: colorForId(roi.id) }} />
                <div className="roi-fields">
                  <input
                    value={roi.id}
                    onChange={(e) => handleRenameRoi(roi.id, e.target.value)}
                  />
                  <input
                    value={roi.description || ''}
                    onChange={(e) => handleDescriptionChange(roi.id, e.target.value)}
                    placeholder="Description"
                  />
                </div>
                <div className="roi-actions">
                  <button className="btn ghost" onClick={() => setActiveRoiId(roi.id)}>Select</button>
                  <button className="btn ghost" onClick={() => handleDeleteRoi(roi.id)}>Delete</button>
                </div>
              </div>
            ))}
          </section>
        </div>
      </aside>
    </>
  );
};

export default RoiEditorDrawer;
