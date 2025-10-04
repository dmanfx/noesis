import React, { useEffect, useMemo, useRef } from 'react';
import { CameraKey, cameraLabel, colorForTrack } from '../lib/camera';
import { TrailStore, drawTrails } from '../lib/trails';

type Bounds = { min_x?: number; max_x?: number; min_z?: number; max_z?: number };
type FloorplanResponse = {
  camera_id?: string;
  bounds?: Bounds;
};

export const MapPanel: React.FC<{
  store: TrailStore;
  visible: boolean;
  floorplans?: Record<string, FloorplanResponse>;
  worldUsage?: Record<CameraKey, boolean>;
}> = ({ store, visible, floorplans = {}, worldUsage = { 'living-room': false, 'kitchen': false, 'family-room': false } }) => {
  const canvLiving = useRef<HTMLCanvasElement>(null);
  const canvKitchen = useRef<HTMLCanvasElement>(null);
  const canvFamily = useRef<HTMLCanvasElement>(null);

  // Resolve bounds per camera key from floorplans map; enforce no-fallback (undefined -> do not render)
  const boundsByKey = useMemo<Record<CameraKey, Bounds | undefined>>(() => {
    const get = (k: CameraKey) => floorplans[k]?.bounds;
    return {
      'living-room': get('living-room'),
      'kitchen': get('kitchen'),
      'family-room': get('family-room'),
    };
  }, [floorplans]);

  // Compute pixel sizes per camera based on bounds aspect; fixed width of 420 px
  const sizeByKey = useMemo<Record<CameraKey, { w: number; h: number } | undefined>>(() => {
    const W = 420;
    const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
    const toSize = (b?: Bounds) => {
      if (!b) return undefined;
      const wx = (b.max_x ?? 0) - (b.min_x ?? 0);
      const hz = (b.max_z ?? 0) - (b.min_z ?? 0);
      if (!(isFinite(wx) && isFinite(hz)) || wx <= 0 || hz <= 0) return undefined;
      const aspect = clamp(hz / wx, 0.5, 3.0);
      const H = Math.round(W * aspect);
      return { w: W, h: H };
    };
    return {
      'living-room': toSize(boundsByKey['living-room']),
      'kitchen': toSize(boundsByKey['kitchen']),
      'family-room': toSize(boundsByKey['family-room']),
    };
  }, [boundsByKey]);

  useEffect(() => {
    if (!visible) {
      [canvLiving.current, canvKitchen.current, canvFamily.current].forEach(c => c?.getContext('2d')?.clearRect(0, 0, c.width, c.height));
      return;
    }
    const drawFor = (cam: CameraKey, ref: React.RefObject<HTMLCanvasElement>) => {
      const canvas = ref.current;
      const size = sizeByKey[cam];
      const b = boundsByKey[cam];
      const useWorld = !!worldUsage[cam];
      if (!canvas || !size || !b) return; // fail-fast: require bounds for rendering
      // Apply size
      if (canvas.width !== size.w) canvas.width = size.w;
      if (canvas.height !== size.h) canvas.height = size.h;
      // Draw using fixed viewport from bounds; invertY for z-forward
      drawTrails(canvas, cam, store, colorForTrack, {
        xMin: Number(b.min_x ?? 0),
        xMax: Number(b.max_x ?? 0),
        yMin: Number(b.min_z ?? 0),
        yMax: Number(b.max_z ?? 0),
        invertY: true,
        drawCameraMarker: useWorld,
      });
    };
    drawFor('living-room', canvLiving);
    drawFor('kitchen', canvKitchen);
    drawFor('family-room', canvFamily);
  }, [visible, store, boundsByKey, sizeByKey, worldUsage]);

  const renderRow = (cam: CameraKey, ref: React.RefObject<HTMLCanvasElement>) => {
    const size = sizeByKey[cam];
    const b = boundsByKey[cam];
    if (!size || !b) return null; // fail-fast: do not render without bounds
    return (
      <div className="vstack panel card">
        <div className="card-title">Top‑Down • {cameraLabel(cam)}</div>
        <div className="map-wrap">
          <canvas
            className="map"
            ref={ref}
            width={size.w}
            height={size.h}
            style={{ width: `${size.w}px`, height: `${size.h}px` }}
          />
        </div>
      </div>
    );
  };

  if (!visible) return null;

  return (
    <div className="vstack" style={{ gap: 16 }}>
      {renderRow('living-room', canvLiving)}
      {renderRow('kitchen', canvKitchen)}
      {renderRow('family-room', canvFamily)}
    </div>
  );
};
