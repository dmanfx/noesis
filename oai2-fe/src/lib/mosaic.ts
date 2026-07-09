import { CameraKey, cameraIndex, detectCameraKey } from './camera';

export type MosaicLayoutLike = {
  mosaic_w?: number;
  mosaic_h?: number;
  rows?: number | null;
  cols?: number | null;
  source_count?: number;
  sources?: Array<{ source_id?: number; camera_id?: string; cameraId?: string }>;
  frame_w?: number;
  frame_h?: number;
  tile_order?: string;
};

export type MosaicTileCrop = {
  sx: number;
  sy: number;
  sw: number;
  sh: number;
  rows: number;
  cols: number;
  tileIndex: number;
};

export type Rect = {
  x: number;
  y: number;
  w: number;
  h: number;
};

const CLASSIC_SOURCE_IDS = new Set<CameraKey>(['living-room', 'kitchen', 'family-room']);

export function normalizeMosaicCameraId(value: unknown): string {
  const id = String(value ?? '').toLowerCase().trim();
  if (!id) return '';
  return detectCameraKey(id) || id;
}

export function resolveMosaicTileIndex(layout: MosaicLayoutLike | null | undefined, camera: CameraKey): number | null {
  if (!layout || !Array.isArray(layout.sources)) return null;

  const target = normalizeMosaicCameraId(camera);
  if (!target) return null;

  for (let idx = 0; idx < layout.sources.length; idx += 1) {
    const src = layout.sources[idx];
    const cameraId = normalizeMosaicCameraId(src?.camera_id ?? src?.cameraId);
    if (cameraId && cameraId === target) return idx;
  }

  if (CLASSIC_SOURCE_IDS.has(camera)) {
    const sourceId = cameraIndex(camera);
    const idx = layout.sources.findIndex((src) => Number(src?.source_id) === sourceId);
    if (idx >= 0) return idx;
  }

  return null;
}

export function resolveMosaicTileCrop(
  layout: MosaicLayoutLike | null | undefined,
  camera: CameraKey,
  videoWidth: number,
  videoHeight: number
): MosaicTileCrop | null {
  if (!layout) return null;
  const rows = Number(layout.rows);
  const cols = Number(layout.cols);
  if (!Number.isFinite(rows) || !Number.isFinite(cols) || rows <= 0 || cols <= 0) return null;
  if (!Number.isFinite(videoWidth) || !Number.isFinite(videoHeight) || videoWidth <= 0 || videoHeight <= 0) return null;

  const tileIndex = resolveMosaicTileIndex(layout, camera);
  if (tileIndex === null) return null;
  const sourceCount = Number(layout.source_count || layout.sources?.length || rows * cols);
  if (Number.isFinite(sourceCount) && sourceCount > 0 && tileIndex >= sourceCount) return null;

  const sourceFrameW = Number(layout.frame_w) > 0 ? Number(layout.frame_w) : 0;
  const sourceFrameH = Number(layout.frame_h) > 0 ? Number(layout.frame_h) : 0;
  const mosaicW = Number(layout.mosaic_w) > 0 ? Number(layout.mosaic_w) : (sourceFrameW > 0 ? sourceFrameW * cols : videoWidth);
  const mosaicH = Number(layout.mosaic_h) > 0 ? Number(layout.mosaic_h) : (sourceFrameH > 0 ? sourceFrameH * rows : videoHeight);
  const tileW = mosaicW / cols;
  const tileH = mosaicH / rows;
  if (![mosaicW, mosaicH, tileW, tileH].every((value) => Number.isFinite(value) && value > 0)) return null;

  const c = tileIndex % cols;
  const r = Math.floor(tileIndex / cols);
  const scaleX = videoWidth / mosaicW;
  const scaleY = videoHeight / mosaicH;
  const sx = Math.max(0, Math.min(videoWidth, c * tileW * scaleX));
  const sy = Math.max(0, Math.min(videoHeight, r * tileH * scaleY));
  const ex = Math.max(sx, Math.min(videoWidth, (c * tileW + tileW) * scaleX));
  const ey = Math.max(sy, Math.min(videoHeight, (r * tileH + tileH) * scaleY));
  const sw = ex - sx;
  const sh = ey - sy;
  if (sw <= 1 || sh <= 1) return null;

  return { sx, sy, sw, sh, rows, cols, tileIndex };
}

export function containRect(containerW: number, containerH: number, mediaW: number, mediaH: number): Rect {
  if (containerW <= 0 || containerH <= 0 || mediaW <= 0 || mediaH <= 0) {
    return { x: 0, y: 0, w: Math.max(0, containerW), h: Math.max(0, containerH) };
  }
  const scale = Math.min(containerW / mediaW, containerH / mediaH);
  const w = mediaW * scale;
  const h = mediaH * scale;
  return {
    x: (containerW - w) / 2,
    y: (containerH - h) / 2,
    w,
    h,
  };
}
