import { CameraKey } from './camera';
import { getExtrinsics, worldToCamera } from './calibration';

export type BevFrameMode = 'world' | 'camera_local_legacy';

export type BoundsHint = {
  min_x?: number;
  max_x?: number;
  min_z?: number;
  max_z?: number;
};

type ProjectionOptions = {
  boundsHint?: BoundsHint | null;
};

type LocalGroundPoint = { x: number; y: number };

export const isWorldFrame = (frame?: string): boolean => {
  const value = String(frame || '').toLowerCase().trim();
  return value === 'world' || value === 'global' || value === 'menon_scene' || value === 'world_frame';
};

export const isCameraLocalFrame = (frame?: string): boolean => {
  const value = String(frame || '').toLowerCase().trim();
  if (!value) return false;
  return value === 'camera_local' || value === 'camera_local_ground' || value === 'camera-local' || value.includes('camera_local');
};

export const resolveBevFrameModeFromPayload = (
  payload: { frame_mode?: unknown; frame?: unknown; world_frame?: unknown },
  fallback: BevFrameMode = 'world'
): BevFrameMode => {
  const frameMode = String(payload.frame_mode || '').trim().toLowerCase();
  const frame = String(payload.frame || '').trim().toLowerCase();
  const worldFrame = String(payload.world_frame || '').trim().toLowerCase();
  if (!frameMode && !frame && !worldFrame) return fallback;
  if (frameMode === 'world') return 'world';
  if (frameMode === 'camera_local_legacy') return 'camera_local_legacy';
  if (frame === 'world') return 'world';
  if (worldFrame === 'world') return 'world';
  if (worldFrame === 'menon_scene' || worldFrame === 'global') return 'world';
  return 'camera_local_legacy';
};

export const projectWorldPointToCameraLocal = (
  camera: CameraKey,
  worldX: number,
  worldY: number,
  worldZ: number,
  _options: ProjectionOptions = {}
): LocalGroundPoint | null => {
  if (!Number.isFinite(worldX) || !Number.isFinite(worldY) || !Number.isFinite(worldZ)) return null;
  const E = getExtrinsics(camera);
  if (!Array.isArray(E) || E.length !== 16) return null;
  const p = worldToCamera(E, [worldX, worldY, worldZ]);
  if (!p) return null;
  const xLocal = Number(p[0]);
  const zLocal = Number(p[2]);
  if (!Number.isFinite(xLocal) || !Number.isFinite(zLocal)) return null;
  return { x: xLocal, y: zLocal };
};
