import { CameraKey } from './camera';
import { extractPoseFromExtrinsics, getExtrinsics, getFrameBinding } from './calibration';

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
  return value === 'world' || value === 'global' || value === 'menon_scene' || value === 'world_frame' || value === 'backend_world_m';
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
  if (frameMode === 'camera_local_legacy' || frameMode === 'camera_local' || frameMode.includes('camera_local')) {
    return 'camera_local_legacy';
  }
  if (frame === 'world') return 'world';
  if (isCameraLocalFrame(frame)) return 'camera_local_legacy';
  if (worldFrame === 'world') return 'world';
  if (isCameraLocalFrame(worldFrame)) return 'camera_local_legacy';
  if (worldFrame === 'menon_scene' || worldFrame === 'global' || worldFrame === 'backend_world_m') return 'world';
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
  if (!Array.isArray(E) || E.length !== 16) {
    // Item 5 guard: missing or late calibration during world-mode projection is a common source of BEV issues.
    // The dashboard will fall back gracefully; this surfaces the root cause for debugging.
    if (typeof console !== 'undefined' && console.warn) {
      console.warn('[BEV] world projection skipped — missing/invalid extrinsics for', camera);
    }
    return null;
  }
  const pose = extractPoseFromExtrinsics(E);
  if (!pose) return null;

  // Calibration-bundle E is camera-from-calibration-frame_raw.  World tracks
  // and the active PCF use the camera's revision-bound target world, so apply
  // target_from_calibration before deriving the presentation basis.  Without
  // this edge a non-identity map-lock (Family Room) projects into the wrong
  // origin even when pitch is handled correctly.
  const binding = getFrameBinding(camera);
  const targetFromCalibration = binding?.target_from_calibration_col_major;
  if (
    !Array.isArray(targetFromCalibration)
    || targetFromCalibration.length !== 16
    || !targetFromCalibration.every((value) => Number.isFinite(Number(value)))
  ) {
    if (typeof console !== 'undefined' && console.warn) {
      console.warn('[BEV] world projection skipped — missing/invalid revision-bound frame binding for', camera);
    }
    return null;
  }

  const m = targetFromCalibration.map(Number);
  if (
    Math.abs(m[3]) > 1e-8
    || Math.abs(m[7]) > 1e-8
    || Math.abs(m[11]) > 1e-8
    || Math.abs(m[15] - 1) > 1e-8
  ) return null;

  // Column-major target<-calibration transform: map the camera center and
  // camera axes from the calibration frame into the active target revision.
  const targetRotation = [
    [m[0], m[4], m[8]],
    [m[1], m[5], m[9]],
    [m[2], m[6], m[10]],
  ];
  const transformTargetPoint = (point: [number, number, number]): [number, number, number] => [
    targetRotation[0][0] * point[0] + targetRotation[0][1] * point[1] + targetRotation[0][2] * point[2] + m[12],
    targetRotation[1][0] * point[0] + targetRotation[1][1] * point[1] + targetRotation[1][2] * point[2] + m[13],
    targetRotation[2][0] * point[0] + targetRotation[2][1] * point[1] + targetRotation[2][2] * point[2] + m[14],
  ];
  const transformTargetAxis = (axis: [number, number, number]): [number, number, number] => [
    targetRotation[0][0] * axis[0] + targetRotation[0][1] * axis[1] + targetRotation[0][2] * axis[2],
    targetRotation[1][0] * axis[0] + targetRotation[1][1] * axis[1] + targetRotation[1][2] * axis[2],
    targetRotation[2][0] * axis[0] + targetRotation[2][1] * axis[1] + targetRotation[2][2] * axis[2],
  ];

  const targetCamera = transformTargetPoint(pose.Cw);
  const targetForward = transformTargetAxis([pose.Rwc[0][2], pose.Rwc[1][2], pose.Rwc[2][2]]);
  const targetRight = transformTargetAxis([pose.Rwc[0][0], pose.Rwc[1][0], pose.Rwc[2][0]]);

  // camera_local_ground_m is a horizontal presentation frame.  Project the
  // camera's +Z/+X axes onto the world XZ plane before taking the dot product;
  // using pitched E*p[2] makes a floor point below a raised camera appear
  // displaced along local Z by the camera height and pitch.
  const forwardX = Number(targetForward[0]);
  const forwardZ = Number(targetForward[2]);
  const forwardNorm = Math.hypot(forwardX, forwardZ);
  if (!Number.isFinite(forwardNorm) || forwardNorm <= 1e-8) return null;
  const groundForwardX = forwardX / forwardNorm;
  const groundForwardZ = forwardZ / forwardNorm;

  let rightX = Number(targetRight[0]);
  let rightZ = Number(targetRight[2]);
  const rightNorm = Math.hypot(rightX, rightZ);
  if (!Number.isFinite(rightNorm) || rightNorm <= 1e-8) return null;
  rightX /= rightNorm;
  rightZ /= rightNorm;
  const rightForwardDot = rightX * groundForwardX + rightZ * groundForwardZ;
  rightX -= rightForwardDot * groundForwardX;
  rightZ -= rightForwardDot * groundForwardZ;
  const groundRightNorm = Math.hypot(rightX, rightZ);
  if (!Number.isFinite(groundRightNorm) || groundRightNorm <= 1e-8) return null;
  rightX /= groundRightNorm;
  rightZ /= groundRightNorm;

  const [cameraX, , cameraZ] = targetCamera;
  const deltaX = worldX - cameraX;
  const deltaZ = worldZ - cameraZ;
  const xLocal = rightX * deltaX + rightZ * deltaZ;
  const zLocal = groundForwardX * deltaX + groundForwardZ * deltaZ;
  if (!Number.isFinite(xLocal) || !Number.isFinite(zLocal)) return null;
  return { x: xLocal, y: zLocal };
};
