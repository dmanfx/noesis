import { CameraKey, detectCameraKey } from './camera';

type CameraTables = {
  K?: Record<string, number[]>;
  E?: Record<string, number[]>;
};

type CalBundle = { cameras?: CameraTables };

let bundle: CalBundle = {};

export function setCalibration(b: any) {
  if (!b || typeof b !== 'object') return;
  bundle = b.data || b;
}

function asRecord<T = any>(value: unknown): Record<string, T> | undefined {
  if (!value || typeof value !== 'object') return undefined;
  return value as Record<string, T>;
}

export function getExtrinsics(cam: CameraKey): number[] | null {
  const cams = bundle.cameras || {};
  const eTable = asRecord<number[]>(cams.E);
  if (!eTable) return null;
  if (Array.isArray(eTable[cam])) return eTable[cam] as number[];
  for (const [key, E] of Object.entries(eTable)) {
    if (detectCameraKey(key) === cam && Array.isArray(E)) return E as number[];
  }

  return null;
}

// Try to fetch extrinsics by either a CameraKey or a raw camera ID.
export function getExtrinsicsAny(idOrKey: string): number[] | null {
  try {
    const key = detectCameraKey(idOrKey) as CameraKey | null;
    if (key) {
      const e = getExtrinsics(key);
      if (e) return e;
    }
  } catch {}

  const cams = bundle.cameras || {};
  const eTable = asRecord<number[]>(cams.E);
  if (eTable && Array.isArray(eTable[idOrKey])) return eTable[idOrKey] as number[];
  return null;
}

// Extract camera pose in world from E (world→camera) column-major 4x4.
// Assumes rigid transform with last row [0,0,0,1]. Returns camera center (world)
// and rotation R_wc (3x3) mapping camera→world.
export function extractPoseFromExtrinsics(EcolMajor: number[]): { Cw: [number, number, number]; Rwc: number[][] } | null {
  if (!Array.isArray(EcolMajor) || EcolMajor.length !== 16) return null;
  const m = EcolMajor as number[];
  // R_cw (world→camera), column-major indices
  const r00 = m[0], r01 = m[4], r02 = m[8];
  const r10 = m[1], r11 = m[5], r12 = m[9];
  const r20 = m[2], r21 = m[6], r22 = m[10];
  const tx = m[12], ty = m[13], tz = m[14];

  // R_wc = R_cw^T
  const Rwc = [
    [r00, r10, r20],
    [r01, r11, r21],
    [r02, r12, r22],
  ];
  // C_world = -R_wc * t_cw
  const Cx = -(Rwc[0][0] * tx + Rwc[0][1] * ty + Rwc[0][2] * tz);
  const Cy = -(Rwc[1][0] * tx + Rwc[1][1] * ty + Rwc[1][2] * tz);
  const Cz = -(Rwc[2][0] * tx + Rwc[2][1] * ty + Rwc[2][2] * tz);
  return { Cw: [Cx, Cy, Cz], Rwc };
}

// Compute the forward direction projected onto the world XZ plane from E.
export function forwardXZFromExtrinsics(EcolMajor: number[]): { fx: number; fz: number } | null {
  const pose = extractPoseFromExtrinsics(EcolMajor);
  if (!pose) return null;
  // Camera forward in camera frame is +Z. Map to world and project to XZ.
  const R = pose.Rwc;
  const fw_x = R[0][2]; // column 2 of R (Z axis)
  const fw_y = R[1][2];
  const fw_z = R[2][2];
  // Project to ground plane (ignore Y)
  const fx = fw_x;
  const fz = fw_z;
  const n = Math.hypot(fx, fz) || 1;
  return { fx: fx / n, fz: fz / n };
}

export function getIntrinsics4(cam: CameraKey): number[] | null {
  const cams = bundle.cameras || {};
  const kTable = asRecord<number[]>(cams.K);
  if (!kTable) return null;
  if (Array.isArray(kTable[cam])) return kTable[cam] as number[];
  for (const [key, K] of Object.entries(kTable)) {
    if (detectCameraKey(key) === cam && Array.isArray(K)) return K as number[];
  }

  return null;
}

export function getIntrinsicsAny(idOrKey: string): number[] | null {
  try {
    const key = detectCameraKey(idOrKey) as CameraKey | null;
    if (key) {
      const k = getIntrinsics4(key);
      if (k) return k;
    }
  } catch {}

  const cams = bundle.cameras || {};
  const kTable = asRecord<number[]>(cams.K);
  if (kTable && Array.isArray(kTable[idOrKey])) return kTable[idOrKey] as number[];
  if (kTable) {
    for (const [key, K] of Object.entries(kTable)) {
      if (detectCameraKey(key) === idOrKey && Array.isArray(K)) return K as number[];
    }
  }
  return null;
}

// Column-major 4x4 multiply (E world→camera) with homogeneous world point
export function worldToCamera(EcolMajor: number[], Pw: [number, number, number]): [number, number, number] | null {
  if (!Array.isArray(EcolMajor) || EcolMajor.length !== 16) return null;
  const m = EcolMajor as number[];
  const x = Pw[0], y = Pw[1], z = Pw[2], w = 1.0;
  // Column-major: [0..3]=col0, [4..7]=col1, [8..11]=col2, [12..15]=col3
  const pcx = m[0]*x + m[4]*y + m[8]*z + m[12]*w;
  const pcy = m[1]*x + m[5]*y + m[9]*z + m[13]*w;
  const pcz = m[2]*x + m[6]*y + m[10]*z + m[14]*w;
  const pcw = m[3]*x + m[7]*y + m[11]*z + m[15]*w;
  if (pcw === 0) return null;
  return [pcx/pcw, pcy/pcw, pcz/pcw];
}
