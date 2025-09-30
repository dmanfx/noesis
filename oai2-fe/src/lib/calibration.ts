import { CameraKey, detectCameraKey } from './camera';

type CameraTables = {
  K?: Record<string, number[]>;
  E?: Record<string, number[]>;
  [legacyKey: string]: any;
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
  if (eTable) {
    if (Array.isArray(eTable[cam])) return eTable[cam] as number[];
    for (const [key, E] of Object.entries(eTable)) {
      if (detectCameraKey(key) === cam && Array.isArray(E)) return E as number[];
    }
  }

  // Legacy fallback: cameras keyed by camId with nested extrinsics
  for (const key of Object.keys(cams)) {
    const legacy = cams[key] as any;
    if (!legacy || typeof legacy !== 'object') continue;
    if (detectCameraKey(key) !== cam) continue;
    const extr = legacy.extrinsics;
    if (extr && Array.isArray(extr.E)) return extr.E as number[];
  }

  return null;
}

export function getIntrinsics4(cam: CameraKey): number[] | null {
  const cams = bundle.cameras || {};
  const kTable = asRecord<number[]>(cams.K);
  if (kTable) {
    if (Array.isArray(kTable[cam])) return kTable[cam] as number[];
    for (const [key, K] of Object.entries(kTable)) {
      if (detectCameraKey(key) === cam && Array.isArray(K)) return K as number[];
    }
  }

  // Legacy fallback: nested intrinsics dict
  for (const key of Object.keys(cams)) {
    const legacy = cams[key] as any;
    if (!legacy || typeof legacy !== 'object') continue;
    if (detectCameraKey(key) !== cam) continue;
    const intr = legacy.intrinsics;
    if (intr && Array.isArray(intr) && intr.length >= 4) return intr as number[];
    if (intr && typeof intr === 'object') {
      const { fx, fy, cx, cy } = intr as any;
      if ([fx, fy, cx, cy].every((v: any) => typeof v === 'number')) {
        return [fx as number, fy as number, cx as number, cy as number];
      }
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
