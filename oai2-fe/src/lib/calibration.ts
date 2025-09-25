import { CameraKey, detectCameraKey } from './camera';

type CamCal = { intrinsics?: any; extrinsics?: { E?: number[] } };
type CalBundle = { cameras?: Record<string, CamCal> };

let bundle: CalBundle = {};

export function setCalibration(b: any) {
  if (!b || typeof b !== 'object') return;
  bundle = b.data || b;
}

export function getExtrinsics(cam: CameraKey): number[] | null {
  const cams = bundle.cameras || {};
  // Try exact key
  const exact = cams[cam];
  if (exact && exact.extrinsics && Array.isArray(exact.extrinsics.E)) return exact.extrinsics.E as number[];
  // Try any key that maps to cam via detectCameraKey
  for (const k of Object.keys(cams)) {
    const ck = detectCameraKey(k);
    if (ck === cam) {
      const E = cams[k]?.extrinsics?.E;
      if (Array.isArray(E)) return E as number[];
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

