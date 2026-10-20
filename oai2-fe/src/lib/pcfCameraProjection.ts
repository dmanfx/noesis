import type { CameraFrameBinding } from './calibration';
import { buildPcfCameraSurface } from './pcfCameraSurface';

export type PcfCameraProjection = {
  camera: string; priorId: string; width: number; height: number; intrinsics: number[];
  cameraFromWorld: number[]; worldFrame: { frame_id: string; revision: string }; transformSha256: string;
};
export type PcfCameraRaster = {
  width: number; height: number; depth: Float32Array; rgb: Uint8ClampedArray; normals: Uint8ClampedArray;
  range: { min: number; max: number }; validCount: number; normalCount: number; triangleCount: number;
};
const requireValue: (ok: unknown, reason: string) => asserts ok = (ok, reason) => { if (!ok) throw new Error(reason); };

function rigid(matrix: unknown): number[] {
  requireValue(Array.isArray(matrix) && matrix.length === 16 && matrix.every(Number.isFinite), 'pcf_camera_transform_invalid');
  const m = matrix as number[];
  requireValue(Math.abs(m[3]) + Math.abs(m[7]) + Math.abs(m[11]) + Math.abs(m[15] - 1) < 1e-5, 'pcf_camera_transform_invalid');
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
    const dot = m[a * 4] * m[b * 4] + m[a * 4 + 1] * m[b * 4 + 1] + m[a * 4 + 2] * m[b * 4 + 2];
    requireValue(Math.abs(dot - (a === b ? 1 : 0)) < 1e-4, 'pcf_camera_transform_not_rigid');
  }
  const determinant = m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2]) + m[8] * (m[1] * m[6] - m[5] * m[2]);
  requireValue(Math.abs(determinant - 1) < 1e-4, 'pcf_camera_transform_not_rigid');
  return m;
}

export function buildPcfCameraProjection(camera: string, priorId: string, width: number, height: number,
  input: { intrinsics: number[] | null; extrinsics: number[] | null; binding: CameraFrameBinding | null } | null): PcfCameraProjection {
  requireValue(Number.isSafeInteger(width) && width > 0 && width <= 8192 && Number.isSafeInteger(height) && height > 0 && height <= 8192, 'pcf_camera_image_size_unavailable');
  const k = input?.intrinsics, binding = input?.binding;
  requireValue(k && k.length === 4 && k.every(Number.isFinite) && k[0] > 0 && k[1] > 0 && k[2] >= 0 && k[2] < width && k[3] >= 0 && k[3] < height, 'pcf_camera_intrinsics_unavailable');
  requireValue(binding?.contract === 'noesis.calibration.frame_binding' && binding.world_frame?.frame_id === 'backend_world_m'
    && binding.world_frame.revision && /^[0-9a-f]{64}$/.test(binding.transform_sha256 ?? '') && binding.scene_prior_id === priorId, 'pcf_camera_frame_binding_mismatch');
  const e = rigid(input?.extrinsics), t = rigid(binding.target_from_calibration_col_major);
  const inverse = [t[0], t[4], t[8], 0, t[1], t[5], t[9], 0, t[2], t[6], t[10], 0,
    -(t[0] * t[12] + t[1] * t[13] + t[2] * t[14]),
    -(t[4] * t[12] + t[5] * t[13] + t[6] * t[14]),
    -(t[8] * t[12] + t[9] * t[13] + t[10] * t[14]), 1];
  // Canonical optical transform: E_calibration * inverse(target_from_calibration).
  const composed = Array(16).fill(0);
  for (let col = 0; col < 4; col++) for (let row = 0; row < 4; row++) for (let n = 0; n < 4; n++) composed[col * 4 + row] += e[n * 4 + row] * inverse[col * 4 + n];
  return { camera, priorId, width, height, intrinsics: [...k], cameraFromWorld: composed,
    worldFrame: { ...binding.world_frame }, transformSha256: binding.transform_sha256! };
}

export function validatePcfCameraMetadata(metadata: any, projection: PcfCameraProjection) {
  requireValue(metadata?.contract === 'noesis.scene_prior.camera_view' && metadata.contract_version === 1
    && metadata.camera_id === projection.camera && metadata.prior_id === projection.priorId
    && metadata.source_coordinate_frame === projection.worldFrame.frame_id
    && metadata.source_coordinate_frame_revision === projection.worldFrame.revision
    && metadata.frame_transform_sha256 === projection.transformSha256, 'pcf_camera_reconstruction_binding_mismatch');
  const url = metadata.artifact_urls?.points_glb;
  requireValue(url === `/api/v1/scene-priors/cameras/${encodeURIComponent(projection.camera)}/artifacts/points_glb`
    && /^[0-9a-f]{64}$/.test(metadata.points?.sha256 ?? '')
    && Number.isSafeInteger(metadata.points?.size_bytes) && metadata.points.size_bytes > 0 && metadata.points.size_bytes <= 32 * 1024 * 1024, 'pcf_camera_artifact_invalid');
  return { url: url as string, sha256: metadata.points.sha256 as string, sizeBytes: metadata.points.size_bytes as number };
}

/** Calibrated static display surface. No extrapolation outside local PCF support. */
export function rasterizePcfCamera(positions: Float32Array, colors: Uint8Array, p: PcfCameraProjection): PcfCameraRaster {
  requireValue(positions.length % 3 === 0 && positions.length / 3 <= 1_000_000 && colors.length === positions.length, 'pcf_camera_points_invalid');
  const scale = Math.min(1, 960 / p.width, 540 / p.height), width = Math.round(p.width * scale), height = Math.round(p.height * scale);
  const intrinsics = p.intrinsics.map(v => v * scale), [fx, fy, cx, cy] = intrinsics, m = p.cameraFromWorld;
  const samples = new Float32Array(width * height); samples.fill(Infinity);
  const sampleU = new Float32Array(samples.length), sampleV = new Float32Array(samples.length), rgb = new Uint8ClampedArray(samples.length * 4);
  for (let offset = 0; offset < positions.length; offset += 3) {
    const x = positions[offset], y = positions[offset + 1], z = positions[offset + 2];
    const px = m[0] * x + m[4] * y + m[8] * z + m[12], py = m[1] * x + m[5] * y + m[9] * z + m[13], pz = m[2] * x + m[6] * y + m[10] * z + m[14];
    if (!Number.isFinite(px + py + pz) || pz <= .05 || pz > 100) continue;
    const u = fx * px / pz + cx, v = fy * py / pz + cy, col = Math.round(u), row = Math.round(v);
    if (col < 0 || col >= width || row < 0 || row >= height) continue;
    const index = row * width + col;
    if (pz >= samples[index]) continue;
    samples[index] = pz; sampleU[index] = u; sampleV[index] = v;
    rgb.set([colors[offset], colors[offset + 1], colors[offset + 2], 255], index * 4);
  }
  const surface = buildPcfCameraSurface(width, height, intrinsics, samples, sampleU, sampleV, rgb);
  requireValue(surface.validCount > 0, 'pcf_camera_no_visible_surface');
  return surface;
}
