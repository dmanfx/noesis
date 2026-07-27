import { decodeFloat32 } from './renderUtils';
import { extractPoseFromExtrinsics } from './calibration';

export type FloorPlaneDepthEntry = {
  ts: number;
  depth: Float32Array;
  conf: Float32Array;
  mask: Uint8Array;
  shape: [number, number];
  snapshotId?: string;
  snapshotRef?: string;
  snapshotContentSha256?: string;
};

export type FloorPlaneFloorplan = {
  snapshot_ts?: number | null;
  snapshot_id?: string;
  snapshot_ref?: string;
  snapshot_content_sha256?: string;
  bounds?: { min_x?: number; max_x?: number; min_z?: number; max_z?: number };
  units?: string;
  s_obj_to_m?: number;
  density?: FloorPlaneLayer;
  walkable?: FloorPlaneLayer;
  obstacle_height?: FloorPlaneLayer;
};

export type FloorPlaneLayer = {
  grid_b64?: string;
  grid_shape?: [number, number];
  value_min?: number;
  value_max?: number;
};

export type FloorPlaneFootprintMesh = {
  vertices: Array<[number, number, number]>;
  indices: number[];
  boundarySegments: Array<[[number, number, number], [number, number, number]]>;
  cellCount: number;
  areaM2: number;
};

export type VisibleFloorPlaneModel = {
  cameraId: string;
  frame: 'camera_local_ground_m';
  depthTsUs: number;
  normalSpace: 'camera' | 'world';
  horizontalThreshold: number;
  confidenceThreshold: number;
  plane: {
    a: number;
    b: number;
    c: number;
    normal: [number, number, number];
    offset: number;
    worldHeightM: number;
  };
  footprint: {
    minX: number;
    maxX: number;
    minZ: number;
    maxZ: number;
    widthM: number;
    depthM: number;
    source: 'floorplan_shape' | 'floorplan_bounds' | 'visible_support';
  };
  footprintMesh: FloorPlaneFootprintMesh;
  corners: Array<[number, number, number]>;
  supportPoints: Array<[number, number, number]>;
  supportPixels: Array<[number, number]>;
  metrics: {
    scanStride: number;
    scannedPixelCount: number;
    candidatePixelCount: number;
    visibleFloorPixelCount: number;
    estimatedVisibleFloorPixelCount: number;
    clusterPixelCount: number;
    inlierRatio: number;
    meanConfidence: number;
    meanHorizontalDot: number;
    planeAreaM2: number;
  };
};

export type VisibleFloorPlaneResult = {
  status:
    | 'ready'
    | 'missing_depth'
    | 'missing_confidence'
    | 'missing_intrinsics'
    | 'missing_extrinsics'
    | 'bad_payload'
    | 'insufficient_horizontal_support'
    | 'plane_fit_failed';
  message: string;
  model: VisibleFloorPlaneModel | null;
};

type BuildOptions = {
  cameraId: string;
  depthEntry?: FloorPlaneDepthEntry;
  intrinsics?: number[] | null;
  extrinsics?: number[] | null;
  floorplan?: FloorPlaneFloorplan | null;
  maxScanPixels?: number;
  maxSupportPoints?: number;
  confidenceThreshold?: number;
  horizontalThreshold?: number;
};

type Candidate = {
  x: number;
  y: number;
  z: number;
  worldY: number;
  confidence: number;
  horizontalDot: number;
  weight: number;
  u: number;
  v: number;
};

const DEFAULT_MAX_SCAN_PIXELS = 240_000;
const DEFAULT_MAX_SUPPORT_POINTS = 1400;
const DEFAULT_CONFIDENCE_THRESHOLD = 0.45;
const DEFAULT_HORIZONTAL_THRESHOLD = 0.82;
const WORLD_UP: [number, number, number] = [0, 1, 0];

const fail = (status: VisibleFloorPlaneResult['status'], message: string): VisibleFloorPlaneResult => ({
  status,
  message,
  model: null,
});

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

export function floorplanMatchesDepthSnapshot(
  depthEntry?: FloorPlaneDepthEntry | null,
  floorplan?: FloorPlaneFloorplan | null,
): boolean {
  if (!depthEntry || !floorplan) return false;
  const depthTs = depthEntry.ts;
  const floorplanTs = floorplan.snapshot_ts;
  return Boolean(
    Number.isSafeInteger(depthTs)
    && Number(depthTs) > 0
    && Number.isSafeInteger(floorplanTs)
    && Number(floorplanTs) === Number(depthTs)
    && typeof depthEntry.snapshotId === 'string'
    && depthEntry.snapshotId.length > 0
    && floorplan.snapshot_id === depthEntry.snapshotId
    && typeof depthEntry.snapshotRef === 'string'
    && depthEntry.snapshotRef.length > 0
    && floorplan.snapshot_ref === depthEntry.snapshotRef
    && typeof depthEntry.snapshotContentSha256 === 'string'
    && /^[0-9a-f]{64}$/.test(depthEntry.snapshotContentSha256)
    && floorplan.snapshot_content_sha256 === depthEntry.snapshotContentSha256
  );
}

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

function normalizeIntrinsics(values?: number[] | null): [number, number, number, number] | null {
  if (!Array.isArray(values)) return null;
  if (values.length >= 4 && values.length !== 9) {
    const [fx, fy, cx, cy] = values.map(Number);
    if ([fx, fy, cx, cy].every((v) => Number.isFinite(v)) && Math.abs(fx) > 1e-6 && Math.abs(fy) > 1e-6) {
      return [fx, fy, cx, cy];
    }
  }
  if (values.length >= 9) {
    const fx = Number(values[0]);
    const fy = Number(values[4]);
    const cx = Number(values[2]);
    const cy = Number(values[5]);
    if ([fx, fy, cx, cy].every((v) => Number.isFinite(v)) && Math.abs(fx) > 1e-6 && Math.abs(fy) > 1e-6) {
      return [fx, fy, cx, cy];
    }
  }
  return null;
}

function normalizeVec3(v: [number, number, number]): [number, number, number] | null {
  const mag = Math.hypot(v[0], v[1], v[2]);
  if (!Number.isFinite(mag) || mag <= 1e-6) return null;
  return [v[0] / mag, v[1] / mag, v[2] / mag];
}

export type DepthNormalSample = {
  depth: Float32Array;
  mask: Uint8Array;
  width: number;
  height: number;
  u: number;
  v: number;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
};

/**
 * Derive a metric camera-space normal from four neighboring depth samples.
 *
 * This deliberately does not consume the display-normal tensor. Each depth
 * neighbor is unprojected with the exact camera intrinsics, then the horizontal
 * and vertical camera-space tangents are crossed. Invalid center/neighbors fail
 * closed so they cannot manufacture horizontal floor support.
 */
export function deriveCameraDepthNormal(sample: DepthNormalSample): [number, number, number] | null {
  const {
    depth,
    mask,
    width,
    height,
    u,
    v,
    fx,
    fy,
    cx,
    cy,
  } = sample;
  if (
    !Number.isSafeInteger(width)
    || !Number.isSafeInteger(height)
    || width <= 2
    || height <= 2
    || !Number.isSafeInteger(u)
    || !Number.isSafeInteger(v)
    || u <= 0
    || v <= 0
    || u >= width - 1
    || v >= height - 1
    || !Number.isFinite(fx)
    || !Number.isFinite(fy)
    || !Number.isFinite(cx)
    || !Number.isFinite(cy)
    || Math.abs(fx) <= 1e-6
    || Math.abs(fy) <= 1e-6
    || !Number.isSafeInteger(width * height)
    || !(depth instanceof Float32Array)
    || !(mask instanceof Uint8Array)
    || depth.length !== width * height
    || mask.length !== width * height
  ) return null;

  const pointAt = (pixelU: number, pixelV: number): [number, number, number] | null => {
    const index = pixelV * width + pixelU;
    const z = depth[index];
    if (mask[index] <= 0 || !Number.isFinite(z) || z <= 0.1 || z >= 50) return null;
    return [
      ((pixelU - cx) * z) / fx,
      -((pixelV - cy) * z) / fy,
      z,
    ];
  };

  if (!pointAt(u, v)) return null;
  const left = pointAt(u - 1, v);
  const right = pointAt(u + 1, v);
  const above = pointAt(u, v - 1);
  const below = pointAt(u, v + 1);
  if (!left || !right || !above || !below) return null;

  const tangentU: [number, number, number] = [
    right[0] - left[0],
    right[1] - left[1],
    right[2] - left[2],
  ];
  const tangentV: [number, number, number] = [
    below[0] - above[0],
    below[1] - above[1],
    below[2] - above[2],
  ];
  return normalizeVec3([
    tangentU[1] * tangentV[2] - tangentU[2] * tangentV[1],
    tangentU[2] * tangentV[0] - tangentU[0] * tangentV[2],
    tangentU[0] * tangentV[1] - tangentU[1] * tangentV[0],
  ]);
}

function mat3Vec(m: number[][], v: [number, number, number]): [number, number, number] {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ];
}

function worldPoint(rwc: number[][], cw: [number, number, number], point: [number, number, number]): [number, number, number] {
  const rotated = mat3Vec(rwc, point);
  return [rotated[0] + cw[0], rotated[1] + cw[1], rotated[2] + cw[2]];
}

function solve3x3(a: number[][], b: number[]): [number, number, number] | null {
  const m = [
    [a[0][0], a[0][1], a[0][2], b[0]],
    [a[1][0], a[1][1], a[1][2], b[1]],
    [a[2][0], a[2][1], a[2][2], b[2]],
  ];

  for (let col = 0; col < 3; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < 3; row += 1) {
      if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    }
    if (Math.abs(m[pivot][col]) < 1e-9) return null;
    if (pivot !== col) {
      const tmp = m[col];
      m[col] = m[pivot];
      m[pivot] = tmp;
    }
    const div = m[col][col];
    for (let j = col; j < 4; j += 1) m[col][j] /= div;
    for (let row = 0; row < 3; row += 1) {
      if (row === col) continue;
      const factor = m[row][col];
      for (let j = col; j < 4; j += 1) m[row][j] -= factor * m[col][j];
    }
  }

  return [m[0][3], m[1][3], m[2][3]];
}

function fitPlaneY(points: Candidate[]): { a: number; b: number; c: number; normal: [number, number, number]; offset: number } | null {
  if (points.length < 3) return null;
  let sW = 0;
  let sX = 0;
  let sZ = 0;
  let sY = 0;
  let sXX = 0;
  let sZZ = 0;
  let sXZ = 0;
  let sXY = 0;
  let sZY = 0;
  for (const p of points) {
    const w = Math.max(1e-4, p.weight);
    sW += w;
    sX += w * p.x;
    sZ += w * p.z;
    sY += w * p.y;
    sXX += w * p.x * p.x;
    sZZ += w * p.z * p.z;
    sXZ += w * p.x * p.z;
    sXY += w * p.x * p.y;
    sZY += w * p.z * p.y;
  }
  const solution = solve3x3(
    [
      [sXX, sXZ, sX],
      [sXZ, sZZ, sZ],
      [sX, sZ, sW],
    ],
    [sXY, sZY, sY],
  );
  if (!solution) return null;
  const [a, b, c] = solution;
  if (![a, b, c].every(Number.isFinite)) return null;
  const invNorm = 1 / Math.hypot(a, 1, b);
  const normal: [number, number, number] = [-a * invNorm, invNorm, -b * invNorm];
  const offset = -c * invNorm;
  return { a, b, c, normal, offset };
}

function percentile(sorted: number[], pct: number): number {
  if (!sorted.length) return 0;
  if (sorted.length === 1) return sorted[0];
  const p = Math.min(100, Math.max(0, pct));
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function summarize(values: number[]): { min: number; max: number; median: number } {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return { min: 0, max: 0, median: 0 };
  return {
    min: sorted[0],
    max: sorted[sorted.length - 1],
    median: percentile(sorted, 50),
  };
}

function clusterByWorldHeight(
  candidates: Candidate[],
  minBinCount: number,
): Array<{ minY: number; maxY: number; count: number; centerY: number }> {
  if (!candidates.length) return [];
  const binSizeM = 0.08;
  const counts = new Map<number, number>();
  for (const p of candidates) {
    const bin = Math.floor(p.worldY / binSizeM);
    counts.set(bin, (counts.get(bin) || 0) + 1);
  }
  const bins = Array.from(counts.keys())
    .filter((bin) => (counts.get(bin) || 0) >= minBinCount)
    .sort((a, b) => a - b);
  if (!bins.length) return [];
  const clusters: Array<{ minY: number; maxY: number; count: number; centerY: number }> = [];
  let start = bins[0];
  let prev = bins[0];
  let count = counts.get(prev) || 0;
  for (let i = 1; i < bins.length; i += 1) {
    const bin = bins[i];
    if (bin <= prev + 1) {
      count += counts.get(bin) || 0;
      prev = bin;
      continue;
    }
    clusters.push({
      minY: start * binSizeM,
      maxY: (prev + 1) * binSizeM,
      count,
      centerY: ((start + prev + 1) * binSizeM) * 0.5,
    });
    start = bin;
    prev = bin;
    count = counts.get(bin) || 0;
  }
  clusters.push({
    minY: start * binSizeM,
    maxY: (prev + 1) * binSizeM,
    count,
    centerY: ((start + prev + 1) * binSizeM) * 0.5,
  });
  return clusters.sort((a, b) => a.centerY - b.centerY);
}

function planeY(plane: { a: number; b: number; c: number }, x: number, z: number): number {
  return plane.a * x + plane.b * z + plane.c;
}

type FootprintResolution = {
  footprint: VisibleFloorPlaneModel['footprint'];
  corners: Array<[number, number, number]>;
  mesh: FloorPlaneFootprintMesh;
};

function layerToMask(layer: FloorPlaneLayer | undefined, threshold: number): { rows: number; cols: number; mask: Uint8Array } | null {
  if (!layer?.grid_b64 || !Array.isArray(layer.grid_shape) || layer.grid_shape.length < 2) return null;
  const rows = Math.floor(Number(layer.grid_shape[0]) || 0);
  const cols = Math.floor(Number(layer.grid_shape[1]) || 0);
  if (rows <= 0 || cols <= 0) return null;
  const values = decodeFloat32(layer.grid_b64);
  if (!values || values.length < rows * cols) return null;
  const mask = new Uint8Array(rows * cols);
  for (let i = 0; i < rows * cols; i += 1) {
    const v = values[i];
    if (Number.isFinite(v) && v > threshold) mask[i] = 1;
  }
  return { rows, cols, mask };
}

function mergeFloorplanMasks(floorplan: FloorPlaneFloorplan | null | undefined): { rows: number; cols: number; mask: Uint8Array } | null {
  const masks = [
    layerToMask(floorplan?.density, 1e-5),
    layerToMask(floorplan?.walkable, 0.15),
    layerToMask(floorplan?.obstacle_height, 0.03),
  ].filter(Boolean) as Array<{ rows: number; cols: number; mask: Uint8Array }>;
  if (!masks.length) return null;
  const rows = masks[0].rows;
  const cols = masks[0].cols;
  const compatible = masks.filter((item) => item.rows === rows && item.cols === cols);
  if (!compatible.length) return null;

  let mask = new Uint8Array(rows * cols);
  for (const item of compatible) {
    for (let i = 0; i < mask.length; i += 1) {
      if (item.mask[i]) mask[i] = 1;
    }
  }

  // Smooth the raster silhouette just enough to fill one-cell holes and drop isolated speckles.
  const closed = new Uint8Array(mask.length);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      let neighbors = 0;
      for (let dr = -1; dr <= 1; dr += 1) {
        for (let dc = -1; dc <= 1; dc += 1) {
          if (dr === 0 && dc === 0) continue;
          const rr = r + dr;
          const cc = c + dc;
          if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) continue;
          neighbors += mask[rr * cols + cc] ? 1 : 0;
        }
      }
      const idx = r * cols + c;
      closed[idx] = mask[idx] || neighbors >= 5 ? 1 : 0;
    }
  }
  const cleaned = new Uint8Array(mask.length);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const idx = r * cols + c;
      if (!closed[idx]) continue;
      let neighbors = 0;
      for (let dr = -1; dr <= 1; dr += 1) {
        for (let dc = -1; dc <= 1; dc += 1) {
          if (dr === 0 && dc === 0) continue;
          const rr = r + dr;
          const cc = c + dc;
          if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) continue;
          neighbors += closed[rr * cols + cc] ? 1 : 0;
        }
      }
      cleaned[idx] = neighbors >= 2 ? 1 : 0;
    }
  }

  return { rows, cols, mask: cleaned };
}

function rectangleFootprintMesh(
  footprint: VisibleFloorPlaneModel['footprint'],
  plane: { a: number; b: number; c: number },
): { corners: Array<[number, number, number]>; mesh: FloorPlaneFootprintMesh } {
  const corners: Array<[number, number, number]> = [
    [footprint.minX, planeY(plane, footprint.minX, footprint.minZ), footprint.minZ],
    [footprint.maxX, planeY(plane, footprint.maxX, footprint.minZ), footprint.minZ],
    [footprint.maxX, planeY(plane, footprint.maxX, footprint.maxZ), footprint.maxZ],
    [footprint.minX, planeY(plane, footprint.minX, footprint.maxZ), footprint.maxZ],
  ];
  return {
    corners,
    mesh: {
      vertices: corners,
      indices: [0, 1, 2, 0, 2, 3],
      boundarySegments: [
        [corners[0], corners[1]],
        [corners[1], corners[2]],
        [corners[2], corners[3]],
        [corners[3], corners[0]],
      ],
      cellCount: 1,
      areaM2: footprint.widthM * footprint.depthM,
    },
  };
}

function floorplanShapeFootprintMesh(
  floorplan: FloorPlaneFloorplan | null | undefined,
  footprint: VisibleFloorPlaneModel['footprint'],
  plane: { a: number; b: number; c: number },
): {
  footprint: VisibleFloorPlaneModel['footprint'];
  corners: Array<[number, number, number]>;
  mesh: FloorPlaneFootprintMesh;
} | null {
  const merged = mergeFloorplanMasks(floorplan);
  if (!merged) return null;
  const { rows, cols, mask } = merged;
  const activeCount = mask.reduce((sum, value) => sum + (value ? 1 : 0), 0);
  if (activeCount <= 0) return null;

  const vertices: Array<[number, number, number]> = [];
  const indices: number[] = [];
  const boundarySegments: Array<[[number, number, number], [number, number, number]]> = [];
  const vertexByCorner = new Map<string, number>();
  const widthM = footprint.maxX - footprint.minX;
  const depthM = footprint.maxZ - footprint.minZ;
  if (widthM <= 0 || depthM <= 0) return null;

  const addVertex = (rCorner: number, cCorner: number): number => {
    const key = `${rCorner}:${cCorner}`;
    const existing = vertexByCorner.get(key);
    if (existing !== undefined) return existing;
    const x = footprint.minX + (cCorner / cols) * widthM;
    const z = footprint.maxZ - (rCorner / rows) * depthM;
    const vertex: [number, number, number] = [x, planeY(plane, x, z), z];
    const idx = vertices.length;
    vertices.push(vertex);
    vertexByCorner.set(key, idx);
    return idx;
  };
  const isActive = (r: number, c: number) => r >= 0 && r < rows && c >= 0 && c < cols && mask[r * cols + c] > 0;

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      if (!isActive(r, c)) continue;
      const tl = addVertex(r, c);
      const tr = addVertex(r, c + 1);
      const br = addVertex(r + 1, c + 1);
      const bl = addVertex(r + 1, c);
      indices.push(tl, bl, br, tl, br, tr);

      if (!isActive(r - 1, c)) boundarySegments.push([vertices[tl], vertices[tr]]);
      if (!isActive(r, c + 1)) boundarySegments.push([vertices[tr], vertices[br]]);
      if (!isActive(r + 1, c)) boundarySegments.push([vertices[br], vertices[bl]]);
      if (!isActive(r, c - 1)) boundarySegments.push([vertices[bl], vertices[tl]]);
    }
  }

  if (!vertices.length || indices.length < 3) return null;
  const xs = vertices.map((v) => v[0]);
  const zs = vertices.map((v) => v[2]);
  const xSummary = summarize(xs);
  const zSummary = summarize(zs);
  const corners: Array<[number, number, number]> = [
    [xSummary.min, planeY(plane, xSummary.min, zSummary.min), zSummary.min],
    [xSummary.max, planeY(plane, xSummary.max, zSummary.min), zSummary.min],
    [xSummary.max, planeY(plane, xSummary.max, zSummary.max), zSummary.max],
    [xSummary.min, planeY(plane, xSummary.min, zSummary.max), zSummary.max],
  ];
  const shapedFootprint: VisibleFloorPlaneModel['footprint'] = {
    minX: xSummary.min,
    maxX: xSummary.max,
    minZ: zSummary.min,
    maxZ: zSummary.max,
    widthM: Math.max(0, xSummary.max - xSummary.min),
    depthM: Math.max(0, zSummary.max - zSummary.min),
    source: 'floorplan_shape',
  };

  return {
    footprint: shapedFootprint,
    corners,
    mesh: {
      vertices,
      indices,
      boundarySegments,
      cellCount: activeCount,
      areaM2: activeCount * (widthM / cols) * (depthM / rows),
    },
  };
}

function resolveFootprint(
  floorplan: FloorPlaneFloorplan | null | undefined,
  support: Candidate[],
  plane: { a: number; b: number; c: number },
): FootprintResolution {
  const bounds = floorplan?.bounds;
  const units = String(floorplan?.units || '').toLowerCase();
  const unitScale = units.includes('scene') && finite(floorplan?.s_obj_to_m) && floorplan!.s_obj_to_m! > 0
    ? floorplan!.s_obj_to_m!
    : 1;
  const hasBounds = bounds &&
    finite(bounds.min_x) &&
    finite(bounds.max_x) &&
    finite(bounds.min_z) &&
    finite(bounds.max_z) &&
    bounds.max_x! > bounds.min_x! &&
    bounds.max_z! > bounds.min_z!;

  const xs = support.map((p) => p.x);
  const zs = support.map((p) => p.z);
  const xSummary = summarize(xs);
  const zSummary = summarize(zs);
  const padX = Math.max(0.35, (xSummary.max - xSummary.min) * 0.08);
  const padZ = Math.max(0.35, (zSummary.max - zSummary.min) * 0.08);

  if (hasBounds) {
    const minX = bounds!.min_x! * unitScale;
    const maxX = bounds!.max_x! * unitScale;
    const minZ = bounds!.min_z! * unitScale;
    const maxZ = bounds!.max_z! * unitScale;
    const footprint: VisibleFloorPlaneModel['footprint'] = {
      minX,
      maxX,
      minZ,
      maxZ,
      widthM: maxX - minX,
      depthM: maxZ - minZ,
      source: 'floorplan_bounds',
    };
    const shaped = floorplanShapeFootprintMesh(floorplan, { ...footprint, source: 'floorplan_shape' }, plane);
    if (shaped) {
      return shaped;
    }
    return { footprint, ...rectangleFootprintMesh(footprint, plane) };
  }

  const minX = xSummary.min - padX;
  const maxX = xSummary.max + padX;
  const minZ = Math.max(0, zSummary.min - padZ);
  const maxZ = zSummary.max + padZ;
  const footprint: VisibleFloorPlaneModel['footprint'] = {
    minX,
    maxX,
    minZ,
    maxZ,
    widthM: Math.max(0, maxX - minX),
    depthM: Math.max(0, maxZ - minZ),
    source: 'visible_support',
  };
  return { footprint, ...rectangleFootprintMesh(footprint, plane) };
}

function sampleSupport(points: Candidate[], maxPoints: number): Candidate[] {
  if (points.length <= maxPoints) return points.slice();
  const out: Candidate[] = [];
  const step = points.length / maxPoints;
  for (let i = 0; i < maxPoints; i += 1) {
    out.push(points[Math.min(points.length - 1, Math.floor(i * step))]);
  }
  return out;
}

export function buildVisibleFloorPlaneModel(options: BuildOptions): VisibleFloorPlaneResult {
  const {
    cameraId,
    depthEntry,
    floorplan,
    maxScanPixels = DEFAULT_MAX_SCAN_PIXELS,
    maxSupportPoints = DEFAULT_MAX_SUPPORT_POINTS,
    confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD,
    horizontalThreshold = DEFAULT_HORIZONTAL_THRESHOLD,
  } = options;

  const floorplanIdentityMatches = floorplanMatchesDepthSnapshot(
    depthEntry,
    floorplan,
  );
  const exactFloorplan = floorplanIdentityMatches ? floorplan : null;

  if (!depthEntry) return fail('missing_depth', 'Waiting for a depth snapshot.');
  if (!(depthEntry.depth instanceof Float32Array) || !Array.isArray(depthEntry.shape)) {
    return fail('bad_payload', 'Depth payload is incomplete.');
  }
  if (!(depthEntry.conf instanceof Float32Array)) {
    return fail('missing_confidence', 'Depth confidence data is required for visible-floor extraction.');
  }
  if (!(depthEntry.mask instanceof Uint8Array)) {
    return fail('bad_payload', 'Depth validity-mask data is required for visible-floor extraction.');
  }

  const intr = normalizeIntrinsics(options.intrinsics);
  if (!intr) return fail('missing_intrinsics', 'Camera intrinsics are not available yet.');
  const pose = extractPoseFromExtrinsics(options.extrinsics || []);
  if (!pose) return fail('missing_extrinsics', 'Camera extrinsics are not available yet.');

  const [heightRaw, widthRaw] = depthEntry.shape;
  const height = Math.floor(Number(heightRaw) || 0);
  const width = Math.floor(Number(widthRaw) || 0);
  if (height <= 0 || width <= 0) return fail('bad_payload', 'Depth payload has an invalid shape.');

  const total = width * height;
  const depth = depthEntry.depth;
  const conf = depthEntry.conf;
  const mask = depthEntry.mask;
  if (depth.length !== total || conf.length !== total || mask.length !== total) {
    return fail('bad_payload', 'Depth, confidence, or mask tensor length does not match its shape.');
  }

  const [fx, fy, cx, cy] = intr;
  const scanStride = Math.max(1, Math.ceil(Math.sqrt(total / Math.max(1, maxScanPixels))));
  const candidates: Candidate[] = [];
  let scannedPixelCount = 0;

  for (let v = 0; v < height; v += scanStride) {
    for (let u = 0; u < width; u += scanStride) {
      scannedPixelCount += 1;
      const idx = v * width + u;
      const d = depth[idx];
      if (!Number.isFinite(d) || d <= 0.1 || d >= 50) continue;
      if (mask && mask.length > idx && mask[idx] <= 0) continue;
      const confidence = conf[idx];
      if (!Number.isFinite(confidence) || confidence < confidenceThreshold) continue;

      const normalLocal = deriveCameraDepthNormal({
        depth,
        mask,
        width,
        height,
        u,
        v,
        fx,
        fy,
        cx,
        cy,
      });
      const normalWorld = normalLocal
        ? normalizeVec3(mat3Vec(pose.Rwc, normalLocal))
        : null;
      if (!normalWorld) continue;
      const horizontalDot = Math.abs(
        normalWorld[0] * WORLD_UP[0] +
        normalWorld[1] * WORLD_UP[1] +
        normalWorld[2] * WORLD_UP[2]
      );
      if (!Number.isFinite(horizontalDot) || horizontalDot < horizontalThreshold) continue;

      const x = ((u - cx) * d) / fx;
      const y = -((v - cy) * d) / fy;
      const z = d;
      const world = worldPoint(pose.Rwc, pose.Cw, [x, y, z]);
      candidates.push({
        x,
        y,
        z,
        worldY: world[1],
        confidence: clamp01(confidence),
        horizontalDot,
        weight: Math.max(0.05, clamp01(confidence)) * horizontalDot,
        u,
        v,
      });
    }
  }

  const minCandidateCount = Math.max(48, Math.floor(scannedPixelCount * 0.0015));
  if (candidates.length < minCandidateCount) {
    return fail(
      'insufficient_horizontal_support',
      `Only ${candidates.length} high-confidence horizontal pixels were found.`,
    );
  }

  const minClusterCount = Math.max(48, Math.floor(candidates.length * 0.003));
  const clusters = clusterByWorldHeight(candidates, minClusterCount);
  const viableClusters = clusters.filter((cluster) => cluster.count >= minClusterCount).slice(0, 10);
  if (!viableClusters.length) {
    return fail('insufficient_horizontal_support', 'No coherent low horizontal plane cluster was found.');
  }

  for (const cluster of viableClusters) {
    const clusterPad = 0.04;
    const clusterPoints = candidates.filter((p) => p.worldY >= cluster.minY - clusterPad && p.worldY <= cluster.maxY + clusterPad);
    if (clusterPoints.length < minClusterCount) continue;
    const firstFit = fitPlaneY(clusterPoints);
    if (!firstFit) continue;
    const residualThresholdM = 0.07;
    const inliers = clusterPoints.filter((p) => Math.abs(p.y - planeY(firstFit, p.x, p.z)) <= residualThresholdM);
    if (inliers.length < minClusterCount) continue;
    const refinedFit = fitPlaneY(inliers) || firstFit;
    const refinedInliers = clusterPoints.filter((p) => Math.abs(p.y - planeY(refinedFit, p.x, p.z)) <= residualThresholdM);
    if (refinedInliers.length < minClusterCount) continue;

    const support = sampleSupport(refinedInliers, maxSupportPoints);
    const { footprint, corners, mesh: footprintMesh } = resolveFootprint(exactFloorplan, refinedInliers, refinedFit);
    if (footprint.widthM <= 0 || footprint.depthM <= 0 || footprintMesh.indices.length < 3) {
      return fail('plane_fit_failed', 'Visible floor support was found, but the plane footprint is invalid.');
    }

    const worldHeights = refinedInliers.map((p) => p.worldY).sort((a, b) => a - b);
    const confidences = refinedInliers.map((p) => p.confidence);
    const dots = refinedInliers.map((p) => p.horizontalDot);
    const mean = (values: number[]) => values.reduce((sum, v) => sum + v, 0) / Math.max(1, values.length);

    const model: VisibleFloorPlaneModel = {
      cameraId,
      frame: 'camera_local_ground_m',
      depthTsUs: depthEntry.ts,
      normalSpace: 'camera',
      horizontalThreshold,
      confidenceThreshold,
      plane: {
        a: refinedFit.a,
        b: refinedFit.b,
        c: refinedFit.c,
        normal: refinedFit.normal,
        offset: refinedFit.offset,
        worldHeightM: percentile(worldHeights, 50),
      },
      footprint,
      footprintMesh,
      corners,
      supportPoints: support.map((p) => [p.x, p.y, p.z]),
      supportPixels: support.map((p) => [p.u, p.v]),
      metrics: {
        scanStride,
        scannedPixelCount,
        candidatePixelCount: candidates.length,
        visibleFloorPixelCount: refinedInliers.length,
        estimatedVisibleFloorPixelCount: refinedInliers.length * scanStride * scanStride,
        clusterPixelCount: clusterPoints.length,
        inlierRatio: refinedInliers.length / Math.max(1, clusterPoints.length),
        meanConfidence: mean(confidences),
        meanHorizontalDot: mean(dots),
        planeAreaM2: footprintMesh.areaM2,
      },
    };

    return {
      status: 'ready',
      message: floorplan && !floorplanIdentityMatches
        ? 'Visible floor plane extracted from metric depth; floorplan footprint withheld because its snapshot identity does not match.'
        : 'Visible floor plane extracted from high-confidence metric depth geometry.',
      model,
    };
  }

  return fail('plane_fit_failed', 'Horizontal support was found, but no stable floor plane fit passed.');
}
