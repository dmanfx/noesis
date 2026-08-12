import { decodeFloat32 } from './renderUtils';
import type {
  FloorPlaneFootprintMesh,
  VisibleFloorPlaneModel,
  VisibleFloorPlaneResult,
} from './visibleFloorPlane';

type GridLayer = {
  grid_b64?: string;
  grid_shape?: [number, number];
};

type FloorplanFrame = {
  snapshot_ts?: number | null;
  bounds?: { min_x?: number; max_x?: number; min_z?: number; max_z?: number };
  units?: string;
  s_obj_to_m?: number;
};

type ScenePriorFloorPlaneOptions = {
  cameraId: string;
  floorplan?: FloorplanFrame | null;
  heightLayer?: GridLayer | null;
  observedLayer?: GridLayer | null;
  minFloorHeightM?: number;
  maxFloorHeightM?: number;
  maxSupportPoints?: number;
};

const finite = (value: unknown): value is number => (
  typeof value === 'number' && Number.isFinite(value)
);

const fail = (message: string): VisibleFloorPlaneResult => ({
  status: 'insufficient_horizontal_support',
  message,
  model: null,
});

/**
 * Build the existing floor-plane representation from an admitted room grid.
 * Low, observed height-AGL cells are floor support; no live inference request
 * is involved.
 */
export function buildScenePriorFloorPlaneModel({
  cameraId,
  floorplan,
  heightLayer,
  observedLayer,
  minFloorHeightM = -0.08,
  maxFloorHeightM = 0.14,
  maxSupportPoints = 1400,
}: ScenePriorFloorPlaneOptions): VisibleFloorPlaneResult {
  if (!heightLayer?.grid_b64 || !heightLayer.grid_shape) {
    return fail('Waiting for conditioned room-walk height data.');
  }
  if (!observedLayer?.grid_b64 || !observedLayer.grid_shape) {
    return fail('Waiting for conditioned room-walk observation data.');
  }
  const [rows, cols] = heightLayer.grid_shape.map((value) => Math.floor(Number(value))) as [number, number];
  if (
    rows <= 0
    || cols <= 0
    || observedLayer.grid_shape[0] !== rows
    || observedLayer.grid_shape[1] !== cols
  ) {
    return fail('Conditioned room-walk floor grids do not share one shape.');
  }
  const heights = decodeFloat32(heightLayer.grid_b64);
  const observed = decodeFloat32(observedLayer.grid_b64);
  if (!heights || !observed || heights.length < rows * cols || observed.length < rows * cols) {
    return fail('Conditioned room-walk floor grids are incomplete.');
  }

  const bounds = floorplan?.bounds;
  const units = String(floorplan?.units || '').toLowerCase();
  const unitScale = units.includes('scene') && finite(floorplan?.s_obj_to_m) && floorplan!.s_obj_to_m! > 0
    ? floorplan!.s_obj_to_m!
    : 1;
  if (
    !finite(bounds?.min_x)
    || !finite(bounds?.max_x)
    || !finite(bounds?.min_z)
    || !finite(bounds?.max_z)
    || bounds!.max_x! <= bounds!.min_x!
    || bounds!.max_z! <= bounds!.min_z!
  ) {
    return fail('Conditioned room-walk floor bounds are unavailable.');
  }

  const fullMinX = bounds!.min_x! * unitScale;
  const fullMaxX = bounds!.max_x! * unitScale;
  const fullMinZ = bounds!.min_z! * unitScale;
  const fullMaxZ = bounds!.max_z! * unitScale;
  const dx = (fullMaxX - fullMinX) / cols;
  const dz = (fullMaxZ - fullMinZ) / rows;
  const active = new Uint8Array(rows * cols);
  let activeCount = 0;
  let minRow = rows;
  let maxRow = -1;
  let minCol = cols;
  let maxCol = -1;

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const index = row * cols + col;
      const height = heights[index];
      const observation = observed[index];
      if (
        !Number.isFinite(height)
        || !Number.isFinite(observation)
        || observation <= 1e-6
        || height < minFloorHeightM
        || height > maxFloorHeightM
      ) continue;
      active[index] = 1;
      activeCount += 1;
      minRow = Math.min(minRow, row);
      maxRow = Math.max(maxRow, row);
      minCol = Math.min(minCol, col);
      maxCol = Math.max(maxCol, col);
    }
  }
  if (activeCount < 16) {
    return fail(`Only ${activeCount} conditioned room-walk floor cells were found.`);
  }

  const vertices: Array<[number, number, number]> = [];
  const indices: number[] = [];
  const boundarySegments: Array<[[number, number, number], [number, number, number]]> = [];
  const vertexByCorner = new Map<string, number>();
  const addVertex = (row: number, col: number): number => {
    const key = `${row}:${col}`;
    const existing = vertexByCorner.get(key);
    if (existing !== undefined) return existing;
    const vertex: [number, number, number] = [
      fullMinX + col * dx,
      0,
      fullMaxZ - row * dz,
    ];
    const index = vertices.length;
    vertices.push(vertex);
    vertexByCorner.set(key, index);
    return index;
  };
  const isActive = (row: number, col: number) => (
    row >= 0 && row < rows && col >= 0 && col < cols && active[row * cols + col] > 0
  );

  const supports: Array<[number, number, number]> = [];
  const supportPixels: Array<[number, number]> = [];
  const supportStride = Math.max(1, Math.ceil(activeCount / Math.max(1, maxSupportPoints)));
  let supportIndex = 0;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      if (!isActive(row, col)) continue;
      const topLeft = addVertex(row, col);
      const topRight = addVertex(row, col + 1);
      const bottomRight = addVertex(row + 1, col + 1);
      const bottomLeft = addVertex(row + 1, col);
      indices.push(topLeft, bottomLeft, bottomRight, topLeft, bottomRight, topRight);
      if (!isActive(row - 1, col)) boundarySegments.push([vertices[topLeft], vertices[topRight]]);
      if (!isActive(row, col + 1)) boundarySegments.push([vertices[topRight], vertices[bottomRight]]);
      if (!isActive(row + 1, col)) boundarySegments.push([vertices[bottomRight], vertices[bottomLeft]]);
      if (!isActive(row, col - 1)) boundarySegments.push([vertices[bottomLeft], vertices[topLeft]]);
      if (supportIndex % supportStride === 0) {
        supports.push([
          fullMinX + (col + 0.5) * dx,
          0,
          fullMaxZ - (row + 0.5) * dz,
        ]);
        supportPixels.push([col, row]);
      }
      supportIndex += 1;
    }
  }

  const minX = fullMinX + minCol * dx;
  const maxX = fullMinX + (maxCol + 1) * dx;
  const maxZ = fullMaxZ - minRow * dz;
  const minZ = fullMaxZ - (maxRow + 1) * dz;
  const corners: Array<[number, number, number]> = [
    [minX, 0, minZ],
    [maxX, 0, minZ],
    [maxX, 0, maxZ],
    [minX, 0, maxZ],
  ];
  const footprintMesh: FloorPlaneFootprintMesh = {
    vertices,
    indices,
    boundarySegments,
    cellCount: activeCount,
    areaM2: activeCount * dx * dz,
  };
  const model: VisibleFloorPlaneModel = {
    cameraId,
    frame: 'camera_local_ground_m',
    depthTsUs: Number.isSafeInteger(floorplan?.snapshot_ts) ? Number(floorplan?.snapshot_ts) : 0,
    normalSpace: 'world',
    horizontalThreshold: 1,
    confidenceThreshold: 0,
    plane: {
      a: 0,
      b: 0,
      c: 0,
      normal: [0, 1, 0],
      offset: 0,
      worldHeightM: 0,
    },
    footprint: {
      minX,
      maxX,
      minZ,
      maxZ,
      widthM: maxX - minX,
      depthM: maxZ - minZ,
      source: 'floorplan_shape',
    },
    footprintMesh,
    corners,
    supportPoints: supports,
    supportPixels,
    metrics: {
      scanStride: 1,
      scannedPixelCount: rows * cols,
      candidatePixelCount: activeCount,
      visibleFloorPixelCount: activeCount,
      estimatedVisibleFloorPixelCount: activeCount,
      clusterPixelCount: activeCount,
      inlierRatio: 1,
      meanConfidence: 1,
      meanHorizontalDot: 1,
      planeAreaM2: footprintMesh.areaM2,
    },
  };
  return {
    status: 'ready',
    message: 'Floor representation generated from promoted room evidence.',
    model,
  };
}
