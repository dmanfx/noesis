export type HeightfieldModel = {
  rows: number;
  cols: number;
  downsampleFactor: number;
  heights: Float32Array;
  observed: Uint8Array;
  coverage: Float32Array;
  positions: Float32Array;
  indices: Uint32Array;
  bounds: {
    minX: number;
    maxX: number;
    minZ: number;
    maxZ: number;
  };
  dx: number;
  dz: number;
  observedCount: number;
  triangleCount: number;
  maxHeightM: number;
  aggregation: {
    coverageThreshold: 1;
    reducer: 'exact' | 'upper_p90';
    maxSourceBlockRows: number;
    maxSourceBlockCols: number;
    maxSourceBlockWidthM: number;
    maxSourceBlockDepthM: number;
  };
};

export function buildMaskedHeightfield(options: {
  heightValues: ArrayLike<number>;
  densityValues?: ArrayLike<number> | null;
  rows: number;
  cols: number;
  bounds?: {
    min_x?: number;
    max_x?: number;
    min_z?: number;
    max_z?: number;
  } | null;
  fallbackGridResM?: number;
  densityThreshold?: number;
  maxVertices?: number;
  heightExaggeration?: number;
}): HeightfieldModel | null;
