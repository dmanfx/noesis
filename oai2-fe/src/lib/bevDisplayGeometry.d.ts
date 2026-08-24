export type BevMetricBounds = {
  min_x: number;
  max_x: number;
  min_z: number;
  max_z: number;
};

export type BevCoverageRegion = {
  id?: string;
  polygonXZ?: Array<[number, number]>;
};

export type BevCoverageEnvelope = {
  boundaryToleranceM?: number;
  bounds?: BevMetricBounds | null;
  regions?: BevCoverageRegion[];
};

export type BevDisplayPoint = {
  x?: number;
  y?: number;
  normX?: number;
  normY?: number;
  floorplanInside?: boolean | null;
  coverageInside?: boolean | null;
  canonicalWorld?: boolean;
};

export function resolveBevDisplayBounds(args?: {
  floorplanBounds?: BevMetricBounds | null;
  advertisedBounds?: BevMetricBounds | null;
  coverageBounds?: BevMetricBounds | null;
  coverageToleranceM?: number;
}): BevMetricBounds | null;

export function coverageRegionForPoint(
  coverage: BevCoverageEnvelope | null | undefined,
  x: number,
  z: number,
): string | null;

export function resolveBevMetricPoint(args?: {
  point?: BevDisplayPoint | null;
  displayBounds?: BevMetricBounds | null;
  normalizedBounds?: BevMetricBounds | null;
  coverage?: BevCoverageEnvelope | null;
}): { x: number; y: number; mapped: boolean } | null;
