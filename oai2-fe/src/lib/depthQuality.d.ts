export type MaskedRange = {
  min: number;
  max: number;
  fullMin: number;
  fullMax: number;
  sampleCount: number;
  lowPercentile: number;
  highPercentile: number;
};

export type SimpleRange = {
  min: number;
  max: number;
};

export function computeMaskedRange(
  values: ArrayLike<number> | null | undefined,
  options?: {
    mask?: ArrayLike<number> | null;
    maskThreshold?: number;
    positiveOnly?: boolean;
    lowPercentile?: number;
    highPercentile?: number;
  },
): MaskedRange | null;

export function normalizeLockedRange(
  range: SimpleRange | null | undefined,
  fallback?: SimpleRange | null,
): SimpleRange | null;

export function chooseDepthRange(
  mode: 'auto' | 'full' | 'locked',
  robustRange: SimpleRange | null,
  fullRange: SimpleRange | null,
  lockedRange: SimpleRange | null,
): SimpleRange | null;

export function integerExportScale(
  rows: number,
  cols: number,
  options?: { maxDimension?: number; maxScale?: number },
): number;

export function snapshotExportStem(
  camera: string,
  tab: string,
  options?: {
    depthTs?: number | string | null;
    floorplanTs?: number | string | null;
    now?: Date | string | number;
  },
): string;

export function floorplanObservationCounts(
  observationMeta: {
    observed_cells?: number;
    unknown_cells?: number;
    total_cells?: number;
  } | null | undefined,
  inferredWalkableValues?: ArrayLike<number> | null,
): {
  observed: number | null;
  unknown: number | null;
  total: number | null;
  inferredWalkable: number;
};
