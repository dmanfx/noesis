export type FloorplanViewport = {
  sourceRect: { x: number; y: number; width: number; height: number };
  observedRect: { x: number; y: number; width: number; height: number };
  observedCount: number;
  totalCount: number;
  cropCount: number;
  fullWidthM: number | null;
  fullDepthM: number | null;
  cropWidthM: number | null;
  cropDepthM: number | null;
  cellWidthM: number | null;
  cellDepthM: number | null;
  paddingM: number;
};

export function computeObservedFloorplanViewport(options: {
  maskValues: ArrayLike<number>;
  rows: number;
  cols: number;
  maskInvert?: boolean;
  maskThreshold?: number;
  bounds?: {
    min_x?: number;
    max_x?: number;
    min_z?: number;
    max_z?: number;
  } | null;
  paddingM?: number;
  fallbackPaddingCells?: number;
}): FloorplanViewport | null;
