export type OverheadBand = {
  cutoffM: number;
  planeHeightM: number;
  hiddenCount: number;
  sampleCount: number;
  peakCount: number;
  windowCount: number;
};

export function detectOverheadBand(
  values: ArrayLike<number>,
  options?: {
    stride?: number;
    offset?: number;
    minimumHeightM?: number;
    binWidthM?: number;
    clearanceM?: number;
    minimumSamples?: number;
    minimumPeakFraction?: number;
    minimumWindowFraction?: number;
    minimumProminence?: number;
  },
): OverheadBand | null;
