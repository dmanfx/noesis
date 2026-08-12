type ClearableCanvas = Pick<HTMLCanvasElement, 'width' | 'height' | 'getContext'>;

const ARCHITECTURAL_HEIGHT_STOPS: ReadonlyArray<{
  heightM: number;
  color: readonly [number, number, number];
}> = [
  { heightM: 0.00, color: [238, 239, 234] },
  { heightM: 0.06, color: [226, 231, 225] },
  { heightM: 0.15, color: [173, 210, 202] },
  { heightM: 0.30, color: [77, 169, 181] },
  { heightM: 0.50, color: [35, 125, 169] },
  { heightM: 0.75, color: [68, 83, 157] },
  { heightM: 1.00, color: [122, 60, 142] },
  { heightM: 1.30, color: [187, 61, 104] },
  { heightM: 1.80, color: [239, 123, 55] },
  { heightM: 2.50, color: [255, 238, 159] },
];

const DETAIL_CONTOURS_M = [0.12, 0.32, 0.70, 1.15, 1.70] as const;

const clamp = (value: number, min: number, max: number): number => (
  Math.min(max, Math.max(min, value))
);

/**
 * A fixed physical-height palette for the primary floorplan. It deliberately
 * gives furniture-height surfaces more color range than a generic heatmap.
 */
export const architecturalHeightColor = (
  heightMValue: number,
): [number, number, number] => {
  const heightM = clamp(
    Number.isFinite(heightMValue) ? heightMValue : 0,
    ARCHITECTURAL_HEIGHT_STOPS[0].heightM,
    ARCHITECTURAL_HEIGHT_STOPS[ARCHITECTURAL_HEIGHT_STOPS.length - 1].heightM,
  );
  for (let index = 0; index < ARCHITECTURAL_HEIGHT_STOPS.length - 1; index += 1) {
    const lower = ARCHITECTURAL_HEIGHT_STOPS[index];
    const upper = ARCHITECTURAL_HEIGHT_STOPS[index + 1];
    if (heightM > upper.heightM) continue;
    const span = upper.heightM - lower.heightM;
    const mix = span > 0 ? (heightM - lower.heightM) / span : 0;
    return [
      Math.round(lower.color[0] + ((upper.color[0] - lower.color[0]) * mix)),
      Math.round(lower.color[1] + ((upper.color[1] - lower.color[1]) * mix)),
      Math.round(lower.color[2] + ((upper.color[2] - lower.color[2]) * mix)),
    ];
  }
  const last = ARCHITECTURAL_HEIGHT_STOPS[ARCHITECTURAL_HEIGHT_STOPS.length - 1];
  return [last.color[0], last.color[1], last.color[2]];
};

export type DetailedFloorplanOverlay = {
  observed: Uint8Array;
  shade: Float32Array;
  contour: Uint8Array;
  frontier: Uint8Array;
  obstacleRim: Uint8Array;
};

type DetailedFloorplanOverlayInput = {
  heightAgl: Float32Array;
  rows: number;
  cols: number;
  observationMask?: Float32Array | null;
  observationMaskInvert?: boolean;
  observationMaskThreshold?: number;
  obstacleHeight?: Float32Array | null;
  obstacleThresholdM?: number;
};

/**
 * Builds display-only relief evidence without changing any floorplan value or
 * semantic label. The result retains continuous height, highlights physical
 * height breaks, and keeps the observed frontier distinct from a claimed wall.
 */
export const buildDetailedFloorplanOverlay = ({
  heightAgl,
  rows: rowsValue,
  cols: colsValue,
  observationMask = null,
  observationMaskInvert = false,
  observationMaskThreshold = 1e-6,
  obstacleHeight = null,
  obstacleThresholdM = 0.05,
}: DetailedFloorplanOverlayInput): DetailedFloorplanOverlay => {
  const rows = Math.max(0, Math.floor(Number(rowsValue) || 0));
  const cols = Math.max(0, Math.floor(Number(colsValue) || 0));
  const count = rows * cols;
  const observed = new Uint8Array(count);
  const shade = new Float32Array(count);
  shade.fill(1);
  const contour = new Uint8Array(count);
  const frontier = new Uint8Array(count);
  const obstacleRim = new Uint8Array(count);
  if (!heightAgl || heightAgl.length < count || count <= 0) {
    return { observed, shade, contour, frontier, obstacleRim };
  }

  const maskUsable = Boolean(observationMask && observationMask.length >= count);
  const maskThreshold = Number.isFinite(observationMaskThreshold)
    ? observationMaskThreshold
    : 1e-6;
  for (let index = 0; index < count; index += 1) {
    const height = heightAgl[index];
    if (!Number.isFinite(height)) continue;
    if (maskUsable && observationMask) {
      const maskValue = observationMask[index];
      if (!Number.isFinite(maskValue)) continue;
      const positive = maskValue > maskThreshold;
      if (observationMaskInvert ? positive : !positive) continue;
    }
    observed[index] = 1;
  }

  const obstacleUsable = Boolean(obstacleHeight && obstacleHeight.length >= count);
  const obstacleThreshold = Number.isFinite(obstacleThresholdM)
    ? obstacleThresholdM
    : 0.05;
  const isObstacle = (index: number): boolean => (
    obstacleUsable
    && Boolean(obstacleHeight)
    && Number.isFinite(obstacleHeight![index])
    && obstacleHeight![index] > obstacleThreshold
  );
  const indexAt = (row: number, col: number): number => (row * cols) + col;

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const index = indexAt(row, col);
      if (!observed[index]) continue;
      const height = heightAgl[index];
      let localMin = height;
      let localMax = height;
      let crossesContour = false;
      let touchesUnknown = false;
      let touchesNonObstacle = false;
      const currentObstacle = isObstacle(index);

      for (let rowOffset = -1; rowOffset <= 1; rowOffset += 1) {
        for (let colOffset = -1; colOffset <= 1; colOffset += 1) {
          if (rowOffset === 0 && colOffset === 0) continue;
          const neighborRow = row + rowOffset;
          const neighborCol = col + colOffset;
          if (
            neighborRow < 0
            || neighborRow >= rows
            || neighborCol < 0
            || neighborCol >= cols
          ) {
            touchesUnknown = true;
            touchesNonObstacle = currentObstacle || touchesNonObstacle;
            continue;
          }
          const neighborIndex = indexAt(neighborRow, neighborCol);
          if (!observed[neighborIndex]) {
            touchesUnknown = true;
            touchesNonObstacle = currentObstacle || touchesNonObstacle;
            continue;
          }
          const neighborHeight = heightAgl[neighborIndex];
          localMin = Math.min(localMin, neighborHeight);
          localMax = Math.max(localMax, neighborHeight);
          if (currentObstacle && !isObstacle(neighborIndex)) {
            touchesNonObstacle = true;
          }
          for (const threshold of DETAIL_CONTOURS_M) {
            if ((height >= threshold) !== (neighborHeight >= threshold)) {
              crossesContour = true;
              break;
            }
          }
        }
      }

      frontier[index] = touchesUnknown ? 1 : 0;
      obstacleRim[index] = currentObstacle && touchesNonObstacle ? 1 : 0;
      contour[index] = (
        crossesContour
        || (height > 0.08 && (localMax - localMin) > 0.18)
      ) ? 1 : 0;

      const neighborHeight = (neighborRow: number, neighborCol: number): number => {
        if (
          neighborRow < 0
          || neighborRow >= rows
          || neighborCol < 0
          || neighborCol >= cols
        ) {
          return height;
        }
        const neighborIndex = indexAt(neighborRow, neighborCol);
        return observed[neighborIndex] ? heightAgl[neighborIndex] : height;
      };
      const gradientX = (
        neighborHeight(row, col + 1) - neighborHeight(row, col - 1)
      ) * 0.5;
      const gradientY = (
        neighborHeight(row + 1, col) - neighborHeight(row - 1, col)
      ) * 0.5;
      shade[index] = clamp(
        0.98 - (gradientX * 0.38) + (gradientY * 0.26),
        0.72,
        1.16,
      );
    }
  }

  return { observed, shade, contour, frontier, obstacleRim };
};

export const clearCanvasPixels = (canvas?: ClearableCanvas | null): boolean => {
  if (!canvas) return false;
  const context = canvas.getContext('2d');
  if (!context) return false;
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, canvas.width, canvas.height);
  return true;
};

/** Clears camera-depth canvases when the selected camera has no depth entry. */
export const clearMissingDepthCanvases = (
  depthEntry: unknown,
  canvases: Array<ClearableCanvas | null | undefined>,
): boolean => {
  if (depthEntry) return false;
  let cleared = false;
  for (const canvas of canvases) {
    cleared = clearCanvasPixels(canvas) || cleared;
  }
  return cleared;
};
