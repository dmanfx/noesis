const finiteNumber = (value, fallback = null) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
};

/**
 * Computes a display-only crop around cells that are explicitly observed.
 *
 * The returned rectangle remains in source-grid coordinates; callers render
 * the original layers through it. No grid values, bounds, or semantic labels
 * are rewritten.
 */
export function computeObservedFloorplanViewport({
  maskValues,
  rows,
  cols,
  maskInvert = false,
  maskThreshold = 1e-6,
  bounds = null,
  paddingM = 0.75,
  fallbackPaddingCells = 4,
} = {}) {
  const sourceRows = Math.max(0, Math.floor(Number(rows) || 0));
  const sourceCols = Math.max(0, Math.floor(Number(cols) || 0));
  const total = sourceRows * sourceCols;
  if (
    !maskValues
    || sourceRows <= 0
    || sourceCols <= 0
    || maskValues.length < total
  ) {
    return null;
  }

  let minRow = sourceRows;
  let maxRow = -1;
  let minCol = sourceCols;
  let maxCol = -1;
  let observedCount = 0;
  const threshold = finiteNumber(maskThreshold, 1e-6);

  for (let row = 0; row < sourceRows; row += 1) {
    for (let col = 0; col < sourceCols; col += 1) {
      const value = Number(maskValues[(row * sourceCols) + col]);
      if (!Number.isFinite(value)) continue;
      const positive = value > threshold;
      const observed = maskInvert ? !positive : positive;
      if (!observed) continue;
      observedCount += 1;
      if (row < minRow) minRow = row;
      if (row > maxRow) maxRow = row;
      if (col < minCol) minCol = col;
      if (col > maxCol) maxCol = col;
    }
  }
  if (!observedCount || maxRow < minRow || maxCol < minCol) return null;

  const minX = finiteNumber(bounds?.min_x);
  const maxX = finiteNumber(bounds?.max_x);
  const minZ = finiteNumber(bounds?.min_z);
  const maxZ = finiteNumber(bounds?.max_z);
  const fullWidthM = minX !== null && maxX !== null && maxX > minX
    ? maxX - minX
    : null;
  const fullDepthM = minZ !== null && maxZ !== null && maxZ > minZ
    ? maxZ - minZ
    : null;
  const cellWidthM = fullWidthM === null ? null : fullWidthM / sourceCols;
  const cellDepthM = fullDepthM === null ? null : fullDepthM / sourceRows;
  const requestedPaddingM = Math.max(0, finiteNumber(paddingM, 0.75));
  const fallback = Math.max(0, Math.floor(Number(fallbackPaddingCells) || 0));
  const padCols = requestedPaddingM === 0
    ? 0
    : cellWidthM && cellWidthM > 0
      ? Math.ceil(requestedPaddingM / cellWidthM)
      : fallback;
  const padRows = requestedPaddingM === 0
    ? 0
    : cellDepthM && cellDepthM > 0
      ? Math.ceil(requestedPaddingM / cellDepthM)
      : fallback;

  const x = Math.max(0, minCol - padCols);
  const y = Math.max(0, minRow - padRows);
  const right = Math.min(sourceCols, maxCol + 1 + padCols);
  const bottom = Math.min(sourceRows, maxRow + 1 + padRows);
  const width = Math.max(1, right - x);
  const height = Math.max(1, bottom - y);

  return {
    sourceRect: { x, y, width, height },
    observedRect: {
      x: minCol,
      y: minRow,
      width: (maxCol - minCol) + 1,
      height: (maxRow - minRow) + 1,
    },
    observedCount,
    totalCount: total,
    cropCount: width * height,
    fullWidthM,
    fullDepthM,
    cropWidthM: cellWidthM === null ? null : width * cellWidthM,
    cropDepthM: cellDepthM === null ? null : height * cellDepthM,
    cellWidthM,
    cellDepthM,
    paddingM: requestedPaddingM,
  };
}
