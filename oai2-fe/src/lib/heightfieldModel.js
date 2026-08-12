const finiteNumber = (value, fallback) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
};

const resolveBounds = (bounds, rows, cols, fallbackGridResM) => {
  const fallback = Math.max(0.001, finiteNumber(fallbackGridResM, 0.15));
  let minX = finiteNumber(bounds?.min_x, 0);
  let maxX = finiteNumber(bounds?.max_x, minX + (cols * fallback));
  let minZ = finiteNumber(bounds?.min_z, 0);
  let maxZ = finiteNumber(bounds?.max_z, minZ + (rows * fallback));
  if (maxX <= minX) maxX = minX + (cols * fallback);
  if (maxZ <= minZ) maxZ = minZ + (rows * fallback);
  return { minX, maxX, minZ, maxZ };
};

/**
 * Builds a bounded metric mesh from an observed height-above-floor grid.
 * Downsampled vertices are observed only when every source cell in their block
 * is observed. Each triangle independently requires all three of its vertices
 * to be observed, so unknown holes remain open without discarding the valid
 * half of every boundary quad.
 */
export function buildMaskedHeightfield({
  heightValues,
  densityValues = null,
  rows,
  cols,
  bounds = null,
  fallbackGridResM = 0.15,
  densityThreshold = 1e-6,
  maxVertices = 65_536,
  heightExaggeration = 1,
  maxVisibleHeightM = Number.POSITIVE_INFINITY,
} = {}) {
  const sourceRows = Math.max(0, Math.floor(Number(rows) || 0));
  const sourceCols = Math.max(0, Math.floor(Number(cols) || 0));
  const sourceCount = sourceRows * sourceCols;
  if (
    !heightValues
    || sourceRows < 2
    || sourceCols < 2
    || heightValues.length < sourceCount
  ) {
    return null;
  }

  const vertexLimit = Math.max(4, Math.floor(Number(maxVertices) || 65_536));
  let factor = Math.max(1, Math.ceil(Math.sqrt(sourceCount / vertexLimit)));
  let targetRows = Math.max(2, Math.ceil(sourceRows / factor));
  let targetCols = Math.max(2, Math.ceil(sourceCols / factor));
  while (targetRows * targetCols > vertexLimit) {
    factor += 1;
    targetRows = Math.max(2, Math.ceil(sourceRows / factor));
    targetCols = Math.max(2, Math.ceil(sourceCols / factor));
  }

  const heights = new Float32Array(targetRows * targetCols);
  heights.fill(Number.NaN);
  const observed = new Uint8Array(targetRows * targetCols);
  const coverage = new Float32Array(targetRows * targetCols);
  const densityCutoff = finiteNumber(densityThreshold, 1e-6);
  const visibleHeightMaximum = finiteNumber(maxVisibleHeightM, Number.POSITIVE_INFINITY);
  const sourceRowRanges = Array.from({ length: targetRows }, (_, row) => {
    const start = Math.floor((row * sourceRows) / targetRows);
    const end = Math.min(
      sourceRows,
      Math.max(start + 1, Math.floor(((row + 1) * sourceRows) / targetRows)),
    );
    return [start, end];
  });
  const sourceColRanges = Array.from({ length: targetCols }, (_, col) => {
    const start = Math.floor((col * sourceCols) / targetCols);
    const end = Math.min(
      sourceCols,
      Math.max(start + 1, Math.floor(((col + 1) * sourceCols) / targetCols)),
    );
    return [start, end];
  });
  let maxSourceBlockRows = 1;
  let maxSourceBlockCols = 1;

  for (let row = 0; row < targetRows; row += 1) {
    const [sourceRow0, sourceRow1] = sourceRowRanges[row];
    maxSourceBlockRows = Math.max(maxSourceBlockRows, sourceRow1 - sourceRow0);
    for (let col = 0; col < targetCols; col += 1) {
      const [sourceCol0, sourceCol1] = sourceColRanges[col];
      maxSourceBlockCols = Math.max(maxSourceBlockCols, sourceCol1 - sourceCol0);
      let highest = Number.NEGATIVE_INFINITY;
      let sourceCellCount = 0;
      let observedSourceCellCount = 0;
      const blockHeights = [];
      for (let sourceRow = sourceRow0; sourceRow < sourceRow1; sourceRow += 1) {
        for (let sourceCol = sourceCol0; sourceCol < sourceCol1; sourceCol += 1) {
          sourceCellCount += 1;
          const sourceIdx = (sourceRow * sourceCols) + sourceCol;
          const height = Number(heightValues[sourceIdx]);
          if (!Number.isFinite(height)) continue;
          if (height >= visibleHeightMaximum) continue;
          if (densityValues) {
            const density = Number(densityValues[sourceIdx]);
            if (!Number.isFinite(density) || density <= densityCutoff) continue;
          }
          observedSourceCellCount += 1;
          blockHeights.push(height);
          if (height > highest) highest = height;
        }
      }
      const idx = (row * targetCols) + col;
      coverage[idx] = sourceCellCount > 0
        ? observedSourceCellCount / sourceCellCount
        : 0;
      if (sourceCellCount > 0 && observedSourceCellCount === sourceCellCount) {
        observed[idx] = 1;
        // Factor-one rendering is exact. For larger, sufficiently populated
        // blocks, a bounded upper statistic prevents one bad source cell from
        // expanding into an entire display block while preserving supported
        // surface peaks.
        let representative = highest;
        if (factor > 1 && blockHeights.length >= 4) {
          blockHeights.sort((left, right) => left - right);
          representative = blockHeights[Math.floor((blockHeights.length - 1) * 0.9)];
        }
        heights[idx] = Math.max(0, representative);
      }
    }
  }

  const resolvedBounds = resolveBounds(bounds, sourceRows, sourceCols, fallbackGridResM);
  const sourceDx = (resolvedBounds.maxX - resolvedBounds.minX) / sourceCols;
  const sourceDz = (resolvedBounds.maxZ - resolvedBounds.minZ) / sourceRows;
  const dx = (resolvedBounds.maxX - resolvedBounds.minX) / targetCols;
  const dz = (resolvedBounds.maxZ - resolvedBounds.minZ) / targetRows;
  const exaggeration = Math.max(0.01, finiteNumber(heightExaggeration, 1));
  const positions = new Float32Array(targetRows * targetCols * 3);
  let observedCount = 0;
  let maxHeightM = 0;

  for (let row = 0; row < targetRows; row += 1) {
    const [sourceRow0, sourceRow1] = sourceRowRanges[row];
    const sourceCenterRow = (sourceRow0 + sourceRow1) * 0.5;
    for (let col = 0; col < targetCols; col += 1) {
      const [sourceCol0, sourceCol1] = sourceColRanges[col];
      const sourceCenterCol = (sourceCol0 + sourceCol1) * 0.5;
      const idx = (row * targetCols) + col;
      const offset = idx * 3;
      positions[offset] = resolvedBounds.minX + (sourceCenterCol * sourceDx);
      positions[offset + 1] = observed[idx] ? heights[idx] * exaggeration : 0;
      positions[offset + 2] = resolvedBounds.maxZ - (sourceCenterRow * sourceDz);
      if (observed[idx]) {
        observedCount += 1;
        if (heights[idx] > maxHeightM) maxHeightM = heights[idx];
      }
    }
  }

  const indices = [];
  for (let row = 0; row < targetRows - 1; row += 1) {
    for (let col = 0; col < targetCols - 1; col += 1) {
      const a = (row * targetCols) + col;
      const b = a + 1;
      const c = a + targetCols;
      const d = c + 1;
      const validCount = observed[a] + observed[b] + observed[c] + observed[d];
      if (validCount < 3) continue;
      if (validCount === 3) {
        if (!observed[a]) indices.push(b, c, d);
        else if (!observed[b]) indices.push(a, c, d);
        else if (!observed[c]) indices.push(a, d, b);
        else indices.push(a, c, b);
        continue;
      }

      // For a fully observed quad, use the diagonal with the smaller height
      // discontinuity. This avoids introducing a long cross-edge facet when a
      // surface step runs through the cell.
      const diagonalAd = Math.abs(heights[a] - heights[d]);
      const diagonalBc = Math.abs(heights[b] - heights[c]);
      if (diagonalAd < diagonalBc) {
        indices.push(a, c, d, a, d, b);
      } else {
        indices.push(a, c, b, b, c, d);
      }
    }
  }

  return {
    rows: targetRows,
    cols: targetCols,
    downsampleFactor: factor,
    heights,
    observed,
    coverage,
    positions,
    indices: Uint32Array.from(indices),
    bounds: resolvedBounds,
    dx,
    dz,
    observedCount,
    triangleCount: indices.length / 3,
    maxHeightM,
    aggregation: {
      coverageThreshold: 1,
      reducer: factor > 1 ? 'upper_p90' : 'exact',
      maxSourceBlockRows,
      maxSourceBlockCols,
      maxSourceBlockWidthM: maxSourceBlockCols * sourceDx,
      maxSourceBlockDepthM: maxSourceBlockRows * sourceDz,
    },
  };
}
