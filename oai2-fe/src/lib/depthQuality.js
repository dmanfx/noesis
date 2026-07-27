const clampPercentile = (value, fallback) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(100, Math.max(0, numeric));
};

const percentileSorted = (sorted, percentile) => {
  if (!sorted.length) return Number.NaN;
  if (sorted.length === 1) return sorted[0];
  const p = clampPercentile(percentile, 0);
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const fraction = idx - lo;
  return sorted[lo] + ((sorted[hi] - sorted[lo]) * fraction);
};

/**
 * Returns the exact display range for finite samples that satisfy the supplied
 * validity mask. The same helper is used by the canvas renderer and export
 * manifest so legend values cannot drift from the rendered raster.
 */
export function computeMaskedRange(values, {
  mask = null,
  maskThreshold = 0,
  positiveOnly = false,
  lowPercentile = 0,
  highPercentile = 100,
} = {}) {
  if (!values || typeof values.length !== 'number') return null;

  const low = clampPercentile(lowPercentile, 0);
  const high = clampPercentile(highPercentile, 100);
  const lowPct = Math.min(low, high);
  const highPct = Math.max(low, high);
  const samples = [];
  const count = values.length;

  for (let idx = 0; idx < count; idx += 1) {
    const value = Number(values[idx]);
    if (!Number.isFinite(value) || (positiveOnly && value <= 0)) continue;
    if (mask) {
      const maskValue = Number(mask[idx]);
      if (!Number.isFinite(maskValue) || maskValue <= maskThreshold) continue;
    }
    samples.push(value);
  }

  if (!samples.length) return null;
  samples.sort((a, b) => a - b);
  const fullMin = samples[0];
  const fullMax = samples[samples.length - 1];
  let min = percentileSorted(samples, lowPct);
  let max = percentileSorted(samples, highPct);

  if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
  if (max <= min) {
    const pad = Math.max(Math.abs(min) * 0.01, 0.001);
    min -= pad;
    max += pad;
  }

  return {
    min,
    max,
    fullMin,
    fullMax,
    sampleCount: samples.length,
    lowPercentile: lowPct,
    highPercentile: highPct,
  };
}

export function normalizeLockedRange(range, fallback = null) {
  const min = Number(range?.min);
  const max = Number(range?.max);
  if (Number.isFinite(min) && Number.isFinite(max) && max > min) {
    return { min, max };
  }
  if (!fallback) return null;
  const fallbackMin = Number(fallback.min);
  const fallbackMax = Number(fallback.max);
  if (!Number.isFinite(fallbackMin) || !Number.isFinite(fallbackMax) || fallbackMax <= fallbackMin) {
    return null;
  }
  return { min: fallbackMin, max: fallbackMax };
}

export function chooseDepthRange(mode, robustRange, fullRange, lockedRange) {
  if (mode === 'locked') {
    return normalizeLockedRange(lockedRange, robustRange || fullRange);
  }
  if (mode === 'full') {
    return normalizeLockedRange(fullRange, robustRange);
  }
  return normalizeLockedRange(robustRange, fullRange);
}

export function integerExportScale(rows, cols, {
  maxDimension = 2048,
  maxScale = 8,
} = {}) {
  const height = Math.max(1, Math.floor(Number(rows) || 1));
  const width = Math.max(1, Math.floor(Number(cols) || 1));
  const dimensionLimit = Math.max(1, Math.floor(Number(maxDimension) || 2048));
  const scaleLimit = Math.max(1, Math.floor(Number(maxScale) || 8));
  return Math.max(1, Math.min(scaleLimit, Math.floor(dimensionLimit / Math.max(height, width)) || 1));
}

export function snapshotExportStem(camera, tab, {
  depthTs = null,
  floorplanTs = null,
  now = new Date(),
} = {}) {
  const safePart = (value) => String(value || 'unknown')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'unknown';
  const date = now instanceof Date ? now : new Date(now);
  const stamp = Number.isFinite(date.getTime())
    ? date.toISOString().replace(/\D/g, '').slice(0, 17)
    : String(Date.now());
  const snapshot = depthTs ?? floorplanTs ?? 'no-snapshot';
  return `${stamp}_${safePart(camera)}_${safePart(tab)}_snapshot-${safePart(snapshot)}`;
}

export function floorplanObservationCounts(observationMeta, inferredWalkableValues = null) {
  const finiteCount = (value) => {
    const numeric = Number(value);
    return Number.isFinite(numeric) && numeric >= 0 ? Math.floor(numeric) : null;
  };
  let inferredWalkable = 0;
  if (inferredWalkableValues && typeof inferredWalkableValues.length === 'number') {
    for (let idx = 0; idx < inferredWalkableValues.length; idx += 1) {
      const value = Number(inferredWalkableValues[idx]);
      if (Number.isFinite(value) && value > 0.5) inferredWalkable += 1;
    }
  }
  return {
    observed: finiteCount(observationMeta?.observed_cells),
    unknown: finiteCount(observationMeta?.unknown_cells),
    total: finiteCount(observationMeta?.total_cells),
    inferredWalkable,
  };
}
