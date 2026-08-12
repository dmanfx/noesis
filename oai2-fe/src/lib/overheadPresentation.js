const finiteNumber = (value, fallback) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
};

const median = (values) => {
  if (!values.length) return 0;
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2
    ? ordered[middle]
    : (ordered[middle - 1] + ordered[middle]) * 0.5;
};

/**
 * Detects a concentrated high horizontal shell, normally a room ceiling.
 * This is a presentation-only heuristic: callers can omit points/cells above
 * the returned cutoff while retaining the complete source artifact.
 */
export function detectOverheadBand(values, {
  stride = 1,
  offset = 0,
  minimumHeightM = 1.75,
  binWidthM = 0.05,
  clearanceM = 0.15,
  minimumSamples = 128,
  minimumPeakFraction = 0.008,
  minimumWindowFraction = 0.025,
  minimumProminence = 1.8,
} = {}) {
  if (!values || !Number.isFinite(Number(values.length))) return null;
  const sampleStride = Math.max(1, Math.floor(finiteNumber(stride, 1)));
  const sampleOffset = Math.max(0, Math.floor(finiteNumber(offset, 0)));
  const candidateMinimum = finiteNumber(minimumHeightM, 1.75);
  const width = Math.max(0.01, finiteNumber(binWidthM, 0.05));
  const finiteHeights = [];
  let maximum = Number.NEGATIVE_INFINITY;

  for (let index = sampleOffset; index < values.length; index += sampleStride) {
    const height = Number(values[index]);
    if (!Number.isFinite(height)) continue;
    finiteHeights.push(height);
    if (height > maximum) maximum = height;
  }
  if (
    finiteHeights.length < Math.max(8, Math.floor(finiteNumber(minimumSamples, 128)))
    || maximum < candidateMinimum + (width * 2)
  ) return null;

  const binCount = Math.max(1, Math.ceil((maximum - candidateMinimum) / width));
  const bins = new Uint32Array(binCount);
  for (const height of finiteHeights) {
    if (height < candidateMinimum) continue;
    const bin = Math.min(binCount - 1, Math.floor((height - candidateMinimum) / width));
    bins[bin] += 1;
  }

  let peakIndex = 0;
  for (let index = 1; index < bins.length; index += 1) {
    if (bins[index] > bins[peakIndex]) peakIndex = index;
  }
  const peakCount = bins[peakIndex];
  const windowCount = (bins[peakIndex - 1] ?? 0)
    + peakCount
    + (bins[peakIndex + 1] ?? 0);
  const baselineBins = Array.from(bins).filter((_count, index) => (
    Math.abs(index - peakIndex) > 1
  ));
  const baseline = Math.max(1, median(baselineBins));
  const sampleCount = finiteHeights.length;
  if (
    peakCount / sampleCount < finiteNumber(minimumPeakFraction, 0.008)
    || windowCount / sampleCount < finiteNumber(minimumWindowFraction, 0.025)
    || peakCount / baseline < finiteNumber(minimumProminence, 1.8)
  ) return null;

  const peakStartM = candidateMinimum + (peakIndex * width);
  const planeHeightM = peakStartM + (width * 0.5);
  const cutoffM = Math.max(
    candidateMinimum,
    planeHeightM - Math.max(width, finiteNumber(clearanceM, 0.15)),
  );
  let hiddenCount = 0;
  for (const height of finiteHeights) {
    if (height >= cutoffM) hiddenCount += 1;
  }
  return {
    cutoffM,
    planeHeightM,
    hiddenCount,
    sampleCount,
    peakCount,
    windowCount,
  };
}
