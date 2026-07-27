export const DEPTH_DIAGNOSTICS_METHOD = 'client_bulk_exact_snapshot_v1' as const;

export type DepthDiagnosticsSummary = {
  median: number;
  p10: number;
  p90: number;
  conf_mean: number;
  valid_ratio: number;
  sample_count: number;
  method: typeof DEPTH_DIAGNOSTICS_METHOD;
};

const percentile = (sorted: Float32Array, fraction: number): number => {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return Number(sorted[0]);
  const position = Math.min(1, Math.max(0, fraction)) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return Number(sorted[lower]);
  const weight = position - lower;
  return Number(sorted[lower]) + ((Number(sorted[upper]) - Number(sorted[lower])) * weight);
};

export const deriveDepthDiagnostics = (
  depth: Float32Array,
  conf: Float32Array,
  mask: Uint8Array,
  shape: [number, number],
): DepthDiagnosticsSummary => {
  const height = Number(shape?.[0]);
  const width = Number(shape?.[1]);
  const total = height * width;
  if (
    !Number.isSafeInteger(height)
    || height <= 0
    || !Number.isSafeInteger(width)
    || width <= 0
    || !Number.isSafeInteger(total)
    || depth.length < total
    || conf.length < total
    || mask.length < total
  ) {
    throw new Error('depth_diagnostics_shape_invalid');
  }

  let sampleCount = 0;
  let confidenceTotal = 0;
  for (let index = 0; index < total; index += 1) {
    const depthValue = depth[index];
    const confidenceValue = conf[index];
    if (
      mask[index] === 0
      || !Number.isFinite(depthValue)
      || depthValue <= 0
      || !Number.isFinite(confidenceValue)
    ) continue;
    sampleCount += 1;
    confidenceTotal += confidenceValue;
  }

  if (sampleCount === 0) {
    return {
      median: 0,
      p10: 0,
      p90: 0,
      conf_mean: 0,
      valid_ratio: 0,
      sample_count: 0,
      method: DEPTH_DIAGNOSTICS_METHOD,
    };
  }

  const validDepth = new Float32Array(sampleCount);
  let outputIndex = 0;
  for (let index = 0; index < total; index += 1) {
    const depthValue = depth[index];
    const confidenceValue = conf[index];
    if (
      mask[index] === 0
      || !Number.isFinite(depthValue)
      || depthValue <= 0
      || !Number.isFinite(confidenceValue)
    ) continue;
    validDepth[outputIndex++] = depthValue;
  }
  validDepth.sort();

  return {
    median: percentile(validDepth, 0.5),
    p10: percentile(validDepth, 0.1),
    p90: percentile(validDepth, 0.9),
    conf_mean: confidenceTotal / sampleCount,
    valid_ratio: sampleCount / total,
    sample_count: sampleCount,
    method: DEPTH_DIAGNOSTICS_METHOD,
  };
};
