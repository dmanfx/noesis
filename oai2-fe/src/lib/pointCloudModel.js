const normalizeIntrinsics = (values) => {
  if (!Array.isArray(values)) return null;
  if (values.length >= 4 && values.length !== 9) {
    const [fx, fy, cx, cy] = values.map(Number);
    if ([fx, fy, cx, cy].every(Number.isFinite) && Math.abs(fx) > 1e-6 && Math.abs(fy) > 1e-6) {
      return [fx, fy, cx, cy];
    }
  }
  if (values.length >= 9) {
    const fx = Number(values[0]);
    const fy = Number(values[4]);
    const cx = Number(values[2]);
    const cy = Number(values[5]);
    if ([fx, fy, cx, cy].every(Number.isFinite) && Math.abs(fx) > 1e-6 && Math.abs(fy) > 1e-6) {
      return [fx, fy, cx, cy];
    }
  }
  return null;
};

/**
 * Backprojects a bounded sample of the dense depth payload into the calibrated
 * camera-local frame (+X right, +Y up, +Z forward).
 */
export function buildCalibratedPointCloud({
  depthValues,
  confidenceValues = null,
  maskValues = null,
  rgbValues = null,
  rgbShape = null,
  width,
  height,
  intrinsics,
  maxPoints = 100_000,
  minDepthM = 0.1,
  maxDepthM = 50,
  minConfidence = 0,
} = {}) {
  const cols = Math.max(0, Math.floor(Number(width) || 0));
  const rows = Math.max(0, Math.floor(Number(height) || 0));
  const total = rows * cols;
  const intr = normalizeIntrinsics(intrinsics);
  if (!intr || !depthValues || cols <= 0 || rows <= 0 || depthValues.length < total) {
    return null;
  }

  const pointLimit = Math.max(1, Math.floor(Number(maxPoints) || 100_000));
  const stride = Math.max(1, Math.ceil(Math.sqrt(total / pointLimit)));
  const positions = new Float32Array(pointLimit * 3);
  const depths = new Float32Array(pointLimit);
  const confidences = new Float32Array(pointLimit);
  const sourcePixels = new Uint32Array(pointLimit * 2);
  const hasExactRgb = (
    rgbValues
    && Array.isArray(rgbShape)
    && Number(rgbShape[0]) === rows
    && Number(rgbShape[1]) === cols
    && Number(rgbShape[2]) === 3
    && rgbValues.length >= total * 3
  );
  const rgbColors = hasExactRgb ? new Uint8Array(pointLimit * 3) : null;
  const [fx, fy, cx, cy] = intr;
  const depthMin = Number.isFinite(Number(minDepthM)) ? Number(minDepthM) : 0.1;
  const depthMax = Number.isFinite(Number(maxDepthM)) ? Number(maxDepthM) : 50;
  const confidenceMin = Number.isFinite(Number(minConfidence)) ? Number(minConfidence) : 0;
  let count = 0;

  outer:
  for (let v = 0; v < rows; v += stride) {
    for (let u = 0; u < cols; u += stride) {
      const sourceIdx = (v * cols) + u;
      const depth = Number(depthValues[sourceIdx]);
      if (!Number.isFinite(depth) || depth <= depthMin || depth >= depthMax) continue;
      if (maskValues && Number(maskValues[sourceIdx]) <= 0) continue;
      const confidence = confidenceValues ? Number(confidenceValues[sourceIdx]) : 1;
      if (!Number.isFinite(confidence) || confidence < confidenceMin) continue;

      const target = count * 3;
      positions[target] = ((u - cx) * depth) / fx;
      positions[target + 1] = -((v - cy) * depth) / fy;
      positions[target + 2] = depth;
      depths[count] = depth;
      confidences[count] = confidence;
      sourcePixels[count * 2] = u;
      sourcePixels[count * 2 + 1] = v;
      if (rgbColors) {
        const rgbSource = sourceIdx * 3;
        rgbColors[target] = Number(rgbValues[rgbSource]);
        rgbColors[target + 1] = Number(rgbValues[rgbSource + 1]);
        rgbColors[target + 2] = Number(rgbValues[rgbSource + 2]);
      }
      count += 1;
      if (count >= pointLimit) break outer;
    }
  }

  return {
    positions: positions.slice(0, count * 3),
    depths: depths.slice(0, count),
    confidences: confidences.slice(0, count),
    sourcePixels: sourcePixels.slice(0, count * 2),
    rgbColors: rgbColors ? rgbColors.slice(0, count * 3) : null,
    count,
    stride,
    width: cols,
    height: rows,
    intrinsics: intr,
  };
}
