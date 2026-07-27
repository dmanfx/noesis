export type CameraIntrinsics = [number, number, number, number];

export type DepthNormalOptions = {
  confidence?: Float32Array | null;
  bilateralRadius?: number;
  sampleRadius?: number;
  normalSmoothingRadius?: number;
};

export const normalizeCameraIntrinsics = (
  values?: number[] | null,
): CameraIntrinsics | null => {
  if (!Array.isArray(values)) return null;
  let fx: number;
  let fy: number;
  let cx: number;
  let cy: number;
  if (values.length >= 9) {
    fx = Number(values[0]);
    fy = Number(values[4]);
    cx = Number(values[2]);
    cy = Number(values[5]);
  } else if (values.length >= 4) {
    fx = Number(values[0]);
    fy = Number(values[1]);
    cx = Number(values[2]);
    cy = Number(values[3]);
  } else {
    return null;
  }
  if (
    ![fx, fy, cx, cy].every(Number.isFinite)
    || fx <= 1e-6
    || fy <= 1e-6
  ) return null;
  return [fx, fy, cx, cy];
};

const validDepth = (value: number): boolean => (
  Number.isFinite(value) && value > 0.1 && value < 50
);

const boundedRadius = (
  value: number | undefined,
  fallback: number,
  maximum: number,
): number => Math.min(
  maximum,
  Math.max(0, Math.floor(Number(value ?? fallback) || 0)),
);

const confidenceWeight = (
  confidence: Float32Array | null,
  index: number,
): number => {
  if (!confidence) return 1;
  const value = confidence[index];
  if (!Number.isFinite(value)) return 0.05;
  return 0.05 + 0.95 * Math.max(0, Math.min(1, value));
};

const filterMetricDepth = (
  depth: Float32Array,
  confidence: Float32Array | null,
  mask: Uint8Array,
  height: number,
  width: number,
  radius: number,
): Float32Array => {
  if (radius === 0) return depth;
  const filtered = new Float32Array(depth.length);
  const diameter = radius * 2 + 1;
  const spatialWeights = new Float32Array(diameter * diameter);
  const spatialSigma = Math.max(0.8, radius * 0.7);
  for (let dy = -radius; dy <= radius; dy += 1) {
    for (let dx = -radius; dx <= radius; dx += 1) {
      spatialWeights[(dy + radius) * diameter + dx + radius] = Math.exp(
        -(dx * dx + dy * dy) / (2 * spatialSigma * spatialSigma),
      );
    }
  }

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const centerIndex = y * width + x;
      const centerDepth = depth[centerIndex];
      if (mask[centerIndex] === 0 || !validDepth(centerDepth)) continue;
      const rangeSigma = Math.max(0.015, centerDepth * 0.006);
      const hardEdgeThreshold = Math.max(0.05, centerDepth * 0.02);
      let weightedDepth = 0;
      let totalWeight = 0;
      const yStart = Math.max(0, y - radius);
      const yEnd = Math.min(height - 1, y + radius);
      const xStart = Math.max(0, x - radius);
      const xEnd = Math.min(width - 1, x + radius);
      for (let ny = yStart; ny <= yEnd; ny += 1) {
        for (let nx = xStart; nx <= xEnd; nx += 1) {
          const neighborIndex = ny * width + nx;
          const neighborDepth = depth[neighborIndex];
          if (mask[neighborIndex] === 0 || !validDepth(neighborDepth)) continue;
          const delta = neighborDepth - centerDepth;
          if (Math.abs(delta) > hardEdgeThreshold) continue;
          const rangeWeight = Math.exp(
            -(delta * delta) / (2 * rangeSigma * rangeSigma),
          );
          const spatialWeight = spatialWeights[
            (ny - y + radius) * diameter + nx - x + radius
          ];
          const weight = (
            spatialWeight
            * rangeWeight
            * confidenceWeight(confidence, neighborIndex)
          );
          weightedDepth += neighborDepth * weight;
          totalWeight += weight;
        }
      }
      filtered[centerIndex] = totalWeight > 1e-8
        ? weightedDepth / totalWeight
        : centerDepth;
    }
  }
  return filtered;
};

const selectNormalRadius = (
  depth: Float32Array,
  mask: Uint8Array,
  height: number,
  width: number,
  x: number,
  y: number,
  preferredRadius: number,
): number => {
  if (
    x < preferredRadius
    || x >= width - preferredRadius
    || y < preferredRadius
    || y >= height - preferredRadius
  ) return 0;
  const centerIndex = y * width + x;
  const centerDepth = depth[centerIndex];
  if (mask[centerIndex] === 0 || !validDepth(centerDepth)) return 0;

  // Missing support is not interpolated. This keeps calibration/FoV holes and
  // invalid model regions explicit instead of painting plausible normals over
  // absent evidence.
  for (let distance = 1; distance <= preferredRadius; distance += 1) {
    const support = [
      centerIndex - distance,
      centerIndex + distance,
      centerIndex - distance * width,
      centerIndex + distance * width,
    ];
    if (support.some((index) => mask[index] === 0 || !validDepth(depth[index]))) {
      return 0;
    }
  }

  for (let radius = preferredRadius; radius >= 1; radius -= 1) {
    const leftDepth = depth[centerIndex - radius];
    const rightDepth = depth[centerIndex + radius];
    const aboveDepth = depth[centerIndex - radius * width];
    const belowDepth = depth[centerIndex + radius * width];
    const hardDelta = Math.max(0.12, centerDepth * 0.08);
    if (
      Math.max(
        Math.abs(leftDepth - centerDepth),
        Math.abs(rightDepth - centerDepth),
        Math.abs(aboveDepth - centerDepth),
        Math.abs(belowDepth - centerDepth),
      ) > hardDelta
    ) continue;
    const curvatureThreshold = Math.max(0.025, centerDepth * 0.015);
    if (
      Math.abs(leftDepth + rightDepth - 2 * centerDepth) > curvatureThreshold
      || Math.abs(aboveDepth + belowDepth - 2 * centerDepth) > curvatureThreshold
    ) continue;
    return radius;
  }
  return 0;
};

const smoothNormalsInPlace = (
  normals: Float32Array,
  depth: Float32Array,
  confidence: Float32Array | null,
  height: number,
  width: number,
): void => {
  if (height < 2 || width < 2) return;
  const rowStride = width * 3;
  let previousRow = new Float32Array(rowStride);
  let currentRow = new Float32Array(rowStride);
  let nextRow = new Float32Array(rowStride);
  currentRow.set(normals.subarray(0, rowStride));
  nextRow.set(normals.subarray(rowStride, Math.min(normals.length, rowStride * 2)));

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const localIndex = x * 3;
      const centerNx = currentRow[localIndex];
      const centerNy = currentRow[localIndex + 1];
      const centerNz = currentRow[localIndex + 2];
      if (Math.hypot(centerNx, centerNy, centerNz) < 0.5) continue;
      const centerPixel = y * width + x;
      const centerDepth = depth[centerPixel];
      const depthThreshold = Math.max(0.04, centerDepth * 0.02);
      let sumX = 0;
      let sumY = 0;
      let sumZ = 0;
      let totalWeight = 0;
      let supportCount = 0;

      for (let dy = -1; dy <= 1; dy += 1) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        const sourceRow = dy < 0 ? previousRow : dy > 0 ? nextRow : currentRow;
        for (let dx = -1; dx <= 1; dx += 1) {
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          const neighborLocalIndex = nx * 3;
          const neighborNx = sourceRow[neighborLocalIndex];
          const neighborNy = sourceRow[neighborLocalIndex + 1];
          const neighborNz = sourceRow[neighborLocalIndex + 2];
          if (Math.hypot(neighborNx, neighborNy, neighborNz) < 0.5) continue;
          const neighborPixel = ny * width + nx;
          if (Math.abs(depth[neighborPixel] - centerDepth) > depthThreshold) continue;
          const dot = (
            centerNx * neighborNx
            + centerNy * neighborNy
            + centerNz * neighborNz
          );
          if (!Number.isFinite(dot) || dot < 0.55) continue;
          const spatialWeight = dx === 0 && dy === 0
            ? 1
            : dx === 0 || dy === 0 ? 0.82 : 0.64;
          const angularWeight = 0.25 + 0.75 * Math.min(1, (dot - 0.55) / 0.45);
          const weight = (
            spatialWeight
            * angularWeight
            * confidenceWeight(confidence, neighborPixel)
          );
          sumX += neighborNx * weight;
          sumY += neighborNy * weight;
          sumZ += neighborNz * weight;
          totalWeight += weight;
          supportCount += 1;
        }
      }
      if (supportCount < 3 || totalWeight <= 1e-8) continue;
      const magnitude = Math.hypot(sumX, sumY, sumZ);
      if (!Number.isFinite(magnitude) || magnitude <= 1e-10) continue;
      const outputIndex = centerPixel * 3;
      normals[outputIndex] = sumX / magnitude;
      normals[outputIndex + 1] = sumY / magnitude;
      normals[outputIndex + 2] = sumZ / magnitude;
    }

    const reusableRow = previousRow;
    previousRow = currentRow;
    currentRow = nextRow;
    nextRow = reusableRow;
    const sourceY = y + 2;
    if (sourceY < height) {
      const sourceStart = sourceY * rowStride;
      nextRow.set(normals.subarray(sourceStart, sourceStart + rowStride));
    } else {
      nextRow.fill(0);
    }
  }
};

/**
 * Derive denoised, perspective-correct camera-space normals from metric depth.
 *
 * A confidence-aware bilateral filter suppresses model jitter without crossing
 * depth discontinuities. Metric camera-space tangents use the widest locally
 * smooth support radius, contract near real edges, and remain invalid across
 * missing evidence. A final unit-vector pass averages only nearby normals that
 * agree in depth and orientation.
 */
export const deriveCalibratedDepthNormals = (
  depth: Float32Array,
  mask: Uint8Array,
  height: number,
  width: number,
  intrinsics: CameraIntrinsics,
  optionsValue: DepthNormalOptions | number = {},
): Float32Array => {
  const pixelCount = height * width;
  if (
    !Number.isSafeInteger(height)
    || !Number.isSafeInteger(width)
    || height <= 0
    || width <= 0
    || !Number.isSafeInteger(pixelCount)
    || depth.length !== pixelCount
    || mask.length !== pixelCount
  ) {
    throw new Error('depth_normals_shape_invalid');
  }
  const maximumRadius = Math.floor((Math.min(height, width) - 1) / 2);
  if (maximumRadius < 1) {
    throw new Error('depth_normals_shape_too_small');
  }
  const options = typeof optionsValue === 'number'
    ? { sampleRadius: optionsValue }
    : optionsValue;
  const confidence = options.confidence ?? null;
  if (confidence && confidence.length !== pixelCount) {
    throw new Error('depth_normals_confidence_shape_invalid');
  }
  const bilateralRadius = boundedRadius(options.bilateralRadius, 2, 3);
  const preferredRadius = Math.max(
    1,
    boundedRadius(options.sampleRadius, 2, maximumRadius),
  );
  const normalSmoothingRadius = boundedRadius(
    options.normalSmoothingRadius,
    1,
    1,
  );
  const normalizedIntrinsics = normalizeCameraIntrinsics(intrinsics);
  if (!normalizedIntrinsics) {
    throw new Error('depth_normals_intrinsics_invalid');
  }
  const [fx, fy, cx, cy] = normalizedIntrinsics;
  const filteredDepth = filterMetricDepth(
    depth,
    confidence,
    mask,
    height,
    width,
    bilateralRadius,
  );
  const normals = new Float32Array(pixelCount * 3);

  for (let y = preferredRadius; y < height - preferredRadius; y += 1) {
    for (let x = preferredRadius; x < width - preferredRadius; x += 1) {
      const centerIndex = y * width + x;
      const radius = selectNormalRadius(
        filteredDepth,
        mask,
        height,
        width,
        x,
        y,
        preferredRadius,
      );
      if (radius === 0) continue;
      const leftIndex = centerIndex - radius;
      const rightIndex = centerIndex + radius;
      const aboveIndex = centerIndex - radius * width;
      const belowIndex = centerIndex + radius * width;
      if (
        mask[centerIndex] === 0
        || mask[leftIndex] === 0
        || mask[rightIndex] === 0
        || mask[aboveIndex] === 0
        || mask[belowIndex] === 0
      ) continue;

      const centerDepth = filteredDepth[centerIndex];
      const leftDepth = filteredDepth[leftIndex];
      const rightDepth = filteredDepth[rightIndex];
      const aboveDepth = filteredDepth[aboveIndex];
      const belowDepth = filteredDepth[belowIndex];
      if (
        ![centerDepth, leftDepth, rightDepth, aboveDepth, belowDepth]
          .every(validDepth)
      ) continue;

      const leftX = ((x - radius - cx) * leftDepth) / fx;
      const leftY = -((y - cy) * leftDepth) / fy;
      const rightX = ((x + radius - cx) * rightDepth) / fx;
      const rightY = -((y - cy) * rightDepth) / fy;
      const aboveX = ((x - cx) * aboveDepth) / fx;
      const aboveY = -((y - radius - cy) * aboveDepth) / fy;
      const belowX = ((x - cx) * belowDepth) / fx;
      const belowY = -((y + radius - cy) * belowDepth) / fy;

      const tangentUx = rightX - leftX;
      const tangentUy = rightY - leftY;
      const tangentUz = rightDepth - leftDepth;
      const tangentVx = belowX - aboveX;
      const tangentVy = belowY - aboveY;
      const tangentVz = belowDepth - aboveDepth;

      let nx = tangentUy * tangentVz - tangentUz * tangentVy;
      let ny = tangentUz * tangentVx - tangentUx * tangentVz;
      let nz = tangentUx * tangentVy - tangentUy * tangentVx;
      const magnitude = Math.hypot(nx, ny, nz);
      if (!Number.isFinite(magnitude) || magnitude <= 1e-10) continue;
      nx /= magnitude;
      ny /= magnitude;
      nz /= magnitude;

      const centerX = ((x - cx) * centerDepth) / fx;
      const centerY = -((y - cy) * centerDepth) / fy;
      if (nx * centerX + ny * centerY + nz * centerDepth > 0) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
      }

      const outputIndex = centerIndex * 3;
      normals[outputIndex] = nx;
      normals[outputIndex + 1] = ny;
      normals[outputIndex + 2] = nz;
    }
  }
  if (normalSmoothingRadius > 0) {
    smoothNormalsInPlace(
      normals,
      filteredDepth,
      confidence,
      height,
      width,
    );
  }
  return normals;
};
