export type CameraIntrinsics = [number, number, number, number];

export type DepthNormalOptions = {
  confidence?: Float32Array | null;
  bilateralRadius?: number;
  sampleRadius?: number;
  normalSmoothingRadius?: number;
};

export type PlaneAwareSurfaceNormalOptions = {
  blockSize?: number;
  minimumBlockCoherence?: number;
  minimumComponentCoherence?: number;
  maximumMergeAngleDeg?: number;
  minimumConfidence?: number;
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

/**
 * Regularize a calibrated detail-normal field into piecewise-planar surfaces.
 *
 * Small image blocks vote for a dominant normal. Adjacent coherent blocks are
 * joined only when their orientations agree and their shared depth boundary is
 * continuous. Large, globally coherent components receive one robust normal;
 * uncertain pixels, curved regions, and structural boundaries retain the
 * detail normal instead of being forced onto a plane.
 */
export const derivePlaneAwareSurfaceNormals = (
  detailNormals: Float32Array,
  depth: Float32Array,
  confidence: Float32Array,
  mask: Uint8Array,
  height: number,
  width: number,
  options: PlaneAwareSurfaceNormalOptions = {},
): Int8Array => {
  const pixelCount = height * width;
  if (
    !Number.isSafeInteger(height)
    || !Number.isSafeInteger(width)
    || height <= 0
    || width <= 0
    || !Number.isSafeInteger(pixelCount)
    || detailNormals.length !== pixelCount * 3
    || depth.length !== pixelCount
    || confidence.length !== pixelCount
    || mask.length !== pixelCount
  ) {
    throw new Error('surface_normals_shape_invalid');
  }

  const blockSize = Math.min(
    24,
    Math.max(4, Math.floor(Number(options.blockSize ?? 8) || 8)),
  );
  const minimumBlockCoherence = Math.max(
    0.8,
    Math.min(0.9999, Number(options.minimumBlockCoherence ?? 0.94)),
  );
  const minimumComponentCoherence = Math.max(
    0.8,
    Math.min(0.9999, Number(options.minimumComponentCoherence ?? 0.97)),
  );
  const maximumMergeAngleDeg = Math.max(
    3,
    Math.min(35, Number(options.maximumMergeAngleDeg ?? 18)),
  );
  const mergeDotThreshold = Math.cos(maximumMergeAngleDeg * Math.PI / 180);
  const blockAssignmentDotThreshold = Math.cos(
    Math.min(35, maximumMergeAngleDeg + 8) * Math.PI / 180,
  );
  const pixelAssignmentDotThreshold = Math.cos(35 * Math.PI / 180);
  const minimumConfidence = Math.max(
    0,
    Math.min(1, Number(options.minimumConfidence ?? 0.2)),
  );

  const blockColumns = Math.ceil(width / blockSize);
  const blockRows = Math.ceil(height / blockSize);
  const blockCount = blockRows * blockColumns;
  const blockNormals = new Float32Array(blockCount * 3);
  const blockCoherence = new Float32Array(blockCount);
  const blockSupport = new Uint32Array(blockCount);
  const planarBlock = new Uint8Array(blockCount);

  for (let blockY = 0; blockY < blockRows; blockY += 1) {
    const yStart = blockY * blockSize;
    const yEnd = Math.min(height, yStart + blockSize);
    for (let blockX = 0; blockX < blockColumns; blockX += 1) {
      const xStart = blockX * blockSize;
      const xEnd = Math.min(width, xStart + blockSize);
      const blockIndex = blockY * blockColumns + blockX;
      let sumX = 0;
      let sumY = 0;
      let sumZ = 0;
      let totalWeight = 0;
      let support = 0;
      for (let y = yStart; y < yEnd; y += 1) {
        for (let x = xStart; x < xEnd; x += 1) {
          const pixelIndex = y * width + x;
          if (mask[pixelIndex] === 0 || !validDepth(depth[pixelIndex])) continue;
          const normalIndex = pixelIndex * 3;
          const nx = detailNormals[normalIndex];
          const ny = detailNormals[normalIndex + 1];
          const nz = detailNormals[normalIndex + 2];
          const magnitude = Math.hypot(nx, ny, nz);
          if (!Number.isFinite(magnitude) || magnitude < 0.5) continue;
          const weight = confidenceWeight(confidence, pixelIndex);
          sumX += (nx / magnitude) * weight;
          sumY += (ny / magnitude) * weight;
          sumZ += (nz / magnitude) * weight;
          totalWeight += weight;
          support += 1;
        }
      }
      if (support < Math.max(6, Math.floor((xEnd - xStart) * (yEnd - yStart) * 0.2))) {
        continue;
      }
      const magnitude = Math.hypot(sumX, sumY, sumZ);
      if (!Number.isFinite(magnitude) || magnitude <= 1e-10 || totalWeight <= 1e-8) {
        continue;
      }
      const normalIndex = blockIndex * 3;
      blockNormals[normalIndex] = sumX / magnitude;
      blockNormals[normalIndex + 1] = sumY / magnitude;
      blockNormals[normalIndex + 2] = sumZ / magnitude;
      blockCoherence[blockIndex] = magnitude / totalWeight;
      blockSupport[blockIndex] = support;
      if (blockCoherence[blockIndex] >= minimumBlockCoherence) {
        planarBlock[blockIndex] = 1;
      }
    }
  }

  const parents = new Int32Array(blockCount);
  for (let index = 0; index < blockCount; index += 1) parents[index] = index;
  const regionSums = new Float64Array(blockCount * 3);
  const regionSupport = new Float64Array(blockCount);
  for (let blockIndex = 0; blockIndex < blockCount; blockIndex += 1) {
    if (planarBlock[blockIndex] === 0) continue;
    const normalIndex = blockIndex * 3;
    const weight = blockSupport[blockIndex];
    regionSums[normalIndex] = blockNormals[normalIndex] * weight;
    regionSums[normalIndex + 1] = blockNormals[normalIndex + 1] * weight;
    regionSums[normalIndex + 2] = blockNormals[normalIndex + 2] * weight;
    regionSupport[blockIndex] = weight;
  }
  const findRoot = (value: number): number => {
    let root = value;
    while (parents[root] !== root) root = parents[root];
    let current = value;
    while (parents[current] !== current) {
      const next = parents[current];
      parents[current] = root;
      current = next;
    }
    return root;
  };
  const join = (left: number, right: number): boolean => {
    const leftRoot = findRoot(left);
    const rightRoot = findRoot(right);
    if (leftRoot === rightRoot) return true;
    const leftIndex = leftRoot * 3;
    const rightIndex = rightRoot * 3;
    const sumX = regionSums[leftIndex] + regionSums[rightIndex];
    const sumY = regionSums[leftIndex + 1] + regionSums[rightIndex + 1];
    const sumZ = regionSums[leftIndex + 2] + regionSums[rightIndex + 2];
    const support = regionSupport[leftRoot] + regionSupport[rightRoot];
    if (
      support <= 0
      || Math.hypot(sumX, sumY, sumZ) / support < minimumComponentCoherence
    ) return false;
    parents[rightRoot] = leftRoot;
    regionSums[leftIndex] = sumX;
    regionSums[leftIndex + 1] = sumY;
    regionSums[leftIndex + 2] = sumZ;
    regionSupport[leftRoot] = support;
    return true;
  };
  const orientationAgrees = (left: number, right: number): boolean => {
    const leftIndex = left * 3;
    const rightIndex = right * 3;
    return (
      blockNormals[leftIndex] * blockNormals[rightIndex]
      + blockNormals[leftIndex + 1] * blockNormals[rightIndex + 1]
      + blockNormals[leftIndex + 2] * blockNormals[rightIndex + 2]
    ) >= mergeDotThreshold;
  };
  const sharedBoundaryIsContinuous = (
    blockY: number,
    blockX: number,
    horizontal: boolean,
  ): boolean => {
    let comparable = 0;
    let continuous = 0;
    if (horizontal) {
      const leftX = Math.min(width - 1, (blockX + 1) * blockSize - 1);
      const rightX = leftX + 1;
      if (rightX >= width) return false;
      const yStart = blockY * blockSize;
      const yEnd = Math.min(height, yStart + blockSize);
      for (let y = yStart; y < yEnd; y += 1) {
        const leftPixel = y * width + leftX;
        const rightPixel = y * width + rightX;
        const leftDepth = depth[leftPixel];
        const rightDepth = depth[rightPixel];
        if (
          mask[leftPixel] === 0
          || mask[rightPixel] === 0
          || !validDepth(leftDepth)
          || !validDepth(rightDepth)
        ) continue;
        comparable += 1;
        if (
          Math.abs(leftDepth - rightDepth)
          <= Math.max(0.1, Math.min(leftDepth, rightDepth) * 0.04)
        ) continuous += 1;
      }
    } else {
      const aboveY = Math.min(height - 1, (blockY + 1) * blockSize - 1);
      const belowY = aboveY + 1;
      if (belowY >= height) return false;
      const xStart = blockX * blockSize;
      const xEnd = Math.min(width, xStart + blockSize);
      for (let x = xStart; x < xEnd; x += 1) {
        const abovePixel = aboveY * width + x;
        const belowPixel = belowY * width + x;
        const aboveDepth = depth[abovePixel];
        const belowDepth = depth[belowPixel];
        if (
          mask[abovePixel] === 0
          || mask[belowPixel] === 0
          || !validDepth(aboveDepth)
          || !validDepth(belowDepth)
        ) continue;
        comparable += 1;
        if (
          Math.abs(aboveDepth - belowDepth)
          <= Math.max(0.1, Math.min(aboveDepth, belowDepth) * 0.04)
        ) continuous += 1;
      }
    }
    return comparable >= 2 && continuous / comparable >= 0.6;
  };

  for (let blockY = 0; blockY < blockRows; blockY += 1) {
    for (let blockX = 0; blockX < blockColumns; blockX += 1) {
      const blockIndex = blockY * blockColumns + blockX;
      if (planarBlock[blockIndex] === 0) continue;
      if (blockX + 1 < blockColumns) {
        const rightBlock = blockIndex + 1;
        if (
          planarBlock[rightBlock] !== 0
          && orientationAgrees(blockIndex, rightBlock)
          && sharedBoundaryIsContinuous(blockY, blockX, true)
        ) join(blockIndex, rightBlock);
      }
      if (blockY + 1 < blockRows) {
        const belowBlock = blockIndex + blockColumns;
        if (
          planarBlock[belowBlock] !== 0
          && orientationAgrees(blockIndex, belowBlock)
          && sharedBoundaryIsContinuous(blockY, blockX, false)
        ) join(blockIndex, belowBlock);
      }
    }
  }

  const componentSums = new Float64Array(blockCount * 3);
  const componentSupport = new Float64Array(blockCount);
  const componentBlockCount = new Uint32Array(blockCount);
  for (let blockIndex = 0; blockIndex < blockCount; blockIndex += 1) {
    if (planarBlock[blockIndex] === 0) continue;
    const root = findRoot(blockIndex);
    const blockNormalIndex = blockIndex * 3;
    const componentNormalIndex = root * 3;
    const weight = blockSupport[blockIndex];
    componentSums[componentNormalIndex] += blockNormals[blockNormalIndex] * weight;
    componentSums[componentNormalIndex + 1] += blockNormals[blockNormalIndex + 1] * weight;
    componentSums[componentNormalIndex + 2] += blockNormals[blockNormalIndex + 2] * weight;
    componentSupport[root] += weight;
    componentBlockCount[root] += 1;
  }

  const componentNormals = new Float32Array(blockCount * 3);
  const componentAccepted = new Uint8Array(blockCount);
  for (let root = 0; root < blockCount; root += 1) {
    if (componentBlockCount[root] < 2 || componentSupport[root] <= 0) continue;
    const normalIndex = root * 3;
    const sumX = componentSums[normalIndex];
    const sumY = componentSums[normalIndex + 1];
    const sumZ = componentSums[normalIndex + 2];
    const magnitude = Math.hypot(sumX, sumY, sumZ);
    const coherence = magnitude / componentSupport[root];
    if (
      !Number.isFinite(magnitude)
      || magnitude <= 1e-10
      || coherence < minimumComponentCoherence
    ) continue;
    componentNormals[normalIndex] = sumX / magnitude;
    componentNormals[normalIndex + 1] = sumY / magnitude;
    componentNormals[normalIndex + 2] = sumZ / magnitude;
    componentAccepted[root] = 1;
  }

  // Surface normals are a display derivative, not the metric geometry source.
  // Signed normalized bytes preserve the RGB visualization while keeping both
  // Surface and Float32 Detail fields resident within the worker memory budget.
  const surfaceNormals = new Int8Array(detailNormals.length);
  for (let index = 0; index < detailNormals.length; index += 1) {
    const value = detailNormals[index];
    surfaceNormals[index] = Number.isFinite(value)
      ? Math.round(Math.max(-1, Math.min(1, value)) * 127)
      : 0;
  }
  for (let blockY = 0; blockY < blockRows; blockY += 1) {
    const yStart = blockY * blockSize;
    const yEnd = Math.min(height, yStart + blockSize);
    for (let blockX = 0; blockX < blockColumns; blockX += 1) {
      const blockIndex = blockY * blockColumns + blockX;
      if (planarBlock[blockIndex] === 0) continue;
      const root = findRoot(blockIndex);
      if (componentAccepted[root] === 0) continue;
      const componentNormalIndex = root * 3;
      const surfaceNx = componentNormals[componentNormalIndex];
      const surfaceNy = componentNormals[componentNormalIndex + 1];
      const surfaceNz = componentNormals[componentNormalIndex + 2];
      const blockNormalIndex = blockIndex * 3;
      const blockDot = (
        blockNormals[blockNormalIndex] * surfaceNx
        + blockNormals[blockNormalIndex + 1] * surfaceNy
        + blockNormals[blockNormalIndex + 2] * surfaceNz
      );
      if (blockDot < blockAssignmentDotThreshold) continue;
      const xStart = blockX * blockSize;
      const xEnd = Math.min(width, xStart + blockSize);
      for (let y = yStart; y < yEnd; y += 1) {
        for (let x = xStart; x < xEnd; x += 1) {
          const pixelIndex = y * width + x;
          if (
            mask[pixelIndex] === 0
            || !Number.isFinite(confidence[pixelIndex])
            || confidence[pixelIndex] < minimumConfidence
          ) continue;
          const normalIndex = pixelIndex * 3;
          const nx = detailNormals[normalIndex];
          const ny = detailNormals[normalIndex + 1];
          const nz = detailNormals[normalIndex + 2];
          const magnitude = Math.hypot(nx, ny, nz);
          if (!Number.isFinite(magnitude) || magnitude < 0.5) continue;
          const pixelDot = (
            (nx / magnitude) * surfaceNx
            + (ny / magnitude) * surfaceNy
            + (nz / magnitude) * surfaceNz
          );
          if (pixelDot < pixelAssignmentDotThreshold) continue;
          surfaceNormals[normalIndex] = Math.round(surfaceNx * 127);
          surfaceNormals[normalIndex + 1] = Math.round(surfaceNy * 127);
          surfaceNormals[normalIndex + 2] = Math.round(surfaceNz * 127);
        }
      }
    }
  }
  return surfaceNormals;
};
