export type CameraIntrinsics = [number, number, number, number];

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

/**
 * Derive perspective-correct camera-space normals from metric depth.
 *
 * Four neighboring pixels are unprojected with the calibrated camera
 * intrinsics. Their horizontal and vertical tangents are crossed and oriented
 * toward the camera. A two-pixel radius suppresses single-pixel depth noise
 * while retaining furniture and structural edges at full snapshot resolution.
 */
export const deriveCalibratedDepthNormals = (
  depth: Float32Array,
  mask: Uint8Array,
  height: number,
  width: number,
  intrinsics: CameraIntrinsics,
  sampleRadius = 2,
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
  const radius = Math.min(
    maximumRadius,
    Math.max(1, Math.floor(Number(sampleRadius) || 1)),
  );
  const normalizedIntrinsics = normalizeCameraIntrinsics(intrinsics);
  if (!normalizedIntrinsics) {
    throw new Error('depth_normals_intrinsics_invalid');
  }
  const [fx, fy, cx, cy] = normalizedIntrinsics;
  const normals = new Float32Array(pixelCount * 3);

  for (let y = radius; y < height - radius; y += 1) {
    for (let x = radius; x < width - radius; x += 1) {
      const centerIndex = y * width + x;
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

      const centerDepth = depth[centerIndex];
      const leftDepth = depth[leftIndex];
      const rightDepth = depth[rightIndex];
      const aboveDepth = depth[aboveIndex];
      const belowDepth = depth[belowIndex];
      if (
        ![centerDepth, leftDepth, rightDepth, aboveDepth, belowDepth]
          .every((value) => Number.isFinite(value) && value > 0.1 && value < 50)
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
  return normals;
};
