/**
 * Close narrow transparent bands in the Surface-normal visualization.
 *
 * Edge-aware metric derivatives deliberately leave normals invalid when their
 * support crosses a depth discontinuity. Those invalid pixels are useful in
 * the Detail view, but become artificial dark outlines in the plane-regularized
 * Surface view. This display-only pass fills a transparent pixel only when
 * visible normals bracket it along a row, column, or diagonal. It never changes
 * depth, confidence, masks, or either stored normal field.
 *
 * The source snapshot makes this a single-pass gap closure: pixels filled by
 * the function cannot expand into unsupported or broadly missing regions.
 */
export const closeThinSurfaceNormalGapsInPlace = (
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  radius = 5,
): number => {
  if (
    !Number.isSafeInteger(width)
    || !Number.isSafeInteger(height)
    || width <= 0
    || height <= 0
    || rgba.length !== width * height * 4
  ) {
    throw new Error('surface_normal_display_shape_invalid');
  }
  const boundedRadius = Math.min(
    8,
    Math.max(0, Math.floor(Number(radius) || 0)),
  );
  if (boundedRadius === 0) return 0;

  const source = new Uint8ClampedArray(rgba);
  const axes = [
    [1, 0],
    [0, 1],
    [1, 1],
    [1, -1],
  ] as const;
  const findSupport = (x: number, y: number, dx: number, dy: number) => {
    for (let distance = 1; distance <= boundedRadius; distance += 1) {
      const candidateX = x + dx * distance;
      const candidateY = y + dy * distance;
      if (
        candidateX < 0
        || candidateX >= width
        || candidateY < 0
        || candidateY >= height
      ) break;
      const candidate = (candidateY * width + candidateX) * 4;
      if (source[candidate + 3] > 0) return { candidate, distance };
    }
    return null;
  };

  let filled = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const target = (y * width + x) * 4;
      if (source[target + 3] > 0) continue;
      let best: { candidate: number; span: number } | null = null;
      for (const [dx, dy] of axes) {
        const forward = findSupport(x, y, dx, dy);
        const backward = findSupport(x, y, -dx, -dy);
        if (!forward || !backward) continue;
        const nearest = forward.distance <= backward.distance ? forward : backward;
        const span = forward.distance + backward.distance;
        if (!best || span < best.span) {
          best = { candidate: nearest.candidate, span };
        }
      }
      if (!best) continue;
      rgba[target] = source[best.candidate];
      rgba[target + 1] = source[best.candidate + 1];
      rgba[target + 2] = source[best.candidate + 2];
      rgba[target + 3] = 255;
      filled += 1;
    }
  }
  return filled;
};
