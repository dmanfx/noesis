const finite = (value) => Number.isFinite(Number(value));

const metricBounds = (raw) => {
  if (!raw || typeof raw !== 'object') return null;
  const min_x = Number(raw.min_x);
  const max_x = Number(raw.max_x);
  const min_z = Number(raw.min_z);
  const max_z = Number(raw.max_z);
  if (![min_x, max_x, min_z, max_z].every(Number.isFinite)) return null;
  if (max_x <= min_x || max_z <= min_z) return null;
  return { min_x, max_x, min_z, max_z };
};

const containsBounds = (outer, inner) => (
  outer.min_x <= inner.min_x + 1e-6
  && outer.max_x >= inner.max_x - 1e-6
  && outer.min_z <= inner.min_z + 1e-6
  && outer.max_z >= inner.max_z - 1e-6
);

/**
 * Resolve the one metric viewport used by the BEV renderer.
 *
 * floorplanBounds is the semantic raster footprint.  advertisedBounds is the
 * producer's exact display envelope and is preferred when it safely contains
 * that raster (and any configured coverage envelope).  The dashboard must not
 * invent a second margin or fit the viewport to observed tracks.
 */
export const resolveBevDisplayBounds = ({
  floorplanBounds,
  advertisedBounds,
  coverageBounds,
  coverageToleranceM = 0,
} = {}) => {
  const floorplan = metricBounds(floorplanBounds);
  const coverage = metricBounds(coverageBounds);
  const tolerance = finite(coverageToleranceM) ? Math.max(0, Number(coverageToleranceM)) : 0;
  const expected = coverage
    ? {
      min_x: Math.min(floorplan?.min_x ?? Number.POSITIVE_INFINITY, coverage.min_x - tolerance),
      max_x: Math.max(floorplan?.max_x ?? Number.NEGATIVE_INFINITY, coverage.max_x + tolerance),
      min_z: Math.min(floorplan?.min_z ?? Number.POSITIVE_INFINITY, coverage.min_z - tolerance),
      max_z: Math.max(floorplan?.max_z ?? Number.NEGATIVE_INFINITY, coverage.max_z + tolerance),
    }
    : floorplan;
  const advertised = metricBounds(advertisedBounds);
  if (advertised && (!expected || containsBounds(advertised, expected))) return advertised;
  return expected || advertised;
};

const pointSegmentDistance = (x, z, ax, az, bx, bz) => {
  const dx = bx - ax;
  const dz = bz - az;
  const denom = (dx * dx) + (dz * dz);
  if (denom <= 1e-18) return Math.hypot(x - ax, z - az);
  const t = Math.min(1, Math.max(0, (((x - ax) * dx) + ((z - az) * dz)) / denom));
  return Math.hypot(x - (ax + (t * dx)), z - (az + (t * dz)));
};

const pointInCoverageRegion = (x, z, region, boundaryToleranceM) => {
  const polygon = Array.isArray(region?.polygonXZ) ? region.polygonXZ : [];
  if (polygon.length < 3) return false;
  const tolerance = Math.max(0, Number(boundaryToleranceM) || 0);
  let inside = false;
  let previous = polygon[polygon.length - 1];
  for (const current of polygon) {
    if (
      pointSegmentDistance(x, z, previous[0], previous[1], current[0], current[1])
      <= Math.max(1e-9, tolerance)
    ) return true;
    const crosses = (current[1] > z) !== (previous[1] > z);
    if (crosses) {
      const intersectionX = current[0]
        + (((z - current[1]) * (previous[0] - current[0])) / (previous[1] - current[1]));
      if (x < intersectionX) inside = !inside;
    }
    previous = current;
  }
  return inside;
};

export const coverageRegionForPoint = (coverage, x, z) => {
  if (!coverage || !Array.isArray(coverage.regions)) return null;
  for (const region of coverage.regions) {
    if (pointInCoverageRegion(x, z, region, coverage.boundaryToleranceM)) return region.id;
  }
  return null;
};

const resolveDisplayPoint = (displayBounds, x, y) => {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  if (
    displayBounds
    && (
      x < displayBounds.min_x
      || x > displayBounds.max_x
      || y < displayBounds.min_z
      || y > displayBounds.max_z
    )
  ) return null;
  return { x, y, mapped: false };
};

/**
 * Resolve a BEV point without moving it between metric coordinate spaces.
 * Canonical world points are already producer-authoritative: coverageInside
 * is diagnostic for them and must not become a second FE-only rejection gate.
 */
export const resolveBevMetricPoint = ({
  point,
  displayBounds,
  normalizedBounds,
  coverage,
} = {}) => {
  if (!point || typeof point !== 'object') return null;
  const metricX = Number(point.x);
  const metricY = Number(point.y);
  const canonicalWorld = point.canonicalWorld === true;
  if (coverage) {
    if (!Number.isFinite(metricX) || !Number.isFinite(metricY)) return null;
    if (!canonicalWorld && point.coverageInside === false) return null;
    if (!canonicalWorld && !coverageRegionForPoint(coverage, metricX, metricY)) return null;
    return resolveDisplayPoint(displayBounds, metricX, metricY);
  }
  if (Number.isFinite(metricX) && Number.isFinite(metricY)) {
    return resolveDisplayPoint(displayBounds, metricX, metricY);
  }
  const normX = Number(point.normX);
  const normY = Number(point.normY);
  if (!normalizedBounds || !Number.isFinite(normX) || !Number.isFinite(normY)) return null;
  if (point.floorplanInside === false || normX < 0 || normX > 1 || normY < 0 || normY > 1) return null;
  const spanX = normalizedBounds.max_x - normalizedBounds.min_x;
  const spanZ = normalizedBounds.max_z - normalizedBounds.min_z;
  if (!Number.isFinite(spanX) || !Number.isFinite(spanZ) || spanX <= 0 || spanZ <= 0) return null;
  const x = normalizedBounds.min_x + (normX * spanX);
  const y = normalizedBounds.max_z - (normY * spanZ);
  const resolved = resolveDisplayPoint(displayBounds, x, y);
  return resolved ? { ...resolved, mapped: true } : null;
};
