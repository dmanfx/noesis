import { derivePlaneAwareSurfaceNormals } from './depthNormals';

/** Display-only surface support; never written back to PCF or live world state. */
export function buildPcfCameraSurface(width: number, height: number, intrinsics: number[],
  samples: Float32Array, sampleU: Float32Array, sampleV: Float32Array, colors: Uint8ClampedArray) {
  const [fx, fy, cx, cy] = intrinsics, count = width * height;
  // One front-surface representative per small image cell. Work and triangle
  // sizes are bounded by display resolution, independently of point count.
  const step = 6, columns = Math.ceil(width / step), rows = Math.ceil(height / step);
  const vertices = new Int32Array(columns * rows); vertices.fill(-1);
  for (let y = 0; y < rows; y++) for (let x = 0; x < columns; x++) {
    const x0 = x * step, y0 = y * step, x1 = Math.min(width, x0 + step), y1 = Math.min(height, y0 + step);
    let nearest = Infinity;
    for (let v = y0; v < y1; v++) for (let u = x0; u < x1; u++) nearest = Math.min(nearest, samples[v * width + u]);
    if (!Number.isFinite(nearest)) continue;
    let distance = Infinity;
    for (let v = y0; v < y1; v++) for (let u = x0; u < x1; u++) {
      const i = v * width + u;
      if (samples[i] > nearest + .025 + .01 * nearest) continue;
      const d = (sampleU[i] - (x0 + x1 - 1) / 2) ** 2 + (sampleV[i] - (y0 + y1 - 1) / 2) ** 2;
      if (d < distance) { distance = d; vertices[y * columns + x] = i; }
    }
  }
  // Estimate a local plane from a neighbourhood of measured vertices, rather
  // than differentiating a tiny triangle and amplifying PCF sampling noise.
  // Inverse optical depth is affine on a plane. Two bounded robust passes
  // reject depth steps; planar averaging below further joins coherent walls.
  const vertexNormals = new Float32Array(vertices.length * 3);
  for (let y = 0; y < rows; y++) for (let x = 0; x < columns; x++) {
    const cell = y * columns + x, centre = vertices[cell]; if (centre < 0) continue;
    const u0 = sampleU[centre], v0 = sampleV[centre], z0 = samples[centre];
    let a = 0, b = 0, d = 1 / z0, accepted = false;
    for (let pass = 0; pass < 2; pass++) {
      let n = 0, sx = 0, sy = 0, sq = 0, sxx = 0, syy = 0, sxy = 0, sxq = 0, syq = 0;
      for (let yy = Math.max(0, y - 3); yy <= Math.min(rows - 1, y + 3); yy++) for (let xx = Math.max(0, x - 3); xx <= Math.min(columns - 1, x + 3); xx++) {
        const i = vertices[yy * columns + xx]; if (i < 0) continue;
        const z = samples[i], u = sampleU[i] - u0, v = sampleV[i] - v0, q = 1 / z;
        if (Math.abs(z - z0) > .12 + .05 * z0 || (pass && Math.abs(1 / (a * u + b * v + d) - z) > .025 + .01 * z)) continue;
        n++; sx += u; sy += v; sq += q; sxx += u * u; syy += v * v; sxy += u * v; sxq += u * q; syq += v * q;
      }
      if (n < 6) { accepted = false; break; }
      const xx = sxx - sx * sx / n, yy = syy - sy * sy / n, xy = sxy - sx * sy / n;
      const xq = sxq - sx * sq / n, yq = syq - sy * sq / n, determinant = xx * yy - xy * xy;
      if (determinant < 1e-6) { accepted = false; break; }
      a = (xq * yy - yq * xy) / determinant; b = (yq * xx - xq * xy) / determinant; d = (sq - a * sx - b * sy) / n; accepted = true;
    }
    if (accepted) {
      const normal = [-a * fx, -b * fy, -(d + a * (cx - u0) + b * (cy - v0))], length = Math.hypot(...normal);
      if (length > 1e-10) for (let axis = 0; axis < 3; axis++) vertexNormals[cell * 3 + axis] = normal[axis] / length;
    }
  }
  const normalAt = new Int32Array(count);
  for (let cell = 0; cell < vertices.length; cell++) if (vertices[cell] >= 0) normalAt[vertices[cell]] = cell * 3;
  const depth = new Float32Array(count); depth.fill(Infinity);
  const rgb = new Uint8ClampedArray(count * 4), detail = new Float32Array(count * 3), mask = new Uint8Array(count);
  let triangleCount = 0;
  const triangle = (a: number, b: number, c: number) => {
    if (a < 0 || b < 0 || c < 0) return;
    const indices = [a, b, c], us = indices.map(i => sampleU[i]), vs = indices.map(i => sampleV[i]), zs = indices.map(i => samples[i]);
    const near = Math.min(...zs), far = Math.max(...zs);
    if (far - near > .08 + .05 * near) return;
    const points = indices.map((_, i) => [(us[i] - cx) / fx * zs[i], (vs[i] - cy) / fy * zs[i], zs[i]]);
    for (let i = 0; i < 3; i++) {
      const j = (i + 1) % 3;
      if (Math.hypot(us[i] - us[j], vs[i] - vs[j]) > step * 2.85
        || Math.hypot(...points[i].map((p, axis) => p - points[j][axis])) > .08 + .08 * near) return;
    }
    const area = (us[1] - us[0]) * (vs[2] - vs[0]) - (vs[1] - vs[0]) * (us[2] - us[0]);
    if (area < .05) return;
    const ab = points[1].map((v, axis) => v - points[0][axis]), ac = points[2].map((v, axis) => v - points[0][axis]);
    const normal = [ac[1] * ab[2] - ac[2] * ab[1], ac[2] * ab[0] - ac[0] * ab[2], ac[0] * ab[1] - ac[1] * ab[0]];
    const length = Math.hypot(...normal); if (length < 1e-10) return;
    const sign = normal.reduce((s, v, axis) => s + v * points[0][axis], 0) > 0 ? -1 : 1;
    for (let axis = 0; axis < 3; axis++) normal[axis] *= sign / length;
    triangleCount++;
    const x0 = Math.max(0, Math.ceil(Math.min(...us))), x1 = Math.min(width - 1, Math.floor(Math.max(...us)));
    const y0 = Math.max(0, Math.ceil(Math.min(...vs))), y1 = Math.min(height - 1, Math.floor(Math.max(...vs)));
    for (let v = y0; v <= y1; v++) for (let u = x0; u <= x1; u++) {
      const w1 = ((u - us[0]) * (vs[2] - vs[0]) - (v - vs[0]) * (us[2] - us[0])) / area;
      const w2 = ((us[1] - us[0]) * (v - vs[0]) - (vs[1] - vs[0]) * (u - us[0])) / area, w0 = 1 - w1 - w2;
      if (Math.min(w0, w1, w2) < -1e-6) continue;
      const z = 1 / (w0 / zs[0] + w1 / zs[1] + w2 / zs[2]), index = v * width + u;
      // A nearer measured point blocks a surface behind it, even if that point
      // did not have enough neighbours to form its own triangle.
      if (z >= depth[index] || z > samples[index] + .025 + .01 * samples[index]) continue;
      depth[index] = z; mask[index] = 1;
      const weights = [w0 / zs[0], w1 / zs[1], w2 / zs[2]];
      let nx = 0, ny = 0, nz = 0;
      for (let n = 0; n < 3; n++) {
        const ni = normalAt[indices[n]];
        const hasFit = Math.hypot(vertexNormals[ni], vertexNormals[ni + 1], vertexNormals[ni + 2]) > .5;
        nx += weights[n] * (hasFit ? vertexNormals[ni] : normal[0]);
        ny += weights[n] * (hasFit ? vertexNormals[ni + 1] : normal[1]);
        nz += weights[n] * (hasFit ? vertexNormals[ni + 2] : normal[2]);
      }
      const norm = Math.hypot(nx, ny, nz);
      detail.set(norm > 1e-10 ? [nx / norm, ny / norm, nz / norm] : normal, index * 3);
      for (let axis = 0; axis < 3; axis++) rgb[index * 4 + axis] = z * (w0 * colors[a * 4 + axis] / zs[0] + w1 * colors[b * 4 + axis] / zs[1] + w2 * colors[c * 4 + axis] / zs[2]);
      rgb[index * 4 + 3] = 255;
    }
  };
  for (let y = 0; y < rows - 1; y++) for (let x = 0; x < columns - 1; x++) {
    const i = y * columns + x, a = vertices[i], b = vertices[i + 1], c = vertices[i + columns], d = vertices[i + columns + 1];
    // Choose the diagonal that changes depth least; a missing corner still
    // permits the triangle supported by the other three cells.
    if (a < 0 || d < 0 || (b >= 0 && c >= 0 && Math.abs(samples[b] - samples[c]) < Math.abs(samples[a] - samples[d]))) {
      triangle(a, b, c); triangle(b, d, c);
    } else { triangle(a, b, d); triangle(a, d, c); }
  }
  // Uniform display weights express geometric support, not model confidence.
  const weights = new Float32Array(count); weights.fill(1);
  const planar = derivePlaneAwareSurfaceNormals(detail, depth, weights, mask, height, width);
  const normals = new Uint8ClampedArray(count * 4);
  let min = Infinity, max = -Infinity, validCount = 0;
  for (let i = 0; i < count; i++) {
    if (!mask[i]) { depth[i] = NaN; continue; }
    min = Math.min(min, depth[i]); max = Math.max(max, depth[i]); validCount++;
    for (let axis = 0; axis < 3; axis++) normals[i * 4 + axis] = Math.round((planar[i * 3 + axis] / 127 * .5 + .5) * 255);
    normals[i * 4 + 3] = 255;
  }
  return { width, height, depth, rgb, normals, range: { min, max }, validCount, normalCount: validCount, triangleCount };
}
