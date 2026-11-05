export type DisplayRect = { x: number; y: number; width: number; height: number };
export type RasterFootprint = { minX: number; maxX: number; minZ: number; maxZ: number };
export type CameraQuadrant = 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';

/** Exact scene-plane bounds of painted raster cells, using the same affine
 * corners as the map renderer. Computed once per cached image, never per frame. */
export function rasterFootprint(width: number, height: number, pixels: Uint8ClampedArray,
  corners: readonly (readonly number[])[]): RasterFootprint | null {
  const [tl, tr, bl] = [corners[3], corners[2], corners[0]];
  const dx = [(tr[0] - tl[0]) / width, (tr[2] - tl[2]) / width];
  const dy = [(bl[0] - tl[0]) / height, (bl[2] - tl[2]) / height];
  const low = [Math.min(0, dx[0]) + Math.min(0, dy[0]), Math.min(0, dx[1]) + Math.min(0, dy[1])];
  const high = [Math.max(0, dx[0]) + Math.max(0, dy[0]), Math.max(0, dx[1]) + Math.max(0, dy[1])];
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (let v = 0; v < height; v++) for (let u = 0; u < width; u++) {
    if (!pixels[(v * width + u) * 4 + 3]) continue;
    const x = tl[0] + u * dx[0] + v * dy[0], z = tl[2] + u * dx[1] + v * dy[1];
    minX = Math.min(minX, x + low[0]); maxX = Math.max(maxX, x + high[0]);
    minZ = Math.min(minZ, z + low[1]); maxZ = Math.max(maxZ, z + high[1]);
  }
  return Number.isFinite(minX) ? { minX, maxX, minZ, maxZ } : null;
}

/** Largest useful camera view in its current quadrant, excluding the home.
 * The returned card may letterbox; neither media nor the map is stretched. */
export function cameraQuadrantExpansion(stage: { width: number; height: number }, rest: DisplayRect,
  home: DisplayRect, chrome = 94): (DisplayRect & { quadrant: CameraQuadrant }) | null {
  const left = rest.x + rest.width / 2 < stage.width / 2;
  const top = rest.y + rest.height / 2 < stage.height / 2;
  const quadrant: CameraQuadrant = `${top ? 'top' : 'bottom'}-${left ? 'left' : 'right'}`;
  const x0 = left ? 12 : stage.width / 2 + 6, x1 = left ? stage.width / 2 - 6 : stage.width - 12;
  const y0 = top ? 12 : stage.height / 2 + 6, y1 = top ? stage.height / 2 - 6 : stage.height - 12;
  const box = (x: number, y: number, right: number, bottom: number) => ({ x, y, width: right - x, height: bottom - y });
  const h = { x: home.x - 12, y: home.y - 12, right: home.x + home.width + 12, bottom: home.y + home.height + 12 };
  const intersects = h.x < x1 && h.right > x0 && h.y < y1 && h.bottom > y0;
  const candidates = intersects ? [
    box(x0, y0, Math.min(x1, h.x), y1), box(Math.max(x0, h.right), y0, x1, y1),
    box(x0, y0, x1, Math.min(y1, h.y)), box(x0, Math.max(y0, h.bottom), x1, y1),
  ] : [box(x0, y0, x1, y1)];
  // Prefer actual 16:9 image size over a wide strip too short for the controls.
  const score = (r: DisplayRect) => Math.min(r.width, Math.max(0, r.height - chrome) * 16 / 9);
  const usable = candidates.filter(r => r.width >= 180 && r.height >= chrome + 60)
    .sort((a, b) => score(b) - score(a) || b.width * b.height - a.width * a.height);
  return usable.length ? { ...usable[0], quadrant } : null;
}
