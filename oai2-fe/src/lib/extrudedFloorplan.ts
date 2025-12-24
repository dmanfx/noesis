import { decodeFloat32, infernoColor } from './renderUtils';

export type GridLayer = {
  grid_b64?: string;
  grid_shape?: [number, number];
  value_min?: number;
  value_max?: number;
};

export type FloorplanBounds = {
  min_x?: number;
  max_x?: number;
  min_z?: number;
  max_z?: number;
};

export type FloorplanLike = {
  bounds?: FloorplanBounds;
  height?: GridLayer;
  density?: GridLayer;
};

export type ObstacleBox = {
  id: string;
  minRow: number;
  maxRow: number;
  minCol: number;
  maxCol: number;
  areaCells: number;
  areaM2: number;
  widthM: number;
  depthM: number;
  heightM: number;
  centerXM: number;
  centerZM: number;
};

export type ExtrudedFloorplanModel = {
  rows: number;
  cols: number;
  heights: Float32Array;
  densities: Float32Array | null;
  bounds: FloorplanBounds;
  dx: number;
  dz: number;
  maxHeightM: number;
  boxes: ObstacleBox[];
};

export type ObstacleBoxSettings = {
  maxCells: number;
  minHeightM: number;
  minDensity: number;
  minFootprintM2: number;
  maxBoxes: number;
};

export const DEFAULT_OBSTACLE_SETTINGS: ObstacleBoxSettings = {
  maxCells: 140,
  minHeightM: 0.25,
  minDensity: 0.06,
  minFootprintM2: 0.35,
  maxBoxes: 12,
};

export type ExtrudedFloorplanRenderOptions = {
  forceAspect?: number;
  heightExaggeration?: number;
  minHeightM?: number;
  minDensity?: number;
  showBoxes?: boolean;
  selectedBoxId?: string | null;
  palette?: (t: number) => [number, number, number];
  background?: string;
};

function decodeGrid(layer?: GridLayer): { rows: number; cols: number; values: Float32Array } | null {
  if (!layer?.grid_b64 || !layer.grid_shape) return null;
  const [rows, cols] = layer.grid_shape;
  if (!rows || !cols) return null;
  const values = decodeFloat32(layer.grid_b64);
  if (!values || values.length < rows * cols) return null;
  return { rows, cols, values };
}

function finiteMax(values: ArrayLike<number>, count: number): number {
  let best = 0;
  for (let i = 0; i < count; i += 1) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (v > best) best = v;
  }
  return best;
}

function downsampleMax(
  src: ArrayLike<number>,
  srcRows: number,
  srcCols: number,
  dstRows: number,
  dstCols: number,
): Float32Array {
  const out = new Float32Array(dstRows * dstCols);
  out.fill(Number.NaN);
  const total = Math.min(srcRows * srcCols, src.length);
  for (let r = 0; r < dstRows; r += 1) {
    const y0 = Math.floor((r * srcRows) / dstRows);
    const y1 = Math.max(y0 + 1, Math.floor(((r + 1) * srcRows) / dstRows));
    for (let c = 0; c < dstCols; c += 1) {
      const x0 = Math.floor((c * srcCols) / dstCols);
      const x1 = Math.max(x0 + 1, Math.floor(((c + 1) * srcCols) / dstCols));
      let best = Number.NEGATIVE_INFINITY;
      let saw = false;
      for (let sy = y0; sy < Math.min(srcRows, y1); sy += 1) {
        for (let sx = x0; sx < Math.min(srcCols, x1); sx += 1) {
          const idx = sy * srcCols + sx;
          if (idx >= total) continue;
          const v = src[idx];
          if (!Number.isFinite(v)) continue;
          saw = true;
          if (v > best) best = v;
        }
      }
      out[r * dstCols + c] = saw ? best : Number.NaN;
    }
  }
  return out;
}

function downsampleMean(
  src: ArrayLike<number>,
  srcRows: number,
  srcCols: number,
  dstRows: number,
  dstCols: number,
): Float32Array {
  const out = new Float32Array(dstRows * dstCols);
  out.fill(Number.NaN);
  const total = Math.min(srcRows * srcCols, src.length);
  for (let r = 0; r < dstRows; r += 1) {
    const y0 = Math.floor((r * srcRows) / dstRows);
    const y1 = Math.max(y0 + 1, Math.floor(((r + 1) * srcRows) / dstRows));
    for (let c = 0; c < dstCols; c += 1) {
      const x0 = Math.floor((c * srcCols) / dstCols);
      const x1 = Math.max(x0 + 1, Math.floor(((c + 1) * srcCols) / dstCols));
      let sum = 0;
      let count = 0;
      for (let sy = y0; sy < Math.min(srcRows, y1); sy += 1) {
        for (let sx = x0; sx < Math.min(srcCols, x1); sx += 1) {
          const idx = sy * srcCols + sx;
          if (idx >= total) continue;
          const v = src[idx];
          if (!Number.isFinite(v)) continue;
          sum += v;
          count += 1;
        }
      }
      out[r * dstCols + c] = count ? sum / count : Number.NaN;
    }
  }
  return out;
}

function normalizeBounds(bounds?: FloorplanBounds): FloorplanBounds {
  if (!bounds) return {};
  const minX = typeof bounds.min_x === 'number' ? bounds.min_x : undefined;
  const maxX = typeof bounds.max_x === 'number' ? bounds.max_x : undefined;
  const minZ = typeof bounds.min_z === 'number' ? bounds.min_z : undefined;
  const maxZ = typeof bounds.max_z === 'number' ? bounds.max_z : undefined;
  return { min_x: minX, max_x: maxX, min_z: minZ, max_z: maxZ };
}

function computeCellSize(bounds: FloorplanBounds, rows: number, cols: number): { dx: number; dz: number } {
  const minX = bounds.min_x ?? 0;
  const maxX = bounds.max_x ?? 0;
  const minZ = bounds.min_z ?? 0;
  const maxZ = bounds.max_z ?? 0;
  const dx = cols > 0 ? Math.abs(maxX - minX) / cols : 0;
  const dz = rows > 0 ? Math.abs(maxZ - minZ) / rows : 0;
  const fallback = 0.15;
  return {
    dx: Number.isFinite(dx) && dx > 0 ? dx : fallback,
    dz: Number.isFinite(dz) && dz > 0 ? dz : fallback,
  };
}

function shade([r, g, b]: [number, number, number], factor: number): [number, number, number] {
  const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v)));
  return [clamp(r * factor), clamp(g * factor), clamp(b * factor)];
}

function extractObstacleBoxes(
  heights: ArrayLike<number>,
  densities: ArrayLike<number> | null,
  rows: number,
  cols: number,
  options: {
    dx: number;
    dz: number;
    bounds: FloorplanBounds;
    minHeightM: number;
    minDensity: number;
    minFootprintM2: number;
    maxBoxes: number;
  },
): ObstacleBox[] {
  const { dx, dz, bounds, minHeightM, minDensity, minFootprintM2, maxBoxes } = options;
  const n = rows * cols;
  const occ = new Uint8Array(n);
  const heightThresh = Math.max(0, minHeightM);
  const densityThresh = Math.max(0, Math.min(1, minDensity));
  for (let i = 0; i < n; i += 1) {
    const h = heights[i];
    if (!Number.isFinite(h) || h < heightThresh) continue;
    if (densities) {
      const d = densities[i];
      if (!Number.isFinite(d) || d < densityThresh) continue;
    }
    occ[i] = 1;
  }

  const visited = new Uint8Array(n);
  const queue = new Int32Array(n);
  const boxes: ObstacleBox[] = [];

  const cellAreaM2 = Math.max(0.000001, dx * dz);
  const minCellsFromArea = Math.max(1, Math.ceil(minFootprintM2 / cellAreaM2));

  for (let start = 0; start < n; start += 1) {
    if (!occ[start] || visited[start]) continue;
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    visited[start] = 1;

    let minRow = Math.floor(start / cols);
    let maxRow = minRow;
    let minCol = start % cols;
    let maxCol = minCol;
    let maxHeight = Number.NEGATIVE_INFINITY;
    let areaCells = 0;

    while (head < tail) {
      const idx = queue[head++];
      const r = Math.floor(idx / cols);
      const c = idx % cols;
      const h = heights[idx];
      if (Number.isFinite(h) && h > maxHeight) maxHeight = h;
      areaCells += 1;
      if (r < minRow) minRow = r;
      if (r > maxRow) maxRow = r;
      if (c < minCol) minCol = c;
      if (c > maxCol) maxCol = c;

      const up = r > 0 ? idx - cols : -1;
      const down = r + 1 < rows ? idx + cols : -1;
      const left = c > 0 ? idx - 1 : -1;
      const right = c + 1 < cols ? idx + 1 : -1;
      if (up >= 0 && occ[up] && !visited[up]) { visited[up] = 1; queue[tail++] = up; }
      if (down >= 0 && occ[down] && !visited[down]) { visited[down] = 1; queue[tail++] = down; }
      if (left >= 0 && occ[left] && !visited[left]) { visited[left] = 1; queue[tail++] = left; }
      if (right >= 0 && occ[right] && !visited[right]) { visited[right] = 1; queue[tail++] = right; }
    }

    if (areaCells < minCellsFromArea) continue;
    if (!Number.isFinite(maxHeight) || maxHeight < heightThresh) continue;

    const widthCells = (maxCol - minCol + 1);
    const depthCells = (maxRow - minRow + 1);
    const widthM = widthCells * dx;
    const depthM = depthCells * dz;
    const areaM2 = areaCells * cellAreaM2;

    const centerCol = (minCol + maxCol + 1) / 2;
    const centerRow = (minRow + maxRow + 1) / 2;

    const minX = bounds.min_x ?? 0;
    const maxZ = bounds.max_z ?? 0;
    const centerXM = minX + dx * centerCol;
    const centerZM = maxZ - dz * centerRow;

    boxes.push({
      id: `${minRow}:${minCol}:${maxRow}:${maxCol}:${Math.round(maxHeight * 100)}`,
      minRow,
      maxRow,
      minCol,
      maxCol,
      areaCells,
      areaM2,
      widthM,
      depthM,
      heightM: maxHeight,
      centerXM,
      centerZM,
    });
  }

  boxes.sort((a, b) => b.areaM2 - a.areaM2);
  return boxes.slice(0, Math.max(0, maxBoxes));
}

export function buildExtrudedFloorplanModel(
  floorplan: FloorplanLike | null | undefined,
  settings: Partial<ObstacleBoxSettings> = {},
): ExtrudedFloorplanModel | null {
  const cfg: ObstacleBoxSettings = { ...DEFAULT_OBSTACLE_SETTINGS, ...settings };
  if (!floorplan) return null;
  const decodedHeight = decodeGrid(floorplan.height);
  if (!decodedHeight) return null;
  const decodedDensity = decodeGrid(floorplan.density);
  const bounds = normalizeBounds(floorplan.bounds);

  let rows = decodedHeight.rows;
  let cols = decodedHeight.cols;
  const maxCells = Math.max(16, Math.min(256, Math.floor(cfg.maxCells)));
  let heights = decodedHeight.values;
  let densities: Float32Array | null = decodedDensity ? decodedDensity.values : null;

  if (rows > maxCells || cols > maxCells) {
    const scale = Math.max(rows / maxCells, cols / maxCells);
    const dstRows = Math.max(1, Math.round(rows / scale));
    const dstCols = Math.max(1, Math.round(cols / scale));
    heights = downsampleMax(heights, rows, cols, dstRows, dstCols);
    if (densities) {
      densities = downsampleMean(densities, rows, cols, dstRows, dstCols);
    }
    rows = dstRows;
    cols = dstCols;
  }

  const { dx, dz } = computeCellSize(bounds, rows, cols);
  const maxHeightM = finiteMax(heights, rows * cols);

  const boxes = extractObstacleBoxes(
    heights,
    densities,
    rows,
    cols,
    {
      dx,
      dz,
      bounds,
      minHeightM: cfg.minHeightM,
      minDensity: cfg.minDensity,
      minFootprintM2: cfg.minFootprintM2,
      maxBoxes: cfg.maxBoxes,
    }
  );

  return {
    rows,
    cols,
    heights,
    densities,
    bounds,
    dx,
    dz,
    maxHeightM,
    boxes,
  };
}

export function renderExtrudedFloorplanToCanvas(
  canvas: HTMLCanvasElement | null,
  model: ExtrudedFloorplanModel | null,
  opts: ExtrudedFloorplanRenderOptions = {},
): void {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const clear = () => {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  if (!model || model.rows <= 0 || model.cols <= 0) {
    clear();
    return;
  }

  const palette = opts.palette ?? infernoColor;
  const forceAspect = opts.forceAspect ?? (16 / 9);
  const rect = canvas.getBoundingClientRect();
  const width = rect.width || canvas.clientWidth || 1;
  const height = width / forceAspect;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(width * dpr));
  canvas.height = Math.max(1, Math.round(height * dpr));

  const background = opts.background ?? '#0d1320';
  const heightExaggeration = Math.max(0.2, Number(opts.heightExaggeration ?? 1.6));
  const minHeightM = Math.max(0, Number(opts.minHeightM ?? 0.02));
  const minDensity = Math.max(0, Math.min(1, Number(opts.minDensity ?? 0)));
  const showBoxes = opts.showBoxes ?? true;
  const selectedBoxId = opts.selectedBoxId ?? null;

  const rows = model.rows;
  const cols = model.cols;
  const heights = model.heights;
  const densities = model.densities;
  const maxHeightM = Math.max(0.001, Number.isFinite(model.maxHeightM) ? model.maxHeightM : 1.0);

  const margin = 14;
  const widthAvail = Math.max(1, width - margin * 2);
  const heightAvail = Math.max(1, height - margin * 2);

  const gridResM = Math.max(0.001, (model.dx + model.dz) * 0.5);
  const heightCells = (maxHeightM * heightExaggeration) / gridResM;
  const denomX = Math.max(2, rows + cols);
  const denomY = Math.max(2, (rows + cols) * 0.25 + heightCells);

  const tileWFromX = (2 * widthAvail) / denomX;
  const tileWFromY = heightAvail / denomY;
  const tileW = Math.max(2, Math.min(26, Math.min(tileWFromX, tileWFromY)));
  const tileH = tileW * 0.5;
  const heightScalePx = (tileW / gridResM) * heightExaggeration;

  const iso = (r: number, c: number) => ({
    x: (c - r) * (tileW / 2),
    y: (c + r) * (tileH / 2),
  });

  // Intersections range from (0,0) to (rows, cols)
  const xMin = iso(rows, 0).x;
  const xMax = iso(0, cols).x;
  const yMax = iso(rows, cols).y;
  const yMin = -maxHeightM * heightScalePx;
  const xRange = xMax - xMin;
  const yRange = yMax - yMin;

  const originX = margin + (widthAvail - xRange) / 2 - xMin;
  const originY = margin + (heightAvail - yRange) / 2 - yMin;

  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, width, height);
  ctx.translate(originX, originY);

  // Subtle base plane outline.
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  const p00 = iso(0, 0);
  const p0c = iso(0, cols);
  const pr0 = iso(rows, 0);
  const prc = iso(rows, cols);
  ctx.moveTo(p00.x, p00.y);
  ctx.lineTo(p0c.x, p0c.y);
  ctx.lineTo(prc.x, prc.y);
  ctx.lineTo(pr0.x, pr0.y);
  ctx.closePath();
  ctx.stroke();

  // Draw blocks back-to-front by diagonal (r+c).
  for (let s = 0; s <= rows + cols - 2; s += 1) {
    for (let r = 0; r < rows; r += 1) {
      const c = s - r;
      if (c < 0 || c >= cols) continue;
      const idx = r * cols + c;
      const hM = heights[idx];
      if (!Number.isFinite(hM) || hM < minHeightM) continue;
      if (densities && minDensity > 0) {
        const d = densities[idx];
        if (!Number.isFinite(d) || d < minDensity) continue;
      }
      const hPx = hM * heightScalePx;
      if (!(hPx > 0)) continue;

      const norm = Math.min(1, Math.max(0, hM / maxHeightM));
      const topColor = palette(norm);
      const leftColor = shade(topColor, 0.72);
      const rightColor = shade(topColor, 0.6);

      const north = iso(r, c);
      const east = iso(r, c + 1);
      const south = iso(r + 1, c + 1);
      const west = iso(r + 1, c);

      const northTop = { x: north.x, y: north.y - hPx };
      const eastTop = { x: east.x, y: east.y - hPx };
      const southTop = { x: south.x, y: south.y - hPx };
      const westTop = { x: west.x, y: west.y - hPx };

      // Right face.
      ctx.beginPath();
      ctx.moveTo(eastTop.x, eastTop.y);
      ctx.lineTo(southTop.x, southTop.y);
      ctx.lineTo(south.x, south.y);
      ctx.lineTo(east.x, east.y);
      ctx.closePath();
      ctx.fillStyle = `rgb(${rightColor[0]},${rightColor[1]},${rightColor[2]})`;
      ctx.fill();

      // Left face.
      ctx.beginPath();
      ctx.moveTo(westTop.x, westTop.y);
      ctx.lineTo(southTop.x, southTop.y);
      ctx.lineTo(south.x, south.y);
      ctx.lineTo(west.x, west.y);
      ctx.closePath();
      ctx.fillStyle = `rgb(${leftColor[0]},${leftColor[1]},${leftColor[2]})`;
      ctx.fill();

      // Top face.
      ctx.beginPath();
      ctx.moveTo(northTop.x, northTop.y);
      ctx.lineTo(eastTop.x, eastTop.y);
      ctx.lineTo(southTop.x, southTop.y);
      ctx.lineTo(westTop.x, westTop.y);
      ctx.closePath();
      ctx.fillStyle = `rgb(${topColor[0]},${topColor[1]},${topColor[2]})`;
      ctx.fill();
    }
  }

  if (showBoxes && model.boxes.length) {
    for (const box of model.boxes) {
      const hPx = Math.max(0, box.heightM) * heightScalePx;
      const a = iso(box.minRow, box.minCol);
      const b = iso(box.minRow, box.maxCol + 1);
      const c = iso(box.maxRow + 1, box.maxCol + 1);
      const d = iso(box.maxRow + 1, box.minCol);

      const at = { x: a.x, y: a.y - hPx };
      const bt = { x: b.x, y: b.y - hPx };
      const ct = { x: c.x, y: c.y - hPx };
      const dt = { x: d.x, y: d.y - hPx };

      const isSelected = selectedBoxId && box.id === selectedBoxId;
      ctx.lineWidth = isSelected ? 2 : 1;
      ctx.strokeStyle = isSelected ? 'rgba(88,214,141,0.95)' : 'rgba(88,214,141,0.55)';

      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.lineTo(c.x, c.y);
      ctx.lineTo(d.x, d.y);
      ctx.closePath();
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(at.x, at.y);
      ctx.lineTo(bt.x, bt.y);
      ctx.lineTo(ct.x, ct.y);
      ctx.lineTo(dt.x, dt.y);
      ctx.closePath();
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(at.x, at.y);
      ctx.moveTo(b.x, b.y);
      ctx.lineTo(bt.x, bt.y);
      ctx.moveTo(c.x, c.y);
      ctx.lineTo(ct.x, ct.y);
      ctx.moveTo(d.x, d.y);
      ctx.lineTo(dt.x, dt.y);
      ctx.stroke();
    }
  }

  ctx.restore();
}
