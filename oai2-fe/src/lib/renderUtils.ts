import type { FloorplanLayer } from '../components/DepthDrawer';
import {
  architecturalHeightColor,
  buildDetailedFloorplanOverlay,
} from './depthPanelRendering';

export type FloorplanRgbLayer = {
  rgb_b64?: string;
  rgb_shape?: [number, number, number];
  observed_b64?: string;
};

export const VIRIDIS = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 36],
];

export function viridisColor(t: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * (VIRIDIS.length - 1);
  const idx = Math.floor(scaled);
  const frac = scaled - idx;
  const a = VIRIDIS[idx];
  const b = VIRIDIS[Math.min(idx + 1, VIRIDIS.length - 1)];
  const r = Math.round(a[0] + (b[0] - a[0]) * frac);
  const g = Math.round(a[1] + (b[1] - a[1]) * frac);
  const bl = Math.round(a[2] + (b[2] - a[2]) * frac);
  return [r, g, bl];
}

export function grayscaleColor(t: number): [number, number, number] {
  const v = Math.round(255 * (1 - Math.min(1, Math.max(0, t))));
  return [v, v, v];
}

// Linear grayscale: 0 -> black, 1 -> white. Useful for binary layers like walkable masks.
export function bwColor(t: number): [number, number, number] {
  const v = Math.round(255 * Math.min(1, Math.max(0, t)));
  return [v, v, v];
}

export function infernoColor(t: number): [number, number, number] {
  const stops: Array<[number, [number, number, number]]> = [
    [0, [0, 0, 4]],
    [0.2, [35, 6, 59]],
    [0.4, [99, 23, 94]],
    [0.6, [159, 43, 73]],
    [0.8, [218, 83, 32]],
    [1, [252, 255, 164]],
  ];
  const clamped = Math.min(1, Math.max(0, t));
  for (let i = 0; i < stops.length - 1; i += 1) {
    const [posA, colorA] = stops[i];
    const [posB, colorB] = stops[i + 1];
    if (clamped >= posA && clamped <= posB) {
      const frac = (clamped - posA) / (posB - posA || 1);
      const r = Math.round(colorA[0] + (colorB[0] - colorA[0]) * frac);
      const g = Math.round(colorA[1] + (colorB[1] - colorA[1]) * frac);
      const b = Math.round(colorA[2] + (colorB[2] - colorA[2]) * frac);
      return [r, g, b];
    }
  }
  const last = stops[stops.length - 1][1];
  return [last[0], last[1], last[2]];
}

export function turboColor(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  const r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
  const g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
  const b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (115.67994485 + x * (-87.60200647 + x * 26.70740952))));
  const clamp = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255);
  return [clamp(r), clamp(g), clamp(b)];
}

export function decodeFloat32(base64?: string): Float32Array | null {
  if (!base64) return null;
  try {
    const binary = atob(base64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return new Float32Array(bytes.buffer);
  } catch (err) {
    console.error('Failed to decode float32 payload', err);
    return null;
  }
}

const halfToFloat = (value: number): number => {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) {
    if (fraction === 0) return sign * 0;
    return sign * Math.pow(2, -14) * (fraction / 1024);
  }
  if (exponent === 31) {
    return fraction === 0 ? sign * Infinity : NaN;
  }
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
};

export function decodeFloat16(base64?: string): Float32Array | null {
  if (!base64) return null;
  try {
    const binary = atob(base64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    const view = new Uint16Array(bytes.buffer);
    const out = new Float32Array(view.length);
    for (let i = 0; i < view.length; i += 1) {
      out[i] = halfToFloat(view[i]);
    }
    return out;
  } catch (err) {
    console.error('Failed to decode float16 payload', err);
    return null;
  }
}

export function decodeUint8(base64?: string): Uint8Array | null {
  if (!base64) return null;
  try {
    const binary = atob(base64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  } catch (err) {
    console.error('Failed to decode uint8 payload', err);
    return null;
  }
}

export const applyCanvasSize = (canvas: HTMLCanvasElement, widthOverride?: number, heightOverride?: number) => {
  const rect = canvas.getBoundingClientRect();
  const width = widthOverride ?? (rect.width || canvas.clientWidth || 1);
  const height = heightOverride ?? (rect.height || canvas.clientHeight || width);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(width * dpr));
  canvas.height = Math.max(1, Math.round(height * dpr));
  return { width, height, dpr };
};

type RenderLayerOptions = {
  forceAspect?: number;
  fit?: 'stretch' | 'contain';
  background?: string;
  contentPaddingPx?: number;
  // Optional overrides for value normalization. Useful for "contrast" views that clamp
  // to a meaningful physical range (e.g., 0..1.2m for floor vs countertop).
  valueMin?: number;
  valueMax?: number;
  valueMinPercentile?: number;
  valueMaxPercentile?: number;
  // Apply gamma to the normalized value after clamping to [0,1]. gamma < 1 boosts low values.
  gamma?: number;
  // Optional validity mask. Failed pixels use the explicit unknown checker
  // instead of a valid-looking semantic zero. maskInvert supports an `unknown`
  // layer whose positive cells are invalid.
  maskLayer?: FloorplanLayer;
  maskThreshold?: number;
  maskInvert?: boolean;
  imageSmoothing?: boolean;
  // Unknown cells use a checker pair so they cannot be mistaken for a valid
  // zero-valued semantic cell (for example, obstacle vs unobserved).
  unknownColor?: [number, number, number, number?];
  unknownAltColor?: [number, number, number, number?];
  inferredWalkableLayer?: FloorplanLayer;
  inferredWalkableColor?: [number, number, number, number?];
  inferredWalkableAltColor?: [number, number, number, number?];
  // Display-only repair for an isolated raster sampling hole. A masked cell
  // is rendered from the median of at least five valid 8-neighbours; larger
  // unknown regions and room edges remain explicitly unknown.
  repairIsolatedMaskHoles?: boolean;
  // Deterministic output sizing is used by source-quality PNG exports.
  targetWidthPx?: number;
  targetHeightPx?: number;
  pixelRatio?: number;
  // Display-only crop in source-grid coordinates. The renderer samples the
  // original grid through this rectangle; it never rewrites metric bounds or
  // semantic values.
  sourceRect?: { x: number; y: number; width: number; height: number };
  // Display-only metric reference lines for the detailed floorplan.
  bounds?: { min_x: number; max_x: number; min_z: number; max_z: number };
  metricGridM?: number;
  flipHorizontal?: boolean;
  flipVertical?: boolean;
};

type RenderLayerResult = {
  contentRectPx: { x: number; y: number; w: number; h: number };
  sourceRectGrid: { x: number; y: number; width: number; height: number };
  valueRange?: { min: number; max: number };
};

const writeRgba = (
  target: Uint8ClampedArray,
  offset: number,
  color: [number, number, number, number?],
) => {
  target[offset] = color[0];
  target[offset + 1] = color[1];
  target[offset + 2] = color[2];
  target[offset + 3] = color[3] ?? 255;
};

const unknownColorForCell = (
  options: RenderLayerOptions | undefined,
  row: number,
  col: number,
): [number, number, number, number?] => {
  const primary = options?.unknownColor ?? [16, 22, 32, 255];
  const alternate = options?.unknownAltColor ?? [26, 34, 47, 255];
  return ((Math.floor(row / 4) + Math.floor(col / 4)) % 2 === 0)
    ? primary
    : alternate;
};

const inferredWalkableColorForCell = (
  options: RenderLayerOptions | undefined,
  row: number,
  col: number,
): [number, number, number, number?] => {
  const primary = options?.inferredWalkableColor ?? [52, 102, 140, 255];
  const alternate = options?.inferredWalkableAltColor ?? [72, 126, 164, 255];
  return ((Math.floor(row / 4) + Math.floor(col / 4)) % 2 === 0)
    ? primary
    : alternate;
};

const normalizeSourceRect = (
  options: RenderLayerOptions | undefined,
  rows: number,
  cols: number,
) => {
  const requested = options?.sourceRect;
  if (!requested) return { x: 0, y: 0, width: cols, height: rows };
  const x = Math.max(0, Math.min(cols - 1, Math.floor(Number(requested.x) || 0)));
  const y = Math.max(0, Math.min(rows - 1, Math.floor(Number(requested.y) || 0)));
  const requestedWidth = Math.max(1, Math.floor(Number(requested.width) || cols));
  const requestedHeight = Math.max(1, Math.floor(Number(requested.height) || rows));
  return {
    x,
    y,
    width: Math.max(1, Math.min(cols - x, requestedWidth)),
    height: Math.max(1, Math.min(rows - y, requestedHeight)),
  };
};

function percentileSorted(sorted: number[], pct: number): number {
  if (!sorted.length) return 0;
  const p = Math.min(100, Math.max(0, pct));
  if (sorted.length === 1) return sorted[0];
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const frac = idx - lo;
  return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
}

export function renderCompositeWalkableObstacleToCanvas(
  canvas: HTMLCanvasElement | null,
  walkable: FloorplanLayer | undefined,
  obstacleHeight: FloorplanLayer | undefined,
  options?: number | RenderLayerOptions
): RenderLayerResult | null {
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  const clear = () => {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const hasWalkable = !!(walkable && walkable.grid_b64 && walkable.grid_shape);
  const hasObstacle = !!(obstacleHeight && obstacleHeight.grid_b64 && obstacleHeight.grid_shape);
  if (!hasWalkable || !hasObstacle) {
    clear();
    return null;
  }

  const [rowsW, colsW] = walkable!.grid_shape!;
  const [rowsO, colsO] = obstacleHeight!.grid_shape!;
  if (!rowsW || !colsW || rowsW !== rowsO || colsW !== colsO) {
    clear();
    return null;
  }

  const walkValues = decodeFloat32(walkable!.grid_b64!);
  const obsValues = decodeFloat32(obstacleHeight!.grid_b64!);
  if (!walkValues || !obsValues || walkValues.length < rowsW * colsW || obsValues.length < rowsW * colsW) {
    clear();
    return null;
  }

  const offscreen = document.createElement('canvas');
  offscreen.width = colsW;
  offscreen.height = rowsW;
  const offCtx = offscreen.getContext('2d');
  if (!offCtx) return null;

  const imageData = offCtx.createImageData(colsW, rowsW);
  const data = imageData.data;

  const obsMin = obstacleHeight!.value_min ?? 0;
  const obsMax = obstacleHeight!.value_max ?? 1;
  const obsDenom = obsMax - obsMin === 0 ? 1 : (obsMax - obsMin);
  const obstacleEps = 0.05;
  const floorColor: [number, number, number] = [230, 228, 222];
  const optObj = (typeof options === 'object') ? options : undefined;
  let maskValues: Float32Array | null = null;
  let inferredWalkableValues: Float32Array | null = null;
  let maskThreshold = 1e-6;
  let maskInvert = false;
  const maskLayer = optObj?.maskLayer;
  if (maskLayer?.grid_b64 && maskLayer.grid_shape) {
    const [maskRows, maskCols] = maskLayer.grid_shape;
    if (maskRows === rowsW && maskCols === colsW) {
      const decoded = decodeFloat32(maskLayer.grid_b64);
      if (decoded && decoded.length >= rowsW * colsW) {
        maskValues = decoded;
        if (typeof optObj?.maskThreshold === 'number' && Number.isFinite(optObj.maskThreshold)) {
          maskThreshold = optObj.maskThreshold;
        }
        maskInvert = !!optObj?.maskInvert;
      }
    }
  }
  const inferredWalkableLayer = optObj?.inferredWalkableLayer;
  if (inferredWalkableLayer?.grid_b64 && inferredWalkableLayer.grid_shape) {
    const [inferredRows, inferredCols] = inferredWalkableLayer.grid_shape;
    if (inferredRows === rowsW && inferredCols === colsW) {
      const decoded = decodeFloat32(inferredWalkableLayer.grid_b64);
      if (decoded && decoded.length >= rowsW * colsW) {
        inferredWalkableValues = decoded;
      }
    }
  }

  for (let idx = 0; idx < (rowsW * colsW); idx += 1) {
    const row = Math.floor(idx / colsW);
    const col = idx % colsW;
    const offset = idx * 4;
    if (maskValues) {
      const mv = maskValues[idx];
      const finiteMask = Number.isFinite(mv);
      const positive = finiteMask && mv > maskThreshold;
      const pass = finiteMask && (maskInvert ? !positive : positive);
      if (!pass) {
        const inferredWalkable = inferredWalkableValues
          && Number.isFinite(inferredWalkableValues[idx])
          && inferredWalkableValues[idx] > 0.5;
        writeRgba(
          data,
          offset,
          inferredWalkable
            ? inferredWalkableColorForCell(optObj, row, col)
            : unknownColorForCell(optObj, row, col),
        );
        continue;
      }
    }

    const w = walkValues[idx];
    const oh = obsValues[idx];
    let r = 0;
    let g = 0;
    let b = 0;

    if (Number.isFinite(oh) && oh > obstacleEps) {
      const t = Math.min(1, Math.max(0, (oh - obsMin) / obsDenom));
      const [rr, gg, bb] = infernoColor(t);
      r = rr; g = gg; b = bb;
    } else if (Number.isFinite(w) && w > 0.5) {
      r = floorColor[0];
      g = floorColor[1];
      b = floorColor[2];
    }

    data[offset] = r;
    data[offset + 1] = g;
    data[offset + 2] = b;
    data[offset + 3] = 255;
  }
  offCtx.putImageData(imageData, 0, 0);

  const fit = typeof options === 'number' ? 'stretch' : (options?.fit ?? 'stretch');
  const forceAspect = typeof options === 'number' ? options : options?.forceAspect;
  const background = (typeof options === 'object' && options?.background) ? options.background : '#000';
  const paddingCss = (typeof options === 'object' && options?.contentPaddingPx)
    ? Math.max(0, Number(options.contentPaddingPx) || 0)
    : 0;
  const imageSmoothing = (typeof options === 'object') ? !!options.imageSmoothing : false;
  const dpr = (typeof optObj?.pixelRatio === 'number' && Number.isFinite(optObj.pixelRatio))
    ? Math.max(0.1, optObj.pixelRatio)
    : (window.devicePixelRatio || 1);
  const sourceRect = normalizeSourceRect(optObj, rowsW, colsW);

  let { width, height } = applyCanvasSize(
    canvas,
    optObj?.targetWidthPx,
    optObj?.targetHeightPx,
  );
  if (fit === 'stretch') {
    const aspect = forceAspect || (sourceRect.width / sourceRect.height);
    height = width / aspect;
  }
  canvas.width = Math.max(1, Math.round(width * dpr));
  canvas.height = Math.max(1, Math.round(height * dpr));

  const contentAspect = forceAspect || (sourceRect.width / sourceRect.height);
  let contentW = width;
  let contentH = height;
  let contentX = 0;
  let contentY = 0;

  if (fit === 'contain') {
    const canvasAspect = width / height;
    if (canvasAspect > contentAspect) {
      contentH = height;
      contentW = height * contentAspect;
      contentX = (width - contentW) * 0.5;
    } else {
      contentW = width;
      contentH = width / contentAspect;
      contentY = (height - contentH) * 0.5;
    }
  }

  if (paddingCss > 0) {
    const shrink = paddingCss * 2;
    if (contentW > shrink && contentH > shrink) {
      contentX += paddingCss;
      contentY += paddingCss;
      contentW -= shrink;
      contentH -= shrink;
    }
  }

  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  if (fit === 'contain' && background) {
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, width, height);
  }
  ctx.imageSmoothingEnabled = imageSmoothing;
  if (imageSmoothing) {
    ctx.imageSmoothingQuality = 'high';
  }
  ctx.drawImage(
    offscreen,
    sourceRect.x,
    sourceRect.y,
    sourceRect.width,
    sourceRect.height,
    contentX,
    contentY,
    contentW,
    contentH,
  );
  ctx.restore();

  return {
    contentRectPx: {
      x: contentX * dpr,
      y: contentY * dpr,
      w: contentW * dpr,
      h: contentH * dpr
    },
    sourceRectGrid: sourceRect,
  };
}

export function renderLayerToCanvas(
  canvas: HTMLCanvasElement | null,
  layer: FloorplanLayer | undefined,
  palette: (t: number) => [number, number, number],
  options?: number | RenderLayerOptions
): RenderLayerResult | null {
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  const clear = () => {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  if (!layer || !layer.grid_b64 || !layer.grid_shape) {
    clear();
    return null;
  }
  const [rows, cols] = layer.grid_shape;
  if (!rows || !cols) {
    clear();
    return null;
  }
  const values = decodeFloat32(layer.grid_b64);
  if (!values || values.length < rows * cols) {
    clear();
    return null;
  }

  const offscreen = document.createElement('canvas');
  offscreen.width = cols;
  offscreen.height = rows;
  const offCtx = offscreen.getContext('2d');
  if (!offCtx) return null;

  const imageData = offCtx.createImageData(cols, rows);
  const data = imageData.data;
  const optObj = (typeof options === 'object') ? options : undefined;
  const gamma = (optObj && typeof optObj.gamma === 'number' && Number.isFinite(optObj.gamma) && optObj.gamma > 0)
    ? optObj.gamma
    : 1.0;

  let maskValues: Float32Array | null = null;
  let inferredWalkableValues: Float32Array | null = null;
  let maskThreshold = 1e-6;
  let maskInvert = false;
  const maskLayer = optObj?.maskLayer;
  if (maskLayer && maskLayer.grid_b64 && maskLayer.grid_shape) {
    const [maskRows, maskCols] = maskLayer.grid_shape;
    if (maskRows === rows && maskCols === cols) {
      const decoded = decodeFloat32(maskLayer.grid_b64);
      if (decoded && decoded.length >= rows * cols) {
        maskValues = decoded;
        if (typeof optObj?.maskThreshold === 'number' && Number.isFinite(optObj.maskThreshold)) {
          maskThreshold = optObj.maskThreshold;
        }
        maskInvert = !!optObj?.maskInvert;
      }
    }
  }
  const inferredWalkableLayer = optObj?.inferredWalkableLayer;
  if (inferredWalkableLayer?.grid_b64 && inferredWalkableLayer.grid_shape) {
    const [inferredRows, inferredCols] = inferredWalkableLayer.grid_shape;
    if (inferredRows === rows && inferredCols === cols) {
      const decoded = decodeFloat32(inferredWalkableLayer.grid_b64);
      if (decoded && decoded.length >= rows * cols) {
        inferredWalkableValues = decoded;
      }
    }
  }

  let min = (optObj && typeof optObj.valueMin === 'number') ? optObj.valueMin : (layer.value_min ?? 0);
  let max = (optObj && typeof optObj.valueMax === 'number') ? optObj.valueMax : (layer.value_max ?? 1);
  const minPct = optObj?.valueMinPercentile;
  const maxPct = optObj?.valueMaxPercentile;
  if (
    (typeof minPct === 'number' || typeof maxPct === 'number') &&
    Number.isFinite(Number(minPct ?? maxPct))
  ) {
    const samples: number[] = [];
    const n = rows * cols;
    for (let idx = 0; idx < n; idx += 1) {
      const v = values[idx];
      if (!Number.isFinite(v)) continue;
      if (maskValues) {
        const mv = maskValues[idx];
        const finiteMask = Number.isFinite(mv);
        const positive = finiteMask && mv > maskThreshold;
        const pass = finiteMask && (maskInvert ? !positive : positive);
        if (!pass) continue;
      }
      samples.push(v);
    }
    if (samples.length >= 16) {
      samples.sort((a, b) => a - b);
      if (typeof minPct === 'number' && Number.isFinite(minPct) && !(optObj && typeof optObj.valueMin === 'number')) {
        min = percentileSorted(samples, minPct);
      }
      if (typeof maxPct === 'number' && Number.isFinite(maxPct) && !(optObj && typeof optObj.valueMax === 'number')) {
        max = percentileSorted(samples, maxPct);
      }
    }
  }
  if (!Number.isFinite(min)) min = 0;
  if (!Number.isFinite(max)) max = min + 1;
  if (max <= min) max = min + 1;
  const denom = max - min;

  const maskPasses = (idx: number): boolean => {
    if (!maskValues || idx < 0 || idx >= rows * cols) return true;
    const mv = maskValues[idx];
    const finiteMask = Number.isFinite(mv);
    const positive = finiteMask && mv > maskThreshold;
    return finiteMask && (maskInvert ? !positive : positive);
  };
  const isolatedHoleValue = (idx: number): number | null => {
    if (!optObj?.repairIsolatedMaskHoles || !maskValues) return null;
    const row = Math.floor(idx / cols);
    const col = idx % cols;
    const neighbours: number[] = [];
    for (let rowOffset = -1; rowOffset <= 1; rowOffset += 1) {
      for (let colOffset = -1; colOffset <= 1; colOffset += 1) {
        if (rowOffset === 0 && colOffset === 0) continue;
        const neighbourRow = row + rowOffset;
        const neighbourCol = col + colOffset;
        if (
          neighbourRow < 0
          || neighbourRow >= rows
          || neighbourCol < 0
          || neighbourCol >= cols
        ) {
          continue;
        }
        const neighbourIdx = (neighbourRow * cols) + neighbourCol;
        const neighbourValue = values[neighbourIdx];
        if (maskPasses(neighbourIdx) && Number.isFinite(neighbourValue)) {
          neighbours.push(neighbourValue);
        }
      }
    }
    if (neighbours.length < 5) return null;
    neighbours.sort((a, b) => a - b);
    const middle = Math.floor(neighbours.length / 2);
    return neighbours.length % 2
      ? neighbours[middle]
      : (neighbours[middle - 1] + neighbours[middle]) / 2;
  };

  const n = rows * cols;
  for (let idx = 0; idx < n; idx += 1) {
    let v = values[idx];
    const offset = idx * 4;

    if (maskValues && !maskPasses(idx)) {
      const repairedValue = isolatedHoleValue(idx);
      if (repairedValue !== null) {
        v = repairedValue;
      } else {
        const inferredWalkable = inferredWalkableValues
          && Number.isFinite(inferredWalkableValues[idx])
          && inferredWalkableValues[idx] > 0.5;
        writeRgba(
          data,
          offset,
          inferredWalkable
            ? inferredWalkableColorForCell(optObj, Math.floor(idx / cols), idx % cols)
            : unknownColorForCell(optObj, Math.floor(idx / cols), idx % cols),
        );
        continue;
      }
    }

    if (!Number.isFinite(v)) {
      writeRgba(
        data,
        offset,
        unknownColorForCell(optObj, Math.floor(idx / cols), idx % cols),
      );
      continue;
    }

    let norm = (v - min) / denom;
    norm = Math.min(1, Math.max(0, norm));
    if (gamma !== 1.0) {
      norm = Math.pow(norm, gamma);
    }
    const [r, g, b] = palette(norm);
    data[offset] = r;
    data[offset + 1] = g;
    data[offset + 2] = b;
    data[offset + 3] = 255;
  }
  offCtx.putImageData(imageData, 0, 0);

  const fit = typeof options === 'number' ? 'stretch' : (options?.fit ?? 'stretch');
  const forceAspect = typeof options === 'number' ? options : options?.forceAspect;
  const background = (typeof options === 'object' && options?.background) ? options.background : '#000';
  const paddingCss = (typeof options === 'object' && options?.contentPaddingPx)
    ? Math.max(0, Number(options.contentPaddingPx) || 0)
    : 0;
  const imageSmoothing = (typeof options === 'object') ? !!options.imageSmoothing : false;
  const dpr = (typeof optObj?.pixelRatio === 'number' && Number.isFinite(optObj.pixelRatio))
    ? Math.max(0.1, optObj.pixelRatio)
    : (window.devicePixelRatio || 1);
  const sourceRect = normalizeSourceRect(optObj, rows, cols);

  let { width, height } = applyCanvasSize(
    canvas,
    optObj?.targetWidthPx,
    optObj?.targetHeightPx,
  );
  if (fit === 'stretch') {
    const aspect = forceAspect || (sourceRect.width / sourceRect.height);
    height = width / aspect;
  }
  canvas.width = Math.max(1, Math.round(width * dpr));
  canvas.height = Math.max(1, Math.round(height * dpr));

  const contentAspect = forceAspect || (sourceRect.width / sourceRect.height);
  let contentW = width;
  let contentH = height;
  let contentX = 0;
  let contentY = 0;

  if (fit === 'contain') {
    const canvasAspect = width / height;
    if (canvasAspect > contentAspect) {
      contentH = height;
      contentW = height * contentAspect;
      contentX = (width - contentW) * 0.5;
    } else {
      contentW = width;
      contentH = width / contentAspect;
      contentY = (height - contentH) * 0.5;
    }
  }

  if (paddingCss > 0) {
    const shrink = paddingCss * 2;
    if (contentW > shrink && contentH > shrink) {
      contentX += paddingCss;
      contentY += paddingCss;
      contentW -= shrink;
      contentH -= shrink;
    }
  }

  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  if (fit === 'contain' && background) {
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, width, height);
  }
  ctx.imageSmoothingEnabled = imageSmoothing;
  if (imageSmoothing) {
    ctx.imageSmoothingQuality = 'high';
  }
  ctx.drawImage(
    offscreen,
    sourceRect.x,
    sourceRect.y,
    sourceRect.width,
    sourceRect.height,
    contentX,
    contentY,
    contentW,
    contentH,
  );
  ctx.restore();

  return {
    contentRectPx: {
      x: contentX * dpr,
      y: contentY * dpr,
      w: contentW * dpr,
      h: contentH * dpr
    },
    sourceRectGrid: sourceRect,
    valueRange: { min, max },
  };
}

export function renderHeightfieldNormalsToCanvas(
  canvas: HTMLCanvasElement,
  heightLayer: FloorplanLayer | undefined,
  observedLayer: FloorplanLayer | undefined,
  {
    resolutionM,
    smoothingRadius = 0,
  }: {
    resolutionM: number;
    smoothingRadius?: number;
  },
): HTMLCanvasElement | null {
  if (!heightLayer?.grid_b64 || !heightLayer.grid_shape || !Number.isFinite(resolutionM) || resolutionM <= 0) {
    return null;
  }
  const [rows, columns] = heightLayer.grid_shape;
  const count = rows * columns;
  const heights = decodeFloat32(heightLayer.grid_b64);
  const observed = observedLayer?.grid_b64
    && observedLayer.grid_shape?.[0] === rows
    && observedLayer.grid_shape?.[1] === columns
    ? decodeFloat32(observedLayer.grid_b64)
    : null;
  if (!heights || heights.length < count) return null;

  const validAt = (index: number) => (
    Number.isFinite(heights[index])
    && (!observed || observed[index] > 0.5)
  );
  const filtered = new Float32Array(count);
  filtered.fill(Number.NaN);
  const radius = Math.max(0, Math.min(2, Math.floor(smoothingRadius)));
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const index = row * columns + column;
      if (!validAt(index)) continue;
      if (radius === 0) {
        filtered[index] = heights[index];
        continue;
      }
      let sum = 0;
      let samples = 0;
      for (let rowOffset = -radius; rowOffset <= radius; rowOffset += 1) {
        const sampleRow = row + rowOffset;
        if (sampleRow < 0 || sampleRow >= rows) continue;
        for (let columnOffset = -radius; columnOffset <= radius; columnOffset += 1) {
          const sampleColumn = column + columnOffset;
          if (sampleColumn < 0 || sampleColumn >= columns) continue;
          const sampleIndex = sampleRow * columns + sampleColumn;
          if (!validAt(sampleIndex)) continue;
          sum += heights[sampleIndex];
          samples += 1;
        }
      }
      if (samples >= 3) filtered[index] = sum / samples;
    }
  }

  const source = document.createElement('canvas');
  source.width = columns;
  source.height = rows;
  const sourceContext = source.getContext('2d');
  if (!sourceContext) return null;
  const image = sourceContext.createImageData(columns, rows);
  for (let row = 1; row < rows - 1; row += 1) {
    for (let column = 1; column < columns - 1; column += 1) {
      const index = row * columns + column;
      const left = filtered[index - 1];
      const right = filtered[index + 1];
      const forward = filtered[index - columns];
      const backward = filtered[index + columns];
      if (![filtered[index], left, right, forward, backward].every(Number.isFinite)) continue;
      const derivativeX = (right - left) / (2 * resolutionM);
      const derivativeZ = (forward - backward) / (2 * resolutionM);
      let normalX = -derivativeX;
      let normalY = 1;
      let normalZ = -derivativeZ;
      const norm = Math.hypot(normalX, normalY, normalZ);
      if (!Number.isFinite(norm) || norm <= 1e-9) continue;
      normalX /= norm;
      normalY /= norm;
      normalZ /= norm;
      const offset = index * 4;
      image.data[offset] = Math.round((normalX * 0.5 + 0.5) * 255);
      image.data[offset + 1] = Math.round((normalY * 0.5 + 0.5) * 255);
      image.data[offset + 2] = Math.round((normalZ * 0.5 + 0.5) * 255);
      image.data[offset + 3] = 255;
    }
  }
  sourceContext.putImageData(image, 0, 0);

  const context = canvas.getContext('2d');
  if (!context) return null;
  const rect = canvas.getBoundingClientRect();
  const width = rect.width || canvas.clientWidth || columns;
  const height = width / (columns / rows);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(width * dpr));
  canvas.height = Math.max(1, Math.round(height * dpr));
  context.save();
  context.scale(dpr, dpr);
  context.clearRect(0, 0, width, height);
  context.imageSmoothingEnabled = false;
  context.drawImage(source, 0, 0, width, height);
  context.restore();
  return source;
}

/**
 * Renders the observed highest surface as a continuous, physical AGL product.
 * Clean semantic layers remain supporting evidence: they add a fine obstacle
 * rim and inferred-space hatch but never replace furniture-height values.
 */
export function renderDetailedFloorplanToCanvas(
  canvas: HTMLCanvasElement | null,
  heightAgl: FloorplanLayer | undefined,
  obstacleHeight: FloorplanLayer | undefined,
  options?: RenderLayerOptions,
): RenderLayerResult | null {
  if (
    !canvas
    || !heightAgl?.grid_b64
    || !heightAgl.grid_shape
  ) {
    if (canvas) {
      const clearContext = canvas.getContext('2d');
      if (clearContext) {
        clearContext.setTransform(1, 0, 0, 1, 0, 0);
        clearContext.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
    return null;
  }
  const [rows, cols] = heightAgl.grid_shape;
  const heightValues = decodeFloat32(heightAgl.grid_b64);
  if (!rows || !cols || !heightValues || heightValues.length < rows * cols) {
    return null;
  }

  let observationMaskValues: Float32Array | null = null;
  if (
    options?.maskLayer?.grid_b64
    && options.maskLayer.grid_shape?.[0] === rows
    && options.maskLayer.grid_shape?.[1] === cols
  ) {
    observationMaskValues = decodeFloat32(options.maskLayer.grid_b64);
  }
  let obstacleValues: Float32Array | null = null;
  if (
    obstacleHeight?.grid_b64
    && obstacleHeight.grid_shape?.[0] === rows
    && obstacleHeight.grid_shape?.[1] === cols
  ) {
    obstacleValues = decodeFloat32(obstacleHeight.grid_b64);
  }

  const result = renderLayerToCanvas(
    canvas,
    heightAgl,
    (normalized) => architecturalHeightColor(normalized * 2.5),
    {
      ...options,
      valueMin: 0,
      valueMax: 2.5,
      gamma: 1,
      imageSmoothing: false,
    },
  );
  if (!result) return null;

  const context = canvas.getContext('2d');
  if (!context) return result;
  const overlay = buildDetailedFloorplanOverlay({
    heightAgl: heightValues,
    rows,
    cols,
    observationMask: observationMaskValues,
    observationMaskInvert: Boolean(options?.maskInvert),
    observationMaskThreshold: options?.maskThreshold,
    obstacleHeight: obstacleValues,
  });
  const sourceRect = result.sourceRectGrid;
  const content = result.contentRectPx;
  const cellWidthPx = content.w / sourceRect.width;
  const cellHeightPx = content.h / sourceRect.height;

  context.save();
  context.setTransform(1, 0, 0, 1, 0, 0);
  for (let row = sourceRect.y; row < sourceRect.y + sourceRect.height; row += 1) {
    for (let col = sourceRect.x; col < sourceRect.x + sourceRect.width; col += 1) {
      const index = (row * cols) + col;
      if (!overlay.observed[index]) continue;
      const x = content.x + ((col - sourceRect.x) * cellWidthPx);
      const y = content.y + ((row - sourceRect.y) * cellHeightPx);
      const width = cellWidthPx + 0.35;
      const height = cellHeightPx + 0.35;
      const shade = overlay.shade[index];
      if (shade < 0.995) {
        context.fillStyle = `rgba(0, 0, 0, ${Math.min(0.26, (1 - shade) * 0.9)})`;
        context.fillRect(x, y, width, height);
      } else if (shade > 1.005) {
        context.fillStyle = `rgba(255, 255, 255, ${Math.min(0.10, (shade - 1) * 0.55)})`;
        context.fillRect(x, y, width, height);
      }
      if (overlay.contour[index]) {
        context.fillStyle = 'rgba(5, 9, 15, 0.38)';
        context.fillRect(x, y, width, height);
      }
      if (overlay.frontier[index]) {
        context.fillStyle = 'rgba(10, 19, 25, 0.18)';
        context.fillRect(x, y, width, height);
      }
      if (overlay.obstacleRim[index]) {
        context.fillStyle = 'rgba(247, 176, 73, 0.22)';
        context.fillRect(x, y, width, height);
      }
    }
  }

  const bounds = options?.bounds;
  const metricStep = Number(options?.metricGridM ?? 1);
  if (
    bounds
    && Number.isFinite(metricStep)
    && metricStep > 0
    && bounds.max_x > bounds.min_x
    && bounds.max_z > bounds.min_z
  ) {
    const cellWidthM = (bounds.max_x - bounds.min_x) / cols;
    const cellDepthM = (bounds.max_z - bounds.min_z) / rows;
    const sourceMinX = bounds.min_x + (sourceRect.x * cellWidthM);
    const sourceMaxX = sourceMinX + (sourceRect.width * cellWidthM);
    const sourceMaxZ = bounds.max_z - (sourceRect.y * cellDepthM);
    const sourceMinZ = sourceMaxZ - (sourceRect.height * cellDepthM);
    context.beginPath();
    context.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    context.lineWidth = 1;
    for (
      let xM = Math.ceil((sourceMinX - 1e-9) / metricStep) * metricStep;
      xM <= sourceMaxX + 1e-9;
      xM += metricStep
    ) {
      const cellX = ((xM - bounds.min_x) / cellWidthM) - sourceRect.x;
      const pixelX = content.x + (cellX * cellWidthPx);
      context.moveTo(pixelX, content.y);
      context.lineTo(pixelX, content.y + content.h);
    }
    for (
      let zM = Math.ceil((sourceMinZ - 1e-9) / metricStep) * metricStep;
      zM <= sourceMaxZ + 1e-9;
      zM += metricStep
    ) {
      const cellY = ((bounds.max_z - zM) / cellDepthM) - sourceRect.y;
      const pixelY = content.y + (cellY * cellHeightPx);
      context.moveTo(content.x, pixelY);
      context.lineTo(content.x + content.w, pixelY);
    }
    context.stroke();
  }
  context.restore();

  return {
    ...result,
    valueRange: { min: 0, max: 2.5 },
  };
}

/**
 * Renders the highest observed horizontal surfaces as a metric orthophoto.
 *
 * MapAnything's mean-height raster is useful numerically but visually mixes
 * floors, furniture, walls, and projection bands. The registered surface RGB
 * retains the room and furniture detail people actually need from a floorplan,
 * so it is the dominant signal here. Structural height contributes only a
 * fixed-range tint and restrained relief. Inferred room footprint cells are
 * never filled; the footprint is shown only as a one-cell perimeter.
 */
export function renderTextureFloorplanToCanvas(
  canvas: HTMLCanvasElement | null,
  structuralHeight: FloorplanLayer | undefined,
  roomFootprint: FloorplanLayer | undefined,
  measuredPerimeter: FloorplanLayer | undefined,
  surfaceRgb: FloorplanRgbLayer | undefined,
  options?: RenderLayerOptions,
): RenderLayerResult | null {
  if (!canvas) return null;
  const clearCanvas = () => {
    const context = canvas.getContext('2d');
    if (!context) return;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, canvas.width, canvas.height);
  };
  if (
    !structuralHeight?.grid_b64
    || !structuralHeight.grid_shape
    || !roomFootprint?.grid_b64
    || !roomFootprint.grid_shape
    || !surfaceRgb?.rgb_b64
    || !surfaceRgb.rgb_shape
    || !surfaceRgb.observed_b64
  ) {
    clearCanvas();
    return null;
  }

  const [rows, cols] = structuralHeight.grid_shape;
  const count = rows * cols;
  if (
    rows <= 0
    || cols <= 0
    || roomFootprint.grid_shape[0] !== rows
    || roomFootprint.grid_shape[1] !== cols
    || surfaceRgb.rgb_shape[0] !== rows
    || surfaceRgb.rgb_shape[1] !== cols
    || surfaceRgb.rgb_shape[2] !== 3
  ) {
    clearCanvas();
    return null;
  }

  const heightValues = decodeFloat32(structuralHeight.grid_b64);
  const footprintValues = decodeFloat32(roomFootprint.grid_b64);
  const measuredPerimeterValues = (
    measuredPerimeter?.grid_b64
    && measuredPerimeter.grid_shape?.[0] === rows
    && measuredPerimeter.grid_shape?.[1] === cols
  )
    ? decodeFloat32(measuredPerimeter.grid_b64)
    : null;
  const rgbValues = decodeUint8(surfaceRgb.rgb_b64);
  const rgbObservedValues = decodeFloat32(surfaceRgb.observed_b64);
  if (
    !heightValues
    || heightValues.length < count
    || !footprintValues
    || footprintValues.length < count
    || !rgbValues
    || rgbValues.length < count * 3
    || !rgbObservedValues
    || rgbObservedValues.length < count
  ) {
    clearCanvas();
    return null;
  }

  const directObserved = new Uint8Array(count);
  const footprint = new Uint8Array(count);
  const luminanceSamples: number[] = [];
  for (let index = 0; index < count; index += 1) {
    const isDirect = Number.isFinite(rgbObservedValues[index])
      && rgbObservedValues[index] > 0.5;
    const isFootprint = Number.isFinite(footprintValues[index])
      && footprintValues[index] > 0.5;
    directObserved[index] = isDirect ? 1 : 0;
    footprint[index] = isFootprint ? 1 : 0;
    if (isDirect) {
      const rgbOffset = index * 3;
      luminanceSamples.push((
        (0.2126 * rgbValues[rgbOffset])
        + (0.7152 * rgbValues[rgbOffset + 1])
        + (0.0722 * rgbValues[rgbOffset + 2])
      ) / 255);
    }
  }
  if (luminanceSamples.length < 16) {
    clearCanvas();
    return null;
  }
  luminanceSamples.sort((a, b) => a - b);
  const luminanceLow = percentileSorted(luminanceSamples, 1);
  const luminanceHigh = percentileSorted(luminanceSamples, 99);
  const luminanceSpan = Math.max(luminanceHigh - luminanceLow, 0.05);

  const toneR = new Float32Array(count);
  const toneG = new Float32Array(count);
  const toneB = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    const rgbOffset = index * 3;
    toneR[index] = Math.min(1, Math.max(
      0,
      ((rgbValues[rgbOffset] / 255) - luminanceLow) / luminanceSpan,
    ));
    toneG[index] = Math.min(1, Math.max(
      0,
      ((rgbValues[rgbOffset + 1] / 255) - luminanceLow) / luminanceSpan,
    ));
    toneB[index] = Math.min(1, Math.max(
      0,
      ((rgbValues[rgbOffset + 2] / 255) - luminanceLow) / luminanceSpan,
    ));
  }

  // The visible repair radius is exactly three 0.04 m cells (0.12 m).
  // A six-cell nearest-source lookup also supplies the two-cell bilateral
  // neighbourhood around every supported output cell.
  const nearestSource = new Int32Array(count);
  const nearestDistanceSquared = new Float32Array(count);
  nearestSource.fill(-1);
  nearestDistanceSquared.fill(Number.POSITIVE_INFINITY);
  const nearestSearchRadius = 6;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const index = (row * cols) + col;
      if (directObserved[index]) {
        nearestSource[index] = index;
        nearestDistanceSquared[index] = 0;
        continue;
      }
      let bestIndex = -1;
      let bestDistanceSquared = Number.POSITIVE_INFINITY;
      for (let rowOffset = -nearestSearchRadius; rowOffset <= nearestSearchRadius; rowOffset += 1) {
        const sourceRow = row + rowOffset;
        if (sourceRow < 0 || sourceRow >= rows) continue;
        for (let colOffset = -nearestSearchRadius; colOffset <= nearestSearchRadius; colOffset += 1) {
          const distanceSquared = (rowOffset * rowOffset) + (colOffset * colOffset);
          if (distanceSquared > 36 || distanceSquared >= bestDistanceSquared) continue;
          const sourceCol = col + colOffset;
          if (sourceCol < 0 || sourceCol >= cols) continue;
          const sourceIndex = (sourceRow * cols) + sourceCol;
          if (!directObserved[sourceIndex]) continue;
          bestIndex = sourceIndex;
          bestDistanceSquared = distanceSquared;
        }
      }
      nearestSource[index] = bestIndex;
      nearestDistanceSquared[index] = bestDistanceSquared;
    }
  }

  const textureSupport = new Uint8Array(count);
  for (let index = 0; index < count; index += 1) {
    textureSupport[index] = (
      footprint[index]
      && nearestSource[index] >= 0
      && nearestDistanceSquared[index] <= 9
    ) ? 1 : 0;
  }

  const bilateralR = new Float32Array(count);
  const bilateralG = new Float32Array(count);
  const bilateralB = new Float32Array(count);
  const sigmaColor = 22 / 255;
  const sigmaSpace = 1.4;
  const colorDenominator = 2 * sigmaColor * sigmaColor;
  const spaceDenominator = 2 * sigmaSpace * sigmaSpace;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const index = (row * cols) + col;
      if (!textureSupport[index]) continue;
      const centerSource = nearestSource[index];
      const centerR = Math.round(toneR[centerSource] * 255) / 255;
      const centerG = Math.round(toneG[centerSource] * 255) / 255;
      const centerB = Math.round(toneB[centerSource] * 255) / 255;
      let weightSum = 0;
      let sumR = 0;
      let sumG = 0;
      let sumB = 0;
      for (let rowOffset = -2; rowOffset <= 2; rowOffset += 1) {
        const sampleRow = row + rowOffset;
        if (sampleRow < 0 || sampleRow >= rows) continue;
        for (let colOffset = -2; colOffset <= 2; colOffset += 1) {
          const sampleCol = col + colOffset;
          if (sampleCol < 0 || sampleCol >= cols) continue;
          const sampleIndex = (sampleRow * cols) + sampleCol;
          const sampleSource = nearestSource[sampleIndex];
          if (sampleSource < 0) continue;
          const sampleR = Math.round(toneR[sampleSource] * 255) / 255;
          const sampleG = Math.round(toneG[sampleSource] * 255) / 255;
          const sampleB = Math.round(toneB[sampleSource] * 255) / 255;
          const diffR = sampleR - centerR;
          const diffG = sampleG - centerG;
          const diffB = sampleB - centerB;
          const colorDistanceSquared = (
            (diffR * diffR) + (diffG * diffG) + (diffB * diffB)
          );
          const spaceDistanceSquared = (
            (rowOffset * rowOffset) + (colOffset * colOffset)
          );
          const weight = Math.exp(
            -(colorDistanceSquared / colorDenominator)
            - (spaceDistanceSquared / spaceDenominator),
          );
          weightSum += weight;
          sumR += sampleR * weight;
          sumG += sampleG * weight;
          sumB += sampleB * weight;
        }
      }
      const inverseWeight = weightSum > 1e-12 ? 1 / weightSum : 0;
      bilateralR[index] = weightSum > 1e-12 ? sumR * inverseWeight : centerR;
      bilateralG[index] = weightSum > 1e-12 ? sumG * inverseWeight : centerG;
      bilateralB[index] = weightSum > 1e-12 ? sumB * inverseWeight : centerB;
    }
  }

  // Approximate scipy's sigma=1 Gaussian with its default four-sigma support.
  const gaussianKernel = new Float32Array(9);
  let gaussianTotal = 0;
  for (let offset = -4; offset <= 4; offset += 1) {
    const value = Math.exp(-(offset * offset) / 2);
    gaussianKernel[offset + 4] = value;
    gaussianTotal += value;
  }
  for (let index = 0; index < gaussianKernel.length; index += 1) {
    gaussianKernel[index] /= gaussianTotal;
  }
  const heightHorizontal = new Float32Array(count);
  const heightSmooth = new Float32Array(count);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      let value = 0;
      for (let offset = -4; offset <= 4; offset += 1) {
        const sampleCol = Math.min(cols - 1, Math.max(0, col + offset));
        const sampleHeight = heightValues[(row * cols) + sampleCol];
        value += (Number.isFinite(sampleHeight) ? sampleHeight : 0)
          * gaussianKernel[offset + 4];
      }
      heightHorizontal[(row * cols) + col] = value;
    }
  }
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      let value = 0;
      for (let offset = -4; offset <= 4; offset += 1) {
        const sampleRow = Math.min(rows - 1, Math.max(0, row + offset));
        value += heightHorizontal[(sampleRow * cols) + col]
          * gaussianKernel[offset + 4];
      }
      heightSmooth[(row * cols) + col] = value;
    }
  }

  const offscreen = document.createElement('canvas');
  offscreen.width = cols;
  offscreen.height = rows;
  const offscreenContext = offscreen.getContext('2d');
  const context = canvas.getContext('2d');
  if (!offscreenContext || !context) {
    clearCanvas();
    return null;
  }
  const imageData = offscreenContext.createImageData(cols, rows);
  const image = imageData.data;
  const heightAt = (row: number, col: number): number => (
    heightSmooth[
      (Math.min(rows - 1, Math.max(0, row)) * cols)
      + Math.min(cols - 1, Math.max(0, col))
    ]
  );
  const writeCompositePixel = (
    index: number,
    red: number,
    green: number,
    blue: number,
  ) => {
    const offset = index * 4;
    image[offset] = Math.round(Math.min(255, Math.max(0, red)));
    image[offset + 1] = Math.round(Math.min(255, Math.max(0, green)));
    image[offset + 2] = Math.round(Math.min(255, Math.max(0, blue)));
    image[offset + 3] = 255;
  };

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const index = (row * cols) + col;
      if (!textureSupport[index]) {
        writeCompositePixel(index, 2, 4, 7);
        continue;
      }
      const sourceIndex = nearestSource[index];
      const textureR = directObserved[index]
        ? (0.78 * toneR[sourceIndex]) + (0.22 * bilateralR[index])
        : bilateralR[index];
      const textureG = directObserved[index]
        ? (0.78 * toneG[sourceIndex]) + (0.22 * bilateralG[index])
        : bilateralG[index];
      const textureB = directObserved[index]
        ? (0.78 * toneB[sourceIndex]) + (0.22 * bilateralB[index])
        : bilateralB[index];
      const heightM = Number.isFinite(heightValues[index]) ? heightValues[index] : 0;
      const [heightR, heightG, heightB] = viridisColor(
        Math.min(1, Math.max(0, heightM / 1.65)),
      );
      const gradientX = (
        heightAt(row - 1, col + 1)
        + (2 * heightAt(row, col + 1))
        + heightAt(row + 1, col + 1)
        - heightAt(row - 1, col - 1)
        - (2 * heightAt(row, col - 1))
        - heightAt(row + 1, col - 1)
      ) / 8;
      const gradientY = (
        heightAt(row + 1, col - 1)
        + (2 * heightAt(row + 1, col))
        + heightAt(row + 1, col + 1)
        - heightAt(row - 1, col - 1)
        - (2 * heightAt(row - 1, col))
        - heightAt(row - 1, col + 1)
      ) / 8;
      const relief = Math.min(1.10, Math.max(
        0.85,
        0.98 + (0.12 * ((-0.55 * gradientX) - (0.83 * gradientY))),
      ));
      writeCompositePixel(
        index,
        (((0.90 * textureR) + (0.10 * (heightR / 255))) * relief) * 255,
        (((0.90 * textureG) + (0.10 * (heightG / 255))) * relief) * 255,
        (((0.90 * textureB) + (0.10 * (heightB / 255))) * relief) * 255,
      );
    }
  }

  // The producer's measured one-cell perimeter remains a soft edge overlay,
  // never a replacement surface or justification for filling unsupported
  // space. Its confidence changes only the blend strength.
  if (
    measuredPerimeterValues
    && measuredPerimeterValues.length >= count
  ) {
    for (let index = 0; index < count; index += 1) {
      if (
        !textureSupport[index]
        || !Number.isFinite(measuredPerimeterValues[index])
        || measuredPerimeterValues[index] <= 0.18
      ) {
        continue;
      }
      const measuredStrength = Math.min(
        1,
        Math.max(
          0,
          (measuredPerimeterValues[index] - 0.18) / 0.82,
        ),
      );
      const measuredMix = 0.35 + (0.30 * measuredStrength);
      const offset = index * 4;
      image[offset] = Math.round(
        ((1 - measuredMix) * image[offset]) + (measuredMix * 205),
      );
      image[offset + 1] = Math.round(
        ((1 - measuredMix) * image[offset + 1]) + (measuredMix * 211),
      );
      image[offset + 2] = Math.round(
        ((1 - measuredMix) * image[offset + 2]) + (measuredMix * 210),
      );
    }
  }

  // Show inferred extent only as a one-cell perimeter (3x3 erosion frontier).
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const index = (row * cols) + col;
      if (!footprint[index]) continue;
      let perimeter = false;
      for (let rowOffset = -1; rowOffset <= 1 && !perimeter; rowOffset += 1) {
        for (let colOffset = -1; colOffset <= 1; colOffset += 1) {
          const neighbourRow = row + rowOffset;
          const neighbourCol = col + colOffset;
          if (
            neighbourRow < 0
            || neighbourRow >= rows
            || neighbourCol < 0
            || neighbourCol >= cols
            || !footprint[(neighbourRow * cols) + neighbourCol]
          ) {
            perimeter = true;
            break;
          }
        }
      }
      if (perimeter) {
        const offset = index * 4;
        image[offset] = Math.max(image[offset], 70);
        image[offset + 1] = Math.max(image[offset + 1], 82);
        image[offset + 2] = Math.max(image[offset + 2], 94);
      }
    }
  }
  offscreenContext.putImageData(imageData, 0, 0);

  const sourceRect = normalizeSourceRect(options, rows, cols);
  const fit = options?.fit ?? 'stretch';
  const forceAspect = options?.forceAspect;
  const background = options?.background ?? '#000';
  const paddingCss = options?.contentPaddingPx
    ? Math.max(0, Number(options.contentPaddingPx) || 0)
    : 0;
  const imageSmoothing = options?.imageSmoothing ?? false;
  const dpr = (
    typeof options?.pixelRatio === 'number'
    && Number.isFinite(options.pixelRatio)
  )
    ? Math.max(0.1, options.pixelRatio)
    : (window.devicePixelRatio || 1);
  let { width, height } = applyCanvasSize(
    canvas,
    options?.targetWidthPx,
    options?.targetHeightPx,
  );
  if (fit === 'stretch') {
    const aspect = forceAspect || (sourceRect.width / sourceRect.height);
    height = width / aspect;
  }
  canvas.width = Math.max(1, Math.round(width * dpr));
  canvas.height = Math.max(1, Math.round(height * dpr));

  const contentAspect = forceAspect || (sourceRect.width / sourceRect.height);
  let contentWidth = width;
  let contentHeight = height;
  let contentX = 0;
  let contentY = 0;
  if (fit === 'contain') {
    const canvasAspect = width / height;
    if (canvasAspect > contentAspect) {
      contentHeight = height;
      contentWidth = height * contentAspect;
      contentX = (width - contentWidth) * 0.5;
    } else {
      contentWidth = width;
      contentHeight = width / contentAspect;
      contentY = (height - contentHeight) * 0.5;
    }
  }
  if (paddingCss > 0) {
    const shrink = paddingCss * 2;
    if (contentWidth > shrink && contentHeight > shrink) {
      contentX += paddingCss;
      contentY += paddingCss;
      contentWidth -= shrink;
      contentHeight -= shrink;
    }
  }

  context.save();
  context.scale(dpr, dpr);
  context.clearRect(0, 0, width, height);
  if (fit === 'contain' && background) {
    context.fillStyle = background;
    context.fillRect(0, 0, width, height);
  }
  context.imageSmoothingEnabled = imageSmoothing;
  if (imageSmoothing) context.imageSmoothingQuality = 'high';
  if (options?.flipHorizontal || options?.flipVertical) {
    context.translate(
      options.flipHorizontal ? width : 0,
      options.flipVertical ? height : 0,
    );
    context.scale(
      options.flipHorizontal ? -1 : 1,
      options.flipVertical ? -1 : 1,
    );
  }
  context.drawImage(
    offscreen,
    sourceRect.x,
    sourceRect.y,
    sourceRect.width,
    sourceRect.height,
    contentX,
    contentY,
    contentWidth,
    contentHeight,
  );
  context.restore();

  return {
    contentRectPx: {
      x: contentX * dpr,
      y: contentY * dpr,
      w: contentWidth * dpr,
      h: contentHeight * dpr,
    },
    sourceRectGrid: sourceRect,
    valueRange: { min: 0, max: 1.65 },
  };
}

export function renderStructuralFloorplanToCanvas(
  canvas: HTMLCanvasElement | null,
  structuralHeight: FloorplanLayer | undefined,
  roomFootprint: FloorplanLayer | undefined,
  surfaceObserved: FloorplanLayer | undefined,
  wallSupport: FloorplanLayer | undefined,
  roomBoundary: FloorplanLayer | undefined,
  surfaceRgb: FloorplanRgbLayer | undefined,
  options?: RenderLayerOptions,
): RenderLayerResult | null {
  if (
    !canvas
    || !structuralHeight?.grid_b64
    || !structuralHeight.grid_shape
    || !roomFootprint?.grid_b64
    || !roomFootprint.grid_shape
  ) {
    if (canvas) {
      const context = canvas.getContext('2d');
      context?.clearRect(0, 0, canvas.width, canvas.height);
    }
    return null;
  }
  const [rows, cols] = structuralHeight.grid_shape;
  if (
    rows <= 0
    || cols <= 0
    || roomFootprint.grid_shape[0] !== rows
    || roomFootprint.grid_shape[1] !== cols
  ) {
    return null;
  }
  const count = rows * cols;
  const heightValues = decodeFloat32(structuralHeight.grid_b64);
  const footprintValues = decodeFloat32(roomFootprint.grid_b64);
  if (
    !heightValues
    || heightValues.length < count
    || !footprintValues
    || footprintValues.length < count
  ) {
    return null;
  }

  const decodeMatchingLayer = (
    layer: FloorplanLayer | undefined,
  ): Float32Array | null => {
    if (
      !layer?.grid_b64
      || layer.grid_shape?.[0] !== rows
      || layer.grid_shape?.[1] !== cols
    ) {
      return null;
    }
    const decoded = decodeFloat32(layer.grid_b64);
    return decoded && decoded.length >= count ? decoded : null;
  };
  const surfaceObservedValues = decodeMatchingLayer(surfaceObserved);
  const wallValues = decodeMatchingLayer(wallSupport);
  const boundaryValues = decodeMatchingLayer(roomBoundary);
  const rgbValues = (
    surfaceRgb?.rgb_b64
    && surfaceRgb.rgb_shape?.[0] === rows
    && surfaceRgb.rgb_shape?.[1] === cols
    && surfaceRgb.rgb_shape?.[2] === 3
  )
    ? decodeUint8(surfaceRgb.rgb_b64)
    : null;
  const rgbObservedValues = (
    surfaceRgb?.observed_b64
    ? decodeFloat32(surfaceRgb.observed_b64)
    : null
  );

  const result = renderLayerToCanvas(
    canvas,
    structuralHeight,
    (normalized) => architecturalHeightColor(normalized * 1.65),
    {
      ...options,
      valueMin: 0,
      valueMax: 1.65,
      gamma: 1,
      maskLayer: roomFootprint,
      maskThreshold: 0.5,
      maskInvert: false,
      imageSmoothing: false,
    },
  );
  if (!result) return null;

  const context = canvas.getContext('2d');
  if (!context) return result;
  const sourceRect = result.sourceRectGrid;
  const content = result.contentRectPx;
  const cellWidthPx = content.w / sourceRect.width;
  const cellHeightPx = content.h / sourceRect.height;
  const indexAt = (row: number, col: number): number => (row * cols) + col;

  context.save();
  context.setTransform(1, 0, 0, 1, 0, 0);
  for (let row = sourceRect.y; row < sourceRect.y + sourceRect.height; row += 1) {
    for (let col = sourceRect.x; col < sourceRect.x + sourceRect.width; col += 1) {
      const index = indexAt(row, col);
      if (
        !Number.isFinite(footprintValues[index])
        || footprintValues[index] <= 0.5
      ) {
        continue;
      }
      const x = content.x + ((col - sourceRect.x) * cellWidthPx);
      const y = content.y + ((row - sourceRect.y) * cellHeightPx);
      const width = cellWidthPx + 0.35;
      const height = cellHeightPx + 0.35;
      const heightM = Number.isFinite(heightValues[index])
        ? heightValues[index]
        : 0;
      const observedSurface = !surfaceObservedValues
        || (
          Number.isFinite(surfaceObservedValues[index])
          && surfaceObservedValues[index] > 0.5
        );

      if (!observedSurface) {
        const alternate = (
          Math.floor(row / 3) + Math.floor(col / 3)
        ) % 2 === 0;
        context.fillStyle = alternate
          ? 'rgba(39, 111, 119, 0.055)'
          : 'rgba(25, 77, 88, 0.025)';
        context.fillRect(x, y, width, height);
      }

      const rgbObserved = (
        rgbValues
        && rgbValues.length >= count * 3
        && rgbObservedValues
        && rgbObservedValues.length >= count
        && Number.isFinite(rgbObservedValues[index])
        && rgbObservedValues[index] > 0.5
      );
      if (rgbObserved && heightM >= 0.12) {
        const rgbOffset = index * 3;
        context.fillStyle = `rgba(${rgbValues![rgbOffset]}, ${rgbValues![rgbOffset + 1]}, ${rgbValues![rgbOffset + 2]}, 0.22)`;
        context.fillRect(x, y, width, height);
      }

      if (heightM >= 0.12) {
        let edge = false;
        for (const [rowOffset, colOffset] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
          const neighborRow = row + rowOffset;
          const neighborCol = col + colOffset;
          if (
            neighborRow < 0
            || neighborRow >= rows
            || neighborCol < 0
            || neighborCol >= cols
          ) {
            edge = true;
            break;
          }
          const neighborIndex = indexAt(neighborRow, neighborCol);
          const neighborHeight = (
            Number.isFinite(heightValues[neighborIndex])
            && footprintValues[neighborIndex] > 0.5
          )
            ? heightValues[neighborIndex]
            : 0;
          if (
            neighborHeight < 0.12
            || Math.abs(neighborHeight - heightM) > 0.12
          ) {
            edge = true;
            break;
          }
        }
        if (edge) {
          context.fillStyle = 'rgba(7, 11, 18, 0.24)';
          context.fillRect(x, y, width, height);
        }
      }

      const wall = wallValues && Number.isFinite(wallValues[index])
        ? Math.min(1, Math.max(0, wallValues[index]))
        : 0;
      if (wall > 0.04) {
        context.fillStyle = `rgba(8, 13, 18, ${0.14 + (wall * 0.48)})`;
        context.fillRect(x, y, width, height);
      } else {
        const boundary = boundaryValues && Number.isFinite(boundaryValues[index])
          ? Math.min(1, Math.max(0, boundaryValues[index]))
          : 0;
        if (boundary > 0.20) {
          context.fillStyle = `rgba(22, 32, 38, ${boundary * 0.42})`;
          context.fillRect(x, y, width, height);
        }
      }
    }
  }

  const bounds = options?.bounds;
  const metricStep = Number(options?.metricGridM ?? 1);
  if (
    bounds
    && Number.isFinite(metricStep)
    && metricStep > 0
    && bounds.max_x > bounds.min_x
    && bounds.max_z > bounds.min_z
  ) {
    const cellWidthM = (bounds.max_x - bounds.min_x) / cols;
    const cellDepthM = (bounds.max_z - bounds.min_z) / rows;
    const sourceMinX = bounds.min_x + (sourceRect.x * cellWidthM);
    const sourceMaxX = sourceMinX + (sourceRect.width * cellWidthM);
    const sourceMaxZ = bounds.max_z - (sourceRect.y * cellDepthM);
    const sourceMinZ = sourceMaxZ - (sourceRect.height * cellDepthM);
    context.beginPath();
    context.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    context.lineWidth = 1;
    for (
      let xM = Math.ceil((sourceMinX - 1e-9) / metricStep) * metricStep;
      xM <= sourceMaxX + 1e-9;
      xM += metricStep
    ) {
      const gridX = ((xM - bounds.min_x) / cellWidthM) - sourceRect.x;
      const pixelX = content.x + (gridX * cellWidthPx);
      context.moveTo(pixelX, content.y);
      context.lineTo(pixelX, content.y + content.h);
    }
    for (
      let zM = Math.ceil((sourceMinZ - 1e-9) / metricStep) * metricStep;
      zM <= sourceMaxZ + 1e-9;
      zM += metricStep
    ) {
      const gridY = ((bounds.max_z - zM) / cellDepthM) - sourceRect.y;
      const pixelY = content.y + (gridY * cellHeightPx);
      context.moveTo(content.x, pixelY);
      context.lineTo(content.x + content.w, pixelY);
    }
    context.stroke();
  }
  context.restore();

  return {
    ...result,
    valueRange: { min: 0, max: 1.65 },
  };
}
