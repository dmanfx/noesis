import type { FloorplanLayer } from '../components/DepthDrawer';

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

const STRUCTURAL_HEIGHT_STOPS: ReadonlyArray<{
  heightM: number;
  color: readonly [number, number, number];
}> = [
  { heightM: 0.00, color: [238, 239, 234] },
  { heightM: 0.12, color: [184, 216, 207] },
  { heightM: 0.30, color: [76, 170, 181] },
  { heightM: 0.50, color: [35, 125, 169] },
  { heightM: 0.75, color: [68, 83, 157] },
  { heightM: 1.00, color: [122, 60, 142] },
  { heightM: 1.30, color: [187, 61, 104] },
  { heightM: 1.65, color: [242, 142, 56] },
];

const structuralHeightColor = (
  heightMValue: number,
): [number, number, number] => {
  const heightM = Math.min(
    STRUCTURAL_HEIGHT_STOPS[STRUCTURAL_HEIGHT_STOPS.length - 1].heightM,
    Math.max(0, Number.isFinite(heightMValue) ? heightMValue : 0),
  );
  for (let index = 0; index < STRUCTURAL_HEIGHT_STOPS.length - 1; index += 1) {
    const lower = STRUCTURAL_HEIGHT_STOPS[index];
    const upper = STRUCTURAL_HEIGHT_STOPS[index + 1];
    if (heightM > upper.heightM) continue;
    const span = upper.heightM - lower.heightM;
    const mix = span > 0 ? (heightM - lower.heightM) / span : 0;
    return [
      Math.round(lower.color[0] + ((upper.color[0] - lower.color[0]) * mix)),
      Math.round(lower.color[1] + ((upper.color[1] - lower.color[1]) * mix)),
      Math.round(lower.color[2] + ((upper.color[2] - lower.color[2]) * mix)),
    ];
  }
  const last = STRUCTURAL_HEIGHT_STOPS[STRUCTURAL_HEIGHT_STOPS.length - 1];
  return [last.color[0], last.color[1], last.color[2]];
};

export type FloorplanRgbLayer = {
  rgb_b64?: string;
  rgb_shape?: [number, number, number];
  observed_b64?: string;
};

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
  bounds?: { min_x: number; max_x: number; min_z: number; max_z: number };
  metricGridM?: number;
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

    if (maskValues) {
      if (!maskPasses(idx)) {
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

/**
 * Render the v9 structural floorplan without mixing walls or ceilings into
 * furniture height. Inferred interior floor remains visibly distinct, while
 * learned RGB supplies only a restrained texture cue and never changes the
 * metric height color.
 */
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
    (normalized) => structuralHeightColor(normalized * 1.65),
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
