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
  // Apply gamma to the normalized value after clamping to [0,1]. gamma < 1 boosts low values.
  gamma?: number;
  // Optional mask layer: if provided, pixels where mask <= threshold render as black.
  // Intended for hiding unobserved cells (e.g., density == 0).
  maskLayer?: FloorplanLayer;
  maskThreshold?: number;
  maskInvert?: boolean;
};

type RenderLayerResult = {
  contentRectPx: { x: number; y: number; w: number; h: number };
};

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
  const outsideColor: [number, number, number] = [0, 0, 0];

  for (let idx = 0; idx < (rowsW * colsW); idx += 1) {
    const w = walkValues[idx];
    const oh = obsValues[idx];
    let r = outsideColor[0];
    let g = outsideColor[1];
    let b = outsideColor[2];

    if (Number.isFinite(oh) && oh > obstacleEps) {
      const t = Math.min(1, Math.max(0, (oh - obsMin) / obsDenom));
      const [rr, gg, bb] = infernoColor(t);
      r = rr; g = gg; b = bb;
    } else if (Number.isFinite(w) && w > 0.5) {
      r = floorColor[0];
      g = floorColor[1];
      b = floorColor[2];
    }

    const offset = idx * 4;
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
  const dpr = window.devicePixelRatio || 1;

  let { width, height } = applyCanvasSize(canvas);
  if (fit === 'stretch') {
    const aspect = forceAspect || (colsW / rowsW);
    height = width / aspect;
  }
  canvas.width = Math.max(1, Math.round(width * dpr));
  canvas.height = Math.max(1, Math.round(height * dpr));

  const contentAspect = forceAspect || (colsW / rowsW);
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
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offscreen, contentX, contentY, contentW, contentH);
  ctx.restore();

  return {
    contentRectPx: {
      x: contentX * dpr,
      y: contentY * dpr,
      w: contentW * dpr,
      h: contentH * dpr
    }
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
  const min = (optObj && typeof optObj.valueMin === 'number') ? optObj.valueMin : (layer.value_min ?? 0);
  const max = (optObj && typeof optObj.valueMax === 'number') ? optObj.valueMax : (layer.value_max ?? 1);
  const denom = max - min === 0 ? 1 : max - min;
  const gamma = (optObj && typeof optObj.gamma === 'number' && Number.isFinite(optObj.gamma) && optObj.gamma > 0)
    ? optObj.gamma
    : 1.0;

  let maskValues: Float32Array | null = null;
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

  const n = rows * cols;
  for (let idx = 0; idx < n; idx += 1) {
    const v = values[idx];
    const offset = idx * 4;
    if (!Number.isFinite(v)) {
      data[offset] = 0;
      data[offset + 1] = 0;
      data[offset + 2] = 0;
      data[offset + 3] = 255;
      continue;
    }

    if (maskValues) {
      const mv = maskValues[idx];
      const ok = Number.isFinite(mv) && (mv > maskThreshold);
      const pass = maskInvert ? !ok : ok;
      if (!pass) {
        data[offset] = 0;
        data[offset + 1] = 0;
        data[offset + 2] = 0;
        data[offset + 3] = 255;
        continue;
      }
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
  const dpr = window.devicePixelRatio || 1;

  let { width, height } = applyCanvasSize(canvas);
  if (fit === 'stretch') {
    const aspect = forceAspect || (cols / rows);
    height = width / aspect;
  }
  canvas.width = Math.max(1, Math.round(width * dpr));
  canvas.height = Math.max(1, Math.round(height * dpr));

  const contentAspect = forceAspect || (cols / rows);
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
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offscreen, contentX, contentY, contentW, contentH);
  ctx.restore();

  return {
    contentRectPx: {
      x: contentX * dpr,
      y: contentY * dpr,
      w: contentW * dpr,
      h: contentH * dpr
    }
  };
}
