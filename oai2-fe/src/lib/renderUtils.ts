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
};

type RenderLayerResult = {
  contentRectPx: { x: number; y: number; w: number; h: number };
};

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
  const min = layer.value_min ?? 0;
  const max = layer.value_max ?? 1;
  const denom = max - min === 0 ? 1 : max - min;

  for (let idx = 0; idx < values.length; idx += 1) {
    const norm = Math.min(1, Math.max(0, (values[idx] - min) / denom));
    const [r, g, b] = palette(norm);
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
