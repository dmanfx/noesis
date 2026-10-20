import { decodeFloat32 } from './rasterEncoding';
import { bwColor, decodeUint8, turboColor, viridisColor } from './renderUtils';
import type { AppearanceTheme, BevLayer, ScientificPalette } from './dashboardPresentation.js';
import type { UnifiedBevRaster, UnifiedBevRasterLayer } from './unifiedBevConsumer.js';
import type { UnifiedBevMap } from './unifiedBevWorld.js';

export type BevHeightRange = Readonly<{ min: number; max: number }>;
export type BevRasterPixels = Readonly<{ width: number; height: number; pixels: Uint8ClampedArray }>;
export const BEV_PRESENTATION_IMAGE_LIMIT = 7; // two structure themes, RGB, normals, three height palettes
export const BEV_LAYER_LABELS: Readonly<Record<BevLayer, string>> = Object.freeze({
  structure: 'Structure', texture: 'Texture', height: 'Height', normals: 'Room normals',
});

/** Shared with the visible metre legend. No percentile clipping or per-room rescaling. */
export function combinedBevHeightRange(ranges: readonly (BevHeightRange | null)[]): BevHeightRange | null {
  const present = ranges.filter((range): range is BevHeightRange => range !== null);
  return present.length ? Object.freeze({ min: Math.min(...present.map(r => r.min)), max: Math.max(...present.map(r => r.max)) }) : null;
}

export function bevPaletteColor(palette: ScientificPalette, value: number): [number, number, number] {
  return palette === 'gray' ? bwColor(value) : palette === 'viridis' ? viridisColor(value) : turboColor(value);
}

/** One owner per currently admitted raster. No module/global cache or retained history. */
export function createBevPresentationRaster(raster: UnifiedBevRaster, map: UnifiedBevMap) {
  const [rows, columns] = map.gridShape;
  const count = rows * columns;
  const grids = new Map<UnifiedBevRasterLayer, Float32Array | null>();
  const decode = (layer: UnifiedBevRasterLayer | null) => {
    if (!layer) return null;
    if (!grids.has(layer)) {
      const values = decodeFloat32(layer.grid_b64);
      grids.set(layer, values?.length === count ? values : null);
    }
    return grids.get(layer) ?? null;
  };
  const mask = decode(raster.mask);
  const validAt = (values: Float32Array | null, index: number) => Boolean(values && mask
    && Number.isFinite(values[index]) && Number.isFinite(mask[index]) && mask[index] > 0);
  // Decode AGL once for a common physical scale; structure may share this object.
  const heightValues = decode(raster.heightLayer);
  let min = Infinity; let max = -Infinity;
  if (heightValues && mask) for (let index = 0; index < count; index++) {
    if (!validAt(heightValues, index)) continue;
    min = Math.min(min, heightValues[index]); max = Math.max(max, heightValues[index]);
  }
  const heightRange = Number.isFinite(min) ? Object.freeze({ min, max }) : null;
  let colors: Uint8Array | null | undefined;
  let colorMask: Float32Array | null | undefined;
  // Inverse transpose of the admitted local-to-scene basis. Translation and the
  // display's horizontal reflection do not change physical normal directions.
  const m = map.sceneFromLocalColMajor;
  const a = m[0]; const b = m[4]; const c = m[8];
  const d = m[1]; const e = m[5]; const f = m[9];
  const g = m[2]; const h = m[6]; const i = m[10];
  const cofactors = [e * i - f * h, f * g - d * i, d * h - e * g,
    c * h - b * i, a * i - c * g, b * g - a * h,
    b * f - c * e, c * d - a * f, a * e - b * d];
  const determinant = a * cofactors[0] + b * cofactors[1] + c * cofactors[2];
  const stepX = (map.localBounds.max_x - map.localBounds.min_x) / columns;
  const stepZ = (map.localBounds.max_z - map.localBounds.min_z) / rows;

  const render = (layer: BevLayer, palette: ScientificPalette, commonHeightRange: BevHeightRange | null,
    theme: AppearanceTheme = 'dark'): BevRasterPixels | null => {
    const values = layer === 'structure' ? decode(raster.layer) : heightValues;
    if (layer !== 'texture' && (!values || !mask)) return null;
    if (layer === 'height' && (!heightRange || !commonHeightRange)) return null;
    if (layer === 'normals' && (!heightRange || !Number.isFinite(determinant) || Math.abs(determinant) < 1e-9)) return null;
    if (layer === 'texture') {
      if (!raster.surfaceRgb) return null;
      if (colors === undefined) colors = decodeUint8(raster.surfaceRgb.rgb_b64);
      if (colorMask === undefined) colorMask = decodeFloat32(raster.surfaceRgb.observed_b64);
      if (!colors || colors.length !== count * 3 || !colorMask || colorMask.length !== count) return null;
    }
    const pixels = new Uint8ClampedArray(count * 4);
    let visible = 0;
    const derivative = (index: number, before: number | null, after: number | null, step: number) => {
      const hasBefore = before !== null && validAt(values, before);
      const hasAfter = after !== null && validAt(values, after);
      if (hasBefore && hasAfter) return (values![after!] - values![before!]) / (2 * step);
      if (hasBefore) return (values![index] - values![before!]) / step;
      if (hasAfter) return (values![after!] - values![index]) / step;
      return null;
    };
    for (let index = 0; index < count; index++) {
      let rgb: ArrayLike<number>;
      if (layer === 'texture') {
        // RGB owns its own observation mask. A sparse general mask must not
        // invent texture, and missing RGB never becomes an AGL substitute.
        if (!Number.isFinite(colorMask![index]) || colorMask![index] <= 0) continue;
        rgb = colors!.subarray(index * 3, index * 3 + 3);
      } else {
        if (!validAt(values, index)) continue;
        if (layer === 'structure') {
          const height = Math.min(1, Math.max(0, values![index] / 2.5));
          rgb = (theme === 'light' ? [58 + height * 85, 77 + height * 85, 99 + height * 82]
            : [116 + height * 128, 134 + height * 114, 154 + height * 100]).map(Math.round);
        } else if (layer === 'height') {
          const range = commonHeightRange!;
          rgb = bevPaletteColor(palette, range.max === range.min ? .5 : (values![index] - range.min) / (range.max - range.min));
        } else {
          const row = Math.floor(index / columns); const col = index % columns;
          const dx = derivative(index, col ? index - 1 : null, col < columns - 1 ? index + 1 : null, stepX);
          // Rows increase toward MIN local Z: the north/upper neighbour is +Z.
          const dz = derivative(index, row < rows - 1 ? index + columns : null, row ? index - columns : null, stepZ);
          if (dx === null || dz === null) continue;
          const local = [-dx, 1, -dz];
          const normal = [0, 1, 2].map(axis => (cofactors[axis * 3] * local[0]
            + cofactors[axis * 3 + 1] * local[1] + cofactors[axis * 3 + 2] * local[2]) / determinant);
          const length = Math.hypot(...normal);
          if (!Number.isFinite(length) || length <= 1e-9) continue;
          rgb = normal.map(component => Math.round((component / length * .5 + .5) * 255));
        }
      }
      pixels.set(rgb, index * 4); pixels[index * 4 + 3] = 255; visible++;
    }
    return visible ? Object.freeze({ width: columns, height: rows, pixels }) : null;
  };
  return Object.freeze({ raster, heightRange, render });
}
