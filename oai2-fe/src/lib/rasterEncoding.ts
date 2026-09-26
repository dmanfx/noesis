// PCF visualization rasters travel as base64 inside the floorplan JSON response.
// Legacy unprefixed layers remain float32; compact PCF layers declare their
// binary type both in the payload prefix and (on newer producers) grid_encoding.
export const PCF_RASTER_ENCODING_VERSION = 1;
const MAX_GRID_CELLS = 16 * 1024 * 1024;

type GridEncoding = 'f32-le-base64' | 'f16-le-base64' | 'bitpack-msb-base64' | 'u8-base64';

function detectedEncoding(encoded: string): GridEncoding | null {
  if (encoded.startsWith('f16:')) return 'f16-le-base64';
  if (encoded.startsWith('bit:')) return 'bitpack-msb-base64';
  if (encoded.startsWith('u8:')) return 'u8-base64';
  return /^[A-Za-z0-9+/]+={0,2}$/.test(encoded) ? 'f32-le-base64' : null;
}

function expectedEncodedLength(byteCount: number): number {
  return 4 * Math.ceil(byteCount / 3);
}

function bytesFromBase64(encoded: string, expectedLength?: number): Uint8Array {
  const binary = atob(encoded);
  if (expectedLength !== undefined && binary.length !== expectedLength) {
    throw new Error('Raster byte length does not match its encoding');
  }
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function halfToFloat(value: number): number {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) return fraction === 0 ? sign * 0 : sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction === 0 ? sign * Infinity : NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

export function decodeFloat16(base64?: string): Float32Array | null {
  if (!base64) return null;
  try {
    const bytes = bytesFromBase64(base64);
    if (bytes.length % 2 || bytes.length / 2 > MAX_GRID_CELLS) throw new Error('Invalid float16 raster length');
    const view = new DataView(bytes.buffer);
    const out = new Float32Array(bytes.length / 2);
    for (let index = 0; index < out.length; index += 1) out[index] = halfToFloat(view.getUint16(index * 2, true));
    return out;
  } catch (error) {
    console.error('Failed to decode float16 payload', error);
    return null;
  }
}

export function decodeFloat32(base64?: string): Float32Array | null {
  if (!base64) return null;
  try {
    const encoding = detectedEncoding(base64);
    if (encoding === 'f16-le-base64') return decodeFloat16(base64.slice(4));
    if (encoding === 'bitpack-msb-base64') {
      const separator = base64.indexOf(':', 4);
      if (separator < 0) throw new Error('Packed mask is missing its cell count');
      const count = Number(base64.slice(4, separator));
      if (!Number.isSafeInteger(count) || count < 0 || count > MAX_GRID_CELLS) {
        throw new Error('Packed mask has an invalid cell count');
      }
      const bytes = bytesFromBase64(base64.slice(separator + 1), Math.ceil(count / 8));
      const values = new Float32Array(count);
      for (let index = 0; index < count; index += 1) {
        values[index] = (bytes[index >> 3] >> (7 - (index & 7))) & 1;
      }
      return values;
    }
    if (encoding === 'u8-base64') {
      const bytes = bytesFromBase64(base64.slice(3));
      if (bytes.length > MAX_GRID_CELLS) throw new Error('Uint8 raster exceeds the cell bound');
      return Float32Array.from(bytes);
    }
    if (encoding !== 'f32-le-base64') throw new Error('Unsupported raster encoding');
    const bytes = bytesFromBase64(base64);
    if (bytes.length % 4 || bytes.length / 4 > MAX_GRID_CELLS) throw new Error('Invalid float32 raster length');
    const view = new DataView(bytes.buffer);
    const values = new Float32Array(bytes.length / 4);
    for (let index = 0; index < values.length; index += 1) values[index] = view.getFloat32(index * 4, true);
    return values;
  } catch (error) {
    console.error('Failed to decode float32 payload', error);
    return null;
  }
}

// Return a user-visible explanation before an unknown producer format becomes
// an empty canvas. An absent version is accepted for already-deployed producers.
export function pcfRasterCompatibilityError(response: unknown): string | null {
  if (!response || typeof response !== 'object') return null;
  const payload = response as Record<string, unknown>;
  if (payload.scene_prior_only !== true || payload.display_source !== 'pcf') return null;
  const meta = payload.scene_prior_diagnostic_meta as Record<string, unknown> | undefined;
  const version = meta?.raster_encoding_version;
  if (version !== undefined && version !== PCF_RASTER_ENCODING_VERSION) {
    return `Unsupported PCF raster encoding version ${String(version)}; update the dashboard.`;
  }
  for (const [name, value] of Object.entries(payload)) {
    if (!name.startsWith('scene_prior_diagnostic_') && !name.startsWith('scene_prior_floor_')) continue;
    if (!value || typeof value !== 'object') continue;
    const layer = value as Record<string, unknown>;
    if (typeof layer.grid_b64 === 'string') {
      const detected = detectedEncoding(layer.grid_b64);
      const shape = layer.grid_shape;
      if (!detected || (layer.grid_encoding !== undefined && layer.grid_encoding !== detected)
        || !Array.isArray(shape) || shape.length !== 2) {
        return `Unsupported PCF raster encoding in ${name}; update the dashboard.`;
      }
      const [rows, cols] = shape;
      const count = Number(rows) * Number(cols);
      if (!Number.isSafeInteger(count) || count <= 0 || count > MAX_GRID_CELLS) {
        return `Invalid PCF raster shape in ${name}.`;
      }
      const colon = detected === 'bitpack-msb-base64' ? layer.grid_b64.indexOf(':', 4) : -1;
      if (detected === 'bitpack-msb-base64'
        && (colon < 0 || Number(layer.grid_b64.slice(4, colon)) !== count)) {
        return `Invalid PCF raster size in ${name}.`;
      }
      const prefixLength = detected === 'f16-le-base64' ? 4
        : detected === 'u8-base64' ? 3
          : detected === 'bitpack-msb-base64' ? colon + 1 : 0;
      const byteCount = detected === 'f32-le-base64' ? count * 4
        : detected === 'f16-le-base64' ? count * 2
          : detected === 'u8-base64' ? count : Math.ceil(count / 8);
      if (layer.grid_b64.length - prefixLength !== expectedEncodedLength(byteCount)) {
        return `Invalid PCF raster size in ${name}.`;
      }
    }
    if (typeof layer.rgb_b64 === 'string') {
      if (layer.rgb_encoding !== undefined && layer.rgb_encoding !== 'rgb-u8-base64') {
        return `Unsupported PCF raster encoding in ${name}; update the dashboard.`;
      }
      const shape = layer.rgb_shape;
      if (!Array.isArray(shape) || shape.length !== 3 || shape[2] !== 3
        || !Number.isSafeInteger(Number(shape[0]) * Number(shape[1]))
        || Number(shape[0]) * Number(shape[1]) <= 0
        || Number(shape[0]) * Number(shape[1]) > MAX_GRID_CELLS) {
        return `Invalid PCF raster shape in ${name}.`;
      }
      const count = Number(shape[0]) * Number(shape[1]);
      if (layer.rgb_b64.length !== expectedEncodedLength(count * 3)) {
        return `Invalid PCF raster size in ${name}.`;
      }
      if (typeof layer.observed_b64 === 'string') {
        const separator = layer.observed_b64.indexOf(':', 4);
        if (!layer.observed_b64.startsWith('bit:') || separator < 0
          || Number(layer.observed_b64.slice(4, separator)) !== count
          || layer.observed_b64.length - separator - 1 !== expectedEncodedLength(Math.ceil(count / 8))
          || (layer.observed_encoding !== undefined && layer.observed_encoding !== 'bitpack-msb-base64')) {
          return `Unsupported PCF raster encoding in ${name}; update the dashboard.`;
        }
      }
    }
  }
  return null;
}
