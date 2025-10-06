// Lightweight binary helpers for the frontend

/**
 * Decode a base64 string that encodes raw little-endian Float32 values
 * into a Float32Array. Returns an empty array for invalid input.
 */
export function decodeFloat32(b64: string): Float32Array {
  try {
    if (!b64 || typeof b64 !== 'string') return new Float32Array(0);
    // atob decodes to a binary string (1 char = 1 byte)
    const bin = atob(b64);
    const len = bin.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i += 1) bytes[i] = bin.charCodeAt(i) & 0xff;
    // The underlying order is little-endian from the server; Float32Array view uses native endianness
    // which is little-endian on modern platforms. If this ever needs to be forced, copy via DataView.
    return new Float32Array(bytes.buffer);
  } catch {
    return new Float32Array(0);
  }
}

