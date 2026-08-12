const SHA256_INITIAL_STATE = new Uint32Array([
  0x6a09e667,
  0xbb67ae85,
  0x3c6ef372,
  0xa54ff53a,
  0x510e527f,
  0x9b05688c,
  0x1f83d9ab,
  0x5be0cd19,
]);

const SHA256_ROUND_CONSTANTS = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
  0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
  0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
  0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
  0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
  0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const rotateRight = (value: number, places: number): number => (
  (value >>> places) | (value << (32 - places))
);

const compressSha256Block = (
  state: Uint32Array,
  bytes: Uint8Array,
  offset: number,
  schedule: Uint32Array,
): void => {
  for (let index = 0; index < 16; index += 1) {
    const byteOffset = offset + (index * 4);
    schedule[index] = (
      (bytes[byteOffset] << 24)
      | (bytes[byteOffset + 1] << 16)
      | (bytes[byteOffset + 2] << 8)
      | bytes[byteOffset + 3]
    ) >>> 0;
  }
  for (let index = 16; index < 64; index += 1) {
    const previous15 = schedule[index - 15];
    const previous2 = schedule[index - 2];
    const sigma0 = (
      rotateRight(previous15, 7)
      ^ rotateRight(previous15, 18)
      ^ (previous15 >>> 3)
    );
    const sigma1 = (
      rotateRight(previous2, 17)
      ^ rotateRight(previous2, 19)
      ^ (previous2 >>> 10)
    );
    schedule[index] = (
      schedule[index - 16]
      + sigma0
      + schedule[index - 7]
      + sigma1
    ) >>> 0;
  }

  let a = state[0];
  let b = state[1];
  let c = state[2];
  let d = state[3];
  let e = state[4];
  let f = state[5];
  let g = state[6];
  let h = state[7];

  for (let index = 0; index < 64; index += 1) {
    const sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
    const choose = (e & f) ^ (~e & g);
    const temporary1 = (
      h
      + sum1
      + choose
      + SHA256_ROUND_CONSTANTS[index]
      + schedule[index]
    ) >>> 0;
    const sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
    const majority = (a & b) ^ (a & c) ^ (b & c);
    const temporary2 = (sum0 + majority) >>> 0;

    h = g;
    g = f;
    f = e;
    e = (d + temporary1) >>> 0;
    d = c;
    c = b;
    b = a;
    a = (temporary1 + temporary2) >>> 0;
  }

  state[0] = (state[0] + a) >>> 0;
  state[1] = (state[1] + b) >>> 0;
  state[2] = (state[2] + c) >>> 0;
  state[3] = (state[3] + d) >>> 0;
  state[4] = (state[4] + e) >>> 0;
  state[5] = (state[5] + f) >>> 0;
  state[6] = (state[6] + g) >>> 0;
  state[7] = (state[7] + h) >>> 0;
};

/**
 * Compute SHA-256 without relying on Web Crypto's secure-context restriction.
 *
 * The depth dashboard is intentionally available on a trusted HTTP LAN
 * origin, where `crypto.subtle` is absent. Bulk snapshot integrity must still
 * be verified rather than skipped, so the worker uses this implementation on
 * every origin.
 */
export const sha256Hex = (bytes: Uint8Array): string => {
  const state = SHA256_INITIAL_STATE.slice();
  const schedule = new Uint32Array(64);
  const completeByteCount = bytes.byteLength - (bytes.byteLength % 64);
  for (let offset = 0; offset < completeByteCount; offset += 64) {
    compressSha256Block(state, bytes, offset, schedule);
  }

  const remainingByteCount = bytes.byteLength - completeByteCount;
  const paddedByteCount = remainingByteCount < 56 ? 64 : 128;
  const tail = new Uint8Array(paddedByteCount);
  tail.set(bytes.subarray(completeByteCount));
  tail[remainingByteCount] = 0x80;

  const bitLength = bytes.byteLength * 8;
  const highBitLength = Math.floor(bitLength / 0x1_0000_0000);
  const lowBitLength = bitLength >>> 0;
  const lengthOffset = paddedByteCount - 8;
  tail[lengthOffset] = (highBitLength >>> 24) & 0xff;
  tail[lengthOffset + 1] = (highBitLength >>> 16) & 0xff;
  tail[lengthOffset + 2] = (highBitLength >>> 8) & 0xff;
  tail[lengthOffset + 3] = highBitLength & 0xff;
  tail[lengthOffset + 4] = (lowBitLength >>> 24) & 0xff;
  tail[lengthOffset + 5] = (lowBitLength >>> 16) & 0xff;
  tail[lengthOffset + 6] = (lowBitLength >>> 8) & 0xff;
  tail[lengthOffset + 7] = lowBitLength & 0xff;

  for (let offset = 0; offset < paddedByteCount; offset += 64) {
    compressSha256Block(state, tail, offset, schedule);
  }

  return Array.from(state)
    .map((word) => word.toString(16).padStart(8, '0'))
    .join('');
};
