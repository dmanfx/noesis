/**
 * Float16 (half precision) to Float32 conversion utilities
 * JavaScript doesn't have native Float16Array support, so we implement conversion manually
 */

/**
 * Convert a Uint16Array of IEEE 754 binary16 (half precision) values to Float32Array
 * Based on the IEEE 754 binary16 format specification
 */
export function halfToFloatArray(u16: Uint16Array): Float32Array {
  const result = new Float32Array(u16.length);

  for (let i = 0; i < u16.length; i++) {
    result[i] = halfToFloat(u16[i]);
  }

  return result;
}

/**
 * Convert a single IEEE 754 binary16 value to float32
 */
function halfToFloat(half: number): number {
  // Extract components from half (16-bit)
  const sign = (half >> 15) & 0x1;
  const exponent = (half >> 10) & 0x1f;
  const mantissa = half & 0x3ff;

  // Handle special cases
  if (exponent === 0) {
    // Zero or subnormal number
    if (mantissa === 0) {
      return sign ? -0.0 : 0.0;
    } else {
      // Subnormal: (-1)^sign * 2^(-14) * (mantissa / 2^10)
      return (sign ? -1 : 1) * Math.pow(2, -14) * (mantissa / 1024);
    }
  } else if (exponent === 31) {
    // Infinity or NaN
    if (mantissa === 0) {
      return sign ? -Infinity : Infinity;
    } else {
      return NaN;
    }
  }

  // Normal number: (-1)^sign * 2^(exponent-15) * (1 + mantissa/2^10)
  const signMultiplier = sign ? -1 : 1;
  const exponentValue = Math.pow(2, exponent - 15);
  const mantissaValue = 1 + (mantissa / 1024);

  return signMultiplier * exponentValue * mantissaValue;
}

/**
 * Convert a Float32Array to Uint16Array of half precision values
 * Useful for testing or round-trip validation
 */
export function floatToHalfArray(f32: Float32Array): Uint16Array {
  const result = new Uint16Array(f32.length);

  for (let i = 0; i < f32.length; i++) {
    result[i] = floatToHalf(f32[i]);
  }

  return result;
}

/**
 * Convert a single float32 to IEEE 754 binary16
 * This is a simplified implementation that may lose precision
 */
function floatToHalf(float: number): number {
  // Handle special cases
  if (!isFinite(float)) {
    if (isNaN(float)) return 0x7e00; // NaN
    if (float === Infinity) return 0x7c00; // +Inf
    if (float === -Infinity) return 0xfc00; // -Inf
  }

  const sign = float < 0 ? 1 : 0;
  const absValue = Math.abs(float);

  // Handle zero
  if (absValue === 0) {
    return sign << 15;
  }

  // Convert to log2 for exponent calculation
  const log2 = Math.log2(absValue);
  let exponent = Math.floor(log2);
  let mantissa = absValue / Math.pow(2, exponent) - 1; // Remove leading 1

  // Adjust for half precision range
  if (exponent < -14) {
    // Too small, make subnormal
    mantissa = absValue / Math.pow(2, -14);
    exponent = 0;
  } else if (exponent > 15) {
    // Too large, clamp to infinity
    return (sign << 15) | (31 << 10);
  } else {
    // Normal range
    exponent += 15; // Bias for half precision
  }

  // Convert mantissa to 10 bits
  mantissa = Math.round(mantissa * 1024);

  // Handle overflow in mantissa
  if (mantissa >= 1024) {
    mantissa = 0;
    exponent++;
    if (exponent > 30) {
      return (sign << 15) | (31 << 10); // Infinity
    }
  }

  return (sign << 15) | (exponent << 10) | mantissa;
}
