/**
 * Camera-ground presentation constants shared by every PCF Three.js view.
 *
 * camera_local_ground_m is +X camera-right, +Y height-up, +Z
 * camera-forward.  Its X/Y/Z basis is display-left-handed, so reflecting a
 * model axis to accommodate a conventional above-floor camera mirrors metric
 * landmarks.  Viewing from below the ground plane instead preserves the
 * geometry and gives +X screen-right and +Z screen-up.
 */
export const CAMERA_GROUND_OBLIQUE_VIEW_DIRECTION = Object.freeze([0, -1.2, 0.68]);
export const CAMERA_GROUND_TOP_DOWN_VIEW_DIRECTION = Object.freeze([0, -1, 0]);
export const CAMERA_GROUND_TOP_DOWN_UP = Object.freeze([0, 0, 1]);

const normalize = (vector) => {
  const length = Math.hypot(...vector);
  if (!(length > 1e-12)) throw new Error('camera_ground_zero_vector');
  return vector.map((value) => value / length);
};

const cross = (left, right) => [
  (left[1] * right[2]) - (left[2] * right[1]),
  (left[2] * right[0]) - (left[0] * right[2]),
  (left[0] * right[1]) - (left[1] * right[0]),
];

/** Return the screen basis used by framePerspectiveBounds. */
export function cameraGroundScreenBasis(
  viewDirection = CAMERA_GROUND_OBLIQUE_VIEW_DIRECTION,
  cameraUp = [0, 1, 0],
) {
  const backward = normalize(Array.from(viewDirection, Number));
  const up = normalize(Array.from(cameraUp, Number));
  const screenRight = normalize(cross(up, backward));
  const screenUp = normalize(cross(backward, screenRight));
  return { screenRight, screenUp, backward };
}

/** Project a row-zero-far camera-local raster into the obstacle isometric view. */
export function cameraLocalRasterToIsometric(row, column, tileWidth, tileHeight) {
  return {
    x: (Number(column) - Number(row)) * (Number(tileWidth) / 2),
    y: (Number(column) + Number(row)) * (Number(tileHeight) / 2),
  };
}
