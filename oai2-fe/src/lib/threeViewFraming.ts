import * as THREE from 'three';

type TargetControls = {
  target: THREE.Vector3;
  update: () => void;
};

/**
 * Frames every corner of a world-space box for a fixed viewing direction.
 * Unlike a max-dimension heuristic, this accounts for the current horizontal
 * and vertical field of view, so shallow depth shells remain legible without
 * clipping on differently shaped canvases.
 */
export function framePerspectiveBounds({
  bounds,
  camera,
  controls,
  viewDirection,
  margin = 1.15,
}: {
  bounds: THREE.Box3;
  camera: THREE.PerspectiveCamera;
  controls: TargetControls;
  viewDirection: THREE.Vector3;
  margin?: number;
}) {
  if (bounds.isEmpty()) return;

  const center = bounds.getCenter(new THREE.Vector3());
  const direction = viewDirection.clone().normalize();
  const worldUp = camera.up.clone().normalize();
  const right = new THREE.Vector3().crossVectors(worldUp, direction).normalize();
  if (right.lengthSq() < 1e-8) {
    right.set(1, 0, 0);
  }
  const screenUp = new THREE.Vector3().crossVectors(direction, right).normalize();
  const verticalHalfFov = THREE.MathUtils.degToRad(camera.fov * 0.5);
  const horizontalHalfFov = Math.atan(
    Math.tan(verticalHalfFov) * Math.max(camera.aspect, 1e-3),
  );
  const tanVertical = Math.max(1e-3, Math.tan(verticalHalfFov));
  const tanHorizontal = Math.max(1e-3, Math.tan(horizontalHalfFov));
  const padding = Math.max(1, margin);

  let distance = 1;
  let maxAlong = 0;
  for (const x of [bounds.min.x, bounds.max.x]) {
    for (const y of [bounds.min.y, bounds.max.y]) {
      for (const z of [bounds.min.z, bounds.max.z]) {
        const relative = new THREE.Vector3(x, y, z).sub(center);
        const along = relative.dot(direction);
        maxAlong = Math.max(maxAlong, Math.abs(along));
        distance = Math.max(
          distance,
          along + (Math.abs(relative.dot(right)) * padding) / tanHorizontal,
          along + (Math.abs(relative.dot(screenUp)) * padding) / tanVertical,
        );
      }
    }
  }

  camera.position.copy(center).addScaledVector(direction, distance);
  camera.near = Math.max(0.01, (distance - maxAlong) * 0.05);
  camera.far = Math.max(100, distance + (maxAlong * 4));
  camera.lookAt(center);
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}
