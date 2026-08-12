import { useCallback, useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { SceneFusionPoints } from './DepthDrawer';
import { framePerspectiveBounds } from '../lib/threeViewFraming';
import { detectOverheadBand } from '../lib/overheadPresentation.js';

type Props = {
  artifact?: SceneFusionPoints | null;
  colorMode?: 'rgb' | 'provenance';
  hideOverhead?: boolean;
  onCanvasReady?: (canvas: HTMLCanvasElement | null) => void;
};

const PROVENANCE_RGB: Record<number, [number, number, number]> = {
  1: [255, 177, 56],
  2: [207, 91, 207],
  3: [88, 220, 112],
};

const SRGB_TO_LINEAR = Float32Array.from({ length: 256 }, (_unused, byte) => {
  const value = byte / 255;
  return value <= 0.04045
    ? value / 12.92
    : ((value + 0.055) / 1.055) ** 2.4;
});

function decodeBytes(base64?: string): Uint8Array | null {
  if (!base64) return null;
  try {
    const binary = atob(base64);
    const values = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      values[index] = binary.charCodeAt(index);
    }
    return values;
  } catch {
    return null;
  }
}

function decodeFloat32(base64?: string): Float32Array | null {
  const bytes = decodeBytes(base64);
  if (!bytes || bytes.byteLength % 4 !== 0) return null;
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Float32Array(copy.buffer);
}

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    const renderable = child as THREE.Mesh;
    renderable.geometry?.dispose?.();
    const material = renderable.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(material)) material.forEach((entry) => entry.dispose());
    else material?.dispose?.();
  });
}

export default function FusedPointCloud3DView({
  artifact,
  colorMode = 'rgb',
  hideOverhead = true,
  onCanvasReady,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const groupRef = useRef<THREE.Group | null>(null);
  const animationRef = useRef<number | null>(null);

  const model = useMemo(() => {
    const count = Math.max(0, Number(artifact?.point_count) || 0);
    const positions = decodeFloat32(artifact?.positions_f32_b64);
    const rgb = decodeBytes(artifact?.colors_rgb_u8_b64);
    const provenance = decodeBytes(artifact?.provenance_u8_b64);
    if (
      count <= 0
      || !positions
      || positions.length !== count * 3
      || !rgb
      || rgb.length !== count * 3
      || !provenance
      || provenance.length !== count
    ) return null;
    const overhead = hideOverhead
      ? detectOverheadBand(positions, { stride: 3, offset: 1 })
      : null;
    if (!overhead) return { count, positions, rgb, provenance };

    const visible = [];
    for (let index = 0; index < count; index += 1) {
      if (positions[(index * 3) + 1] < overhead.cutoffM) visible.push(index);
    }
    const visiblePositions = new Float32Array(visible.length * 3);
    const visibleRgb = new Uint8Array(visible.length * 3);
    const visibleProvenance = new Uint8Array(visible.length);
    visible.forEach((sourceIndex, targetIndex) => {
      visiblePositions.set(
        positions.subarray(sourceIndex * 3, (sourceIndex * 3) + 3),
        targetIndex * 3,
      );
      visibleRgb.set(rgb.subarray(sourceIndex * 3, (sourceIndex * 3) + 3), targetIndex * 3);
      visibleProvenance[targetIndex] = provenance[sourceIndex];
    });
    return {
      count: visible.length,
      positions: visiblePositions,
      rgb: visibleRgb,
      provenance: visibleProvenance,
    };
  }, [artifact, hideOverhead]);

  const clearModel = useCallback(() => {
    if (!groupRef.current) return;
    sceneRef.current?.remove(groupRef.current);
    disposeObject(groupRef.current);
    groupRef.current = null;
  }, []);

  const frameModel = useCallback((group: THREE.Group) => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;
    const bounds = new THREE.Box3().setFromObject(group);
    if (bounds.isEmpty()) return;
    framePerspectiveBounds({
      bounds,
      camera,
      controls,
      viewDirection: new THREE.Vector3(0, 1.2, -0.68),
      margin: 1.12,
    });
  }, []);

  const buildSceneModel = useCallback(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    clearModel();
    if (!model) return;
    const group = new THREE.Group();
    group.name = 'anchored-fixed-phone-fusion';
    // Match the heatmap and the other conditioned views: camera right is
    // screen-right and camera forward is screen-up without mutating evidence.
    group.scale.x = -1;
    const colors = new Float32Array(model.count * 3);
    for (let index = 0; index < model.count; index += 1) {
      const source = colorMode === 'provenance'
        ? (PROVENANCE_RGB[model.provenance[index]] ?? [170, 170, 170])
        : [model.rgb[index * 3], model.rgb[index * 3 + 1], model.rgb[index * 3 + 2]];
      colors[index * 3] = SRGB_TO_LINEAR[source[0]];
      colors[index * 3 + 1] = SRGB_TO_LINEAR[source[1]];
      colors[index * 3 + 2] = SRGB_TO_LINEAR[source[2]];
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(model.positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    const points = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        vertexColors: true,
        size: 1.6,
        sizeAttenuation: false,
        transparent: true,
        opacity: 0.95,
        toneMapped: false,
      }),
    );
    points.frustumCulled = false;
    group.add(points);
    group.add(new THREE.AxesHelper(0.75));
    scene.add(group);
    groupRef.current = group;
    frameModel(group);
  }, [clearModel, colorMode, frameModel, model]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || rendererRef.current) return;
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(container.clientWidth || 1, container.clientHeight || 1, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;
    onCanvasReady?.(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1320);
    sceneRef.current = scene;
    const camera = new THREE.PerspectiveCamera(
      54,
      (container.clientWidth || 1) / Math.max(1, container.clientHeight || 1),
      0.01,
      100,
    );
    camera.up.set(0, 1, 0);
    cameraRef.current = camera;
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controlsRef.current = controls;
    const resize = new ResizeObserver(() => {
      const width = Math.max(10, container.clientWidth || 1);
      const height = Math.max(10, container.clientHeight || 1);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      if (groupRef.current) frameModel(groupRef.current);
    });
    resize.observe(container);
    const render = () => {
      controls.update();
      renderer.render(scene, camera);
      animationRef.current = requestAnimationFrame(render);
    };
    render();
    return () => {
      resize.disconnect();
      if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
      clearModel();
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      rendererRef.current = null;
      sceneRef.current = null;
      cameraRef.current = null;
      controlsRef.current = null;
      onCanvasReady?.(null);
    };
  }, [clearModel, frameModel, onCanvasReady]);

  useEffect(() => {
    buildSceneModel();
  }, [buildSceneModel]);

  return <div ref={containerRef} className="calibrated-point-cloud-view" />;
}
