import { useCallback, useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { VisibleFloorPlaneModel } from '../lib/visibleFloorPlane';
import { computeMaskedRange } from '../lib/depthQuality.js';
import { buildCalibratedPointCloud } from '../lib/pointCloudModel.js';
import { turboColor, viridisColor } from '../lib/renderUtils';
import { framePerspectiveBounds } from '../lib/threeViewFraming';

type CalibratedPointCloud3DViewProps = {
  depthValues?: Float32Array | null;
  confidenceValues?: Float32Array | null;
  maskValues?: Uint8Array | null;
  rgbValues?: Uint8Array | null;
  rgbShape?: [number, number, number] | null;
  width?: number;
  height?: number;
  intrinsics?: number[] | null;
  floorModel?: VisibleFloorPlaneModel | null;
  colorMode?: 'rgb' | 'depth' | 'confidence';
  onCanvasReady?: (canvas: HTMLCanvasElement | null) => void;
};

const SRGB_BYTE_TO_LINEAR = Float32Array.from(
  { length: 256 },
  (_unused, byte) => {
    const value = byte / 255;
    return value <= 0.04045
      ? value / 12.92
      : ((value + 0.055) / 1.055) ** 2.4;
  },
);

function linearPalette(
  palette: (value: number) => [number, number, number],
): Float32Array {
  const output = new Float32Array(256 * 3);
  for (let index = 0; index < 256; index += 1) {
    const [red, green, blue] = palette(index / 255);
    output[index * 3] = SRGB_BYTE_TO_LINEAR[red];
    output[index * 3 + 1] = SRGB_BYTE_TO_LINEAR[green];
    output[index * 3 + 2] = SRGB_BYTE_TO_LINEAR[blue];
  }
  return output;
}

const TURBO_LINEAR_PALETTE = linearPalette(turboColor);
const VIRIDIS_LINEAR_PALETTE = linearPalette(viridisColor);

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    const renderable = child as THREE.Mesh;
    renderable.geometry?.dispose?.();
    const material = renderable.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(material)) {
      material.forEach((entry) => entry.dispose());
    } else {
      material?.dispose?.();
    }
  });
}

export default function CalibratedPointCloud3DView({
  depthValues,
  confidenceValues,
  maskValues,
  rgbValues,
  rgbShape,
  width = 0,
  height = 0,
  intrinsics,
  floorModel,
  colorMode = 'depth',
  onCanvasReady,
}: CalibratedPointCloud3DViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const groupRef = useRef<THREE.Group | null>(null);
  const animationRef = useRef<number | null>(null);

  const model = useMemo(() => {
    if (!depthValues || !intrinsics) return null;
    return buildCalibratedPointCloud({
      depthValues,
      confidenceValues,
      maskValues,
      rgbValues,
      rgbShape,
      width,
      height,
      intrinsics,
      maxPoints: 250_000,
      minDepthM: 0.1,
      maxDepthM: 50,
      minConfidence: 0.05,
    });
  }, [confidenceValues, depthValues, height, intrinsics, maskValues, rgbShape, rgbValues, width]);

  const clearModel = useCallback(() => {
    const group = groupRef.current;
    if (!group) return;
    sceneRef.current?.remove(group);
    disposeObject(group);
    groupRef.current = null;
  }, []);

  const frameModel = useCallback((group: THREE.Group) => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;
    const bounds = new THREE.Box3().setFromObject(group);
    if (bounds.isEmpty()) return;
    // Backprojection is a camera-facing depth shell. Start close to the
    // optical-axis view with a slight oblique offset, preserving depth cues
    // without presenting the shell edge-on.
    framePerspectiveBounds({
      bounds,
      camera,
      controls,
      viewDirection: new THREE.Vector3(0.08, 0.06, -1),
      margin: 1.14,
    });
  }, []);

  const buildSceneModel = useCallback(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    clearModel();
    if (!model || !model.count) return;
    // RGB is already an exact same-snapshot diagnostic. Keep it free from
    // synthetic distance fog; depth/confidence modes retain fog as a cue.
    scene.fog = colorMode === 'rgb'
      ? null
      : new THREE.Fog(0x0d1320, 18, 55);

    const group = new THREE.Group();
    group.name = 'calibrated-dense-point-cloud';
    const colors = new Float32Array(model.count * 3);
    const colorValues = colorMode === 'confidence' ? model.confidences : model.depths;
    const range = colorMode === 'rgb'
      ? null
      : computeMaskedRange(colorValues, {
        positiveOnly: colorMode === 'depth',
        lowPercentile: 2,
        highPercentile: 98,
      });
    const min = range?.min ?? 0;
    const max = range?.max ?? 1;
    const span = Math.max(0.001, max - min);
    for (let idx = 0; idx < model.count; idx += 1) {
      if (colorMode === 'rgb') {
        const rgb = model.rgbColors;
        const offset = idx * 3;
        const r = rgb ? rgb[offset] : 150;
        const g = rgb ? rgb[offset + 1] : 156;
        const b = rgb ? rgb[offset + 2] : 166;
        colors[offset] = SRGB_BYTE_TO_LINEAR[r];
        colors[offset + 1] = SRGB_BYTE_TO_LINEAR[g];
        colors[offset + 2] = SRGB_BYTE_TO_LINEAR[b];
        continue;
      }
      const normalized = Math.min(1, Math.max(0, (colorValues[idx] - min) / span));
      const palette = colorMode === 'confidence'
        ? VIRIDIS_LINEAR_PALETTE
        : TURBO_LINEAR_PALETTE;
      const paletteOffset = Math.round(normalized * 255) * 3;
      colors[idx * 3] = palette[paletteOffset];
      colors[idx * 3 + 1] = palette[paletteOffset + 1];
      colors[idx * 3 + 2] = palette[paletteOffset + 2];
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(model.positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const points = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        vertexColors: true,
        size: 0.024,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.9,
        toneMapped: false,
      }),
    );
    points.name = 'depth-points';
    group.add(points);

    if (floorModel) {
      const floorPositions = new Float32Array(floorModel.footprintMesh.vertices.length * 3);
      floorModel.footprintMesh.vertices.forEach((point, idx) => {
        floorPositions[idx * 3] = point[0];
        floorPositions[idx * 3 + 1] = point[1];
        floorPositions[idx * 3 + 2] = point[2];
      });
      const floorGeometry = new THREE.BufferGeometry();
      floorGeometry.setAttribute('position', new THREE.BufferAttribute(floorPositions, 3));
      floorGeometry.setIndex(floorModel.footprintMesh.indices);
      floorGeometry.computeVertexNormals();
      const floor = new THREE.Mesh(
        floorGeometry,
        new THREE.MeshBasicMaterial({
          color: 0x58d68d,
          transparent: true,
          opacity: 0.18,
          side: THREE.DoubleSide,
          depthWrite: false,
        }),
      );
      floor.name = 'fitted-floor-overlay';
      group.add(floor);
    }

    const axes = new THREE.AxesHelper(0.75);
    axes.name = 'camera-local-axes';
    group.add(axes);

    scene.add(group);
    groupRef.current = group;
    frameModel(group);
  }, [clearModel, colorMode, floorModel, frameModel, model]);

  useEffect(() => {
    if (rendererRef.current || !containerRef.current) return;
    const container = containerRef.current;
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(container.clientWidth || 1, container.clientHeight || 1, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    rendererRef.current = renderer;
    container.appendChild(renderer.domElement);
    onCanvasReady?.(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1320);
    scene.fog = new THREE.Fog(0x0d1320, 18, 55);
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

    const renderLoop = () => {
      if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;
      controlsRef.current?.update();
      rendererRef.current.render(sceneRef.current, cameraRef.current);
      animationRef.current = window.requestAnimationFrame(renderLoop);
    };
    renderLoop();

    return () => {
      if (animationRef.current !== null) {
        window.cancelAnimationFrame(animationRef.current);
      }
      clearModel();
      controlsRef.current?.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
      onCanvasReady?.(null);
      rendererRef.current = null;
      sceneRef.current = null;
      cameraRef.current = null;
      controlsRef.current = null;
    };
  }, [clearModel, onCanvasReady]);

  useEffect(() => {
    const renderer = rendererRef.current;
    const camera = cameraRef.current;
    const container = containerRef.current;
    if (!renderer || !camera || !container) return;
    const resize = () => {
      const nextWidth = Math.max(10, container.clientWidth || 1);
      const nextHeight = Math.max(10, container.clientHeight || 1);
      camera.aspect = nextWidth / nextHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(nextWidth, nextHeight, false);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    buildSceneModel();
  }, [buildSceneModel]);

  return <div className="calibrated-point-cloud-view" ref={containerRef} />;
}
