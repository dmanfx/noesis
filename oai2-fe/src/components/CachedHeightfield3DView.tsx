import { useCallback, useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { FloorplanLayer, FloorplanResponse } from './DepthDrawer';
import { computeMaskedRange } from '../lib/depthQuality.js';
import { buildMaskedHeightfield } from '../lib/heightfieldModel.js';
import { decodeFloat32, turboColor } from '../lib/renderUtils';
import { framePerspectiveBounds } from '../lib/threeViewFraming';

type CachedHeightfield3DViewProps = {
  floorplan?: FloorplanResponse | null;
  heightLayer?: FloorplanLayer;
  densityLayer?: FloorplanLayer;
  heightExaggeration?: number;
  onCanvasReady?: (canvas: HTMLCanvasElement | null) => void;
};

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

export default function CachedHeightfield3DView({
  floorplan,
  heightLayer,
  densityLayer,
  heightExaggeration = 1,
  onCanvasReady,
}: CachedHeightfield3DViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const groupRef = useRef<THREE.Group | null>(null);
  const animationRef = useRef<number | null>(null);

  const model = useMemo(() => {
    if (!heightLayer?.grid_b64 || !heightLayer.grid_shape) return null;
    const [rows, cols] = heightLayer.grid_shape;
    const heightValues = decodeFloat32(heightLayer.grid_b64);
    if (!heightValues || heightValues.length < rows * cols) return null;
    let densityValues: Float32Array | null = null;
    if (
      densityLayer?.grid_b64
      && densityLayer.grid_shape?.[0] === rows
      && densityLayer.grid_shape?.[1] === cols
    ) {
      densityValues = decodeFloat32(densityLayer.grid_b64);
    }
    return buildMaskedHeightfield({
      heightValues,
      densityValues,
      rows,
      cols,
      bounds: floorplan?.bounds,
      fallbackGridResM: floorplan?.scale_m_per_px ?? 0.15,
      densityThreshold: 1e-6,
      maxVertices: 65_536,
      heightExaggeration,
    });
  }, [
    densityLayer?.grid_b64,
    densityLayer?.grid_shape,
    floorplan?.bounds,
    floorplan?.scale_m_per_px,
    heightExaggeration,
    heightLayer?.grid_b64,
    heightLayer?.grid_shape,
  ]);

  const clearModel = useCallback(() => {
    const group = groupRef.current;
    if (!group) return;
    sceneRef.current?.remove(group);
    disposeObject(group);
    groupRef.current = null;
  }, []);

  const frameModel = useCallback((focusObject: THREE.Object3D) => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;
    const bounds = new THREE.Box3().setFromObject(focusObject);
    if (bounds.isEmpty()) return;
    // A high, oblique initial view makes the observed footprint readable
    // while retaining enough elevation to show the actual height variation.
    framePerspectiveBounds({
      bounds,
      camera,
      controls,
      viewDirection: new THREE.Vector3(0.48, 1.15, 0.62),
      margin: 1.16,
    });
  }, []);

  const buildSceneModel = useCallback(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    clearModel();
    if (!model) return;

    const group = new THREE.Group();
    group.name = 'cached-height-agl-surface';
    const colorRange = computeMaskedRange(model.heights, {
      mask: model.observed,
      maskThreshold: 0,
      lowPercentile: 2,
      highPercentile: 98,
    });
    const min = colorRange?.min ?? 0;
    const max = colorRange?.max ?? Math.max(1, model.maxHeightM);
    const span = Math.max(0.001, max - min);
    const colors = new Float32Array(model.heights.length * 3);

    for (let idx = 0; idx < model.heights.length; idx += 1) {
      if (!model.observed[idx]) continue;
      const normalized = Math.min(1, Math.max(0, (model.heights[idx] - min) / span));
      const [r, g, b] = turboColor(normalized);
      colors[idx * 3] = r / 255;
      colors[idx * 3 + 1] = g / 255;
      colors[idx * 3 + 2] = b / 255;
    }

    if (model.indices.length) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(model.positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      geometry.setIndex(new THREE.BufferAttribute(model.indices, 1));
      geometry.computeVertexNormals();
      const mesh = new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({
          vertexColors: true,
          side: THREE.DoubleSide,
          roughness: 0.88,
          metalness: 0.02,
        }),
      );
      mesh.name = 'observed-heightfield-triangles';
      group.add(mesh);
    }

    const pointPositions = new Float32Array(model.observedCount * 3);
    const pointColors = new Float32Array(model.observedCount * 3);
    let point = 0;
    for (let idx = 0; idx < model.observed.length; idx += 1) {
      if (!model.observed[idx]) continue;
      pointPositions.set(model.positions.subarray(idx * 3, idx * 3 + 3), point * 3);
      pointColors.set(colors.subarray(idx * 3, idx * 3 + 3), point * 3);
      point += 1;
    }
    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute('position', new THREE.BufferAttribute(pointPositions, 3));
    pointGeometry.setAttribute('color', new THREE.BufferAttribute(pointColors, 3));
    const points = new THREE.Points(
      pointGeometry,
      new THREE.PointsMaterial({
        vertexColors: true,
        size: Math.max(0.015, Math.min(model.dx, model.dz) * 0.32),
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.82,
      }),
    );
    points.name = 'observed-heightfield-points';
    group.add(points);

    const width = model.bounds.maxX - model.bounds.minX;
    const depth = model.bounds.maxZ - model.bounds.minZ;
    const centerX = (model.bounds.minX + model.bounds.maxX) * 0.5;
    const centerZ = (model.bounds.minZ + model.bounds.maxZ) * 0.5;
    const gridSize = Math.max(width, depth, 1);
    const gridDivisions = Math.max(4, Math.min(80, Math.round(gridSize / Math.max(model.dx, model.dz))));
    const grid = new THREE.GridHelper(gridSize, gridDivisions, 0x6b7280, 0x293241);
    grid.position.set(centerX, 0, centerZ);
    grid.name = 'metric-floor-grid';
    group.add(grid);

    const axes = new THREE.AxesHelper(Math.max(0.4, Math.min(gridSize * 0.12, 1.5)));
    axes.name = 'camera-local-axes';
    group.add(axes);

    scene.add(group);
    groupRef.current = group;
    // Frame the explicitly observed points, not the full authoritative grid.
    // Unknown vertices remain omitted and the user can still zoom out to the
    // complete metric grid.
    frameModel(model.observedCount ? points : group);
  }, [clearModel, frameModel, model]);

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
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      52,
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

    scene.add(new THREE.HemisphereLight(0xffffff, 0x26313e, 0.75));
    const key = new THREE.DirectionalLight(0xffffff, 1.15);
    key.position.set(-5, 9, -5);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x8bc7ff, 0.45);
    rim.position.set(5, 5, 6);
    scene.add(rim);

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
      const width = Math.max(10, container.clientWidth || 1);
      const height = Math.max(10, container.clientHeight || 1);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    buildSceneModel();
  }, [buildSceneModel]);

  return <div className="cached-heightfield-view" ref={containerRef} />;
}
