import React, { useCallback, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { VisibleFloorPlaneModel } from '../lib/visibleFloorPlane';

const MODEL_COMPARISON_TOP_DOWN_UP = new THREE.Vector3(-1, 0, 0);

type FloorPlane3DViewProps = {
  model: VisibleFloorPlaneModel | null;
  onCanvasReady?: (canvas: HTMLCanvasElement | null) => void;
};

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    const mesh = child as THREE.Mesh;
    mesh.geometry?.dispose?.();
    const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(material)) {
      material.forEach((m) => m.dispose());
    } else {
      material?.dispose?.();
    }
  });
}

const FloorPlane3DView: React.FC<FloorPlane3DViewProps> = ({ model, onCanvasReady }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const groupRef = useRef<THREE.Group | null>(null);
  const animRef = useRef<number | null>(null);

  const clearModel = useCallback(() => {
    const group = groupRef.current;
    if (!group) return;
    if (sceneRef.current) sceneRef.current.remove(group);
    disposeObject(group);
    groupRef.current = null;
  }, []);

  const frameModel = useCallback((group: THREE.Group) => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;
    const box = new THREE.Box3().setFromObject(group);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    const maxDim = Math.max(size.x, size.y, size.z, 1);
    const dist = maxDim * 1.18;
    camera.position.set(center.x, center.y - dist, center.z);
    camera.near = Math.max(0.01, dist * 0.01);
    camera.far = Math.max(50, dist * 8);
    // Keep geometry in camera-local ground (X/Z with Y height), but present it
    // in the same top-down screen basis used by the model-alignment view.
    camera.up.copy(MODEL_COMPARISON_TOP_DOWN_UP);
    camera.lookAt(center);
    camera.updateProjectionMatrix();
    controls.target.copy(center);
    controls.update();
  }, []);

  const buildModel = useCallback(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    clearModel();
    if (!model) return;

    const group = new THREE.Group();
    group.name = 'visible-floor-plane-model';

    const vertices = new Float32Array(model.footprintMesh.vertices.length * 3);
    model.footprintMesh.vertices.forEach((point, idx) => {
      vertices[idx * 3] = point[0];
      vertices[idx * 3 + 1] = point[1];
      vertices[idx * 3 + 2] = point[2];
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geometry.setIndex(model.footprintMesh.indices);
    geometry.computeVertexNormals();

    const mesh = new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({
        color: 0x58d68d,
        transparent: true,
        opacity: 0.48,
        side: THREE.DoubleSide,
        roughness: 0.92,
        metalness: 0.02,
      }),
    );
    mesh.name = 'extended-visible-floor-plane';
    group.add(mesh);

    const edgeVertices = new Float32Array(model.footprintMesh.boundarySegments.length * 2 * 3);
    model.footprintMesh.boundarySegments.forEach(([start, end], idx) => {
      const base = idx * 6;
      edgeVertices[base] = start[0];
      edgeVertices[base + 1] = start[1];
      edgeVertices[base + 2] = start[2];
      edgeVertices[base + 3] = end[0];
      edgeVertices[base + 4] = end[1];
      edgeVertices[base + 5] = end[2];
    });
    const edgeGeometry = new THREE.BufferGeometry();
    edgeGeometry.setAttribute('position', new THREE.BufferAttribute(edgeVertices, 3));
    const edge = new THREE.LineSegments(
      edgeGeometry,
      new THREE.LineBasicMaterial({ color: 0xe9fff3, transparent: true, opacity: 0.95 }),
    );
    edge.name = 'extended-visible-floor-plane-outline';
    group.add(edge);

    if (model.supportPoints.length) {
      const support = new Float32Array(model.supportPoints.length * 3);
      model.supportPoints.forEach((p, idx) => {
        support[idx * 3] = p[0];
        support[idx * 3 + 1] = p[1];
        support[idx * 3 + 2] = p[2];
      });
      const supportGeometry = new THREE.BufferGeometry();
      supportGeometry.setAttribute('position', new THREE.BufferAttribute(support, 3));
      const supportPoints = new THREE.Points(
        supportGeometry,
        new THREE.PointsMaterial({
          color: 0x8bc7ff,
          size: 0.045,
          sizeAttenuation: true,
          transparent: true,
          opacity: 0.9,
        }),
      );
      supportPoints.name = 'visible-floor-support-pixels';
      group.add(supportPoints);
    }

    const axes = new THREE.AxesHelper(Math.max(0.45, Math.min(model.footprint.widthM, model.footprint.depthM) * 0.12));
    axes.name = 'camera-local-axes';
    group.add(axes);

    scene.add(group);
    groupRef.current = group;
    frameModel(group);
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
      80,
    );
    camera.position.set(2, 1.4, -2);
    cameraRef.current = camera;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controlsRef.current = controls;

    const ambient = new THREE.AmbientLight(0xffffff, 0.45);
    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(-4, 7, -5);
    const rim = new THREE.DirectionalLight(0xffffff, 0.45);
    rim.position.set(4, 4, 5);
    scene.add(ambient);
    scene.add(key);
    scene.add(rim);

    const renderLoop = () => {
      if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;
      controlsRef.current?.update();
      rendererRef.current.render(sceneRef.current, cameraRef.current);
      animRef.current = requestAnimationFrame(renderLoop);
    };
    renderLoop();

    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      clearModel();
      controlsRef.current?.dispose();
      rendererRef.current?.dispose();
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
      const w = Math.max(10, container.clientWidth || 1);
      const h = Math.max(10, container.clientHeight || 1);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
    };
    resize();
    const obs = new ResizeObserver(resize);
    obs.observe(container);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    buildModel();
  }, [buildModel]);

  return <div className="floor-plane-view" ref={containerRef} />;
};

export default FloorPlane3DView;
