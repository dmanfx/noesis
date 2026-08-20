import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { framePerspectiveBounds } from '../lib/threeViewFraming';
import { CAMERA_GROUND_OBLIQUE_VIEW_DIRECTION } from '../lib/cameraGroundPresentation.js';

type ScenePriorCameraView = {
  prior_id?: string;
  space_id?: string;
  source_type?: string;
  source_model?: string;
  world_to_camera_local_row_major?: number[][];
  quality?: { selected_point_count?: number };
  artifact_urls?: { points_glb?: string };
  points?: { sha256?: string; size_bytes?: number };
};

type ScenePriorPointCloud3DViewProps = {
  cameraId: string;
  expectedPriorId?: string | null;
  calibrationEpoch?: number;
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

function parseTransform(rows?: number[][]): THREE.Matrix4 {
  if (
    !Array.isArray(rows)
    || rows.length !== 4
    || rows.some((row) => !Array.isArray(row) || row.length !== 4)
  ) {
    throw new Error('scene_prior_transform_invalid');
  }
  const flat = rows.flat().map(Number);
  if (flat.length !== 16 || flat.some((value) => !Number.isFinite(value))) {
    throw new Error('scene_prior_transform_invalid');
  }
  return new THREE.Matrix4().set(...flat as [
    number, number, number, number,
    number, number, number, number,
    number, number, number, number,
    number, number, number, number,
  ]);
}

export default function ScenePriorPointCloud3DView({
  cameraId,
  expectedPriorId,
  calibrationEpoch = 0,
  onCanvasReady,
}: ScenePriorPointCloud3DViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const groupRef = useRef<THREE.Group | null>(null);
  const animationRef = useRef<number | null>(null);
  const [status, setStatus] = useState('Loading conditioned room cloud…');

  const clearModel = useCallback(() => {
    const group = groupRef.current;
    if (!group) return;
    sceneRef.current?.remove(group);
    disposeObject(group);
    groupRef.current = null;
  }, []);

  const frameModel = useCallback((object: THREE.Object3D) => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;
    const bounds = new THREE.Box3().setFromObject(object);
    if (bounds.isEmpty()) return;
    // Frame the unmodified camera-local geometry from below the ground plane.
    // That camera basis, unlike a negative model scale, keeps +X screen-right
    // and +Z screen-up without mirroring metric landmarks.
    framePerspectiveBounds({
      bounds,
      camera,
      controls,
      viewDirection: new THREE.Vector3(...CAMERA_GROUND_OBLIQUE_VIEW_DIRECTION),
      margin: 1.12,
    });
  }, []);

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
    scene.fog = new THREE.Fog(0x0d1320, 14, 42);
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
    const scene = sceneRef.current;
    if (!scene || !cameraId) return;
    const controller = new AbortController();
    let cancelled = false;
    setStatus('Loading conditioned room cloud…');
    clearModel();

    const load = async () => {
      const encodedCamera = encodeURIComponent(cameraId);
      const metadataResponse = await fetch(
        `/api/v1/scene-priors/cameras/${encodedCamera}`,
        { signal: controller.signal, cache: 'no-store' },
      );
      if (!metadataResponse.ok) {
        throw new Error(`scene_prior_metadata_http_${metadataResponse.status}`);
      }
      const metadata = await metadataResponse.json() as ScenePriorCameraView;
      if (expectedPriorId && metadata.prior_id !== expectedPriorId) {
        throw new Error('scene_prior_identity_mismatch');
      }
      const pointsUrl = metadata.artifact_urls?.points_glb;
      if (!pointsUrl || !pointsUrl.startsWith('/api/v1/scene-priors/')) {
        throw new Error('scene_prior_points_url_invalid');
      }
      const artifactResponse = await fetch(pointsUrl, {
        signal: controller.signal,
        cache: 'no-store',
      });
      if (!artifactResponse.ok) {
        throw new Error(`scene_prior_points_http_${artifactResponse.status}`);
      }
      const expectedSha = String(metadata.points?.sha256 || '');
      const servedSha = String(
        artifactResponse.headers.get('X-Noesis-Artifact-Sha256') || '',
      );
      if (!expectedSha || servedSha !== expectedSha) {
        throw new Error('scene_prior_points_identity_mismatch');
      }
      const payload = await artifactResponse.arrayBuffer();
      if (cancelled) return;
      const expectedBytes = Number(metadata.points?.size_bytes);
      if (!Number.isSafeInteger(expectedBytes) || payload.byteLength !== expectedBytes) {
        throw new Error('scene_prior_points_size_mismatch');
      }

      const transform = parseTransform(metadata.world_to_camera_local_row_major);
      const gltf = await new Promise<Awaited<ReturnType<GLTFLoader['parseAsync']>>>((resolve, reject) => {
        new GLTFLoader().parse(payload, '', resolve, reject);
      });
      if (cancelled) {
        disposeObject(gltf.scene);
        return;
      }

      const group = new THREE.Group();
      group.name = 'conditioned-room-walk-cloud-camera-display';
      gltf.scene.applyMatrix4(transform);
      gltf.scene.traverse((child) => {
        if (!(child instanceof THREE.Points)) return;
        const previous = child.material;
        child.material = new THREE.PointsMaterial({
          vertexColors: Boolean(child.geometry.getAttribute('color')),
          color: 0xcfe8f6,
          size: 0.026,
          sizeAttenuation: true,
          transparent: true,
          opacity: 0.94,
          toneMapped: false,
        });
        if (Array.isArray(previous)) previous.forEach((material) => material.dispose());
        else previous?.dispose();
      });
      group.add(gltf.scene);

      const cloudBounds = new THREE.Box3().setFromObject(gltf.scene);
      if (cloudBounds.isEmpty()) throw new Error('scene_prior_points_empty');
      const size = cloudBounds.getSize(new THREE.Vector3());
      const center = cloudBounds.getCenter(new THREE.Vector3());
      const gridSize = Math.max(size.x, size.z, 1);
      const grid = new THREE.GridHelper(
        gridSize,
        Math.max(4, Math.min(80, Math.round(gridSize / 0.25))),
        0x6b7280,
        0x293241,
      );
      grid.position.set(center.x, 0, center.z);
      group.add(grid);
      group.add(new THREE.AxesHelper(Math.max(0.5, Math.min(gridSize * 0.12, 1.5))));

      clearModel();
      scene.add(group);
      groupRef.current = group;
      frameModel(group);
      const pointCount = Number(metadata.quality?.selected_point_count || 0);
      setStatus(
        `${pointCount.toLocaleString()} conditioned-fusion points · ${metadata.space_id || cameraId}`,
      );
    };

    load().catch((error) => {
      if (controller.signal.aborted || cancelled) return;
      setStatus(error instanceof Error ? error.message : String(error));
    });
    return () => {
      cancelled = true;
      controller.abort();
      clearModel();
    };
  }, [calibrationEpoch, cameraId, clearModel, expectedPriorId, frameModel]);

  return (
    <div className="calibrated-point-cloud-view scene-prior-point-cloud-view" ref={containerRef}>
      <div className="scene-prior-point-cloud-status">{status}</div>
    </div>
  );
}
