import React, { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { HeightMapResponse } from '../types/heightMap';

const MAX_GRID_VERTICES = 256 * 256;

type ColorMode = 'height' | 'distance' | 'density' | 'none';
type GeometrySource = 'height' | 'distance';

export type Depth3DViewHandle = {
  setTopDownView: () => void;
  setIsometricView: () => void;
  resetView: () => void;
};

type Depth3DViewProps = {
  heightMap?: HeightMapResponse;
  exaggeration: number;
  objectThresholdM: number;
  showEdges: boolean;
  colorBy: ColorMode;
  geometrySource: GeometrySource;
  requestedGridResM?: number;
};

type DebugInfo = {
  cols: number;
  rows: number;
  gridResM: number;
  zMin: number;
  zMax: number;
  floorLevel: number;
  worldWidthM: number;
  worldDepthM: number;
};

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

function downsampleGrid(
  data: number[],
  srcCols: number,
  srcRows: number,
  targetCols: number,
  targetRows: number,
): Float32Array {
  const out = new Float32Array(targetCols * targetRows);
  const total = Math.min(data.length, srcCols * srcRows);
  for (let r = 0; r < targetRows; r += 1) {
    const y0 = Math.floor((r * srcRows) / targetRows);
    const y1 = Math.max(y0 + 1, Math.floor(((r + 1) * srcRows) / targetRows));
    for (let c = 0; c < targetCols; c += 1) {
      const x0 = Math.floor((c * srcCols) / targetCols);
      const x1 = Math.max(x0 + 1, Math.floor(((c + 1) * srcCols) / targetCols));
      let sum = 0;
      let count = 0;
      for (let sy = y0; sy < Math.min(srcRows, y1); sy += 1) {
        for (let sx = x0; sx < Math.min(srcCols, x1); sx += 1) {
          const idx = sy * srcCols + sx;
          if (idx >= total) continue;
          const v = data[idx];
          if (!Number.isFinite(v)) continue;
          sum += v;
          count += 1;
        }
      }
      out[r * targetCols + c] = count ? sum / count : Number.NaN;
    }
  }
  return out;
}

function finiteMinMax(values: ArrayLike<number>) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const hasFinite = Number.isFinite(min) && Number.isFinite(max);
  return {
    hasFinite,
    min: hasFinite ? min : 0,
    max: hasFinite ? max : 0,
  };
}

const Depth3DView = forwardRef<Depth3DViewHandle, Depth3DViewProps>(({
  heightMap,
  exaggeration,
  objectThresholdM,
  showEdges,
  colorBy,
  geometrySource,
  requestedGridResM,
}, ref) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const edgesRef = useRef<THREE.LineSegments | null>(null);
  const animRef = useRef<number | null>(null);
  const boundsRef = useRef<{ center: THREE.Vector3; radius: number }>({
    center: new THREE.Vector3(0, 0, 0),
    radius: 1,
  });
  const userControlledRef = useRef(false);
  const lastPayloadLogRef = useRef<string>('');
  const statsLogRef = useRef<string>('');
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);
  const showDebugOverlay = process.env.NODE_ENV === 'development';

  const disposeEdges = useCallback(() => {
    const edges = edgesRef.current;
    if (edges) {
      edges.geometry?.dispose();
      (edges.material as THREE.Material | undefined)?.dispose?.();
      if (sceneRef.current) {
        sceneRef.current.remove(edges);
      }
      edgesRef.current = null;
    }
  }, []);

  const disposeMesh = useCallback(() => {
    const mesh = meshRef.current;
    if (mesh) {
      mesh.geometry?.dispose();
      if (Array.isArray(mesh.material)) {
        mesh.material.forEach((m) => m.dispose());
      } else {
        (mesh.material as THREE.Material | undefined)?.dispose?.();
      }
      if (sceneRef.current) {
        sceneRef.current.remove(mesh);
      }
      meshRef.current = null;
    }
    disposeEdges();
  }, [disposeEdges]);

  const frameToBounds = useCallback(() => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const mesh = meshRef.current;
    if (!camera || !controls || !mesh) return;
    const box = new THREE.Box3().setFromObject(mesh);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    const maxDim = Math.max(size.x, size.y, size.z, 0.01);
    const radius = Math.max(0.6, maxDim * 0.75);
    const dist = radius * 2.4;
    camera.position.set(center.x + dist, center.y + radius * 1.1, center.z + dist);
    camera.near = Math.max(0.01, dist * 0.02);
    camera.far = Math.max(20, dist * 8);
    camera.updateProjectionMatrix();
    controls.target.copy(center);
    controls.update();
    boundsRef.current = { center, radius: dist };
  }, []);

  const setTopDownView = useCallback(() => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const bounds = boundsRef.current;
    if (!camera || !controls) return;
    userControlledRef.current = true;
    const dist = Math.max(bounds.radius, 0.5);
    camera.position.set(bounds.center.x, bounds.center.y + dist, bounds.center.z);
    camera.up.set(0, 0, 1);
    camera.lookAt(bounds.center);
    controls.target.copy(bounds.center);
    controls.update();
  }, []);

  const setIsometricView = useCallback(() => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const bounds = boundsRef.current;
    if (!camera || !controls) return;
    userControlledRef.current = true;
    const dist = Math.max(bounds.radius, 0.8);
    camera.position.set(bounds.center.x + dist * 0.8, bounds.center.y + dist * 0.9, bounds.center.z + dist * 0.8);
    camera.up.set(0, 1, 0);
    camera.lookAt(bounds.center);
    controls.target.copy(bounds.center);
    controls.update();
  }, []);

  useImperativeHandle(ref, () => ({
    setTopDownView,
    setIsometricView,
    resetView: () => {
      userControlledRef.current = false;
      frameToBounds();
    },
  }), [frameToBounds, setIsometricView, setTopDownView]);

  useEffect(() => {
    if (!heightMap) return;
    const logKey = String(
      heightMap.request_id ||
      heightMap.ts ||
      heightMap.meta?.generated_at ||
      `${heightMap.camera_id || heightMap.camera || 'cam'}-${heightMap.width}x${heightMap.height}`
    );
    if (lastPayloadLogRef.current === logKey) return;
    lastPayloadLogRef.current = logKey;
    const data = heightMap.data ?? [];
    let dataMin: number | null = null;
    let dataMax: number | null = null;
    for (let i = 0; i < data.length; i += 1) {
      const v = data[i];
      if (!Number.isFinite(v)) continue;
      if (dataMin === null || v < dataMin) dataMin = v;
      if (dataMax === null || v > dataMax) dataMax = v;
    }
    // eslint-disable-next-line no-console
    console.log('Depth3DView: height map payload', {
      width: heightMap.width,
      height: heightMap.height,
      units: heightMap.units,
      z_offset: heightMap.z_offset,
      z_min: heightMap.z_min,
      z_max: heightMap.z_max,
      grid_res_m: heightMap.grid_res_m,
      max_extent_m: heightMap.max_extent_m,
      room_bbox_m: heightMap.meta?.room_bbox_m,
      dataMin,
      dataMax,
    });
    if (process.env.NODE_ENV === 'development') {
      try {
        // eslint-disable-next-line no-console
        console.debug('Depth3DView: map meta', { served_from_cache: heightMap.served_from_cache, meta: heightMap.meta });
      } catch {
        // ignore
      }
    }
  }, [heightMap]);

  const buildMesh = useCallback(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    setDebugInfo(null);

    if (!heightMap || !heightMap.width || !heightMap.height) {
      return;
    }

    const cols = Math.max(1, Math.floor(heightMap.width));
    const rows = Math.max(1, Math.floor(heightMap.height));

    const heightData = heightMap.data ?? [];
    if (!heightData.length) {
      // eslint-disable-next-line no-console
      console.warn('Depth3DView: missing height data in HeightMapResponse');
    }

    const geometryValues = geometrySource === 'distance'
      ? (heightMap.distance?.data?.length ? heightMap.distance.data : heightData)
      : heightData;

    const total = Math.min(geometryValues.length, cols * rows);
    if (!total) {
      return;
    }

    const finiteCount = geometryValues.reduce((acc, v) => acc + (Number.isFinite(v as number) ? 1 : 0), 0);
    if (!finiteCount) {
      // eslint-disable-next-line no-console
      console.warn('Depth3DView: no finite height samples, keeping previous mesh');
      return;
    }

    const effectiveGridRes = heightMap.grid_res_m ?? requestedGridResM ?? 0.1;
    const worldWidthM = cols * effectiveGridRes;
    const worldDepthM = rows * effectiveGridRes;

    const factor = cols * rows > MAX_GRID_VERTICES ? Math.ceil(Math.sqrt((cols * rows) / MAX_GRID_VERTICES)) : 1;
    const targetCols = Math.max(2, Math.floor(cols / factor));
    const targetRows = Math.max(2, Math.floor(rows / factor));

    const sampledHeights = downsampleGrid(geometryValues, cols, rows, targetCols, targetRows);
    const densityLayer = heightMap.density;
    const distanceLayer = heightMap.distance;
    const sampledDensity = densityLayer?.data?.length
      ? downsampleGrid(densityLayer.data, densityLayer.width ?? cols, densityLayer.height ?? rows, targetCols, targetRows)
      : null;
    const sampledDistance = distanceLayer?.data?.length
      ? downsampleGrid(distanceLayer.data, distanceLayer.width ?? cols, distanceLayer.height ?? rows, targetCols, targetRows)
      : null;

    const validHeights = heightData.filter((v) => Number.isFinite(v));
    validHeights.sort((a, b) => a - b);
    const floorPercentile = 0.1;
    const floorIndex = Math.floor(validHeights.length * floorPercentile);
    const floorLevel = validHeights.length ? validHeights[Math.min(floorIndex, validHeights.length - 1)] : 0;

    const zMin = typeof heightMap.z_min === 'number' ? heightMap.z_min : floorLevel;
    const zMax = typeof heightMap.z_max === 'number'
      ? heightMap.z_max
      : (validHeights.length ? validHeights[validHeights.length - 1] : (zMin + 1));
    const dataRange = finiteMinMax(heightData);

    const geometry = new THREE.PlaneGeometry(
      worldWidthM,
      worldDepthM,
      targetCols - 1,
      targetRows - 1,
    );
    const positionAttr = geometry.getAttribute('position') as THREE.BufferAttribute;

    let colorField: Float32Array | null = null;
    if (colorBy === 'height') {
      colorField = sampledHeights;
    } else if (colorBy === 'distance' && sampledDistance) {
      colorField = sampledDistance;
    } else if (colorBy === 'density' && sampledDensity) {
      colorField = sampledDensity;
    }
    const { hasFinite: hasColorFinite, min: colorMin, max: colorMax } = colorField ? finiteMinMax(colorField) : { hasFinite: false, min: 0, max: 1 };
    const colorRange = Math.max(1e-6, colorMax - colorMin);
    const colors = colorField ? new Float32Array(positionAttr.count * 3) : null;
    const color = new THREE.Color();

    for (let r = 0; r < targetRows; r += 1) {
      for (let c = 0; c < targetCols; c += 1) {
        const idx = r * targetCols + c;
        const h = sampledHeights[idx];
        let y = 0;
        if (Number.isFinite(h)) {
          const relative = Math.max(0, h - floorLevel);
          const thresholded = relative < objectThresholdM ? 0 : relative;
          y = thresholded * exaggeration;
        }
        positionAttr.setZ(idx, y);

        if (colors && colorField) {
          const v = colorField[idx];
          const t = hasColorFinite ? clamp01((v - colorMin) / colorRange) : 0.5;
          color.setRGB(0.3 + 0.7 * t, 0.3 + 0.7 * t, 0.4 + 0.6 * t);
          colors[idx * 3] = color.r;
          colors[idx * 3 + 1] = color.g;
          colors[idx * 3 + 2] = color.b;
        }
      }
    }

    positionAttr.needsUpdate = true;
    if (colors) {
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    } else {
      geometry.deleteAttribute('color');
    }
    geometry.computeVertexNormals();
    geometry.rotateX(-Math.PI / 2);

    disposeMesh();

    const material = new THREE.MeshStandardMaterial({
      color: 0xb8beca,
      flatShading: true,
      roughness: 0.95,
      metalness: 0.05,
      vertexColors: !!colors,
    });

    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);
    meshRef.current = mesh;

    if (showEdges) {
      const edges = new THREE.EdgesGeometry(geometry);
      const line = new THREE.LineSegments(
        edges,
        new THREE.LineBasicMaterial({ color: 0x111318, linewidth: 1 })
      );
      line.position.copy(mesh.position);
      scene.add(line);
      edgesRef.current = line;
    } else {
      disposeEdges();
    }

    if (!userControlledRef.current) {
      frameToBounds();
    }

    if (showDebugOverlay) {
      setDebugInfo({
        cols: targetCols,
        rows: targetRows,
        gridResM: effectiveGridRes,
        zMin,
        zMax,
        floorLevel,
        worldWidthM,
        worldDepthM,
      });
    }

    const statsKey = String(heightMap.request_id || heightMap.ts || heightMap.meta?.generated_at || `${heightMap.camera_id || heightMap.camera || ''}-${heightMap.width}x${heightMap.height}`);
    if (process.env.NODE_ENV === 'development' && statsLogRef.current !== statsKey) {
      // eslint-disable-next-line no-console
      console.log('Height map stats', {
        cols: targetCols,
        rows: targetRows,
        gridResM: effectiveGridRes,
        zMin,
        zMax,
        floorLevel,
        dataMin: dataRange.hasFinite ? dataRange.min : null,
        dataMax: dataRange.hasFinite ? dataRange.max : null,
      });
      statsLogRef.current = statsKey;
    }
  }, [colorBy, disposeEdges, disposeMesh, exaggeration, frameToBounds, geometrySource, heightMap, objectThresholdM, requestedGridResM, showDebugOverlay, showEdges]);

  useEffect(() => {
    if (rendererRef.current || !containerRef.current) return;
    const container = containerRef.current;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(container.clientWidth || 1, container.clientHeight || 1, false);
    rendererRef.current = renderer;
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0f16);
    scene.fog = new THREE.Fog(0x0a0f16, 10, 28);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      55,
      (container.clientWidth || 1) / Math.max(1, container.clientHeight || 1),
      0.01,
      80,
    );
    camera.position.set(1.8, 1.6, 1.8);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0);
    controlsRef.current = controls;
    const handleStart = () => {
      userControlledRef.current = true;
    };
    controls.addEventListener('start', handleStart);

    const ambient = new THREE.AmbientLight(0xffffff, 0.35);
    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(-4, 8, 6);
    key.target.position.set(0, 0, 0);
    const rim = new THREE.DirectionalLight(0xffffff, 0.4);
    rim.position.set(4, 6, -6);
    scene.add(ambient);
    scene.add(key);
    scene.add(key.target);
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
      disposeMesh();
      controlsRef.current?.dispose();
      rendererRef.current?.dispose();
      if (rendererRef.current?.domElement && rendererRef.current.domElement.parentNode === container) {
        container.removeChild(rendererRef.current.domElement);
      }
      sceneRef.current = null;
      cameraRef.current = null;
      controlsRef.current = null;
      rendererRef.current = null;
    };
  }, [disposeMesh]);

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
    buildMesh();
  }, [buildMesh]);

  return (
    <div className="depth3d-view" ref={containerRef}>
      {showDebugOverlay && debugInfo && (
        <div className="depth3d-debug">
          <div>Grid: {debugInfo.cols} × {debugInfo.rows}</div>
          <div>Res: {debugInfo.gridResM.toFixed(2)} m</div>
          <div>Floor: {debugInfo.floorLevel.toFixed(2)} m</div>
          <div>Z: {debugInfo.zMin.toFixed(2)}–{debugInfo.zMax.toFixed(2)} m</div>
          <div>Span: {debugInfo.worldWidthM.toFixed(1)} m × {debugInfo.worldDepthM.toFixed(1)} m</div>
        </div>
      )}
    </div>
  );
});

Depth3DView.displayName = 'Depth3DView';

export default Depth3DView;
