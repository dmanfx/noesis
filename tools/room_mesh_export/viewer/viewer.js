import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';

const el = (id) => document.getElementById(id);
const viewport = document.querySelector('.viewport');
const canvas = document.querySelector('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;
renderer.setClearColor(0x080d11);
renderer.localClippingEnabled = true;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 16/9, .03, 100);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = .12;
const normal = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide });
const wire = new THREE.MeshBasicMaterial({ color: 0xc0d5db, wireframe: true, side: THREE.DoubleSide });
const cutaway = new THREE.Plane(new THREE.Vector3(0, -1, 0), 1.8);
let root = null, asset = null, frame = null, benchmarking = false, cameras = {}, serial = 0;
let originals = new Map();
let metrics = { loaded: false };
const mb = (bytes) => `${(bytes / 1_000_000).toFixed(2)} MB`;

function requestRender() {
  if (benchmarking) return;
  if (frame === null) frame = requestAnimationFrame(render);
}
function render() {
  frame = null;
  if (benchmarking) return;
  const changed = controls.update();
  renderer.render(scene, camera);
  if (changed) requestRender();
}
controls.addEventListener('change', requestRender);
new ResizeObserver(() => {
  renderer.setSize(viewport.clientWidth, viewport.clientHeight, false);
  requestRender();
}).observe(viewport);

function resetCamera() {
  const row = cameras[el('camera').value];
  const cv = new THREE.Matrix4().set(...row.camera_to_world.flat());
  cv.multiply(new THREE.Matrix4().makeScale(1, -1, -1));
  camera.position.setFromMatrixPosition(cv);
  camera.quaternion.setFromRotationMatrix(cv);
  camera.up.setFromMatrixColumn(cv, 1).normalize();
  controls.target.copy(camera.position).addScaledVector(new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion), 3);
  controls.update();
  const n = .03, K = row.K, width = row.width, height = row.height;
  camera.projectionMatrix.makePerspective(-K[0][2]/K[0][0]*n, (width-K[0][2])/K[0][0]*n,
    K[1][2]/K[1][1]*n, -(height-K[1][2])/K[1][1]*n, n, 100);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  requestRender();
}
function overview() {
  if (!root) return;
  const box = new THREE.Box3().setFromObject(root), center = box.getCenter(new THREE.Vector3());
  camera.up.set(0, 1, 0);
  camera.position.copy(center).add(new THREE.Vector3(1, .8, -1.3).normalize().multiplyScalar(box.getSize(new THREE.Vector3()).length() * .85));
  camera.fov = 52; camera.aspect = 16/9; camera.updateProjectionMatrix();
  controls.target.copy(center); controls.update(); requestRender();
}
function changeMode() {
  const mode = el('mode').value;
  root?.traverse((node) => {
    if (!node.isMesh) return;
    node.material = mode === 'rgb' ? originals.get(node) : mode === 'normal' ? normal : wire;
    node.material.clippingPlanes = el('cutaway').checked ? [cutaway] : [];
  });
  requestRender();
}
function dispose(previous, materials) {
  if (!previous) return;
  scene.remove(previous);
  const textures = new Set(), uniqueMaterials = new Set();
  previous.traverse((node) => {
    if (!node.isMesh) return;
    node.geometry.dispose();
    const material = materials.get(node);
    for (const entry of Array.isArray(material) ? material : [material]) {
      if (!entry) continue;
      uniqueMaterials.add(entry);
      Object.values(entry).forEach((value) => value?.isTexture && textures.add(value));
    }
  });
  textures.forEach((texture) => { texture.image?.close?.(); texture.dispose(); });
  uniqueMaterials.forEach((material) => material.dispose());
}
async function load(selected) {
  const current = ++serial;
  metrics = { loaded: false, url: selected.url };
  el('loading').hidden = false;
  el('loading').textContent = `Loading ${selected.label}…`;
  el('error').textContent = '';
  el('asset').disabled = true;
  try {
    // Release the previous level before fetching the next; do not retain both GPU surfaces.
    dispose(root, originals); root = null; originals = new Map();
    const start = performance.now();
    const response = await fetch(selected.url);
    if (!response.ok) throw new Error(`Model download failed (${response.status})`);
    const bytes = await response.arrayBuffer();
    if (selected.sha256) {
      const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (b) => b.toString(16).padStart(2, '0')).join('');
      if (hash !== selected.sha256) throw new Error('The downloaded model differs from the trial manifest.');
    }
    const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
    const gltf = await loader.parseAsync(bytes, '');
    if (current !== serial) { dispose(gltf.scene, new Map()); return; }
    root = gltf.scene; scene.add(root); root.updateMatrixWorld(true); asset = selected;
    let triangles = 0, geometryBytes = 0, textureBytes = 0, draws = 0;
    const textures = new Set(), buffers = new Set();
    root.traverse((node) => {
      if (!node.isMesh) return;
      originals.set(node, node.material); draws++;
      triangles += (node.geometry.index?.count || node.geometry.attributes.position.count) / 3;
      const attributes = Object.values(node.geometry.attributes);
      if (node.geometry.index) attributes.push(node.geometry.index);
      attributes.forEach((attribute) => {
        const array = attribute.isInterleavedBufferAttribute ? attribute.data.array : attribute.array;
        if (!buffers.has(array)) { buffers.add(array); geometryBytes += array.byteLength; }
      });
      (Array.isArray(node.material) ? node.material : [node.material]).forEach((material) => {
        Object.values(material).forEach((value) => value?.isTexture && textures.add(value));
      });
    });
    textures.forEach((texture) => { textureBytes += texture.image.width * texture.image.height * 4 * (texture.generateMipmaps ? 4/3 : 1); texture.anisotropy = Math.min(4, renderer.capabilities.getMaxAnisotropy()); });
    changeMode(); resetCamera();
    metrics = { loaded: true, label: selected.label, url: selected.url, bytes: bytes.byteLength, triangles, draws,
      geometryBytes, textureBytes, textureCount: textures.size, loadMs: performance.now()-start,
      renderer: renderer.getContext().getParameter(renderer.getContext().RENDERER) };
    el('download-size').textContent = mb(bytes.byteLength);
    el('triangles').textContent = triangles.toLocaleString();
    el('draws').textContent = draws;
    el('gpu-memory').textContent = `${mb(textureBytes + geometryBytes)} est.`;
    el('load-time').textContent = `${Math.round(metrics.loadMs)} ms`;
    el('download').href = selected.url;
    el('compatible').href = selected.compatibility_url || selected.url;
    el('benchmark-result').textContent = '';
    el('loading').hidden = true;
  } catch (error) {
    el('error').textContent = error.message;
    el('loading').textContent = 'Room could not be loaded.';
    metrics = { loaded: false, error: error.message };
  } finally { el('asset').disabled = false; }
}
async function benchmark() {
  if (!root || benchmarking) return;
  benchmarking = true; el('benchmark').disabled = true;
  if (frame !== null) { cancelAnimationFrame(frame); frame = null; }
  const saved = camera.matrixWorld.clone(), projection = camera.projectionMatrix.clone();
  const target = controls.target.clone(), up = camera.up.clone();
  const offset = camera.position.clone().sub(target), frames = [];
  let begin = null, previous = null;
  await new Promise((resolve) => {
    const tick = (time) => {
      if (begin === null) begin = time;
      if (previous !== null) frames.push(time-previous);
      previous = time;
      camera.position.copy(offset).applyAxisAngle(camera.up, .18 * Math.sin((time-begin)/900)).add(target);
      camera.lookAt(target);
      renderer.render(scene, camera);
      if (time-begin < 3000) requestAnimationFrame(tick); else resolve();
    };
    requestAnimationFrame(tick);
  });
  benchmarking = false;
  frames.sort((a, b) => a-b);
  metrics.benchmark = { frameCount: frames.length, medianFrameMs: frames[Math.floor(frames.length*.5)],
    p95FrameMs: frames[Math.floor(frames.length*.95)], fps: 1000 / (frames.reduce((a,b)=>a+b, 0)/frames.length),
    interpretation: 'display_and_browser_frame_intervals_on_this_machine_not_cross_device_guarantee' };
  camera.position.setFromMatrixPosition(saved); camera.quaternion.setFromRotationMatrix(saved);
  camera.up.copy(up); controls.target.copy(target); controls.update();
  camera.projectionMatrix.copy(projection); camera.projectionMatrixInverse.copy(projection).invert();
  el('benchmark-result').textContent = `${metrics.benchmark.fps.toFixed(1)} fps · ${metrics.benchmark.p95FrameMs.toFixed(1)} ms p95 frame interval`;
  el('benchmark').disabled = false; requestRender();
}

el('mode').addEventListener('change', changeMode);
el('cutaway').addEventListener('change', changeMode);
el('camera').addEventListener('change', resetCamera);
el('reset').addEventListener('click', resetCamera);
el('overview').addEventListener('click', overview);
el('benchmark').addEventListener('click', benchmark);
const response = await fetch('/trial.json');
if (!response.ok) throw new Error('Trial manifest is unavailable.');
const trial = await response.json();
if (trial.title) { document.title = trial.title; document.querySelector('h1').textContent = trial.title; }
if (trial.asset_selector_label) el('asset-label').textContent = trial.asset_selector_label;
cameras = trial.cameras;
for (const item of trial.assets) el('asset').add(new Option(`${item.label} · ${mb(item.size_bytes)}`, item.id));
const names = { fixed: 'Reference camera', phone_0000: 'Walk start', phone_0032: 'Walk view 32', phone_0064: 'Walk view 64', phone_0096: 'Walk view 96', phone_0129: 'Walk end' };
for (const name of Object.keys(cameras)) el('camera').add(new Option(names[name] || name, name));
el('description').textContent = `${trial.dataset_label}. One colour atlas, one material, and selectable variants.`;
el('source-size-note').textContent = `The detailed source is ${mb(trial.source.mesh.size_bytes)}. It is loaded only when you choose this comparison.`;
el('source-note').textContent = `The existing surface includes ${(trial.inferred_area_fraction*100).toFixed(2)}% inferred area. Its geometry noise and missing capture detail remain part of this trial; exporting a compact mesh does not establish physical accuracy.`;
el('asset').value = trial.default_asset;
el('asset').addEventListener('change', () => load(trial.assets.find((item) => item.id === el('asset').value)));
el('source').addEventListener('click', () => load({ label: 'Original detailed surface', url: '/source.glb' }));
// Read-only inspection surface for the bounded browser validation.
window.roomMeshTrial = { get metrics() { return metrics; }, get rendererInfo() { return renderer.info; } };
await load(trial.assets.find((item) => item.id === trial.default_asset));
