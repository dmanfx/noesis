/// <reference lib="webworker" />
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { sha256Hex } from '../lib/sha256';
import { rasterizePcfCamera, validatePcfCameraMetadata, type PcfCameraProjection } from '../lib/pcfCameraProjection';

declare const self: DedicatedWorkerGlobalScope;

async function readBounded(response: Response, limit: number) {
  if (!response.ok) throw new Error(`pcf_camera_http_${response.status}`);
  const reader = response.body?.getReader(); if (!reader) throw new Error('pcf_camera_body_unavailable');
  const chunks: Uint8Array[] = []; let count = 0;
  try {
    while (true) { const { done, value } = await reader.read(); if (done) break; count += value.byteLength; if (count > limit) throw new Error('pcf_camera_transfer_limit'); chunks.push(value); }
  } finally { await reader.cancel().catch(() => {}); }
  const bytes = new Uint8Array(count); let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}

self.onmessage = async ({ data }: MessageEvent<{ projection: PcfCameraProjection; origin: string }>) => {
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 30_000);
  try {
    const { projection: p } = data;
    if (data.origin !== self.location.origin) throw new Error('pcf_camera_origin_invalid');
    const response = await fetch(`${data.origin}/api/v1/scene-priors/cameras/${encodeURIComponent(p.camera)}`, { signal: controller.signal, cache: 'no-store' });
    const metadata = JSON.parse(new TextDecoder().decode(await readBounded(response, 256 * 1024)));
    const artifact = validatePcfCameraMetadata(metadata, p);
    const artifactResponse = await fetch(new URL(artifact.url, data.origin), { signal: controller.signal, cache: 'no-store' });
    if (artifactResponse.headers.get('X-Noesis-Artifact-Sha256') !== artifact.sha256) throw new Error('pcf_camera_artifact_header_mismatch');
    const payload = await readBounded(artifactResponse, artifact.sizeBytes);
    if (payload.byteLength !== artifact.sizeBytes || sha256Hex(payload) !== artifact.sha256) throw new Error('pcf_camera_artifact_digest_mismatch');
    // This immutable point artifact must be self-contained; no external GLTF resources.
    const view = new DataView(payload.buffer), jsonLength = view.getUint32(12, true);
    if (view.getUint32(0, true) !== 0x46546c67 || view.getUint32(4, true) !== 2 || view.getUint32(8, true) !== payload.byteLength || jsonLength > payload.byteLength - 20) throw new Error('pcf_camera_glb_invalid');
    const manifest = JSON.parse(new TextDecoder().decode(payload.subarray(20, 20 + jsonLength)));
    if (manifest.images?.length || manifest.buffers?.some((buffer: any) => buffer.uri)) throw new Error('pcf_camera_glb_external_resource');
    const gltf = await new GLTFLoader().parseAsync(payload.buffer, '');
    gltf.scene.updateMatrixWorld(true);
    let count = 0;
    gltf.scene.traverse(child => { if (child instanceof THREE.Points) count += child.geometry.getAttribute('position')?.count ?? 0; });
    if (!count || count > 1_000_000) throw new Error('pcf_camera_point_limit');
    const positions = new Float32Array(count * 3), colors = new Uint8Array(count * 3), point = new THREE.Vector3(), color = new THREE.Color(); let index = 0, hasColor = true;
    gltf.scene.traverse(child => {
      if (!(child instanceof THREE.Points)) return;
      const attribute = child.geometry.getAttribute('position'), rgb = child.geometry.getAttribute('color');
      for (let n = 0; n < attribute.count; n++, index += 3) {
        point.fromBufferAttribute(attribute, n).applyMatrix4(child.matrixWorld); positions.set(point.toArray(), index);
        if (rgb) { color.fromBufferAttribute(rgb, n).convertLinearToSRGB(); colors.set([Math.round(color.r * 255), Math.round(color.g * 255), Math.round(color.b * 255)], index); }
        else hasColor = false;
      }
      child.geometry.dispose();
      const materials = Array.isArray(child.material) ? child.material : [child.material]; materials.forEach(material => material.dispose());
    });
    const raster = rasterizePcfCamera(positions, colors, p);
    self.postMessage({ ok: true, priorId: p.priorId, hasColor, ...raster }, { transfer: [raster.depth.buffer, raster.rgb.buffer, raster.normals.buffer] });
  } catch (error) { self.postMessage({ ok: false, error: error instanceof Error ? error.message : String(error) }); }
  finally { clearTimeout(timer); }
};
