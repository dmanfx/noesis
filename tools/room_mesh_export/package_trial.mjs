#!/usr/bin/env node
// Package completed offline atlas exports into explicit compatibility/compact levels.
import { createHash } from 'node:crypto';
import { readFile, writeFile, access } from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, index, array) => {
  if (value.startsWith('--')) pairs.push([value.slice(2), array[index + 1]]);
  return pairs;
}, []));
for (const required of ['trial-root', 'gltf-cli', 'camera-manifest']) {
  if (!args[required]) throw new Error(`--${required} is required`);
}
const root = path.resolve(args['trial-root']);
const cli = path.resolve(args['gltf-cli']);
const presets = [
  { id: 'preview', label: 'Overview', edge: 1024 },
  { id: 'balanced', label: 'Balanced', edge: 2048 },
  { id: 'detail', label: 'High detail', edge: 4096 },
];
if (args['include-photo'] === 'true') presets.push({ id: 'photo-blend', label: 'Direct photographs / 4K blend', edge: 4096 });
async function record(filename) {
  const bytes = await readFile(filename);
  return { path: filename, size_bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') };
}
async function exists(filename) { try { await access(filename); return true; } catch { return false; } }
function run(command) {
  const result = spawnSync(cli, command, { encoding: 'utf8', timeout: 120_000 });
  if (result.status !== 0 || result.error) throw new Error(`${command[0]} failed: ${result.error?.message || result.stderr}`);
  process.stdout.write(result.stdout);
}
function header(bytes) {
  if (bytes.readUInt32LE(0) !== 0x46546c67 || bytes.readUInt32LE(4) !== 2) throw new Error('Invalid packaged GLB');
  return JSON.parse(bytes.subarray(20, 20 + bytes.readUInt32LE(12)).toString());
}
const assets = [];
let source = null;
for (const preset of presets) {
  const directory = path.join(root, preset.id);
  const manifestFile = path.join(directory, 'mesh_manifest.json');
  if (!await exists(manifestFile)) continue;
  const manifest = JSON.parse(await readFile(manifestFile, 'utf8'));
  const input = path.join(directory, 'room.glb');
  const actual = await record(input);
  if (actual.sha256 !== manifest.output.sha256 || actual.size_bytes !== manifest.output.size_bytes) {
    throw new Error(`${preset.id}: atlas export binding mismatch`);
  }
  if (source && source.source.mesh.sha256 !== manifest.source.mesh.sha256) throw new Error('Levels use different source meshes');
  source ||= manifest;
  const resized = path.join(directory, 'room-resized.glb');
  const compatible = path.join(directory, 'room-compatible.glb');
  const compact = path.join(directory, 'room-meshopt.glb');
  const assetManifestFile = path.join(directory, 'renderer_asset_manifest.json');
  if (await exists(assetManifestFile)) {
    const cached = JSON.parse(await readFile(assetManifestFile, 'utf8'));
    const compactBytes = await record(compact), compatibleBytes = await record(compatible);
    if (cached.source_glb_sha256 !== actual.sha256 || cached.edge !== preset.edge
      || cached.sha256 !== compactBytes.sha256 || cached.size_bytes !== compactBytes.size_bytes
      || cached.compatibility.sha256 !== compatibleBytes.sha256
      || cached.compatibility.size_bytes !== compatibleBytes.size_bytes) {
      throw new Error(`${preset.id}: packaged asset binding mismatch`);
    }
    assets.push(cached);
    continue;
  }
  const quantization = ['--quantize-position', '16', '--quantize-normal', '8', '--quantize-texcoord', '16'];
  const commands = [
    ['resize', input, resized, '--width', String(preset.edge), '--height', String(preset.edge)],
    ['quantize', resized, compatible, ...quantization],
    ['meshopt', resized, compact, ...quantization, '--level', 'high'],
  ];
  for (const command of commands) {
    if (await exists(command[2])) throw new Error(`Refusing to overwrite packaged asset: ${command[2]}`);
    run(command);
  }
  const compactRecord = await record(compact), compatibility = await record(compatible);
  const doc = header(await readFile(compact));
  const primitives = doc.meshes.flatMap((mesh) => mesh.primitives);
  if (primitives.length !== 1 || doc.materials.length !== 1 || doc.images.length !== 1) throw new Error('Packaging expanded material/draw counts');
  const row = { ...preset, source_glb_sha256: actual.sha256,
    url: `/${preset.id}/room-meshopt.glb`, compatibility_url: `/${preset.id}/room-compatible.glb`,
    size_bytes: compactRecord.size_bytes, sha256: compactRecord.sha256,
    triangle_count: doc.accessors[primitives[0].indices].count / 3,
    compatibility, extensions_required: doc.extensionsRequired || [],
    export_geometry_error: manifest.export_geometry_error,
    estimated_texture_bytes_rgba_with_mips: Math.round(preset.edge ** 2 * 4 * 4 / 3),
    packaging_commands: commands, gltf_transform_version: '4.5.0' };
  assets.push(row);
  await writeFile(assetManifestFile, JSON.stringify(row, null, 2)+'\n', { flag: 'wx' });
  console.log(JSON.stringify({ id: row.id, bytes: row.size_bytes, compatibility_bytes: compatibility.size_bytes, triangles: row.triangle_count }));
}
if (!source || !assets.some((asset) => asset.id === 'balanced')) throw new Error('Completed balanced export is required');
const cameraBytes = await readFile(args['camera-manifest']);
const originalCameras = JSON.parse(cameraBytes).cameras;
const cameras = Object.fromEntries(Object.entries(originalCameras).map(([name, row]) => [name,
  { K: row.K, camera_to_world: row.camera_to_world, width: row.width, height: row.height }]));
const trial = { schema: 'noesis.living_room_renderer_trial.v1', default_asset: args['default-asset'] || 'balanced',
  dataset_label: args['dataset-label'] || '130-view Living Room fusion · September 12 capture / October 4 reconstruction',
  inferred_area_fraction: source.inherited_source_inferred_area_fraction,
  coordinate_frame: source.coordinate_frame, source_scan_id: source.source_scan_id,
  source: source.source, camera_manifest: await record(path.resolve(args['camera-manifest'])),
  assets, cameras, runtime_publication: false };
if (!assets.some((asset) => asset.id === trial.default_asset)) throw new Error('Requested default asset is unavailable');
if (args.title) trial.title = args.title;
await writeFile(path.join(root, 'trial.json'), JSON.stringify(trial, null, 2)+'\n');
