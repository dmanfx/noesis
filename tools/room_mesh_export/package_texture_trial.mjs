#!/usr/bin/env node
// Publish only local preview assets for an exact same-geometry texture comparison.
import { createHash } from 'node:crypto';
import { constants } from 'node:fs';
import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, index, array) => {
  if (value.startsWith('--')) pairs.push([value.slice(2), array[index+1]]);
  return pairs;
}, []));
for (const required of ['trial-root', 'baseline-root', 'gltf-cli']) {
  if (!args[required]) throw new Error(`--${required} is required`);
}
const root = path.resolve(args['trial-root']), baseline = path.resolve(args['baseline-root']);
const cli = path.resolve(args['gltf-cli']);
const sourceTrial = JSON.parse(await readFile(path.join(baseline, 'trial.json'), 'utf8'));
async function record(filename) {
  const data = await readFile(filename);
  return { path: filename, size_bytes: data.length, sha256: createHash('sha256').update(data).digest('hex') };
}
function run(command) {
  const result = spawnSync(cli, command, { encoding: 'utf8', timeout: 120_000 });
  if (result.status !== 0 || result.error) throw new Error(result.error?.message || result.stderr);
  process.stdout.write(result.stdout);
}
const original = sourceTrial.assets.find((row) => row.id === 'detail');
const copied = [];
await mkdir(path.join(root, 'baseline'), { recursive: true });
for (const filename of ['room-meshopt.glb', 'room-compatible.glb']) {
  const from = path.join(baseline, 'detail', filename), to = path.join(root, 'baseline', filename);
  try { await copyFile(from, to, constants.COPYFILE_EXCL); }
  catch (error) { if (error.code !== 'EEXIST') throw error; }
  const actual = await record(to);
  const expected = filename === 'room-meshopt.glb' ? original : original.compatibility;
  if (actual.sha256 !== expected.sha256 || actual.size_bytes !== expected.size_bytes) throw new Error('Baseline copy binding mismatch');
  copied.push(actual);
}
const assets = [{ ...original, id: 'baseline', label: 'Original colour / original packing',
  url: '/baseline/room-meshopt.glb', compatibility_url: '/baseline/room-compatible.glb', compatibility: copied[1] }];
let geometry = null;
for (const [id, label] of [['packed', 'Packing test / narrow gutters'], ['packed-padded', 'Original colour / improved packing'], ['photo-best', 'Direct photographs / sharpest view'],
  ['photo-blend', 'Direct photographs / robust blend']]) {
  const directory = path.join(root, id), manifestFile = path.join(directory, 'mesh_manifest.json');
  let manifest;
  try { manifest = JSON.parse(await readFile(manifestFile, 'utf8')); }
  catch (error) { if (error.code === 'ENOENT') continue; throw error; }
  geometry ||= manifest.texture_refinement.world_geometry_sha256;
  if (manifest.texture_refinement.world_geometry_sha256 !== geometry
      || manifest.source.mesh.sha256 !== sourceTrial.source.mesh.sha256) throw new Error('Comparison mixes geometry or source');
  const input = path.join(directory, 'room.glb');
  const binding = await record(input);
  if (binding.sha256 !== manifest.output.sha256 || binding.size_bytes !== manifest.output.size_bytes) throw new Error(`${id}: input binding mismatch`);
  const web = path.join(directory, 'room-web.glb');
  const compatible = path.join(directory, 'room-web-compatible.glb'), compact = path.join(directory, 'room-web-meshopt.glb');
  const quantization = ['--quantize-position', '16', '--quantize-normal', '8', '--quantize-texcoord', '16'];
  // A new trial directory is required; never overwrite a completed variant.
  const recordFile = path.join(directory, 'renderer_asset_manifest.json');
  let cached;
  try { cached = JSON.parse(await readFile(recordFile, 'utf8')); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  if (cached) {
    const compactBinding = await record(compact), compatibleBinding = await record(compatible);
    if (cached.input_sha256 !== binding.sha256 || cached.sha256 !== compactBinding.sha256
        || cached.compatibility.sha256 !== compatibleBinding.sha256) throw new Error('Cached texture asset binding mismatch');
  } else {
    for (const output of [web, compatible, compact]) {
      try { await readFile(output); throw new Error(`Refusing to overwrite ${output}`); }
      catch (error) { if (error.code !== 'ENOENT') throw error; }
    }
    run(['jpeg', input, web, '--quality', '85']);
    run(['quantize', web, compatible, ...quantization]);
    run(['meshopt', web, compact, ...quantization, '--level', 'high']);
  }
  const final = await record(compact);
  const bytes = await readFile(compact);
  const document = JSON.parse(bytes.subarray(20, 20 + bytes.readUInt32LE(12)).toString());
  const primitives = document.meshes.flatMap((mesh) => mesh.primitives);
  if (primitives.length !== 1 || document.materials.length !== 1 || document.images.length !== 1) throw new Error('Variant exceeds one draw/material/texture budget');
  const triangles = document.accessors[primitives[0].indices].count / 3;
  if (triangles !== manifest.output.triangle_count) throw new Error('Packaging changed triangle count');
  const row = { id, label, edge: 4096, url: `/${id}/room-web-meshopt.glb`, compatibility_url: `/${id}/room-web-compatible.glb`,
    size_bytes: final.size_bytes, sha256: final.sha256, triangle_count: triangles, jpeg_quality: 85,
    compatibility: await record(compatible), extensions_required: document.extensionsRequired,
    world_geometry_sha256: geometry, surface_texture_fraction: manifest.texture_bake.fraction,
    direct_photo_fraction: manifest.texture_bake.direct_photo_fraction_of_covered ?? null, input_sha256: binding.sha256 };
  assets.push(row);
  if (!cached) await writeFile(recordFile, JSON.stringify(row, null, 2)+'\n', { flag: 'wx' });
  console.log(JSON.stringify({ id, bytes: final.size_bytes }));
}
if (!assets.some((row) => row.id === 'packed')) throw new Error('Completed packing trial required');
const trial = { ...sourceTrial, schema: 'noesis.living_room_texture_trial.v1', assets,
  default_asset: args['default-asset'] || 'baseline',
  dataset_label: 'Living Room texture comparison · same 400,000-triangle geometry / 4K atlas',
  title: 'Living Room · texture quality trial', asset_selector_label: 'Texture',
  interpretation: 'texture_only_same_world_geometry_no_new_depth_inference', world_geometry_sha256: geometry,
  runtime_publication: false };
if (!assets.some((row) => row.id === trial.default_asset)) throw new Error('Requested default is unavailable');
await writeFile(path.join(root, 'trial.json'), JSON.stringify(trial, null, 2)+'\n');
