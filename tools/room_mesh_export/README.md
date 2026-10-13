# Browser-sized Living Room mesh trial

Status: completed offline presentation trial, 2026-10-07. This is independent
of the RoomWalk utility and is not a live Noesis or Menon scene replacement.

The trial derives compact Three.js assets from the retained 130-view Living
Room reconstruction: September 12 capture, October 4 phone-only consensus
rebuild and source-specific floor alignment, then the October 5 closed,
continuous-video-colour surface. It does not rerun inference or add geometry.

The subsequent [same-geometry texture comparison](TEXTURE_TRIAL.md) tests denser
4K atlas packing and direct sampling of the existing 2560-pixel photographs.
Its separate preview defaults to a robust photo blend; the original three-level
mesh trial and its balanced default remain unchanged.

## Source and authority

The input is `final_closed_video_color_v2/surface_mesh.glb`, bound to its
adjacent `surface_mesh_manifest.json`. Its SHA-256 is
`c9bf7bc33522bcf48d7b27aa93339bcba206314546b60d25f8b3410d17f60052`:
74,343,880 bytes, 2,867,524 triangles and 1,426,138 vertices. The manifest binds
the October 4 `prior_conditioned_consensus_da3_carrier` source and
`alignment_consensus_floor_v2` transform. Later timestamps are not sufficient
to select a replacement reconstruction: the later joint comparison was not
selected as better geometry, and the September 22 DA3 result was not established
as a preferred replacement fused PCF.

The input already uses `backend_world_m_stream_points`. Keep its coordinates
unchanged; do not apply the phone-to-world alignment again. Menon presentation
would additionally require the correct target-revision world-to-scene transform
exactly once. This trial does not authorize that integration.

The source is `review_only` and `accepted_for_canonical_use=false`. About
23.64% of its surface area is existing inferred completion. The export inherits
that completion, rough geometry, colour noise and unassessed self-intersections.
Neither compression nor a closed-looking mesh establishes physical accuracy,
measured completeness or live tracking authority. See the
[reconstruction fitting guide](../../docs/room_reconstruction_fitting.md) and
[existing dense/completed surface workflow](../mapanything_phone_scan/README.md#dense-and-explicitly-completed-room-surfaces-offline).

## Delivered levels

Sizes are decimal MB. Only the selected level is fetched; these are selectable
detail levels, not simultaneous downloads or an automatic streaming system.

| Level | Triangles | Colour atlas | Compact GLB | Loader-compatible GLB | Estimated decoded geometry + texture |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overview | 40,000 | 1024² | 0.92 MB | 1.48 MB | 6.71 MB |
| Balanced, default | 150,000 | 2048² | 3.04 MB | 5.91 MB | 26.98 MB |
| High detail | 400,000 | 4096² | 8.24 MB | 15.58 MB | 100.80 MB |

Each asset has one embedded JPEG atlas, one unlit material and one draw call.
Compact files require `EXT_meshopt_compression`; both variants use
`KHR_mesh_quantization`. The compatible balanced file loaded through Menon's
existing `BoundedGltfLoader.js` with `embeddedImages: true`, without modifying
Menon. High-detail texture memory remains substantial despite its small download.

Quadratic-error simplification preserves the source's dense colour field by
baking closest-source-triangle barycentric linear RGB into an sRGB UV atlas.
Spatial UV patches bound chart-solving work; an exact corner/winding assertion
ensures unwrap adds only seams, not geometry changes. Small ambiguous UV samples
remain neutral grey, with counts recorded; ambiguity over 0.1% rejects the
layout. Padding is restricted to outside-chart pixels. There is no inpainting.

Area-weighted, deterministic 100,000-point samples in each direction measured
the following **simplification loss before final quantization**, relative to
the saved dense mesh, not to the physical room:

| Level | Simplified → source p95 | Source → simplified p95 | Largest sampled source → simplified distance |
| --- | ---: | ---: | ---: |
| Overview | 18.16 mm | 22.27 mm | 889.73 mm |
| Balanced | 6.92 mm | 7.17 mm | 67.33 mm |
| High detail | 2.82 mm | 2.83 mm | 18.41 mm |

Overview removes some tiny isolated source components and is not a close-up
detail substitute. These samples are neither worst-case bounds nor independent
physical-accuracy measurements.

## Reproduce the offline export

Run from the repository root. Use a new external trial directory and explicit
source bindings. Keep dependencies local to that directory rather than changing
the shared RoomWalk or Noesis environment. The selected Python environment must
already supply NumPy, Open3D, SciPy, Pillow and pytest; the existing phone-scan
environment supplies them for this trial.

```bash
export ROOM_MESH_ROOT=/path/to/new-room-mesh-trial
export ROOM_MESH_SOURCE_DIR=/path/to/final_closed_video_color_v2
export ROOM_MESH_CAMERA_MANIFEST=/path/to/render_final_video_color/render_manifest.json
export ROOM_MESH_MENON=/path/to/Menon
export ROOM_MESH_PYTHON="$PWD/data/mapanything_phone_scan_runtime/venv/bin/python"
mkdir -p "$ROOM_MESH_ROOT"
"$ROOM_MESH_PYTHON" -m pip install --no-deps \
  --target "$ROOM_MESH_ROOT/python-deps" xatlas==0.0.11 fast-simplification==0.1.13
npm install --prefix "$ROOM_MESH_ROOT/js-tools" --no-audit --no-fund @gltf-transform/cli@4.5.0

for ROOM_MESH_SPEC in preview:40000:2048 balanced:150000:4096 detail:400000:4096; do
  IFS=: read -r ROOM_MESH_LEVEL ROOM_MESH_TRIANGLES ROOM_MESH_EDGE <<< "$ROOM_MESH_SPEC"
  PYTHONPATH="$ROOM_MESH_ROOT/python-deps:." OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
    nice -n 10 "$ROOM_MESH_PYTHON" tools/room_mesh_export/build_browser_mesh.py \
      --source-glb "$ROOM_MESH_SOURCE_DIR/surface_mesh.glb" \
      --source-manifest "$ROOM_MESH_SOURCE_DIR/surface_mesh_manifest.json" \
      --scan-id 20260912-210621-2f755956 \
      --output-dir "$ROOM_MESH_ROOT/$ROOM_MESH_LEVEL" \
      --triangles "$ROOM_MESH_TRIANGLES" --texture-edge "$ROOM_MESH_EDGE" --threads 2
done

node tools/room_mesh_export/package_trial.mjs \
  --trial-root "$ROOM_MESH_ROOT" \
  --gltf-cli "$ROOM_MESH_ROOT/js-tools/node_modules/.bin/gltf-transform" \
  --camera-manifest "$ROOM_MESH_CAMERA_MANIFEST"
```

The source hash, byte size, geometry counts, colour encoding and transform-free
primitive are checked before export. Each completed level records its source,
geometry loss, UV coverage, output hash and size. Packaging records both final
variants and requires every level to share the same source mesh. Existing assets
are reused only when their recorded hashes match; mismatches fail explicitly.

## Preview and Three.js loading

```bash
python3 tools/room_mesh_export/serve_trial.py \
  --trial-root "$ROOM_MESH_ROOT" \
  --three-root "$ROOM_MESH_MENON/node_modules/three" \
  --source-glb "$ROOM_MESH_SOURCE_DIR/surface_mesh.glb" \
  --menon-loader "$ROOM_MESH_MENON/src/three/BoundedGltfLoader.js" --port 8842
```

Open `http://127.0.0.1:8842` on the same host. The temporary server binds loopback
only and runs until stopped; it is not installed as a service. The viewer offers
RGB, normals, wireframe, source-bound fixed/phone cameras, orbit and cutaway.
Loading another level disposes the old geometry, texture and material. Rendering
is on demand when idle. The original 74.34 MB source downloads only after an
explicit comparison-button click.

For the compact asset, an ordinary Three.js consumer needs its bundled decoder:

```js
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';

const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
const gltf = await loader.loadAsync('/room-meshopt.glb');
scene.add(gltf.scene); // Already in the source world frame; no second alignment.
```

Use `room-compatible.glb` when the consumer has no meshopt decoder. Neither variant
replaces an authored Menon scene or changes canonical people/world positions.

## Focused validation

```bash
PYTHONPATH="$ROOM_MESH_ROOT/python-deps:." OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  "$ROOM_MESH_PYTHON" -m pytest -q tools/room_mesh_export/test_browser_mesh.py
node tools/room_mesh_export/validate_browser.mjs \
  --dependency-root "$ROOM_MESH_MENON" --browser-executable /path/to/chrome \
  --output-dir "$ROOM_MESH_ROOT/browser-checks" --url http://127.0.0.1:8842
```

Nine focused tests passed. The actual browser consumer passed all three compact
levels, existing Menon compatible-loader parsing, one-initial-model request,
one retained geometry/texture after switching and 412-pixel mobile layout with
no console errors. Fixed-camera and phone-camera RGB/normal PNGs and
`browser_validation.json` record the results and original-source comparison.

Chrome 150 on Intel UHD 630 hardware WebGL measured approximately 60 fps and
16.7 ms p95 animation-frame intervals in each three-second orbit. The original
source also reached this cap. This is bounded local browser evidence, not a
claim of faster rendering, isolated GPU throughput or cross-device performance.
Local load times are not WAN download measurements. No runtime restart,
RoomWalk utility edit, scene publication, commit or push was performed.
