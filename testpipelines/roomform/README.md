# Roomform living-room phone-walk pipeline

This offline pipeline runs the released Roomform 55M structural model and the
released Pointcept PTv3 ScanNet-20 object lifter locally. It accepts either an
aligned `living-room` MapAnything phone walk or an explicit point-preserving
DA3+MapAnything fusion with its matching camera solution. It preserves metric
RGB points and camera stations, converts the declared source frame to Roomform
Z-up, and produces the structured scene plus object geometry.

No Modal account, API key, cloud function, or paid inference service is used.

## One-time setup

Keep the environment, checkouts, and weights on a storage volume:

```bash
export ROOMFORM_STATE=/path/on/storage/roomform-local
export ROOMFORM_ROOT="$ROOMFORM_STATE/external/roomform"
export POINTCEPT_ROOT="$ROOMFORM_STATE/external/Pointcept"
export PTV3_ENV="$ROOMFORM_STATE/envs/roomform-ptv3"
export ROOMFORM_MODEL_DIR="$ROOMFORM_STATE/models/roomform"
export POINTCEPT_MODEL_DIR="$ROOMFORM_STATE/models/pointcept"
export PTV3_TMP="$ROOMFORM_STATE/tmp/roomform-ptv3"
export PTV3_PIP_CACHE="$ROOMFORM_STATE/cache/pip-roomform-ptv3"

mkdir -p "$ROOMFORM_STATE/external" "$PTV3_TMP" "$PTV3_PIP_CACHE"
git clone https://github.com/johnathanchiu/roomform.git "$ROOMFORM_ROOT"
git clone --depth 1 --branch v1.5.1 \
  https://github.com/Pointcept/Pointcept.git "$POINTCEPT_ROOT"

python3 -m venv "$PTV3_ENV"
env TMPDIR="$PTV3_TMP" PIP_CACHE_DIR="$PTV3_PIP_CACHE" \
  "$PTV3_ENV/bin/python" -m pip install -U pip setuptools wheel
env TMPDIR="$PTV3_TMP" PIP_CACHE_DIR="$PTV3_PIP_CACHE" \
  "$PTV3_ENV/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  'torch==2.4.1' 'torchvision==0.19.1'
env TMPDIR="$PTV3_TMP" PIP_CACHE_DIR="$PTV3_PIP_CACHE" \
  "$PTV3_ENV/bin/python" -m pip install \
  'numpy==1.26.4' 'pydantic>=2,<3' trimesh scikit-image scipy addict \
  timm 'spconv-cu124==2.3.8' plyfile huggingface_hub pillow matplotlib
env TMPDIR="$PTV3_TMP" PIP_CACHE_DIR="$PTV3_PIP_CACHE" \
  "$PTV3_ENV/bin/python" -m pip install \
  'https://data.pyg.org/whl/torch-2.4.0%2Bcu124/torch_scatter-2.1.2%2Bpt24cu124-cp312-cp312-linux_x86_64.whl'

"$PTV3_ENV/bin/python" testpipelines/roomform/setup_model.py \
  --model-dir "$ROOMFORM_MODEL_DIR"
"$PTV3_ENV/bin/python" testpipelines/roomform/setup_ptv3.py \
  --model-dir "$POINTCEPT_MODEL_DIR"
```

The isolated Torch 2.4/CUDA 12.4 stack matches available `spconv` and
`torch-scatter` wheels. PTv3 runs FP32 with TF32 enabled. The Roomform shell
uses BF16 and channels-last 3D tensors on Ampere.

## Run an aligned MapAnything phone walk

```bash
export REPO_ROOT="$(pwd)"
export PHONE_SCAN_DIR="$REPO_ROOT/data/mapanything_phone_scans/20260801-112036-2d9225dd"
export ROOMFORM_ARTIFACT_ROOT="$ROOMFORM_STATE/artifacts/roomform"
run_dir="$ROOMFORM_ARTIFACT_ROOT/living-room-phone-$(date -u +%Y%m%dT%H%M%SZ)"

env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PTV3_ENV/bin/python" testpipelines/roomform/main.py \
  --roomform-root "$ROOMFORM_ROOT" \
  --checkpoint "$ROOMFORM_MODEL_DIR/patch-graph-joint-rgb-55m-offset-head-r2.pt" \
  --phone-scan-dir "$PHONE_SCAN_DIR" \
  --local-ptv3 \
  --pointcept-root "$POINTCEPT_ROOT" \
  --pointcept-checkpoint \
    "$POINTCEPT_MODEL_DIR/scannet-semseg-pt-v3m1-0-base-model_best.pth" \
  --ptv3-grid-m 0.02 \
  --ptv3-max-points 150000 \
  --ptv3-tile-overlap-m 0.75 \
  --precision bf16 \
  --output-dir "$run_dir"

"$PTV3_ENV/bin/python" testpipelines/roomform/render_views.py "$run_dir"
"$PTV3_ENV/bin/python" testpipelines/roomform/render_standalone_objects.py "$run_dir"
"$PTV3_ENV/bin/python" testpipelines/roomform/validate_run.py "$run_dir"
```

## Build a point-preserving consensus cloud

The conservative consensus builder stores full-resolution, consistency-gated
per-frame evidence under its `raw/` directory, but its standard review GLB
collapses that evidence into 4 cm surfels. For PTv3, rebuild the accepted
evidence as weighted 2 cm points without rerunning MapAnything or DA3:

```bash
export SCAN_DIR="$REPO_ROOT/data/mapanything_phone_scans/<scan-id>"
export CONSENSUS_DIR="$SCAN_DIR/<consensus-output>"
export FUSION_DIR="$CONSENSUS_DIR/point_preserving_fusion_2cm"

python3 testpipelines/roomform/build_point_preserving_fusion.py \
  "$CONSENSUS_DIR/raw" \
  "$FUSION_DIR"

python3 testpipelines/roomform/render_point_cloud_views.py \
  "$FUSION_DIR/point_preserving_fusion_2cm.npz" \
  "$FUSION_DIR/renders" \
  --title "DA3 + MapAnything point-preserving fusion"
```

The rebuilder keeps all evidence that already passed cross-model and
multiview gating, requires at least two accepted samples and 0.80 accumulated
weight per voxel, and writes GLB, PLY, and NPZ forms. The NPZ additionally
preserves accumulated weight, sample count, and distinct-view support.

## Run the fused cloud through Roomform and PTv3

Supply both the point-preserving NPZ and the exact camera solution that created
its cached fusion evidence:

```bash
run_dir="$ROOMFORM_ARTIFACT_ROOT/living-room-fusion-$(date -u +%Y%m%dT%H%M%SZ)"

env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PTV3_ENV/bin/python" testpipelines/roomform/main.py \
  --roomform-root "$ROOMFORM_ROOT" \
  --checkpoint "$ROOMFORM_MODEL_DIR/patch-graph-joint-rgb-55m-offset-head-r2.pt" \
  --fusion-cloud-npz "$FUSION_DIR/point_preserving_fusion_2cm.npz" \
  --fusion-camera-solution "$CONSENSUS_DIR/camera_solution.npz" \
  --local-ptv3 \
  --pointcept-root "$POINTCEPT_ROOT" \
  --pointcept-checkpoint \
    "$POINTCEPT_MODEL_DIR/scannet-semseg-pt-v3m1-0-base-model_best.pth" \
  --ptv3-grid-m 0.02 \
  --ptv3-max-points 150000 \
  --ptv3-tile-overlap-m 0.75 \
  --precision bf16 \
  --output-dir "$run_dir"

"$PTV3_ENV/bin/python" testpipelines/roomform/render_views.py "$run_dir"
"$PTV3_ENV/bin/python" testpipelines/roomform/render_standalone_objects.py "$run_dir"
"$PTV3_ENV/bin/python" testpipelines/roomform/validate_run.py "$run_dir"
```

The fusion input mode is explicit and fail-closed. It validates the fusion
report, required NPZ arrays, RGB/point shapes, finite camera poses, and matching
view counts. It uses the documented DA3 Y-down to Roomform Z-up transform; it
does not impersonate an aligned MapAnything artifact.

`scene.glb` contains the RGB evidence, learned wall/floor/ceiling shell,
door/window predictions, and oriented object boxes. `objects/object-*.glb`
contains PTv3-segmented RGB point geometry per retained object. `scene.json`
is the editable structured Roomform document; `labels.npz`, `patchgraph.npz`,
and `evidence.npz` preserve the intermediate results.

`standalone_objects/contact_sheet.png` shows every retained PTv3 cluster in its
own oriented box. `standalone_objects/objects-standalone.glb` contains only the
boxed RGB clusters in scene position, without the Roomform shell.

The 2 cm semantic grid matches PTv3 training resolution. On a 12 GB RTX GPU,
the runner keeps all 2 cm points and evaluates overlapping spatial tiles capped
at 150k points each, then merges the labels before object clustering. There is
no silent coarsening, CPU fallback, or cloud fallback.

PTv3 was trained on ScanNet rather than MapAnything phone reconstructions, so
its object classes and boxes are review candidates, not ground truth.

## Validated 2 cm fusion result

The living-room fusion run validated on 2026-08-10 used 417,845 RGB input
points and retained 333,800 semantic points after the exact 2 cm PTv3 sampling.
Four overlapping slabs stayed below 150,000 points each. PTv3 peaked at
5.29 GiB on the RTX 3060 and produced 13 unfiltered review clusters:

```text
cabinet  6
sofa     4
table    1
window   2
```

This is a clear reduction from the 32 noisier clusters produced from the
earlier aligned phone cloud, but the semantic names remain hypotheses. Sparse
cluster rejection is a separate post-PTv3 review step and was not silently
applied to this result.

Roomform's separate SAM3D stage is the only upstream path that creates
triangulated object surfaces, and it currently calls paid external FAL. This
local path does not invoke or impersonate that stage. The delivered per-object
GLBs are segmented point geometry, not generated solid meshes.

Roomform's model weights are CC BY-NC 4.0. Pointcept code/checkpoints are MIT,
but this PTv3 checkpoint was trained on ScanNet v2 data with non-commercial
terms; review licensing before commercial use.
