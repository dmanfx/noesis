# Separate larger-grid reconstruction lane

This opt-in offline lane reuses an exact prepared phone cohort, the existing
DA3 and MapAnything weights, the existing metric engine, and the normal PCF
fusion/registration algorithms. It does not change RoomWalk defaults, the
retained scan state, old outputs, live calibration, or Noesis/Menon geometry.
New inference outputs require new source-specific static-world registration;
an old transform is not copied onto a new reconstruction. Static frames remain
alignment/validation evidence, not phone inference inputs.

## October 7 experiment and interpretation

The retained September 12 Living Room walk supplies the same 130 prepared
1280×720 frames and existing, ordinal-bound 2560×1440 texture photographs.
DA3's grid increased from 504×280 to 756×420; conditioned MapAnything's grid
increased from 518×294 to the same 756×420. The metric TensorRT engine remains
the existing fixed 294×518, batch-three FP16 engine. Increasing the joint
reconstruction grid does not rebuild or resize that metric engine.

DA3 retains 64-view windows with 32-view overlap. MapAnything uses 32/16 instead
of the old 80/24 to fit the larger grid on the 12 GiB GPU. Thus this is an
operational larger-grid experiment, not an isolated resolution-only ablation.
The MapAnything default preprocessor and 80/24 defaults remain unchanged when
the opt-in flag is absent. Initial preparation and sparse-prior top-up use the
same selected resize options.

Both new DA3 and consensus alignments passed their existing 15 checks. At 2 cm,
the same multi-view support gates retain 1,364,482 surfels rather than 1,083,628.
However, equal-budget comparison against temporally withheld static model depth
favors the old geometry: common-support median disagreement is 0.304 m for the
baseline versus 0.579 m for the new lane; p80 is 0.579 versus 1.037 m. These
are model-depth agreement metrics, not surveyed physical accuracy. More pixels
and passed internal alignment gates do not make this a better replacement.
The baseline remains the preferred geometry; photo texturing can be assessed
without replacing it. All changed artifacts remain review-only.

The optional model-pose disagreement diagnostic does not reconcile each
window's local gauge before comparing across windows. Do not interpret its
aggregate as validated global pose error. Saved/fused geometry uses the supplied
DA3 carrier and independently bound source-specific registration. The static
comparison tests the actual saved geometry rather than that diagnostic.

## Run inference without overwriting a scan

Run from the repository root with the existing phone-scan Python environment.
Use explicit existing inputs and a fresh external output directory:

```bash
export HIGHRES_PYTHON="$PWD/data/mapanything_phone_scan_runtime/venv/bin/python"
export HIGHRES_ROOT=/path/to/new-experiment
export HIGHRES_SCAN=/path/to/scan130
export HIGHRES_TARGET=/path/to/paired_static_reference_revision
export HIGHRES_METRIC_ENGINE=/path/to/current-native-metric.engine

NOESIS_PHONE_SCAN_PCF_PAUSE_APPLIANCE=1 PYTHONPATH=. \
  "$HIGHRES_PYTHON" tools/mapanything_phone_scan/run_highres_reconstruction_lane.py \
  --scan-dir "$HIGHRES_SCAN" --output-root "$HIGHRES_ROOT/reconstruction" \
  --metric-engine "$HIGHRES_METRIC_ENGINE" \
  --target-revision "$HIGHRES_TARGET" --calibration "$HIGHRES_TARGET/calibration.json" \
  --camera-id living-room --process-res 756 \
  --da3-window 64 --da3-overlap 32 --ma-window 32 --ma-overlap 16
```

The runner records a runtime lease before stopping the documented native
appliance target and its exact Noesis GPU-owning service. It verifies Noesis is
stopped before inference, uses separate child CUDA contexts, and restores the
originally active appliance in `finally` before CPU fusion. It does not modify
systemd units, runtime configs, models, or engines. Restoration must pass the
normal native startup/readiness path; also inspect appliance guard health.
An originally inactive appliance is not started. A failed attempt and its lease
remain available instead of silently falling back. If a process is externally
killed before `finally` can run, recover the recorded lease explicitly:

```bash
"$HIGHRES_PYTHON" tools/mapanything_phone_scan/run_highres_reconstruction_lane.py \
  --output-root "$HIGHRES_ROOT/reconstruction" --restore-runtime
```

## Regenerate the existing review artifact families

Use the existing bound native texture photos and explicit retained CPU renderer,
rasterizer and fixed-camera visibility artifact:

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  "$HIGHRES_PYTHON" tools/mapanything_phone_scan/finish_highres_reconstruction_lane.py \
  --lane-root "$HIGHRES_ROOT/reconstruction" --output-root "$HIGHRES_ROOT/presentation" \
  --texture-photo-dir "$HIGHRES_TEXTURE_PHOTOS" \
  --renderer "$HIGHRES_CPU_RENDERER" --rasterizer "$HIGHRES_CPU_RASTERIZER" \
  --reference-visibility "$HIGHRES_FIXED_VISIBILITY"
```

This produces static-world diagnostic images; point-preserving GLB/PLY/NPZ;
dense accepted and withheld surfels; a measured surface; an explicitly inferred
closed shell; exact-corner topology cleanup; new mesh-specific photo visibility
and exposure fits; photo-patch textures; continuous photo color and witnesses;
and fixed/phone RGB, normals, wireframe, depth and face-ID renders. Inference
also retains raw per-view depth/confidence/masks/intrinsics/poses, provider
manifests, trajectory previews and consensus evidence.

The 130 full 756×420 grids total 41,277,600 pixels, exceeding the previous
40-million-pixel surfel budget. This lane explicitly requests 48 million. The
closed source has about 4.5 million faces and 2.24 million vertices, so cleanup,
visibility, texture and browser-source processing explicitly request limits of
5 million faces and 3 million vertices. Legacy defaults remain unchanged.
Only finite work limits increase; confidence, distinct-view support, depth
agreement, topology checks, and provenance gates are not relaxed.

After a failure, `--resume` reuses completed stages only with identical input
bindings, commands and output-manifest hashes. The failed stage's log/history
remain. Derivatives are CPU-only; inference does not run alongside Noesis.

Use [browser mesh export](../room_mesh_export/README.md) for compact assets and
[photo-atlas refinement](../room_mesh_export/TEXTURE_TRIAL.md) for a direct 4K
photo blend. Pass `--max-source-faces 5000000 --max-source-vertices 3000000`
for this source, and `--atlas-padding 2 --bake-padding 8` for denser packing
with outside-chart seam protection. Refined textures inherit the base export's
bound source limits. Keep simplification loss, inferred completion, unsupported
color fallback and physical-accuracy limitations visible.

`tools/room_mesh_export/build_highres_browser_assets.py` runs the three compact
levels, a direct 4K photo blend and packaging with source-bound checkpoints.
It requires the final dense mesh and that same mesh's completed camera-render
manifest. Supply the existing task-local Python dependencies via `PYTHONPATH`
and an explicit `--gltf-cli`; no shared environment installation is needed.
For this rougher source it uses two chart iterations and 0.25 m² chart areas.
The first overview layout was rejected for 0.142% ambiguous texels; the existing
0.1% rejection gate remains unchanged, and the adjusted layout passed. Failed
layouts and exports remain separate evidence, not published fallbacks.
