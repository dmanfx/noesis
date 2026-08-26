# MapAnything ONNX export helper

Status: DS9.1 model-export support utility invoked by
`DS9/scripts/mapanything_profile_tool.py`. It does not own runtime selection or
engine realization. Former generic ONNX-to-TensorRT instructions are archived
under `docs/history/experiments/onnx2trt/`.

Canonical DS9.1 engines are built and finalized through
`DS9/scripts/run_canonical_engine_maintenance_host.sh` against TensorRT
10.16.0.72/CUDA 13.2. Use `DS9/asset_manifest.yaml` and
`DS9/DS9_REBUILD_AND_SMOKE_GATES.md` for the selected artifact contract.
