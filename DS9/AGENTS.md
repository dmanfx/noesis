# AGENTS.md - DS9 Port

This folder contains DeepStream 9 specific runtime overlays, generated assets, and migration notes for the Noesis Service Maker app.

## Rules

1. Run DS9 only on a real DeepStream 9 install/container.
   - Required SDK path: `/opt/nvidia/deepstream/deepstream-9.0` or a `deepstream` symlink that resolves to DS9.
   - Do not create DeepStream 8 compatibility symlinks.

2. Keep DS9 artifacts under `DS9/`.
   - TensorRT engines: `DS9/models/engines/`
   - Generated ONNX/configs: `DS9/models/onnx/`, `DS9/build/`
   - Custom parser libs: `DS9/pipelines/*/*.so`
   - Native Python extensions: `DS9/native_extensions/`

3. Do not reuse DS8 TensorRT engines or native extension binaries.
   - Rebuild engines with DS9 TensorRT 10.14.1.48.
   - Rebuild parser/native `.so` files against DS9 headers/libs.

4. Fail fast.
   - If DS9 dependencies, parser libs, native extensions, engines, or source model artifacts are missing, report the missing item and stop.
   - Do not route DS9 execution through DS8 install paths or fallback workflows.
   - Do not import or spawn `noesis/ds8_runtime.py` from DS9 executable code.
     DS9 runtime ownership lives in `DS9/noesis/ds9_runtime.py` and
     `DS9/noesis/ds9_runtime_core.py`.

5. Document every DS9 migration decision in `DS9/README.md` or a dedicated DS9 doc.

6. Promote DS9 changes by affected capability.
   - Follow the root `AGENTS.md` capability-scoped production-promotion policy.
   - For MapAnything/depth-panel work, default to the focused depth,
     exact-capture, storage/fusion, frontend-build, manual-Refresh, cache-only,
     and service-readiness gates affected by the change.
   - A network change requires exact listener, reachability, TLS/auth, and
     rollback checks. A systemd change requires unit validation,
     dependency/start/stop/restart/readiness, and rollback checks. Neither
     automatically requires model-quality, tracking, identity, occupied-scene,
     long-soak, or sealed full-runtime evidence.
   - A model or native change requires the affected DS9 engine/native loading,
     tensor/metadata contract, and direct-consumer live path. It does not
     automatically require every unrelated DS9 capability.
   - Use full-system assurance only when the change crosses core pipeline
     topology or shared source/frame/timestamp/identity/world semantics, when
     its impact cannot be bounded after inspection, for a named release
     candidate, or when explicitly requested.
   - Reuse unchanged realized engines, native builds, release artifacts, and
     passed evidence. Do not restart the whole promotion sequence after an
     unrelated failure; rerun only invalidated phases.
