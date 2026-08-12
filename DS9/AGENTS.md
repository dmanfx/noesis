# AGENTS.md - DeepStream 9.1 Runtime

This folder owns the DeepStream 9.1 runtime, generated assets, and migration
notes for the Noesis Service Maker app.

## Rules

1. Run DS9 only on the pinned DeepStream 9.1 stack.
   - Required SDK path: `/opt/nvidia/deepstream/deepstream-9.1` or the vendor
     `deepstream` link resolving to 9.1.
   - Required CUDA/TensorRT baseline: CUDA 13.2 and TensorRT 10.16.0.72.
   - Do not create DeepStream 8 or 9.0 compatibility symlinks.

2. Use the matching NVIDIA skill first.
   - Read the relevant `.agents/skills/*/SKILL.md` and routed references before
     direct SDK, pipeline, model, profiling, or MV3DT work.
   - Follow `DS9/docs/deepstream_9_1_agent_skills.md`; this file's 9.1 pins
     override older example versions in upstream skill text.
   - AMC execution is deferred. MV3DT remains disabled and has only one future
     candidate edge, Kitchen/Family Room. Living Room/Family Room is not an
     overlap pair.

3. Keep DS9 artifacts under `DS9/`.
   - TensorRT engines: `DS9/models/engines/`
   - Generated ONNX/configs: `DS9/models/onnx/`, `DS9/build/`
   - Custom parser libs: `DS9/pipelines/*/*.so`
   - Native Python extensions: `DS9/native_extensions/`

4. Do not reuse DS8 or DS9.0 TensorRT engines or native extension binaries.
   - Rebuild engines with DS9.1 TensorRT 10.16.0.72.
   - Rebuild parser/native `.so` files against DS9.1 headers/libs.

5. Fail fast.
   - If DS9 dependencies, parser libs, native extensions, engines, or source model artifacts are missing, report the missing item and stop.
   - Do not route DS9 execution through DS8 install paths or fallback workflows.
   - Do not import or spawn `noesis/ds8_runtime.py` from DS9 executable code.
     DS9 runtime ownership lives in `DS9/noesis/ds9_runtime.py` and
     `DS9/noesis/ds9_runtime_core.py`.

6. Document every DS9 migration decision in `DS9/README.md` or a dedicated DS9 doc.

7. Promote DS9 changes by affected capability.
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
