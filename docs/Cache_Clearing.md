# GStreamer registry and engine cache repair

Status: native DS9.1 guidance, 2026-08-15.

Cache repair is targeted. Never recursively delete model or engine trees to
"see if it helps"; the canonical runtime requires a content-bound realization.

## Native GStreamer registry

The native runtime uses the registry selected by `GST_REGISTRY`, normally below
`NOESIS_DS91_NATIVE_ROOT`. If a newly rebuilt plugin is not discovered:

1. Stop the managed Noesis service if it owns that registry.
2. Resolve and inspect the exact path:

   ```bash
   printf '%s\n' "$GST_REGISTRY"
   test -f "$GST_REGISTRY" && ls -l "$GST_REGISTRY"
   ```

3. Remove only that exact registry file, then run the native supervisor check
   or start the service so GStreamer rebuilds it.
4. Verify only the affected factory with `gst-inspect-1.0`.

Do not clear `$HOME/.cache/gstreamer-1.0` unless the process actually uses that
registry. The managed service normally does not.

## TensorRT engines

Never delete all `.engine` or `.plan` files. Rebuild the specific declared
engine with `DS9/scripts/run_canonical_engine_maintenance_host.sh`, let the
maintenance finalizer update its realization record, and run its focused
deserialize/output check. See `DS9/DS9_REBUILD_AND_SMOKE_GATES.md`.

## Useful inspection

```bash
gst-inspect-1.0 nvurisrcbin
gst-inspect-1.0 nvinfer
gst-inspect-1.0 nvtracker
```

For a custom plugin, set the native `GST_PLUGIN_PATH` and inspect its exact
factory name. A plugin discovery failure is not a reason to change the pipeline
or load a legacy binary.
