# Noesis native-host DeepStream setup

Noesis development, builds, validation, and operation use the installed
DeepStream 9.1 SDK directly on the host. The canonical runtime does not use a
container backend.

## Required platform

- Ubuntu 24.04
- DeepStream 9.1 under `/opt/nvidia/deepstream/deepstream-9.1`
- CUDA 13.2
- TensorRT 10.16.0.72
- GStreamer 1.24.2
- Python 3.12
- NVIDIA driver 595.58.03 or newer

Use `DS9/scripts/ds9_build_env.sh` and the native host supervisor/build scripts
to validate these pins. Do not substitute an SDK or artifact root from another
DeepStream release.

## Python virtual environment

First reuse the environment selected by the native supervisor or the task's
build/conversion configuration. Verify `sys.executable` and the imported module
origin in that interpreter. Do not create a second environment to repair an
unverified import failure.

Download/conversion utilities install only the dependencies needed by that
utility. Service Maker and the full runtime requirements are needed only when
the environment executes SDK/runtime code. If a new runtime environment is
required within the task's scope, its setup includes:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r DS9/requirements-runtime.txt
python -m pip install \
  /opt/nvidia/deepstream/deepstream-9.1/service-maker/python/pyservicemaker*.whl
```

Run commands from the repository root so the requirements path and shared
`noesis/` modules resolve consistently.

## Runtime environment

The native supervisor establishes the approved import, library, and plugin
paths. For focused manual inspection, derive paths from the installed SDK and
the DS9 tree:

```bash
export NOESIS_DEEPSTREAM_HOME=/opt/nvidia/deepstream/deepstream-9.1
export CUDA_HOME=/usr/local/cuda-13.2
export GST_PLUGIN_PATH="$PWD/DS9/gst-plugins:${NOESIS_DEEPSTREAM_HOME}/lib/gst-plugins"
export LD_LIBRARY_PATH="${NOESIS_DEEPSTREAM_HOME}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

Prefer `DS9/scripts/run_canonical_runtime_host.py` for application operation and
the matching `DS9/scripts/*host*` or focused native build script for artifact
maintenance.

## Optional codec packages

URI sources may require host GStreamer demuxer or codec packages beyond the
DeepStream installation. Install only what the selected source actually needs,
for example `gstreamer1.0-plugins-bad` for HLS or DASH demuxing. Confirm the
selected decoder and NVMM path with `gst-inspect-1.0` and a bounded application
smoke.

## Common setup failure

If SDK execution reports `No module named 'pyservicemaker'`, first confirm that
the command uses the selected native runtime interpreter. If that environment
lacks the module, install its matching wheel from the native DeepStream 9.1
Service Maker directory shown above. Verify the import origin afterward. Do not
rewrite the pipeline to a different API to bypass the missing wheel.
