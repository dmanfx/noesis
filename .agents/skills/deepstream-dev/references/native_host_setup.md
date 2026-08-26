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

When a task needs an isolated environment, install the repository runtime
requirements and the Service Maker wheel from the native SDK:

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

If a virtual environment reports `No module named 'pyservicemaker'`, install
the wildcard wheel from the DeepStream 9.1 Service Maker directory shown
above. Do not rewrite the pipeline to a different API to bypass the missing
wheel.
