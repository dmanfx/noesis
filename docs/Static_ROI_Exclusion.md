# Static ROI Exclusion (`nvdsroiexclude`)
_Status: current as of 2026-07-10._

This document describes the canonical DS8, V3DT, and DS9 exclusion path. The
repo-owned `nvdsroiexclude` element is an in-place, metadata-only
`GstBaseTransform` inserted before `nvtracker`:

```text
PGIE -> main_tee -> analytics_exclude (nvdsroiexclude) -> nvtracker -> analytics
```

It removes matching `NvDsObjectMeta` before tracking, ReID, pose, identity, and
analytics consume the object. There is no Python pruning hook, shadow polygon
cache, post-tracker deletion path, or fallback implementation.

## Configuration ownership

The durable source of truth is the `exclude` stage in the analytics YAML named
by `NOESIS_ANALYTICS_CONFIG` (normally `config/nvdsanalytics.yaml`). The runtime
requires one canonical stream policy for every configured source and renders an
exact native INI at `NOESIS_ANALYTICS_EXCLUDE_CONFIG` before constructing the
graph. The tracked DS8 default is `config/config_nvdsanalytics_exclude.ini`;
the canonical DS9 supervisor mounts its persistent pair at
`/var/lib/noesis/state/analytics` inside the container.

The REST contract is `GET/POST /api/v1/analytics/rois` and is defined in
`docs/DS8_api_contracts_rest.md`. The frontend edits YAML semantics, not the INI
directly.

For each stream:

- `enable=true` requires at least one ROI.
- `enable=false` may carry zero ROIs. Deleting the final ROI must disable the
  stream in the same request.
- `class-id=-1` applies to all classes; another value filters that class.
- `inverse-roi=false` removes an object fully contained by any enabled polygon;
  `true` removes objects not fully contained by one.
- ROI IDs are strict 1–64 character identifiers. Polygons require at least
  three finite, in-bounds, distinct integer-rounded points and non-zero area.

The renderer writes the native format below. It rejects duplicate groups/keys,
ambiguous stream IDs, unsafe labels, non-finite/out-of-bounds/degenerate
polygons, missing source coverage, and INI output larger than 1 MiB.

```ini
[property]
enable = 1
osd-mode = 0
display-font-size = 4
config-width = 1920
config-height = 1080

[roi-filtering-stream-0]
enable = 1
class-id = -1
inverse-roi = 0
roi-ART = 1434;102;1777;183;1762;412;1419;363

[roi-filtering-stream-1]
enable = 0
class-id = -1
inverse-roi = 0
```

The plugin defaults to `id-mode=source-id`, matching
`NvDsFrameMeta.source_id`. `pad-index` remains an explicit element mode for a
reviewed pad-index-authored config, but the canonical runtime requires
contiguous source IDs and exact source/config coverage. If a frame arrives with
no stream policy, the plugin raises a streaming error instead of passing it
unfiltered.

ROI points are authored in `config-width`/`config-height` coordinates. The
plugin maps them to each frame's pipeline dimensions with uniform contain scale
and centered padding. An object matches only when all four bounding-box corners
are inside a polygon; polygon boundaries count as inside. With `osd-mode != 0`,
the element attaches green `NvDsDisplayMeta` line segments for later rendering
by `nvdsosd`; it never changes pixels.

## Startup and reload transaction

At startup the plugin opens the INI with `O_NOFOLLOW`, requires a non-empty
single-link regular file no larger than 1 MiB, parses it strictly, and publishes
its active SHA-256. After pipeline activation the runtime verifies that initial
hash, `last-reload-ok`, zero reload errors, and an empty error string. A mismatch
is fatal.

For a REST update:

1. The API validates and bounds the complete YAML (4 MiB) and derived INI
   (1 MiB), then snapshots both files.
2. It atomically writes/fsyncs the durable state and sets
   `expected-config-sha256` on the native node.
3. It dispatches one strictly monotonic `reload-request-sequence`.
4. The native setter synchronously parses and swaps the config under its mutex.
5. REST returns success only after reading a complete receipt whose request and
   accepted sequences equal the new request, failed sequence differs, active
   hash equals the expected INI hash, error count did not rise, and error text is
   empty.

The public receipt also includes `objects_removed_count`. A rejected candidate
keeps the previous native config active and leaves the pipeline playing while
the API restores the exact YAML/INI/cache snapshots. If native activation may
have succeeded but the complete receipt/publication cannot be proven, or file
rollback fails, the process is poisoned and begins fatal shutdown. It does not
claim recovery from an ambiguous commit.

## Build and validation

The C++ implementation is an intentionally byte-identical owned mirror, while
the SDK binaries and build roots remain separate:

- DS8 source: `gst-plugins/nvdsroiexclude/gstnvdsroiexclude.cpp`
- DS8 build/output: `bash gst-plugins/build_nvdsroiexclude.sh` ->
  `gst-plugins/libgstnvdsroiexclude.so`
- DS9 source: `DS9/csrc/nvdsroiexclude/gstnvdsroiexclude.cpp`
- DS9 build/output, inside the pinned DS9 image/install:
  `NOESIS_DEEPSTREAM_HOME=/opt/nvidia/deepstream/deepstream-9.0 bash DS9/scripts/build_nvdsroiexclude_ds9.sh`
  -> `DS9/gst-plugins/libgstnvdsroiexclude.so`

Both CMake projects compile with warnings as errors and reject the wrong
DeepStream major. Run:

```bash
python3 -m pytest -q \
  tests/test_analytics_api.py \
  tests/test_pipeline_build.py \
  tests/test_runtime_shutdown_contract.py \
  DS9/tests/test_nvdsroiexclude_plugin.py \
  DS9/tests/test_preflight_plugin_origin.py \
  DS9/tests/test_runtime_container_boundary.py
```

The final behavior gate requires an authenticated runtime and a real person in
the target camera:

```bash
python3 scripts/roi_reload_smoke_test.py \
  --hot-restore \
  --camera <occupied-camera> \
  --auth-token-file "$HOME/.local/state/noesis/gateway-token" \
  --evidence "$HOME/.local/state/noesis/diagnostics/roi-hot-restore.json"
```

It must prove advancing real frames, disappearance under a temporary full-frame
exclusion, a rising native removal count, exact restore hash/receipt/readback,
and the person's return. An unoccupied camera exits blocked; HTTP success or an
older counter-only smoke is not current acceptance evidence.

## Troubleshooting

- Startup fails: inspect `last-reload-error`; do not replace the native element
  or bypass strict parsing. Check exact source coverage, file type/mode, size,
  duplicate keys, stream IDs, and polygon geometry.
- Reload returns 503: compare request/accepted/failed sequences, active hash,
  error count, and error text. A poisoned process requires resolving the defect
  and a clean restart; further mutation is intentionally refused.
- ROI appears offset: confirm stage dimensions match the authored image space
  and the current pipeline frame geometry. Do not pre-apply another letterbox
  transform.
- Wrong plugin loaded: use the plugin-origin tests. DS9 must resolve the
  DS9-owned binary and must never load the DS8 build.
- New/reordered source: update the public pipeline and analytics YAML together.
  Startup requires the canonical source IDs and exclusion stream keys to match
  exactly; no default policy is synthesized.
