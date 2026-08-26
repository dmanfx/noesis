# 11. nvmultiurisrcbin Cheat Sheet (DS8)

`nvmultiurisrcbin` is a convenience source bin that wraps multiple `nvurisrcbin` instances and a `nvstreammux` into a single element with one `src` pad that outputs batched GPU surfaces. It is purpose-built for multi-camera ingest and dynamic add/remove of sources at runtime.

## Why use it

- Single element to manage many URIs and batching.
- Dynamic lifecycle via REST API (add/remove/update sources without rebuilding the pipeline).
- Internal muxing and capability negotiation.

## Minimal usage

Pipeline sketch:

```
gst-launch-1.0 nvmultiurisrcbin name=srcbin \
  ! queue \
  ! nvinfer config-file-path=pgie.txt \
  ! nvtracker \
  ! nvdsosd \
  ! nveglglessink
```

Add sources via REST (example):

```
# Start the REST server (if not auto-started by your app)
# Then add a sensor:
curl -X POST http://localhost:9000/api/v1/sensors \
  -H 'Content-Type: application/json' \
  -d '{
        "id": "cam-01",
        "uri": "rtsp://user:pass@host/stream1",
        "enable": true,
        "type": "rtsp"
      }'
```

Remove a sensor:

```
curl -X DELETE http://localhost:9000/api/v1/sensors/cam-01
```

Actual endpoints and payloads can vary by build; see 13_REST_API_Server.md and your DS8 install for exact schemas.

## Key properties (categories)

Properties are grouped into three areas. Names below reflect common DS8 usage; check `gst-inspect-1.0 nvmultiurisrcbin` for the full list and exact names.

- nvmultiurisrcbin itself
  - `gpu-id`: GPU to use for decode/mux.
  - `sync-inputs`: synchronize live sources when batching.
  - `drop-pipeline-eos`: keep pipeline alive when last source EOS is seen.
- Per-source defaults (applies to internal `nvurisrcbin` instances)
  - `live-source`: mark sources as live for better latency behavior.
  - `max-fps`: limit per-source FPS.
  - `reconnect-interval`: attempt reconnect for network sources.
  - `drop-frame-interval` / `skip-frame` / `interval`: sampling controls.
- Internal mux configuration (applies to the embedded `nvstreammux`)
  - `width`, `height`: output surface dimensions.
  - `batch-size`/`max-batch-size`: batching parameters.
  - `batched-push-timeout`: microseconds to wait before pushing partial batches.

Notes:

- Set `live-source=1` in mux when number of active sources is less than `max-batch-size` to allow immediate batching of available frames.
- Prefer `drop-pipeline-eos=1` for long-running pipelines that may add/remove sources over time.
- The new `nvstreammux` variant is not universally supported in all audio+video cases within this bin; use legacy mux when mixing A/V unless validated.

## Debugging tips

- List internal structure: run with `GST_DEBUG_DUMP_DOT_DIR` to export the graph and verify internal `nvurisrcbin` and `nvstreammux` instances.
- Use REST GET endpoints to list current sensors and states.
- Watch for caps negotiation issues when adding sources with differing formats; set explicit caps or limits via per-source properties.

## Stream indices and tiler tiles

- Stream indices (0..N-1) follow your `[source-list].list` order in `pipelines/noesis_multiurisrcbin.ini`.
- In a 2x2 `nvmultistreamtiler`, tile order is `0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right`.
- If other components (e.g., ROI configs) are authored by tile position, prefer `id-mode=pad-index` there and keep pad ordering stable (consider `sensorID-padID-mapping=1` and `sort-batch=1`).
