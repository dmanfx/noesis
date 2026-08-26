# 12. nvstreammux (New) Guide

DeepStream 8 introduces a new `nvstreammux` implementation alongside the legacy mux. This guide summarizes when to use it and how to tune for common scenarios.

## Legacy vs New mux

- Legacy mux
  - Stable and widely used; good default for video-only batching.
- New mux
  - Adds richer configuration for synchronization, cascaded muxing, and metadata handling.
  - Designed for more complex multi-source scenarios.

You can run either; ensure your environment caches aren’t stale when switching implementations.

## Key concepts

- Batching: combines `N` decoded frames into a single `NvBufSurface` batch.
- Timeouts: `batched-push-timeout` controls how long to wait for a complete batch before pushing a partial one.
- Live mode: `live-source=1` enables push-as-available behavior suitable for RTSP/cameras.

## Tuning patterns

- Low latency
  - Use `live-source=1`.
  - Reduce `batched-push-timeout` to minimize waiting.
  - Keep `batch-size` modest (e.g., 2–4) to reduce buffering.
- Max throughput
  - Increase `batch-size` to saturate inference.
  - Allow a larger `batched-push-timeout` so batches are full.
  - Consider frame skipping via `nvinfer` `interval` or source-level skip.
- Mixed FPS sources
  - Enable live mode and set a finite timeout so slow sources don’t stall.
  - Optionally cap fast sources with `max-fps` to balance compute.
- Cascaded muxing
  - For very large camera counts, you can mux in stages (e.g., per-node mux then aggregate). Validate latency budgets and metadata propagation through demux/mux boundaries.

## Metadata behavior

- Frame/user meta added pre-mux can be accessed post-mux by mapping the correct `batch_id` and `frame_meta`. Ensure any custom `GstMeta` you attach pre-mux is designed to survive batching.
- Latency measurement utilities rely on user meta added after mux; verify placement when instrumenting.

## Practical tips

- Align mux `batch-size` with `nvinfer` batch.
- When changing between legacy and new mux, clear GStreamer registry cache:
  - `rm -f ~/.cache/gstreamer-1.0/registry.*.bin`
- If you see warnings about dropped buffers, review timeouts and live settings, and confirm downstream can keep up.
