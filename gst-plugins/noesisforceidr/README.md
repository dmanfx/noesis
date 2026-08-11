# `noesisforceidr` (DS8)

`noesisforceidr` is an in-place, caps-transparent `GstBaseTransform`. It does
not map or copy buffers, including NVMM surfaces. Advancing its writable
`request-sequence` property pushes NVIDIA's official downstream
`gst_nvevent_enc_force_idr(stream-id, 1)` event. The request is successful only
when `accepted-sequence` reaches the requested value and `last-request-ok` is
true.

Build the DS8-owned binary with:

```bash
./gst-plugins/build_noesisforceidr.sh
```

Inspect it without installing it globally:

```bash
GST_PLUGIN_PATH="$PWD/gst-plugins" gst-inspect-1.0 noesisforceidr
```

The DS9 source and binary are independently owned under `DS9/gst-plugins/`.
Neither runtime may load the other runtime's plugin binary.
