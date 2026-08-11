# `noesisforceidr` (DS9)

This is the DS9-owned source for Noesis' NVMM-transparent force-IDR event
bridge. Advancing `request-sequence` emits NVIDIA's official downstream
`gst_nvevent_enc_force_idr(stream-id, 1)` event. Success is explicit through
the read-only `accepted-sequence` and `last-request-ok` properties.

Build the DS9-owned binary only against the DS9 SDK:

```bash
./DS9/gst-plugins/build_noesisforceidr.sh
```

Inspect it without installing it globally:

```bash
GST_PLUGIN_PATH="$PWD/DS9/gst-plugins" gst-inspect-1.0 noesisforceidr
```

The source is intentionally mirrored instead of linking or loading the DS8
plugin. DS9 must produce and load its own binary.
