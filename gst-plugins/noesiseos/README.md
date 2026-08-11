# `noesiseos` (DS8)

`noesiseos` is a zero-copy, caps-transparent `GstBaseTransform` used only for
orderly pipeline quiescence. It never maps or modifies buffers. Before a
request, buffers pass normally; while EOS is pending and after downstream
accepts it, late input buffers are dropped inside the transform and cannot
enter the EOS-closed branch. Advancing
its monotonic `request-sequence` enters that terminal drop state and dispatches
a one-shot worker, so the property setter returns without waiting for the
downstream graph. The worker pushes a standard GStreamer EOS event. The request
is acknowledged only when `accepted-sequence` reaches the requested value and
`last-request-ok` is true; rejection restores normal passthrough.

Build and inspect the DS8-owned artifact with:

```bash
./gst-plugins/build_noesiseos.sh
GST_PLUGIN_PATH="$PWD/gst-plugins" gst-inspect-1.0 noesiseos
```

DS9 owns an independent source/build/binary under `DS9/gst-plugins/`. The
control element is not a substitute for pipeline lifecycle validation: the
runtime must still prove that its pipeline wait completed before closing
callback-owned resources.

Any downstream valve that may remain closed during quiescence must use
`drop-mode=1` (`forward-sticky-events`) or `drop-mode=2`; EOS is sticky and
passes those modes. `drop-mode=0` drops EOS and requires opening the valve
before requesting quiescence.
