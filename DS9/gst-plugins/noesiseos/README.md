# `noesiseos` (DS9)

This is DS9's independent zero-copy orderly-EOS control plugin. It passes all
caps and NVMM buffers untouched before a request. While EOS is pending and
after downstream accepts it, the plugin drops late input buffers without
mapping or copying them. A strictly increasing
`request-sequence` starts a self-owned one-shot worker and returns immediately;
the worker pushes standard downstream EOS after the setter unwinds.
`accepted-sequence` and `last-request-ok` provide the later exact
acknowledgement, and a rejected event restores passthrough.

Build and inspect only the DS9-owned artifact:

```bash
./DS9/gst-plugins/build_noesiseos.sh
GST_PLUGIN_PATH="$PWD/DS9/gst-plugins" gst-inspect-1.0 noesiseos
```

The DS9.1 binary is built against the installed native SDK. It must not load an
archived DS8/DS9.0 artifact. Runtime graph placement and shutdown orchestration
remain outside this plugin package.

Closed downstream valves must use `drop-mode=1` or `drop-mode=2` so sticky EOS
continues downstream. A `drop-mode=0` valve must be opened before an EOS
request; otherwise an accepted event does not prove whole-pipeline quiescence.
