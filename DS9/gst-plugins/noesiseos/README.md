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

Looping DS9.1 `nvurisrcbin` file sources also create an internal terminal
`nvurisrc_bin__fakesink` upstream of the post-mux bridge. The graph declares
these exact relative paths in `upstream-sink-paths`, separated by commas. The
worker resolves every declared terminal sink before emitting EOS, drops late
shutdown buffers at its upstream pad, and sends that sink standard EOS after
the main downstream request. Missing, duplicate, disconnected, or nonterminal
targets fail closed. Resolution is limited to 64 paths, eight path components
and 8192 characters; an empty declaration preserves the live-source path.

The acknowledgement covers every declared event path. The runtime still
requires the genuine pipeline EOS callback and `Pipeline.wait()` completion;
neither a synthetic bus message nor a shortened watchdog substitutes for them.
Successful shutdown probes remain pad-owned through pipeline teardown.
