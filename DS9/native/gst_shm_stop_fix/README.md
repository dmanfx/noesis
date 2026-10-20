# Narrow GStreamer SHM stop/poll correction

Vendored from the official GStreamer repository, tag `1.24.2`, commit
`2d8273151571fcab887cc81de48e87aeb61b5c06`, directory
`subprojects/gst-plugins-bad/sys/shm/`. The accompanying provenance JSON records
the exact original blob identities and SHA256 values. `meson.build` preserves
the upstream five-C-file source list. No other upstream subtree is vendored.

The GStreamer plugin files retain their LGPL-2.0-or-later notices; upstream
`COPYING` supplies the LGPL-2.1 text. The pipe/allocator helpers retain their
original permissive MIT-style notices. All copyright/license headers are intact.

Only `gstshmsink.c:pollthread_func` changes: when the existing stop flag is set
AND `gst_poll_wait` returns EBUSY, exit without posting a resource error. This
is the deliberate wakeup produced by `gst_shm_sink_stop` calling
`gst_poll_set_flushing`. Non-stopping EBUSY and every unrelated errno remain
errors. No protocol, allocator, factory, inference, media or graph changes.

`bash DS9/scripts/build_shm_stop_fix_host.sh EXISTING_OUTPUT_DIR` builds only
`libgstshm.so` against installed GStreamer/base 1.24.2. It does not install or
select it. Test children explicitly load this private plugin and use a private
registry; the canonical application's origin, configuration and binding remain
unchanged. Runtime wiring/manifests and actual application acceptance are a
separate root-owned operation, not implied by the CPU tests.
