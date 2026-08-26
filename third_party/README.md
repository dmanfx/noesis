# External dependency provenance

No third-party repository is vendored here. `dependencies.lock.json` records
the exact origin, commit, branch state, and working-tree state of each external
checkout that existed during the 2026-08-26 workspace consolidation. The
parent repository does not register these checkouts as Git submodules.

Set `NOESIS_WORKSPACE_PRESERVATION_ROOT` to the mounted preservation store
before using an archive locator from the lock. Each stored path is relative to
that variable so the repository does not depend on a workstation-specific home
directory.

For a clean dependency, clone the recorded origin and check out the recorded
commit. For `external/DeepStream-Yolo-Seg`, also verify and apply the preserved
full-index patch, then copy the two untracked source/config files from the
preservation archive to the same repo-relative paths. The archive contains a
manifest with per-file SHA-256 digests and a working-tree copy of the modified
Python source for byte comparison.

Example verification, from the repository root:

```bash
preservation_root="${NOESIS_WORKSPACE_PRESERVATION_ROOT:?set the preservation root}"
archive="$preservation_root/2026-08-26/third_party/deepstream_yolo_seg_local_source_a8ec7eccd1e5_20260826.tar.gz"
patch="$preservation_root/2026-08-26/third_party/deepstream_yolo_seg_export_rfdetr_seg_a8ec7eccd1e5_20260826.patch"
printf '%s  %s\n' 'c7eafbbd678b4ada0abbd5021579ebce0ba2f50a0633811b85e3e5e43f3a5856' "$archive" | sha256sum --check
printf '%s  %s\n' '133854bbcf26382e605e381c17d9caa8043c0102f92785953fe5e39d6a703a1a' "$patch" | sha256sum --check
```

The DeepStream-Yolo-Seg preservation set is source-only. Eight `.o`/`.so`
build products, MapAnything package metadata, models, runtime artifacts, and
credentials are deliberately excluded. This lock establishes provenance only;
it does not create an alternate runtime authority or supersede the canonical
DeepStream 9.1 host path.
