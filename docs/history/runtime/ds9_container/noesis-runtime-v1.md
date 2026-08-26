# `noesis-runtime-v1` identity

`noesis-runtime-v1` is the only accepted Noesis checkout snapshot algorithm
for appliance deployment selector version 1. There is no Git-only compatibility
kind. The identity proves a clean committed source tree plus the ignored host
runtime material that DS8 or the DS9 supervisor may load. Secrets, writable
state, logs, caches, diagnostic output, and operating-system/DeepStream/CUDA/
Python packages are outside this descriptor and have separate preflight and
private-state contracts.

## Descriptor and digest

The descriptor object has this exact insertion order and no other keys:

1. `algorithm`: the literal `noesis-runtime-v1`;
2. `revision`: lowercase 40- or 64-hex `git rev-parse HEAD`;
3. `tree`: lowercase 40- or 64-hex `git rev-parse HEAD^{tree}`;
4. `submodules`: lexically sorted, nonempty lines from
   `git submodule status --recursive`;
5. `runtime_inventory_sha256`: the inventory digest below;
6. `runtime_file_count`: count of rows whose `type` is `file`;
7. `runtime_byte_count`: sum of every file row's `bytes`.

The checkout must have empty output from
`git status --porcelain=v1 -z --untracked-files=all --ignore-submodules=none`.
Submodule status must succeed and no line may begin with `-`, `+`, or `U`.

Serialize the descriptor as compact UTF-8 JSON with keys in the insertion
order above, Unicode unescaped, no insignificant whitespace, and one trailing
LF. `snapshot_sha256` is the lowercase SHA-256 of those bytes. The selector's
`software_revision` and `snapshot_sha256` must match the recomputed descriptor.

## Inventory rows

Rows use normalized printable-ASCII POSIX paths relative to the Noesis checkout
and are sorted by ASCII lexical order of `path`. Symlink targets must also be
printable ASCII. Controls and non-ASCII names fail closed, so JavaScript string
ordering, Python string ordering, and UTF-8 byte ordering are identical.
Duplicate paths are forbidden. Objects use these exact key orders:

- directory: `{path,type,mode}`;
- internal symlink: `{path,type,mode,target}`;
- external models symlink: `{path,type,mode,target}`;
- file: `{path,type,mode,bytes,sha256}`.

`mode` is the decimal value of `st_mode & 07777`. `target` is the raw `readlink`
text. File SHA-256 is over exact bytes. Serialize the row array as compact
UTF-8 JSON with unescaped Unicode and one trailing LF; its lowercase SHA-256 is
`runtime_inventory_sha256`.

The inventory is limited to 100,000 rows and 4 GiB of regular-file payload.
Every entry and the checkout root must be owned by the service UID. Files must
be regular, single-link entries. Opens use `O_NOFOLLOW`; device/inode, size,
mtime, opened real path, directory membership, and final path identity are
checked across reads. Directory and model-root membership is rechecked after
walking. A detected race fails the identity.

## Canonical checkout roots

These checkout-contained roots are walked recursively, including ignored
files, modes, directories, and internal symlinks:

```text
config
pipelines
gst-plugins
native_extensions
artifacts/native
plugins
external/ds_preprocess_shim
DS9/artifacts
DS9/native_extensions
DS9/plugins
DS9/gst-plugins
DS9/pipelines
```

The external RF-DETR/YOLO11 TensorRT plugin is also required exactly at:

```text
external/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg/libnvdsinfer_custom_impl_Yolo_seg.so
```

Exactly one top-level ABI-suffixed `.so` must exist for each module:

```text
noesis_depth_meta_ext
noesis_depth_tracking_tensor_ext
noesis_latency_ext
noesis_pose_meta_ext
noesis_reid_meta_ext
noesis_v3dt_meta_ext
```

`models` may be a real checkout directory or one explicit symlink to an
owner-controlled external directory. A symlink produces the row type
`external-directory-symlink`; its raw target is part of the digest. Every model
file is resolved before opening and must remain beneath that one real model
root. Nested escape links, final-component links, hard links, missing files,
and model-root replacement fail closed.

The exact model list is exported as `NOESIS_RUNTIME_MODEL_FILES` by both
implementations. It contains 32 files: COCO/Wholebody labels; every approved
DS8 primary engine for YOLO11, YOLO11-seg, YOLO26, YOLO26-seg, RF-DETR,
RF-DETR-seg, and Wholebody49; plus the shared Swin ReID, YOLO26 pose, DAv2
depth, MapAnything, NvDCF ReID, and BodyPose3D assets.

DS9 selectors additionally bind runtime/build image IDs, asset-realization
SHA-256, and ownership-matrix SHA-256. Those selector fields remain mandatory
evidence and are not replaced by this host-checkout descriptor.

## Cross-language fixture

`contracts/fixtures/v1/noesis_runtime_snapshot_vector.json` freezes row key
order, UTF-8/LF serialization, the external-model-symlink row, inventory
digest, descriptor key order, and final snapshot digest. Every implementation
must pass it byte for byte before it may produce or accept appliance health.
