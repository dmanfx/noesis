# DS8 YOLO26-Seg Engine Maintenance

Status: guarded CPU plan and the canonical `n`, `s`, and `m` rebuilds validated
on 2026-07-10.

## Purpose

Use `scripts/ds8_yolo26_seg_engine_maintenance.py` to rebuild the canonical
DS8 YOLO26 segmentation `n`, `s`, or `m` TensorRT engines explicitly. Do not
start `nvinfer` and rely on its opportunistic rebuild behavior. A missing,
stale, or incompatible engine is a maintenance failure until this command
builds and validates a replacement.

At the start of this maintenance pass, the `n/s/m` engine files were serialized
by TensorRT 10.14 and could not be loaded by the canonical DS8 TensorRT
10.13.3.9 runtime. The guarded command has since rebuilt all three sizes. A
further trap is that this host's
`trtexec --loadEngine ... --skipInference` can return zero and print its final
`PASSED` banner even after `Error[6]`, `Error[4]`, and
`Engine deserialization failed`. Exit status and the final banner are therefore
not sufficient validation evidence.

## Validated `n`, `s`, And `m` Rebuilds

Guarded run `20260710T222107305588Z` completed under the caller-selected
external evidence root. It preserved the prior 24,969,612-byte engine with
SHA-256
`c65a86eca785feea4e967a38fca8b7e9b8b1f6caa424ad038fc874e5f201ca08`,
installed a 26,341,932-byte engine with SHA-256
`2edd286489841c09b4af09919d9d0b92a4665c8407c5e8c476140fed03a2b670`,
and independently rechecked the installed hash.

The build took 219.514 seconds, candidate deserialization took 1.076 seconds,
final-path deserialization took 1.075 seconds, and peak observed GPU memory was
658 MiB. Both load logs contain positive engine-size, deserialization, skipped
inference, and `PASSED` markers with no fail-closed TensorRT error signature.
The maintenance command and its `trtexec` child both exited, and no compute
owner remained after completion.

Guarded run `20260710T222623433238Z` then preserved the prior 9,470,244-byte
`n` engine with SHA-256
`39f341a1da881053b7d636d9fd98f523f629d2ed8101f6b7b73617854a218e67`
and installed a 10,613,572-byte engine with SHA-256
`ffc2666aac5a762e2aeb88ab054df3234e42225dc45450112c7dc6b239c7bcb0`.
The build took 218.526 seconds, candidate deserialization took 1.075 seconds,
final-path deserialization took 1.078 seconds, and peak observed GPU memory was
622 MiB. Its candidate and final-path logs passed the same positive-marker and
error-signature gate, its installed hash was independently checked, and it
also left no compute owner.

Guarded run `20260710T223108901671Z` preserved the prior 50,848,460-byte `m`
engine with SHA-256
`5b7a18699c1cf885ca98e9da4e021c17c46c1942017e3e6ba107881f34ed11a2`
and installed a 53,060,412-byte engine with SHA-256
`2ce7cb72809709405c5a5e018e9794170ac9bce6003abaa8346b9a347d367ac8`.
The build took 277.868 seconds, candidate deserialization took 1.075 seconds,
final-path deserialization took 1.074 seconds, and peak observed GPU memory was
904 MiB. Its candidate and final-path logs passed the same positive-marker and
error-signature gate, its installed hash was independently checked, and it
also left no compute owner.

These runs provide serialization/deserialization evidence only; the focused
DS8 runtime-quality gate is still required for each selected size.

## Review The Plan

The plan is CPU-only, invokes no subprocess, writes nothing, and does not query
or initialize the GPU:

```bash
python3 scripts/ds8_yolo26_seg_engine_maintenance.py \
  --plan \
  --sizes n,s,m
```

Use `--dry-run` as an alias for `--plan`. The plan validates and hashes:

- each fixed-batch fused ONNX source;
- the parser binary and parser source;
- the nvinfer and preprocess templates;
- the labels file;
- the current engine bytes, when present; and
- the selected `trtexec` binary and CUDA version manifest.

It prints the exact build and separate load commands without executing them.

## Exact Build Profile

Real mode fails unless the host matches the reviewed DS8 profile:

- DeepStream 8;
- TensorRT Python `10.13.3.9` and `trtexec` `v101303` build `b9`;
- CUDA SDK `13.0.2` and CUDA runtime `13.0.96`;
- NVIDIA driver `595.71.05`;
- NVIDIA GeForce RTX 3060, compute capability `8.6`, 12,288 MiB;
- static `images[3,3,640,640] -> output0[3,30,4102]` FLOAT tensors;
- FP16 enabled with TensorRT's FP32 fallback semantics;
- 4 GiB workspace, builder optimization level 3, and zero auxiliary streams.

Any future SDK, driver, GPU, tensor, parser, or template change requires a new
review rather than silently broadening this profile.

The real-mode profile probe uses `trtexec --help`. TensorRT 10.13 does not have
a functional `--version` mode: it prints the correct version banner, then emits
`[E] Model missing`, prints `FAILED`, and exits nonzero. The guard rejects that
deceptive transcript and accepts only a zero-exit, error-free help transcript
containing the exact `v101303 [b9]` banner.

## Authorized Build

Run a real build only in an announced exclusive-GPU window after reviewing the
plan. Build one size first when repairing a production blocker:

```bash
python3 scripts/ds8_yolo26_seg_engine_maintenance.py \
  --build \
  --sizes s
```

The default evidence root is
`models/engine_maintenance/yolo26_seg/<run-id>/`. It follows the configured
repository model store rather than embedding a machine-local path. Override it
with `--evidence-root` only when the selected filesystem has the required
headroom.

Real mode:

1. takes a nonblocking process-wide advisory lock shared by every invocation;
2. refuses any existing GPU compute owner;
3. verifies the exact host profile and rechecks all planned input hashes;
4. requires 5 GiB residual free space after bounded temporary, log, and prior
   engine storage;
5. copies and SHA-256 verifies every prior engine before starting a build;
6. writes the candidate to a sibling `.building-<run-id>` file;
7. bounds build time (`n=900s`, `s=1200s`, `m=1800s`), each load (`120s`),
   GPU memory (`11000 MiB`), each log (`32 MiB`), and candidate size;
8. runs a separate `trtexec --loadEngine ... --skipInference` process;
9. rejects nonzero exit, TensorRT error signatures, or missing positive
   `Loaded engine size`, `Engine deserialized in`, and inference-skipped
   markers; and
10. atomically replaces the target only after candidate validation, then fsyncs
   and verifies the installed hash; and
11. launches a third process to deserialize the final installed path and
    applies the same error-signature and positive-marker gate again.

The run manifest records input, old-engine, candidate, platform, command, log,
duration, memory, and installed hashes. A failed candidate is removed; prior
bytes and logs remain in the evidence run. The tool does not automatically
restore or select an alternate engine.

## CPU Regression Gate

```bash
pytest -q tests/test_ds8_yolo26_seg_engine_maintenance.py
python3 scripts/ds8_yolo26_seg_engine_maintenance.py --plan --sizes n,s,m
```

The regression suite includes the observed false-positive `trtexec` transcript
where exit zero and `PASSED` accompany `Error[6]`/`Error[4]`. It also proves a
failed load preserves the prior target and that installation occurs only after
positive deserialization evidence.

After an authorized engine build, run the focused DS8 runtime gate for the
selected size. Engine serialization and standalone deserialization do not by
themselves prove parser, mask, throughput, media, or clean-shutdown behavior.
