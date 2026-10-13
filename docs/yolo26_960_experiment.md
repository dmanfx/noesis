# YOLO26 detector resolution experiment

Owner-authorized on 2026-10-10. Start with stock M at 960×960 and nano pose;
full-pipeline L and Watch trials wait for owner review of M. This experiment
evaluates detector resolution before adding crop recovery or changing NvDCF.

## Preserved baseline and selections

The normal [pipeline YAML](../DS9/config/infer.yaml), current Watch selection,
640 detector engines and small pose remain intact. The separate
[960 test YAML](../DS9/config/infer_yolo26_960_test.yaml) uses nano pose through
its [own INI](../DS9/pipelines/config_infer_secondary_yolo26_pose_n_test.ini).
The [640 comparison YAML](../DS9/config/infer_yolo26_640_comparison.yaml) has
the same nano pose, sources, tracker, thresholds, cadence and outputs.

`models.pgie.input_size` selects `[640, 640]` (default) or `[960, 960]` for
YOLO26 detection. The runtime resolves separately named ONNX/engine files,
generates matching preprocessing, and rejects unsupported/mismatched shapes.
Missing engines fail startup; there is no automatic substitute or rebuild.

| Selection | 960 engine in the selected artifact root |
| --- | --- |
| `--pgie-profile yolo26 --model-size m` | `models/engines/yolo26m_960_b3_fp16.engine` |
| `--pgie-profile yolo26 --model-size l` | `models/engines/yolo26l_960_b3_fp16.engine` |
| `--pgie-profile yolo26-watch --model-size m` | `models/engines/watch-26m-v1-e44_960_b3_fp16.engine` |

The 960 export contract uses fixed `images [3,3,960,960]`, FLOAT boundary tensors and
`output0 [3,300,6]`. The native host builds FP16 engines with TensorRT
10.16.0.72 on the target RTX3060. ONNX/engine provenance sidecars retain source
hashes, settings and synthetic-validation limits.

The [CPU exporter](../DS9/scripts/export_yolo26_resolution.py) consumes exact
existing checkpoints and writes separate artifacts outside the checkout. Stock
M/L use their installed export environment. Watch requires its original epoch44
checkpoint and matching Ultralytics 8.4.173 conversion environment. A fixed 640
ONNX cannot be promoted to 960 by changing only its input shape. The supplied
Watch package currently lacks that checkpoint; its 960 build remains pending.

## Downstream dimensions and authority

Only detector preprocessing and the detector engine change to 960. Installed
DeepStream 9.1 maps network boxes through the actual preprocessing scale and
padding back to the unchanged 1920×1080 mux frame. The native parser uses the
engine network dimensions. No additional downstream bbox rescaling is needed.

Nano pose retains its independent 640×640 crop input, GIE 4 and batch 3.
Its retained `secondary-reinfer-interval=8` setting does not throttle this
`network-type=100` path in DeepStream 9.1; eligible pose objects infer each frame.
Swin ReID retains 256×128 crops. NvDCF, DAv2,
calibration, ROI geometry, world localization, exact publication cohorts and
BEV/OSD/media dimensions retain their existing contracts. Inactive tracker
evidence does not become a new detection or renew world-observation age.

## Run and revert

The native supervisor accepts the parallel configuration directly:

```bash
"$NOESIS_DS91_NATIVE_ROOT/venv/bin/python" \
  DS9/scripts/run_canonical_runtime_host.py check \
  --pipeline-config "$PWD/DS9/config/infer_yolo26_960_test.yaml" \
  --pgie-profile yolo26 --model-size m
```

The persistent M test uses the added user-service drop-in
`96-yolo26-960-test.conf`. Removing only that drop-in, reloading user systemd
and restarting Noesis restores the untouched `95-yolo26-watch.conf` selection
and the normal YAML/small pose. Explicit copies of the baseline YAML, tracker,
pose INI and prior service overrides are retained under
`$NOESIS_DS9_RUNTIME_ROOT/experiments/yolo26-960-20261010/baseline`.

## Validation and interpretation

Compare stock M640+nano with stock M960+nano on the same three occupied
recordings using [the bounded replay runner](../scripts/capture_world_bev_replay.py)
with `--base-config` and explicit `--pgie-profile yolo26`. The runner validates
input rasters, uses fresh test persistence, captures tracking/world/BEV/stats,
and restores the live service. This does not clone household state.

Record per-source progress, stage latency/queue counters, exact cohort joins,
pose/world availability and encoded access-unit cadence. Check decoded WebRTC
frames separately on the live M960 trial. Isolated TensorRT times and
synthetic finite outputs do not establish occupied-pipeline throughput,
detection quality or physical localization accuracy. Captured AU consumer
drops do not measure every pre-encode queue drop.

The existing M640 ONNX and new M960 export share all 224 learned initializers
exactly and the same Ultralytics 8.4.0 export version. The existing 640 engine
retains its original build realization (FP16 with the default TF32 flag); the
new 960 engines use FP16 with TF32 disabled and a 2048 MiB build-workspace cap.
Consequently the runs compare these actual engine realizations under matched
pipeline settings, rather than proving a pure resolution-only timing delta.

Trial evidence lives under
`$NOESIS_DS9_RUNTIME_ROOT/experiments/yolo26-960-20261010/`; the observed results
and remaining limitations are recorded there after the runs.

## Observed result, 2026-10-10

Stock M960 with nano pose is selected for live owner review. Stock M/L 960
engines are built and passed native deserialization/synthetic inference; L has
not been selected in a full pipeline. Watch960 remains pending its original
checkpoint or a verified new 960 ONNX export.

The matched occupied runs retained approximately 30 FPS average per source
and encoded mosaic. Median GPU use rose from 54% to 87%, with p95 reaching
100% at 960. Warm source-rate samples ranged from 29.8–30.4 FPS at 640 and
24.6–33.4 FPS at 960. Encoded AU gap p99 rose from 81.29 to 109.37 ms;
peak GPU memory rose from 6279 to 6557 MiB. Average throughput therefore does
not imply spare capacity for another inference pass.

Every captured tracking cohort joined its exact world and BEV outputs in both
occupied replays. Each capture included one additional BEV at its closing
boundary. No tracking sequence gaps, source recovery or canonical overflow
were observed. Changed person/pose/world row counts require visual review;
they do not establish detector or physical localization accuracy.

Thirty-five focused tests passed. Live producer health and Menon readiness
were healthy; a separate headless WebRTC smoke decoded 862 frames in 30
seconds. Menon's existing native binding was refreshed for only the changed
runtime-core source hash, preserving its revision, built assets and all other
producer hashes. The experiment's `review.md`, `comparison-summary.json`,
`final-validation.json` and `menon-binding-refresh.json` retain the details.
