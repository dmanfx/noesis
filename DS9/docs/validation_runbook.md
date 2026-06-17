# DS9 Validation Runbook

Last updated: 2026-06-16

Commands assume the repo is mounted at `/workspace` inside the DS9 container.
Do not run these against a DS8 install.

For host validation, run the same gates from the repo checkout. First verify the
host stack:

```bash
deepstream-app --version-all
python3 - <<'PY'
import tensorrt as trt
import torch
print("tensorrt", trt.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
```

The 2026-06-16 host cutover snapshot was:

- Driver `595.71.05` on `NVIDIA GeForce RTX 3060`.
- DeepStream `9.0.0`.
- CUDA runtime `13.1`.
- TensorRT `10.14.1.48`.
- Torch `2.12.0+cu130` with CUDA available.

## Start A DS9 Container

```bash
cd <repo>
docker run --rm -it --gpus all --network host --ipc host \
  -v "$PWD:/workspace" \
  -w /workspace \
  nvcr.io/nvidia/deepstream:9.0-triton-multiarch \
  bash
```

Inside the container:

```bash
apt-get update
apt-get install -y \
  python3-pip \
  gstreamer1.0-nice \
  gstreamer1.0-libav \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  libvpx9 \
  libmp3lame0 \
  libx264-164 \
  libx265-199 \
  libmpg123-0t64
python3 -m pip install --break-system-packages -r DS9/requirements-runtime.txt
```

If `gst-inspect-1.0 avdec_h264` still fails after the install, the container may
have package metadata without the shared objects. Reinstall the codec packages
and clear the GStreamer registry:

```bash
apt-get install -y --reinstall --no-install-recommends \
  libvpx9 libmp3lame0 libx264-164 libx265-199 libmpg123-0t64
rm -f ~/.cache/gstreamer-1.0/registry*.bin
gst-inspect-1.0 avdec_h264
```

## Preflight And Rebuild Gates

Run preflight first:

```bash
python3 DS9/scripts/ds9_preflight.py
```

Confirm the output reports `pyservicemaker` from
`/usr/local/lib/python3.12/dist-packages`. A stale user-site
`pyservicemaker` wheel can shadow the DeepStream 9 binding and cause native
heap corruption during construction/destruction and shutdown; the DS9 preflight
and launcher pin the system binding before importing Service Maker.

If native/plugin/parser/engine artifacts need rebuilding:

```bash
bash DS9/scripts/build_gst_plugins.sh
bash DS9/scripts/build_trt_plugins.sh
bash DS9/scripts/build_custom_parsers.sh
bash DS9/scripts/build_native_extensions.sh
python3 DS9/scripts/rebuild_engines.py
python3 DS9/scripts/ds9_preflight.py
```

If MapAnything needs a fresh ONNX/plan:

```bash
python3 utils/onnx2trt/export_ma_onnx/export_to_onnx.py \
  --repo external/map-anything \
  --outdir DS9/models/onnx \
  --h 294 --w 518 \
  --skip-ort --skip-simplify --skip-shape-inference

# Ensure the images-input export is named:
# DS9/models/onnx/mapanything_images_294x518_b3.onnx

NOESIS_MAPANYTHING_PYTHON=/path/to/export-venv/bin/python \
NOESIS_MAPANYTHING_GPU_GUARD_MB=7000 \
NOESIS_MAPANYTHING_GUARD_POLL_SECONDS=1 \
DS9/scripts/build_mapanything_guarded.sh

python3 DS9/scripts/ds9_preflight.py
```

## Launch DS9 Runtime

Use one terminal for the runtime:

```bash
cd /workspace
export PYTHONUNBUFFERED=1
export NOESIS_MOSAIC_RTSP_ENABLED=1
export NOESIS_MOSAIC_WEBRTC_ENABLED=1
export NOESIS_POSE_FEATURE_DEBUG=1
python3 DS9/noesis/ds9_runtime.py \
  --cameras-config config/cameras.yaml \
  --enable-rest \
  2>&1 | tee /tmp/noesis-ds9-runtime.log
```

Expected startup evidence:

- `DS9/scripts/ds9_preflight.py` passes.
- Engines load for `mapanything_fullframe`, `depth_tracking_fullframe`,
  `yolo26_pose`, `reid_osnet`, and the active PGIE such as `yolo11_pgie`.
- RTSP port `8554` opens.
- WebSocket/WebRTC signaling port `6008` opens.

Check ports from another shell in the same container:

```bash
ss -ltnp | rg ':(8554|6008)\b'
```

Check runtime log:

```bash
rg 'mapanything_fullframe|depth_tracking_fullframe|yolo26_pose|reid_osnet|yolo11_pgie|Pose features debug|world_quality_reason|pose_present' /tmp/noesis-ds9-runtime.log
```

On the host, `DS9/noesis/ds9_runtime.py` defaults `--storage-base` to
`DS9/data/depth` to avoid root-owned `data/depth` directories left by Docker
validation. If you override `--storage-base`, verify the target is writable by
the host user before running MapAnything or floorplan gates.

## Live Validation Runner

For future occupied-camera live-RTSP evidence passes, prefer the DS9 runner so
all gate logs and the shutdown result land in one evidence bundle:

```bash
python3 DS9/scripts/ds9_live_validation_runner.py
```

By default the runner:

- runs `DS9/scripts/ds9_preflight.py`;
- launches `DS9/noesis/ds9_runtime.py` with `--enable-rest`,
  RTSP mosaic enabled, WebRTC enabled, ReID enabled, and pose debug enabled;
- waits for WebSocket `:6008`, REST `:8080`, and RTSP `:8554`;
- runs RTSP decode, WebRTC, ReID stable-ID, BEV/track parity, strict DS9 bridge,
  floorplan, MapAnything depth, zero-copy stats, and REST-backed zero-copy gates;
- sends SIGINT to the runtime and scans the runtime log for fatal Python,
  segfault, malloc, double-free, and heap-corruption signatures.

Evidence is written under `DS9/build/live_validation/<timestamp>/` as
`summary.json`, `summary.md`, `runtime.log`, and one log per gate. If a runtime
is already running, attach without spawning it:

```bash
python3 DS9/scripts/ds9_live_validation_runner.py --no-spawn
```

Use `--skip <gate>` only to isolate a known external blocker, and document the
skip in the resulting summary. Valid gate names are `rtsp`, `webrtc`, `reid`,
`bev`, `bridge`, `floorplan`, `ma-depth`, `zero-copy-stats`, and
`zero-copy-rest`.

If live RTSP ingress starts logging repeated source reset/reconnect warnings,
do not treat late WebRTC, fresh MapAnything, BEV, ReID, or timing failures as
clean DS9 migration evidence until the source state is recovered. The observed
host failure signature was:

- `No data from source since last 10 sec. Trying reconnection`.
- RTSP `Received end-of-file` or `System error`.
- WebRTC warning `No RTSP frames yet; refusing to answer after 15s`.
- fresh `ma_depth_response` timeout with cached floorplan still available.
- zero-copy p99 timing above 3 ms during the degraded period, while fresh-start
  zero-copy runs passed.

## Pose Activation Gate

Run this with the runtime still running. Do not force the DS8 native pose
metadata bridge; under DS9 it segfaulted at the first analytics batch. The
supported DS9 path decodes YOLO26 tensor metadata through Service Maker and
attaches `NOESIS.POSE_FEATURES` with the DS9-built native extension.

```bash
python3 - <<'PY'
import asyncio
import json
import time
import websockets

async def main():
    uri = "ws://127.0.0.1:6008"
    deadline = time.time() + 45
    tracks = 0
    pose_true = 0
    world_valid = 0
    backend_world_m = 0
    reasons = {}
    async with websockets.connect(uri, max_size=None) as ws:
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            if isinstance(raw, (bytes, bytearray)):
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if msg.get("type") != "tracking":
                continue
            for track in msg.get("tracks") or []:
                tracks += 1
                if track.get("pose_present") is True:
                    pose_true += 1
                if track.get("world_valid") is True:
                    world_valid += 1
                if track.get("world_frame") == "backend_world_m":
                    backend_world_m += 1
                reason = track.get("world_quality_reason")
                if reason:
                    reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    print(json.dumps({
        "tracks": tracks,
        "pose_present_true": pose_true,
        "world_valid_true": world_valid,
        "backend_world_m": backend_world_m,
        "world_quality_reasons": reasons,
    }, indent=2))
    raise SystemExit(0 if pose_true > 0 and world_valid > 0 and backend_world_m > 0 else 1)

asyncio.run(main())
PY
```

If the implementation still uses the existing debug counter, require this log
line to show increasing `attached` counts:

```bash
rg 'Pose features debug:.*attached=[1-9]' /tmp/noesis-ds9-runtime.log
```

Do not proceed to readiness claims until this gate passes with
`pose_present_true > 0`, `world_valid_true > 0`, and
`world_frame=backend_world_m` in live telemetry.

## BEV And Track Parity Smoke

Run only after pose activation is proven:

```bash
python3 scripts/menon_bev_track_parity_smoke_test.py \
  --no-spawn \
  --ws ws://127.0.0.1:6008 \
  --pipeline-config DS9/config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --duration 45
```

Pass criteria:

- `track_world_valid > 0`
- `track_world_frame_backend_world_m > 0`
- `bev_world_frame_backend_world_m > 0`
- `comparisons > 0`
- p95 error is at or below the script threshold

Latest passing DS9 run:

```json
{
  "track_total": 2769,
  "track_world_valid": 2769,
  "track_world_frame_backend_world_m": 2769,
  "bev_total": 1632,
  "bev_world_frame_backend_world_m": 1632,
  "comparisons": 1887,
  "p95_err_m": 0.0
}
```

## WebRTC Smoke

Run against the already running DS9 runtime:

```bash
python3 scripts/webrtc_gateway_smoke_test.py \
  --ws ws://127.0.0.1:6008 \
  --duration 6 \
  --pt 103 \
  --min-rtp 10 \
  --min-decoded 1
```

Recent known-good DS9 result after dependency installation:
`rtp_packets=2287`, `decoded_frames=184`.

## RTSP Mosaic Gate

```bash
gst-launch-1.0 -e rtspsrc location=rtsp://127.0.0.1:8554/mosaic latency=100 ! \
  rtph264depay ! h264parse ! avdec_h264 ! fakesink sync=false
```

This should preroll/play without connection or decode errors.

## MapAnything, Depth, Floorplan, And ReID Gates

Run these with DS9 already running and pass `--no-spawn` so every gate uses the
same runtime instance. The DS9-copied smoke scripts spawn
`DS9/noesis/ds9_runtime.py` by default when `--no-spawn` is omitted.

```bash
python3 DS9/scripts/ma_depth_rpc_smoke_test.py \
  --no-spawn \
  --ws ws://127.0.0.1:6008 \
  --pipeline-config DS9/config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --camera family-room

python3 DS9/scripts/zero_copy_stats_smoke_test.py \
  --no-spawn \
  --stats-ws ws://127.0.0.1:6008 \
  --pipeline-config DS9/config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --duration-s 60

python3 DS9/scripts/zero_copy_smoke_test.py \
  --no-spawn \
  --stats-ws ws://127.0.0.1:6008 \
  --rest-url http://127.0.0.1:8080/api/v1/depth/refresh \
  --pipeline-config DS9/config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --duration-s 90

python3 scripts/floorplan_rpc_smoke_test.py \
  --no-spawn \
  --ws ws://127.0.0.1:6008 \
  --pipeline-config DS9/config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --max-age-sec 1200

NOESIS_REID_ENABLED=1 python3 DS9/scripts/reid_stable_id_smoke_test.py \
  --no-spawn \
  --ws ws://127.0.0.1:6008 \
  --pipeline-config DS9/config/infer.yaml \
  --cameras-config config/cameras.yaml \
  --duration 20
```

`zero_copy_smoke_test.py` requires the DS9 runtime to be launched with
`--enable-rest`; it is not covered by runtime runs that use `--disable-rest`.

If live RTSP ingest is degraded and the goal is to isolate ReID rather than
prove the full production graph, use a validation-only MP4 config under
`DS9/build/` with local file URIs, `streammux.live-source=0`,
`models.reid.enable=true`, pose/depth/MapAnything disabled, and
`validation.reid_smoke_depthless=true`. Launch it with:

```bash
NOESIS_ALLOW_DEPTHLESS_REID_SMOKE=1 \
NOESIS_REID_ENABLED=1 \
NOESIS_MOSAIC_RTSP_ENABLED=0 \
NOESIS_MOSAIC_WEBRTC_ENABLED=0 \
python3 DS9/noesis/ds9_runtime.py \
  --pipeline-config DS9/build/infer_reid_mp4.yaml \
  --cameras-config config/cameras.yaml \
  --disable-rest
```

Then run:

```bash
python3 DS9/scripts/reid_stable_id_smoke_test.py \
  --no-spawn \
  --ws ws://127.0.0.1:6008 \
  --pipeline-config DS9/build/infer_reid_mp4.yaml \
  --cameras-config config/cameras.yaml \
  --duration 35
```

This focused gate passed on 2026-06-16 for `family-room:18`. Treat it only as
ReID-path evidence; it does not replace the live-RTSP production ReID, BEV, or
world parity gates.

## Broader Migration Plan Gates

The core production path gates above are necessary but are not the same as full
DS8 option-surface parity. Keep these broader gates separate in reports:

- ROI pruning: prove `nvdsroiexclude` removes excluded objects before tracker,
  ReID, and pose work. Plugin presence from preflight is not enough.
- V3DT: if DS9 is expected to run a V3DT profile, prove
  `noesis_v3dt_meta_ext` extracts visibility, image-foot, and 3D bbox metadata
  in a DS9 runtime smoke.
- Alternate model profiles: smoke enabled YOLO26-seg and RF-DETR parser/profile
  combinations rather than relying only on build/preflight success.
- Native bridge behavior: when object-depth, depth tensor, ReID, or V3DT
  native paths are touched, run focused smokes for those contracts in addition
  to the end-to-end core runtime gates.

Run the focused DS9 bridge smoke against an already-running DS9 runtime:

```bash
python3 DS9/scripts/ds9_bridge_contract_smoke_test.py \
  --ws ws://127.0.0.1:6008 \
  --duration 75 \
  --require-embedding-track
```

For focused ROI hot-restore validation on local MP4 input, run DS9 with REST
enabled and point both analytics config paths at disposable `DS9/build/`
artifacts:

```bash
cp config/nvdsanalytics.yaml DS9/build/nvdsanalytics_roi_hot_restore.yaml

NOESIS_ALLOW_DEPTHLESS_REID_SMOKE=1 \
NOESIS_REID_ENABLED=1 \
NOESIS_ANALYTICS_CONFIG=DS9/build/nvdsanalytics_roi_hot_restore.yaml \
NOESIS_ANALYTICS_EXCLUDE_CONFIG=DS9/build/config_nvdsanalytics_exclude_roi_hot_restore.ini \
NOESIS_MOSAIC_RTSP_ENABLED=0 \
NOESIS_MOSAIC_WEBRTC_ENABLED=0 \
python3 DS9/noesis/ds9_runtime.py \
  --pipeline-config DS9/build/infer_reid_mp4.yaml \
  --cameras-config config/cameras.yaml \
  --enable-rest
```

Then run:

```bash
python3 scripts/roi_reload_smoke_test.py \
  --no-spawn \
  --hot-restore \
  --ws ws://127.0.0.1:6008 \
  --rest http://127.0.0.1:8080 \
  --pipeline-config DS9/build/infer_reid_mp4.yaml \
  --cameras-config config/cameras.yaml \
  --baseline-timeout 45 \
  --excluded-timeout 45 \
  --restore-timeout 60
```

This gate passed on 2026-06-16 with `baseline_total=2`,
`excluded_total=0`, and `restored_total=4`.

Pass criteria:

- `depth_tracking_device_frames_total > 0`
- `object_depth_gpu_roi_copies_total > 0`
- `object_depth_attach_total > 0`
- `object_depth_status_total.ok > 0`
- `tensor_host_copies_total.reid > 0`
- `core_path.cpu_copy_violation.total == 0`

Latest broader-gate evidence from 2026-06-15 and 2026-06-16:

- Zero-copy stats passed:
  `samples=43`, `max_zero_copy_violations=0`,
  `max_boundary_p99_ms=1.034443`, `ws_depth_requests=43`,
  `ws_depth_responses=5`.
- Zero-copy REST depth passed with DS9 launched using `--enable-rest`:
  `samples=62`, `max_zero_copy_violations=0`,
  `max_boundary_p99_ms=1.146929`, `rest_refresh_attempts=8`,
  `rest_refresh_success=8`, `rest_refresh_last_status=ok`.
- ROI unit/API tests passed with:
  `pytest -q tests/test_exclude_prune_hook.py tests/test_analytics_api.py`.
  Live full-frame exclusion pruned pose/debug object counts to zero after the
  `nvdsroiexclude` reload.
- Focused MP4 ROI hot-restore passed:
  `baseline_total=2`, `excluded_total=0`, `restored_total=4`.
- YOLO26 segmentation profile smoke passed:
  `python3 DS9/noesis/ds9_runtime.py --cameras-config config/cameras.yaml --pgie-profile yolo26_seg --size s --disable-rest`.
- RF-DETR segmentation profile smoke passed after generated parser/label path
  fixes:
  `python3 DS9/noesis/ds9_runtime.py --cameras-config config/cameras.yaml --pgie-profile rfdetr_seg --size m --disable-rest`.
- Bridge-specific object-depth, depth tensor, and ReID native extraction smoke
  passed with
  `python3 DS9/scripts/ds9_bridge_contract_smoke_test.py --ws ws://127.0.0.1:6008 --duration 75 --require-embedding-track`:
  `stats_samples=76`, `tracking_messages=1472`, `tracks_seen=2614`,
  `depth_ok_tracks=2248`, `embedding_tracks=2609`,
  `depth_tracking_device_frames_total=1533`,
  `object_depth_gpu_roi_copies_total=2246`, `object_depth_attach_total=2612`,
  `object_depth_status_total.ok=2246`, `tensor_host_copies_total.reid=354`,
  `core_path.cpu_copy_violation.total=0`, and `pipeline_errors=[]`.
- RF-DETR detect-only profile smoke passed after staging
  `DS9/models/onnx/rfdetr_{n,s,m}_*.onnx` and building
  `DS9/models/engines/rfdetr_{n,s,m}_*_b3_fp16.engine`:
  `python3 DS9/noesis/ds9_runtime.py --cameras-config config/cameras.yaml --pgie-profile rfdetr --size s --disable-rest`.
  The runtime loaded `DS9/models/engines/rfdetr_s_512_b3_fp16.engine` and
  stayed alive until the bounded smoke timed out with `RUNTIME_RC=124`.
- Occupied-camera live-RTSP host validation passed:
  - BEV/track parity:
    `track_total=11435`, `track_world_valid=11426`,
    `bev_world_frame_backend_world_m=6980`, `comparisons=11065`,
    `p95_err_m=0.0`, and lagged p95 `0.11610091475856413`.
  - ReID stable-ID smoke: `family-room:16`.
  - Bridge smoke with `--require-embedding-track`:
    `tracking_messages=5373`, `tracks_seen=11436`,
    `depth_ok_tracks=11066`, `embedding_tracks=11398`,
    `depth_tracking_device_frames_total=4833`,
    `object_depth_gpu_roi_copies_total=10980`,
    `object_depth_attach_total=11350`,
    `object_depth_status_total.ok=10980`,
    `tensor_host_copies_total.reid=971`,
    `core_path.cpu_copy_violation.total=0`, and `pipeline_errors=[]`.
  - WebRTC smoke: `rtp_packets=1438`, `decoded_frames=92`.
  - RTSP mosaic decode held for 25 seconds and sustained decode held for
    60 seconds.
  - Zero-copy stats:
    `samples=7`, `max_zero_copy_violations=0`,
    `max_boundary_p99_ms=1.123324`, `ws_depth_requests=38`, and
    `ws_depth_responses=4`.
  - MapAnything depth RPC passed cache-first plus fresh for `family-room`.
  - Floorplan RPC passed for `kitchen` and then `family-room`; the first
    `family-room` attempt timed out before succeeding on retry.
  - Live RTSP SIGINT shutdown exited `0`, closed ports `6008`, `8080`, and
    `8554`, and left no DS9 runtime process. The shutdown tail logged
    `Wait thread did not terminate cleanly` and one `source_2` reconnect warning
    after EOS, with no native heap/fatal markers.
- V3DT is blocked until DS9-scoped V3DT pipeline/camera/tracker assets exist.
  `DS9/scripts/sv3dt_meta_smoke_test.py` refuses to auto-spawn without an
  explicit DS9 V3DT pipeline config; do not use root DS8 V3DT configs as DS9
  evidence.
- Host cutover evidence from 2026-06-16 UTC:
  - preflight passed with host DeepStream 9.0.0 / TensorRT 10.14.1.48.
  - fresh host runtime loaded all five engines and opened `:8554`, `:6008`, and
    `:8080`.
  - WebRTC passed with `rtp_packets=6945`, `decoded_frames=602`.
  - RTSP mosaic decode held until bounded timeout with no decode error.
  - zero-copy stats passed with `samples=41`,
    `max_boundary_p99_ms=1.002647`, `max_zero_copy_violations=0`.
  - REST-backed zero-copy passed with `samples=56`,
    `max_boundary_p99_ms=2.549196`, `rest_refresh_success=9`.
  - floorplan RPC passed.
  - native bridge counters moved for object-depth/depth/ReID:
    `depth_tracking_device_frames_total=4389`,
    `object_depth_gpu_roi_copies_total=31`, `object_depth_attach_total=46`,
    `object_depth_status_total.ok=31`, `tensor_host_copies_total.reid=5`,
    `core_path.cpu_copy_violation.total=0`.
  - focused host MP4 ReID stable-ID smoke passed with the validation-only
    depthless ReID config: `family-room:18`.
  - MP4 shutdown/native cleanup smoke passed after pinning the DS9 system
    `pyservicemaker` and removing the stale user-site shadow install: SIGINT
    posted EOS and exited `0` with no fatal Python, segfault, malloc,
    double-free, or heap-corruption markers. Finite MP4 EOS
    (`streammux.live-source=0`, `NOESIS_DS8_LOOP_LOCAL_MP4=0`) also exits `0`.
    Looping MP4 sources can still log `Wait thread did not terminate cleanly`.
  - occupied-camera live-RTSP host validation later proved ReID stable IDs,
    BEV/track parity, strict embedding-track bridge, MapAnything depth, floorplan,
    RTSP/WebRTC media output, zero-copy stats, and live RTSP SIGINT shutdown.
    See the broader-gate evidence above for the exact counters.

Suggested bounded alternate-profile commands:

```bash
timeout 45s python3 DS9/noesis/ds9_runtime.py \
  --cameras-config config/cameras.yaml \
  --pgie-profile yolo26_seg \
  --size s \
  --disable-rest

timeout 75s python3 DS9/noesis/ds9_runtime.py \
  --cameras-config config/cameras.yaml \
  --pgie-profile rfdetr_seg \
  --size m \
  --disable-rest

timeout 75s python3 DS9/noesis/ds9_runtime.py \
  --cameras-config config/cameras.yaml \
  --pgie-profile rfdetr \
  --size s \
  --disable-rest
```

## DS8-vs-DS9 Config/Artifact Review

This is a review gate, not an equality check. DS9 paths should differ where
expected, but they must not point at DS8-only engines, parser outputs, or label
paths.

```bash
python3 - <<'PY'
from pathlib import Path

required = [
    "DS9/noesis/ds9_runtime.py",
    "DS9/config/infer.yaml",
    "DS9/config/depth_registration.json",
    "DS9/models/coco_labels.txt",
    "DS9/pipelines/config_infer_primary_yolo11_seg.ini",
    "DS9/pipelines/config_infer_primary_yolo11.ini",
    "DS9/pipelines/config_infer_primary_rfdetr_seg.ini",
    "DS9/pipelines/config_infer_primary_rfdetr.template.ini",
    "DS9/pipelines/config_infer_primary_rfdetr_seg.template.ini",
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit("missing required DS9 paths:\n" + "\n".join(missing))

bad_tokens = [
    "../../models/coco_labels.txt",
    "deepstream-8",
    "deepstream-8.0",
    "DS8/models/engines",
]
bad = []
scan_roots = [
    Path("DS9/noesis"),
    Path("DS9/config"),
    Path("DS9/pipelines"),
    Path("DS9/scripts"),
    Path("DS9/csrc"),
]
for root in scan_roots:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml", ".ini", ".txt", ".sh", ".cpp", ".hpp", ".h", ".cmake"}:
            continue
        text = path.read_text(errors="ignore")
        for token in bad_tokens:
            if token in text:
                bad.append(f"{path}: {token}")
if bad:
    raise SystemExit("unexpected DS8/stale references:\n" + "\n".join(bad[:50]))
print("[OK] Required DS9 paths exist and obvious stale references were not found.")
PY
```

For manual diff review:

```bash
diff -u config/infer.yaml DS9/config/infer.yaml || true
diff -u config/depth_registration.json DS9/config/depth_registration.json || true
```

Review differences intentionally; do not force equality.

## Readiness Criteria

Production readiness requires:

- Preflight passes in the DS9 container.
- Runtime starts through `DS9/noesis/ds9_runtime.py`.
- All expected DS9 engines load.
- Pose activation gate passes.
- BEV/track parity smoke passes.
- WebRTC and RTSP mosaic gates pass.
- MapAnything, depth, floorplan, and ReID gates pass.
- No DS8 fallback/shim/degraded path is required.
- `DS9/README.md`, `DS9/docs/migration_state.md`, and
  `DS9/docs/known_blockers.md` are updated with final validation results.

Latest DS9 validation satisfied the startup, engine-load, pose, BEV, WebRTC,
RTSP, MapAnything, floorplan, ReID, config-review, zero-copy, REST depth, ROI
prune, YOLO26-seg, RF-DETR-seg, object-depth bridge, depth tensor bridge, and
ReID native extraction bridge gates listed above, plus RF-DETR detect-only
startup for `s`, and occupied-camera live-RTSP host tracking/BEV/shutdown
evidence. Full DS8 option-surface parity is still blocked by DS9-native V3DT
staging where V3DT support is required.
