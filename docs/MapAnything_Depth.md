# MapAnything Depth (DS8)
_Status: validated against code on 2026-03-16._

MapAnything is still a first-class DS8 depth component, but it is no longer the
only depth-related path in the runtime.

## Roles

DS8 currently uses two distinct depth lanes:

- `models.mapanything`
  - Full-frame SGIE branch in the live DS8 pipeline.
  - On-demand and gate-controlled.
  - Owns the `depth_result` WebSocket payload and `get_ma_depth` RPC contract.
  - Persists dense full-frame snapshots under `data/depth/...`.
- `models.depth_tracking`
  - Full-frame Depth Anything V2 metric lane used by baseline non-`v3dt`
    world tracking.
  - Always on in baseline mode.
  - Does not publish a second full-frame depth WebSocket stream.
  - Contributes through `NOESIS.OBJECT_DEPTH` and the fused backend world
    estimator in `noesis/pipelines/hooks.py`.

MapAnything also acts as the offline reference source for the DAv2 room
registration artifact used by baseline tracking.

## Runtime Topology

MapAnything stays on its own DS8 branch:

- `main_tee -> mapanything_queue -> mapanything_valve -> mapanything_fullframe -> mapanything_fullframe_sink`

Config source:

- `config/infer.yaml`
- `models.mapanything.*`
- default `gie_id=2`

Gate control:

- REST: `GET /api/v1/depth/refresh?seconds=N`
- WebSocket RPC: `get_ma_depth` can open a short refresh window
- Runtime: `DS8Pipeline.mark_depth_enabled()` controls `mapanything_valve.drop`

The gate applies only to MapAnything. It does not control the always-on baseline
DAv2 tracking lane.

## Postprocess, Storage, and RPC

Live MapAnything processing is owned by `MapAnythingProcessor` in
`noesis/pipelines/hooks.py`.

Responsibilities:

- decode MapAnything tensor outputs
- align depth/conf/mask to camera frame geometry
- store dense snapshots through `geometry.depth_source.DepthStorageManager`
- publish `DepthResult`
- serve `get_ma_depth` via the runtime provider path

The full-frame `depth_result` and `ma_depth_response` contracts remain
MapAnything-specific. They are not reused for the baseline DAv2 tracking lane.

## Offline DAv2 -> MapAnything Registration

Baseline non-`v3dt` world tracking now requires a prebuilt registration artifact
that maps raw DAv2 anchor range into MapAnything-aligned room range.

Canonical pieces:

- builder: [build_depth_registration.py](/home/mayor/Noesis_Devel/scripts/build_depth_registration.py)
- artifact: [depth_registration.json](/home/mayor/Noesis_Devel/config/depth_registration.json)
- schema/loader: [depth_registration.py](/home/mayor/Noesis_Devel/noesis/calibration/depth_registration.py)
- fitter: [depth_registration_builder.py](/home/mayor/Noesis_Devel/noesis/calibration/depth_registration_builder.py)

Operational rules:

- The artifact is generated offline and loaded read-only by DS8 at startup.
- DS8 does not auto-generate, auto-refresh, or auto-download this artifact.
- Missing or stale entries are a fatal startup error in baseline mode.
- Empty-room RTSP captures are preferred, but the builder now filters samples to
  temporally stable pixels so minor/static occupancy does not automatically
  poison the fit.

Typical workflow:

```bash
bash services/mapanything_svc/run.sh
env CUDA_VISIBLE_DEVICES='' python3 scripts/build_depth_registration.py \
  --output config/depth_registration.json
timeout 25s python3 noesis/ds8_runtime.py --pgie-profile yolo26_seg --size s --disable-rest
```

The builder uses live RTSP sources from `config/infer.yaml` by default.

## MapAnything Service Notes

The local MapAnything service is still an active tool for offline registration
work. The deprecated part is the old live-runtime microservice/adapter depth
path; the canonical runtime depth path is the DS8 SGIE branch.

Current service ownership:

- startup script: `services/mapanything_svc/run.sh`
- app: `services/mapanything_svc/server.py`
- weight pin: `docs/ma-integration/weights.sha`

The service is primarily used to produce reference dense depth for registration
builds, not to replace the DS8 in-pipeline MapAnything branch.

## What MapAnything Does Not Own

MapAnything is not the canonical owner of baseline person world tracking.

It does not:

- own `track.world` in baseline mode
- emit `NOESIS.OBJECT_DEPTH`
- replace the pose-first anchor chain
- act as a runtime fallback for missing DAv2 registration

Baseline room-relative tracking remains:

- pose-first anchor authority
- DAv2 object depth on the same pose ray
- optional offline DAv2 -> MapAnything registration correction
- one fused backend world estimator

## Quick Validation

- `python3 scripts/ma_depth_rpc_smoke_test.py --no-spawn`
  - validates live `ma_depth_response`
- `python3 scripts/build_depth_registration.py --help`
  - validates builder surface
- `timeout 25s python3 noesis/ds8_runtime.py --pgie-profile yolo26_seg --size s --disable-rest`
  - proves DS8 can start with the current registration artifact
