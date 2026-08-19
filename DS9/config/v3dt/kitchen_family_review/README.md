# Kitchen-Family Room MV3DT profile

Status: accepted explicit runtime option. Select `--tracking-mode mv3dt` (or
`NOESIS_TRACKING_MODE=mv3dt`) to use `DS9/config/infer_mv3dt.yaml`. Baseline and
SV3DT are not modified by this profile; baseline remains the default.

The canonical MV3DT selector uses the same YOLO26-m detection PGIE as baseline.
The detector does not emit instance masks, so the replacement cuboid uses its
tracked-bbox bottom-center fallback for the image gravity anchor. The recorded
evaluation profile remains available for segmentation-specific comparisons.

On the native host, run the same supervisor used by the normal appliance with
an explicit lane selection:

```bash
python3 DS9/scripts/run_canonical_runtime_host.py check --tracking-mode mv3dt
python3 DS9/scripts/run_canonical_runtime_host.py run --tracking-mode mv3dt
```

Omitting `--tracking-mode` starts the unchanged baseline lane.

Family Room is the fixed gauge. Kitchen and Family Room use independently
reviewed static-camera anchors in one backend-world frame. Living Room retains
its accepted local SV3DT geometry and has no cross-camera peer edge. Its MQTT
self-topic exists only because the ordered DS9.1 communicator blocks a batched
tracker when a stream has an empty peer entry; the self-loop exposes no other
camera's measurements or IDs to Living Room.

The production profile uses the accepted binding in
`geometry_binding_accepted.json`. The sibling `geometry_binding.json` and
`infer_mv3dt_kitchen_family_review.yaml` preserve the recorded-input evaluation
lane and still require `NOESIS_MV3DT_EVALUATION=1`.

The accepted tuning is deliberately profile-local:

- all communicators share the MQTT connection and remain non-threaded so every
  stream is online before the first synchronized batch;
- the person cylinder is 1.7 m, matching the observed foot projection in these
  rooms;
- two common frames, a 0.18 peer score, a 4.75 m peer-prediction safety radius,
  and 0.05 minimum peer visibility retain brief occluded doorway handoffs;
- two-frame probation lets short overlap tracks participate in late peer
  reassociation;
- Noesis treats MV3DT's batch-global object ID as one legacy StableID-manager
  key, while baseline and SV3DT keep their historical per-camera keys.

Validation used three frame-aligned, equal-length July clips and two May
multi-person spans. The July replay produced native shared IDs in all three
Kitchen/Family doorway episodes, including late reassociation after the peer
left view. The multi-person spans produced shared native IDs only on visually
confirmed same-person pairs; the separate person and partial-body duplicate
tracks were not merged. A rendered 22-second mosaic was inspected at near,
far, doorway, and full-body positions; that segmentation-PGIE run kept the
replacement cuboid bottom-face centroid on the mask feet/gravity anchor. The
canonical YOLO26-m fallback must be judged from the live occupied view.

Both recorded evaluation and live opt-in profiles require complete batches
(`batched-push-timeout: -1`) with `sync-inputs: 0` and
`BaseConfig.useBatchNumForFrameId: 1`. The RTSP encoders do not expose one
shared PTP/NTP timestamp domain, so timestamp synchronization remains disabled;
the mux instead waits for one current frame from every camera and the tracker
batch number supplies their common MV3DT frame ID. A finite live timeout can
emit a partial batch when one encoder jitters and deadlock the ordered peer
message synchronizer. A camera that stops producing is handled by Noesis's
bounded per-source progress recovery and supervisor restart rather than by
feeding an incomplete batch to MV3DT.
