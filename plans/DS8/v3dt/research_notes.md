# SV3DT + MV3DT Research Notes (DeepStream 8)

This document is a consolidated deep-dive on **SV3DT** and **MV3DT** as implemented in DeepStream 8, with an emphasis on: how they fit into a pipeline, how they are configured, what metadata they output, and what is realistically accessible from Python in a DS8 (Service Maker) deployment.

## 0) Current implementation addendum (2026-07-11)

The January baseline below remains useful historical tuning evidence, but it is
not the current coordinate contract. The active DS8, protected reimplementation,
and DS9 V3DT profiles now share `config/camera_calibration.json`, whose camera
centers are separated in one metric `backend_world_m` frame. Their locked
camInfo files are byte-identical and use `WORLD_AXES=xzy`.

The tracker-owned `bbox3d` and `velocity3d` values remain diagnostic values in
the profile-specific tracker tuple. At the producer boundary, Noesis derives
the tracker ground endpoint using `zCentre - zLen / 2`, applies the exact `xzy`
signed-permutation contract, and only then publishes `world` in canonical Y-up
`backend_world_m`. Missing or malformed axis metadata fails closed; it does not
fall back to a ray-projected V3DT world point.

This completes the static shared-world and axis correction for SV3DT. It does
not prove MV3DT overlap fusion. MV3DT still requires one occupied, synchronized
kitchen/family-room session that exercises peer association and fused output.

### Historical locked baseline (2026-01-22)

The current SV3DT baseline is locked and summarized in:
`plans/DS8/v3dt/status_summary_2026-01-22_locked_baseline.md`.

Key points to carry into any new work:

- **Working baseline config:**  
  `config/infer_v3dt_baseline.yaml`
- **Per-camera pitch (preview extrinsics):** family-room **-16**, kitchen **-21**, living-room **-15**
- **Model height:** **2.2 m** (2.4/2.6 degraded family-room)
- **CamInfo conventions:** `w2p`, `INVERT_E=0`, `Y_FLIP=1`, `WORLD_AXES=xzy`, `WORLD_SCALE=1`
- **Confirmed no-go:** do not enable PGIE aspect ratio/padding; do not change `s_obj_to_m` away from `1.0`

Those dated shortfalls describe the January tuning snapshot. See the summary
document for its exact historical gaps and non-negotiable projection constraints.

## 1) Where SV3DT/MV3DT live in DeepStream

- **GStreamer element:** `nvtracker`
- **Low-level tracker library:** typically `libnvds_nvmultiobjecttracker.so`
- **Enablement mechanism:** tracker **low-level YAML config** (`nvtracker` property `ll-config-file`)

In other words: SV3DT/MV3DT are **not separate GStreamer elements** in the classic video pipeline sense; they are **modes/features inside the low-level tracker library** that `nvtracker` hosts.

DeepStream documentation confirms SV3DT is enabled through a new low-level section `ObjectModelProjection` and state estimator type `3`, and MV3DT is enabled via `MultiViewAssociator` + `Communicator`.

## 2) SV3DT (Single‑View 3D Tracking)

### 2.1 What SV3DT does

SV3DT runs tracking in a **3D world coordinate system** rather than purely in 2D image space. Its key idea is:

- Use a **camera projection matrix** (3×4) + a **3D object model** to estimate where a target is in 3D, especially the **foot location on the ground plane**.
- Use that 3D state (foot position + velocity) for tracking, which makes it substantially more robust under:
  - partial occlusions (only upper body visible)
  - truncated / clipped detections
  - abrupt bbox size changes

SV3DT can also **recover full-body boxes** from partial detections by fitting the 3D model and projecting it back into the image.

### 2.2 SV3DT’s required inputs

Per camera/stream you must provide a **camera info YAML** (“camInfo”) that includes:

- A **3×4 projection matrix**:
  - `projectionMatrix_3x4` (DeepStream assumes principal point at (0,0) and internally adds `(img_w/2, img_h/2)`), or
  - `projectionMatrix_3x4_w2p` (world→pixel; matrix already includes principal point offset, origin at top-left; no extra translation).
- A **human model** (`modelInfo`) with at least:
  - `height`
  - `radius` (cylinder radius, or half bottom-edge for cuboid)

DeepStream docs clarify the difference between `projectionMatrix_3x4` vs `_w2p` in “The 3x4 Camera Projection Matrix” section.

#### 2.2.1 Building `camInfo_*.yml` from Noesis calibration (practical recipe)

Noesis already stores the ingredients needed to generate camInfo files:

- Camera models + heights: `config/cameras.yaml`
- Intrinsics: `config/cameras.yaml` (and/or `intrinsics.json`)
- Extrinsics: `config/camera_calibration.json` (`E` is stored as **column-major**, and the canonical convention is **world→camera**)
- Tracker pixel space: `config/infer.yaml` streammux `width/height` (currently 1920×1080 for all streams; family-room is scaled up from 1280×720)

The active locked profiles use `projectionMatrix_3x4_w2p`; the generator also
supports `projectionMatrix_3x4`. For either variant, compute per camera:

- Reshape `E` into a 4×4 matrix with `order="F"` (column-major).
- Treat `E` as **World→Camera** and extract `[R|t] = E[:3, :]` (3×4).
- Build the intrinsics `K` *in the tracker pixel coordinates*:
  - For cameras already at 1920×1080 (living-room, kitchen): use `fx,fy,cx,cy` as-is.
  - For family-room (`unifi_g4_instant`) where intrinsics are defined at 1280×720 but streammux is 1920×1080:
    - scale `fx,fy,cx,cy` by 1.5.
  - If streammux ever introduces letterboxing (aspect mismatch + `enable-padding=1`), incorporate the pad offsets into `cx,cy` before forming `K`.
- Compute `P = K @ [R|t]` (3×4).

Note: If your calibration source provides `Twc` (Camera→World) instead, you must invert it first to obtain `E` (World→Camera) before building camInfo.
- Flatten row-major 12 floats into `projectionMatrix_3x4_w2p`.
  - For `projectionMatrix_3x4`, flatten the same way but keep the principal point zero-centered (DeepStream shifts internally).

The current `config/camera_calibration.json` produces one shared metric world
frame across all three cameras. That is necessary for MV3DT but is not, by
itself, evidence that overlap synchronization or peer fusion works live.

### 2.3 SV3DT configuration knobs (confirmed)

In the tracker low-level YAML:

- Enable SV3DT:
  - `ObjectModelProjection.cameraModelFilepath: [<camInfo0>, <camInfo1>, ...]`
  - `StateEstimator.stateEstimatorType: 3` (documented as `SIMPLE_LOCATION_KF`)
- Optional (high-value) SV3DT outputs:
  - `ObjectModelProjection.outputVisibility: 1`
  - `ObjectModelProjection.outputFootLocation: 1` (also controls 3D bbox output)
  - `ObjectModelProjection.outputConvexHull: 1`
- Pose integration for SV3DT:
  - `PoseEstimator` section (BodyPose3DNet) and `poseEstimatorType: 1`
  - SV3DT uses **2D keypoints** (from pose estimator) as anchors to estimate per-person height and refine 3D fit.

The NVIDIA MV3DT reference tracker config includes additional, verified keys that are safe to use (but must be tuned for your scene), including:
- `StateEstimator.processNoiseVar4Loc`, `processNoiseVar4Vel`, `measurementNoiseVar4Detector`, `measurementNoiseVar4Tracker`
- `ObjectModelProjection.objectModelType` (reference sets `0`)

DeepStream doc notes:
- The pose model’s 3D keypoints are in target-local coordinates; SV3DT currently uses only 2D keypoints for world estimation.
- There is a configurable `waistHeightRatio` in `ObjectModelProjection` (upper-body proportion assumption).

### 2.4 What SV3DT outputs (misc/user meta)

When enabled, `nvtracker` can attach the following per-object outputs as **NvDsUserMeta**:

- **Visibility** (a float):
  - `NVDS_OBJ_VISIBILITY`
- **Foot location in image coordinates** (float[2]):
  - `NVDS_OBJ_IMAGE_FOOT_LOCATION`
- **Foot location on world ground plane** (float[2]):
  - `NVDS_OBJ_WORLD_FOOT_LOCATION`
- **Convex hull in image coords** (2D polygon points):
  - `NVDS_OBJ_IMAGE_CONVEX_HULL`
- **3D bounding box in world coordinates** (`NvDsObj3DBbox`):
  - `NVDS_OBJ_3D_META`

This is verified in DeepStream’s tracker implementation at:
- `/opt/nvidia/deepstream/deepstream-8.0/sources/gst-plugins/gst-nvtracker/nvtracker_proc.cpp` (`updateObjectProjectionMeta`)

#### 2.4.1 `NvDsObj3DBbox` contents

`NvDsObj3DBbox` (the payload of `NVDS_OBJ_3D_META`) contains:

- Center: `xCentre, yCentre, zCentre`
- Dimensions: `xLen, yLen, zLen`
- Rotation (radians): `xRot, yRot, zRot`
- Velocity (m/s or “world units per second”): `xVel, yVel, zVel`

This is defined in:
- `/opt/nvidia/deepstream/deepstream-8.0/sources/includes/nvds_tracker_meta.h`

DeepStream sample app `deepstream-test5` demonstrates extracting `NVDS_OBJ_3D_META` from `obj_user_meta_list` and copying its fields into schema output:
- `/opt/nvidia/deepstream/deepstream-8.0/sources/apps/sample_apps/deepstream-test5/deepstream_test5_app_main.c` (`generate_event_msg_meta`)

## 3) MV3DT (Multi‑View 3D Tracking; Developer Preview)

### 3.1 What MV3DT does (conceptually)

MV3DT builds on top of SV3DT and adds **distributed multi-camera collaboration** for overlapping FoVs:

- **Global ID assignment + propagation**:
  - Cameras negotiate and assign **globally unique target IDs** via MQTT messaging.
  - Global IDs propagate among “vision neighbors” (overlapping cameras) to preserve identity through occlusions and handovers.
- **Multi-view measurement fusion**:
  - Cameras exchange short 3D tracklets (positions/velocities over recent frames) and fuse peer measurements to improve robustness and reduce occlusion failures.

### 3.2 MV3DT requirements (non-negotiable)

MV3DT requires:

- **Shared global world coordinate system** across cameras:
  - All cameras’ SV3DT 3D measurements must align in the same frame (meters recommended).
- A **vision neighbor graph**:
  - only overlapping FoVs should be connected (to avoid false cross-camera associations).
- **MQTT connectivity**:
  - MV3DT uses an MQTT communicator (“peer-to-peer via broker”), even when cameras are in the same machine.

### 3.3 MV3DT config knobs (confirmed from NVIDIA docs + reference configs)

In the tracker low-level YAML:

- Enable MV3DT:
  - `MultiViewAssociator` section
  - `Communicator` section
- Key MV3DT knobs (documented):
  - `maxPeerTrackletSize`
  - `recentlyActiveAge`
  - `minCommonFrames4MatchScore`
  - `minPeerToPredDistance4Fusion`
  - `minPeerVisibility4Fusion`
  - `minPeerTrackletMatchScore`
- MQTT communicator selection (documented):
  - `Communicator.communicatorType: 2` (MQTT)
  - `Communicator.pubSubInfoConfigPath: <path>`
  - `Communicator.mqttProtoAdaptorConfigPath: <path>`

NVIDIA’s MV3DT reference config also toggles advanced features:
- `enableLatePeerReAssoc`
- `enableIDCorrection`
- `enableSeeThrough`
- `enableMsgSync`

Those toggles appear in NVIDIA reference configs, but are not fully enumerated in the short parameter tables.

#### 3.3.1 Verified MV3DT keys from NVIDIA reference config (ground truth)

The following keys/sections are present in NVIDIA’s `deepstream-tracker-3d-multi-view` template tracker config:
- `/tmp/deepstream_reference_apps/deepstream-tracker-3d-multi-view/config_templates/config_tracker.yml`

Key takeaways:
- `MultiViewAssociator.multiViewAssociatorType: 1` exists (reference uses `1`).
- The advanced toggles (`enableLatePeerReAssoc`, `enableIDCorrection`, `enableSeeThrough`, `enableMsgSync`) live under `MultiViewAssociator`.
- MV3DT MQTT wiring is done via:
  - `Communicator.communicatorType: 2`
  - `Communicator.pubSubInfoConfigPath: <pub_sub_info_config_0.yml>`
  - `Communicator.mqttProtoAdaptorConfigPath: <config_mqtt.txt>`
- Pose is enabled via a `PoseEstimator` section with `poseEstimatorType: 1` and BodyPose3DNet ONNX/engine paths.

### 3.4 MV3DT advanced features (how to think about them)

From MV3DT documentation:

- **Peer-target re-association**:
  - Matches tracklets across cameras using overlapping timestamps + 3D foot locations.
- **LatePeerReAssoc**:
  - Allows adopting a peer ID after a short delay (targets “recently active”), mitigating missed early matches.
- **ID Correction**:
  - Detects and fixes incorrect associations by re-checking tracklet similarity constraints.
- **Quasi-active tracking**:
  - Keep tracking using peer measurements when ego camera has weak/no measurement.
- **See-through tracking**:
  - Initiate new tracks based on peer evidence even when fully occluded in ego camera.

## 4) MQTT / pub-sub config file format (what MV3DT expects)

NVIDIA’s `deepstream-tracker-3d-multi-view` repo generates `pub_sub_info_config_0.yml` with:

- `pubBrokerTopicStr`: list of `<broker_host:port>;<topic>` strings, one per camera stream in the DS instance.
- `subPeerBrokerTopicStrs`: list-of-lists, where each element corresponds to a camera stream and contains its neighbor subscriptions, each as `<broker_host:port>;<topic>`.

This format is visible in:
- `/tmp/deepstream_reference_apps/deepstream-tracker-3d-multi-view/utils/generate_pub_sub_configs.py`
- `/tmp/deepstream_reference_apps/deepstream-tracker-3d-multi-view/utils/deepstream_auto_configurator.py` (`topic_template: "/trck/cam%d"`, broker `"127.0.0.1:1883"`)

For your deployment, we will keep **only the overlap pair** as subscribers (kitchen ↔ family-room), and keep living-room unsubscribed initially.

## 5) Python access to MV3DT/SV3DT outputs in DS8

### 5.1 Service Maker vs pyds reality

In DS8 Service Maker pipelines, you still ultimately have `NvDsBatchMeta / NvDsObjectMeta` behind the scenes. However, the **Service Maker Python metadata wrappers** (`pyservicemaker._pydeepstream.*`) do **not** expose arbitrary per-object user-meta lists (no `obj_user_meta_list` on `ObjectMetadata`), so you cannot directly read `NVDS_OBJ_3D_META` from pure-Python DS8 hooks.

Practical observation:
- Service Maker C++ metadata API *does* support iterating object user meta by meta type (see `ObjectMetadata::iterate(..., meta_type)` and `Object3DBBoxUserMetadata` in `/opt/nvidia/deepstream/deepstream/service-maker/includes/metadata.hpp`).
- `pyds` **does** expose `NVDS_OBJ_3D_META` and `NvDsObj3DBbox`, but it requires access to the underlying `NvDsObjectMeta` / `obj_user_meta_list`, which the DS8 Python wrappers do not provide.

Implication for Noesis DS8:
- Use a minimal native bridge module (`noesis_v3dt_meta_ext`, built via `scripts/build_noesis_v3dt_meta_ext.sh`) to call the Service Maker C++ iterators and return `bbox3d`/`velocity3d` into Python hooks without `nvmsgconv` and without repurposing preprocess meta.

But `pyds` in this environment does **not** expose constants/classes for:
- `NVDS_OBJ_VISIBILITY`
- `NVDS_OBJ_IMAGE_FOOT_LOCATION`
- `NVDS_OBJ_WORLD_FOOT_LOCATION`
- `NVDS_OBJ_IMAGE_CONVEX_HULL`
- `NvDsObjConvexHull`

Implication:
- We can reliably extract **3D bbox (and velocity)** via `NVDS_OBJ_3D_META`.
- If we want visibility / footpoint / convex hull in Python, we either:
  - derive them from the 3D bbox (preferred for minimal risk), or
  - extend/patch `pyds` bindings to include those meta types/structs (higher effort, but possible).

### 5.2 What we can leverage immediately in Noesis

Using only `NvDsObj3DBbox`, we can produce:

- `world` / `position3d`:
  - footpoint ≈ `(xCentre, yCentre - 0.5*yLen, zCentre)`
  - or keep center if that’s what the algorithm outputs in practice (must be validated)
- `velocity3d`:
  - `(xVel, yVel, zVel)`
- `bbox3d`:
  - center, dimensions, rotation

This is enough to:
- improve BEV stability (no bbox-ray intersection)
- add true world-space speed/heading analytics
- inform StableID matching (3D gating)

## 6) What “success” looks like for your house deployment

With good global calibration + SV3DT + MV3DT:

- **Kitchen ↔ Family-room overlap**:
  - The same person gets the same internal MV3DT global tracker ID in both cameras.
  - Quasi-active / see-through behaviors reduce ID switches during occlusions.
  - 3D position estimates (meters, Y-up) are consistent between the two views.
- **Living-room ↔ Kitchen adjacency**:
  - MV3DT is not relied upon (no overlap).
  - StableIDManager + (optional) 3D gating provides robust handoff without false merges.

## References (primary)

- DeepStream nvtracker docs (SV3DT, meta outputs, pose):
  - https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html
- DeepStream MV3DT developer preview:
  - https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_MV3DT.html
- NVIDIA reference apps:
  - https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/master/deepstream-tracker-3d
  - https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/master/deepstream-tracker-3d-multi-view
