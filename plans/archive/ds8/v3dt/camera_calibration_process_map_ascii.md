# Camera Calibration — Process Map (ASCII)
_Generated: 2025-12-31_

Legend:
  [file]            on-disk config/artifact
  (WS msg)          WebSocket JSON message type/cmd
  {component}       runtime component / major module
  -->               data/control flow
  === writes ===>   persists/overwrites on disk

--------------------------------------------------------------------------------
Quick Summary (layman)
--------------------------------------------------------------------------------

Basic idea:

  +------------------------+                     +------------------------------------+
  | config/cameras.yaml    |                     | DS8 runtime + WebSocket server     |
  | “lens settings” (K)    |-------------------->| builds + broadcasts calibration     |
  +------------------------+                     | bundle (K + pose + alignment)      |
                                                 +------------------+-----------------+
                                                                    |
  +------------------------+                                        |  (sent on connect + after updates)
  | Menon (UI)             |                                        v
  | - set_extrinsics       |------------------------------->   (WS msg) calibration-bundle
  |   “camera pose” (E)    |
  | - set_align            |------------------------------->   [config/ply_alignment.json]
  |   “scale + floor”      |
  | - pixel_to_world        \------------------------------->   “click pixel -> world point”
  |   (query only)          \---> reply: pixel_to_world_response
  | - {cmd:"calibrate"}     \------------------------------->   triggers “auto-cal” (compat)
  +------------------------+

  +------------------------+
  | MapAnything depth       |-----> (optional) “auto-calibrate from depth”
  | snapshots per camera    |       (estimates a rough pose; not Menon’s absolute pose)
  +------------------------+              |
                                         === writes ===> [config/camera_calibration.json]

  Calibration outputs are then used for:
    - overlays / BEV projection / track footpoints
    - pixel_to_world conversions for UI tools
    - V3DT camInfo regeneration: === writes ===> config/v3dt/camInfo_*.yml

  Legacy note:
    - The deprecated pre-DS8 runtime (`main.py`) also reads/writes the same calibration files and serves similar WS messages,
      but DS8 is the canonical path for new work.

--------------------------------------------------------------------------------
0) Sources of Truth (as the code works today)
--------------------------------------------------------------------------------

  Intrinsics (K):
    Primary: [config/cameras.yaml] intrinsics_models + cameras[*].model
    Secondary/legacy: [intrinsics.json] + (config.py -> CalibrationSettings)

  Extrinsics (E, pose):
    Manual/authoritative: (WS msg) set_extrinsics from Menon
    Auto/fallback: (WS msg) auto_calibrate_pose OR (WS cmd) {cmd:"calibrate"} using MapAnything depth

  Alignment / scale / floor height (align):
    (WS msg) set_align  === writes ===> [config/ply_alignment.json]

  Broadcast bundle to clients:
    (WS msg) calibration-bundle  (assembled from K/E/align and sent on connect + updates)

--------------------------------------------------------------------------------
1) DS8 (canonical) Calibration Path — noesis/ds8_runtime.py + websocket_server.py
--------------------------------------------------------------------------------

  Startup:
    [config/cameras.yaml] --> {noesis/ds8_runtime.py:_load_camera_labels} --> camera_labels
                       \\--> {noesis/metadata/intrinsics.py:CameraConfigLoader} --> intrinsics per source_id
    [intrinsics.json] + (config.py calibration settings) --> {calibration_bundle.py:assemble_calibration_bundle}
    [config/camera_calibration.json] --> {calibration_bundle.py:load_extrinsics}
    [config/ply_alignment.json]     --> {calibration_bundle.py:load_alignment}
         --> {noesis/ds8_runtime.py:_CalibrationProvider} --> cached snapshots + calibration_bundle()

  WebSocket connect:
    {websocket_server.py:WebSocketServer.handle_client}
      --> calls ws_server.calibration_getter()
      --> sends (WS msg) calibration-bundle to the client (if non-empty)

  Menon/manual pose update:
    Menon UI --> (WS msg) set_extrinsics {cameraId, E|Twc}
      --> {websocket_server.py} dispatch
          --> ws_server.set_extrinsics_handler(req)
              --> {noesis/ds8_runtime.py:_ds8_set_extrinsics_handler}
                  --> cameraId resolution (supports string OR {id,name,source} object)
                  === writes ===> [config/camera_calibration.json] cameras[<camera-name>].E
                  --> reload extrinsics + refresh bundle
                  --> broadcast (WS msg) calibration-bundle
      --> replies (WS msg) set_extrinsics_result {ok:true|false,...}

  Menon/manual alignment update:
    Menon UI --> (WS msg) set_align {align:{matrix[16], floor_y, units:{s_obj_to_m}}}
      --> {websocket_server.py} dispatch
          --> ws_server.set_align_handler(req)
              --> {noesis/ds8_runtime.py:_ds8_set_align_handler}
                  === writes ===> [config/ply_alignment.json]
                  --> reload alignment + refresh bundle
                  --> broadcast (WS msg) calibration-bundle
      --> replies (WS msg) set_align_result {ok:true|false,...}

  Coordinate query (uses K + E + align):
    Menon UI --> (WS msg) pixel_to_world {cameraId, u, v, depth?}
      --> {websocket_server.py} dispatch
          --> ws_server.pixel_to_world_handler(req)
              --> {noesis/ds8_runtime.py:_ds8_pixel_to_world_handler}
                  --> snapshot() -> K (scaled to streammux) + E(world->cam) + floor_y + unit_scale
                  --> if depth provided: project pixel ray to depth, then cam->world
                  --> else: intersect pixel ray with floor plane at Y=floor_y
      --> replies (WS msg) pixel_to_world_response {ok, world_point, method, ...}

  Auto-calibration trigger from Menon (legacy compat):
    Menon UI --> (WS cmd) {cmd:"calibrate", cameraId}
      --> {websocket_server.py} compat handler
          --> if cmd arrives <2s after set_extrinsics for same camera:
                - SKIP depth-based auto-calibration (guardrail against overwriting Menon pose)
                - (optionally) send calibration-bundle + calibration-complete + auto_calibrate_result{skipped:true}
          --> else:
                --> ws_server.auto_calibrate_handler(cameraId)
                    --> {noesis/ds8_runtime.py:_ds8_auto_calibrate_handler}

  Auto-calibration (depth-based pose estimation):
    Any client --> (WS msg) auto_calibrate_pose {cameraId?}
      --> {websocket_server.py} dispatch --> ws_server.auto_calibrate_handler(cameraId)
          --> {noesis/ds8_runtime.py:_ds8_auto_calibrate_handler}
              --> enables MapAnything depth “burst” (pipeline valve)
              --> waits for fresh depth snapshots in storage
              --> calls {scripts/auto_calibrate_from_depth.py:auto_calibrate_from_latest_depth(persist=False)}
              === writes ===> [config/camera_calibration.json] for each camera that succeeded
              --> reload extrinsics + refresh bundle
              --> broadcast (WS msg) calibration-bundle
      --> replies (WS msg) auto_calibrate_result {ok, results[], updated[], error?}

--------------------------------------------------------------------------------
2) MapAnything Depth → Auto-Calibrate (what it actually computes)
--------------------------------------------------------------------------------

  DS8 pipeline (MapAnything SGIE) -> {hooks.attach_mapanything_postprocess_hook}
    --> {storage_manager} stores latest depth/conf/mask per camera (snapshots)

  Auto-calibration script:
    {scripts/auto_calibrate_from_depth.py}
      Inputs:
        - latest depth/conf/mask (meters) from MapAnything snapshots
        - intrinsics K for that camera (intrinsics.json/config settings, fallback cameras.yaml)
      Computes (per camera):
        - fits a floor plane in camera space from 3D points
        - estimates orientation (pitch/roll) and camera height above floor
        - returns an E(world->camera) in a *canonicalized* “depth world” frame:
            * world origin fixed at (x=0,z=0) under the camera (no lateral translation)
            * yaw is normalized so camera forward aligns to +Z
      Output keys include:
        ok, E, height_m, pitch_deg, roll_deg, plane_normal_cam, floor_ratio, floor_std, floor_score, reject_reason, ...

  Implication:
    This depth auto-calibration is a *fallback pose estimator*; it cannot reproduce
    Menon’s absolute real-world camera positions/orientations (especially X/Z and yaw).

--------------------------------------------------------------------------------
3) Deprecated Calibration Path — main.py + websocket_server.py
--------------------------------------------------------------------------------

  Startup (deprecated path):
    {main.py} loads calibration bundle (intrinsics + extrinsics + align)
      --> wires WS handlers:
          - calibration_getter
          - set_extrinsics_handler
          - set_align_handler
          - pixel_to_world_handler
          - auto_calibrate_handler

  set_extrinsics (deprecated path):
    (WS msg) set_extrinsics --> {main.py:_set_extrinsics_rpc}
      === writes ===> [config/camera_calibration.json]
      --> rebuilds calibration-bundle + broadcasts it

  auto_calibrate_pose (deprecated path):
    (WS msg) auto_calibrate_pose --> {main.py:_auto_calibrate_from_depth}
      --> runs scripts/auto_calibrate_from_depth.py (persist=False)
      --> applies each E via _set_extrinsics_rpc (so it persists + broadcasts)

  pixel_to_world (deprecated path):
    (WS msg) pixel_to_world --> {main.py:_pixel_to_world_rpc}
      --> prefers MapAnything depth pixel sample (if available + confident)
      --> else falls back to floor-plane intersection
      --> applies align matrix + units scale via geometry/transform.py

--------------------------------------------------------------------------------
4) Derived Artifacts / Consumers (where calibration is “used”)
--------------------------------------------------------------------------------

  A) Web UI + Menon:
     - Consumes (WS msg) calibration-bundle (K/E/align) for visualization + tools.
     - Calls pixel_to_world for user interactions (click → world point).

  B) BEV / overlays (DS8):
     - {noesis/} hooks + {BevRenderer} use calibration snapshots to project tracks.

  C) V3DT (SV3DT/MV3DT) projection files:
     - DS8 startup hook {noesis/ds8_runtime.py:_maybe_sync_v3dt_caminfo}
         reads [config/camera_calibration.json] + [config/cameras.yaml]
         === writes ===> [config/v3dt/camInfo_<camera>.yml] (projectionMatrix_3x4 / projectionMatrix_3x4_w2p)

--------------------------------------------------------------------------------
5) Where the “duplicate / overlapping” paths are
--------------------------------------------------------------------------------

  1) Two writers to the same pose store:
       Menon/manual set_extrinsics  === writes ===> [config/camera_calibration.json]
       Depth auto_calibrate_pose    === writes ===> [config/camera_calibration.json]
     If both run, last-write-wins (and depth auto-cal can overwrite real-world Menon poses).

  2) Two intrinsics sources:
       [config/cameras.yaml] intrinsics_models
       [intrinsics.json] + CalibrationSettings model map/specs

  3) Alignment is separate from pose:
       [config/ply_alignment.json] is applied on top of camera/world transforms for
       “aligned world” coordinates (and unit conversion).
