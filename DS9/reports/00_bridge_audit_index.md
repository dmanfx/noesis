# DS9 Bridge Audit Index

This index tracks the DS9 exposure questions before any live DS8 path is
changed. A DS9 replacement is accepted only when official DS9 docs or installed
DS9 headers/modules prove equivalent behavior.

| Report | Owner Scope | Status | Final decision |
| --- | --- | --- | --- |
| `01_servicemaker_pyds_intrinsics_latency.md` | Service Maker, PyDS, intrinsics, latency | Complete | Keep Service Maker as canonical; quarantine PyDS raw metadata paths; use DS9 OTel as default latency; route intrinsics through the calibration provider. |
| `02_pose_bridge.md` | Pose metadata bridge and pose parser | Complete | Keep/rebuild `noesis_pose_meta_ext`; keep tensor-output pose SGIE config; remove the no-op YOLO26 pose parser from the active DS9 target. |
| `03_depth_bridges.md` | Object-depth and DAv2 tensor bridges | Complete | Keep/rebuild `noesis_depth_meta_ext` and `noesis_depth_tracking_tensor_ext`; validate DAv2 device tensors and `disable-output-host-copy=1` under DS9. |
| `04_reid_v3dt_bridges.md` | ReID and V3DT metadata bridges | Complete | Keep/rebuild `noesis_reid_meta_ext` and `noesis_v3dt_meta_ext`; treat tracker-internal ReID as a later experiment, not a first-port replacement. |
| `05_roi_exclude.md` | ROI pruning and `nvdsroiexclude` | Complete | Keep pre-tracker ROI pruning semantics; reconstruct or recover `nvdsroiexclude` source; do not replace it with `nvdsanalytics` or Python pruning. |
| `06_custom_parsers.md` | Custom `nvinfer` parsers | Complete | Keep/rebuild YOLO26-Seg, YOLO11-Seg, and RF-DETR-Seg parsers; remove the YOLO26 pose no-op parser from DS9 build/runtime planning. |
| `07_custom_python_hooks.md` | MapAnything, hooks, BEV, overlays | Complete | Keep Service Maker metadata hooks, MapAnything tensor branch, overlays, and BEV JSON; quarantine raw PyDS and silent native fallback paths. |
| `08_no_install_transition_prep.md` | No-install DS9 prep work | Complete | Added DS9 rebuild guardrails, active parser build guards, reconstructed `nvdsroiexclude` source, and documented the DS9-header blocker for actual rebuilds. |

Working rule: not proven means keep or rebuild the existing bridge/parser for
the first DS9 transition.

All first-port audit/prep reports are complete. The consolidated decision
ledger is [`../DS9_PREP_DECISIONS.md`](../DS9_PREP_DECISIONS.md).
