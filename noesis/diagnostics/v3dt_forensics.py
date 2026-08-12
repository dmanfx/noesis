from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from noesis.diagnostics.telemetry_log import public_v3dt_environment
from noesis_core.runtime_secrets import public_pipeline_config

_ENV_TRUE = {"1", "true", "yes", "y", "on"}


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
    except Exception:
        return default
    if not math.isfinite(num):
        return default
    return num


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _scale_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    target_w: int,
    target_h: int,
) -> Tuple[float, float, float, float, float, float, float, float]:
    base_w = 2.0 * float(cx) if cx else 0.0
    base_h = 2.0 * float(cy) if cy else 0.0
    scale_x = float(target_w) / base_w if base_w else 1.0
    scale_y = float(target_h) / base_h if base_h else 1.0
    return fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y, base_w, base_h, scale_x, scale_y


def _axis_mapping_from_env() -> Optional[np.ndarray]:
    spec = str(os.environ.get("NOESIS_V3DT_CAMINFO_WORLD_AXES", "xzy") or "").strip().lower()
    if not spec or spec == "xyz":
        return None

    tokens = [tok for tok in spec.replace(",", " ").split() if tok]
    if len(tokens) == 1 and len(tokens[0]) == 3 and all(ch in "xyz" for ch in tokens[0]):
        tokens = list(tokens[0])

    if len(tokens) != 3:
        return None

    basis = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    used = set()
    cols = []
    for token in tokens:
        sign = -1.0 if token.startswith("-") else 1.0
        axis = token.lstrip("+-")
        if axis not in basis or axis in used:
            return None
        used.add(axis)
        cols.append(sign * basis[axis])
    return np.stack(cols, axis=1)


def _apply_axis_mapping(E_col_major: Sequence[float], axis_map: Optional[np.ndarray]) -> np.ndarray:
    E = np.array(E_col_major, dtype=np.float64).reshape((4, 4), order="F")
    if axis_map is None:
        return E
    axis_map_4 = np.eye(4, dtype=np.float64)
    axis_map_4[:3, :3] = axis_map
    return E @ axis_map_4


def _camera_pose_from_E(E_col_major: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    E = np.array(E_col_major, dtype=np.float64).reshape((4, 4), order="F")
    R_cw = E[:3, :3]
    t_cw = E[:3, 3]
    R_wc = R_cw.T
    C_world = -R_wc @ t_cw
    return R_wc, C_world


def _yaw_pitch_roll(R_wc: np.ndarray) -> Tuple[float, float, float]:
    forward = R_wc @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    yaw = math.degrees(math.atan2(float(forward[0]), float(forward[2])))
    fy = max(-1.0, min(1.0, float(forward[1])))
    pitch = math.degrees(math.asin(fy))
    R_cw = R_wc.T
    up_cam = R_cw @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    roll = math.degrees(math.atan2(float(up_cam[0]), float(up_cam[1])))
    return yaw, pitch, roll


def _ray_floor_intersection(
    K: np.ndarray,
    E_col_major: Sequence[float],
    floor_y: float,
    u: float,
    v: float,
) -> Dict[str, Any]:
    try:
        R_wc, C_world = _camera_pose_from_E(E_col_major)
        uv1 = np.array([u, v, 1.0], dtype=np.float64)
        Kinv = np.linalg.inv(K)
        dir_cam = Kinv @ uv1
        norm = float(np.linalg.norm(dir_cam))
        if norm <= 1e-9:
            return {"ok": False, "reason": "invalid_ray"}
        dir_cam = dir_cam / norm
        dir_world = R_wc @ dir_cam
        denom = float(dir_world[1])
        if abs(denom) < 1e-9:
            return {"ok": False, "reason": "parallel"}
        t = (float(floor_y) - float(C_world[1])) / denom
        hit = C_world + float(t) * dir_world
        return {
            "ok": True,
            "t": float(t),
            "hit": [float(hit[0]), float(hit[1]), float(hit[2])],
            "dir_world": [float(dir_world[0]), float(dir_world[1]), float(dir_world[2])],
            "origin": [float(C_world[0]), float(C_world[1]), float(C_world[2])],
        }
    except Exception as exc:
        return {"ok": False, "reason": f"error:{exc}"}


def _load_caminfo(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _projection_from_caminfo(caminfo: Dict[str, Any]) -> Tuple[str, Optional[np.ndarray]]:
    if "projectionMatrix_3x4" in caminfo:
        data = caminfo.get("projectionMatrix_3x4")
        key = "projectionMatrix_3x4"
    else:
        data = caminfo.get("projectionMatrix_3x4_w2p")
        key = "projectionMatrix_3x4_w2p"
    if not isinstance(data, list) or len(data) != 12:
        return key, None
    P = np.array(data, dtype=np.float64).reshape((3, 4))
    return key, P


def _projection_from_E(K: np.ndarray, E_col_major: Sequence[float]) -> np.ndarray:
    E = np.array(E_col_major, dtype=np.float64).reshape((4, 4), order="F")
    return K @ E[:3, :]


def _apply_y_flip(P: np.ndarray, img_h: int) -> np.ndarray:
    out = P.copy()
    out[1, :] = -P[1, :] + float(img_h) * P[2, :]
    return out


def _project_point(P: np.ndarray, xyz: Sequence[float]) -> Optional[Tuple[float, float]]:
    try:
        X = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0], dtype=np.float64)
        uvw = P @ X
        w = float(uvw[2])
        if abs(w) < 1e-9:
            return None
        u = float(uvw[0] / w)
        v = float(uvw[1] / w)
        return u, v
    except Exception:
        return None


def _collect_env_snapshot() -> Dict[str, str]:
    return public_v3dt_environment()


def build_snapshot(
    *,
    pipeline_path: Path,
    cameras_path: Path,
    calibration_path: Path,
    caminfo_dir: Path,
    alignment_path: Path,
    tracker_config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    pipeline_cfg = _load_yaml(pipeline_path)
    cameras_cfg = _load_yaml(cameras_path)
    calibration_cfg = _load_json(calibration_path)
    align_cfg = _load_json(alignment_path)

    streammux_cfg = pipeline_cfg.get("streammux", {}) if isinstance(pipeline_cfg, dict) else {}
    stream_w = _safe_int(streammux_cfg.get("width", 1920), 1920)
    stream_h = _safe_int(streammux_cfg.get("height", 1080), 1080)

    tracker_cfg = pipeline_cfg.get("tracker", {}) if isinstance(pipeline_cfg, dict) else {}
    tracker_w = _safe_int(tracker_cfg.get("tracker-width", tracker_cfg.get("tracker_width", stream_w)), stream_w)
    tracker_h = _safe_int(tracker_cfg.get("tracker-height", tracker_cfg.get("tracker_height", stream_h)), stream_h)
    tracker_cfg_path = tracker_config_path
    if tracker_cfg_path is None:
        tracker_cfg_value = str(
            tracker_cfg.get("config-file")
            or tracker_cfg.get("ll-config-file")
            or ""
        ).strip()
        tracker_cfg_path = Path(tracker_cfg_value) if tracker_cfg_value else None
    tracker_cfg_data = (
        _load_yaml(tracker_cfg_path)
        if tracker_cfg_path is not None and tracker_cfg_path.is_file()
        else {}
    )

    align = align_cfg or {}
    floor_y = _safe_float(align.get("floor_y", 0.0), 0.0)
    unit_scale = _safe_float((align.get("units") or {}).get("s_obj_to_m", 1.0), 1.0)

    intrinsics_models = cameras_cfg.get("intrinsics_models", {}) if isinstance(cameras_cfg, dict) else {}
    cameras = cameras_cfg.get("cameras", {}) if isinstance(cameras_cfg, dict) else {}
    calibration_cameras = calibration_cfg.get("cameras", {}) if isinstance(calibration_cfg, dict) else {}

    issues: List[Dict[str, Any]] = []
    per_camera: Dict[str, Any] = {}

    for cam_id, cam_info in sorted(cameras.items(), key=lambda item: int(item[0])):
        if not isinstance(cam_info, dict):
            continue
        cam_name = str(cam_info.get("name") or "")
        model_name = str(cam_info.get("model") or "")
        height_m = _safe_float(cam_info.get("height_m", 0.0), 0.0)
        if not cam_name or not model_name:
            continue

        model = intrinsics_models.get(model_name, {}) if isinstance(intrinsics_models, dict) else {}
        intr = model.get("intrinsics", {}) if isinstance(model, dict) else {}
        fx = _safe_float(intr.get("fx", 0.0))
        fy = _safe_float(intr.get("fy", 0.0))
        cx = _safe_float(intr.get("cx", 0.0))
        cy = _safe_float(intr.get("cy", 0.0))
        fx_s, fy_s, cx_s, cy_s, base_w, base_h, scale_x, scale_y = _scale_intrinsics(
            fx, fy, cx, cy, stream_w, stream_h
        )

        E = None
        if isinstance(calibration_cameras, dict):
            entry = calibration_cameras.get(cam_name) or {}
            if isinstance(entry, dict):
                E = entry.get("E")
        if not isinstance(E, list) or len(E) != 16:
            issues.append({
                "camera": cam_name,
                "level": "error",
                "code": "missing_extrinsics",
                "message": "camera_calibration.json missing E",
            })
            continue

        axis_map = _axis_mapping_from_env()
        E_caminfo = _apply_axis_mapping(E, axis_map)

        R_wc, C_world = _camera_pose_from_E(E)
        yaw, pitch, roll = _yaw_pitch_roll(R_wc)
        cam_height = float(C_world[1]) - float(floor_y)
        if height_m > 0.0 and abs(cam_height - height_m) > 0.5:
            issues.append({
                "camera": cam_name,
                "level": "warn",
                "code": "height_mismatch",
                "message": "camera height differs from cameras.yaml",
                "data": {"expected_m": height_m, "derived_m": cam_height},
            })

        caminfo_path = caminfo_dir / f"camInfo_{cam_name}.yml"
        caminfo = _load_caminfo(caminfo_path)
        caminfo_type, P_caminfo = _projection_from_caminfo(caminfo)

        use_w2p = caminfo_type.endswith("w2p")
        K_full = np.array([[fx_s, 0.0, cx_s], [0.0, fy_s, cy_s], [0.0, 0.0, 1.0]], dtype=np.float64)
        K_zero = np.array([[fx_s, 0.0, 0.0], [0.0, fy_s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        K = K_full if use_w2p else K_zero

        P_expected = _projection_from_E(K, E_caminfo.flatten(order="F").tolist())
        P_expected_flip = _apply_y_flip(P_expected, stream_h)
        caminfo_diff = None
        caminfo_diff_flip = None
        t_ratio = None
        if P_caminfo is not None:
            caminfo_diff = float(np.linalg.norm(P_caminfo - P_expected))
            caminfo_diff_flip = float(np.linalg.norm(P_caminfo - P_expected_flip))
            try:
                Rt = np.linalg.inv(K) @ P_caminfo
                t_caminfo = Rt[:, 3]
                t_E = np.array(E, dtype=np.float64).reshape((4, 4), order="F")[:3, 3]
                ratios = []
                for i in range(3):
                    if abs(float(t_E[i])) < 1e-9:
                        ratios.append(float("nan"))
                    else:
                        ratios.append(abs(float(t_caminfo[i] / t_E[i])))
                t_ratio = ratios
                for ratio in ratios:
                    if math.isfinite(ratio) and not (0.9 <= ratio <= 1.1):
                        if 90.0 <= ratio <= 110.0:
                            issues.append({
                                "camera": cam_name,
                                "level": "warn",
                                "code": "caminfo_scale_cm",
                                "message": "camInfo translation looks 100x larger (cm vs m)",
                                "data": {"ratio": ratios},
                            })
                        else:
                            issues.append({
                                "camera": cam_name,
                                "level": "warn",
                                "code": "caminfo_scale_mismatch",
                                "message": "camInfo translation ratio not near 1",
                                "data": {"ratio": ratios},
                            })
                        break
            except Exception:
                t_ratio = None

        bottom_u = float(stream_w) / 2.0
        bottom_v = float(stream_h - 1)
        ray = _ray_floor_intersection(K_full, E, floor_y, bottom_u, bottom_v)
        ray_flip = _ray_floor_intersection(K_full, E, floor_y, bottom_u, float(stream_h - 1 - bottom_v))
        if ray.get("ok") and isinstance(ray.get("t"), float):
            if float(ray["t"]) <= 0.0:
                issues.append({
                    "camera": cam_name,
                    "level": "warn",
                    "code": "floor_intersection_behind_camera",
                    "message": "bottom-center ray intersects floor behind camera (Y-axis mismatch)",
                    "data": {"t": ray.get("t"), "t_flip": ray_flip.get("t")},
                })
        elif ray.get("ok") is False:
            issues.append({
                "camera": cam_name,
                "level": "warn",
                "code": "floor_intersection_failed",
                "message": "failed to intersect floor for bottom-center ray",
                "data": {"reason": ray.get("reason")},
            })

        per_camera[cam_name] = {
            "camera_id": cam_id,
            "model": model_name,
            "height_m_expected": height_m,
            "intrinsics_raw": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "intrinsics_scaled": {"fx": fx_s, "fy": fy_s, "cx": cx_s, "cy": cy_s},
            "intrinsics_scale": {"base_w": base_w, "base_h": base_h, "scale_x": scale_x, "scale_y": scale_y},
            "E": E,
            "camera_pose": {
                "C_world": [float(C_world[0]), float(C_world[1]), float(C_world[2])],
                "yaw_deg": yaw,
                "pitch_deg": pitch,
                "roll_deg": roll,
                "camera_height_m": cam_height,
            },
            "axes_world": {
                "right": [float(x) for x in (R_wc @ np.array([1.0, 0.0, 0.0]))],
                "up": [float(x) for x in (R_wc @ np.array([0.0, 1.0, 0.0]))],
                "forward": [float(x) for x in (R_wc @ np.array([0.0, 0.0, 1.0]))],
            },
            "caminfo": {
                "path": str(caminfo_path),
                "type": caminfo_type,
                "modelInfo": caminfo.get("modelInfo"),
                "projectionMatrix": P_caminfo.tolist() if P_caminfo is not None else None,
                "diff_to_expected": caminfo_diff,
                "diff_to_expected_yflip": caminfo_diff_flip,
                "translation_ratio": t_ratio,
            },
            "ray_check": {
                "bottom_center": ray,
                "bottom_center_y_flip": ray_flip,
            },
        }

    snapshot = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "inputs": {
            "pipeline_path": str(pipeline_path),
            "cameras_path": str(cameras_path),
            "calibration_path": str(calibration_path),
            "alignment_path": str(alignment_path),
            "caminfo_dir": str(caminfo_dir),
            "tracker_config_path": str(tracker_cfg_path) if tracker_cfg_path else "",
            "env": _collect_env_snapshot(),
            "raw": {
                "pipeline": public_pipeline_config(pipeline_cfg),
                "cameras": cameras_cfg,
                "calibration": calibration_cfg,
                "alignment": align_cfg,
                "tracker": tracker_cfg_data,
            },
        },
        "pipeline": {
            "streammux": {"width": stream_w, "height": stream_h},
            "tracker": {"width": tracker_w, "height": tracker_h},
        },
        "units": {"floor_y": floor_y, "unit_scale": unit_scale},
        "cameras": per_camera,
        "issues": issues,
    }
    return snapshot


def render_snapshot_markdown(snapshot: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# V3DT Forensics Snapshot")
    lines.append("")
    lines.append(f"Generated: {snapshot.get('generated_at')}")
    lines.append("")
    pipeline = snapshot.get("pipeline", {})
    streammux = (pipeline.get("streammux") or {}) if isinstance(pipeline, Mapping) else {}
    tracker = (pipeline.get("tracker") or {}) if isinstance(pipeline, Mapping) else {}
    lines.append("## Pipeline")
    lines.append("")
    lines.append(f"- streammux: {streammux.get('width')}x{streammux.get('height')}")
    lines.append(f"- tracker: {tracker.get('width')}x{tracker.get('height')}")
    lines.append("")

    issues = snapshot.get("issues", []) or []
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            cam = issue.get("camera", "<unknown>")
            level = issue.get("level", "warn")
            code = issue.get("code", "")
            msg = issue.get("message", "")
            lines.append(f"- [{level}] {cam}: {code} - {msg}")
        lines.append("")

    lines.append("## Cameras")
    lines.append("")
    lines.append("| Camera | Model | Height(m) | C_world (m) | Yaw/Pitch/Roll | CamInfo | t_ratio | Ray t | Ray t (y-flip) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    cameras = snapshot.get("cameras", {}) or {}
    for name, entry in cameras.items():
        caminfo = entry.get("caminfo", {}) or {}
        pose = entry.get("camera_pose", {}) or {}
        ray = entry.get("ray_check", {}) or {}
        t_ratio = caminfo.get("translation_ratio")
        ratio_str = "n/a"
        if isinstance(t_ratio, list):
            ratio_str = ",".join(f"{r:.3f}" if isinstance(r, float) and math.isfinite(r) else "nan" for r in t_ratio)
        t_main = (ray.get("bottom_center") or {}).get("t")
        t_flip = (ray.get("bottom_center_y_flip") or {}).get("t")
        lines.append(
            "| {cam} | {model} | {height:.2f} | {c_world} | {ypr} | {ctype} | {ratio} | {t} | {tf} |".format(
                cam=name,
                model=entry.get("model"),
                height=_safe_float(entry.get("height_m_expected", 0.0)),
                c_world="[{:.2f},{:.2f},{:.2f}]".format(*pose.get("C_world", [0.0, 0.0, 0.0])),
                ypr="{:.1f}/{:.1f}/{:.1f}".format(
                    _safe_float(pose.get("yaw_deg", 0.0)),
                    _safe_float(pose.get("pitch_deg", 0.0)),
                    _safe_float(pose.get("roll_deg", 0.0)),
                ),
                ctype=caminfo.get("type"),
                ratio=ratio_str,
                t="{:.2f}".format(_safe_float(t_main, float("nan"))) if t_main is not None else "n/a",
                tf="{:.2f}".format(_safe_float(t_flip, float("nan"))) if t_flip is not None else "n/a",
            )
        )

    lines.append("")
    return "\n".join(lines)


def _iter_log_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                yield payload


@dataclass
class _CameraStats:
    frames: int = 0
    tracks: int = 0
    people_tracks: int = 0
    bbox3d_tracks: int = 0
    world_tracks: int = 0
    visibility: List[float] = None
    heights: List[float] = None
    foot_z: List[float] = None
    reproj_err: List[float] = None
    track_lengths: List[int] = None
    proj_bbox_heights: List[float] = None
    proj_bbox_widths: List[float] = None
    proj_height_ratios: List[float] = None
    proj_width_ratios: List[float] = None
    proj_bottom_offsets: List[float] = None
    proj_top_offsets: List[float] = None
    proj_center_offsets: List[float] = None
    rotation_samples: int = 0
    rotation_nonzero: int = 0
    depth_implied: List[float] = None
    depth_ratio: List[float] = None
    implied_fy: List[float] = None

    def __post_init__(self) -> None:
        self.visibility = []
        self.heights = []
        self.foot_z = []
        self.reproj_err = []
        self.track_lengths = []
        self.proj_bbox_heights = []
        self.proj_bbox_widths = []
        self.proj_height_ratios = []
        self.proj_width_ratios = []
        self.proj_bottom_offsets = []
        self.proj_top_offsets = []
        self.proj_center_offsets = []
        self.depth_implied = []
        self.depth_ratio = []
        self.implied_fy = []


@dataclass
class _ScaleSweepStats:
    reproj_err: List[float] = None
    heights: List[float] = None

    def __post_init__(self) -> None:
        self.reproj_err = []
        self.heights = []


def _normalize_scale_sweep(scales: Optional[Sequence[float]]) -> List[float]:
    if not scales:
        return []
    normalized: List[float] = []
    seen: set[float] = set()
    for value in scales:
        try:
            scale = float(value)
        except Exception:
            continue
        if not math.isfinite(scale) or scale <= 0.0:
            continue
        if scale in seen:
            continue
        seen.add(scale)
        normalized.append(scale)
    return normalized


def analyze_tracking_log(
    log_path: Path,
    *,
    snapshot: Optional[Mapping[str, Any]] = None,
    scale_sweep: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    stats: Dict[str, _CameraStats] = {}
    last_seen: Dict[Tuple[str, int], Tuple[int, int]] = {}
    warnings: List[Dict[str, Any]] = []

    snapshot_cams = (snapshot or {}).get("cameras", {}) if isinstance(snapshot, Mapping) else {}
    proj_by_camera: Dict[str, Optional[np.ndarray]] = {}
    fy_by_camera: Dict[str, float] = {}
    if snapshot_cams:
        for cam_id, cam_snap in snapshot_cams.items():
            caminfo = cam_snap.get("caminfo", {}) if isinstance(cam_snap, Mapping) else {}
            P_list = caminfo.get("projectionMatrix")
            P = None
            if isinstance(P_list, list):
                if len(P_list) == 12:
                    P = np.array(P_list, dtype=np.float64).reshape((3, 4))
                elif len(P_list) == 3 and all(isinstance(row, list) and len(row) == 4 for row in P_list):
                    P = np.array(P_list, dtype=np.float64).reshape((3, 4))
            proj_by_camera[str(cam_id)] = P
            intrinsics = cam_snap.get("intrinsics_scaled") or {}
            fy = _safe_float(intrinsics.get("fy"), float("nan"))
            if math.isfinite(fy) and fy > 0.0:
                fy_by_camera[str(cam_id)] = fy

    scale_sweep_values = _normalize_scale_sweep(scale_sweep)
    scale_stats: Dict[str, Dict[float, _ScaleSweepStats]] = {}

    for record in _iter_log_records(log_path):
        if record.get("type") not in ("v3dt_tracking_frame", "tracking"):
            continue
        camera_id = str(record.get("camera_id") or record.get("camera") or "")
        if not camera_id:
            continue
        cam_stats = stats.setdefault(camera_id, _CameraStats())
        cam_stats.frames += 1
        frame_id = _safe_int(record.get("frame_id", -1), -1)
        tracks = record.get("tracks") or []
        if not isinstance(tracks, list):
            continue
        cam_stats.tracks += len(tracks)
        for track in tracks:
            if not isinstance(track, dict):
                continue
            class_id = _safe_int(track.get("class_id", -1), -1)
            if class_id != 0:
                continue
            cam_stats.people_tracks += 1
            if "bbox3d" in track and isinstance(track.get("bbox3d"), dict):
                cam_stats.bbox3d_tracks += 1
                bbox3d = track.get("bbox3d") or {}
                x_c = _safe_float(bbox3d.get("xCentre"), float("nan"))
                y_c = _safe_float(bbox3d.get("yCentre"), float("nan"))
                z_c = _safe_float(bbox3d.get("zCentre"), float("nan"))
                x_len = _safe_float(bbox3d.get("xLen"), float("nan"))
                y_len = _safe_float(bbox3d.get("yLen"), float("nan"))
                z_len = _safe_float(bbox3d.get("zLen"), float("nan"))
                x_rot = _safe_float(bbox3d.get("xRot"), float("nan"))
                y_rot = _safe_float(bbox3d.get("yRot"), float("nan"))
                z_rot = _safe_float(bbox3d.get("zRot"), float("nan"))
                if math.isfinite(x_rot) or math.isfinite(y_rot) or math.isfinite(z_rot):
                    cam_stats.rotation_samples += 1
                    if any(math.isfinite(r) and abs(r) > 1e-6 for r in (x_rot, y_rot, z_rot)):
                        cam_stats.rotation_nonzero += 1
                if math.isfinite(z_len):
                    cam_stats.heights.append(z_len)
                if math.isfinite(z_c) and math.isfinite(z_len):
                    cam_stats.foot_z.append(z_c - 0.5 * z_len)

                bbox = track.get("bbox") or []
                bbox_valid = isinstance(bbox, list) and len(bbox) >= 4
                u_bbox = _safe_float(bbox[0]) + _safe_float(bbox[2]) / 2.0 if bbox_valid else float("nan")
                v_bbox = _safe_float(bbox[1]) + _safe_float(bbox[3]) if bbox_valid else float("nan")
                bbox_w = _safe_float(bbox[2]) if bbox_valid else float("nan")
                bbox_h = _safe_float(bbox[3]) if bbox_valid else float("nan")
                bbox_top = _safe_float(bbox[1]) if bbox_valid else float("nan")
                bbox_bottom = _safe_float(bbox[1]) + _safe_float(bbox[3]) if bbox_valid else float("nan")

                P = proj_by_camera.get(camera_id) if proj_by_camera else None
                if P is not None and bbox_valid and math.isfinite(x_c) and math.isfinite(y_c) and math.isfinite(z_c):
                    z_fp = z_c - 0.5 * z_len if math.isfinite(z_len) else z_c
                    uvd = _project_point(P, [x_c, y_c, z_fp])
                    if uvd:
                        u_proj, v_proj = uvd
                        cam_stats.reproj_err.append(float(math.hypot(u_proj - u_bbox, v_proj - v_bbox)))

                if math.isfinite(z_len) and z_len > 0.0 and math.isfinite(bbox_h) and bbox_h > 0.0:
                    fy = fy_by_camera.get(camera_id)
                    if fy is not None:
                        depth_implied = (fy * z_len) / bbox_h
                        cam_stats.depth_implied.append(depth_implied)
                        if math.isfinite(y_c):
                            cam_stats.depth_ratio.append(y_c / depth_implied)
                            cam_stats.implied_fy.append((bbox_h * y_c) / z_len)

                if (
                    P is not None
                    and bbox_valid
                    and math.isfinite(x_c)
                    and math.isfinite(y_c)
                    and math.isfinite(z_c)
                    and math.isfinite(x_len)
                    and math.isfinite(y_len)
                    and math.isfinite(z_len)
                    and x_len > 0.0
                    and y_len > 0.0
                    and z_len > 0.0
                ):
                    hx = 0.5 * x_len
                    hy = 0.5 * y_len
                    hz = 0.5 * z_len
                    corners = []
                    for dx in (-hx, hx):
                        for dy in (-hy, hy):
                            for dz in (-hz, hz):
                                uv = _project_point(P, [x_c + dx, y_c + dy, z_c + dz])
                                if uv and all(math.isfinite(val) for val in uv):
                                    corners.append(uv)
                    if len(corners) >= 4:
                        u_vals = [pt[0] for pt in corners]
                        v_vals = [pt[1] for pt in corners]
                        u_min = min(u_vals)
                        u_max = max(u_vals)
                        v_min = min(v_vals)
                        v_max = max(v_vals)
                        proj_w = float(u_max - u_min)
                        proj_h = float(v_max - v_min)
                        if proj_w > 0.0:
                            cam_stats.proj_bbox_widths.append(proj_w)
                            if math.isfinite(bbox_w) and bbox_w > 0.0:
                                cam_stats.proj_width_ratios.append(proj_w / bbox_w)
                        if proj_h > 0.0:
                            cam_stats.proj_bbox_heights.append(proj_h)
                            if math.isfinite(bbox_h) and bbox_h > 0.0:
                                cam_stats.proj_height_ratios.append(proj_h / bbox_h)
                        if math.isfinite(bbox_bottom):
                            cam_stats.proj_bottom_offsets.append(v_max - bbox_bottom)
                        if math.isfinite(bbox_top):
                            cam_stats.proj_top_offsets.append(v_min - bbox_top)
                        if math.isfinite(u_bbox):
                            cam_stats.proj_center_offsets.append((u_min + u_max) / 2.0 - u_bbox)

                if scale_sweep_values:
                    cam_scale_stats = scale_stats.setdefault(camera_id, {})
                    for scale in scale_sweep_values:
                        sweep_stats = cam_scale_stats.setdefault(scale, _ScaleSweepStats())
                        if math.isfinite(z_len):
                            sweep_stats.heights.append(z_len * scale)
                        if P is None or not bbox_valid:
                            continue
                        if not (math.isfinite(x_c) and math.isfinite(y_c) and math.isfinite(z_c)):
                            continue
                        x_s = x_c * scale
                        y_s = y_c * scale
                        z_s = z_c * scale
                        z_fp = z_s
                        if math.isfinite(z_len):
                            z_fp = z_s - 0.5 * (z_len * scale)
                        uvd = _project_point(P, [x_s, y_s, z_fp])
                        if uvd:
                            u_proj, v_proj = uvd
                            sweep_stats.reproj_err.append(float(math.hypot(u_proj - u_bbox, v_proj - v_bbox)))

            if "world" in track and isinstance(track.get("world"), list):
                cam_stats.world_tracks += 1

            vis = track.get("visibility")
            if vis is not None:
                v = _safe_float(vis, float("nan"))
                if math.isfinite(v):
                    cam_stats.visibility.append(v)

            track_id = track.get("track_id")
            if track_id is None:
                track_id = track.get("stable_id")
            if track_id is None:
                continue
            key = (camera_id, int(track_id))
            prev = last_seen.get(key)
            if prev is None:
                last_seen[key] = (frame_id, frame_id)
            else:
                start, last = prev
                last_seen[key] = (start, frame_id)

    for key, (start, end) in last_seen.items():
        camera_id, _ = key
        if camera_id in stats and start >= 0 and end >= start:
            stats[camera_id].track_lengths.append(int(end - start + 1))

    report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "log_path": str(log_path),
        "cameras": {},
        "warnings": warnings,
    }
    if scale_sweep_values:
        report["scale_sweep"] = {"scales": scale_sweep_values}

    for camera_id, cam_stats in stats.items():
        coverage = cam_stats.bbox3d_tracks / cam_stats.people_tracks if cam_stats.people_tracks else 0.0
        height_med = statistics.median(cam_stats.heights) if cam_stats.heights else None
        height_mean = statistics.mean(cam_stats.heights) if cam_stats.heights else None
        foot_z_mean = statistics.mean(cam_stats.foot_z) if cam_stats.foot_z else None
        reproj_med = statistics.median(cam_stats.reproj_err) if cam_stats.reproj_err else None
        track_len_med = statistics.median(cam_stats.track_lengths) if cam_stats.track_lengths else None
        proj_height_med = statistics.median(cam_stats.proj_bbox_heights) if cam_stats.proj_bbox_heights else None
        proj_width_med = statistics.median(cam_stats.proj_bbox_widths) if cam_stats.proj_bbox_widths else None
        proj_height_ratio_med = (
            statistics.median(cam_stats.proj_height_ratios) if cam_stats.proj_height_ratios else None
        )
        proj_width_ratio_med = (
            statistics.median(cam_stats.proj_width_ratios) if cam_stats.proj_width_ratios else None
        )
        proj_bottom_offset_med = (
            statistics.median(cam_stats.proj_bottom_offsets) if cam_stats.proj_bottom_offsets else None
        )
        proj_top_offset_med = statistics.median(cam_stats.proj_top_offsets) if cam_stats.proj_top_offsets else None
        proj_center_offset_med = (
            statistics.median(cam_stats.proj_center_offsets) if cam_stats.proj_center_offsets else None
        )
        depth_implied_med = statistics.median(cam_stats.depth_implied) if cam_stats.depth_implied else None
        depth_ratio_med = statistics.median(cam_stats.depth_ratio) if cam_stats.depth_ratio else None
        implied_fy_med = statistics.median(cam_stats.implied_fy) if cam_stats.implied_fy else None
        if cam_stats.people_tracks and coverage < 0.95:
            warnings.append({
                "camera": camera_id,
                "level": "warn",
                "code": "bbox3d_coverage_low",
                "message": "bbox3d present on <95% of person tracks",
                "data": {"coverage": coverage},
            })
        if height_med is not None:
            if not (1.2 <= height_med <= 2.4):
                warnings.append({
                    "camera": camera_id,
                    "level": "warn",
                    "code": "bbox3d_height_out_of_range",
                    "message": "median bbox3d height out of expected adult range",
                    "data": {"median_height": height_med},
                })
        if reproj_med is not None and reproj_med > 30.0:
            warnings.append({
                "camera": camera_id,
                "level": "warn",
                "code": "reprojection_error_high",
                "message": "median reprojection error exceeds 30px",
                "data": {"median_px": reproj_med},
            })
        if proj_height_ratio_med is not None and (proj_height_ratio_med < 0.6 or proj_height_ratio_med > 1.6):
            warnings.append({
                "camera": camera_id,
                "level": "warn",
                "code": "bbox3d_projection_height_ratio_off",
                "message": "projected 3D box height diverges from 2D bbox height",
                "data": {"median_ratio": proj_height_ratio_med},
            })
        if depth_ratio_med is not None and (depth_ratio_med < 0.6 or depth_ratio_med > 1.6):
            warnings.append({
                "camera": camera_id,
                "level": "warn",
                "code": "depth_ratio_off",
                "message": "bbox3d depth diverges from depth implied by 2D bbox height",
                "data": {"median_ratio": depth_ratio_med},
            })

        report["cameras"][camera_id] = {
            "frames": cam_stats.frames,
            "tracks": cam_stats.tracks,
            "people_tracks": cam_stats.people_tracks,
            "bbox3d_tracks": cam_stats.bbox3d_tracks,
            "bbox3d_coverage": coverage,
            "world_tracks": cam_stats.world_tracks,
            "visibility": {
                "mean": statistics.mean(cam_stats.visibility) if cam_stats.visibility else None,
                "median": statistics.median(cam_stats.visibility) if cam_stats.visibility else None,
            },
            "bbox3d_height": {
                "min": min(cam_stats.heights) if cam_stats.heights else None,
                "max": max(cam_stats.heights) if cam_stats.heights else None,
                "mean": height_mean,
                "median": height_med,
            },
            "foot_z": {
                "mean": foot_z_mean,
                "median": statistics.median(cam_stats.foot_z) if cam_stats.foot_z else None,
            },
            "reprojection_error_px": {
                "mean": statistics.mean(cam_stats.reproj_err) if cam_stats.reproj_err else None,
                "median": reproj_med,
            },
            "track_length_frames": {
                "mean": statistics.mean(cam_stats.track_lengths) if cam_stats.track_lengths else None,
                "median": track_len_med,
                "min": min(cam_stats.track_lengths) if cam_stats.track_lengths else None,
                "max": max(cam_stats.track_lengths) if cam_stats.track_lengths else None,
            },
            "projection_bbox": {
                "count": len(cam_stats.proj_bbox_heights),
                "height_px": {
                    "mean": statistics.mean(cam_stats.proj_bbox_heights) if cam_stats.proj_bbox_heights else None,
                    "median": proj_height_med,
                },
                "width_px": {
                    "mean": statistics.mean(cam_stats.proj_bbox_widths) if cam_stats.proj_bbox_widths else None,
                    "median": proj_width_med,
                },
                "height_ratio_to_2d": {
                    "mean": statistics.mean(cam_stats.proj_height_ratios) if cam_stats.proj_height_ratios else None,
                    "median": proj_height_ratio_med,
                },
                "width_ratio_to_2d": {
                    "mean": statistics.mean(cam_stats.proj_width_ratios) if cam_stats.proj_width_ratios else None,
                    "median": proj_width_ratio_med,
                },
                "bottom_offset_px": {
                    "mean": statistics.mean(cam_stats.proj_bottom_offsets)
                    if cam_stats.proj_bottom_offsets
                    else None,
                    "median": proj_bottom_offset_med,
                },
                "top_offset_px": {
                    "mean": statistics.mean(cam_stats.proj_top_offsets) if cam_stats.proj_top_offsets else None,
                    "median": proj_top_offset_med,
                },
                "center_x_offset_px": {
                    "mean": statistics.mean(cam_stats.proj_center_offsets)
                    if cam_stats.proj_center_offsets
                    else None,
                    "median": proj_center_offset_med,
                },
                "rotation_nonzero_ratio": (
                    cam_stats.rotation_nonzero / cam_stats.rotation_samples
                    if cam_stats.rotation_samples
                    else None
                ),
            },
            "depth_consistency": {
                "depth_implied_m": {
                    "mean": statistics.mean(cam_stats.depth_implied) if cam_stats.depth_implied else None,
                    "median": depth_implied_med,
                },
                "depth_ratio": {
                    "mean": statistics.mean(cam_stats.depth_ratio) if cam_stats.depth_ratio else None,
                    "median": depth_ratio_med,
                },
                "implied_fy": {
                    "mean": statistics.mean(cam_stats.implied_fy) if cam_stats.implied_fy else None,
                    "median": implied_fy_med,
                },
            },
        }
        if scale_sweep_values:
            sweep_entries: List[Dict[str, Any]] = []
            cam_scale_stats = scale_stats.get(camera_id, {})
            for scale in scale_sweep_values:
                sweep_stats = cam_scale_stats.get(scale)
                if not sweep_stats:
                    continue
                sweep_height_med = statistics.median(sweep_stats.heights) if sweep_stats.heights else None
                sweep_height_mean = statistics.mean(sweep_stats.heights) if sweep_stats.heights else None
                sweep_reproj_med = statistics.median(sweep_stats.reproj_err) if sweep_stats.reproj_err else None
                sweep_reproj_mean = statistics.mean(sweep_stats.reproj_err) if sweep_stats.reproj_err else None
                sweep_entries.append({
                    "scale": scale,
                    "bbox3d_height": {
                        "mean": sweep_height_mean,
                        "median": sweep_height_med,
                    },
                    "reprojection_error_px": {
                        "mean": sweep_reproj_mean,
                        "median": sweep_reproj_med,
                    },
                })
            if sweep_entries:
                report["cameras"][camera_id]["scale_sweep"] = sweep_entries
                best_entry = None
                best_med = None
                for entry in sweep_entries:
                    med = (entry.get("reprojection_error_px") or {}).get("median")
                    if med is None:
                        continue
                    if best_med is None or med < best_med:
                        best_med = med
                        best_entry = entry
                if best_entry and best_med is not None:
                    report["cameras"][camera_id]["scale_sweep_best_by_reproj"] = {
                        "scale": best_entry.get("scale"),
                        "median_px": best_med,
                    }

    return report


def render_report_markdown(report: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# V3DT Forensics Report")
    lines.append("")
    lines.append(f"Generated: {report.get('generated_at')}")
    lines.append("")

    warnings = report.get("warnings") or []
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            cam = warning.get("camera", "<unknown>")
            level = warning.get("level", "warn")
            code = warning.get("code", "")
            msg = warning.get("message", "")
            lines.append(f"- [{level}] {cam}: {code} - {msg}")
        lines.append("")

    lines.append("## Cameras")
    lines.append("")
    lines.append("| Camera | Frames | People Tracks | bbox3d% | Height med | Reproj med (px) | Track len med |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    cameras = report.get("cameras", {}) or {}
    for cam, data in cameras.items():
        lines.append(
            "| {cam} | {frames} | {people} | {cov:.1f}% | {hmed} | {rmed} | {tmed} |".format(
                cam=cam,
                frames=_safe_int(data.get("frames", 0)),
                people=_safe_int(data.get("people_tracks", 0)),
                cov=float(data.get("bbox3d_coverage", 0.0)) * 100.0,
                hmed="{:.2f}".format(_safe_float((data.get("bbox3d_height") or {}).get("median"), float("nan")))
                if (data.get("bbox3d_height") or {}).get("median") is not None
                else "n/a",
                rmed="{:.1f}".format(_safe_float((data.get("reprojection_error_px") or {}).get("median"), float("nan")))
                if (data.get("reprojection_error_px") or {}).get("median") is not None
                else "n/a",
                tmed="{:.1f}".format(_safe_float((data.get("track_length_frames") or {}).get("median"), float("nan")))
                if (data.get("track_length_frames") or {}).get("median") is not None
                else "n/a",
            )
        )

    lines.append("")

    if cameras:
        lines.append("## Projection Diagnostics")
        lines.append("")
        lines.append(
            "| Camera | Proj H med | H ratio med | W ratio med | Bottom off med | Top off med | Center off med | Rot nonzero |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for cam, data in cameras.items():
            proj = data.get("projection_bbox") or {}
            hmed = (proj.get("height_px") or {}).get("median")
            hratio = (proj.get("height_ratio_to_2d") or {}).get("median")
            wratio = (proj.get("width_ratio_to_2d") or {}).get("median")
            bmed = (proj.get("bottom_offset_px") or {}).get("median")
            tmed = (proj.get("top_offset_px") or {}).get("median")
            cmed = (proj.get("center_x_offset_px") or {}).get("median")
            rnon = proj.get("rotation_nonzero_ratio")
            lines.append(
                "| {cam} | {hmed} | {hr} | {wr} | {b} | {t} | {c} | {r} |".format(
                    cam=cam,
                    hmed="{:.1f}px".format(_safe_float(hmed, float("nan"))) if hmed is not None else "n/a",
                    hr="{:.2f}".format(_safe_float(hratio, float("nan"))) if hratio is not None else "n/a",
                    wr="{:.2f}".format(_safe_float(wratio, float("nan"))) if wratio is not None else "n/a",
                    b="{:.1f}px".format(_safe_float(bmed, float("nan"))) if bmed is not None else "n/a",
                    t="{:.1f}px".format(_safe_float(tmed, float("nan"))) if tmed is not None else "n/a",
                    c="{:.1f}px".format(_safe_float(cmed, float("nan"))) if cmed is not None else "n/a",
                    r="{:.1f}%".format(_safe_float(rnon, float("nan")) * 100.0)
                    if rnon is not None
                    else "n/a",
                )
            )
        lines.append("")

        lines.append("## Depth Consistency (Y-forward)")
        lines.append("")
        lines.append("| Camera | Depth implied med (m) | Depth ratio med | Implied fy med |")
        lines.append("| --- | --- | --- | --- |")
        for cam, data in cameras.items():
            depth = data.get("depth_consistency") or {}
            dmed = (depth.get("depth_implied_m") or {}).get("median")
            rmed = (depth.get("depth_ratio") or {}).get("median")
            fmed = (depth.get("implied_fy") or {}).get("median")
            lines.append(
                "| {cam} | {d} | {r} | {f} |".format(
                    cam=cam,
                    d="{:.2f}".format(_safe_float(dmed, float("nan"))) if dmed is not None else "n/a",
                    r="{:.2f}".format(_safe_float(rmed, float("nan"))) if rmed is not None else "n/a",
                    f="{:.1f}".format(_safe_float(fmed, float("nan"))) if fmed is not None else "n/a",
                )
            )
        lines.append("")

    has_scale_sweep = any((data.get("scale_sweep") or []) for data in cameras.values())
    if has_scale_sweep:
        lines.append("## Scale Sweep")
        lines.append("")
        for cam, data in cameras.items():
            sweep = data.get("scale_sweep") or []
            if not sweep:
                continue
            best = data.get("scale_sweep_best_by_reproj") or {}
            if best:
                best_scale = _safe_float(best.get("scale"), float("nan"))
                best_med = _safe_float(best.get("median_px"), float("nan"))
                lines.append(
                    "### {cam} (best reproj: scale {scale} @ {px:.1f}px)".format(
                        cam=cam,
                        scale="{:.4g}".format(best_scale) if math.isfinite(best_scale) else "n/a",
                        px=best_med,
                    )
                )
            else:
                lines.append(f"### {cam}")
            lines.append("| Scale | Height med | Reproj med (px) |")
            lines.append("| --- | --- | --- |")
            for entry in sweep:
                scale = _safe_float(entry.get("scale"), float("nan"))
                hmed = (entry.get("bbox3d_height") or {}).get("median")
                rmed = (entry.get("reprojection_error_px") or {}).get("median")
                lines.append(
                    "| {scale} | {hmed} | {rmed} |".format(
                        scale="{:.4g}".format(scale) if math.isfinite(scale) else "n/a",
                        hmed="{:.2f}".format(_safe_float(hmed, float("nan"))) if hmed is not None else "n/a",
                        rmed="{:.1f}".format(_safe_float(rmed, float("nan"))) if rmed is not None else "n/a",
                    )
                )
            lines.append("")

    return "\n".join(lines)


def render_panel_html(snapshot: Mapping[str, Any], report: Mapping[str, Any]) -> str:
    snapshot_json = json.dumps(snapshot, ensure_ascii=True)
    report_json = json.dumps(report, ensure_ascii=True)
    template = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>V3DT Forensics Panel</title>
  <style>
    :root {{
      --bg: #f6f1ea;
      --bg-alt: #fbf8f3;
      --ink: #1b1b1b;
      --muted: #6b5f55;
      --accent: #d2633c;
      --accent-2: #2d6b7d;
      --warn: #b0322a;
      --ok: #1f6f3d;
      --card: #ffffff;
      --grid: rgba(0,0,0,0.08);
      --radius: 14px;
      --shadow: 0 18px 40px rgba(0,0,0,0.12);
      --mono: "IBM Plex Mono", "Fira Mono", "Courier New", monospace;
      --body: "Space Grotesk", "Montserrat", "Segoe UI", sans-serif;
      --title: "DM Serif Display", "Georgia", serif;
    }}
    body {{
      margin: 0;
      font-family: var(--body);
      color: var(--ink);
      background: radial-gradient(circle at top left, #fef2e7, var(--bg));
    }}
    header {{
      padding: 32px 6vw 16px;
    }}
    h1 {{
      font-family: var(--title);
      font-size: clamp(28px, 4vw, 48px);
      margin: 0 0 8px;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 16px;
      max-width: 720px;
    }}
    main {{
      padding: 0 6vw 48px;
      display: grid;
      gap: 24px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--card);
      border-radius: var(--radius);
      padding: 18px 20px;
      box-shadow: var(--shadow);
    }}
    .card h2 {{
      font-size: 18px;
      margin: 0 0 12px;
      font-weight: 600;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      background: rgba(210, 99, 60, 0.12);
      color: var(--accent);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 6px;
      border-bottom: 1px solid var(--grid);
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    .warn {{ color: var(--warn); font-weight: 600; }}
    .ok {{ color: var(--ok); font-weight: 600; }}
    details {{
      background: var(--bg-alt);
      border-radius: 12px;
      padding: 12px 14px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    pre {{
      white-space: pre-wrap;
      font-family: var(--mono);
      font-size: 12px;
      margin: 8px 0 0;
    }}
    .tag {{
      font-family: var(--mono);
      font-size: 11px;
      background: rgba(45,107,125,0.12);
      color: var(--accent-2);
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>V3DT Forensics Panel</h1>
    <div class=\"subtitle\">Snapshot + telemetry report with explicit calibration math, projection checks, and track stability metrics.</div>
    <div class=\"pill\" id=\"generated-pill\"></div>
  </header>
  <main>
    <section class=\"grid\">
      <div class=\"card\">
        <h2>Pipeline</h2>
        <div id=\"pipeline-info\"></div>
      </div>
      <div class=\"card\">
        <h2>Warnings</h2>
        <div id=\"warning-list\"></div>
      </div>
      <div class=\"card\">
        <h2>Telemetry Summary</h2>
        <div id=\"telemetry-summary\"></div>
      </div>
    </section>
    <section class=\"card\">
      <h2>Per-Camera Metrics</h2>
      <div id=\"camera-table\"></div>
    </section>
    <section class=\"card\">
      <h2>Projection Diagnostics</h2>
      <div id=\"projection-diagnostics\"></div>
    </section>
    <section class=\"card\">
      <h2>Depth Consistency (Y-forward)</h2>
      <div id=\"depth-consistency\"></div>
    </section>
    <section class=\"card\">
      <h2>Scale Sweep</h2>
      <div id=\"scale-sweep\"></div>
    </section>
    <section class=\"grid\">
      <details class=\"card\">
        <summary>Raw Snapshot JSON</summary>
        <pre id=\"snapshot-json\"></pre>
      </details>
      <details class=\"card\">
        <summary>Raw Report JSON</summary>
        <pre id=\"report-json\"></pre>
      </details>
    </section>
  </main>
  <script>
    const SNAPSHOT = __SNAPSHOT_JSON__;
    const REPORT = __REPORT_JSON__;

    const setText = (id, text) => {{
      const el = document.getElementById(id);
      if (el) el.textContent = text;
    }};

    setText('generated-pill', `Snapshot ${SNAPSHOT.generated_at || ''} · Report ${REPORT.generated_at || ''}`);

    const pipeline = SNAPSHOT.pipeline || {{}};
    const streammux = (pipeline.streammux || {{}});
    const tracker = (pipeline.tracker || {{}});
    setText('pipeline-info', `streammux ${streammux.width || '?'}x${streammux.height || '?'} · tracker ${tracker.width || '?'}x${tracker.height || '?'}`);

    const warnings = (SNAPSHOT.issues || []).concat(REPORT.warnings || []);
    const warnEl = document.getElementById('warning-list');
    if (warnings.length === 0) {{
      warnEl.textContent = 'No warnings flagged.';
    }} else {{
      warnEl.innerHTML = warnings.map(w => `\n<div class=\"warn\">${{w.camera || 'global'}} · ${{w.code || 'warning'}} — ${{w.message || ''}}</div>`).join('');
    }}

    const telemetrySummary = document.getElementById('telemetry-summary');
    const camCount = Object.keys(REPORT.cameras || {{}}).length;
    telemetrySummary.textContent = `${{camCount}} cameras analyzed, log: ${REPORT.log_path || 'n/a'}`;

    const camTable = document.getElementById('camera-table');
    const rows = Object.entries(REPORT.cameras || {{}}).map(([name, data]) => {{
      const coverage = (data.bbox3d_coverage || 0) * 100;
      const heightMed = data.bbox3d_height?.median;
      const reprojMed = data.reprojection_error_px?.median;
      const tlenMed = data.track_length_frames?.median;
      return `<tr>
        <td><span class=\"tag\">${name}</span></td>
        <td>${data.frames || 0}</td>
        <td>${data.people_tracks || 0}</td>
        <td>${coverage.toFixed(1)}%</td>
        <td>${heightMed ? heightMed.toFixed(2) : 'n/a'}</td>
        <td>${reprojMed ? reprojMed.toFixed(1) : 'n/a'}</td>
        <td>${tlenMed ? tlenMed.toFixed(1) : 'n/a'}</td>
      </tr>`;
    }}).join('');

    camTable.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>Camera</th>
            <th>Frames</th>
            <th>People Tracks</th>
            <th>bbox3d%</th>
            <th>Height med</th>
            <th>Reproj med</th>
            <th>Track len med</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;

    const projDiagEl = document.getElementById('projection-diagnostics');
    const projRows = Object.entries(REPORT.cameras || {{}}).map(([name, data]) => {{
      const proj = data.projection_bbox || {{}};
      const hmed = Number.isFinite(proj.height_px?.median) ? proj.height_px.median.toFixed(1) : 'n/a';
      const hratio = Number.isFinite(proj.height_ratio_to_2d?.median) ? proj.height_ratio_to_2d.median.toFixed(2) : 'n/a';
      const wratio = Number.isFinite(proj.width_ratio_to_2d?.median) ? proj.width_ratio_to_2d.median.toFixed(2) : 'n/a';
      const bmed = Number.isFinite(proj.bottom_offset_px?.median) ? proj.bottom_offset_px.median.toFixed(1) : 'n/a';
      const tmed = Number.isFinite(proj.top_offset_px?.median) ? proj.top_offset_px.median.toFixed(1) : 'n/a';
      const cmed = Number.isFinite(proj.center_x_offset_px?.median) ? proj.center_x_offset_px.median.toFixed(1) : 'n/a';
      const rnon = Number.isFinite(proj.rotation_nonzero_ratio) ? (proj.rotation_nonzero_ratio * 100).toFixed(1) + '%' : 'n/a';
      return `<tr>
        <td><span class=\"tag\">${{name}}</span></td>
        <td>${{hmed}}</td>
        <td>${{hratio}}</td>
        <td>${{wratio}}</td>
        <td>${{bmed}}</td>
        <td>${{tmed}}</td>
        <td>${{cmed}}</td>
        <td>${{rnon}}</td>
      </tr>`;
    }}).join('');

    projDiagEl.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>Camera</th>
            <th>Proj H med (px)</th>
            <th>H ratio med</th>
            <th>W ratio med</th>
            <th>Bottom off med (px)</th>
            <th>Top off med (px)</th>
            <th>Center X off med (px)</th>
            <th>Rot nonzero</th>
          </tr>
        </thead>
        <tbody>${{projRows}}</tbody>
      </table>
    `;

    const depthEl = document.getElementById('depth-consistency');
    const depthRows = Object.entries(REPORT.cameras || {{}}).map(([name, data]) => {{
      const depth = data.depth_consistency || {{}};
      const dmed = Number.isFinite(depth.depth_implied_m?.median) ? depth.depth_implied_m.median.toFixed(2) : 'n/a';
      const rmed = Number.isFinite(depth.depth_ratio?.median) ? depth.depth_ratio.median.toFixed(2) : 'n/a';
      const fmed = Number.isFinite(depth.implied_fy?.median) ? depth.implied_fy.median.toFixed(1) : 'n/a';
      return `<tr>
        <td><span class=\"tag\">${{name}}</span></td>
        <td>${{dmed}}</td>
        <td>${{rmed}}</td>
        <td>${{fmed}}</td>
      </tr>`;
    }}).join('');

    depthEl.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>Camera</th>
            <th>Depth implied med (m)</th>
            <th>Depth ratio med</th>
            <th>Implied fy med</th>
          </tr>
        </thead>
        <tbody>${{depthRows}}</tbody>
      </table>
    `;

    const scaleSweepEl = document.getElementById('scale-sweep');
    const sweepBlocks = Object.entries(REPORT.cameras || {{}}).map(([name, data]) => {{
      const sweep = data.scale_sweep || [];
      if (!sweep.length) return '';
      const best = data.scale_sweep_best_by_reproj || null;
      const bestScale = best && Number.isFinite(best.scale) ? best.scale.toPrecision(4) : 'n/a';
      const bestPx = best && Number.isFinite(best.median_px) ? best.median_px.toFixed(1) : 'n/a';
      const rows = sweep.map(entry => {{
        const scale = Number.isFinite(entry.scale) ? entry.scale.toPrecision(4) : 'n/a';
        const hmed = Number.isFinite(entry.bbox3d_height?.median) ? entry.bbox3d_height.median.toFixed(2) : 'n/a';
        const rmed = Number.isFinite(entry.reprojection_error_px?.median) ? entry.reprojection_error_px.median.toFixed(1) : 'n/a';
        return `<tr><td>${{scale}}</td><td>${{hmed}}</td><td>${{rmed}}</td></tr>`;
      }}).join('');
      return `
        <details>
          <summary><span class=\"tag\">${{name}}</span> · best reproj ${bestScale} @ ${bestPx}px</summary>
          <table>
            <thead>
              <tr><th>Scale</th><th>Height med</th><th>Reproj med (px)</th></tr>
            </thead>
            <tbody>${{rows}}</tbody>
          </table>
        </details>
      `;
    }}).filter(Boolean).join('');

    if (!sweepBlocks) {{
      scaleSweepEl.textContent = 'No scale sweep data in this report.';
    }} else {{
      scaleSweepEl.innerHTML = sweepBlocks;
    }}

    setText('snapshot-json', JSON.stringify(SNAPSHOT, null, 2));
    setText('report-json', JSON.stringify(REPORT, null, 2));
  </script>
</body>
</html>
"""
    return template.replace("__SNAPSHOT_JSON__", snapshot_json).replace("__REPORT_JSON__", report_json)
