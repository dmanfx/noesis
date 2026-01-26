#!/usr/bin/env python3
"""Calibrate camera intrinsics from ChArUco images or video frames.

This tool estimates intrinsics (fx, fy, cx, cy) and distortion coefficients
using OpenCV's ChArUco calibration. It can optionally update intrinsics.json
and/or config/cameras.yaml so V3DT camInfo generation uses the new intrinsics.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _as_paths(items: Sequence[str]) -> List[Path]:
    return [Path(item).expanduser().resolve() for item in items if item]


def _iter_images(image_paths: Sequence[Path]) -> Iterable[np.ndarray]:
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[warn] skipping unreadable image: {path}")
            continue
        yield img


def _iter_video(video_path: Path, frame_stride: int, max_frames: int) -> Iterable[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed_to_open_video:{video_path}")
    idx = 0
    used = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_stride == 0:
            yield frame
            used += 1
            if max_frames and used >= max_frames:
                break
        idx += 1
    cap.release()


def _build_detectors(
    dictionary: cv2.aruco_Dictionary,
    board: cv2.aruco_CharucoBoard,
) -> Tuple[Optional[object], Optional[object], cv2.aruco_DetectorParameters]:
    params = cv2.aruco.DetectorParameters()
    aruco_detector = None
    if hasattr(cv2.aruco, "ArucoDetector"):
        aruco_detector = cv2.aruco.ArucoDetector(dictionary, params)
    charuco_detector = None
    if hasattr(cv2.aruco, "CharucoDetector"):
        try:
            charuco_detector = cv2.aruco.CharucoDetector(board)
            if hasattr(charuco_detector, "setDetectorParameters"):
                charuco_detector.setDetectorParameters(params)
        except Exception:
            charuco_detector = None
    return aruco_detector, charuco_detector, params


def _detect_charuco(
    gray: np.ndarray,
    dictionary: cv2.aruco_Dictionary,
    board: cv2.aruco_CharucoBoard,
    aruco_detector: Optional[object],
    charuco_detector: Optional[object],
    params: cv2.aruco_DetectorParameters,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    if charuco_detector is not None:
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
        if charuco_ids is None or len(charuco_ids) == 0:
            return None, None, 0
        return charuco_corners, charuco_ids, int(len(charuco_ids))

    corners = ids = None
    if aruco_detector is not None:
        corners, ids, _ = aruco_detector.detectMarkers(gray)
    elif hasattr(cv2.aruco, "detectMarkers"):
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    if ids is None or len(ids) == 0:
        return None, None, 0
    if not hasattr(cv2.aruco, "interpolateCornersCharuco"):
        return None, None, 0
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if retval is None:
        return None, None, 0
    return charuco_corners, charuco_ids, int(retval)


def _compute_view_error(
    board: cv2.aruco_CharucoBoard,
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    ids = charuco_ids.flatten().astype(int)
    obj_points = board.getChessboardCorners()[ids]
    img_points = charuco_corners.reshape(-1, 2)
    proj, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(img_points - proj, axis=1)
    return float(np.mean(err)) if len(err) else float("nan")


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _update_intrinsics_json(
    json_path: Path,
    model_key: str,
    model_name: str,
    resolution: Tuple[int, int],
    K: np.ndarray,
    dist: np.ndarray,
    meta: Dict,
) -> None:
    data = _load_json(json_path)
    entry = dict(data.get(model_key) or {})
    entry["model"] = model_name or entry.get("model") or model_key
    entry["resolution"] = [int(resolution[0]), int(resolution[1])]
    entry["intrinsics"] = {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "distortion_coeffs": [float(x) for x in dist.flatten().tolist()],
        "K_matrix": [
            [float(K[0, 0]), 0.0, float(K[0, 2])],
            [0.0, float(K[1, 1]), float(K[1, 2])],
            [0.0, 0.0, 1.0],
        ],
    }
    entry["calibration"] = meta
    data[model_key] = entry
    _write_json(json_path, data)


def _update_cameras_yaml(
    yaml_path: Path,
    model_key: str,
    K: np.ndarray,
    dist: np.ndarray,
    meta: Dict,
) -> None:
    import yaml  # lazy import to keep startup light

    data = {}
    if yaml_path.exists():
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    models = data.setdefault("intrinsics_models", {})
    entry = dict(models.get(model_key) or {})
    entry["intrinsics"] = {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "k1": float(dist[0]) if dist.size > 0 else 0.0,
        "k2": float(dist[1]) if dist.size > 1 else 0.0,
        "k3": float(dist[4]) if dist.size > 4 else 0.0,
    }
    entry["calibration"] = meta
    models[model_key] = entry
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate intrinsics from ChArUco captures.")
    parser.add_argument("--image-dir", default="", help="Directory of calibration images")
    parser.add_argument("--images", nargs="*", default=[], help="Explicit image paths")
    parser.add_argument("--video", default="", help="Video path (optional)")
    parser.add_argument("--frame-stride", type=int, default=10, help="Use every Nth frame from video")
    parser.add_argument("--max-frames", type=int, default=300, help="Limit frames sampled from video")
    parser.add_argument("--squares-x", type=int, default=7, help="ChArUco squares in X (columns)")
    parser.add_argument("--squares-y", type=int, default=5, help="ChArUco squares in Y (rows)")
    parser.add_argument("--square-length-mm", type=float, required=True, help="Square size in mm")
    parser.add_argument("--marker-length-mm", type=float, required=True, help="Marker size in mm")
    parser.add_argument("--dictionary", default="DICT_4X4_50", help="Aruco dictionary name")
    parser.add_argument("--min-corners", type=int, default=12, help="Minimum ChArUco corners per frame")
    parser.add_argument("--min-frames", type=int, default=15, help="Minimum accepted frames")
    parser.add_argument("--output", default="", help="Write calibration JSON to this path")
    parser.add_argument("--update-intrinsics-json", default="", help="Update intrinsics.json at this path")
    parser.add_argument("--json-model-key", default="", help="Model key to update in intrinsics.json")
    parser.add_argument("--json-model-name", default="", help="Model name string for intrinsics.json")
    parser.add_argument("--update-cameras-yaml", default="", help="Update cameras.yaml at this path")
    parser.add_argument("--yaml-model-key", default="", help="Model key to update in cameras.yaml")
    args = parser.parse_args()

    if not hasattr(cv2, "aruco"):
        print("[fail] OpenCV ArUco module not available. Install opencv-contrib-python.")
        return 2

    image_paths: List[Path] = []
    if args.image_dir:
        base = Path(args.image_dir).expanduser().resolve()
        if base.exists():
            for ext in _IMAGE_EXTS:
                image_paths.extend(sorted(base.glob(f"*{ext}")))
    if args.images:
        image_paths.extend(_as_paths(args.images))
    video_path = Path(args.video).expanduser().resolve() if args.video else None

    if not image_paths and not video_path:
        print("[fail] Provide --image-dir, --images, or --video.")
        return 2

    dict_name = args.dictionary
    if not hasattr(cv2.aruco, dict_name):
        print(f"[fail] Unknown dictionary '{dict_name}'")
        return 2
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    board = cv2.aruco.CharucoBoard(
        (int(args.squares_x), int(args.squares_y)),
        float(args.square_length_mm) * 1e-3,
        float(args.marker_length_mm) * 1e-3,
        dictionary,
    )
    aruco_detector, charuco_detector, params = _build_detectors(dictionary, board)

    charuco_corners_list: List[np.ndarray] = []
    charuco_ids_list: List[np.ndarray] = []
    image_size: Optional[Tuple[int, int]] = None
    corner_counts: List[int] = []
    total_frames = 0

    def _handle_frame(frame: np.ndarray) -> None:
        nonlocal image_size, total_frames
        total_frames += 1
        if frame is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if image_size is None:
            image_size = (w, h)
        corners, ids, count = _detect_charuco(
            gray,
            dictionary,
            board,
            aruco_detector,
            charuco_detector,
            params,
        )
        if corners is None or ids is None or count < int(args.min_corners):
            return
        charuco_corners_list.append(corners)
        charuco_ids_list.append(ids)
        corner_counts.append(count)

    for img in _iter_images(image_paths):
        _handle_frame(img)
    if video_path:
        for frame in _iter_video(video_path, int(args.frame_stride), int(args.max_frames)):
            _handle_frame(frame)

    if not image_size:
        print("[fail] No valid images/frames found.")
        return 2
    if len(charuco_corners_list) < int(args.min_frames):
        print(f"[fail] Not enough valid frames ({len(charuco_corners_list)}/{args.min_frames}).")
        return 2

    if hasattr(cv2.aruco, "calibrateCameraCharuco"):
        ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charuco_corners_list,
            charuco_ids_list,
            board,
            image_size,
            None,
            None,
        )
    else:
        obj_points: List[np.ndarray] = []
        img_points: List[np.ndarray] = []
        for corners, ids in zip(charuco_corners_list, charuco_ids_list):
            obj, img = board.matchImagePoints(corners, ids)
            if obj is None or img is None:
                continue
            obj_points.append(obj)
            img_points.append(img)
        if len(obj_points) < int(args.min_frames):
            print(f"[fail] Not enough valid frames after matchImagePoints ({len(obj_points)}/{args.min_frames}).")
            return 2
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points,
            img_points,
            image_size,
            None,
            None,
        )

    view_errors: List[float] = []
    for corners, ids, rvec, tvec in zip(charuco_corners_list, charuco_ids_list, rvecs, tvecs):
        view_errors.append(_compute_view_error(board, corners, ids, K, dist, rvec, tvec))

    meta = {
        "method": "charuco",
        "generated_at": _now_stamp(),
        "dictionary": dict_name,
        "squares_x": int(args.squares_x),
        "squares_y": int(args.squares_y),
        "square_length_mm": float(args.square_length_mm),
        "marker_length_mm": float(args.marker_length_mm),
        "frames_total": int(total_frames),
        "frames_used": int(len(charuco_corners_list)),
        "avg_corners": float(statistics.mean(corner_counts)) if corner_counts else 0.0,
        "reprojection_error_px": float(ret),
        "reprojection_error_median_px": float(statistics.median(view_errors)) if view_errors else None,
    }

    output = {
        "resolution": [int(image_size[0]), int(image_size[1])],
        "intrinsics": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "distortion_coeffs": [float(x) for x in dist.flatten().tolist()],
            "K_matrix": [
                [float(K[0, 0]), 0.0, float(K[0, 2])],
                [0.0, float(K[1, 1]), float(K[1, 2])],
                [0.0, 0.0, 1.0],
            ],
        },
        "calibration": meta,
    }

    out_path = Path(args.output) if args.output else Path("diagnostics") / f"charuco_intrinsics_{_now_stamp()}.json"
    _write_json(out_path, output)
    print(f"Wrote calibration: {out_path}")
    print(f"fx={output['intrinsics']['fx']:.2f} fy={output['intrinsics']['fy']:.2f} "
          f"cx={output['intrinsics']['cx']:.2f} cy={output['intrinsics']['cy']:.2f} "
          f"reproj={meta['reprojection_error_px']:.3f}px")

    if args.update_intrinsics_json and args.json_model_key:
        json_path = Path(args.update_intrinsics_json).expanduser().resolve()
        _update_intrinsics_json(
            json_path,
            args.json_model_key,
            args.json_model_name,
            image_size,
            K,
            dist,
            meta,
        )
        print(f"Updated intrinsics.json: {json_path} (model={args.json_model_key})")

    if args.update_cameras_yaml and args.yaml_model_key:
        yaml_path = Path(args.update_cameras_yaml).expanduser().resolve()
        _update_cameras_yaml(yaml_path, args.yaml_model_key, K, dist, meta)
        print(f"Updated cameras.yaml: {yaml_path} (model={args.yaml_model_key})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
