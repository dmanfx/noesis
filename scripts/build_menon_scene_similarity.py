#!/usr/bin/env python3
"""Build or verify the backend-world to Menon-scene camera-anchor similarity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.calibration.scene_registration import (  # noqa: E402
    camera_anchor_state_sha256,
    solve_scene_similarity,
)


class SceneSimilarityBuildError(ValueError):
    """Raised when the camera-anchor fit cannot be admitted."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _camera_center_world_m(E_col_major: Sequence[Any]) -> list[float]:
    if not isinstance(E_col_major, Sequence) or isinstance(E_col_major, (str, bytes)) or len(E_col_major) != 16:
        raise SceneSimilarityBuildError("camera E must be a 16-value column-major matrix")
    matrix = np.asarray([float(value) for value in E_col_major], dtype=np.float64).reshape(
        (4, 4), order="F"
    )
    if not np.all(np.isfinite(matrix)):
        raise SceneSimilarityBuildError("camera E contains non-finite values")
    try:
        center = np.linalg.inv(matrix)[:3, 3]
    except np.linalg.LinAlgError as exc:
        raise SceneSimilarityBuildError("camera E is singular") from exc
    return [float(value) for value in center]


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _device_camera_positions(
    payload: Any,
    camera_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, list):
        raise SceneSimilarityBuildError("Menon virtual-device state must be a JSON array")
    wanted = {_slug(camera_id): str(camera_id) for camera_id in camera_ids}
    result: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, Mapping) or _slug(item.get("type")) != "camera":
            continue
        name_slug = _slug(item.get("name"))
        matches = [camera_id for key, camera_id in wanted.items() if key and key in name_slug]
        if len(matches) != 1:
            continue
        camera_id = matches[0]
        if camera_id in result:
            raise SceneSimilarityBuildError(f"multiple Menon camera anchors match {camera_id!r}")
        anchor_id = str(item.get("id") or "").strip()
        position = item.get("position")
        if not anchor_id or not isinstance(position, Mapping):
            raise SceneSimilarityBuildError(f"Menon camera anchor {camera_id!r} is incomplete")
        try:
            scene_position = [float(position[key]) for key in ("x", "y", "z")]
        except (KeyError, TypeError, ValueError) as exc:
            raise SceneSimilarityBuildError(
                f"Menon camera anchor {camera_id!r} has an invalid position"
            ) from exc
        if not all(math.isfinite(value) for value in scene_position):
            raise SceneSimilarityBuildError(
                f"Menon camera anchor {camera_id!r} has a non-finite position"
            )
        result[camera_id] = {
            "anchor_id": anchor_id,
            "scene_position": scene_position,
        }
    missing = sorted(set(camera_ids) - set(result))
    if missing:
        raise SceneSimilarityBuildError(f"Menon camera anchors are missing: {missing}")
    return result


def build_similarity(
    *,
    camera_calibration_path: Path,
    menon_devices_path: Path,
    anchor_residual_limit_m: float = 0.05,
) -> dict[str, Any]:
    limit_m = float(anchor_residual_limit_m)
    if not math.isfinite(limit_m) or limit_m <= 0.0 or limit_m > 0.25:
        raise SceneSimilarityBuildError("anchor residual limit must be within (0, 0.25] meters")
    calibration = _load_json(camera_calibration_path)
    cameras = calibration.get("cameras") if isinstance(calibration, Mapping) else None
    if not isinstance(cameras, Mapping) or len(cameras) < 3:
        raise SceneSimilarityBuildError("camera calibration must contain at least three cameras")
    camera_ids = sorted(str(camera_id) for camera_id in cameras)
    device_positions = _device_camera_positions(_load_json(menon_devices_path), camera_ids)

    correspondences: list[dict[str, Any]] = []
    for camera_id in camera_ids:
        camera = cameras.get(camera_id)
        if not isinstance(camera, Mapping):
            raise SceneSimilarityBuildError(f"camera calibration entry {camera_id!r} is invalid")
        anchor = device_positions[camera_id]
        correspondences.append(
            {
                "anchor_id": anchor["anchor_id"],
                "camera_id": camera_id,
                "world_position_m": _camera_center_world_m(camera.get("E") or ()),
                "scene_position": anchor["scene_position"],
            }
        )

    result = solve_scene_similarity(
        correspondences,
        source="menon_virtual_device_camera_similarity_v1",
        residual_units="scene_units",
    )
    result["anchor_state_sha256"] = camera_anchor_state_sha256(correspondences)
    result["camera_calibration_sha256"] = _sha256_file(camera_calibration_path)
    result["anchor_residual_limit_m"] = limit_m
    if (
        float(result["position_rmse_m"]) > limit_m
        or float(result["max_residual_m"]) > limit_m
    ):
        raise SceneSimilarityBuildError(
            "camera-anchor similarity exceeds the admitted residual limit: "
            f"rmse={result['position_rmse_m']:.6f}m "
            f"max={result['max_residual_m']:.6f}m limit={limit_m:.6f}m"
        )
    return result


def _alignment_with_similarity(path: Path, similarity: Mapping[str, Any]) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise SceneSimilarityBuildError("alignment config must be a JSON object")
    result = dict(payload)
    result["scene_similarity"] = dict(similarity)
    return result


def _verify_alignment(alignment: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    current = alignment.get("scene_similarity")
    if not isinstance(current, Mapping):
        raise SceneSimilarityBuildError("alignment has no scene_similarity object")
    for key in ("anchor_state_sha256", "camera_calibration_sha256", "source"):
        if current.get(key) != expected.get(key):
            raise SceneSimilarityBuildError(f"alignment scene_similarity.{key} is stale")
    actual_matrix = np.asarray(current.get("world_to_scene_col_major"), dtype=np.float64)
    expected_matrix = np.asarray(expected.get("world_to_scene_col_major"), dtype=np.float64)
    if actual_matrix.shape != (16,) or not np.allclose(actual_matrix, expected_matrix, rtol=0, atol=1e-9):
        raise SceneSimilarityBuildError("alignment world_to_scene matrix does not match current anchors")
    if float(current.get("max_residual_m", math.inf)) > float(expected["anchor_residual_limit_m"]):
        raise SceneSimilarityBuildError("alignment camera-anchor residual exceeds its limit")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify the exact camera-anchor similarity from Noesis into Menon."
    )
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        default=REPO_ROOT / "config" / "camera_calibration.json",
    )
    parser.add_argument("--menon-devices", type=Path, required=True)
    parser.add_argument(
        "--alignment",
        type=Path,
        default=REPO_ROOT / "config" / "ply_alignment.json",
    )
    parser.add_argument("--max-anchor-residual-m", type=float, default=0.05)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        similarity = build_similarity(
            camera_calibration_path=args.camera_calibration.expanduser().resolve(),
            menon_devices_path=args.menon_devices.expanduser().resolve(),
            anchor_residual_limit_m=args.max_anchor_residual_m,
        )
        alignment = _alignment_with_similarity(args.alignment.expanduser().resolve(), similarity)
        if args.check:
            _verify_alignment(_load_json(args.alignment.expanduser().resolve()), similarity)
        encoded = json.dumps(alignment, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.expanduser().resolve().write_text(encoded, encoding="utf-8")
        else:
            sys.stdout.write(encoded)
    except Exception as exc:
        print(f"scene similarity build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
