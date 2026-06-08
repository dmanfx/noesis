from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_image(path: str | Path) -> np.ndarray:
    try:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    except Exception as exc:
        raise RuntimeError(f"unable to load image {path}: {exc}") from exc


def _write_diff_image(diff: np.ndarray, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    scaled = np.clip(diff * 4, 0, 255).astype(np.uint8)
    try:
        from PIL import Image

        Image.fromarray(scaled, mode="RGB").save(target)
    except Exception as exc:
        raise RuntimeError(f"unable to write image diff {target}: {exc}") from exc
    return target


def compare_image_files(
    actual_path: str | Path,
    golden_path: str | Path,
    *,
    diff_path: str | Path | None = None,
    pixel_tolerance: int = 4,
    max_mean_abs_error: float = 1.0,
    max_changed_ratio: float = 0.01,
) -> dict[str, Any]:
    actual = _load_image(actual_path)
    golden = _load_image(golden_path)
    if actual.shape != golden.shape:
        return {
            "status": "fail",
            "reason": "shape_mismatch",
            "actual_shape": list(actual.shape),
            "golden_shape": list(golden.shape),
        }
    diff = np.abs(actual - golden)
    mean_abs_error = float(np.mean(diff))
    changed_ratio = float(np.count_nonzero(np.any(diff > int(pixel_tolerance), axis=2)) / max(1, diff.shape[0] * diff.shape[1]))
    status = "pass" if mean_abs_error <= max_mean_abs_error and changed_ratio <= max_changed_ratio else "fail"
    payload: dict[str, Any] = {
        "status": status,
        "mean_abs_error": mean_abs_error,
        "changed_ratio": changed_ratio,
        "pixel_tolerance": int(pixel_tolerance),
        "max_mean_abs_error": float(max_mean_abs_error),
        "max_changed_ratio": float(max_changed_ratio),
    }
    if diff_path is not None:
        payload["diff_path"] = str(_write_diff_image(diff, diff_path))
    return payload


def compare_json_subset(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, Mapping) and isinstance(actual_value, Mapping):
            for child in compare_json_subset(actual_value, expected_value):
                failures.append(f"{key}.{child}")
        elif actual_value != expected_value:
            failures.append(f"{key}: actual={actual_value!r} expected={expected_value!r}")
    return failures


def _artifact_path(report_payload: Mapping[str, Any], key: str, *, run_dir: Path) -> Path | None:
    artifacts = report_payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    entry = artifacts.get(key)
    if isinstance(entry, Mapping):
        raw_path = entry.get("path")
    else:
        raw_path = entry
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = run_dir / path
    return path


def compare_artifact_expectations(
    report_payload: Mapping[str, Any],
    *,
    run_dir: str | Path,
    expectations: Sequence[Mapping[str, Any]],
    registry_dir: str | Path,
    diff_dir: str | Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    failures: list[str] = []
    comparisons: list[dict[str, Any]] = []
    run_root = Path(run_dir)
    registry_root = Path(registry_dir)
    diff_root = Path(diff_dir)
    for idx, expectation in enumerate(expectations):
        key = str(expectation.get("key") or "").strip()
        if not key:
            failures.append(f"artifact expectation {idx} is missing key")
            continue
        actual = _artifact_path(report_payload, key, run_dir=run_root)
        comparison: dict[str, Any] = {"key": key, "status": "pass"}
        if actual is None:
            comparison.update({"status": "fail", "reason": "missing_artifact_reference"})
            failures.append(f"{key}: missing artifact reference")
            comparisons.append(comparison)
            continue
        comparison["actual_path"] = str(actual)
        if not actual.is_file():
            comparison.update({"status": "fail", "reason": "missing_artifact_file"})
            failures.append(f"{key}: missing artifact file {actual}")
            comparisons.append(comparison)
            continue
        min_bytes = expectation.get("min_bytes")
        if min_bytes is not None and actual.stat().st_size < int(min_bytes):
            comparison.update({"status": "fail", "reason": "artifact_too_small", "size_bytes": actual.stat().st_size, "min_bytes": int(min_bytes)})
            failures.append(f"{key}: artifact size {actual.stat().st_size} below {int(min_bytes)}")
        expected_sha256 = expectation.get("sha256")
        if isinstance(expected_sha256, str) and expected_sha256:
            digest = sha256_file(actual)
            comparison["sha256"] = digest
            if digest != expected_sha256:
                comparison.update({"status": "fail", "reason": "sha256_mismatch", "expected_sha256": expected_sha256})
                failures.append(f"{key}: sha256 mismatch")
        golden_path = expectation.get("golden_path")
        if isinstance(golden_path, str) and golden_path:
            golden = Path(golden_path)
            if not golden.is_absolute():
                golden = registry_root / golden
            diff_path = diff_root / f"{key}_diff.png"
            image_result = compare_image_files(
                actual,
                golden,
                diff_path=diff_path,
                pixel_tolerance=int(expectation.get("pixel_tolerance", 4)),
                max_mean_abs_error=float(expectation.get("max_mean_abs_error", 1.0)),
                max_changed_ratio=float(expectation.get("max_changed_ratio", 0.01)),
            )
            comparison["image_diff"] = image_result
            if image_result.get("status") != "pass":
                comparison["status"] = "fail"
                failures.append(f"{key}: image diff failed {json.dumps(image_result, sort_keys=True)}")
        comparisons.append(comparison)
    return failures, comparisons
