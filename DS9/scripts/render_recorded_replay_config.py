#!/usr/bin/env python3
"""Render a restartable DS9.1 config for paced local-video replay."""

from __future__ import annotations

import argparse
import copy
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


CANONICAL_CAMERA_ORDER = ("living-room", "kitchen", "family-room")
SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".mov"})


class ReplayConfigError(ValueError):
    """The requested replay configuration is unsafe or incomplete."""


def _camera_video(value: str) -> tuple[str, Path]:
    camera_id, separator, raw_path = str(value or "").partition("=")
    camera_id = camera_id.strip()
    raw_path = raw_path.strip()
    if not separator or camera_id not in CANONICAL_CAMERA_ORDER or not raw_path:
        raise argparse.ArgumentTypeError(
            "--source must be CAMERA_ID=/absolute/video with CAMERA_ID one of "
            + ", ".join(CANONICAL_CAMERA_ORDER)
        )
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("replay video paths must be absolute")
    path = path.resolve(strict=False)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"replay video is missing: {path}")
    if path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        raise argparse.ArgumentTypeError(f"unsupported replay video extension: {path.suffix}")
    return camera_id, path


def render_replay_config(
    baseline: Mapping[str, Any],
    videos: Mapping[str, Path],
    *,
    baseline_path: Path | None = None,
) -> dict[str, Any]:
    if set(videos) != set(CANONICAL_CAMERA_ORDER):
        missing = sorted(set(CANONICAL_CAMERA_ORDER) - set(videos))
        extra = sorted(set(videos) - set(CANONICAL_CAMERA_ORDER))
        details = []
        if missing:
            details.append(f"missing cameras: {', '.join(missing)}")
        if extra:
            details.append(f"unknown cameras: {', '.join(extra)}")
        raise ReplayConfigError("canonical replay requires exactly three sources; " + "; ".join(details))

    payload = copy.deepcopy(dict(baseline))
    sources = payload.get("sources")
    if not isinstance(sources, list) or len(sources) != len(CANONICAL_CAMERA_ORDER):
        raise ReplayConfigError("baseline pipeline config must contain the three canonical sources")
    for index, camera_id in enumerate(CANONICAL_CAMERA_ORDER):
        source = sources[index]
        if not isinstance(source, dict):
            raise ReplayConfigError(f"source {index} must be a mapping")
        source.pop("uri_secret", None)
        source["uri"] = videos[camera_id].resolve().as_uri()

    streammux = payload.get("streammux")
    if not isinstance(streammux, dict):
        raise ReplayConfigError("baseline streammux config must be a mapping")
    streammux["live-source"] = 0
    payload["recorded_replay"] = {
        "realtime": True,
        "preserve_frames": True,
    }
    if baseline_path is not None:
        _preserve_baseline_path_scope(payload, baseline_path)
    return payload


def _preserve_baseline_path_scope(payload: dict[str, Any], baseline_path: Path) -> None:
    baseline_path = baseline_path.expanduser().resolve(strict=True)
    if len(baseline_path.parents) < 3:
        raise ReplayConfigError("baseline config path is too shallow")
    ds9_root = baseline_path.parents[1]
    if ds9_root.name != "DS9":
        raise ReplayConfigError("baseline config must live under the DS9 tree")

    def preserve(raw: Any) -> str:
        value = str(raw or "").strip()
        if not value:
            return value
        candidate = Path(value)
        if candidate.is_absolute() or value.startswith("DS9/"):
            return value
        if value.startswith(("config/", "pipelines/", "build/")):
            return str((ds9_root / candidate).resolve())
        return str((baseline_path.parent / candidate).resolve())

    for section_name in ("depth_registration", "scene_priors", "scene_fusions"):
        section = payload.get(section_name)
        if isinstance(section, dict) and str(section.get("path") or "").strip():
            section["path"] = preserve(section["path"])

    for source in payload.get("sources", []):
        if not isinstance(source, dict):
            continue
        dewarper = source.get("dewarper")
        if isinstance(dewarper, dict) and str(dewarper.get("config-file") or "").strip():
            dewarper["config-file"] = preserve(dewarper["config-file"])

    for section_name in ("preprocess", "tracker"):
        section = payload.get(section_name)
        if isinstance(section, dict) and str(section.get("config-file") or "").strip():
            section["config-file"] = preserve(section["config-file"])

    analytics = payload.get("analytics")
    if isinstance(analytics, dict):
        for key in ("config-file", "stages_config"):
            if str(analytics.get(key) or "").strip():
                analytics[key] = preserve(analytics[key])
        exclude = analytics.get("exclude")
        if isinstance(exclude, dict) and str(exclude.get("config-file") or "").strip():
            exclude["config-file"] = preserve(exclude["config-file"])


def write_atomic_yaml(output: Path, payload: Mapping[str, Any]) -> None:
    output = output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(payload), handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("DS9/config/infer.yaml"),
        help="canonical DS9.1 pipeline config to copy without quality changes",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        type=_camera_video,
        required=True,
        help="CAMERA_ID=/absolute/video; provide each canonical camera exactly once",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    baseline_path = args.base_config.expanduser().resolve(strict=True)
    baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict):
        raise ReplayConfigError("baseline pipeline config must be a mapping")
    videos: dict[str, Path] = {}
    for camera_id, video_path in args.source:
        if camera_id in videos:
            raise ReplayConfigError(f"duplicate replay source: {camera_id}")
        videos[camera_id] = video_path
    payload = render_replay_config(baseline, videos, baseline_path=baseline_path)
    write_atomic_yaml(args.output, payload)
    print(args.output.expanduser().resolve(strict=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
