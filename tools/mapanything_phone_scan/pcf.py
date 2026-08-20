from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from .alignment import NoesisAlignmentSettings
from .inference import MapAnythingScanSettings


REPO_ROOT = Path(__file__).resolve().parents[2]
ProgressCallback = Callable[[float, str], None]


class PCFReviewError(RuntimeError):
    """Raised when a Prior-Conditioned Fusion review candidate cannot be built."""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PCFReviewError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PCFReviewError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise PCFReviewError(f"{label} must contain a JSON object: {path}")
    return payload


def _require_file(path: Path, *, label: str) -> Path:
    if not path.is_file():
        raise PCFReviewError(f"{label} is missing: {path}")
    return path


def _run_logged(command: Sequence[str], *, log_path: Path, stage: str) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{stage}] {' '.join(command)}\n")
        log.flush()
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise PCFReviewError(
            f"{stage} failed with exit code {return_code}; inspect {log_path.name}"
        )


def _file_row(run_root: Path, path: Path) -> dict[str, Any]:
    _require_file(path, label="PCF artifact")
    return {
        "path": path.relative_to(run_root).as_posix(),
        "size_bytes": int(path.stat().st_size),
    }


def run_pcf_review_candidate(
    scan_dir: Path,
    run_root: Path,
    state: dict[str, Any],
    target: NoesisAlignmentSettings,
    mapanything: MapAnythingScanSettings,
    progress: ProgressCallback,
) -> dict[str, Any]:
    """Run the canonical review-only PCF path for an aligned DA3 phone walk."""
    scan_dir = scan_dir.resolve()
    run_root = run_root.resolve()
    if any(run_root.iterdir()):
        raise PCFReviewError(f"PCF run directory is not empty: {run_root}")

    if str(state.get("provider") or state.get("outputs", {}).get("provider")) != "da3":
        raise PCFReviewError("PCF requires DA3 to be the base reconstruction provider")
    alignment = state.get("alignment")
    if not isinstance(alignment, dict) or alignment.get("status") != "complete":
        raise PCFReviewError("PCF requires a completed Noesis alignment")
    alignment_results = alignment.get("results")
    if not isinstance(alignment_results, dict):
        raise PCFReviewError("PCF alignment results are missing")
    quality_gate = alignment_results.get("quality_gate")
    if not isinstance(quality_gate, dict) or quality_gate.get("passed") is not True:
        raise PCFReviewError("PCF requires a passed alignment quality gate")
    if alignment_results.get("target_camera_id") != target.camera_id:
        raise PCFReviewError("PCF alignment camera does not match the selected target")
    if alignment_results.get("target_revision_id") != target.target_revision.name:
        raise PCFReviewError("PCF alignment revision does not match the selected target")

    prepared_manifest = _require_file(
        scan_dir / "prepared_frames_manifest.json",
        label="prepared-frame manifest",
    )
    da3_raw = scan_dir / "outputs" / "raw"
    if not da3_raw.is_dir():
        raise PCFReviewError(f"DA3 raw output is missing: {da3_raw}")
    world_from_da3 = _require_file(
        scan_dir / "alignment" / "phone_ma_to_noesis_world.json",
        label="DA3-to-Noesis transform",
    )
    _require_file(target.target_revision / "room_points.npz", label="static room cloud")
    _require_file(target.calibration_path, label="camera calibration")

    conditioned_dir = run_root / "mapanything_da3_pose_sparse_depth"
    consensus_dir = run_root / "prior_conditioned_consensus_da3_carrier"
    evaluation_dir = run_root / "evaluation_static_world"
    log_path = run_root / "pcf_run.log"
    log_path.write_text(
        json.dumps(
            {
                "scan_id": state.get("id"),
                "prepared_manifest": str(prepared_manifest),
                "target_camera_id": target.camera_id,
                "target_revision": str(target.target_revision),
                "review_only": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    progress(0.03, "Starting DA3-conditioned MapAnything")
    conditioned_command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "mapanything_phone_scan" / "run_mapanything_prior_variants.py"),
        str(scan_dir),
        "--da3-raw",
        str(da3_raw),
        "--world-from-da3",
        str(world_from_da3),
        "--target-revision",
        str(target.target_revision),
        "--calibration",
        str(target.calibration_path),
        "--camera",
        target.camera_id,
        "--variants",
        "da3_pose_sparse_depth",
        "--output-root",
        str(run_root),
        "--model-id",
        mapanything.model_id,
        "--device",
        mapanything.device,
        "--amp-dtype",
        mapanything.amp_dtype,
        "--point-budget",
        str(mapanything.point_budget),
        "--max-joint-views",
        str(mapanything.max_joint_views),
        "--window-overlap-views",
        str(mapanything.window_overlap_views),
    ]
    if not mapanything.local_files_only:
        conditioned_command.append("--allow-download")
    _run_logged(
        conditioned_command,
        log_path=log_path,
        stage="DA3-conditioned MapAnything",
    )

    progress(0.62, "Fusing conditioned MapAnything with DA3")
    _run_logged(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "mapanything_phone_scan" / "build_consensus_fusion.py"),
            str(scan_dir),
            "--mapanything-raw",
            str(conditioned_dir / "raw"),
            "--da3-raw",
            str(da3_raw),
            "--pose-carrier",
            "da3",
            "--output-dir",
            str(consensus_dir),
        ],
        log_path=log_path,
        stage="PCF consistency fusion",
    )

    progress(0.78, "Evaluating PCF in the selected static-camera world")
    _run_logged(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "mapanything_phone_scan" / "evaluate_mapanything_prior_variants.py"),
            str(scan_dir),
            "--suite-root",
            str(run_root),
            "--da3-raw",
            str(da3_raw),
            "--prior-consensus-raw",
            str(consensus_dir / "raw"),
            "--variants",
            "da3_pose_sparse_depth",
            "--world-from-da3",
            str(world_from_da3),
            "--target-revision",
            str(target.target_revision),
            "--calibration",
            str(target.calibration_path),
            "--camera",
            target.camera_id,
            "--point-budget",
            str(mapanything.point_budget),
            "--output-dir",
            str(evaluation_dir),
        ],
        log_path=log_path,
        stage="PCF static-world evaluation",
    )

    progress(0.97, "Collecting saved PCF review artifacts")
    consensus_manifest_path = _require_file(
        consensus_dir / "consensus_manifest.json",
        label="PCF consensus manifest",
    )
    evaluation_metrics_path = _require_file(
        evaluation_dir / "evaluation_metrics.json",
        label="PCF evaluation metrics",
    )
    consensus = _read_json(consensus_manifest_path, label="PCF consensus manifest")
    evaluation = _read_json(evaluation_metrics_path, label="PCF evaluation metrics")
    candidates = evaluation.get("candidates")
    candidate = (
        candidates.get("prior_conditioned_consensus")
        if isinstance(candidates, dict)
        else None
    )
    if not isinstance(candidate, dict):
        raise PCFReviewError("PCF evaluation did not contain the selected candidate")

    artifact_paths = {
        "pcf_glb": consensus_dir / "consensus_surfel_reconstruction.glb",
        "collaboration_diagnostics": consensus_dir
        / "consensus_collaboration_diagnostics.png",
        "consensus_manifest": consensus_manifest_path,
        "conditioned_mapanything_glb": conditioned_dir / "reconstruction_points.glb",
        "conditioned_mapanything_manifest": conditioned_dir / "variant_manifest.json",
        "evaluation_overview": evaluation_dir / "prior_variant_static_world_overview.png",
        "diagnostic_layers": evaluation_dir
        / "prior_conditioned_consensus"
        / "static_world_heatmap_diagnostics.png",
        "point_preserving_layers": evaluation_dir
        / "prior_conditioned_consensus"
        / "phone_walk_point_layers_2p5cm.png",
        "static_alignment_topdown": evaluation_dir
        / "prior_conditioned_consensus"
        / "static_alignment_topdown.png",
        "fixed_camera_reprojection": evaluation_dir
        / "prior_conditioned_consensus"
        / "fixed_camera_reprojection.jpg",
        "evaluation_metrics": evaluation_metrics_path,
        "run_log": log_path,
    }
    artifacts = {
        key: _require_file(path, label=key).relative_to(run_root).as_posix()
        for key, path in artifact_paths.items()
    }
    files = [_file_row(run_root, path) for path in artifact_paths.values()]
    progress(1.0, "PCF review candidate and diagnostics are saved")
    return {
        "schema": "noesis.phone_scan.pcf_review.v1",
        "method": "prior_conditioned_consensus_da3_carrier",
        "review_only": True,
        "published_to_scene_prior": False,
        "coordinate_frame": evaluation.get("coordinate_frame"),
        "target_camera_id": target.camera_id,
        "target_revision_id": target.target_revision.name,
        "view_count": int(consensus.get("view_count") or 0),
        "fusion": consensus.get("fusion") or {},
        "surfel_fusion": consensus.get("surfel_fusion") or {},
        "multiview_consistency": consensus.get("multiview_consistency") or {},
        "heldout_even_to_odd_reprojection": consensus.get(
            "heldout_even_to_odd_reprojection"
        )
        or {},
        "evaluation": candidate,
        "artifacts": artifacts,
        "files": files,
    }


__all__ = ["PCFReviewError", "run_pcf_review_candidate"]
