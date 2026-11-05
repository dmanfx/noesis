#!/usr/bin/env python3
"""Review-only Atlas benchmark operator. No PCF/runtime imports or mutations."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

from runners.worldlabs import BenchError, WorldLabs, private_json, redact

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise BenchError("Run IDs must be 1-80 letters/digits/dots/underscores/hyphens")
    return value


class Workspace:
    def __init__(self, root: str | None = None):
        self.local = read_json(REPO / "data/atlas_bench/operator.json") if (REPO / "data/atlas_bench/operator.json").exists() else {}
        root_value = root or os.environ.get("NOESIS_ATLAS_BENCH_ROOT") or self.local.get("output_root")
        self.root = Path(root_value).expanduser().resolve() if root_value else Path.home() / ".local/share/noesis/atlas_bench"
        if self.root.is_relative_to(REPO):
            raise BenchError("NOESIS_ATLAS_BENCH_ROOT must resolve outside the repository")
        self.baseline_path = HERE / "manifests/baseline/living179.json"
        self.lock = read_json(HERE / "config/baseline.lock.json")
        self.baseline_hash = self.lock["baseline_sha256"]
        self.baseline = read_json(self.baseline_path)
        self.holdouts_path = HERE / "manifests/holdouts/living179.json"
        self.holdouts = read_json(self.holdouts_path)
        self.catalog = read_json(HERE / "config/experiments.json")

    def path(self, value: str) -> Path:
        bindings = {"NOESIS_REPO_ROOT": str(REPO),
                    "NOESIS_STORAGE_ROOT": os.environ.get("NOESIS_STORAGE_ROOT", self.local.get("storage_root", "/mnt/noesis_storage/noesis_devel"))}
        for key, replacement in bindings.items():
            value = value.replace("${" + key + "}", replacement)
        value = os.path.expandvars(value)
        if "${" in value:
            raise BenchError(f"Unresolved path variable in {value}")
        return Path(value).expanduser()

    def verify(self) -> dict:
        if sha256(self.baseline_path) != self.baseline_hash or sha256(self.holdouts_path) != self.lock["holdouts_sha256"]:
            raise BenchError("Benchmark baseline/holdout manifest changed; freeze and review a new baseline")
        checked = 0
        for item in self.baseline["evidence"]:
            path = self.path(item["path"])
            if not path.is_file():
                raise BenchError(f"Frozen evidence missing: {item['path']}")
            stat = path.stat()
            if stat.st_size != item["size_bytes"]:
                raise BenchError(f"Frozen evidence size changed: {item['path']}")
            if item.get("sha256") and sha256(path) != item["sha256"]:
                raise BenchError(f"Frozen evidence hash changed: {item['path']}")
            if item.get("mtime_ns") and stat.st_mtime_ns != item["mtime_ns"]:
                raise BenchError(f"Frozen provider evidence timestamp changed: {item['path']}")
            checked += 1
        for frame in self.baseline["frames"]:
            if sha256(self.path(frame["path"])) != frame["sha256"]:
                raise BenchError(f"Prepared frame changed: {frame['frame_id']}")
        holdout_ids = set(self.holdouts["frame_ids"])
        known_ids = {f["frame_id"] for f in self.baseline["frames"]}
        if not holdout_ids <= known_ids:
            raise BenchError("Holdouts are not members of the frozen frame cohort")
        for name, selected in self.baseline["cohorts"].items():
            if len(selected) != len(set(selected)) or not set(selected) <= known_ids or set(selected) & holdout_ids:
                raise BenchError(f"Cohort {name} contains duplicates, unknown frames, or holdouts")
        return {"status": "matched", "metadata_and_provider_files": checked,
                "prepared_images": len(known_ids), "baseline_sha256": self.baseline_hash}

    def experiment(self, experiment_id: str) -> dict:
        try:
            return read_json(HERE / self.catalog[experiment_id]["config"])
        except KeyError as exc:
            raise BenchError(f"Unknown experiment {experiment_id}; use list") from exc

    def runs(self, experiment_id: str) -> list[dict]:
        rows = []
        for path in sorted((self.root / "runs" / experiment_id).glob("*/run.json")):
            row = read_json(path)
            row["directory"] = str(path.parent)
            rows.append(row)
        return sorted(rows, key=lambda row: row.get("started_at", ""))

    def completed(self, experiment_id: str) -> list[dict]:
        config = self.experiment(experiment_id)
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        return [row for row in self.runs(experiment_id)
                if row.get("state") == "completed" and row.get("baseline_sha256") == self.baseline_hash
                and row.get("experiment_sha256") == digest and not row.get("refine_poses", False)]

    def reviewed(self, experiment_id: str) -> bool:
        for path in (self.root / "reviews" / experiment_id).glob("*.json"):
            review = read_json(path)
            source = self.root / "runs" / experiment_id / identifier(review["run_id"]) / "run.json"
            if (review.get("baseline_sha256") == self.baseline_hash and source.exists()
                    and sha256(source) == review["run_sha256"]
                    and any(row["run_id"] == review["run_id"] for row in self.completed(experiment_id))):
                return True
        return False

    def blockers(self, experiment_id: str) -> list[str]:
        config = self.experiment(experiment_id)
        result = [f"complete {dependency}" for dependency in config["dependencies"] if not self.completed(dependency)]
        result += [f"review {gate}" for gate in config["review_gates"] if not self.reviewed(gate)]
        if config["input_mode"] == "prepared_bundle" and not (self.root / "preparations" / experiment_id / "default.json").exists():
            result.append(f"prepare {experiment_id} --bundle <reviewed-local-inputs.json>")
        return result


def walk_assets(value: object, prefix: str = ""):
    if isinstance(value, dict):
        if "assetId" in value or "local_path" in value or "url" in value or "base64" in value:
            yield prefix, value
        else:
            for key, nested in value.items():
                yield from walk_assets(nested, f"{prefix}.{key}".lstrip("."))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from walk_assets(nested, f"{prefix}.{index}".lstrip("."))


def input_bundle(ws: Workspace, experiment_id: str) -> dict:
    path = ws.root / "preparations" / experiment_id / "default.json"
    if not path.exists():
        raise BenchError(f"Run prepare {experiment_id} --bundle <local-inputs.json>; see experiment README")
    bundle = read_json(path)
    validate_bundle(ws, experiment_id, bundle)
    return bundle


def validate_bundle(ws: Workspace, experiment_id: str, bundle: dict) -> None:
    from metrics.inspect_artifacts import validate_camera
    import cv2
    import numpy as np
    config = ws.experiment(experiment_id)
    if config["input_mode"] != "prepared_bundle":
        raise BenchError("This experiment uses the frozen prepared-image cohort directly")
    if bundle.get("baseline_sha256") != ws.baseline_hash:
        raise BenchError("Prepared inputs refer to another baseline")
    if not bundle.get("world_frame") or not bundle.get("distance_unit") or not bundle.get("provenance"):
        raise BenchError("Prepared inputs need explicit world_frame, distance_unit and provenance")
    request = bundle["request"]
    targets = request.get("targetCameras", [])
    if not targets or len(targets) > 32:
        raise BenchError("Prepared generation needs 1-32 explicit target cameras")
    for camera in targets:
        validate_camera(camera, [1280, 720])
    holdouts = set(ws.holdouts["frame_ids"])
    holdout_hashes = {frame["sha256"] for frame in ws.holdouts["frames"]}
    static_rgb_hashes = {item["sha256"] for item in ws.baseline["evidence"]
                         if item["purpose"] == "independent fixed-camera evidence"
                         and item["path"].endswith(("keyframe.png", "keyframe_valid_mask.png"))}
    known = {frame["frame_id"]: frame for frame in ws.baseline["frames"]}
    for field, asset in walk_assets(request):
        if not asset.get("local_path") or not asset.get("sha256"):
            raise BenchError(f"Preparation assets must be local hash-bound files: {field}")
        source = ws.path(asset["local_path"])
        if sha256(source) != asset["sha256"]:
            raise BenchError(f"Prepared asset changed: {field}")
        if asset.get("source_frame_id") in holdouts or asset["sha256"] in holdout_hashes:
            raise BenchError("Untouched holdouts cannot be appearance/context inputs")
        if asset["sha256"] in static_rgb_hashes:
            raise BenchError("Independent static reference cannot become a phone/appearance context input")
        if field.startswith("sourceFrames"):
            source_frame = known.get(asset.get("source_frame_id"))
            if not source_frame or source_frame["sha256"] != asset["sha256"]:
                raise BenchError("Chisel appearance anchors must identify exact real non-holdout prepared images")
        if "depthAsset" in field:
            depth = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            if depth is None or depth.ndim != 2 or depth.shape != (720, 1280) or depth.dtype.kind != "f":
                raise BenchError("Context depth must be 1280x720 single-channel float EXR")
            if not (np.isfinite(depth) & (depth > 0)).any():
                raise BenchError("Context depth has no finite positive surfaces")
    context = request.get("contextFrames", [])
    if config["task"] == "atlasChisel":
        if len(context) != len(targets) or not request.get("sourceFrames") or not request.get("prompt"):
            raise BenchError("Chisel hybrid requires matched depth context, real RGB anchors and factual prompt")
        if bundle.get("depth_semantics") != "planar_z" or bundle["distance_unit"] != "m":
            raise BenchError("PCF Chisel depth must be explicitly metric planar_z")
        for index, frame in enumerate(context):
            if set(frame) != {"camera", "depth"} or set(frame["depth"]) != {"depthAsset"}:
                raise BenchError("Chisel context must contain camera and depth only")
            if frame["camera"] != targets[index]:
                raise BenchError("Chisel context cameras differ from targets or their order")
        if 2 * len(targets) + len(request["sourceFrames"]) > 64:
            raise BenchError("Chisel exceeds its 64-view rollout limit")
        for frame in request["sourceFrames"]:
            if set(frame) != {"imageAsset", "camera"}:
                raise BenchError("Chisel RGB anchors must contain image and camera only")
            validate_camera(frame["camera"])
        if request.get("returnDepth") and (len(targets) < 2 or len({tuple(c["extrinsics"]["position"]) for c in targets}) < 2):
            raise BenchError("Chisel depth needs two translated target cameras")
        if experiment_id == "C03" and not request.get("returnDepth"):
            raise BenchError("C03 geometry preservation requires actual returned depth")
    else:
        if not context:
            raise BenchError("Atlas generation requires posed RGBD context")
        for frame in context:
            if not all(key in frame for key in ("imageAsset", "camera", "depth")):
                raise BenchError("Atlas context must be a complete posed RGBD bundle")
            validate_camera(frame["camera"], [1280, 720])
        if experiment_id in {"A04", "R01", "R02"} and not bundle.get("registration"):
            raise BenchError("Matched real-camera generation requires a recorded offline Atlas/PCF registration")
        if experiment_id in {"A04", "R01", "R02"}:
            registration = bundle["registration"]
            required = {"source_frame", "target_frame", "target_from_source_row_major",
                        "fit_frame_ids", "report_path", "report_sha256"}
            if not required <= set(registration):
                raise BenchError("Registration needs explicit frames, Sim3, input fit IDs and a hash-bound diagnostic report")
            matrix = np.asarray(registration["target_from_source_row_major"], dtype=float)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
                raise BenchError("Registration must be a finite homogeneous 4x4 Sim3")
            scale = float(np.linalg.det(matrix[:3, :3]))
            if scale <= 0:
                raise BenchError("Registration must preserve handedness and positive scale")
            rotation = matrix[:3, :3] / np.cbrt(scale)
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
                raise BenchError("Registration may not shear cameras or depth")
            fit_ids = set(registration["fit_frame_ids"])
            if not fit_ids or not fit_ids <= set(known) or fit_ids & holdouts:
                raise BenchError("Registration fit must use input camera correspondences, excluding holdouts")
            if sha256(ws.path(registration["report_path"])) != registration["report_sha256"]:
                raise BenchError("Registration diagnostic report changed")
    if experiment_id == "R02":
        bounds = bundle.get("pose_neighborhood")
        if (not bounds or not 0 <= bounds.get("max_translation_m", 999) <= 0.10
                or not 0 <= bounds.get("max_rotation_deg", 999) <= 2):
            raise BenchError("R02 needs a bounded diagnostic neighborhood <=10 cm and <=2 degrees")
        reference = bounds.get("reference_camera")
        if not reference or bundle["distance_unit"] != "m":
            raise BenchError("R02 needs a reference camera in a registered metric frame")
        validate_camera(reference, [1280, 720])
        for target in targets:
            origin, candidate = reference["extrinsics"], target["extrinsics"]
            translation = float(np.linalg.norm(np.asarray(origin["position"]) - candidate["position"]))
            angle = float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(origin["quaternion"], candidate["quaternion"])), 0, 1))))
            if (origin["coordinateSystem"] != candidate["coordinateSystem"]
                    or translation > bounds["max_translation_m"] + 1e-8
                    or angle > bounds["max_rotation_deg"] + 1e-6):
                raise BenchError("R02 target lies outside its declared metric pose neighborhood")


def make_request(ws: Workspace, config: dict, run: Path, api: WorldLabs, state: dict) -> dict:
    uploads = run / "uploads"
    uploads.mkdir(exist_ok=True, mode=0o700)
    if config["input_mode"] == "cohort":
        selected = ws.baseline["cohorts"][config["cohort"]]
        by_id = {row["frame_id"]: row for row in ws.baseline["frames"]}
        request = copy.deepcopy(config["parameters"])
        request["refinePoses"] = state["refine_poses"]
        request["frames"] = []
        inputs = []
        for index, frame_id in enumerate(selected):
            frame = by_id[frame_id]
            record = uploads / f"input_{index:04d}.json"
            result = read_json(record) if record.exists() else api.upload(ws.path(frame["path"]), record)
            request["frames"].append({"imageAsset": {"assetId": result["id"]}})
            inputs.append({"frame_id": frame_id, "timestamp_s": frame["timestamp_s"],
                           "source_path": frame["path"], "sha256": frame["sha256"], "asset_id": result["id"]})
    else:
        bundle = input_bundle(ws, config["id"])
        private_json(run / "prepared-inputs.json", bundle)
        request = copy.deepcopy(bundle["request"])
        inputs = []
        for index, (field, asset) in enumerate(list(walk_assets(request))):
            source = ws.path(asset["local_path"])
            record = uploads / f"input_{index:04d}.json"
            result = read_json(record) if record.exists() else api.upload(source, record, asset.get("mime"))
            inputs.append({"field": field, "source_path": asset["local_path"],
                           "sha256": asset["sha256"], "asset_id": result["id"]})
            asset.clear()
            asset["assetId"] = result["id"]
    private_json(run / "input-assets.json", inputs)
    private_json(run / "request.json", request)
    return request


def download_outputs(api: WorldLabs, run: Path, response: dict) -> list[dict]:
    artifacts = []
    for field, asset in walk_assets(response):
        if not asset.get("assetId"):
            raise BenchError(f"Returned asset {field} lacks an asset ID; preserve result and inspect before retry")
        extension = ".exr" if field.endswith(".depthAsset") else ".png"
        destination = run / "artifacts" / (re.sub(r"[^A-Za-z0-9_-]", "_", field) + extension)
        destination.parent.mkdir(exist_ok=True, mode=0o700)
        receipt = destination.with_suffix(destination.suffix + ".download.json")
        if not destination.exists():
            api.download(asset["assetId"], destination, receipt)
        elif not receipt.exists():
            raise BenchError("Incomplete download found; preserve it and recover explicitly before resume")
        artifacts.append({"field": field, "asset_id": asset["assetId"],
                          "path": str(destination.relative_to(run)), "sha256": sha256(destination),
                          "size_bytes": destination.stat().st_size})
    private_json(run / "artifacts.json", artifacts)
    cameras = [frame["camera"] for frame in response.get("frames", [])]
    private_json(run / "cameras.json", cameras)
    return artifacts


def write_report(run: Path, state: dict, metrics: dict, artifacts: list[dict]) -> None:
    lines = [f"# {state['experiment_id']} / {state['run_id']}", "", "Offline review evidence.", "",
             f"- State: {state['state']}", f"- Task: {state['task']}",
             f"- Operation: {state.get('operation_id', 'not submitted')}",
             f"- Baseline SHA256: `{state['baseline_sha256']}`",
             f"- Elapsed seconds: {state.get('elapsed_seconds', 0):.2f}",
             f"- API cost: {json.dumps(metrics.get('api_cost', {}))}",
             "", "Smoke parsing does not establish metric accuracy, registration or PCF superiority.",
             "", "[RGB/depth/confidence review](review.png)", "",
             "[Request](request.json) | [API operation](operation.json) | [Cameras](cameras.json) | "
             "[Metrics](metrics.json) | [Inputs](input-assets.json)", "", "## Returned artifacts", ""]
    lines += [f"- [{a['field']}]({a['path']}) — {a['size_bytes']} bytes; SHA256 `{a['sha256']}`" for a in artifacts]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n")
    (run / "REPORT.md").chmod(0o600)


def execute(ws: Workspace, experiment_id: str, run_id: str, refine: bool = False,
            resume: bool = False, wait_seconds: int = 900) -> dict:
    from metrics.inspect_artifacts import dependencies, inspect_frames, metric_hooks
    from visualization.review import contact_sheet
    config = ws.experiment(experiment_id)
    ws.verify()
    dependencies()
    blockers = ws.blockers(experiment_id)
    if blockers:
        raise BenchError("Blocked: " + "; ".join(blockers))
    if refine and (config["task"] != "images2PosedRGBD" or run_id == "default"):
        raise BenchError("Pose refinement is a separate controlled variant: supply an explicit new --run-id")
    api = WorldLabs(ws.local)
    api.runtime()
    auth = api.auth()
    run = ws.root / "runs" / experiment_id / identifier(run_id)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if resume:
        state = read_json(run / "run.json")
        if state["state"] == "completed":
            raise BenchError("Completed runs are immutable; use report or an explicit new --run-id")
        if state["baseline_sha256"] != ws.baseline_hash or state["experiment_sha256"] != digest:
            raise BenchError("Cannot resume against changed baseline or experiment configuration")
        if (run / "request.json").exists() and state.get("request_sha256") != sha256(run / "request.json"):
            raise BenchError("Saved request changed; refusing idempotent retry")
    else:
        run.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            run.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise BenchError("Run directory exists. Use resume for interrupted work or supply a new --run-id.") from exc
        state = {"schema": "noesis.atlas_bench.run.v1", "experiment_id": experiment_id,
                 "run_id": run_id, "task": config["task"], "started_at": utc(),
                 "state": "preparing", "idempotency_key": str(uuid.uuid4()),
                 "baseline_sha256": ws.baseline_hash, "experiment_sha256": digest,
                 "refine_poses": refine, "authority": "review_only", "runtime_mutations": False}
        private_json(run / "baseline.json", ws.baseline)
        private_json(run / "holdouts.json", ws.holdouts)
        private_json(run / "experiment.json", config)
    private_json(run / "run.json", state)
    started = time.monotonic()
    try:
        private_json(run / f"auth-{uuid.uuid4().hex[:8]}.json", auth)
        private_json(run / "tool-versions.json", {"cli": api.call(["--version"], json_output=False),
                                                 "python": sys.version, "parsers": dependencies()})
        if not (run / "request.json").exists():
            request = make_request(ws, config, run, api, state)
            state["request_sha256"] = sha256(run / "request.json")
            private_json(run / "run.json", state)
        else:
            request = read_json(run / "request.json")
        state["state"] = "submitted"
        private_json(run / "run.json", state)
        operation = read_json(run / "submission.json") if (run / "submission.json").exists() else api.submit(
            config["task"], run / "request.json", state["idempotency_key"], run / "submission.json")
        if not operation.get("id"):
            raise BenchError("Submission has no operation ID; recover with saved request and idempotency key")
        state["operation_id"] = operation["id"]
        private_json(run / "operation-id.json", {"id": operation["id"], "name": operation.get("name")})
        private_json(run / "run.json", state)
        print(f"Operation {operation['id']} accepted; saving artifacts under {run}", flush=True)
        poll_index = len(list((run / "polls").glob("*.json")))
        deadline = time.monotonic() + wait_seconds
        while not operation.get("done"):
            if time.monotonic() >= deadline:
                raise BenchError(f"Local wait ended; server work continues. resume {experiment_id} --run-id {run_id}")
            time.sleep(3)
            poll_index += 1
            operation = api.poll(operation["id"], run / "polls" / f"{poll_index:05d}.json")
        private_json(run / "operation.json", operation)
        if operation.get("error"):
            raise BenchError("API operation failed; inspect operation.json")
        response = operation.get("response")
        if not isinstance(response, dict):
            raise BenchError("Successful operation is missing its response")
        private_json(run / "response.json", response)
        artifacts = download_outputs(api, run, response)
        expected = len(request.get("frames", request.get("targetCameras", [])))
        parsed = inspect_frames(run, response, artifacts, expected,
                                require_depth=config["task"] == "images2PosedRGBD" or request.get("returnDepth", False))
        private_json(run / "parse-report.json", parsed)
        contact_sheet(run, artifacts)
        elapsed = state.get("elapsed_seconds", 0) + time.monotonic() - started
        metrics = metric_hooks()
        metrics["artifact_parsing"] = parsed
        metrics["runtime"] = {"state": "measured", "client_elapsed_seconds": elapsed,
                              "operation_created_at_secs": operation.get("createdAtSecs")}
        cost = {key: operation.get(key, response.get(key)) for key in ("cost", "usage", "billing")
                if operation.get(key) is not None or response.get(key) is not None}
        metrics["api_cost"] = {"state": "returned" if cost else "unavailable", "data": cost,
                               "note": "Never infer zero cost from absent billing data"}
        private_json(run / "metrics.json", metrics)
        state.update(state="completed", finished_at=utc(), elapsed_seconds=elapsed,
                     artifact_count=len(artifacts), frame_count=expected)
        write_report(run, state, metrics, artifacts)
        private_json(run / "run.json", state)
        return {"state": "completed", "experiment_id": experiment_id, "run_id": run_id,
                "operation_id": operation["id"], "directory": str(run), "artifacts": len(artifacts)}
    except BaseException as exc:
        state.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     elapsed_seconds=state.get("elapsed_seconds", 0) + time.monotonic() - started,
                     error=redact(str(exc)), last_attempt_at=utc())
        private_json(run / f"failure-{uuid.uuid4().hex[:8]}.json", state)
        private_json(run / "run.json", state)
        raise


def doctor(ws: Workspace) -> dict:
    from metrics.inspect_artifacts import dependencies
    result = {}
    def tasks():
        names = WorldLabs(ws.local).call(["tasks", "list"])
        if not {"images2PosedRGBD", "atlasGenerate", "atlasChisel"} <= set(names):
            raise BenchError("Required Atlas tasks missing from the installed CLI")
        return names

    def skill():
        path = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "skills/worldlabs/SKILL.md"
        if not path.is_file():
            raise BenchError("Install the Codex skill: worldlabs skills install --client codex")
        return {"path": str(path), "sha256": sha256(path)}

    checks = {"baseline": ws.verify, "parsers": dependencies,
              "node": lambda: WorldLabs(ws.local).runtime(), "codex_skill": skill,
              "auth": lambda: WorldLabs(ws.local).auth(),
              "cli_version": lambda: WorldLabs(ws.local).call(["--version"], json_output=False),
              "tasks": tasks}
    for name, check in checks.items():
        try:
            result[name] = {"ok": True, "value": check()}
        except (BenchError, OSError, ImportError, ValueError) as exc:
            result[name] = {"ok": False, "error": redact(str(exc))}
    try:
        ws.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        import tempfile
        with tempfile.TemporaryFile(dir=ws.root) as handle:
            handle.write(b"Atlas output permission check")
        usage = shutil.disk_usage(ws.root)
        result["output_root"] = {"ok": True, "path": str(ws.root), "free_bytes": usage.free}
    except OSError as exc:
        result["output_root"] = {"ok": False, "error": str(exc)}
    result["note"] = "Task list is local SDK discovery; auth checks assets. Actual task permission is proven by smoke."
    result["ready"] = all(value.get("ok", False) for value in result.values() if isinstance(value, dict))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="External output root (overrides NOESIS_ATLAS_BENCH_ROOT)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "status", "list", "next"):
        sub.add_parser(name)
    for name in ("run", "resume", "smoke"):
        item = sub.add_parser(name)
        if name != "smoke":
            item.add_argument("id")
        item.add_argument("--run-id", default="default")
        item.add_argument("--refine-poses", action="store_true")
        item.add_argument("--wait-seconds", type=int, default=900)
    for name in ("report", "review"):
        item = sub.add_parser(name)
        item.add_argument("id")
        item.add_argument("--run-id", default="default")
        if name == "review":
            item.add_argument("--note", required=True, help="Operator's review finding; unlocks dependent API runs")
        else:
            item.add_argument("--open", action="store_true")
    item = sub.add_parser("prepare")
    item.add_argument("id")
    item.add_argument("--bundle", type=Path, required=True)
    item = sub.add_parser("derive")
    item.add_argument("id")
    item.add_argument("--run-id", default="default")
    item.add_argument("--derived-id", default="cloud_stride8")
    item.add_argument("--stride", type=int, default=8)
    item.add_argument("--max-points", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    try:
        ws = Workspace(args.root)
        if args.command == "doctor":
            result = doctor(ws)
            print(json.dumps(result, indent=2))
            return 0 if result["ready"] else 1
        if args.command == "status":
            result = doctor(ws)
            result["experiments"] = {key: {"runs": [{k: row.get(k) for k in ("run_id", "state", "directory")}
                                                    for row in ws.runs(key)], "blockers": ws.blockers(key)}
                                     for key in ws.catalog}
        elif args.command == "list":
            result = [{"id": key, "purpose": ws.experiment(key)["purpose"],
                       "dependencies": ws.experiment(key)["dependencies"],
                       "state": "completed" if ws.completed(key) else (ws.runs(key)[-1]["state"] if ws.runs(key) else "pending"),
                       "blockers": ws.blockers(key)} for key in ws.catalog]
        elif args.command == "next":
            ws.verify()
            result = {"message": "All configured experiments completed; review reports."}
            for key in ws.catalog:
                if not ws.completed(key):
                    result = {"experiment": key, "purpose": ws.experiment(key)["purpose"],
                              "blockers": ws.blockers(key),
                              "command": f"python3 tools/atlas_bench/atlas_bench.py {'smoke' if key == 'SMOKE' else 'run ' + key}"}
                    runs = ws.runs(key)
                    if runs and runs[-1]["state"] != "completed":
                        result["existing_run"] = {k: runs[-1].get(k) for k in ("run_id", "state", "error")}
                        result["command"] = f"python3 tools/atlas_bench/atlas_bench.py resume {key} --run-id {runs[-1]['run_id']}"
                        result["note"] = "Inspect terminal API failures before choosing a new paid run. Resume preserves the original operation/key."
                    break
        elif args.command in ("smoke", "run", "resume"):
            experiment_id = "SMOKE" if args.command == "smoke" else args.id.upper()
            result = execute(ws, experiment_id, args.run_id, args.refine_poses,
                             args.command == "resume", args.wait_seconds)
        elif args.command == "derive":
            from runners.derive_cloud import derive
            ws.verify()
            run = ws.root / "runs" / args.id.upper() / identifier(args.run_id)
            destination = ws.root / "derived" / args.id.upper() / identifier(args.run_id) / identifier(args.derived_id)
            result = derive(run, destination, args.stride, args.max_points)
        elif args.command == "prepare":
            ws.verify()
            experiment_id = args.id.upper()
            bundle = read_json(args.bundle)
            validate_bundle(ws, experiment_id, bundle)
            path = ws.root / "preparations" / experiment_id / "default.json"
            if path.exists():
                raise BenchError("Preparation already exists; retain it and define a separately reviewed experiment variant")
            private_json(path, bundle)
            result = {"prepared": experiment_id, "path": str(path), "submitted": False}
        elif args.command == "review":
            ws.verify()
            experiment_id = args.id.upper()
            path = ws.root / "runs" / experiment_id / identifier(args.run_id) / "run.json"
            if not any(row["run_id"] == args.run_id for row in ws.completed(experiment_id)):
                raise BenchError("Only a successful run on the current frozen baseline can be reviewed")
            if not args.note.strip():
                raise BenchError("A review finding is required")
            review_path = ws.root / "reviews" / experiment_id / f"{uuid.uuid4().hex}.json"
            private_json(review_path, {"reviewed_at": utc(), "run_id": args.run_id,
                                      "run_sha256": sha256(path), "baseline_sha256": ws.baseline_hash,
                                      "operator_note": args.note, "authority": "offline_experiment_only"})
            result = {"review_recorded": experiment_id, "path": str(review_path)}
        else:
            path = ws.root / "runs" / args.id.upper() / identifier(args.run_id)
            if not (path / "run.json").exists():
                raise BenchError("No such run")
            report = path / "REPORT.md"
            if not report.exists():
                state = read_json(path / "run.json")
                report = ws.root / "reports" / args.id.upper() / identifier(args.run_id) / f"{uuid.uuid4().hex}.md"
                report.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                report.write_text(f"# {args.id.upper()} / {args.run_id}\n\nState: {state['state']}\n\n"
                                  f"{state.get('error', 'Still in progress')}\n\n"
                                  f"Private operation, command and failure records: {path}\n")
                report.chmod(0o600)
            result = {"run": read_json(path / "run.json"), "report": str(report),
                      "artifact_index": str(path / "artifacts.json")}
            if args.open:
                import webbrowser
                webbrowser.open(report.as_uri())
        print(json.dumps(result, indent=2))
        return 0
    except (BenchError, OSError, ValueError, KeyError, ImportError) as exc:
        print(json.dumps({"error": redact(str(exc))}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
