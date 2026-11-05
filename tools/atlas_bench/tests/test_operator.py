"""Exercise spend/authority gates and the recorded producer/consumer path offline."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import atlas_bench as bench
from runners.worldlabs import BenchError, private_json
from metrics.inspect_artifacts import validate_camera


def camera():
    return {"intrinsics": {"width": 1280, "height": 720, "fx": 900, "fy": 900, "cx": 640, "cy": 360},
            "extrinsics": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "coordinateSystem": "rdf"}}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    here, repo, output = tmp_path / "tool", tmp_path / "repo", tmp_path / "outputs"
    here.mkdir()
    repo.mkdir()
    source = tmp_path / "source.jpg"
    source.write_bytes(b"exact prepared image")
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"identity":"fixed"}')
    frame = {"frame_id": "prepared:0:hash", "path": str(source), "sha256": bench.sha256(source), "timestamp_s": 0.5}
    baseline = {"id": "test", "frames": [frame], "cohorts": {"smoke": [frame["frame_id"]]},
                "evidence": [{"path": str(metadata), "size_bytes": metadata.stat().st_size,
                              "sha256": bench.sha256(metadata), "purpose": "test source"}]}
    private_json(here / "manifests/baseline/living179.json", baseline)
    private_json(here / "manifests/holdouts/living179.json", {"frame_ids": [], "frames": []})
    private_json(here / "config/baseline.lock.json", {
        "baseline_sha256": bench.sha256(here / "manifests/baseline/living179.json"),
        "holdouts_sha256": bench.sha256(here / "manifests/holdouts/living179.json")})
    config = {"id": "SMOKE", "purpose": "fixture", "task": "images2PosedRGBD", "input_mode": "cohort",
              "cohort": "smoke", "parameters": {"targetResolution": [1280, 720]}, "dependencies": [], "review_gates": []}
    private_json(here / "experiments/SMOKE/experiment.json", config)
    a01 = copy.deepcopy(config)
    a01.update(id="A01", dependencies=["SMOKE"], review_gates=["SMOKE"])
    private_json(here / "experiments/A01/experiment.json", a01)
    private_json(here / "config/experiments.json", {key: {"config": f"experiments/{key}/experiment.json"}
                                                for key in ("SMOKE", "A01")})
    monkeypatch.setattr(bench, "HERE", here)
    monkeypatch.setattr(bench, "REPO", repo)
    monkeypatch.setattr(bench.time, "sleep", lambda _: None)
    return bench.Workspace(str(output))


class FakeWorldLabs:
    submissions = []
    fail_once = False

    def __init__(self, local):
        pass

    def auth(self):
        return {"authenticated": True, "source": "fixture"}

    def runtime(self):
        return {"version": "v24.fixture"}

    def call(self, *args, **kwargs):
        return "fixture"

    def upload(self, source, record, mime=None):
        result = {"id": "asset_input"}
        private_json(record, result)
        return result

    def submit(self, task, request, key, record):
        self.submissions.append((task, key, json.loads(request.read_text())))
        if self.fail_once:
            type(self).fail_once = False
            raise BenchError("uncertain submission; resume same key")
        result = {"id": "op_fixture", "done": False}
        private_json(record, result)
        return result

    def poll(self, operation_id, record):
        result = {"id": operation_id, "done": True, "error": None, "response": {"frames": [{
            "camera": camera(), "imageAsset": {"assetId": "asset_rgb"},
            "depth": {"depthAsset": {"assetId": "asset_depth"}, "confidenceAsset": {"assetId": "asset_conf"}}}]}}
        private_json(record, result)
        return result

    def download(self, asset_id, destination, record):
        import cv2
        import numpy as np
        from PIL import Image
        if asset_id == "asset_depth":
            depth = np.full((720, 1280), 2., dtype=np.float32)
            depth[0, 0] = np.nan
            cv2.imwrite(str(destination), depth)
        else:
            Image.new("RGB", (1280, 720), (128, 64, 32)).save(destination)
        private_json(record, {"assetId": asset_id, "path": str(destination)})


@pytest.fixture
def api(monkeypatch):
    FakeWorldLabs.submissions = []
    FakeWorldLabs.fail_once = False
    monkeypatch.setattr(bench, "WorldLabs", FakeWorldLabs)
    return FakeWorldLabs


def test_source_and_manifest_drift_refuse(workspace):
    assert workspace.verify()["status"] == "matched"
    source = Path(workspace.baseline["frames"][0]["path"])
    source.write_bytes(b"changed")
    with pytest.raises(BenchError, match="Prepared frame changed"):
        workspace.verify()


def test_hash_guard_catches_same_size_metadata_drift(workspace):
    path = Path(workspace.baseline["evidence"][0]["path"])
    path.write_text(path.read_text().replace("fixed", "other"))
    with pytest.raises(BenchError, match="hash changed"):
        workspace.verify()


def test_holdouts_never_join_cohort(workspace):
    workspace.holdouts["frame_ids"] = [workspace.baseline["frames"][0]["frame_id"]]
    with pytest.raises(BenchError, match="holdouts"):
        workspace.verify()


def test_dependencies_and_reviews_block_before_api(workspace, api):
    with pytest.raises(BenchError, match="complete SMOKE.*review SMOKE"):
        bench.execute(workspace, "A01", "default")
    assert api.submissions == []
    assert not workspace.root.exists()


def test_authentication_blocks_before_output_or_upload(workspace, api, monkeypatch):
    def deny(self):
        raise BenchError("authentication missing")
    monkeypatch.setattr(api, "auth", deny)
    with pytest.raises(BenchError, match="authentication missing"):
        bench.execute(workspace, "SMOKE", "default")
    assert api.submissions == []
    assert not workspace.root.exists()


def test_recorded_smoke_consumes_parsers_and_preserves_completed_run(workspace, api):
    result = bench.execute(workspace, "SMOKE", "default")
    run = Path(result["directory"])
    assert result["artifacts"] == 3
    assert result["state"] == "completed"
    assert json.loads((run / "parse-report.json").read_text())["frames"][0]["depth"]["dtype"] == "float32"
    assert (run / "review.png").is_file()
    assert json.loads((run / "metrics.json").read_text())["api_cost"]["state"] == "unavailable"
    assert api.submissions[0][2]["refinePoses"] is False
    before = {p: bench.sha256(p) for p in run.rglob("*") if p.is_file()}
    with pytest.raises(BenchError, match="directory exists"):
        bench.execute(workspace, "SMOKE", "default")
    with pytest.raises(BenchError, match="Completed runs are immutable"):
        bench.execute(workspace, "SMOKE", "default", resume=True)
    assert before == {p: bench.sha256(p) for p in before}
    assert workspace.blockers("A01") == ["review SMOKE"]


def test_uncertain_submit_reuses_saved_key_and_identical_request(workspace, api):
    api.fail_once = True
    with pytest.raises(BenchError, match="uncertain submission"):
        bench.execute(workspace, "SMOKE", "default")
    bench.execute(workspace, "SMOKE", "default", resume=True)
    assert len(api.submissions) == 2
    assert api.submissions[0] == api.submissions[1]


def test_changed_saved_request_refuses_idempotent_retry(workspace, api):
    api.fail_once = True
    with pytest.raises(BenchError):
        bench.execute(workspace, "SMOKE", "default")
    path = workspace.root / "runs/SMOKE/default/request.json"
    value = json.loads(path.read_text())
    value["refinePoses"] = True
    private_json(path, value)
    with pytest.raises(BenchError, match="Saved request changed"):
        bench.execute(workspace, "SMOKE", "default", resume=True)


def test_refinement_requires_distinct_variant_id(workspace, api):
    with pytest.raises(BenchError, match="separate controlled variant"):
        bench.execute(workspace, "SMOKE", "default", refine=True)
    assert not api.submissions


@pytest.mark.parametrize("mutate", [
    lambda c: c["extrinsics"].update(coordinateSystem="unknown"),
    lambda c: c["extrinsics"].update(quaternion=[0, 0, 0, 0]),
    lambda c: c["intrinsics"].update(width=640),
    lambda c: c["intrinsics"].update(fx=float("nan")),
])
def test_camera_contract_rejects_invalid_geometry(mutate):
    value = camera()
    mutate(value)
    with pytest.raises(BenchError, match="Unparseable Atlas camera"):
        validate_camera(value, [1280, 720])


@pytest.mark.parametrize("value", ["../other", "/tmp/other", ".", "a/b"])
def test_run_ids_cannot_escape_workspace(value):
    with pytest.raises(BenchError):
        bench.identifier(value)


def test_outputs_must_stay_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "REPO", tmp_path)
    with pytest.raises(BenchError, match="outside the repository"):
        bench.Workspace(str(tmp_path / "large_outputs"))


def chisel_bundle(workspace, tmp_path):
    import cv2
    import numpy as np
    config = {"id": "C01", "task": "atlasChisel", "input_mode": "prepared_bundle"}
    private_json(bench.HERE / "experiments/C01/experiment.json", config)
    workspace.catalog["C01"] = {"config": "experiments/C01/experiment.json"}
    depth = tmp_path / "layout.exr"
    cv2.imwrite(str(depth), np.full((720, 1280), 2., dtype=np.float32))
    source = workspace.baseline["frames"][0]
    return {"baseline_sha256": workspace.baseline_hash, "world_frame": "backend_world_m",
            "distance_unit": "m", "depth_semantics": "planar_z", "provenance": {"geometry": "frozen PCF"},
            "request": {"contextFrames": [{"camera": camera(), "depth": {"depthAsset": {
                "local_path": str(depth), "sha256": bench.sha256(depth)}}}],
                "sourceFrames": [{"camera": camera(), "imageAsset": {"local_path": source["path"],
                    "sha256": source["sha256"], "source_frame_id": source["frame_id"]}}],
                "targetCameras": [camera()], "prompt": "The same real room", "returnDepth": False}}


def test_chisel_refuses_ray_depth_and_mismatched_camera_order(workspace, tmp_path):
    bundle = chisel_bundle(workspace, tmp_path)
    bench.validate_bundle(workspace, "C01", bundle)
    bundle["depth_semantics"] = "ray_distance"
    with pytest.raises(BenchError, match="planar_z"):
        bench.validate_bundle(workspace, "C01", bundle)
    bundle["depth_semantics"] = "planar_z"
    bundle["request"]["contextFrames"][0]["camera"]["extrinsics"]["position"][0] = 1.
    with pytest.raises(BenchError, match="cameras differ"):
        bench.validate_bundle(workspace, "C01", bundle)


def test_holdout_cannot_be_uploaded_under_an_alias_or_without_id(workspace, tmp_path):
    bundle = chisel_bundle(workspace, tmp_path)
    source = workspace.baseline["frames"][0]
    workspace.holdouts["frames"] = [source]
    del bundle["request"]["sourceFrames"][0]["imageAsset"]["source_frame_id"]
    with pytest.raises(BenchError, match="holdouts"):
        bench.validate_bundle(workspace, "C01", bundle)


def test_static_reference_cannot_become_an_appearance_anchor(workspace, tmp_path):
    bundle = chisel_bundle(workspace, tmp_path)
    source = workspace.baseline["frames"][0]
    workspace.baseline["evidence"].append({"purpose": "independent fixed-camera evidence",
        "path": "reference/keyframe.png", "sha256": source["sha256"]})
    with pytest.raises(BenchError, match="Independent static reference"):
        bench.validate_bundle(workspace, "C01", bundle)


def test_single_chisel_target_cannot_claim_returned_depth(workspace, tmp_path):
    bundle = chisel_bundle(workspace, tmp_path)
    bundle["request"]["returnDepth"] = True
    with pytest.raises(BenchError, match="two translated"):
        bench.validate_bundle(workspace, "C01", bundle)


def test_point_cloud_is_separate_bounded_and_cannot_overwrite(workspace, api):
    from runners.derive_cloud import derive
    result = bench.execute(workspace, "SMOKE", "default")
    run = Path(result["directory"])
    before = bench.sha256(run / "run.json")
    destination = workspace.root / "derived/SMOKE/default/cloud"
    result = derive(run, destination, stride=16, max_points=100)
    assert result["point_count"] <= 100
    assert bench.sha256(run / "run.json") == before
    with pytest.raises(BenchError, match="Derived output exists"):
        derive(run, destination)


def test_refinement_variant_does_not_unlock_control_dependencies(workspace, api):
    bench.execute(workspace, "SMOKE", "refinement01", refine=True)
    assert not workspace.completed("SMOKE")
    assert "complete SMOKE" in workspace.blockers("A01")


def test_registration_neighborhood_checks_actual_target_pose(workspace, tmp_path):
    bundle = chisel_bundle(workspace, tmp_path)
    source = bundle["request"]["sourceFrames"][0]
    bundle["request"]["contextFrames"][0]["imageAsset"] = source["imageAsset"]
    del bundle["request"]["sourceFrames"]
    config = {"id": "R02", "task": "atlasGenerate", "input_mode": "prepared_bundle"}
    private_json(bench.HERE / "experiments/R02/experiment.json", config)
    workspace.catalog["R02"] = {"config": "experiments/R02/experiment.json"}
    report = tmp_path / "registration-report.json"
    private_json(report, {"fitted_on": "input camera poses only"})
    bundle["registration"] = {"source_frame": "atlas", "target_frame": "backend_world_m",
        "target_from_source_row_major": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        "fit_frame_ids": [workspace.baseline["frames"][0]["frame_id"]],
        "report_path": str(report), "report_sha256": bench.sha256(report)}
    bundle["pose_neighborhood"] = {"max_translation_m": 0.1, "max_rotation_deg": 2,
                                   "reference_camera": camera()}
    bench.validate_bundle(workspace, "R02", bundle)
    bundle["request"]["targetCameras"][0]["extrinsics"]["position"][0] = 0.2
    with pytest.raises(BenchError, match="outside its declared"):
        bench.validate_bundle(workspace, "R02", bundle)
