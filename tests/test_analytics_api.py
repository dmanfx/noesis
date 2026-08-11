from __future__ import annotations

import copy
import configparser
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, List

from fastapi.testclient import TestClient
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from noesis.server import analytics_api  # noqa: E402
from noesis.pipelines import ds8_pipeline  # noqa: E402
from noesis.pipelines import hooks  # noqa: E402


def _reset_pipeline() -> None:
    """Ensure the DS8 pipeline singleton is cleared between tests."""
    try:
        graph = ds8_pipeline.get_pipeline()
    except Exception:
        graph = None

    if graph is not None and getattr(graph, "_timer", None):
        graph._timer.cancel()

    ds8_pipeline._PIPELINE_SINGLETON = None  # type: ignore[attr-defined]


def _reset_analytics_state() -> None:
    analytics_api._CONFIG_CACHE = None  # type: ignore[attr-defined]
    analytics_api._CONFIG_PATH = None  # type: ignore[attr-defined]
    analytics_api._RELOAD_HOOK = None  # type: ignore[attr-defined]
    analytics_api._RELOAD_COUNTER = 0  # type: ignore[attr-defined]
    analytics_api._STATE_POISONED = None  # type: ignore[attr-defined]
    analytics_api._POISON_HOOK = None  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def analytics_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provide an isolated analytics config file and reset globals."""
    _reset_pipeline()
    _reset_analytics_state()

    cfg_src = Path("config/nvdsanalytics.yaml")
    cfg_dst = tmp_path / "nvdsanalytics.yaml"
    cfg_dst.write_text(cfg_src.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv(analytics_api.ANALYTICS_CONFIG_ENV, str(cfg_dst))
    monkeypatch.setenv("NOESIS_ANALYTICS_EXCLUDE_CONFIG", str(tmp_path / "config_nvdsanalytics_exclude.ini"))

    yield

    _reset_pipeline()
    _reset_analytics_state()
    monkeypatch.delenv(analytics_api.ANALYTICS_CONFIG_ENV, raising=False)
    monkeypatch.delenv("NOESIS_ANALYTICS_EXCLUDE_CONFIG", raising=False)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(analytics_api.app)


def _update_body() -> Dict[str, object]:
    return {
        "stage": "exclude",
        "streams": [
            {
                "stream_id": "0",
                "label": "living_room",
                "enable": True,
                "rois": [
                    {
                        "id": "TEST1",
                        "description": "pytest override",
                        "points_px": [[0, 0], [32, 0], [32, 32]],
                    }
                ],
            }
        ],
    }


def test_list_rois_returns_stage_payload(client: TestClient):
    response = client.get("/api/v1/analytics/rois")
    assert response.status_code == 200

    payload = response.json()
    assert payload["stage"] == "exclude"
    streams: List[Dict[str, str]] = payload["streams"]
    assert any(stream["stream_id"] == "0" for stream in streams)
    first = streams[0]
    assert "rois" in first
    assert isinstance(first["rois"], list)


def test_exclusion_path_lookup_has_no_default_when_pipeline_is_absent(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("NOESIS_ANALYTICS_EXCLUDE_CONFIG", raising=False)
    _reset_pipeline()

    with pytest.raises(RuntimeError, match="before pipeline construction"):
        analytics_api._resolve_exclude_config_path()


def test_update_rolls_back_when_native_reload_cannot_be_acknowledged(client: TestClient):
    graph = ds8_pipeline.build_pipeline(Path("config/infer.yaml"))
    hooks.attach_analytics_reload_bridge(graph)
    node = graph.ds_pipeline["analytics_exclude"]
    real_set = node.set

    def _drop_reload_request(props: Dict[str, object]) -> None:
        if "reload-request-sequence" in props:
            real_set(
                {
                    key: value
                    for key, value in props.items()
                    if key != "reload-request-sequence"
                }
            )
            return
        real_set(props)

    node.set = _drop_reload_request
    config_path = Path(analytics_api._resolve_analytics_config())
    exclude_path = Path(analytics_api._resolve_exclude_config_path())
    config_before = config_path.read_bytes()
    exclude_before = exclude_path.read_bytes() if exclude_path.exists() else None
    cache_before = copy.deepcopy(analytics_api._load_config(force=True))
    graph_path_before = graph.components["analytics_exclude"].config["config-file"]

    response = client.post("/api/v1/analytics/rois", json=_update_body())

    assert response.status_code == 503
    assert "Failed to apply analytics reload" in response.json()["detail"]
    assert config_path.read_bytes() == config_before
    assert (exclude_path.read_bytes() if exclude_path.exists() else None) == exclude_before
    assert analytics_api._CONFIG_CACHE == cache_before
    assert analytics_api._RELOAD_COUNTER == 0
    assert graph.analytics_reload_count == 0
    assert graph.components["analytics_exclude"].config["config-file"] == graph_path_before


def test_persist_failure_leaves_cache_counter_and_hook_unchanged(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    cache_before = copy.deepcopy(analytics_api._load_config(force=True))
    path_before = analytics_api._CONFIG_PATH
    hook_calls: list[tuple[str, Dict[str, object]]] = []
    analytics_api.register_reload_hook(
        lambda stage, cfg, _context: hook_calls.append((stage, cfg)) or {}
    )

    real_write = analytics_api._atomic_write_text
    attempts = 0

    def _fail_write(path: Path, content: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("read-only analytics state")
        real_write(path, content)

    monkeypatch.setattr(analytics_api, "_atomic_write_text", _fail_write)
    response = client.post("/api/v1/analytics/rois", json=_update_body())

    assert response.status_code == 500
    assert analytics_api._CONFIG_CACHE == cache_before
    assert analytics_api._CONFIG_PATH == path_before
    assert analytics_api._RELOAD_COUNTER == 0
    assert hook_calls == []


def test_atomic_write_failure_removes_temporary_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "analytics.yaml"
    target.write_text("before\n", encoding="utf-8")

    def _fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(analytics_api.os, "replace", _fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        analytics_api._atomic_write_text(target, "after\n")

    assert target.read_text(encoding="utf-8") == "before\n"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda body: body["streams"][0].update({"stream_id": "0]\n[property]\nenable"}),
        lambda body: body["streams"][0]["rois"][0].update({"id": "X\n[property]\nenable"}),
        lambda body: body["streams"][0].update(
            {"rois": [body["streams"][0]["rois"][0], body["streams"][0]["rois"][0]]}
        ),
        lambda body: body["streams"][0]["rois"][0].update(
            {"points_px": [[-1, 0], [32, 0], [32, 32]]}
        ),
    ],
)
def test_update_rejects_ambiguous_or_unsafe_roi_state(client: TestClient, mutate):
    body = _update_body()
    mutate(body)
    response = client.post("/api/v1/analytics/rois", json=body)
    assert response.status_code == 422


def test_update_rejects_nonfinite_coordinates(client: TestClient):
    body = _update_body()
    body["streams"][0]["rois"][0]["points_px"][0][0] = float("nan")
    response = client.post(
        "/api/v1/analytics/rois",
        content=json.dumps(body, allow_nan=True),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_update_returns_exact_native_reload_receipt(client: TestClient):
    graph = ds8_pipeline.build_pipeline(Path("config/infer.yaml"))
    config_path = Path(graph.components["analytics_exclude"].config["config-file"])
    analytics_cfg = analytics_api._load_config(force=True)
    stage_cfg = analytics_cfg["analytics"]["stages"]["exclude"]
    analytics_api._sync_exclude_stage("exclude", stage_cfg)

    class AckNode:
        def __init__(self) -> None:
            digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
            self.values = {
                "config-file": str(config_path),
                "reload-request-sequence": 0,
                "reload-accepted-sequence": 0,
                "reload-failed-sequence": 0,
                "last-reload-ok": True,
                "expected-config-sha256": "",
                "active-config-sha256": digest,
                "reload-error-count": 0,
                "objects-removed-count": 7,
                "last-reload-error": "",
            }

        def get(self, name: str):
            return self.values[name]

        def set(self, values: Dict[str, object]) -> None:
            self.values.update(values)
            if "reload-request-sequence" not in values:
                return
            sequence = int(values["reload-request-sequence"])
            digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
            self.values["reload-request-sequence"] = sequence
            if digest == self.values["expected-config-sha256"]:
                self.values["reload-accepted-sequence"] = sequence
                self.values["last-reload-ok"] = True
                self.values["active-config-sha256"] = digest
                self.values["last-reload-error"] = ""

    class AckPipeline:
        def __init__(self, node: AckNode) -> None:
            self.node = node

        def __getitem__(self, _name: str) -> AckNode:
            return self.node

    node = AckNode()
    graph.ds_pipeline = AckPipeline(node)
    hooks.attach_analytics_reload_bridge(graph)

    response = client.post("/api/v1/analytics/rois", json=_update_body())

    assert response.status_code == 200
    payload = response.json()
    assert payload["reloaded"] is True
    assert payload["reload_receipt"] == {
        "request_sequence": 1,
        "accepted_sequence": 1,
        "failed_sequence": 0,
        "active_config_sha256": node.values["active-config-sha256"],
        "reload_error_count": 0,
        "objects_removed_count": 7,
    }
    assert graph.analytics_reload_count == 1
    assert analytics_api._RELOAD_COUNTER == 1


def test_receipt_read_failure_after_native_dispatch_poison_is_fatal(client: TestClient):
    graph = ds8_pipeline.build_pipeline(Path("config/infer.yaml"))
    config_path = Path(analytics_api._resolve_analytics_config())
    exclude_path = Path(analytics_api._resolve_exclude_config_path())
    config_before = config_path.read_bytes()
    analytics_cfg = analytics_api._load_config(force=True)
    stage_cfg = analytics_cfg["analytics"]["stages"]["exclude"]
    analytics_api._sync_exclude_stage("exclude", stage_cfg)
    exclude_before = exclude_path.read_bytes()

    class PartialCommitNode:
        def __init__(self) -> None:
            self.dispatched = False
            self.values: Dict[str, object] = {
                "config-file": str(exclude_path),
                "reload-request-sequence": 0,
                "reload-accepted-sequence": 0,
                "reload-failed-sequence": 0,
                "last-reload-ok": True,
                "expected-config-sha256": "",
                "active-config-sha256": hashlib.sha256(exclude_path.read_bytes()).hexdigest(),
                "reload-error-count": 0,
                "objects-removed-count": 0,
                "last-reload-error": "",
            }

        def get(self, name: str):
            if self.dispatched and name == "reload-error-count":
                raise RuntimeError("receipt transport failed")
            return self.values[name]

        def set(self, values: Dict[str, object]) -> None:
            self.values.update(values)
            if "reload-request-sequence" in values:
                sequence = int(values["reload-request-sequence"])
                self.values["reload-accepted-sequence"] = sequence
                self.values["active-config-sha256"] = self.values[
                    "expected-config-sha256"
                ]
                self.dispatched = True

    class PartialCommitPipeline:
        def __init__(self, node: PartialCommitNode) -> None:
            self.node = node

        def __getitem__(self, _name: str) -> PartialCommitNode:
            return self.node

    graph.ds_pipeline = PartialCommitPipeline(PartialCommitNode())
    poison_reasons: list[str] = []
    analytics_api.register_poison_hook(poison_reasons.append)
    hooks.attach_analytics_reload_bridge(graph)

    response = client.post("/api/v1/analytics/rois", json=_update_body())

    assert response.status_code == 503
    assert config_path.read_bytes() == config_before
    assert (exclude_path.read_bytes() if exclude_path.exists() else None) == exclude_before
    assert analytics_api._STATE_POISONED is not None
    assert "may have committed" in analytics_api._STATE_POISONED
    assert poison_reasons == [analytics_api._STATE_POISONED]


def test_rollback_failure_poison_invokes_runtime_fatal_hook(monkeypatch: pytest.MonkeyPatch):
    poison_reasons: list[str] = []
    analytics_api.register_poison_hook(poison_reasons.append)
    monkeypatch.setattr(analytics_api, "_rollback_files", lambda _files: ["analytics: restore failed"])

    errors = analytics_api._rollback_transaction([], None, None)

    assert errors == ["analytics: restore failed"]
    assert analytics_api._STATE_POISONED == "analytics rollback failed: analytics: restore failed"
    assert poison_reasons == ["analytics rollback failed: analytics: restore failed"]


def test_enabled_stream_cannot_delete_its_final_roi(client: TestClient):
    body = _update_body()
    body["streams"][0]["rois"] = []

    response = client.post("/api/v1/analytics/rois", json=body)

    assert response.status_code == 422
    assert "disable the stream" in response.json()["detail"]


def test_disabled_stream_may_persist_without_rois(tmp_path: Path):
    target = tmp_path / "disabled-empty.ini"
    stage = {
        "config_width": 64,
        "config_height": 64,
        "streams": {
            "0": {
                "roi_filtering": {
                    "enable": False,
                    "rois": [],
                }
            }
        },
    }

    analytics_api._persist_exclude_ini(stage, target)

    rendered = target.read_text(encoding="utf-8")
    assert "[roi-filtering-stream-0]" in rendered
    assert "enable = 0" in rendered
    parser = configparser.ConfigParser()
    parser.read_string(rendered)
    assert not any(key.startswith("roi-") for key in parser["roi-filtering-stream-0"])


@pytest.mark.parametrize("publication_failure", ["invalid_receipt", "graph_update"])
def test_post_commit_publication_failure_poison_is_fatal(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    publication_failure: str,
):
    graph = ds8_pipeline.build_pipeline(Path("config/infer.yaml"))
    config_path = Path(analytics_api._resolve_analytics_config())
    exclude_path = Path(analytics_api._resolve_exclude_config_path())
    config_before = config_path.read_bytes()
    exclude_before = exclude_path.read_bytes() if exclude_path.exists() else None
    poison_reasons: list[str] = []
    analytics_api.register_poison_hook(poison_reasons.append)

    if publication_failure == "invalid_receipt":
        receipt: Dict[str, object] = {"accepted_sequence": 1}
    else:
        receipt = {
            "request_sequence": 1,
            "accepted_sequence": 1,
            "failed_sequence": 0,
            "active_config_sha256": "a" * 64,
            "reload_error_count": 0,
            "objects_removed_count": 0,
        }

        class ExplodingConfig(dict):
            def setdefault(self, *_args, **_kwargs):
                raise RuntimeError("graph publication failed")

        graph.config = ExplodingConfig(graph.config)

    analytics_api.register_reload_hook(lambda _stage, _cfg, _context: receipt)

    response = client.post("/api/v1/analytics/rois", json=_update_body())

    assert response.status_code == 503
    assert config_path.read_bytes() == config_before
    assert (exclude_path.read_bytes() if exclude_path.exists() else None) == exclude_before
    assert analytics_api._STATE_POISONED is not None
    assert "may have committed" in analytics_api._STATE_POISONED
    assert poison_reasons == [analytics_api._STATE_POISONED]
    assert analytics_api._CONFIG_CACHE is None
    assert analytics_api._CONFIG_PATH is None
